"""NBV-style visualization: for each method, render its trained GS at the
poses the *planner itself* chose in the last sortie, and compare against
habitat GT renders at the exact same poses.

If a planner is doing its job, the views it chose should be either:
  - near training viewpoints (so the GS can reconstruct them well), or
  - in regions the GS predicts will benefit from more observation.

Failure mode: planner chooses views where the GS produces garbage, which
means the planner's belief about the world is unreliable.

Output:
  <out_root>/<method>/nbv_renders/
    rgb_gs_NNNNNN.png        (GS rendering at the planner's chosen pose)
    rgb_gt_NNNNNN.png        (Habitat GT at the same pose)
  <out_root>/nbv_grid.png    (compact 4 methods x [GS|GT] x 6 columns grid)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/xiaoming/GS_planning")
sys.path.insert(0, str(ROOT))


def _gt_render(scene: str, poses: np.ndarray, intr: dict, out_dir: Path):
    """Render habitat GT RGB at each provided OpenCV world-from-cam pose."""
    import habitat_sim
    import quaternion
    import imageio.v2 as imageio
    from src.sim.habitat_runner import HABITAT_TO_OPENCV

    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = scene
    cfg.enable_physics = False

    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "rgb"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = [intr["height"], intr["width"]]
    rgb_spec.hfov = intr["hfov_deg"]

    ag = habitat_sim.agent.AgentConfiguration()
    ag.sensor_specifications = [rgb_spec]
    sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [ag]))
    agent = sim.initialize_agent(0)

    try:
        for i, wp in enumerate(poses):
            T_hab = wp @ np.linalg.inv(HABITAT_TO_OPENCV)
            state = habitat_sim.AgentState()
            state.position = T_hab[:3, 3]
            state.rotation = quaternion.from_rotation_matrix(T_hab[:3, :3])
            agent.set_state(state, reset_sensors=True)
            obs = sim.get_sensor_observations()
            imageio.imwrite(out_dir / f"rgb_gt_{i:06d}.png", obs["rgb"][..., :3])
    finally:
        sim.close()


def _gs_render(gs_dir: Path, poses: np.ndarray, intr: dict, out_dir: Path):
    import imageio.v2 as imageio
    from src.queries.gs_render_queries import GSRenderer
    out_dir.mkdir(parents=True, exist_ok=True)
    r = GSRenderer(gs_dir)
    for i, p in enumerate(poses):
        rgb, _ = r.render(p, intr["width"], intr["height"], intr["hfov_deg"])
        arr = (rgb.detach().cpu().numpy().transpose(1, 2, 0).clip(0, 1) * 255).astype(np.uint8)
        imageio.imwrite(out_dir / f"rgb_gs_{i:06d}.png", arr)


def _build_grid(out_root: Path, methods: list[str], n_cols: int):
    import imageio.v2 as imageio
    rows = []
    sample_size = None
    for m in methods:
        d = out_root / m / "nbv_renders"
        if not d.exists():
            print(f"[skip] {m}"); continue
        gs_imgs, gt_imgs = [], []
        # take first n_cols available
        for i in range(n_cols):
            gs_p = d / f"rgb_gs_{i:06d}.png"
            gt_p = d / f"rgb_gt_{i:06d}.png"
            if not gs_p.exists() or not gt_p.exists(): break
            gs_imgs.append(imageio.imread(gs_p)[..., :3])
            gt_imgs.append(imageio.imread(gt_p)[..., :3])
        if gs_imgs:
            if sample_size is None:
                sample_size = gs_imgs[0].shape[:2]
            rows.append((f"{m}/GS", gs_imgs))
            rows.append((f"{m}/GT", gt_imgs))
    if not rows: return
    H, W = sample_size
    label_w = 130
    n_cols_actual = min(n_cols, len(rows[0][1]))
    grid = np.full((len(rows) * H, label_w + n_cols_actual * W, 3), 240, dtype=np.uint8)
    try:
        from PIL import Image, ImageDraw, ImageFont
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except Exception:
            font = ImageFont.load_default()
    except Exception:
        font = None
    for r_idx, (name, imgs) in enumerate(rows):
        y0 = r_idx * H
        for c_idx, img in enumerate(imgs[:n_cols_actual]):
            x0 = label_w + c_idx * W
            grid[y0:y0+H, x0:x0+W] = img
        if font is not None:
            pil = Image.fromarray(grid)
            d = ImageDraw.Draw(pil)
            d.text((6, y0 + H // 2 - 10), name, fill=(0, 0, 0), font=font)
            grid = np.array(pil)
    out = out_root / "nbv_grid.png"
    imageio.imwrite(out, grid)
    print(f"wrote {out} ({grid.shape[1]}x{grid.shape[0]})")


def main(scene: str, out_root: Path, methods: list[str], n_cols: int = 6,
         which_sortie: str = "last"):
    # Collect (method, poses, intr) once
    plan = []
    for m in methods:
        m_dir = out_root / m
        gs_dir = m_dir / "gs"
        if not gs_dir.exists():
            print(f"[skip] {m}: no GS at {gs_dir}"); continue
        sortie_dirs = sorted(m_dir.glob("sortie_*"))
        if not sortie_dirs:
            print(f"[skip] {m}: no sortie dirs"); continue
        sd = sortie_dirs[-1] if which_sortie == "last" else sortie_dirs[int(which_sortie)]
        poses = np.load(sd / "poses.npy")[:n_cols]
        intr = json.loads((sd / "intrinsics.json").read_text())
        plan.append((m, gs_dir, poses, intr, m_dir / "nbv_renders"))

    # Pass 1 (habitat FIRST, before any torch/scipy that pulls llvmlite):
    # ONE habitat-sim session, switch poses across methods.
    import habitat_sim, quaternion
    import imageio.v2 as imageio
    from src.sim.habitat_runner import HABITAT_TO_OPENCV

    for m, _, _, _, nbv_dir in plan:
        nbv_dir.mkdir(parents=True, exist_ok=True)

    intr = plan[0][3]
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = scene
    cfg.enable_physics = False
    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "rgb"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = [intr["height"], intr["width"]]
    rgb_spec.hfov = intr["hfov_deg"]
    ag = habitat_sim.agent.AgentConfiguration()
    ag.sensor_specifications = [rgb_spec]
    sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [ag]))
    agent = sim.initialize_agent(0)
    try:
        for m, _, poses, _, nbv_dir in plan:
            print(f"[render-GT] {m}")
            for i, wp in enumerate(poses):
                T_hab = wp @ np.linalg.inv(HABITAT_TO_OPENCV)
                state = habitat_sim.AgentState()
                state.position = T_hab[:3, 3]
                state.rotation = quaternion.from_rotation_matrix(T_hab[:3, :3])
                agent.set_state(state, reset_sensors=True)
                obs = sim.get_sensor_observations()
                imageio.imwrite(nbv_dir / f"rgb_gt_{i:06d}.png", obs["rgb"][..., :3])
    finally:
        sim.close()

    # Pass 2: GS renders -- safe to do after habitat is shut down.
    for m, gs_dir, poses, intr, nbv_dir in plan:
        print(f"[render-GS] {m}")
        _gs_render(gs_dir, poses, intr, nbv_dir)

    _build_grid(out_root, methods, n_cols)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--methods", nargs="+",
                   default=["frontier", "volumetric", "reactive_gs", "deliberative"])
    p.add_argument("--n-cols", type=int, default=6)
    p.add_argument("--which-sortie", default="last",
                   help="'last' or an integer index")
    a = p.parse_args()
    main(a.scene, Path(a.out), a.methods, a.n_cols, a.which_sortie)
