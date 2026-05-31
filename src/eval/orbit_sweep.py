"""Render a smooth orbit through the scene for each method's GS, and the
habitat GT alongside. Produces a strip-grid image per method (rows: time)
and a side-by-side video-frame contact sheet.

Unlike comparison_grid.py (random held-out poses), orbit_sweep renders a
*continuous trajectory* so reconstruction quality is visually obvious as
the camera moves through the apartment.

CLI::
    python -m src.eval.orbit_sweep --scene ...glb --out <bench_root> \
        --bounds-min ... --bounds-max ... --n-frames 24
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List

import numpy as np

ROOT = Path("/home/xiaoming/GS_planning")
sys.path.insert(0, str(ROOT))


def build_orbit_poses(bounds_min, bounds_max, n_frames: int = 24,
                      camera_height: float = -1.2, radius_scale: float = 0.35):
    """Circular orbit around the scene xz-center, camera looks at the center.

    radius_scale: orbit radius as a fraction of the half-extent (so the
    camera stays in free space).
    """
    from src.planner.pose_sampling import look_at_opencv
    cx = 0.5 * (bounds_min[0] + bounds_max[0])
    cz = 0.5 * (bounds_min[2] + bounds_max[2])
    r = radius_scale * min(bounds_max[0] - bounds_min[0],
                            bounds_max[2] - bounds_min[2])
    poses = []
    for k in range(n_frames):
        theta = 2 * math.pi * k / n_frames
        eye = np.array([cx + r * math.cos(theta), camera_height, cz + r * math.sin(theta)])
        target = np.array([cx, camera_height, cz])
        poses.append(look_at_opencv(eye, target))
    return poses


def render_gt(scene: str, poses, intr: dict, out_dir: Path):
    import habitat_sim, quaternion
    import imageio.v2 as imageio
    from src.sim.habitat_runner import HABITAT_TO_OPENCV
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = scene; cfg.enable_physics = False
    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid = "rgb"; rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.resolution = [intr["height"], intr["width"]]; rgb.hfov = intr["hfov_deg"]
    ag = habitat_sim.agent.AgentConfiguration(); ag.sensor_specifications = [rgb]
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
            imageio.imwrite(out_dir / f"gt_{i:04d}.png", obs["rgb"][..., :3])
    finally:
        sim.close()


def render_gs(gs_dir: Path, poses, intr: dict, out_dir: Path):
    import imageio.v2 as imageio
    from src.queries.gs_render_queries import GSRenderer
    out_dir.mkdir(parents=True, exist_ok=True)
    r = GSRenderer(gs_dir)
    for i, p in enumerate(poses):
        rgb, _ = r.render(p, intr["width"], intr["height"], intr["hfov_deg"])
        arr = (rgb.detach().cpu().numpy().transpose(1, 2, 0).clip(0, 1) * 255).astype(np.uint8)
        imageio.imwrite(out_dir / f"gs_{i:04d}.png", arr)


def build_strip(out_root: Path, methods: List[str], n_frames: int,
                 scale: float = 0.4):
    """Stack rows: GT / frontier / volumetric / reactive_gs / deliberative."""
    import imageio.v2 as imageio
    from PIL import Image
    rows = []
    sample = imageio.imread(out_root / "orbit" / "gt" / "gt_0000.png")
    H, W = sample.shape[:2]
    Hs, Ws = int(H * scale), int(W * scale)

    def resize(arr):
        return np.array(Image.fromarray(arr).resize((Ws, Hs)))

    # GT row
    gt = [resize(imageio.imread(out_root / "orbit" / "gt" / f"gt_{i:04d}.png")[..., :3])
          for i in range(n_frames)]
    rows.append(("GT", gt))
    for m in methods:
        d = out_root / "orbit" / m
        if not d.exists():
            continue
        imgs = [resize(imageio.imread(d / f"gs_{i:04d}.png")[..., :3]) for i in range(n_frames)]
        rows.append((m, imgs))

    label_w = 120
    grid_h = len(rows) * Hs
    grid_w = label_w + n_frames * Ws
    grid = np.full((grid_h, grid_w, 3), 240, dtype=np.uint8)
    try:
        from PIL import Image, ImageDraw, ImageFont
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except Exception:
            font = ImageFont.load_default()
    except Exception:
        font = None
    for r_idx, (name, imgs) in enumerate(rows):
        y0 = r_idx * Hs
        for c_idx, img in enumerate(imgs):
            x0 = label_w + c_idx * Ws
            grid[y0:y0+Hs, x0:x0+Ws] = img
        if font is not None:
            pil = Image.fromarray(grid)
            d = ImageDraw.Draw(pil)
            d.text((6, y0 + Hs // 2 - 8), name, fill=(0, 0, 0), font=font)
            grid = np.array(pil)
    out = out_root / "orbit_strip.png"
    imageio.imwrite(out, grid)
    print(f"wrote {out} ({grid.shape[1]}x{grid.shape[0]})")


def main(scene: str, bench_root: Path, methods, bounds_min, bounds_max,
         n_frames: int, camera_height: float):
    intr = {"width": 640, "height": 480, "hfov_deg": 90.0,
            "fx": 0.5 * 640 / math.tan(0.5 * math.radians(90)),
            "fy": 0.5 * 640 / math.tan(0.5 * math.radians(90)),
            "cx": 320, "cy": 240}

    poses = build_orbit_poses(bounds_min, bounds_max, n_frames, camera_height)

    # ONE habitat session for GT, then per-method GS renders.
    print(f"[render-GT] orbit ({n_frames} frames)")
    render_gt(scene, poses, intr, bench_root / "orbit" / "gt")

    for m in methods:
        gs_dir = bench_root / m / "gs"
        if not gs_dir.exists():
            print(f"[skip] {m}"); continue
        print(f"[render-GS] {m}")
        render_gs(gs_dir, poses, intr, bench_root / "orbit" / m)

    build_strip(bench_root, methods, n_frames)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--methods", nargs="+",
                   default=["frontier", "volumetric", "reactive_gs", "deliberative"])
    p.add_argument("--bounds-min", nargs=3, type=float, default=[-0.5, -1.4, -2.0])
    p.add_argument("--bounds-max", nargs=3, type=float, default=[6.5, -1.0, 7.5])
    p.add_argument("--n-frames", type=int, default=18)
    p.add_argument("--camera-height", type=float, default=-1.2)
    a = p.parse_args()
    main(a.scene, Path(a.out), a.methods, a.bounds_min, a.bounds_max,
         a.n_frames, a.camera_height)
