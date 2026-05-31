"""Render every method's final GS field at a fixed set of held-out test
poses, then build a comparison grid (rows: methods + GT, cols: poses).

Run after the benchmark; consumes::

    data/apartment_1/test/{poses.npy, intrinsics.json, images/}
    data/results/<scene>_<tag>/<method>/gs/point_cloud/iteration_*/point_cloud.ply

and writes::

    data/results/<scene>_<tag>/comparison_grid.png
    data/results/<scene>_<tag>/<method>/test_renders/rgb_NNNNNN.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/xiaoming/GS_planning")
sys.path.insert(0, str(ROOT))


def _load_test_set(test_dir: Path):
    poses = np.load(test_dir / "poses.npy")
    intr = json.loads((test_dir / "intrinsics.json").read_text())
    return poses, intr


def render_method(gs_dir: Path, poses: np.ndarray, intr: dict, out_dir: Path,
                  indices=None):
    import imageio.v2 as imageio
    from src.queries.gs_render_queries import GSRenderer
    out_dir.mkdir(parents=True, exist_ok=True)
    r = GSRenderer(gs_dir)
    indices = list(range(len(poses))) if indices is None else indices
    rendered = {}
    for i in indices:
        rgb, _ = r.render(poses[i], intr["width"], intr["height"], intr["hfov_deg"])
        arr = (rgb.detach().cpu().numpy().transpose(1, 2, 0).clip(0, 1) * 255).astype(np.uint8)
        imageio.imwrite(out_dir / f"rgb_{i:06d}.png", arr)
        rendered[i] = arr
    return rendered


def build_grid(out_root: Path, methods: list[str], pose_indices: list[int],
               test_dir: Path):
    import imageio.v2 as imageio
    # Load GT row
    intr = json.loads((test_dir / "intrinsics.json").read_text())
    H, W = intr["height"], intr["width"]
    rows = []
    header = []
    # GT row
    gt_imgs = []
    for i in pose_indices:
        img = imageio.imread(test_dir / "images" / f"rgb_{i:06d}.png")[..., :3]
        gt_imgs.append(img)
    rows.append(("GT", gt_imgs))
    # Method rows
    for m in methods:
        m_imgs = []
        for i in pose_indices:
            p = out_root / m / "test_renders" / f"rgb_{i:06d}.png"
            if p.exists():
                m_imgs.append(imageio.imread(p)[..., :3])
            else:
                m_imgs.append(np.zeros((H, W, 3), dtype=np.uint8))
        rows.append((m, m_imgs))

    # Compose into a (R*H, C*W, 3) image with a label strip on the left
    label_w = 110
    grid_h = len(rows) * H
    grid_w = label_w + len(pose_indices) * W
    grid = np.full((grid_h, grid_w, 3), 240, dtype=np.uint8)

    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
        font = None
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        except Exception:
            font = ImageFont.load_default()
    except Exception:
        font = None

    for r_idx, (name, imgs) in enumerate(rows):
        y0 = r_idx * H
        for c_idx, img in enumerate(imgs):
            x0 = label_w + c_idx * W
            grid[y0:y0+H, x0:x0+W] = img
        # label
        if font is not None:
            pil = Image.fromarray(grid)
            d = ImageDraw.Draw(pil)
            d.text((6, y0 + H // 2 - 12), name, fill=(0, 0, 0), font=font)
            grid = np.array(pil)

    out_path = out_root / "comparison_grid.png"
    imageio.imwrite(out_path, grid)
    print(f"wrote {out_path}  ({grid.shape[1]}x{grid.shape[0]} px)")


def main(out_root: Path, test_dir: Path, methods: list[str], n_cols: int = 6):
    poses, intr = _load_test_set(test_dir)
    # Even sampling across test set
    n = len(poses)
    pose_indices = list(np.linspace(0, n - 1, n_cols, dtype=int))

    for m in methods:
        gs_dir = out_root / m / "gs"
        if not gs_dir.exists():
            print(f"[skip] {m}: no GS at {gs_dir}")
            continue
        print(f"[render] {m}")
        render_method(gs_dir, poses, intr,
                      out_root / m / "test_renders",
                      indices=pose_indices)

    build_grid(out_root, methods, pose_indices, test_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="benchmark output root")
    p.add_argument("--test-dir", required=True, help="path to test/ dir")
    p.add_argument("--methods", nargs="+",
                   default=["frontier", "volumetric", "reactive_gs", "deliberative"])
    p.add_argument("--n-cols", type=int, default=6)
    a = p.parse_args()
    main(Path(a.out), Path(a.test_dir), a.methods, a.n_cols)
