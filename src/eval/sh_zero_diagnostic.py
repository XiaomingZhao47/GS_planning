"""Lighting-vs-geometry diagnostic.

A novel-view render that returns ``black`` could be either:
  (a) geometry hole -- no Gaussian projects into that pixel
  (b) SH garbage    -- a Gaussian projects there but its view-dependent SH
                       coefficients return ~0 brightness for the unseen
                       viewing direction.

This script writes two variants of each method's GS field, both pinned to the
same trained means / scales / rotations / opacities:

    *_dc_only.ply       -- SH degree 0 only (DC color); zero out f_rest_*
    *_dc_plus_sh1.ply   -- SH degrees 0+1 only; zero f_rest_* above index 9

Then it re-renders a fixed set of test poses with each variant.

Interpretation:
- If ``black_in_full`` -> ``not_black_in_dc_only``:  SH garbage. Our planner
  needs to encourage angular-coverage observations (each Gaussian seen from
  multiple directions) so the higher SH bands are constrained.
- If ``black_in_full`` -> ``still_black_in_dc_only``: real geometry hole. Our
  planner needs to encourage spatial coverage of those regions.

CLI::

    python -m src.eval.sh_zero_diagnostic \
        --gs   data/results/apartment_1_K3_7k_v6_hybrid/volumetric/gs \
        --poses data/apartment_1/test/poses.npy \
        --intr  data/apartment_1/test/intrinsics.json \
        --out  data/results/apartment_1_K3_7k_v6_hybrid/volumetric/sh_diag
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement   # type: ignore

ROOT = Path("/home/xiaoming/GS_planning")
sys.path.insert(0, str(ROOT))


def _ply_field_names(ply_path: Path) -> list[str]:
    return list(PlyData.read(str(ply_path))["vertex"].data.dtype.names)


def _make_zeroed_variant(src_ply: Path, dst_ply: Path, keep_sh_degree: int):
    """Copy ``src_ply`` to ``dst_ply``, setting every ``f_rest_*`` element
    above the per-channel SH-coef threshold to 0.

    Inria 3DGS stores SH as f_dc_{0,1,2} (DC, 3 channels) and f_rest_{0..N-1}
    where N = 3 * ((max_sh+1)^2 - 1). The order is (sh_coef, channel).
    For max_sh=3 we have 15 extra coefs per channel = 45 f_rest fields,
    indexed 0..14 for channel R, 15..29 for G, 30..44 for B (Inria's
    convention -- channels are the *outer* axis after transposition).
    Empirically the Inria load_ply does:
        features_extra = features_extra.reshape((N, 3, (max_sh+1)**2 - 1))
    so f_rest_N is *channel = N // num_extra*, sh_coef = N % num_extra.

    With max_sh=3, num_extra = 15 -> the first 15 f_rest_* fields (0..14)
    are channel 0's 15 higher-SH coefs in increasing degree order, etc.
    SH degree-1 occupies coefs 1..3 (0-indexed), degree-2 coefs 4..8,
    degree-3 coefs 9..14.

    keep_sh_degree == 0 -> zero every f_rest (only DC remains).
    keep_sh_degree == 1 -> zero coefs with index >= 3 (keep DC + L=1 band).
    """
    ply = PlyData.read(str(src_ply))
    v = ply["vertex"]
    new = v.data.copy()
    rest_keys = [k for k in v.data.dtype.names if k.startswith("f_rest_")]
    n_rest = len(rest_keys)
    assert n_rest % 3 == 0, f"unexpected f_rest count {n_rest}"
    num_extra = n_rest // 3   # per-channel count
    threshold = max(0, (keep_sh_degree + 1) ** 2 - 1)  # # of SH coefs to keep beyond DC
    for k in rest_keys:
        idx = int(k.split("_")[-1])
        local = idx % num_extra
        if local >= threshold:
            new[k] = 0.0
    PlyData([PlyElement.describe(new, "vertex")] + list(ply.elements[1:])).write(str(dst_ply))


def _make_variant_model_dir(src_model: Path, dst_model: Path, keep_sh_degree: int):
    """Clone the Inria model layout but with the PLY ``f_rest_*`` zeroed."""
    dst_model.mkdir(parents=True, exist_ok=True)
    # Copy cfg_args (sets sh_degree); we leave it at the original max_sh
    if (src_model / "cfg_args").exists():
        shutil.copy(src_model / "cfg_args", dst_model / "cfg_args")
    # Find latest iteration_N dir
    pc_dirs = sorted((src_model / "point_cloud").iterdir(),
                     key=lambda p: int(p.name.split("_")[-1]))
    latest = pc_dirs[-1]
    (dst_model / "point_cloud" / latest.name).mkdir(parents=True, exist_ok=True)
    _make_zeroed_variant(latest / "point_cloud.ply",
                          dst_model / "point_cloud" / latest.name / "point_cloud.ply",
                          keep_sh_degree)


def render_set(model_dir: Path, poses: np.ndarray, intr: dict, out_dir: Path,
               tag: str):
    import imageio.v2 as imageio
    from src.queries.gs_render_queries import GSRenderer
    out_dir.mkdir(parents=True, exist_ok=True)
    r = GSRenderer(model_dir)
    for i, p in enumerate(poses):
        rgb, _ = r.render(p, intr["width"], intr["height"], intr["hfov_deg"])
        arr = (rgb.detach().cpu().numpy().transpose(1, 2, 0).clip(0, 1) * 255).astype(np.uint8)
        imageio.imwrite(out_dir / f"{tag}_{i:04d}.png", arr)


def build_strip(out_dir: Path, n_frames: int):
    """3-row strip: full | dc+sh1 | dc only."""
    import imageio.v2 as imageio
    from PIL import Image
    tags = ["full", "dc_sh1", "dc_only"]
    rows = []
    sample = imageio.imread(out_dir / f"{tags[0]}_0000.png")
    H, W = sample.shape[:2]
    scale = 0.5
    Hs, Ws = int(H * scale), int(W * scale)

    def resize(arr):
        return np.array(Image.fromarray(arr).resize((Ws, Hs)))

    for tag in tags:
        imgs = [resize(imageio.imread(out_dir / f"{tag}_{i:04d}.png")[..., :3])
                for i in range(n_frames)]
        rows.append((tag, imgs))

    label_w = 110
    grid_h = len(rows) * Hs
    grid_w = label_w + n_frames * Ws
    grid = np.full((grid_h, grid_w, 3), 240, dtype=np.uint8)
    try:
        from PIL import Image, ImageDraw, ImageFont
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
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
            d.text((6, y0 + Hs // 2 - 10), name, fill=(0, 0, 0), font=font)
            grid = np.array(pil)

    out = out_dir / "sh_strip.png"
    imageio.imwrite(out, grid)
    return out


def main(gs_dir: Path, poses_file: Path, intr_file: Path, out_dir: Path,
         n_frames: int = 8):
    poses = np.load(poses_file)
    intr = json.loads(intr_file.read_text())
    # Subsample for speed
    n = min(n_frames, len(poses))
    idx = np.linspace(0, len(poses) - 1, n, dtype=int)
    poses = poses[idx]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Variants
    dc_only_dir = out_dir / "_dc_only_model"
    dc_sh1_dir = out_dir / "_dc_sh1_model"
    print("building dc-only variant ...")
    _make_variant_model_dir(gs_dir, dc_only_dir, keep_sh_degree=0)
    print("building dc+sh1 variant ...")
    _make_variant_model_dir(gs_dir, dc_sh1_dir, keep_sh_degree=1)

    print("rendering full model ...")
    render_set(gs_dir, poses, intr, out_dir, "full")
    print("rendering dc+sh1 ...")
    render_set(dc_sh1_dir, poses, intr, out_dir, "dc_sh1")
    print("rendering dc-only ...")
    render_set(dc_only_dir, poses, intr, out_dir, "dc_only")

    strip = build_strip(out_dir, n)
    print(f"wrote {strip}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gs", required=True, help="path to a method's gs/ directory")
    p.add_argument("--poses", required=True)
    p.add_argument("--intr", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-frames", type=int, default=8)
    a = p.parse_args()
    main(Path(a.gs), Path(a.poses), Path(a.intr), Path(a.out), a.n_frames)
