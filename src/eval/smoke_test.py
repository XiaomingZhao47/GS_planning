"""End-to-end smoke test, run once after the Replica bundle is unpacked.

Validates:
  1. habitat-sim can load the Replica room0 mesh
  2. run_sortie writes RGB-D + poses
  3. sortie_to_nerf_synthetic produces transforms_train.json that Inria reads
  4. train_gs trains a tiny GS (1000 iters) without crashing
  5. GSRenderer can load that model and re-render one of the training poses

This deliberately uses 1 sortie of 6 frames and 1000 iters so total runtime
is ~2 min on a V100, even from a cold cache.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/xiaoming/GS_planning")
sys.path.insert(0, str(ROOT))

from src.sim.habitat_runner import CameraIntrinsics, run_sortie
from src.sim.sortie_to_nerf_synthetic import build_transforms
from src.recon.train_gs import train_gs
from src.planner.pose_sampling import look_at_opencv


def main(scene_path: str, out_root: Path,
         eye_xyz: tuple = (3.0, -1.2, 2.8)):
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    intr = CameraIntrinsics(width=640, height=480, hfov_deg=90.0)

    eye = np.array(eye_xyz, dtype=np.float64)
    waypoints = []
    for i in range(6):
        yaw = (2 * math.pi) * i / 6
        target = eye + 1.5 * np.array([math.sin(yaw), 0.0, math.cos(yaw)])
        waypoints.append(look_at_opencv(eye, target))

    sortie_dir = out_root / "sortie_00"
    print(">> rendering 6 frames via habitat-sim")
    info = run_sortie(scene_path, waypoints, sortie_dir, intr)
    print("   ", info)

    print(">> converting to NeRF-Synthetic")
    nerf_root = out_root / "nerf_synth"
    info = build_transforms([sortie_dir], nerf_root)
    print("   ", info)

    print(">> training GS for 1000 iters")
    gs_dir = out_root / "gs"
    rc = train_gs(nerf_root, gs_dir, iterations=1000,
                  extra=["--test_iterations", "-1", "--save_iterations", "1000"])
    assert rc == 0, f"train_gs rc={rc}"

    print(">> re-rendering pose 0 from trained GS")
    from src.queries.gs_render_queries import GSRenderer
    r = GSRenderer(gs_dir)
    rgb, depth = r.render(waypoints[0], intr.width, intr.height, intr.hfov_deg)
    import imageio.v2 as imageio
    out_img = (rgb.detach().cpu().numpy().transpose(1, 2, 0).clip(0, 1) * 255).astype(np.uint8)
    imageio.imwrite(out_root / "smoke_rerender.png", out_img)
    print("   wrote", out_root / "smoke_rerender.png")
    print(">> SMOKE TEST OK")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True, help="path to a Replica scene mesh (PLY)")
    p.add_argument("--out", default=str(ROOT / "data" / "smoke"))
    args = p.parse_args()
    main(args.scene, Path(args.out))
