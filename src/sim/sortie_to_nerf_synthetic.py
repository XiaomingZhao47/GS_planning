"""Convert one or more sortie output directories into a NeRF-Synthetic-style
transforms.json that the Inria 3DGS train.py can ingest via
``readNerfSyntheticInfo``.

The Inria reader expects::

    root/
      transforms_train.json   (camera_angle_x + frames[{file_path, transform_matrix}])
      transforms_test.json    (optional; we duplicate train if absent)
      train/r_00000.png ...

We symlink (or copy) all sortie images into ``root/train/`` with stable names
that encode (sortie_idx, frame_idx) so downstream tools can trace each image
back to its sortie.

NeRF-Synthetic convention: transform_matrix is camera-from-world in OpenGL
basis (x-right, y-up, z-back).  We start from OpenCV world-from-cam, invert
to camera-from-world, then flip axes y,z to OpenGL.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import List

import numpy as np


# OpenCV (x-right, y-down, z-forward) -> OpenGL (x-right, y-up, z-back)
# Applied on the *camera basis*, i.e. on the right of cam-from-world.
OPENCV_TO_OPENGL = np.diag([1.0, -1.0, -1.0, 1.0])


def _world_from_cam_opencv_to_transform_matrix_opengl(wfc_cv: np.ndarray) -> np.ndarray:
    """NeRF-Synthetic stores camera-to-world in OpenGL convention,
    matching Inria's reader which computes ``c2w = transform_matrix`` and then
    flips y,z again to get OpenCV. Net effect: we should hand it OpenGL c2w.
    """
    # OpenCV world-from-cam == c2w in OpenCV basis. Flip basis to OpenGL.
    return wfc_cv @ OPENCV_TO_OPENGL


def build_transforms(
    sortie_dirs: List[Path],
    out_root: Path,
    link_mode: str = "symlink",  # symlink | copy
) -> dict:
    out_root = Path(out_root)
    train_dir = out_root / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    intrinsics = None
    for s_idx, sd in enumerate(sortie_dirs):
        intr = json.loads((sd / "intrinsics.json").read_text())
        if intrinsics is None:
            intrinsics = intr
        else:
            # Mixed intrinsics across sorties not supported in this minimal path.
            assert intr["width"] == intrinsics["width"]
            assert intr["height"] == intrinsics["height"]
            assert abs(intr["hfov_deg"] - intrinsics["hfov_deg"]) < 1e-6

        poses = np.load(sd / "poses.npy")
        img_dir = sd / "images"
        for f_idx in range(poses.shape[0]):
            src = img_dir / f"rgb_{f_idx:06d}.png"
            dst_name = f"s{s_idx:02d}_f{f_idx:06d}.png"
            dst = train_dir / dst_name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            if link_mode == "symlink":
                dst.symlink_to(src.resolve())
            else:
                shutil.copy2(src, dst)

            T = _world_from_cam_opencv_to_transform_matrix_opengl(poses[f_idx])
            frames.append({
                "file_path": f"./train/{Path(dst_name).stem}",
                "transform_matrix": T.tolist(),
            })

    cam_angle_x = math.radians(intrinsics["hfov_deg"])
    manifest = {
        "camera_angle_x": cam_angle_x,
        "w": intrinsics["width"],
        "h": intrinsics["height"],
        "fl_x": intrinsics["fx"],
        "fl_y": intrinsics["fy"],
        "cx": intrinsics["cx"],
        "cy": intrinsics["cy"],
        "frames": frames,
    }
    (out_root / "transforms_train.json").write_text(json.dumps(manifest, indent=2))
    # Reuse train as test if no held-out provided; later we'll generate held-out poses.
    (out_root / "transforms_test.json").write_text(json.dumps(manifest, indent=2))

    # Seed point cloud for the Inria GS optimizer. Without this, train.py
    # samples 100k random points in [-1.3, 1.3]^3, which may not overlap with
    # the actual scene -- causing densification to produce 0 visible
    # Gaussians from many viewpoints and crashing the rasterizer backward pass.
    _write_seed_pointcloud(sortie_dirs, intrinsics, out_root / "points3d.ply")
    return {"n_frames": len(frames), "root": str(out_root)}


def _write_seed_pointcloud(sortie_dirs, intrinsics, out_path: Path,
                            max_points: int = 100000, stride: int = 8) -> None:
    """Unproject depth maps from each sortie to a world-frame point cloud
    written as ``points3d.ply`` for Inria's NeRF-Synthetic reader.
    """
    from plyfile import PlyData, PlyElement  # type: ignore
    import imageio.v2 as imageio  # type: ignore

    fx = intrinsics["fx"]; fy = intrinsics["fy"]
    cx = intrinsics["cx"]; cy = intrinsics["cy"]

    pts_all = []
    cols_all = []
    for sd in sortie_dirs:
        poses = np.load(sd / "poses.npy")
        for f_idx in range(poses.shape[0]):
            depth = np.load(sd / "depth" / f"depth_{f_idx:06d}.npy")
            rgb = imageio.imread(sd / "images" / f"rgb_{f_idx:06d}.png")[..., :3]
            H, W = depth.shape
            ys = np.arange(0, H, stride)
            xs = np.arange(0, W, stride)
            yy, xx = np.meshgrid(ys, xs, indexing="ij")
            zz = depth[yy, xx]
            mask = (zz > 0.1) & (zz < 8.0)
            if not mask.any(): continue
            yy = yy[mask]; xx = xx[mask]; zz = zz[mask]
            x_cam = (xx - cx) * zz / fx
            y_cam = (yy - cy) * zz / fy
            cam_pts = np.stack([x_cam, y_cam, zz], axis=1)
            wfc = poses[f_idx]
            world_pts = (wfc[:3, :3] @ cam_pts.T).T + wfc[:3, 3]
            pts_all.append(world_pts.astype(np.float32))
            cols_all.append(rgb[yy, xx])

    if not pts_all:
        return
    pts = np.concatenate(pts_all, axis=0)
    cols = np.concatenate(cols_all, axis=0)
    if pts.shape[0] > max_points:
        sel = np.random.default_rng(0).choice(pts.shape[0], max_points, replace=False)
        pts = pts[sel]; cols = cols[sel]

    # Inria's fetchPly requires nx/ny/nz fields even though the optimizer
    # ignores them for seeded clouds. Write zeros.
    vert = np.empty(pts.shape[0],
                    dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                           ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
                           ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vert["x"] = pts[:, 0]; vert["y"] = pts[:, 1]; vert["z"] = pts[:, 2]
    vert["nx"] = 0.0; vert["ny"] = 0.0; vert["nz"] = 0.0
    vert["red"] = cols[:, 0]; vert["green"] = cols[:, 1]; vert["blue"] = cols[:, 2]
    PlyData([PlyElement.describe(vert, "vertex")]).write(str(out_path))
    print(f"  seed pointcloud: wrote {pts.shape[0]} pts -> {out_path}")


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sorties", nargs="+", required=True,
                   help="one or more sortie output directories")
    p.add_argument("--out", required=True, help="GS training root")
    p.add_argument("--copy", action="store_true", help="copy images instead of symlink")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    info = build_transforms(
        [Path(s) for s in args.sorties],
        Path(args.out),
        link_mode="copy" if args.copy else "symlink",
    )
    print(json.dumps(info, indent=2))
