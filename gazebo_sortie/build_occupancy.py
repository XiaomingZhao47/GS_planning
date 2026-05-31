#!/usr/bin/env python3
"""Build a log-odds occupancy grid from the Gazebo sortie's perception
(depth + poses) using the project's OccupancyGrid3D, voxelize the trained
3DGS field on the same grid, and cross-modally compare the two (OctoMap-style
vs 3DGS) the same way src/eval/cross_modal_eval.py does.

No GT mesh here (Gazebo), so instead of scoring against GT we report:
  - per-modality occupied/free/unknown voxel counts
  - OctoMap-vs-3DGS surface agreement (IoU / precision / recall, GS as pred,
    Octo-occupied as the reference "surface")
and write a benchmark row + top-down occupancy figures.

Runs in the gs_planning env (numpy, plyfile, PIL).
"""
from __future__ import annotations
import argparse, json, sys, math
from pathlib import Path
import numpy as np

ROOT = Path("/home/xiaoming/GS_planning_handoff")
sys.path.insert(0, str(ROOT))
from src.planner.occupancy_grid import OccupancyGrid3D  # noqa: E402


def iou(a, b):
    u = int((a | b).sum())
    return int((a & b).sum()) / max(1, u)

def precision(pred, ref):
    p = int(pred.sum())
    return int((pred & ref).sum()) / max(1, p)

def recall(pred, ref):
    g = int(ref.sum())
    return int((pred & ref).sum()) / max(1, g)


def estimate_bounds(sortie, intr, frame_stride, depth_max):
    """Robust bbox of unprojected endpoints (2-98 pct) to size the grid."""
    poses = np.load(sortie / "poses.npy")
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    pts = []
    for i in range(0, poses.shape[0], frame_stride):
        d = np.load(sortie / "depth" / f"depth_{i:06d}.npy")
        H, W = d.shape
        ys = np.arange(0, H, 12); xs = np.arange(0, W, 12)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        z = d[yy, xx]
        m = (z > 0.1) & (z < depth_max) & np.isfinite(z)
        if not m.any():
            continue
        yy, xx, z = yy[m], xx[m], z[m]
        cam = np.stack([(xx - cx) * z / fx, (yy - cy) * z / fy, z], 1)
        wfc = poses[i]
        w = (wfc[:3, :3] @ cam.T).T + wfc[:3, 3]
        pts.append(w)
    pts = np.concatenate(pts, 0)
    lo = np.percentile(pts, 2, axis=0)
    hi = np.percentile(pts, 98, axis=0)
    return lo, hi


def gs_voxel_mask(ply, origin, dims, voxel, opacity_thresh=0.05):
    from plyfile import PlyData
    v = PlyData.read(str(ply))["vertex"]
    means = np.stack([v["x"], v["y"], v["z"]], 1)
    raw_op = np.asarray(v["opacity"])
    op = 1.0 / (1.0 + np.exp(-raw_op))   # sigmoid (ply stores pre-activation)
    means = means[op >= opacity_thresh]
    cells = ((means - origin) / voxel).astype(int)
    keep = np.all((cells >= 0) & (cells < dims), axis=1)
    cells = cells[keep]
    out = np.zeros(tuple(dims), dtype=bool)
    out[cells[:, 0], cells[:, 1], cells[:, 2]] = True
    return out, int((op >= opacity_thresh).sum()), int(keep.sum())


def save_topdown(int2d, path, colormap):
    """int2d: (Nx,Nz) -> RGB PNG. colormap: dict val->(r,g,b)."""
    from PIL import Image
    Nx, Nz = int2d.shape
    img = np.zeros((Nz, Nx, 3), dtype=np.uint8)  # row=z (flip for view), col=x
    for val, col in colormap.items():
        ys, xs = np.where(int2d.T == val)
        img[ys, xs] = col
    Image.fromarray(img[::-1]).resize((Nx * 3, Nz * 3), Image.NEAREST).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sortie", default=str(ROOT / "data/gazebo_house_wf/sortie_00"))
    ap.add_argument("--ply", default=str(ROOT / "data/gazebo_house_wf/gs_model/point_cloud/iteration_7000/point_cloud.ply"))
    ap.add_argument("--out", default=str(ROOT / "data/gazebo_house_wf/benchmark"))
    ap.add_argument("--voxel", type=float, default=0.10)
    ap.add_argument("--frame-stride", type=int, default=3)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--depth-max", type=float, default=5.0)
    a = ap.parse_args()

    sortie = Path(a.sortie); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    intr = json.loads((sortie / "intrinsics.json").read_text())

    lo, hi = estimate_bounds(sortie, intr, a.frame_stride, a.depth_max)
    origin = lo - 0.5
    dims = np.ceil((hi + 0.5 - origin) / a.voxel).astype(int)
    print(f"bounds lo={np.round(lo,2)} hi={np.round(hi,2)} -> dims={dims} ({np.prod(dims)/1e6:.1f}M cells)")

    grid = OccupancyGrid3D(origin=origin, cell=a.voxel, dims=dims)
    poses = np.load(sortie / "poses.npy")
    n_used = 0
    for i in range(0, poses.shape[0], a.frame_stride):
        d = np.load(sortie / "depth" / f"depth_{i:06d}.npy")
        grid.integrate_depth(poses[i], d, fx=intr["fx"], fy=intr["fy"],
                             cx=intr["cx"], cy=intr["cy"],
                             depth_max=a.depth_max, stride=a.stride)
        n_used += 1
    print(f"integrated {n_used} frames")

    octo_occ = grid.occupied(); octo_free = grid.free(); octo_unk = grid.unknown()
    gs_mask, gs_total, gs_in = gs_voxel_mask(Path(a.ply), origin, dims, a.voxel)

    rec = {
        "method": "gazebo_wall_follower",
        "frames_total": int(poses.shape[0]), "frames_integrated": n_used,
        "voxel": a.voxel, "dims": [int(x) for x in dims],
        "octo_occupied": int(octo_occ.sum()),
        "octo_free": int(octo_free.sum()),
        "octo_unknown": int(octo_unk.sum()),
        "gs_total_gaussians": gs_total, "gs_in_bounds": gs_in,
        "gs_occupied_voxels": int(gs_mask.sum()),
        "iou_octo_gs": iou(octo_occ, gs_mask),
        "gs_vs_octo_precision": precision(gs_mask, octo_occ),
        "gs_vs_octo_recall": recall(gs_mask, octo_occ),
    }
    print(json.dumps(rec, indent=2))
    (out / "gazebo_cross_modal.json").write_text(json.dumps(rec, indent=2))

    # top-down figures (squash full height band)
    occ2d = grid.project_to_2d(origin[1], origin[1] + dims[1] * a.voxel)
    save_topdown(occ2d, out / "occupancy_topdown.png",
                 {1: (220, 40, 40), 0: (40, 180, 40), -1: (150, 150, 150)})
    gs2d = np.where(gs_mask.any(axis=1), 1, -1)  # squash height -> (Nx,Nz)
    save_topdown(gs2d, out / "gs_voxels_topdown.png",
                 {1: (40, 90, 220), -1: (150, 150, 150)})
    print(f"wrote figures + json to {out}")

    md = ["# Gazebo wall-follower sortie -- cross-modal occupancy (no GT)", "",
          "OctoMap (depth log-odds) vs 3DGS (voxelized Gaussian means, opacity>=0.05),",
          f"voxel={a.voxel} m, {n_used}/{poses.shape[0]} frames integrated.", "",
          "| Method | Octo occ | Octo free | Octo unk | GS occ vox | IoU(octo,gs) | GS->Octo prec | GS->Octo recall |",
          "|---|---|---|---|---|---|---|---|",
          f"| {rec['method']} | {rec['octo_occupied']} | {rec['octo_free']} | "
          f"{rec['octo_unknown']} | {rec['gs_occupied_voxels']} | {rec['iou_octo_gs']:.3f} | "
          f"{rec['gs_vs_octo_precision']:.2%} | {rec['gs_vs_octo_recall']:.2%} |"]
    (out / "gazebo_cross_modal.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
