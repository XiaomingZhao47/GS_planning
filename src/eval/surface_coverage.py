"""GT-mesh-based surface coverage metric (proposal §V.B M1).

Decouples planner evaluation from 3DGS reconstruction noise:

    coverage(method, sortie_k) = | OBS_k  intersect  GT_voxels | / | GT_voxels |

where:
    GT_voxels  = voxelization of the apartment_1.glb mesh surface
    OBS_k      = world-frame voxels intersected by ray endpoints (or rays) of
                  every depth pixel across sorties 0..k for that method.

Two coverage levels are reported:

    surface_coverage  : how much of the ground-truth surface has any training
                        frame *observed* it (depth endpoint within voxel).
    free_coverage     : how much *free* space the rays passed through; useful
                        for path-planning evaluation.

CLI::

    python -m src.eval.surface_coverage \
        --scene .../apartment_1.glb \
        --out data/results/apartment_1_K3_7k_v5_translation \
        --voxel 0.1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path("/home/xiaoming/GS_planning")
sys.path.insert(0, str(ROOT))


# ----- GT mesh -----

def gt_surface_voxels(scene_glb: Path, voxel: float, bounds_min, bounds_max,
                      n_samples: int = 500_000, seed: int = 0):
    """Voxelize the scene mesh surface inside an axis-aligned box.

    Returns
    -------
    voxel_set : set of (i, j, k) int tuples
    origin    : world-frame anchor of cell (0,0,0)
    dims      : (Nx, Ny, Nz) of the bounding voxel grid
    """
    import trimesh  # type: ignore
    mesh = trimesh.load(str(scene_glb), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        # Concatenate child meshes into one
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])
    rng = np.random.default_rng(seed)
    pts, _ = trimesh.sample.sample_surface(mesh, n_samples, seed=int(rng.integers(2**31)))
    pts = np.asarray(pts)
    # trimesh loads glTF with axes permuted vs habitat:
    #   trimesh_x = habitat_x
    #   trimesh_y = -habitat_z
    #   trimesh_z =  habitat_y
    # Reorder to habitat world frame so our depth-unprojected points match.
    pts = np.stack([pts[:, 0], pts[:, 2], -pts[:, 1]], axis=1)

    bounds_min = np.asarray(bounds_min, dtype=np.float64)
    bounds_max = np.asarray(bounds_max, dtype=np.float64)
    origin = bounds_min - voxel  # 1-cell pad
    extent = (bounds_max + voxel) - origin
    dims = np.ceil(extent / voxel).astype(int)

    # Clip to box, voxelize
    inside = np.all((pts >= bounds_min) & (pts <= bounds_max), axis=1)
    pts = pts[inside]
    cells = ((pts - origin) / voxel).astype(int)
    cells = np.unique(cells, axis=0)
    keep = np.all((cells >= 0) & (cells < dims), axis=1)
    cells = cells[keep]
    voxel_set = {tuple(c) for c in cells}
    return voxel_set, origin, dims


# ----- observed voxels from sortie depth maps -----

def observed_voxels_from_sortie(sortie_dir: Path, intr: dict,
                                origin: np.ndarray, voxel: float, dims: np.ndarray,
                                stride: int = 4, depth_min: float = 0.1,
                                depth_max: float = 6.0) -> set:
    """Unproject every (stride-subsampled) depth pixel of every frame in the
    sortie to a world point, drop it into a voxel cell.

    Returns a set of (i,j,k) cells. Uses *only ray endpoints*, not the
    traversed cells -- that's the correct definition of "surface observed".
    """
    poses = np.load(sortie_dir / "poses.npy")
    fx = intr["fx"]; fy = intr["fy"]; cx = intr["cx"]; cy = intr["cy"]
    obs = set()
    for f_idx in range(poses.shape[0]):
        depth = np.load(sortie_dir / "depth" / f"depth_{f_idx:06d}.npy")
        H, W = depth.shape
        ys, xs = np.meshgrid(np.arange(0, H, stride), np.arange(0, W, stride),
                             indexing="ij")
        zz = depth[ys, xs]
        mask = (zz > depth_min) & (zz < depth_max)
        if not mask.any():
            continue
        ys = ys[mask]; xs = xs[mask]; zz = zz[mask]
        x_cam = (xs - cx) * zz / fx
        y_cam = (ys - cy) * zz / fy
        cam_pts = np.stack([x_cam, y_cam, zz], axis=1)
        wfc = poses[f_idx]
        world_pts = (wfc[:3, :3] @ cam_pts.T).T + wfc[:3, 3]
        cells = ((world_pts - origin) / voxel).astype(int)
        keep = np.all((cells >= 0) & (cells < dims), axis=1)
        for c in cells[keep]:
            obs.add(tuple(c))
    return obs


def compute_coverage_per_sortie(out_dir: Path, gt_voxels: set, origin: np.ndarray,
                                voxel: float, dims: np.ndarray):
    """For each sortie_NN in out_dir, integrate cumulative observed cells and
    return [(sortie_idx, coverage_frac, observed_cells, gt_hits)]"""
    sortie_dirs = sorted(out_dir.glob("sortie_*"))
    if not sortie_dirs:
        return []
    intr = json.loads((sortie_dirs[0] / "intrinsics.json").read_text())
    cumulative: set = set()
    records = []
    for sd in sortie_dirs:
        new_obs = observed_voxels_from_sortie(sd, intr, origin, voxel, dims)
        cumulative |= new_obs
        gt_hits = len(cumulative & gt_voxels)
        records.append((int(sd.name.split("_")[-1]),
                        gt_hits / max(1, len(gt_voxels)),
                        len(cumulative), gt_hits))
    return records


# ----- driver -----

def main(scene_glb: Path, bench_root: Path, voxel: float = 0.1,
         bounds_min=(-0.96, -1.80, -2.60), bounds_max=(6.94, 1.14, 8.14),
         methods=None):
    print(f"voxelizing GT mesh {scene_glb} (this may take ~30 s)")
    gt, origin, dims = gt_surface_voxels(scene_glb, voxel, bounds_min, bounds_max)
    print(f"  GT voxels: {len(gt):,}  grid {tuple(dims)}  origin {origin}")

    if methods is None:
        methods = [d.name for d in bench_root.iterdir() if d.is_dir()]

    summary_rows = []
    for m in methods:
        m_dir = bench_root / m
        if not m_dir.is_dir():
            continue
        if not any(m_dir.glob("sortie_*")):
            continue
        recs = compute_coverage_per_sortie(m_dir, gt, origin, voxel, dims)
        for sidx, cov, n_obs, n_hit in recs:
            print(f"[{m}] sortie {sidx}: cov={cov:.2%} "
                  f"obs_cells={n_obs}  gt_hits={n_hit}")
            summary_rows.append({
                "method": m, "sortie": sidx, "surface_coverage": cov,
                "observed_cells": n_obs, "gt_hits": n_hit,
            })

    out_json = bench_root / "surface_coverage.json"
    out_json.write_text(json.dumps({
        "voxel": voxel,
        "n_gt_voxels": len(gt),
        "rows": summary_rows,
    }, indent=2))
    print(f"wrote {out_json}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--out", required=True, help="benchmark output root (contains <method>/sortie_NN/)")
    p.add_argument("--voxel", type=float, default=0.1)
    a = p.parse_args()
    main(Path(a.scene), Path(a.out), a.voxel)
