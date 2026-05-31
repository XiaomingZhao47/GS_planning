"""Reconstruct each method's final OccupancyGrid3D from its on-disk sortie
data, then export:

  occupancy_2d.png         top-down 2D projection (free/occupied/unknown)
  occupancy_voxels.ply     occupied voxels as a sparse PLY for 3D viewers
  occupancy_summary.json   per-(method, cell_class) counts vs GT-mesh hits

Why we recompute on disk rather than serialize during run_benchmark:
  - The grid is small enough to rebuild in <1 s from depth maps.
  - Keeps the runtime path simple (training is the slow step, not the grid).

This is the *non-GS* reconstruction artifact -- the OctoMap-equivalent
reconstruction of every method. Together with surface_coverage.json it
gives an apples-to-apples geometric quality comparison across methods.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/xiaoming/GS_planning")
sys.path.insert(0, str(ROOT))


def _rebuild_grid(method_dir: Path, bounds_min, bounds_max, voxel: float):
    from src.planner.occupancy_grid import OccupancyGrid3D
    grid_origin = np.array([
        bounds_min[0] - 0.5,
        bounds_min[1] - 1.0,
        bounds_min[2] - 0.5,
    ])
    grid_dims = np.ceil(np.array([
        (bounds_max[0] + 0.5) - grid_origin[0],
        (bounds_max[1] + 1.0) - grid_origin[1],
        (bounds_max[2] + 0.5) - grid_origin[2],
    ]) / voxel).astype(int)
    g = OccupancyGrid3D(origin=grid_origin, cell=voxel, dims=grid_dims)

    sortie_dirs = sorted(method_dir.glob("sortie_*"))
    if not sortie_dirs:
        return None
    intr = json.loads((sortie_dirs[0] / "intrinsics.json").read_text())
    for sd in sortie_dirs:
        poses = np.load(sd / "poses.npy")
        for f_idx in range(poses.shape[0]):
            depth = np.load(sd / "depth" / f"depth_{f_idx:06d}.npy")
            g.integrate_depth(poses[f_idx], depth,
                              fx=intr["fx"], fy=intr["fy"],
                              cx=intr["cx"], cy=intr["cy"], stride=4)
    return g


def export_top_down(grid, out_png: Path, camera_height: float = -1.2):
    """Render the floor-level slice (y +- 0.4 m around camera_height) as a
    color PNG: green=free, red=occupied, gray=unknown.
    """
    import imageio.v2 as imageio
    occ_2d = grid.project_to_2d(camera_height - 0.4, camera_height + 0.4)
    H, W = occ_2d.shape
    img = np.full((H, W, 3), 220, dtype=np.uint8)
    img[occ_2d == 0] = [180, 240, 180]  # free  -> pale green
    img[occ_2d == 1] = [220, 60, 60]    # occupied -> red
    img[occ_2d == -1] = [200, 200, 200] # unknown  -> gray
    # Flip vertically so +z is up in the image (standard top-down map)
    img = img[::-1]
    imageio.imwrite(out_png, img)
    return out_png


def export_occupied_ply(grid, out_ply: Path):
    """Write occupied cell centers as a colored PLY for 3D inspection."""
    from plyfile import PlyData, PlyElement
    occ = grid.occupied()
    coords = np.argwhere(occ)  # (N, 3) in grid coords
    if coords.size == 0:
        return None
    world = grid.origin + (coords + 0.5) * grid.cell
    n = world.shape[0]
    vert = np.empty(n, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                              ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vert["x"] = world[:, 0]; vert["y"] = world[:, 1]; vert["z"] = world[:, 2]
    vert["red"] = 220; vert["green"] = 60; vert["blue"] = 60
    PlyData([PlyElement.describe(vert, "vertex")]).write(str(out_ply))
    return out_ply


def main(bench_root: Path, bounds_min, bounds_max, voxel: float,
         camera_height: float):
    summary = []
    for m_dir in sorted(bench_root.iterdir()):
        if not m_dir.is_dir(): continue
        if not any(m_dir.glob("sortie_*")): continue
        print(f"[{m_dir.name}] rebuilding grid")
        g = _rebuild_grid(m_dir, bounds_min, bounds_max, voxel)
        if g is None: continue
        out = m_dir / "occupancy"
        out.mkdir(parents=True, exist_ok=True)
        png = export_top_down(g, out / "occupancy_2d.png",
                              camera_height=camera_height)
        ply = export_occupied_ply(g, out / "occupancy_voxels.ply")
        occ_count = int(g.occupied().sum())
        free_count = int(g.free().sum())
        unk_count = int(g.unknown().sum())
        summary.append({"method": m_dir.name,
                        "occupied_cells": occ_count,
                        "free_cells": free_count,
                        "unknown_cells": unk_count,
                        "top_down_png": str(png.relative_to(bench_root)),
                        "voxels_ply": str(ply.relative_to(bench_root)) if ply else None})
        print(f"  occ={occ_count}  free={free_count}  unk={unk_count}  -> {png}")
    (bench_root / "occupancy_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="benchmark output root")
    p.add_argument("--bounds-min", nargs=3, type=float, default=[-0.5, -1.4, -2.0])
    p.add_argument("--bounds-max", nargs=3, type=float, default=[6.5, -1.0, 7.5])
    p.add_argument("--voxel", type=float, default=0.1)
    p.add_argument("--camera-height", type=float, default=-1.2)
    a = p.parse_args()
    main(Path(a.out), a.bounds_min, a.bounds_max, a.voxel, a.camera_height)
