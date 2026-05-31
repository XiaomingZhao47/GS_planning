"""Build a 3-class ground-truth occupancy grid for the scene.

Classes:
    GT_OCC      cells where the GT mesh surface passes through
    GT_FREE     cells inside the apartment that are traversable (from navmesh)
    GT_OUTSIDE  everything else

Outputs (in --out dir):
    gt_occ.npy        bool (Nx, Ny, Nz)
    gt_free.npy       bool (Nx, Ny, Nz)
    gt_outside.npy    bool (Nx, Ny, Nz)
    gt_metadata.json  origin, dims, voxel, class counts
    gt_top_down.png   visualization at camera_height

GT free space is derived from Habitat's pre-computed navmesh: sample dense
points on the navmesh, extrude vertically by [floor_eps, ceiling_eps] meters
into voxels.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/xiaoming/GS_planning")
sys.path.insert(0, str(ROOT))


def _make_grid_geometry(bounds_min, bounds_max, voxel):
    """Match the grid layout used by run_benchmark.run() (and occupancy_export.py).
    Pads x,z by 0.5 m and y by 1.0 m so the floor + ceiling regions are included.
    """
    bounds_min = np.asarray(bounds_min, dtype=np.float64)
    bounds_max = np.asarray(bounds_max, dtype=np.float64)
    origin = np.array([
        bounds_min[0] - 0.5,
        bounds_min[1] - 1.0,
        bounds_min[2] - 0.5,
    ])
    extent = np.array([
        (bounds_max[0] + 0.5) - origin[0],
        (bounds_max[1] + 1.0) - origin[1],
        (bounds_max[2] + 0.5) - origin[2],
    ])
    dims = np.ceil(extent / voxel).astype(int)
    return origin, dims


# ----- GT_OCC from mesh surface -----
def voxelize_gt_surface(scene_glb: Path, voxel: float, origin: np.ndarray,
                        dims: np.ndarray, n_samples: int = 500_000,
                        seed: int = 0) -> np.ndarray:
    """Same logic as surface_coverage.py: sample mesh surface, voxelize."""
    import trimesh  # type: ignore
    mesh = trimesh.load(str(scene_glb), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    rng = np.random.default_rng(seed)
    pts, _ = trimesh.sample.sample_surface(mesh, n_samples,
                                            seed=int(rng.integers(2**31)))
    pts = np.asarray(pts)
    # trimesh axes -> habitat axes
    pts = np.stack([pts[:, 0], pts[:, 2], -pts[:, 1]], axis=1)

    grid = np.zeros(tuple(dims), dtype=bool)
    cells = ((pts - origin) / voxel).astype(int)
    keep = np.all((cells >= 0) & (cells < dims), axis=1)
    cells = cells[keep]
    grid[cells[:, 0], cells[:, 1], cells[:, 2]] = True
    return grid


# ----- GT_FREE from habitat navmesh extrusion -----
def voxelize_gt_free(scene_glb: Path, voxel: float, origin: np.ndarray,
                     dims: np.ndarray, camera_height: float,
                     ceiling_offset: float = 0.4, floor_offset: float = 0.6,
                     n_samples: int = 200_000) -> np.ndarray:
    """Sample the navmesh (free agent positions) and extrude vertically
    around camera_height to mark a volume of GT-free cells.
    """
    import habitat_sim
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = str(scene_glb)
    cfg.enable_physics = False
    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid = "rgb"; rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.resolution = [64, 64]; rgb.hfov = 90.0
    ag = habitat_sim.agent.AgentConfiguration(); ag.sensor_specifications = [rgb]
    sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [ag]))

    grid = np.zeros(tuple(dims), dtype=bool)
    try:
        if not sim.pathfinder.is_loaded:
            print("WARN: navmesh not loaded; GT_FREE will be empty")
            return grid
        rng = np.random.default_rng(0)
        n_collected = 0
        # Get random navigable points -- each one represents a 2D floor cell
        # we then extrude vertically.
        nav_pts = []
        attempts = 0
        while n_collected < n_samples and attempts < 5 * n_samples:
            p = sim.pathfinder.get_random_navigable_point()
            attempts += 1
            if np.all(np.isfinite(p)):
                nav_pts.append(p); n_collected += 1
        nav_pts = np.array(nav_pts, dtype=np.float64)
        print(f"  collected {nav_pts.shape[0]} navmesh points in {attempts} tries")

        # Vertical extrusion: for each navmesh point, mark cells in
        # [navmesh_y - floor_offset, navmesh_y + ceiling_offset] as free.
        y_steps = int((floor_offset + ceiling_offset) / voxel) + 1
        y_offsets = np.linspace(-floor_offset, +ceiling_offset, y_steps)

        for dy in y_offsets:
            pts_y = nav_pts.copy()
            pts_y[:, 1] += dy
            cells = ((pts_y - origin) / voxel).astype(int)
            keep = np.all((cells >= 0) & (cells < dims), axis=1)
            c = cells[keep]
            if c.size:
                grid[c[:, 0], c[:, 1], c[:, 2]] = True
    finally:
        sim.close()

    return grid


def export_top_down(gt_occ, gt_free, gt_outside, out_png: Path,
                    camera_height: float, origin: np.ndarray, voxel: float):
    """3-channel top-down map: red=occ, green=free, gray=outside."""
    import imageio.v2 as imageio
    y_lo = camera_height - 0.4
    y_hi = camera_height + 0.4
    j_lo = max(0, int((y_lo - origin[1]) / voxel))
    j_hi = min(gt_occ.shape[1], int((y_hi - origin[1]) / voxel))
    occ_band = gt_occ[:, j_lo:j_hi, :].any(axis=1)
    free_band = gt_free[:, j_lo:j_hi, :].any(axis=1)
    # outside_band by exclusion
    out_band = ~(occ_band | free_band)

    H, W = occ_band.shape
    img = np.full((H, W, 3), 220, dtype=np.uint8)
    img[free_band] = [180, 240, 180]
    img[occ_band] = [220, 60, 60]
    img[out_band] = [200, 200, 200]
    img = img[::-1]
    imageio.imwrite(out_png, img)


def main(scene_glb: Path, out_dir: Path, bounds_min, bounds_max,
         voxel: float, camera_height: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    origin, dims = _make_grid_geometry(bounds_min, bounds_max, voxel)
    print(f"grid: origin={origin}, dims={dims}, voxel={voxel}")

    print(">> voxelizing GT mesh surface")
    gt_occ = voxelize_gt_surface(scene_glb, voxel, origin, dims)
    print(f"   GT_OCC: {int(gt_occ.sum())} cells")

    print(">> voxelizing GT free space from navmesh")
    gt_free = voxelize_gt_free(scene_glb, voxel, origin, dims, camera_height)
    print(f"   GT_FREE: {int(gt_free.sum())} cells")

    # Free should not include surface cells (in case of voxel overlap)
    gt_free = gt_free & ~gt_occ

    gt_outside = ~(gt_occ | gt_free)
    print(f"   GT_OUTSIDE: {int(gt_outside.sum())} cells")

    np.save(out_dir / "gt_occ.npy", gt_occ)
    np.save(out_dir / "gt_free.npy", gt_free)
    np.save(out_dir / "gt_outside.npy", gt_outside)
    (out_dir / "gt_metadata.json").write_text(json.dumps({
        "origin": origin.tolist(),
        "dims": dims.tolist(),
        "voxel": voxel,
        "camera_height": camera_height,
        "gt_occ_cells": int(gt_occ.sum()),
        "gt_free_cells": int(gt_free.sum()),
        "gt_outside_cells": int(gt_outside.sum()),
    }, indent=2))

    export_top_down(gt_occ, gt_free, gt_outside, out_dir / "gt_top_down.png",
                    camera_height, origin, voxel)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--bounds-min", nargs=3, type=float, default=[-0.5, -1.4, -2.0])
    p.add_argument("--bounds-max", nargs=3, type=float, default=[6.5, -1.0, 7.5])
    p.add_argument("--voxel", type=float, default=0.1)
    p.add_argument("--camera-height", type=float, default=-1.2)
    a = p.parse_args()
    main(Path(a.scene), Path(a.out), a.bounds_min, a.bounds_max,
         a.voxel, a.camera_height)
