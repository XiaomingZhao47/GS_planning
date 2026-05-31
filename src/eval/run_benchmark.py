"""Top-level driver for the sortie-budget benchmark (§V).

Per method (frontier / volumetric / reactive-gs / deliberative), for K sorties:

    1. plan  sortie_k -> waypoints (using state + GS model from sortie k-1)
    2. exec  habitat_runner.run_sortie     -> sortie_k/ {images, depth, poses}
    3. integrate depth into occupancy grid (for baselines A/B + planning bounds)
    4. union sortie_{1..k} -> NeRF-synth root
    5. train GS up to iteration_{train_iters_per_sortie * k}
    6. eval  vs held-out test poses -> per-sortie metrics

Output layout::

    data/results/<scene>/<method>/
      sortie_<k>/ ...                  (raw frames)
      gs/                              (gs model dir, growing)
      nerf_synth/                      (transforms_train.json)
      occupancy.npz                    (latest log-odds grid)
      metrics.jsonl                    (one row per sortie)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/xiaoming/GS_planning")
sys.path.insert(0, str(ROOT))

from src.sim.habitat_runner import CameraIntrinsics, run_sortie
from src.sim.sortie_to_nerf_synthetic import build_transforms
from src.recon.train_gs import train_gs
from src.planner.occupancy_grid import OccupancyGrid3D
from src.planner.baselines import (PlannerState, plan_frontier_A,
                                    plan_volumetric_B, plan_reactive_gs_C)
from src.planner.deliberative import plan_deliberative
from src.planner.gs_completion import plan_gs_completion
from src.planner.pose_sampling import look_at_opencv
from src.eval.metrics import psnr, coverage_fraction, path_length, SortieMetrics


METHODS = ("frontier", "volumetric", "reactive_gs", "deliberative", "gs_completion")


def _initial_sortie_poses(bounds_min, bounds_max, camera_height=-1.2,
                           look_radius=1.2):
    """Hybrid v6 initial scan: 4 spatial points (small square around the
    center) with 4 full-circle yaws at each = 16 frames total.

    Combines the v4 full-360-degree angular coverage (so the GT surface near
    the centre is well-observed from the start) with v5's translational
    diversity (4 separated eye points provide stereo baselines that 3DGS
    needs to triangulate geometry).
    """
    cx = 0.5 * (bounds_min[0] + bounds_max[0])
    cz = 0.5 * (bounds_min[2] + bounds_max[2])
    # Square half-side ~ 0.6 m: tight enough to stay in free space, wide
    # enough to give real baselines.
    half = 0.6
    eyes = [
        np.array([cx + half, camera_height, cz + half]),
        np.array([cx - half, camera_height, cz + half]),
        np.array([cx - half, camera_height, cz - half]),
        np.array([cx + half, camera_height, cz - half]),
    ]

    poses = []
    yaws_per_eye = 4
    for eye in eyes:
        for k in range(yaws_per_eye):
            yaw = (2 * math.pi) * k / yaws_per_eye
            target = eye + look_radius * np.array(
                [math.sin(yaw), 0.0, math.cos(yaw)])
            poses.append(look_at_opencv(eye, target))
    return poses


def _load_test_set(test_dir: Path) -> tuple[np.ndarray, dict]:
    """Held-out evaluation poses + reference images, layout per gen_test_set.py::

        test_dir/
          poses.npy
          intrinsics.json
          images/rgb_000000.png ...
    """
    poses = np.load(test_dir / "poses.npy")
    intr = json.loads((test_dir / "intrinsics.json").read_text())
    return poses, intr


def _eval_against_test_set(gs_model_dir: Path, test_poses: np.ndarray,
                           test_dir: Path, intr: dict, threshold_db: float):
    import imageio.v2 as imageio
    from src.queries.gs_render_queries import GSRenderer
    r = GSRenderer(gs_model_dir)
    psnrs = []
    img_dir = test_dir / "images"
    for i, p in enumerate(test_poses):
        rgb_t, _ = r.render(p, intr["width"], intr["height"], intr["hfov_deg"])
        rgb = rgb_t.detach().cpu().numpy().transpose(1, 2, 0).clip(0, 1)
        gt = imageio.imread(img_dir / f"rgb_{i:06d}.png").astype(np.float32) / 255.0
        psnrs.append(psnr(rgb, gt[..., :3]))
    psnrs = np.asarray(psnrs)
    return psnrs, {
        threshold_db: coverage_fraction(psnrs, threshold_db),
        20.0: coverage_fraction(psnrs, 20.0),
        25.0: coverage_fraction(psnrs, 25.0),
    }


def run(method: str, scene_path: str, scene_dir: Path, out_root: Path,
        bounds_min: np.ndarray, bounds_max: np.ndarray,
        K: int = 5, train_iters_per_sortie: int = 3000,
        camera_height: float = -1.2, threshold_db: float = 25.0):
    # camera_height is the literal y-coordinate in world frame (Habitat is
    # y-up). For apartment_1 with floor at y=-1.80 and ceiling at y=1.14,
    # -1.2 puts the camera ~0.6 m above the floor.

    out_dir = out_root / method
    out_dir.mkdir(parents=True, exist_ok=True)
    intr = CameraIntrinsics(width=640, height=480, hfov_deg=90.0)

    # The occupancy grid must straddle the camera y range or depth rays from
    # the cameras will start outside the grid and be ignored. Extend a 1 m
    # pad both vertically and horizontally so the grid covers floor + ceiling.
    grid_origin = np.array([
        bounds_min[0] - 0.5,
        bounds_min[1] - 1.0,
        bounds_min[2] - 0.5,
    ])
    grid_dims = np.ceil(np.array([
        (bounds_max[0] + 0.5) - grid_origin[0],
        (bounds_max[1] + 1.0) - grid_origin[1],
        (bounds_max[2] + 0.5) - grid_origin[2],
    ]) / 0.1).astype(int)
    grid = OccupancyGrid3D(origin=grid_origin, cell=0.1, dims=grid_dims)

    state = PlannerState(grid=grid, last_pose=np.eye(4),
                         bounds_min=np.asarray(bounds_min),
                         bounds_max=np.asarray(bounds_max))

    metrics_path = out_dir / "metrics.jsonl"
    metrics_path.write_text("")
    sortie_dirs = []
    cumulative_path = 0.0

    # Test poses for evaluation
    test_poses, test_intr = _load_test_set(scene_dir / "test")

    for k in range(K):
        # --- plan ---
        if k == 0:
            waypoints = _initial_sortie_poses(bounds_min, bounds_max,
                                              camera_height=camera_height)
        else:
            if method == "frontier":
                waypoints = plan_frontier_A(state, camera_height=camera_height)
            elif method == "volumetric":
                waypoints = plan_volumetric_B(state, camera_height=camera_height)
            elif method == "reactive_gs":
                from src.queries.gs_render_queries import GSRenderer
                from src.queries.gs_queries import load_ply, under_reconstructed_cells
                r = GSRenderer(out_dir / "gs")
                ply = (out_dir / "gs" / "point_cloud").glob("iteration_*/point_cloud.ply")
                f = load_ply(sorted(ply, key=lambda p: int(p.parent.name.split("_")[-1]))[-1])
                ur = under_reconstructed_cells(
                    f, grid_origin, grid_origin + grid_dims * 0.1, cell=0.2, rho_min=4.0)
                waypoints = plan_reactive_gs_C(
                    state, r, ur, grid_origin, cell=0.2,
                    camera_height=camera_height)
            elif method == "deliberative":
                from src.queries.gs_render_queries import GSRenderer
                from src.queries.gs_queries import load_ply, under_reconstructed_cells
                r = GSRenderer(out_dir / "gs")
                ply = sorted(
                    (out_dir / "gs" / "point_cloud").glob("iteration_*/point_cloud.ply"),
                    key=lambda p: int(p.parent.name.split("_")[-1]),
                )[-1]
                f = load_ply(ply)
                ur = under_reconstructed_cells(
                    f, grid_origin, grid_origin + grid_dims * 0.1, cell=0.2, rho_min=4.0)
                waypoints = plan_deliberative(
                    state, r, f, ur, grid_origin, cell=0.2,
                    # Loosened from phi_min=0.03, L_max=8 -- with an under-
                    # trained GS the Q5 path-feature density is small almost
                    # everywhere, which was rejecting valid waypoints.
                    phi_min=0.005, L_max=20.0,
                    camera_height=camera_height,
                )
            elif method == "gs_completion":
                from src.queries.gs_queries import load_ply, under_reconstructed_cells
                ply = sorted(
                    (out_dir / "gs" / "point_cloud").glob("iteration_*/point_cloud.ply"),
                    key=lambda p: int(p.parent.name.split("_")[-1]),
                )[-1]
                f = load_ply(ply)
                # under_rec_grid MUST match state.grid resolution (cell=0.1)
                # so cell-coord lookups inside q3_completion_gain are valid.
                ur = under_reconstructed_cells(
                    f, grid_origin, grid_origin + grid_dims * 0.1, cell=0.1,
                    rho_min=20.0)  # rho_min scales with cell**3 (0.1^3 vs 0.2^3 = 8x denser cells)
                waypoints = plan_gs_completion(
                    state, f, ur, camera_height=camera_height)
            else:
                raise ValueError(method)

        # --- exec ---
        sortie_dir = out_dir / f"sortie_{k:02d}"
        run_sortie(scene_path, waypoints, sortie_dir, intr)
        sortie_dirs.append(sortie_dir)
        # Path length: inter-sortie transit (skipped on k=0 since we don't know
        # the true start pose) + intra-sortie waypoint chain length.
        if waypoints:
            if k > 0:
                cumulative_path += float(np.linalg.norm(
                    waypoints[0][:3, 3] - state.last_pose[:3, 3]))
            if len(waypoints) > 1:
                cumulative_path += path_length(waypoints)

        # --- integrate depth into occupancy grid ---
        poses = np.load(sortie_dir / "poses.npy")
        for f_idx in range(poses.shape[0]):
            depth = np.load(sortie_dir / "depth" / f"depth_{f_idx:06d}.npy")
            grid.integrate_depth(poses[f_idx], depth,
                                 fx=intr.fx, fy=intr.fy, cx=intr.cx, cy=intr.cy)
        if waypoints:
            state.last_pose = waypoints[-1]

        # --- union -> nerf-synth root ---
        nerf_root = out_dir / "nerf_synth"
        build_transforms(sortie_dirs, nerf_root)

        # --- train GS ---
        # Fixed iters per sortie (was train_iters * (k+1) -- scaling pushed
        # later sorties past ~10k iters which overfits per user guidance).
        # Each sortie still re-trains from scratch on the union of frames.
        gs_dir = out_dir / "gs"
        it = train_iters_per_sortie
        rc = train_gs(nerf_root, gs_dir, iterations=it)
        if rc != 0:
            print(f"[!] train_gs returned {rc} for sortie {k}", flush=True)

        # --- eval ---
        psnrs, cov = _eval_against_test_set(
            gs_dir, test_poses,
            test_dir=scene_dir / "test",
            intr=test_intr, threshold_db=threshold_db)
        sm = SortieMetrics(
            sortie_idx=k, n_test_views=int(psnrs.size),
            psnr_per_view=psnrs, coverage_at=cov,
            path_length=cumulative_path,
            expected_slam_failures=0,  # populated by deliberative path post-hoc
        )
        with metrics_path.open("a") as fh:
            fh.write(json.dumps(sm.summary()) + "\n")
        print(f"[{method}] sortie {k}: psnr={sm.summary()['psnr_mean']:.2f}dB "
              f"cov@{threshold_db}={cov[threshold_db]:.2%} "
              f"path={cumulative_path:.2f}m", flush=True)


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=METHODS)
    p.add_argument("--scene", required=True, help="path to a Replica scene mesh (PLY/GLB)")
    p.add_argument("--scene-dir", required=True,
                   help="directory containing test/ subdir with eval poses+images")
    p.add_argument("--out", default=str(ROOT / "data" / "results" / "room0"))
    p.add_argument("--bounds-min", nargs=3, type=float, required=True,
                   help="x y z lower corner of nav volume")
    p.add_argument("--bounds-max", nargs=3, type=float, required=True)
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--train-iters", type=int, default=3000)
    p.add_argument("--camera-height", type=float, default=-1.2,
                   help="literal y-coordinate (Habitat y-up) of the camera "
                        "above the floor. For apartment_1, -1.2 is ~0.6 m "
                        "above the floor.")
    return p.parse_args()


if __name__ == "__main__":
    a = _parse()
    run(a.method, a.scene, Path(a.scene_dir), Path(a.out),
        np.asarray(a.bounds_min), np.asarray(a.bounds_max),
        K=a.K, train_iters_per_sortie=a.train_iters,
        camera_height=a.camera_height)
