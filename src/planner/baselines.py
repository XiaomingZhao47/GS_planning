"""Baseline planners (A, B, C) from §IV-E of the report.

All planners share a signature so the benchmark driver can swap them::

    plan(state) -> list[np.ndarray]   # next sortie waypoints (OpenCV w-from-cam)

A: OctoMap + Frontier             [Yamauchi 1997]
B: OctoMap + Volumetric info gain [Delmerico 2018]
C: Reactive GS NBV                [Jin 2024]

The "state" object is intentionally minimal -- the driver wires it up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
from scipy import ndimage  # type: ignore

from .occupancy_grid import OccupancyGrid3D
from .pose_sampling import GridSamplerConfig, look_at_opencv, sample_grid_poses


@dataclass
class PlannerState:
    grid: OccupancyGrid3D
    last_pose: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    sortie_idx: int = 0
    # Optional GS-side state, set by drivers that train a GS field after each sortie.
    gs_model_dir: Optional[str] = None


# ---------- Baseline A: Frontier ----------

def _frontiers_2d(occ_xz: np.ndarray) -> np.ndarray:
    """Find frontier cells: free cells adjacent to unknown cells.
    occ_xz: -1 unknown, 0 free, 1 occupied. Returns (N, 2) int cells.
    """
    free = occ_xz == 0
    unknown = occ_xz == -1
    unknown_neighbors = ndimage.binary_dilation(unknown)
    frontier = free & unknown_neighbors
    ys, xs = np.where(frontier)
    return np.stack([ys, xs], axis=1)


def plan_frontier_A(state: PlannerState, n_stops: int = 3, yaws_per_stop: int = 2,
                    camera_height: float = -1.2) -> List[np.ndarray]:
    """Baseline A: pick the largest frontier cluster, then walk from the
    current pose TOWARD its centroid in ``n_stops`` stops, capturing
    ``yaws_per_stop`` yaws at each stop. This gives 3DGS translational
    baselines (a pure look-around at the centroid does not).

    Picks the largest cluster that the planner hasn't already centred on this
    sortie -- if every cluster has been visited recently we still take the
    biggest, but at least the walk-along-the-way adds new translation.

    Returns OpenCV world-from-cam poses.
    """
    g = state.grid
    occ_2d = g.project_to_2d(camera_height - 0.4, camera_height + 0.4)
    cells = _frontiers_2d(occ_2d)
    if cells.size == 0:
        return [state.last_pose.copy()]

    mask = np.zeros_like(occ_2d, dtype=bool)
    mask[cells[:, 0], cells[:, 1]] = True
    labels, nlabels = ndimage.label(mask, structure=np.ones((3, 3)))
    sizes = ndimage.sum(mask, labels, index=np.arange(1, nlabels + 1))
    if sizes.size == 0:
        return [state.last_pose.copy()]

    # Damp clusters near the current pose so we don't keep visiting the same one
    cur_x, _, cur_z = state.last_pose[:3, 3]
    damped = sizes.copy()
    for li in range(1, nlabels + 1):
        ys_l, xs_l = np.where(labels == li)
        cy_l = g.origin[2] + ys_l.mean() * g.cell
        cx_l = g.origin[0] + xs_l.mean() * g.cell
        d = np.hypot(cx_l - cur_x, cy_l - cur_z)
        if d < 0.8:
            damped[li - 1] *= 0.2
    biggest = int(np.argmax(damped)) + 1
    ys, xs = np.where(labels == biggest)
    cy, cx = float(ys.mean()), float(xs.mean())

    cz = g.origin[2] + cy * g.cell
    cx_w = g.origin[0] + cx * g.cell
    target_pt = np.array([cx_w, camera_height, cz])

    # Walk from last_pose toward the centroid in n_stops equal segments
    start = state.last_pose[:3, 3].copy()
    # If we have no real start (sortie 0 init was overridden), use the centroid alone
    if not np.any(start):
        stops = [target_pt]
    else:
        stops = [start + (target_pt - start) * (i / n_stops) for i in range(1, n_stops + 1)]

    poses = []
    forward = target_pt - start if np.any(start) else np.array([0, 0, 1.0])
    base_yaw = math.atan2(forward[0], forward[2])
    spread = math.radians(45)
    for eye in stops:
        for k in range(yaws_per_stop):
            yaw = base_yaw + (k - (yaws_per_stop - 1) / 2.0) * spread
            target = eye + 1.5 * np.array([math.sin(yaw), 0.0, math.cos(yaw)])
            poses.append(look_at_opencv(eye, target))
    return poses


# ---------- Baseline B: Volumetric information gain ----------

def _count_unknown_along_rays(g: OccupancyGrid3D, pose: np.ndarray,
                              n_rays: int = 256, max_steps: int = 60) -> int:
    """Cheap Delmerico-style volumetric gain: count unknown voxels visible
    from this pose along a uniform set of rays through the FoV."""
    eye = pose[:3, 3]
    R = pose[:3, :3]
    # Sample ray directions in cam frame (square spiral over 90deg FoV)
    rng = np.random.default_rng(0)
    cam_dirs = rng.standard_normal((n_rays, 3))
    cam_dirs[:, 2] = np.abs(cam_dirs[:, 2]) + 0.5
    cam_dirs /= np.linalg.norm(cam_dirs, axis=1, keepdims=True)
    unknown_mask = g.unknown()
    occ_mask = g.occupied()

    total = 0
    for cam_dir in cam_dirs:
        world_dir = R @ cam_dir
        c = g.world_to_cell(eye)
        if not g.in_bounds(c):
            continue
        # March in fixed cell steps; stop on occupied or out-of-bounds.
        step = np.sign(world_dir).astype(np.int32)
        d_safe = np.where(np.abs(world_dir) < 1e-12, 1e-12, world_dir)
        cell_world = g.origin + (c + (step > 0)) * g.cell
        t_max = (cell_world - eye) / d_safe
        t_delta = g.cell / np.abs(d_safe)
        seen_unknown = 0
        for _ in range(max_steps):
            axis = int(np.argmin(t_max))
            c[axis] += step[axis]
            t_max[axis] += t_delta[axis]
            if not g.in_bounds(c):
                break
            if occ_mask[c[0], c[1], c[2]]:
                break
            if unknown_mask[c[0], c[1], c[2]]:
                seen_unknown += 1
        total += seen_unknown
    return total


def plan_volumetric_B(state: PlannerState, n_waypoints: int = 6,
                      n_candidates: int = 80, camera_height: float = 0.6) -> List[np.ndarray]:
    cfg = GridSamplerConfig(xy_step=0.7, z_levels=(camera_height,),
                            yaws_deg=tuple(range(0, 360, 45)))
    cands = sample_grid_poses(state.bounds_min, state.bounds_max, cfg)
    if cands.shape[0] == 0:
        return [state.last_pose.copy()]
    # Subsample for speed
    if cands.shape[0] > n_candidates:
        idx = np.random.default_rng(0).choice(cands.shape[0], n_candidates, replace=False)
        cands = cands[idx]
    scores = np.array([_count_unknown_along_rays(state.grid, p) for p in cands])
    order = np.argsort(-scores)
    return [cands[i] for i in order[:n_waypoints]]


# ---------- Baseline C: Reactive GS NBV ----------

def plan_reactive_gs_C(state: PlannerState, gs_renderer, under_rec_grid, grid_min,
                       cell: float, top_k: int = 3, yaws_per_stop: int = 2,
                       n_candidates: int = 60,
                       camera_height: float = -1.2) -> List[np.ndarray]:
    """Top-K Q4 NBVs, each captured with a small yaw spread. This gives
    translational baselines instead of a single look-around at one spot.

    No SLAM-reliability constraint, no multi-sortie state -- still
    'reactive' in spirit, just multi-NBV.
    """
    from src.queries.gs_render_queries import q4_information_gain

    cfg = GridSamplerConfig(xy_step=0.7, z_levels=(camera_height,),
                            yaws_deg=tuple(range(0, 360, 60)))
    cands = sample_grid_poses(state.bounds_min, state.bounds_max, cfg)
    if cands.shape[0] == 0:
        return [state.last_pose.copy()]
    if cands.shape[0] > n_candidates:
        idx = np.random.default_rng(0).choice(cands.shape[0], n_candidates, replace=False)
        cands = cands[idx]
    scores = np.array([
        q4_information_gain(gs_renderer, p, under_rec_grid, grid_min, cell)
        for p in cands
    ])
    order = np.argsort(-scores)

    # Suppress nearby duplicates (Q4-greedy without NMS picks clustered yaws
    # at the same eye position).
    chosen_eyes: list = []
    chosen_idx: list = []
    min_dist = 0.6  # meters
    for i in order:
        eye = cands[i][:3, 3]
        if all(np.linalg.norm(eye - e) >= min_dist for e in chosen_eyes):
            chosen_idx.append(int(i))
            chosen_eyes.append(eye)
            if len(chosen_idx) >= top_k:
                break

    poses = []
    spread = math.radians(40)
    for i in chosen_idx:
        eye = cands[i][:3, 3]
        # Use the NBV's own forward direction as the central yaw; spread around it.
        R = cands[i][:3, :3]
        # OpenCV cam forward in world frame = R[:, 2]
        fwd = R[:, 2]
        base_yaw = math.atan2(fwd[0], fwd[2])
        for k in range(yaws_per_stop):
            yaw = base_yaw + (k - (yaws_per_stop - 1) / 2.0) * spread
            target = eye + 1.2 * np.array([math.sin(yaw), 0.0, math.cos(yaw)])
            poses.append(look_at_opencv(eye, target))
    return poses
