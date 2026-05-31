"""GS-completion planner -- our contribution method.

Replaces Q4 (rendering entropy) with Q3 (coverage density)-based gain. Same
overall structure as plan_reactive_gs_C: sample candidates, score, NMS,
expand to 2 yaws per chosen pose.

Why a different planner:
  - reactive_gs uses Q4 = entropy(render) + lam * voxel_hits_in_render_depth
    The entropy term goes to 0 in unobserved regions because the renderer
    returns black, so the planner is BLINDEST exactly where it should be
    most aggressive. The sh_zero_diagnostic confirms those black regions are
    geometry holes, not SH artifacts.
  - gs_completion uses Q3 = unknown_voxels + alpha * under_reconstructed_voxels
    along forward rays from the candidate pose. The gain is computed
    entirely from the (shared) OccupancyGrid3D + the GS field's density,
    NOT from the GS field's rendered output. This makes the gain stable in
    regions the GS hasn't covered yet.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

from .baselines import PlannerState
from .pose_sampling import GridSamplerConfig, look_at_opencv, sample_grid_poses


def plan_gs_completion(state: PlannerState, gs_field, under_rec_grid,
                       top_k: int = 3, yaws_per_stop: int = 2,
                       n_candidates: int = 80, camera_height: float = -1.2,
                       alpha_fill: float = 0.5) -> List[np.ndarray]:
    """Top-K GS-completion NBVs with 0.6 m NMS and 2-yaw spread per stop."""
    from src.queries.gs_completion_gain import (
        CompletionGainConfig, q3_completion_gain,
    )

    cfg_pose = GridSamplerConfig(xy_step=0.7, z_levels=(camera_height,),
                                 yaws_deg=tuple(range(0, 360, 60)))
    cands = sample_grid_poses(state.bounds_min, state.bounds_max, cfg_pose)
    if cands.shape[0] == 0:
        return [state.last_pose.copy()]

    # Drop candidates in occupied or unknown 2D cells -- only navigate
    # through free space.
    occ_2d = state.grid.project_to_2d(camera_height - 0.4, camera_height + 0.4)
    keep = []
    H_g, W_g = occ_2d.shape
    for p in cands:
        x, _, z = p[:3, 3]
        i = int((z - state.grid.origin[2]) / state.grid.cell)
        j = int((x - state.grid.origin[0]) / state.grid.cell)
        if 0 <= i < H_g and 0 <= j < W_g and occ_2d[i, j] == 0:
            keep.append(p)
    cands = np.stack(keep, axis=0) if keep else cands

    if cands.shape[0] > n_candidates:
        idx = np.random.default_rng(0).choice(cands.shape[0], n_candidates, replace=False)
        cands = cands[idx]

    cfg = CompletionGainConfig(alpha_fill=alpha_fill)
    scores = np.array([
        q3_completion_gain(state.grid, under_rec_grid, p, cfg=cfg)
        for p in cands
    ])
    order = np.argsort(-scores)

    chosen_eyes: list = []
    chosen_idx: list = []
    min_dist = 0.6
    for i in order:
        if scores[i] <= 0:
            break
        eye = cands[i][:3, 3]
        if all(np.linalg.norm(eye - e) >= min_dist for e in chosen_eyes):
            chosen_idx.append(int(i))
            chosen_eyes.append(eye)
            if len(chosen_idx) >= top_k:
                break

    if not chosen_idx:
        # All candidates scored zero -- fall back to highest path-length-reachable
        chosen_idx = [int(order[0])]

    poses = []
    spread = math.radians(40)
    for i in chosen_idx:
        eye = cands[i][:3, 3]
        R = cands[i][:3, :3]
        fwd = R[:, 2]
        base_yaw = math.atan2(fwd[0], fwd[2])
        for k in range(yaws_per_stop):
            yaw = base_yaw + (k - (yaws_per_stop - 1) / 2.0) * spread
            target = eye + 1.2 * np.array([math.sin(yaw), 0.0, math.cos(yaw)])
            poses.append(look_at_opencv(eye, target))
    return poses
