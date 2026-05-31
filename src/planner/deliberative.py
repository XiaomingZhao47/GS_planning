"""Our sequential deliberative planner (§III-D of the report).

For each sortie, the planner:
  1. Samples candidate poses on a grid over the current bounds (free cells only)
  2. Scores each by  J = I(Q4) + beta * S(Q6)   -- semantic relevance plugged in via callback
  3. Filters by Q5 path-feature-density along the candidate path from last_pose
     (proxy for SLAM reliability through textureless corridors)
  4. Greedy-select top-n waypoints under a path-length budget L_max

The Q6 callback is optional; if omitted we reduce to pure Q4 + Q5 filtering.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

from .baselines import PlannerState
from .pose_sampling import GridSamplerConfig, sample_grid_poses


def plan_deliberative(state: PlannerState, gs_renderer, gs_field,
                      under_rec_grid, grid_min, cell: float,
                      phi_min: float, L_max: float,
                      n_waypoints: int = 6, n_candidates: int = 80,
                      camera_height: float = 0.6,
                      beta: float = 0.0,
                      semantic_score: Optional[Callable[[np.ndarray], float]] = None
                      ) -> List[np.ndarray]:
    from src.queries.gs_queries import path_feature_density
    from src.queries.gs_render_queries import q4_information_gain, q1_render

    cfg = GridSamplerConfig(xy_step=0.6, z_levels=(camera_height,),
                            yaws_deg=tuple(range(0, 360, 60)))
    cands = sample_grid_poses(state.bounds_min, state.bounds_max, cfg)
    if cands.shape[0] == 0:
        return [state.last_pose.copy()]

    # Drop candidates whose (x,z) is in an occupied/unknown cell of the 2D map
    occ_2d = state.grid.project_to_2d(camera_height - 0.4, camera_height + 0.4)
    keep = []
    for p in cands:
        x, _, z = p[:3, 3]
        i = int((z - state.grid.origin[2]) / state.grid.cell)
        j = int((x - state.grid.origin[0]) / state.grid.cell)
        if 0 <= i < occ_2d.shape[0] and 0 <= j < occ_2d.shape[1] and occ_2d[i, j] == 0:
            keep.append(p)
    cands = np.stack(keep, axis=0) if keep else cands

    if cands.shape[0] > n_candidates:
        idx = np.random.default_rng(0).choice(cands.shape[0], n_candidates, replace=False)
        cands = cands[idx]

    # ---- score Q4 + beta*Q6 ----
    j_scores = np.zeros(cands.shape[0])
    for k, p in enumerate(cands):
        i_gain = q4_information_gain(gs_renderer, p, under_rec_grid, grid_min, cell)
        s = 0.0
        if beta > 0.0 and semantic_score is not None:
            rgb = q1_render(gs_renderer, p)
            s = semantic_score(rgb.detach().cpu().numpy().transpose(1, 2, 0))
        j_scores[k] = i_gain + beta * s

    order = np.argsort(-j_scores)

    # ---- greedy selection under path-length + phi_min constraints ----
    chosen: List[np.ndarray] = []
    cur = state.last_pose.copy()
    cur_len = 0.0
    for idx in order:
        p = cands[idx]
        seg = np.linalg.norm(p[:3, 3] - cur[:3, 3])
        if cur_len + seg > L_max:
            continue
        # Q5 along the straight-line segment from cur -> p
        polyline = np.stack([cur[:3, 3], p[:3, 3]], axis=0)
        phi = path_feature_density(
            gs_field, polyline, radius=0.6,
            z_clip=(camera_height - 0.4, camera_height + 0.4),
        )
        if phi < phi_min:
            continue
        chosen.append(p)
        cur = p
        cur_len += seg
        if len(chosen) >= n_waypoints:
            break

    if not chosen:
        # Bail out: take top Q4 anyway (so we always make progress) but warn.
        chosen = [cands[int(order[0])]]
    return chosen
