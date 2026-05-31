"""Q3-based information-gain function (replacement for Q4 in our planner).

Diagnostic outcome (sh_zero_diagnostic.py): the "black novel view" failure
mode is NOT view-dependent SH garbage -- it is genuine geometry holes. So
the right NBV primitive is "how many under-reconstructed cells will this
candidate pose actually see?", NOT "what's the rendering entropy of this
candidate pose?"

Q4 used: H(rendered) + lam * |under_rec voxels in rendered depth|
   problem 1: rendering entropy of a black view is 0 -> planner under-scores
              exactly the regions the planner should be visiting.
   problem 2: depth from the GS at unobserved regions is unreliable; the
              voxel-count term gets noise from spurious depth values.

This module computes the gain WITHOUT relying on the GS field's output --
it ray-casts in the OccupancyGrid3D (the OctoMap-equivalent) and counts:

    new_unknown_voxels         (volumetric-B-style)
  + alpha * under_reconstructed_voxels_in_known_space
                               (cells we've SEEN with depth but the GS
                                density is below rho_min == sparse
                                reconstruction)

so it captures both "explore unknown space" and "fill in sparse known
space". The latter is what frontier/volumetric miss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class CompletionGainConfig:
    n_rays: int = 256
    max_steps: int = 60
    alpha_fill: float = 0.5   # weight for under-reconstructed cells
    cam_fov_deg: float = 90.0


def _sample_ray_dirs_in_fov(n_rays: int, fov_deg: float, rng: np.random.Generator):
    """Sample n_rays direction vectors uniformly inside a forward cone."""
    cam_dirs = rng.standard_normal((n_rays, 3))
    cam_dirs[:, 2] = np.abs(cam_dirs[:, 2]) + 0.5  # bias forward (z+)
    # Optional cone cap
    half_fov_rad = math.radians(fov_deg) / 2
    # Re-normalize, then enforce angle from forward axis <= half_fov
    cam_dirs /= np.linalg.norm(cam_dirs, axis=1, keepdims=True)
    cos_thresh = math.cos(half_fov_rad)
    keep = cam_dirs[:, 2] >= cos_thresh
    cam_dirs = cam_dirs[keep]
    return cam_dirs


def q3_completion_gain(grid, under_rec_grid, pose: np.ndarray,
                       cfg: CompletionGainConfig | None = None,
                       rng_seed: int = 0) -> float:
    """Score a candidate pose by how many under-reconstructed *or* unknown
    voxels its rays would intersect.

    Parameters
    ----------
    grid : OccupancyGrid3D
        The shared 3D log-odds occupancy from depth integration.
    under_rec_grid : (Nx, Ny, Nz) bool
        Output of ``gs_queries.under_reconstructed_cells`` -- True where the
        GS field has fewer Gaussians per m^3 than rho_min.
    pose : (4, 4) world-from-cam (OpenCV).

    Returns
    -------
    score : float -- weighted count of (unknown + alpha * under_rec) cells
        intersected along forward rays, normalized by the number of rays.
    """
    cfg = cfg or CompletionGainConfig()
    eye = pose[:3, 3]
    R = pose[:3, :3]
    rng = np.random.default_rng(rng_seed)
    cam_dirs = _sample_ray_dirs_in_fov(cfg.n_rays, cfg.cam_fov_deg, rng)
    if cam_dirs.shape[0] == 0:
        return 0.0

    unknown = grid.unknown()
    occupied = grid.occupied()

    total = 0.0
    n_eval = 0
    for cam_dir in cam_dirs:
        world_dir = R @ cam_dir
        c = grid.world_to_cell(eye)
        if not grid.in_bounds(c):
            continue
        n_eval += 1
        step = np.sign(world_dir).astype(np.int32)
        d_safe = np.where(np.abs(world_dir) < 1e-12, 1e-12, world_dir)
        cell_world = grid.origin + (c + (step > 0)) * grid.cell
        t_max = (cell_world - eye) / d_safe
        t_delta = grid.cell / np.abs(d_safe)

        seen = 0.0
        for _ in range(cfg.max_steps):
            axis = int(np.argmin(t_max))
            c[axis] += step[axis]
            t_max[axis] += t_delta[axis]
            if not grid.in_bounds(c):
                break
            ci = (c[0], c[1], c[2])
            if occupied[ci]:
                break
            # Unknown OR known-but-under-reconstructed both count.
            if unknown[ci]:
                seen += 1.0
            elif under_rec_grid[ci]:
                seen += cfg.alpha_fill
        total += seen
    return total / max(1, n_eval)
