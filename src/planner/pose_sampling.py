"""Pose sampling utilities shared by all planners.

A candidate pose is a 4x4 OpenCV world-from-camera matrix.  These functions
emit pools of candidates from which a planner can score-and-select.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np


def look_at_opencv(eye: np.ndarray, target: np.ndarray, up: np.ndarray | None = None) -> np.ndarray:
    """Build an OpenCV world-from-camera matrix that looks from ``eye`` to ``target``."""
    if up is None:
        up = np.array([0.0, 1.0, 0.0])  # habitat y-up
    f = target - eye
    f = f / (np.linalg.norm(f) + 1e-9)
    # OpenCV camera: z forward, x right, y down
    right = np.cross(f, up)
    right = right / (np.linalg.norm(right) + 1e-9)
    down = np.cross(f, right)
    R = np.stack([right, down, f], axis=1)  # world basis vectors as columns
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = eye
    return T


@dataclass
class GridSamplerConfig:
    xy_step: float = 0.5          # meters between candidate (x, z) cells
    z_levels: tuple = (0.6,)      # camera heights above floor (Habitat y-up)
    yaws_deg: tuple = tuple(range(0, 360, 30))  # 12 yaws per (x, z)


def sample_grid_poses(bounds_min: np.ndarray, bounds_max: np.ndarray,
                      cfg: GridSamplerConfig | None = None,
                      look_radius: float = 1.5) -> np.ndarray:
    """Discretize a horizontal grid over [bounds_min, bounds_max] in (x, z)
    and emit yawed look-at poses at each cell.

    Returns
    -------
    poses : (N, 4, 4) array of OpenCV world-from-camera matrices.
    """
    cfg = cfg or GridSamplerConfig()
    xs = np.arange(bounds_min[0], bounds_max[0], cfg.xy_step)
    zs = np.arange(bounds_min[2], bounds_max[2], cfg.xy_step)
    poses = []
    for x in xs:
        for z in zs:
            for y in cfg.z_levels:
                eye = np.array([x, y, z])
                for yaw_deg in cfg.yaws_deg:
                    yaw = math.radians(yaw_deg)
                    target = eye + look_radius * np.array(
                        [math.sin(yaw), 0.0, math.cos(yaw)])
                    poses.append(look_at_opencv(eye, target))
    if not poses:
        return np.empty((0, 4, 4))
    return np.stack(poses, axis=0)


def filter_in_freespace(poses: np.ndarray, occupied_xz: np.ndarray,
                        grid_origin: np.ndarray, grid_res: float) -> np.ndarray:
    """Drop poses whose (x, z) falls in an occupied cell of a 2D top-down grid.

    Parameters
    ----------
    occupied_xz : (H, W) bool array, True == occupied.
    grid_origin : (2,) world-frame (x, z) of cell [0, 0].
    grid_res    : meters per cell.
    """
    keep = []
    H, W = occupied_xz.shape
    for p in poses:
        x, _, z = p[:3, 3]
        i = int((z - grid_origin[1]) / grid_res)
        j = int((x - grid_origin[0]) / grid_res)
        if 0 <= i < H and 0 <= j < W and not occupied_xz[i, j]:
            keep.append(p)
    return np.stack(keep, axis=0) if keep else np.empty((0, 4, 4))
