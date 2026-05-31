"""3D log-odds occupancy mapper.  Direct extension of PA4
(``occupancy_grid_mapper.py``) to 3D RGB-D depth input.

PA4 raycasted 2D laser ranges via Bresenham; here we raycast 3D depth pixels
via Amanatides-Woo voxel traversal, but the log-odds Bayesian update is the
same form. Reused parameters keep the same semantics:

    l_occ   = +0.85   (one occupied observation)
    l_free  = -0.40   (one free observation)
    l_min, l_max = -2.0, +3.5  (clip)
    p_thresh_occ = 0.65, p_thresh_free = 0.35  (decision boundaries)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


@dataclass
class OccupancyGrid3D:
    origin: np.ndarray            # (3,) world-frame corner of cell [0,0,0]
    cell: float                   # meters
    dims: np.ndarray              # (3,) Nx, Ny, Nz
    l_occ: float = 0.85
    l_free: float = -0.40
    l_min: float = -2.0
    l_max: float = 3.5
    p_occ_thresh: float = 0.65
    p_free_thresh: float = 0.35
    log_odds: np.ndarray = field(init=False)

    def __post_init__(self):
        self.log_odds = np.zeros(tuple(self.dims), dtype=np.float32)

    # ---- cell <-> world ----
    def world_to_cell(self, p: np.ndarray) -> np.ndarray:
        return ((p - self.origin) / self.cell).astype(np.int32)

    def in_bounds(self, c: np.ndarray) -> bool:
        return bool(np.all(c >= 0) and np.all(c < self.dims))

    # ---- decisions ----
    @property
    def prob(self) -> np.ndarray:
        return 1.0 - 1.0 / (1.0 + np.exp(self.log_odds))

    def occupied(self) -> np.ndarray:
        return self.prob > self.p_occ_thresh

    def free(self) -> np.ndarray:
        return self.prob < self.p_free_thresh

    def unknown(self) -> np.ndarray:
        p = self.prob
        return (p >= self.p_free_thresh) & (p <= self.p_occ_thresh)

    # ---- ray traversal (Amanatides-Woo) ----
    def _traverse(self, origin: np.ndarray, direction: np.ndarray, t_end: float,
                  on_voxel):
        d = direction
        d = d / (np.linalg.norm(d) + 1e-12)
        c = self.world_to_cell(origin)
        if not self.in_bounds(c):
            return

        step = np.sign(d).astype(np.int32)
        # avoid divide-by-zero
        d_safe = np.where(np.abs(d) < 1e-12, 1e-12, d)
        cell_origin_world = self.origin + (c + (step > 0).astype(np.int32)) * self.cell
        t_max = (cell_origin_world - origin) / d_safe
        t_delta = self.cell / np.abs(d_safe)

        t = 0.0
        while t < t_end:
            if not on_voxel(c):
                return
            axis = int(np.argmin(t_max))
            t = t_max[axis]
            c = c.copy()
            c[axis] += step[axis]
            t_max[axis] += t_delta[axis]
            if not self.in_bounds(c):
                return

    def integrate_depth(self, world_from_cam: np.ndarray, depth_hw: np.ndarray,
                        fx: float, fy: float, cx: float, cy: float,
                        depth_max: float = 6.0, stride: int = 4) -> None:
        """Update log-odds from one depth frame."""
        H, W = depth_hw.shape
        cam_eye = world_from_cam[:3, 3]
        R = world_from_cam[:3, :3]
        ys = np.arange(0, H, stride)
        xs = np.arange(0, W, stride)
        for y in ys:
            for x in xs:
                z = float(depth_hw[y, x])
                if not (0.05 < z < depth_max):
                    continue
                ray_cam = np.array([(x - cx) / fx, (y - cy) / fy, 1.0])
                ray_world = R @ ray_cam
                hit = cam_eye + ray_world * (z / np.linalg.norm(ray_world))

                # Free along the ray, occupied at endpoint.
                def mark(c, hit_cell=self.world_to_cell(hit)):
                    if np.all(c == hit_cell):
                        self.log_odds[c[0], c[1], c[2]] = float(np.clip(
                            self.log_odds[c[0], c[1], c[2]] + self.l_occ,
                            self.l_min, self.l_max))
                        return False
                    self.log_odds[c[0], c[1], c[2]] = float(np.clip(
                        self.log_odds[c[0], c[1], c[2]] + self.l_free,
                        self.l_min, self.l_max))
                    return True
                self._traverse(cam_eye, ray_world, z, mark)

    # ---- 2D projection for ground-plane planning ----
    def project_to_2d(self, y_lo: float, y_hi: float) -> np.ndarray:
        """Squash y in [y_lo, y_hi] -> 2D (x,z) occupancy. A column counts as
        occupied if any cell in [y_lo, y_hi] is occupied; free if all known
        cells in that band are free; otherwise unknown.
        Returns ints: 0 free, 1 occupied, -1 unknown.
        """
        j_lo = max(0, int((y_lo - self.origin[1]) / self.cell))
        j_hi = min(self.dims[1], int((y_hi - self.origin[1]) / self.cell))
        col = self.log_odds[:, j_lo:j_hi, :]   # (Nx, band, Nz)
        col_p = 1.0 - 1.0 / (1.0 + np.exp(col))
        any_occ = (col_p > self.p_occ_thresh).any(axis=1)
        all_free = (col_p < self.p_free_thresh).all(axis=1)
        out = np.full(col_p.shape[::2], -1, dtype=np.int32)  # (Nx, Nz)
        out[all_free] = 0
        out[any_occ] = 1
        return out
