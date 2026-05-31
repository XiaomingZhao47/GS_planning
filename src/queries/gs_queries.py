"""Planner-queryable interface over a trained 3DGS field (§III-C of the report).

Loads the Inria-format point_cloud.ply that train.py emits to
``<model>/point_cloud/iteration_<N>/point_cloud.ply`` and exposes:

  Q2  regional_uncertainty   tr(Sigma) averaged over Gaussians in a region
  Q3  coverage_density       Gaussians per cubic meter in a region
  Q5  visual_feature_density mean color-gradient magnitude over SH coeffs

Q1 (rendered view) and Q4 (information gain) need the differentiable
rasterizer and live in ``gs_render_queries.py``.

Q6 (semantic relevance) is a model-side query (CLIP / classifier) over a
rendered image — composed on top of Q1.

All "region" arguments are axis-aligned boxes (xyz-min, xyz-max) in world
coordinates. We deliberately keep the implementation pure-numpy + plyfile so
it runs without CUDA.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from plyfile import PlyData


@dataclass
class GSField:
    means: np.ndarray            # (N, 3)
    scales_log: np.ndarray       # (N, 3) -- Inria stores log-scales
    rotations: np.ndarray        # (N, 4) -- quaternions, real-first
    opacity_logit: np.ndarray    # (N,)   -- Inria stores logits
    sh_dc: np.ndarray            # (N, 3) -- band 0 (DC) of SH
    sh_rest: np.ndarray          # (N, 3 * (deg+1)^2 - 3) -- higher SH bands

    @property
    def n(self) -> int:
        return self.means.shape[0]

    @property
    def scales(self) -> np.ndarray:
        return np.exp(self.scales_log)

    @property
    def opacity(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.opacity_logit))


def load_ply(path: str | Path) -> GSField:
    """Load an Inria-format 3DGS PLY."""
    ply = PlyData.read(str(path))
    v = ply["vertex"]

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    scales = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1).astype(np.float32)
    rotations = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype(np.float32)
    opacity = np.asarray(v["opacity"], dtype=np.float32)
    sh_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float32)

    rest_keys = [k for k in v.data.dtype.names if k.startswith("f_rest_")]
    rest_keys.sort(key=lambda k: int(k.split("_")[-1]))
    if rest_keys:
        sh_rest = np.stack([v[k] for k in rest_keys], axis=1).astype(np.float32)
    else:
        sh_rest = np.zeros((xyz.shape[0], 0), dtype=np.float32)

    return GSField(xyz, scales, rotations, opacity, sh_dc, sh_rest)


# ---------- region helpers ----------

def _box_mask(means: np.ndarray, box_min: np.ndarray, box_max: np.ndarray) -> np.ndarray:
    return np.all((means >= box_min) & (means <= box_max), axis=1)


def covariance_traces(field: GSField) -> np.ndarray:
    """Per-Gaussian tr(Sigma) = sum of squared world-frame scales.
    Since Sigma = R diag(s^2) R^T, tr(Sigma) = sum(s_i^2) regardless of R.
    """
    s = field.scales
    return np.sum(s * s, axis=1)


# ---------- Q2 ----------

def regional_uncertainty(field: GSField, box_min, box_max,
                         opacity_thresh: float = 0.05) -> float:
    """Q2: average tr(Sigma) over Gaussians whose mean lies in [box_min, box_max].

    Low-opacity floaters are filtered (they aren't really contributing to the
    surface estimate and otherwise dominate the average).
    Returns ``nan`` if the region contains no qualifying Gaussians.
    """
    box_min = np.asarray(box_min); box_max = np.asarray(box_max)
    mask = _box_mask(field.means, box_min, box_max)
    mask &= (field.opacity >= opacity_thresh)
    if not mask.any():
        return float("nan")
    return float(np.mean(covariance_traces(field)[mask]))


# ---------- Q3 ----------

def coverage_density(field: GSField, box_min, box_max,
                     opacity_thresh: float = 0.05) -> float:
    """Q3: count of opaque Gaussians per unit volume of the query box."""
    box_min = np.asarray(box_min); box_max = np.asarray(box_max)
    mask = _box_mask(field.means, box_min, box_max)
    mask &= (field.opacity >= opacity_thresh)
    vol = float(np.prod(box_max - box_min))
    if vol <= 0:
        return float("nan")
    return float(mask.sum()) / vol


def under_reconstructed_cells(field: GSField, grid_min, grid_max,
                              cell: float, rho_min: float) -> np.ndarray:
    """Voxelize [grid_min, grid_max] at ``cell`` resolution and return a bool
    array of cells with density < rho_min (Q3 thresholded).

    Returns
    -------
    flagged : (Nx, Ny, Nz) bool
    """
    g0 = np.asarray(grid_min, dtype=np.float32)
    g1 = np.asarray(grid_max, dtype=np.float32)
    dims = np.ceil((g1 - g0) / cell).astype(int)
    flagged = np.zeros(tuple(dims), dtype=bool)
    cell_vol = cell ** 3
    # Bucket Gaussian indices by cell to avoid an N x Nvox loop.
    means = field.means
    mask_o = field.opacity >= 0.05
    inside = np.all((means >= g0) & (means < g1), axis=1) & mask_o
    if not inside.any():
        return flagged | True   # everything is under-reconstructed
    idx = ((means[inside] - g0) / cell).astype(int)
    flat = idx[:, 0] * (dims[1] * dims[2]) + idx[:, 1] * dims[2] + idx[:, 2]
    counts = np.bincount(flat, minlength=int(np.prod(dims))).reshape(tuple(dims))
    flagged = (counts.astype(np.float32) / cell_vol) < rho_min
    return flagged


# ---------- Q5 ----------

def visual_feature_density(field: GSField, box_min, box_max,
                           opacity_thresh: float = 0.05) -> float:
    """Q5: mean L2 norm of higher-band SH coefficients (proxy for color
    gradient / texture richness) over Gaussians in the box.

    Rationale: DC term is base color; higher SH bands encode the view-dependent
    component which is what view-invariant texture detectors latch onto. Empty
    SH (degree 0 model) returns 0 by definition; the caller should then use a
    fallback such as ``sh_dc`` variance.
    """
    box_min = np.asarray(box_min); box_max = np.asarray(box_max)
    mask = _box_mask(field.means, box_min, box_max)
    mask &= (field.opacity >= opacity_thresh)
    if not mask.any():
        return float("nan")
    rest = field.sh_rest[mask]
    if rest.size == 0:
        # Degree-0 model -- fall back to per-Gaussian DC variance.
        dc = field.sh_dc[mask]
        return float(np.linalg.norm(dc - dc.mean(axis=0, keepdims=True), axis=1).mean())
    return float(np.linalg.norm(rest, axis=1).mean())


def path_feature_density(field: GSField, polyline: np.ndarray, radius: float = 0.6,
                         z_clip: tuple | None = None) -> float:
    """Q5 evaluated along a 2D top-down polyline: average visual feature
    density within ``radius`` of each segment. Used by the deliberative planner
    to enforce ``phi(path) >= phi_min``.
    """
    means_xz = field.means[:, [0, 2]]
    if z_clip is not None:
        y = field.means[:, 1]
        in_y = (y >= z_clip[0]) & (y <= z_clip[1])
    else:
        in_y = np.ones(field.n, dtype=bool)

    rest = field.sh_rest if field.sh_rest.size else field.sh_dc - field.sh_dc.mean(0)
    feat_norm = np.linalg.norm(rest, axis=1)

    samples = []
    for a, b in zip(polyline[:-1], polyline[1:]):
        seg = b - a
        L = np.linalg.norm(seg)
        n_steps = max(2, int(np.ceil(L / (radius * 0.5))))
        ts = np.linspace(0, 1, n_steps)
        for t in ts:
            c = a + t * seg
            c_xz = c[[0, 2]]   # polyline points are (x,y,z); compare in (x,z)
            dxz = means_xz - c_xz[None, :]
            in_r = (dxz[:, 0] ** 2 + dxz[:, 1] ** 2) < (radius * radius)
            sel = in_r & in_y & (field.opacity >= 0.05)
            if sel.any():
                samples.append(feat_norm[sel].mean())
            else:
                samples.append(0.0)
    return float(np.mean(samples)) if samples else 0.0
