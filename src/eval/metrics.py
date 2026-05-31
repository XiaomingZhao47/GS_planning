"""Evaluation metrics for the sortie benchmark (§V.B of the report).

  M1  Surface coverage (%) vs sortie count   -- fraction of held-out test
                                                viewpoints whose rendered PSNR
                                                exceeds a threshold.
  M2  Final reconstruction PSNR              -- mean PSNR over held-out views.
  M3  SLAM failure events                    -- proxy via path-feature-density:
                                                each waypoint-to-waypoint
                                                segment whose Q5 < phi_min
                                                counts as one expected failure.
  M4  Total exploration path length          -- cumulative L2 distance over
                                                all sortie waypoints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np


def psnr(rgb_pred: np.ndarray, rgb_gt: np.ndarray) -> float:
    """PSNR in dB. Both inputs are float32 in [0, 1], shape (H, W, 3)."""
    mse = float(np.mean((rgb_pred.astype(np.float32) - rgb_gt.astype(np.float32)) ** 2))
    if mse <= 1e-12:
        return 100.0
    return -10.0 * math.log10(mse)


@dataclass
class SortieMetrics:
    sortie_idx: int
    n_test_views: int
    psnr_per_view: np.ndarray   # (n_test_views,)
    coverage_at: dict[float, float]  # threshold (dB) -> fraction passing
    path_length: float
    expected_slam_failures: int

    def summary(self) -> dict:
        return {
            "sortie_idx": self.sortie_idx,
            "n_test_views": self.n_test_views,
            "psnr_mean": float(np.mean(self.psnr_per_view)),
            "psnr_median": float(np.median(self.psnr_per_view)),
            "coverage": {k: float(v) for k, v in self.coverage_at.items()},
            "path_length": float(self.path_length),
            "expected_slam_failures": int(self.expected_slam_failures),
        }


def coverage_fraction(psnr_per_view: np.ndarray, threshold_db: float) -> float:
    return float((psnr_per_view >= threshold_db).mean())


def path_length(waypoints: Iterable[np.ndarray]) -> float:
    pts = np.stack([w[:3, 3] for w in waypoints], axis=0)
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
