"""Cross-modality OctoMap vs 3DGS evaluation against the same GT occupancy.

For each method:
   1. Rebuild its OccupancyGrid3D (from sortie depth maps).
   2. Load its trained 3DGS field and extract per-voxel Gaussian-cell mask
      (cells that contain >= 1 Gaussian mean with opacity > eps).
   3. Score both representations against the GT 3-class grid:
         GT_OCC   : surface voxels (from gt_occupancy.py)
         GT_FREE  : interior free voxels
         GT_OUTSIDE : everything else

Metrics emitted per (method, modality):
    surface_recall (= coverage)
    surface_precision
    surface_iou
    free_recall
    free_precision
    free_iou
    floater_rate   (predicted-occupied voxels in GT_OUTSIDE / total predicted-occ)

CLI::

    python -m src.eval.cross_modal_eval \
        --gt   data/apartment_1/gt_occupancy \
        --out  data/results/apartment_1_K3_7k_v6_hybrid
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/xiaoming/GS_planning")
sys.path.insert(0, str(ROOT))


def _iou(a, b):
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return inter / max(1, union)


def _precision(pred, gt):
    """Of predicted-positive cells, what fraction lie in gt-positive cells."""
    p = int(pred.sum())
    if p == 0: return 0.0
    return int((pred & gt).sum()) / p


def _recall(pred, gt):
    """Of gt-positive cells, what fraction are predicted-positive."""
    g = int(gt.sum())
    if g == 0: return 0.0
    return int((pred & gt).sum()) / g


# ----- OctoMap reconstruction -----

def rebuild_octomap(method_dir: Path, bounds_min, bounds_max, voxel: float):
    from src.planner.occupancy_grid import OccupancyGrid3D
    g_origin = np.array([
        bounds_min[0] - 0.5,
        bounds_min[1] - 1.0,
        bounds_min[2] - 0.5,
    ])
    g_dims = np.ceil(np.array([
        (bounds_max[0] + 0.5) - g_origin[0],
        (bounds_max[1] + 1.0) - g_origin[1],
        (bounds_max[2] + 0.5) - g_origin[2],
    ]) / voxel).astype(int)
    grid = OccupancyGrid3D(origin=g_origin, cell=voxel, dims=g_dims)

    sortie_dirs = sorted(method_dir.glob("sortie_*"))
    if not sortie_dirs:
        return None
    intr = json.loads((sortie_dirs[0] / "intrinsics.json").read_text())
    for sd in sortie_dirs:
        poses = np.load(sd / "poses.npy")
        for f_idx in range(poses.shape[0]):
            depth = np.load(sd / "depth" / f"depth_{f_idx:06d}.npy")
            grid.integrate_depth(poses[f_idx], depth,
                                 fx=intr["fx"], fy=intr["fy"],
                                 cx=intr["cx"], cy=intr["cy"], stride=4)
    return grid


# ----- 3DGS reconstruction (voxelize Gaussian centers) -----

def gs_voxel_mask(method_dir: Path, origin: np.ndarray, dims: np.ndarray,
                  voxel: float, opacity_thresh: float = 0.05) -> np.ndarray:
    from src.queries.gs_queries import load_ply
    plys = sorted(
        (method_dir / "gs" / "point_cloud").glob("iteration_*/point_cloud.ply"),
        key=lambda p: int(p.parent.name.split("_")[-1]),
    )
    if not plys:
        return np.zeros(tuple(dims), dtype=bool)
    field = load_ply(plys[-1])
    means = field.means
    mask = field.opacity >= opacity_thresh
    means = means[mask]
    cells = ((means - origin) / voxel).astype(int)
    keep = np.all((cells >= 0) & (cells < dims), axis=1)
    cells = cells[keep]
    out = np.zeros(tuple(dims), dtype=bool)
    out[cells[:, 0], cells[:, 1], cells[:, 2]] = True
    return out


# ----- driver -----

def evaluate_method(method_dir: Path, gt_occ, gt_free, gt_outside, origin, dims,
                    bounds_min, bounds_max, voxel: float):
    out = {"method": method_dir.name}
    grid = rebuild_octomap(method_dir, bounds_min, bounds_max, voxel)
    if grid is None:
        return None
    octo_occ = grid.occupied()
    octo_free = grid.free()

    out["octo_surface_recall"] = _recall(octo_occ, gt_occ)
    out["octo_surface_precision"] = _precision(octo_occ, gt_occ)
    out["octo_surface_iou"] = _iou(octo_occ, gt_occ)
    out["octo_free_recall"] = _recall(octo_free, gt_free)
    out["octo_free_precision"] = _precision(octo_free, gt_free)
    out["octo_free_iou"] = _iou(octo_free, gt_free)
    out["octo_floater_rate"] = _precision(octo_occ, gt_outside)

    gs_mask = gs_voxel_mask(method_dir, origin, dims, voxel)
    out["gs_surface_recall"] = _recall(gs_mask, gt_occ)
    out["gs_surface_precision"] = _precision(gs_mask, gt_occ)
    out["gs_surface_iou"] = _iou(gs_mask, gt_occ)
    # 3DGS doesn't natively express FREE -- skip free metrics. Floaters yes.
    out["gs_floater_rate"] = _precision(gs_mask, gt_outside)

    return out


def main(gt_dir: Path, bench_root: Path, bounds_min, bounds_max, voxel: float):
    meta = json.loads((gt_dir / "gt_metadata.json").read_text())
    origin = np.asarray(meta["origin"])
    dims = np.asarray(meta["dims"])
    assert abs(meta["voxel"] - voxel) < 1e-9, "voxel mismatch"
    gt_occ = np.load(gt_dir / "gt_occ.npy")
    gt_free = np.load(gt_dir / "gt_free.npy")
    gt_outside = np.load(gt_dir / "gt_outside.npy")

    methods_order = ["frontier", "volumetric", "reactive_gs",
                     "deliberative", "gs_completion"]
    rows = []
    for m in methods_order:
        m_dir = bench_root / m
        if not m_dir.is_dir(): continue
        if not any(m_dir.glob("sortie_*")): continue
        print(f"[{m}] evaluating ...")
        rec = evaluate_method(m_dir, gt_occ, gt_free, gt_outside, origin, dims,
                              bounds_min, bounds_max, voxel)
        if rec is None: continue
        rows.append(rec)
        print(f"  octo_surf  iou={rec['octo_surface_iou']:.3f}  "
              f"recall={rec['octo_surface_recall']:.2%}  "
              f"prec={rec['octo_surface_precision']:.2%}  "
              f"floater={rec['octo_floater_rate']:.2%}")
        print(f"  octo_free  iou={rec['octo_free_iou']:.3f}  "
              f"recall={rec['octo_free_recall']:.2%}  "
              f"prec={rec['octo_free_precision']:.2%}")
        print(f"  gs_surf    iou={rec['gs_surface_iou']:.3f}  "
              f"recall={rec['gs_surface_recall']:.2%}  "
              f"prec={rec['gs_surface_precision']:.2%}  "
              f"floater={rec['gs_floater_rate']:.2%}")

    out_json = bench_root / "cross_modal_eval.json"
    out_json.write_text(json.dumps({"rows": rows,
                                     "gt_meta": meta}, indent=2))
    print(f"wrote {out_json}")

    # Markdown table
    md = ["# Cross-modality reconstruction quality vs GT",
          "",
          "All metrics computed in the same voxel grid as the runtime OccupancyGrid3D.",
          "",
          "| Method | Octo IoU (surf) | Octo recall | Octo prec | Octo floater | GS IoU (surf) | GS recall | GS prec | GS floater |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(
            f"| {r['method']} | "
            f"{r['octo_surface_iou']:.3f} | "
            f"{r['octo_surface_recall']:.2%} | "
            f"{r['octo_surface_precision']:.2%} | "
            f"{r['octo_floater_rate']:.2%} | "
            f"{r['gs_surface_iou']:.3f} | "
            f"{r['gs_surface_recall']:.2%} | "
            f"{r['gs_surface_precision']:.2%} | "
            f"{r['gs_floater_rate']:.2%} |"
        )
    (bench_root / "cross_modal_eval.md").write_text("\n".join(md) + "\n")
    print(f"wrote {bench_root / 'cross_modal_eval.md'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gt", required=True, help="dir with gt_occ.npy etc")
    p.add_argument("--out", required=True, help="benchmark output root")
    p.add_argument("--bounds-min", nargs=3, type=float, default=[-0.5, -1.4, -2.0])
    p.add_argument("--bounds-max", nargs=3, type=float, default=[6.5, -1.0, 7.5])
    p.add_argument("--voxel", type=float, default=0.1)
    a = p.parse_args()
    main(Path(a.gt), Path(a.out), a.bounds_min, a.bounds_max, a.voxel)
