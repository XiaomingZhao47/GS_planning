#!/usr/bin/env python3
"""Exp 3 — same planner ablation as Exp 1+2, but on the TurtleBot3 *2D ground*
sortie of `turtlebot3_house` instead of the 6-DoF wreck.

Candidate pool = 154 non-heldout frames from data/gazebo_house_wf/sortie_00
                 (193 total, every-5th held-out -> 39 test frames).
Bootstrap = 6 frames evenly spaced along the trajectory.
Methods = uniform | random | volumetric | q3_iter (same as wreck).
NMS for IG planners uses Euclidean distance ≥ 0.5 m between camera positions.

This isolates the *robot-platform* axis (TB3 ground vs free-fly) with everything
else (planner, budget, NMS, training discipline) held fixed.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np

ROOT   = Path("/home/xiaoming/GS_planning_handoff")
SORTIE = ROOT / "data/gazebo_house_wf/sortie_00"
OUT    = ROOT / "data/tb3_exp3"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).parent))
from exp1_run import (
    _score_unknown, _score_q3, _train_mini_gs, _gs_density_mask,
    write_seed_ply, make_dataset, train_one, parse_psnr,
)

sys.path.insert(0, str(ROOT))
from src.planner.occupancy_grid import OccupancyGrid3D  # noqa: E402

BOUNDS_MIN = (-10.0, -2.0, -10.0)   # y-up convention (comp1 = robot height ~0)
BOUNDS_MAX = ( 10.0,  2.0,  10.0)
VOXEL      = 0.30
NMS_DIST_M = 0.5
B          = 18


def load_full_sortie(d: Path):
    poses = np.load(d / "poses.npy")
    intr  = json.loads((d / "intrinsics.json").read_text())
    n     = poses.shape[0]
    imgs  = [d / "images" / f"rgb_{i:06d}.png" for i in range(n)]
    deps  = [d / "depth"  / f"depth_{i:06d}.npy" for i in range(n)]
    return poses, intr, imgs, deps


def build_grid_tb3(poses, depths, intr, indices, voxel=VOXEL, depth_max=8.0):
    lo = np.asarray(BOUNDS_MIN, np.float64); hi = np.asarray(BOUNDS_MAX, np.float64)
    dims = np.ceil((hi - lo) / voxel).astype(int)
    g = OccupancyGrid3D(origin=lo, cell=voxel, dims=dims)
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    for i in indices:
        g.integrate_depth(poses[i], np.load(depths[i]),
                          fx=fx, fy=fy, cx=cx, cy=cy,
                          depth_max=depth_max, stride=8)
    return g


def euclidean_nms(cands, scores, n_extra, poses, min_dist=NMS_DIST_M):
    order = np.argsort(-scores)
    chosen, fallback = [], []
    for j in order:
        i = int(cands[int(j)])
        eye = poses[i][:3, 3]
        ok = all(np.linalg.norm(eye - poses[ci][:3, 3]) >= min_dist for ci in chosen)
        (chosen if ok else fallback).append(i)
        if len(chosen) >= n_extra:
            break
    if len(chosen) < n_extra:
        chosen += fallback[: n_extra - len(chosen)]
    return chosen


def plan_uniform(cand_idx, B, bootstrap):
    n_extra = B - len(bootstrap)
    rem = [i for i in cand_idx if i not in bootstrap]
    step = max(1, len(rem) // n_extra)
    return sorted(set(bootstrap + rem[::step][:n_extra]))


def plan_random(cand_idx, B, bootstrap, seed=42):
    n_extra = B - len(bootstrap)
    rem = [i for i in cand_idx if i not in bootstrap]
    return sorted(set(bootstrap + [int(i) for i in
                  np.random.default_rng(seed).choice(rem, n_extra, replace=False)]))


def plan_volumetric(poses, depths, intr, bootstrap, B, cand_idx):
    n_extra = B - len(bootstrap)
    g = build_grid_tb3(poses, depths, intr, bootstrap)
    print(f"  octo: occ={int(g.occupied().sum())} free={int(g.free().sum())} unk={int(g.unknown().sum())}")
    rem = [i for i in cand_idx if i not in bootstrap]
    rng = np.random.default_rng(0)
    scores = np.array([_score_unknown(g, poses[i], rng=rng) for i in rem])
    return sorted(set(bootstrap + euclidean_nms(rem, scores, n_extra, poses)))


def plan_q3_iter(poses, imgs, depths, intr, bootstrap, B, cand_idx, work_dir, iterations=7000):
    n_extra = B - len(bootstrap)
    g = build_grid_tb3(poses, depths, intr, bootstrap)
    print(f"  octo: occ={int(g.occupied().sum())} free={int(g.free().sum())} unk={int(g.unknown().sum())}")
    print(f"  training sortie-1 GS on {len(bootstrap)} frames ({iterations} iters)...")
    bs_p = poses[bootstrap]; bs_i = [imgs[i] for i in bootstrap]; bs_d = [depths[i] for i in bootstrap]
    ply = _train_mini_gs(bs_p, bs_i, bs_d, intr, work_dir, iterations=iterations)
    gs_mask = _gs_density_mask(ply, g.origin, g.dims, g.cell)
    print(f"  gs voxels: {int(gs_mask.sum())}")
    rem = [i for i in cand_idx if i not in bootstrap]
    rng = np.random.default_rng(0)
    scores = np.array([_score_q3(g, gs_mask, poses[i], rng=rng) for i in rem])
    return sorted(set(bootstrap + euclidean_nms(rem, scores, n_extra, poses)))


def main():
    poses, intr, imgs, deps = load_full_sortie(SORTIE)
    n = len(poses)
    heldout_idx = list(range(0, n, 5))
    cand_idx    = [i for i in range(n) if i not in heldout_idx]
    held_poses  = poses[heldout_idx]
    held_imgs   = [imgs[i] for i in heldout_idx]
    step = max(1, len(cand_idx) // 6)
    bootstrap   = sorted(cand_idx[::step][:6])
    print(f"loaded {n} frames | candidates={len(cand_idx)} | heldout={len(heldout_idx)}")
    print(f"bootstrap: {bootstrap}\n")

    results = {}
    for m in ["uniform", "random", "volumetric", "q3_iter"]:
        print(f"=== {m} (B={B}) ===")
        if m == "uniform":
            sel = plan_uniform(cand_idx, B, bootstrap)
        elif m == "random":
            sel = plan_random(cand_idx, B, bootstrap)
        elif m == "volumetric":
            sel = plan_volumetric(poses, deps, intr, bootstrap, B, cand_idx)
        elif m == "q3_iter":
            sel = plan_q3_iter(poses, imgs, deps, intr, bootstrap, B, cand_idx,
                               OUT / "q3_iter_work")
        print(f"  selected ({len(sel)}): {sel[:8]}...")

        ds  = OUT / f"ds_{m}_B{B}"
        mdl = OUT / f"model_{m}_B{B}"
        sp  = poses[sel]; si = [imgs[i] for i in sel]; sd = [deps[i] for i in sel]
        make_dataset(ds, sp, si, sd, held_poses, held_imgs, intr)
        log = train_one(ds, mdl)
        test_psnr, train_psnr = parse_psnr(log)
        print(f"  test PSNR  = {test_psnr}")
        print(f"  train PSNR = {train_psnr}\n")
        results[m] = {"selected": list(map(int, sel)),
                      "test_psnr": test_psnr,
                      "train_psnr": train_psnr}

    (OUT / "results_B18.json").write_text(json.dumps(results, indent=2))
    print(f"\nSUMMARY (TB3 house, B={B}, held-out PSNR):")
    for m, r in results.items():
        print(f"  {m:12s}: {r['test_psnr']:.2f} dB  (train {r['train_psnr']:.2f} dB)")


if __name__ == "__main__":
    main()
