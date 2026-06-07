#!/usr/bin/env python3
"""Exp 1 driver — planner ablation for 3DGS reconstruction quality on the
Herkules wreck. Each method picks B candidates from a fixed 72-pose pool, then
3DGS is trained on those B with the same 18-pose held-out test set.

Methods:
  uniform     : every-(N/B)-th candidate, no planning
  random      : RNG selection (seeded)
  volumetric  : 6 fixed bootstrap + (B-6) candidates that maximize unknown-voxel
                gain along forward rays in an OccupancyGrid3D built from the
                bootstrap depths (Delmerico-style Baseline B, 3D version).

Run in the gs_planning env. Reuses src/planner/occupancy_grid.py.
"""
from __future__ import annotations
import argparse, json, math, os, subprocess, sys
from pathlib import Path
import numpy as np

ROOT    = Path("/home/xiaoming/GS_planning_handoff")
DATA    = ROOT / "data/wreck_exp1"
GS_REPO = Path("/home/xiaoming/gaussian-splatting")
PY      = str(Path.home() / "miniconda3/envs/gs_planning/bin/python")
SEEDED_TRAIN = ROOT / "gazebo_sortie/seeded_train.py"


def _train_env(seed):
    """Subprocess env that forwards GS_SEED when set."""
    if seed is None:
        return None
    env = os.environ.copy()
    env["GS_SEED"] = str(int(seed))
    return env


def _train_entry(seed):
    """Return the python script that drives one Inria GS run. When `seed`
    is None we invoke Inria's train.py directly (legacy). When `seed` is
    set we go through seeded_train.py, which monkey-patches safe_state."""
    return str(SEEDED_TRAIN) if seed is not None else str(GS_REPO / "train.py")

sys.path.insert(0, str(ROOT))
from src.planner.occupancy_grid import OccupancyGrid3D  # noqa: E402

OPENCV_TO_OPENGL = np.diag([1.0, -1.0, -1.0, 1.0])

# 18 yaws × 4 alts, altitude-major. Bootstrap 6 frames across **two** altitudes
# (alt 0 = z-3, alt 2 = z+3) at staggered yaws — gives planners a non-trivial
# initial observation set that already spans the vertical axis.
BOOTSTRAP_INDICES = [0, 6, 12, 39, 45, 51]   # alt0 yaws 0/120/240, alt2 yaws 60/180/300


# ─── utilities ───────────────────────────────────────────────

def load_sortie(d: Path):
    poses = np.load(d / "poses.npy")
    intr  = json.loads((d / "intrinsics.json").read_text())
    n     = poses.shape[0]
    imgs  = [d / "images"  / f"rgb_{i:06d}.png" for i in range(n)]
    deps  = [d / "depth"   / f"depth_{i:06d}.npy" for i in range(n)]
    return poses, intr, imgs, deps


def write_seed_ply(out_path: Path, sel_poses, sel_depths, intr,
                   max_pts=80000, stride=8, depth_max=30.0):
    from plyfile import PlyData, PlyElement
    import imageio.v2 as imageio
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    pts_all, cols_all = [], []
    for P, dpath in zip(sel_poses, sel_depths):
        d = np.load(dpath)
        H, W = d.shape
        yy, xx = np.meshgrid(np.arange(0, H, stride), np.arange(0, W, stride), indexing="ij")
        z = d[yy, xx]; m = (z > 0.1) & (z < depth_max) & np.isfinite(z)
        if not m.any():
            continue
        cam = np.stack([(xx[m] - cx) * z[m] / fx, (yy[m] - cy) * z[m] / fy, z[m]], 1)
        w = (P[:3, :3] @ cam.T).T + P[:3, 3]
        pts_all.append(w.astype(np.float32))
        rgb_path = str(dpath).replace("/depth/depth_", "/images/rgb_").replace(".npy", ".png")
        rgb = imageio.imread(rgb_path)[..., :3]
        cols_all.append(rgb[yy[m], xx[m]])
    pts = np.concatenate(pts_all, 0); cols = np.concatenate(cols_all, 0)
    if pts.shape[0] > max_pts:
        sel = np.random.default_rng(0).choice(pts.shape[0], max_pts, replace=False)
        pts = pts[sel]; cols = cols[sel]
    vert = np.empty(pts.shape[0],
                    dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                           ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
                           ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vert["x"], vert["y"], vert["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    vert["nx"] = vert["ny"] = vert["nz"] = 0.0
    vert["red"], vert["green"], vert["blue"] = cols[:, 0], cols[:, 1], cols[:, 2]
    PlyData([PlyElement.describe(vert, "vertex")]).write(str(out_path))


def make_dataset(out_dir: Path,
                 train_poses, train_imgs, train_depths,
                 test_poses,  test_imgs,
                 intr):
    out_dir.mkdir(parents=True, exist_ok=True)
    train_d = out_dir / "train"; train_d.mkdir(exist_ok=True)

    def transforms(name, poses, imgs, prefix):
        frames = []
        for i, (P, src) in enumerate(zip(poses, imgs)):
            dst_name = f"{prefix}_{i:04d}.png"
            link = train_d / dst_name
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(src.resolve())
            T_gl = P @ OPENCV_TO_OPENGL
            frames.append({"file_path": f"./train/{dst_name[:-4]}",
                           "transform_matrix": T_gl.tolist()})
        manifest = {"camera_angle_x": math.radians(intr["hfov_deg"]),
                    "w": intr["width"], "h": intr["height"],
                    "fl_x": intr["fx"], "fl_y": intr["fy"],
                    "cx": intr["cx"], "cy": intr["cy"],
                    "frames": frames}
        (out_dir / f"transforms_{name}.json").write_text(json.dumps(manifest, indent=2))

    transforms("train", train_poses, train_imgs, "t")
    transforms("test",  test_poses,  test_imgs,  "h")
    write_seed_ply(out_dir / "points3d.ply", train_poses, train_depths, intr)


# ─── planners ──────────────────────────────────────────────────

def plan_uniform(n_total, B, bootstrap):
    """Bootstrap + uniform spread over the remaining candidates."""
    n_extra = B - len(bootstrap)
    remaining = [i for i in range(n_total) if i not in bootstrap]
    step = max(1, len(remaining) // n_extra)
    chosen = remaining[::step][:n_extra]
    return sorted(set(bootstrap + chosen))


def plan_random(n_total, B, bootstrap, seed=42):
    n_extra = B - len(bootstrap)
    remaining = [i for i in range(n_total) if i not in bootstrap]
    chosen = np.random.default_rng(seed).choice(remaining, n_extra, replace=False).tolist()
    return sorted(set(bootstrap + [int(i) for i in chosen]))


def _build_grid(poses, depths, intr, indices, voxel=0.6, depth_max=30.0,
                bounds_min=(-25.0, -25.0, -70.0),
                bounds_max=( 25.0,  25.0, -40.0)):
    """Fixed wreck-shell AABB; percentile-based bounds were too tight and
    clipped the depth endpoints out of grid -> 0 occupied cells."""
    lo = np.asarray(bounds_min, dtype=np.float64)
    hi = np.asarray(bounds_max, dtype=np.float64)
    dims = np.ceil((hi - lo) / voxel).astype(int)
    g = OccupancyGrid3D(origin=lo, cell=voxel, dims=dims)
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    for i in indices:
        g.integrate_depth(poses[i], np.load(depths[i]),
                          fx=fx, fy=fy, cx=cx, cy=cy,
                          depth_max=depth_max, stride=8)
    return g


def _score_unknown(g, pose, n_rays=128, max_steps=40, rng=np.random.default_rng(0)):
    eye = pose[:3, 3]; R = pose[:3, :3]
    cam = rng.standard_normal((n_rays, 3))
    cam[:, 2] = np.abs(cam[:, 2]) + 0.5
    cam /= np.linalg.norm(cam, axis=1, keepdims=True)
    unk = g.unknown(); occ = g.occupied()
    total = 0
    for cd in cam:
        wd = R @ cd
        c = g.world_to_cell(eye)
        if not g.in_bounds(c):
            continue
        step  = np.sign(wd).astype(np.int32)
        d_safe = np.where(np.abs(wd) < 1e-12, 1e-12, wd)
        cw = g.origin + (c + (step > 0)) * g.cell
        t_max = (cw - eye) / d_safe
        t_delta = g.cell / np.abs(d_safe)
        seen = 0
        for _ in range(max_steps):
            ax = int(np.argmin(t_max))
            c = c.copy(); c[ax] += step[ax]
            t_max[ax] += t_delta[ax]
            if not g.in_bounds(c):
                break
            if occ[c[0], c[1], c[2]]:
                break
            if unk[c[0], c[1], c[2]]:
                seen += 1
        total += seen
    return total


def _nms_select(cands, scores, n_extra, n_yaws=18, min_yaw_step=2, max_per_alt=None):
    """Greedy-by-score with non-maximum suppression on the (alt_idx, yaw_idx) grid:
      (a) within the same altitude, require cyclic yaw distance >= min_yaw_step;
      (b) cap the number of picks per altitude at max_per_alt — prevents the
          greedy IG signal from stacking all picks at the high vantage point.
    Falls back to taking remaining best if fewer than n_extra pass NMS."""
    order = np.argsort(-scores)
    chosen, fallback = [], []
    alt_count: dict[int, int] = {}
    for j in order:
        i = int(cands[int(j)])
        a, y = i // n_yaws, i % n_yaws
        ok = True
        # (a) yaw spacing within same altitude
        for ci in chosen:
            ca, cy = ci // n_yaws, ci % n_yaws
            if ca == a:
                d = abs(y - cy); d = min(d, n_yaws - d)
                if d < min_yaw_step:
                    ok = False; break
        # (b) altitude cap (only when set)
        if ok and max_per_alt is not None and alt_count.get(a, 0) >= max_per_alt:
            ok = False
        if ok:
            chosen.append(i)
            alt_count[a] = alt_count.get(a, 0) + 1
        else:
            fallback.append(i)
        if len(chosen) >= n_extra:
            break
    if len(chosen) < n_extra:
        chosen += fallback[: n_extra - len(chosen)]
    return chosen


def plan_volumetric(poses, depths, intr, bootstrap, B):
    n_extra = B - len(bootstrap)
    g = _build_grid(poses, depths, intr, bootstrap)
    print(f"  octo: occ={int(g.occupied().sum())} free={int(g.free().sum())} unk={int(g.unknown().sum())}")
    cands = [i for i in range(poses.shape[0]) if i not in bootstrap]
    rng = np.random.default_rng(0)
    scores = np.array([_score_unknown(g, poses[i], rng=rng) for i in cands])
    chosen = _nms_select(cands, scores, n_extra)
    return sorted(set(bootstrap + chosen))


# --- Q3: bootstrap mini-GS + occupancy. Q3 gain = unknown + alpha * under-rec ---

def _train_mini_gs(sel_poses, sel_imgs, sel_depths, intr, work_dir, iterations=1500, seed=None):
    work_dir = Path(work_dir); work_dir.mkdir(parents=True, exist_ok=True)
    ds = work_dir / "mini_ds"; ds.mkdir(exist_ok=True); (ds / "train").mkdir(exist_ok=True)
    frames = []
    for i, (P, src) in enumerate(zip(sel_poses, sel_imgs)):
        dn = f"b_{i:04d}.png"
        link = ds / "train" / dn
        if link.exists() or link.is_symlink(): link.unlink()
        link.symlink_to(Path(src).resolve())
        T_gl = P @ OPENCV_TO_OPENGL
        frames.append({"file_path": f"./train/{dn[:-4]}", "transform_matrix": T_gl.tolist()})
    manifest = {"camera_angle_x": math.radians(intr["hfov_deg"]),
                "w": intr["width"], "h": intr["height"],
                "fl_x": intr["fx"], "fl_y": intr["fy"],
                "cx": intr["cx"], "cy": intr["cy"], "frames": frames}
    (ds / "transforms_train.json").write_text(json.dumps(manifest, indent=2))
    (ds / "transforms_test.json").write_text(json.dumps(manifest, indent=2))
    write_seed_ply(ds / "points3d.ply", sel_poses, sel_depths, intr)
    mdir = work_dir / "mini_model"
    cmd = [PY, _train_entry(seed),
           "-s", str(ds), "-m", str(mdir),
           "--iterations", str(iterations),
           "--disable_viewer",
           "--opacity_reset_interval", "999999",
           "--densify_until_iter", str(min(iterations - 200, 1200))]
    subprocess.run(cmd, cwd=str(GS_REPO), env=_train_env(seed),
                   capture_output=True, text=True, check=True)
    return mdir / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"


def _gs_density_mask(ply_path, origin, dims, voxel, opacity_thresh=0.05):
    from plyfile import PlyData
    v = PlyData.read(str(ply_path))["vertex"]
    means = np.stack([v["x"], v["y"], v["z"]], 1)
    op = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"])))
    means = means[op >= opacity_thresh]
    cells = ((means - origin) / voxel).astype(int)
    keep = np.all((cells >= 0) & (cells < dims), axis=1)
    cells = cells[keep]
    counts = np.zeros(tuple(dims), dtype=np.int32)
    np.add.at(counts, (cells[:, 0], cells[:, 1], cells[:, 2]), 1)
    return counts >= 1


def _score_q3(g, gs_mask, pose, alpha=0.5, n_rays=128, max_steps=40,
              near_surface_iters=0,
              rng=np.random.default_rng(0)):
    """Q3 gain = unknown + alpha * under-reconstructed (along forward rays).
    With near_surface_iters > 0 the 'under-reconstructed' set is restricted to
    free cells that are within `near_surface_iters` voxel dilations of an
    occupied cell — the refined Q3 that suppresses the empty-sky noise."""
    eye = pose[:3, 3]; R = pose[:3, :3]
    cam = rng.standard_normal((n_rays, 3))
    cam[:, 2] = np.abs(cam[:, 2]) + 0.5
    cam /= np.linalg.norm(cam, axis=1, keepdims=True)
    unk = g.unknown(); occ = g.occupied(); free = g.free()
    if near_surface_iters > 0:
        from scipy.ndimage import binary_dilation
        near_surface = binary_dilation(occ, iterations=near_surface_iters)
        under = free & near_surface & ~gs_mask
    else:
        under = free & ~gs_mask
    total = 0.0
    for cd in cam:
        wd = R @ cd
        c = g.world_to_cell(eye)
        if not g.in_bounds(c): continue
        step = np.sign(wd).astype(np.int32)
        d_safe = np.where(np.abs(wd) < 1e-12, 1e-12, wd)
        cw = g.origin + (c + (step > 0)) * g.cell
        t_max = (cw - eye) / d_safe
        t_delta = g.cell / np.abs(d_safe)
        seen_unk = seen_under = 0
        for _ in range(max_steps):
            ax = int(np.argmin(t_max))
            c = c.copy(); c[ax] += step[ax]
            t_max[ax] += t_delta[ax]
            if not g.in_bounds(c): break
            if occ[c[0], c[1], c[2]]: break
            if unk[c[0], c[1], c[2]]: seen_unk += 1
            elif under[c[0], c[1], c[2]]: seen_under += 1
        total += seen_unk + alpha * seen_under
    return total


def plan_q3(poses, imgs, depths, intr, bootstrap, B, work_dir, iterations=1500,
            near_surface_iters=0, max_per_alt=None, seed=None):
    n_extra = B - len(bootstrap)
    g = _build_grid(poses, depths, intr, bootstrap)
    print(f"  octo: occ={int(g.occupied().sum())} free={int(g.free().sum())} unk={int(g.unknown().sum())}")
    print(f"  training sortie-1 GS on {len(bootstrap)} frames ({iterations} iters)...")
    bs_p = poses[bootstrap]; bs_i = [imgs[i] for i in bootstrap]; bs_d = [depths[i] for i in bootstrap]
    ply = _train_mini_gs(bs_p, bs_i, bs_d, intr, work_dir, iterations=iterations, seed=seed)
    gs_mask = _gs_density_mask(ply, g.origin, g.dims, g.cell)
    print(f"  gs voxels: {int(gs_mask.sum())} | near-surface dilations: {near_surface_iters} | altcap: {max_per_alt}")
    cands = [i for i in range(poses.shape[0]) if i not in bootstrap]
    rng = np.random.default_rng(0)
    scores = np.array([_score_q3(g, gs_mask, poses[i],
                                  near_surface_iters=near_surface_iters, rng=rng)
                       for i in cands])
    chosen = _nms_select(cands, scores, n_extra, max_per_alt=max_per_alt)
    return sorted(set(bootstrap + chosen))


# ─── train / parse ──────────────────────────────────────────────

def train_one(source_dir, model_dir, iterations=7000, seed=None):
    cmd = [PY, _train_entry(seed),
           "-s", str(source_dir), "-m", str(model_dir),
           "--iterations", str(iterations),
           "--disable_viewer",
           "--opacity_reset_interval", "999999",
           "--densify_until_iter", "5000",
           "--eval"]
    print("  RUN:", " ".join(cmd), f"(seed={seed})" if seed is not None else "")
    r = subprocess.run(cmd, cwd=str(GS_REPO), env=_train_env(seed),
                       capture_output=True, text=True)
    return r.stdout + "\n" + r.stderr


def parse_psnr(log_text):
    test_psnr = train_psnr = None
    for line in log_text.splitlines():
        if "Evaluating test" in line and "PSNR" in line:
            test_psnr = float(line.split("PSNR")[-1].strip().split()[0])
        elif "Evaluating train" in line and "PSNR" in line:
            train_psnr = float(line.split("PSNR")[-1].strip().split()[0])
    return test_psnr, train_psnr


# ─── main ──────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=18)
    ap.add_argument("--methods", nargs="+",
                    default=["uniform", "random", "volumetric", "q3"])
    ap.add_argument("--seed", type=int, default=None,
                    help="Override Inria GS RNG seed. Outputs are routed "
                         "to data/wreck_exp1/seed_{N}/... When omitted, "
                         "legacy flat layout + Inria's hardcoded seed=0.")
    args = ap.parse_args()

    # Per-seed output root. Bootstrap, candidate pool, and heldout are
    # always read from the canonical (seed-independent) location.
    out_root = DATA / f"seed_{args.seed}" if args.seed is not None else DATA
    out_root.mkdir(parents=True, exist_ok=True)

    cand_poses, intr, cand_imgs, cand_deps = load_sortie(DATA / "candidates")
    held_poses, _,    held_imgs, _         = load_sortie(DATA / "heldout")
    n_cand = len(cand_poses)
    print(f"loaded {n_cand} candidates, {len(held_poses)} held-out")
    print(f"bootstrap indices: {BOOTSTRAP_INDICES}")
    print(f"seed: {args.seed}  ->  out_root: {out_root}\n")

    results = {}
    for m in args.methods:
        print(f"=== {m} (B={args.budget}) ===")
        if m == "uniform":
            sel = plan_uniform(n_cand, args.budget, BOOTSTRAP_INDICES)
        elif m == "random":
            sel = plan_random(n_cand, args.budget, BOOTSTRAP_INDICES)
        elif m == "volumetric":
            sel = plan_volumetric(cand_poses, cand_deps, intr,
                                  BOOTSTRAP_INDICES, args.budget)
        elif m == "q3":
            sel = plan_q3(cand_poses, cand_imgs, cand_deps, intr,
                          BOOTSTRAP_INDICES, args.budget, out_root / "q3_work",
                          iterations=1500, seed=args.seed)
        elif m == "q3_iter":
            # Exp 2: real multi-sortie loop. Train a full sortie-1 GS (7k iters)
            # then run Q3 on its actual density field, not a 1500-iter bootstrap.
            sel = plan_q3(cand_poses, cand_imgs, cand_deps, intr,
                          BOOTSTRAP_INDICES, args.budget, out_root / "q3_iter_work",
                          iterations=7000, seed=args.seed)
        elif m == "q3_refined":
            # Full refined Q3 = near-surface mask + altitude-capped NMS.
            sel = plan_q3(cand_poses, cand_imgs, cand_deps, intr,
                          BOOTSTRAP_INDICES, args.budget, out_root / "q3_refined_work",
                          iterations=7000, near_surface_iters=2, max_per_alt=4,
                          seed=args.seed)
        elif m == "q3_altcap":
            # Ablation: altitude cap only (naive under-rec score).
            sel = plan_q3(cand_poses, cand_imgs, cand_deps, intr,
                          BOOTSTRAP_INDICES, args.budget, out_root / "q3_altcap_work",
                          iterations=7000, near_surface_iters=0, max_per_alt=4,
                          seed=args.seed)
        elif m == "q3_nearsurf":
            # Ablation: near-surface mask only (no altitude cap).
            sel = plan_q3(cand_poses, cand_imgs, cand_deps, intr,
                          BOOTSTRAP_INDICES, args.budget, out_root / "q3_nearsurf_work",
                          iterations=7000, near_surface_iters=2, max_per_alt=None,
                          seed=args.seed)
        else:
            print(f"  unknown: {m}"); continue
        print(f"  selected ({len(sel)}): {sel}")

        ds  = out_root / f"ds_{m}_B{args.budget}"
        mdl = out_root / f"model_{m}_B{args.budget}"
        sp  = cand_poses[sel]
        si  = [cand_imgs[i] for i in sel]
        sd  = [cand_deps[i] for i in sel]
        make_dataset(ds, sp, si, sd, held_poses, held_imgs, intr)
        log = train_one(ds, mdl, seed=args.seed)
        test_psnr, train_psnr = parse_psnr(log)
        print(f"  test PSNR  = {test_psnr}")
        print(f"  train PSNR = {train_psnr}\n")
        results[m] = {"selected": list(map(int, sel)),
                      "test_psnr": test_psnr,
                      "train_psnr": train_psnr}

    out = out_root / f"results_B{args.budget}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")
    print("\nSUMMARY (held-out PSNR per method, B={}):".format(args.budget))
    for m, r in results.items():
        print(f"  {m:12s} : {r['test_psnr']:.2f} dB  (train {r['train_psnr']:.2f} dB)")


if __name__ == "__main__":
    main()
