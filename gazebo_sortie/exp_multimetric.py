#!/usr/bin/env python3
"""Multi-metric evaluation across wreck-experiment planners.

For each model under <root>/model_<method>_B18:
  - ensure test views are rendered (render.py --skip_train if missing)
  - run Inria metrics.py -> PSNR, SSIM, LPIPS in results.json
  - count Gaussians in the trained ply
  - record selected indices + bootstrap

Outputs a consolidated table to <root>/multimetric_results.json
and a markdown summary.

By default <root> is data/wreck_exp1/ (legacy flat layout). With --seed N
<root> becomes data/wreck_exp1/seed_{N}/, matching the layout produced by
`exp1_run.py --seed N`.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path

ROOT = Path("/home/xiaoming/GS_planning_handoff")
GS   = Path("/home/xiaoming/gaussian-splatting")
PY   = str(Path.home() / "miniconda3/envs/gs_planning/bin/python")
DATA = ROOT / "data/wreck_exp1"

METHODS = ["uniform", "random", "volumetric",
           "q3_iter", "q3_nearsurf", "q3_altcap", "q3_refined"]

def count_gaussians(model_dir: Path, iter_=7000):
    ply = model_dir / "point_cloud" / f"iteration_{iter_}" / "point_cloud.ply"
    if not ply.exists(): return None
    # PLY header: count from line "element vertex N"
    with open(ply, "rb") as f:
        for _ in range(40):
            line = f.readline().decode("latin-1", errors="ignore").strip()
            if line.startswith("element vertex"):
                return int(line.split()[2])
    return None


def ensure_renders(model_dir: Path):
    renders = model_dir / "test" / "ours_7000" / "renders"
    if renders.exists() and any(renders.iterdir()):
        return True
    print(f"  rendering test views for {model_dir.name}...")
    cmd = [PY, str(GS / "render.py"), "-m", str(model_dir), "--skip_train"]
    r = subprocess.run(cmd, cwd=str(GS), capture_output=True, text=True)
    return r.returncode == 0


def run_metrics(model_dir: Path):
    cmd = [PY, str(GS / "metrics.py"), "-m", str(model_dir)]
    subprocess.run(cmd, cwd=str(GS), capture_output=True, text=True)
    rj = model_dir / "results.json"
    if not rj.exists(): return None
    return json.loads(rj.read_text()).get("ours_7000", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None,
                    help="Read from data/wreck_exp1/seed_{N}/ instead of "
                         "the flat legacy layout.")
    args = ap.parse_args()
    root = (DATA / f"seed_{args.seed}") if args.seed is not None else DATA
    print(f"root: {root}")

    summary = {}
    for m in METHODS:
        mdir = root / f"model_{m}_B18"
        if not mdir.exists():
            print(f"SKIP {m}: model dir missing ({mdir})")
            continue
        print(f"=== {m} ===")
        ensure_renders(mdir)
        metrics = run_metrics(mdir)
        gauss = count_gaussians(mdir)
        summary[m] = {"PSNR": metrics.get("PSNR") if metrics else None,
                      "SSIM": metrics.get("SSIM") if metrics else None,
                      "LPIPS": metrics.get("LPIPS") if metrics else None,
                      "n_gaussians": gauss}
        print(f"  PSNR  {summary[m]['PSNR']:.3f}  SSIM {summary[m]['SSIM']:.4f}  LPIPS {summary[m]['LPIPS']:.4f}  |  {gauss} Gaussians")

    out = root / "multimetric_results.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")

    # markdown table
    md = ["| Method | PSNR ↑ (dB) | SSIM ↑ | LPIPS ↓ | # Gaussians |",
          "|---|---|---|---|---|"]
    for m, r in summary.items():
        if r["PSNR"] is None: continue
        md.append(f"| {m} | {r['PSNR']:.2f} | {r['SSIM']:.4f} | {r['LPIPS']:.4f} | {r['n_gaussians']:,} |")
    (root / "multimetric_results.md").write_text("\n".join(md) + "\n")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
