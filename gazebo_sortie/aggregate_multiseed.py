#!/usr/bin/env python3
"""Multi-seed aggregator with the pre-registered sign+magnitude verdict
from docs/adr/0001-multi-seed-evaluation-design.md.

Reads data/wreck_exp1/seed_*/multimetric_results.json and emits:

  data/wreck_exp1/multiseed_summary.json
  data/wreck_exp1/multiseed_summary.md

The verdict for each (Q3_refined, metric) pair, against the strongest
non-Q3 baseline on that metric:

  PASS  iff (a) Q3_refined is ahead in >= K_min of the seeds AND
            (b) |mean_gap| > 1 * seed_std_of_Q3_refined
  FAIL otherwise; that metric is reported as single-seed only.

Defaults: K_min = 4 of 5 seeds. ``ahead'' is direction-aware
(PSNR/SSIM higher-better; LPIPS lower-better; n_gaussians is reported
but not adjudicated -- the model-complexity story is qualitative, not a
pre-registered claim).

Exits non-zero if any (seed, method) cell is missing -- silent partial
aggregation would be misleading.
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
from statistics import mean, stdev

ROOT = Path("/home/xiaoming/GS_planning_handoff")
DATA = ROOT / "data/wreck_exp1"

# Methods to compare against Q3_refined in the verdict. Per the ADR the
# multi-seed sweep covers uniform / volumetric / q3_iter / q3_refined;
# extras (random / q3_altcap / q3_nearsurf) are tolerated if present but
# don't expand the verdict.
CORE_METHODS = ["uniform", "volumetric", "q3_iter", "q3_refined"]
BASELINES = ["uniform", "volumetric", "q3_iter"]  # adjudication targets
METRIC_DIRECTION = {  # +1 = higher-is-better, -1 = lower-is-better
    "PSNR": +1, "SSIM": +1, "LPIPS": -1, "n_gaussians": 0,
}


def _collect(root: Path, expected_seeds=None):
    """Return {seed: {method: {metric: value}}} for every seed_*/ under root.
    Raises SystemExit if any (seed, method) is missing."""
    seed_dirs = sorted(root.glob("seed_*/"))
    if not seed_dirs:
        sys.exit(f"no seed_*/ subdirs under {root}; nothing to aggregate")
    seeds = []
    by_seed: dict[int, dict] = {}
    for sd in seed_dirs:
        n = int(sd.name.split("_", 1)[1])
        results_file = sd / "multimetric_results.json"
        if not results_file.exists():
            sys.exit(f"missing {results_file}; run exp_multimetric --seed {n} first")
        per_method = json.loads(results_file.read_text())
        # Sanity: every CORE_METHODS row must be present and complete.
        for m in CORE_METHODS:
            if m not in per_method:
                sys.exit(f"seed_{n}: method '{m}' missing from {results_file}")
            row = per_method[m]
            for k in ("PSNR", "SSIM", "LPIPS", "n_gaussians"):
                if row.get(k) is None:
                    sys.exit(f"seed_{n}/{m}: metric '{k}' is null in {results_file}")
        by_seed[n] = per_method
        seeds.append(n)
    if expected_seeds is not None and len(seeds) < expected_seeds:
        sys.exit(f"expected >= {expected_seeds} seeds; found {len(seeds)}: {seeds}")
    return seeds, by_seed


def _per_method_stats(seeds, by_seed):
    """{method: {metric: {mean, std, values: [per-seed list]}}}."""
    methods = set()
    for s in seeds:
        methods.update(by_seed[s].keys())
    out = {}
    for m in sorted(methods):
        vals_by_metric = {k: [] for k in METRIC_DIRECTION}
        for s in seeds:
            if m not in by_seed[s]:
                continue
            row = by_seed[s][m]
            for k in vals_by_metric:
                if row.get(k) is not None:
                    vals_by_metric[k].append(float(row[k]))
        per_metric = {}
        for k, vals in vals_by_metric.items():
            if not vals:
                continue
            per_metric[k] = {
                "mean": mean(vals),
                "std": stdev(vals) if len(vals) > 1 else 0.0,
                "values": vals,
            }
        out[m] = per_metric
    return out


def _verdict(stats, k_min):
    """Per-metric verdict for Q3_refined vs each baseline.

    For each baseline and each metric (PSNR/SSIM/LPIPS), compute:
      - per-seed sign: q3_refined ahead of baseline?
      - mean gap (q3_refined_mean - baseline_mean), direction-corrected
      - q3_refined seed-std (the reference noise floor)
      - PASS iff (#ahead >= k_min) AND (|mean_gap| > q3_refined_std)
    """
    out = {}
    if "q3_refined" not in stats:
        sys.exit("aggregator cannot adjudicate without q3_refined; "
                 "rerun with --methods including q3_refined")
    q3 = stats["q3_refined"]
    for b in BASELINES:
        if b not in stats:
            print(f"  WARN: baseline '{b}' missing; skipping in verdict")
            continue
        bstats = stats[b]
        per_metric_verdict = {}
        for metric, direction in METRIC_DIRECTION.items():
            if direction == 0 or metric not in q3 or metric not in bstats:
                continue
            q_vals = q3[metric]["values"]
            b_vals = bstats[metric]["values"]
            n = min(len(q_vals), len(b_vals))
            if n == 0:
                continue
            ahead = sum(
                1 for i in range(n)
                if (q_vals[i] - b_vals[i]) * direction > 0
            )
            mean_gap = (q3[metric]["mean"] - bstats[metric]["mean"]) * direction
            q_std = q3[metric]["std"]
            ok_sign = ahead >= k_min
            ok_mag = abs(mean_gap) > q_std if q_std > 0 else True
            verdict = "PASS" if (ok_sign and ok_mag and mean_gap > 0) else "FAIL"
            per_metric_verdict[metric] = {
                "ahead_count": ahead,
                "of": n,
                "mean_gap_correct_direction": mean_gap,
                "q3_refined_std": q_std,
                "verdict": verdict,
            }
        out[b] = per_metric_verdict
    return out


def _format_md(seeds, stats, verdicts):
    out = []
    out.append(f"# Multi-seed summary (n={len(seeds)} seeds: {seeds})\n")
    out.append("## Per-method mean ± std\n")
    out.append("| Method | PSNR (dB) | SSIM | LPIPS | # Gaussians |")
    out.append("|---|---|---|---|---|")
    for m, per_metric in stats.items():
        def cell(k, fmt):
            if k not in per_metric: return ""
            return f"{per_metric[k]['mean']:{fmt}} ± {per_metric[k]['std']:{fmt}}"
        out.append(f"| {m} | {cell('PSNR', '.2f')} | {cell('SSIM', '.4f')} | {cell('LPIPS', '.4f')} | {cell('n_gaussians', '.0f')} |")
    out.append("")
    out.append("## Pre-registered verdict (Q3_refined vs baselines)\n")
    out.append("Per metric: PASS iff Q3_refined ahead in ≥ K_min seeds AND |mean_gap| > Q3_refined seed-std.\n")
    out.append("| Baseline | Metric | Ahead | Mean gap (Q3-refined direction) | Q3_refined std | Verdict |")
    out.append("|---|---|---|---|---|---|")
    for b, per_metric in verdicts.items():
        for metric, v in per_metric.items():
            out.append(
                f"| {b} | {metric} | {v['ahead_count']}/{v['of']} "
                f"| {v['mean_gap_correct_direction']:+.4f} "
                f"| {v['q3_refined_std']:.4f} | **{v['verdict']}** |"
            )
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DATA,
                    help="root containing seed_*/ subdirs")
    ap.add_argument("--k-min", type=int, default=4,
                    help="minimum #ahead seeds to pass the sign criterion (default 4)")
    ap.add_argument("--expected-seeds", type=int, default=None,
                    help="error out if fewer seeds are present")
    args = ap.parse_args()

    seeds, by_seed = _collect(args.root, args.expected_seeds)
    print(f"aggregating seeds: {seeds}")
    stats = _per_method_stats(seeds, by_seed)
    verdicts = _verdict(stats, args.k_min)

    summary = {
        "seeds": seeds,
        "k_min": args.k_min,
        "per_method": stats,
        "verdict_vs_baselines": verdicts,
    }
    out_json = args.root / "multiseed_summary.json"
    out_md = args.root / "multiseed_summary.md"
    out_json.write_text(json.dumps(summary, indent=2))
    out_md.write_text(_format_md(seeds, stats, verdicts))
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    print("\n--- verdict ---")
    for b, per_metric in verdicts.items():
        for metric, v in per_metric.items():
            print(f"  {b:12s} {metric:8s} {v['ahead_count']}/{v['of']} "
                  f"gap={v['mean_gap_correct_direction']:+.4f} std={v['q3_refined_std']:.4f} -> {v['verdict']}")


if __name__ == "__main__":
    main()
