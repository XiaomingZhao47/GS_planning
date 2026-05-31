"""Aggregate per-method metrics.jsonl files into a comparison table + plot.

Reads ``<out_root>/<method>/metrics.jsonl`` for every method present and
emits:
  - <out_root>/summary.csv   : long-form table (method, sortie, ...)
  - <out_root>/coverage.png  : coverage-vs-sortie line plot
  - <out_root>/psnr.png      : mean-PSNR-vs-sortie line plot
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main(out_root: Path):
    rows = []
    methods = []
    for method_dir in sorted(out_root.iterdir()):
        if not method_dir.is_dir(): continue
        mfile = method_dir / "metrics.jsonl"
        if not mfile.exists(): continue
        methods.append(method_dir.name)
        for line in mfile.read_text().splitlines():
            if not line.strip(): continue
            r = json.loads(line)
            rows.append({
                "method": method_dir.name,
                "sortie": r["sortie_idx"],
                "psnr_mean": r["psnr_mean"],
                "psnr_median": r["psnr_median"],
                "cov@20": r["coverage"].get("20.0"),
                "cov@25": r["coverage"].get("25.0"),
                "path_length": r["path_length"],
            })

    csv_path = out_root / "summary.csv"
    if rows:
        with csv_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"wrote {csv_path}  ({len(rows)} rows, methods: {methods})")
    else:
        print("no metrics found")
        return

    # plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots"); return

    by_method = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)

    for metric, ylabel, fname in [
        ("psnr_mean", "Mean PSNR (dB)", "psnr.png"),
        ("cov@20", "Coverage @ PSNR>=20 dB", "coverage_20.png"),
        ("cov@25", "Coverage @ PSNR>=25 dB", "coverage_25.png"),
    ]:
        plt.figure(figsize=(6, 4))
        for m, recs in by_method.items():
            recs.sort(key=lambda r: r["sortie"])
            xs = [r["sortie"] for r in recs]
            ys = [r[metric] for r in recs]
            plt.plot(xs, ys, marker="o", label=m)
        plt.xlabel("Sortie #")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} vs sortie count")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_root / fname, dpi=120)
        plt.close()
        print(f"wrote {out_root / fname}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="benchmark output root (contains <method>/metrics.jsonl)")
    args = p.parse_args()
    main(Path(args.out))
