"""Plot surface coverage vs sortie count from surface_coverage.json files.

Usage::
    python -m src.eval.plot_coverage \
        --runs data/results/apartment_1_K3_7k_v4_no_translation \
               data/results/apartment_1_K3_7k_v5_translation \
        --labels v4 v5 \
        --out data/results/coverage_v4_v5.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(runs, labels, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    style_per_label = ["-", "--", ":", "-."][: len(runs)]
    color_per_method = {
        "frontier": "tab:orange",
        "volumetric": "tab:red",
        "reactive_gs": "tab:green",
        "deliberative": "tab:blue",
    }
    for run, label, ls in zip(runs, labels, style_per_label):
        cov_file = Path(run) / "surface_coverage.json"
        if not cov_file.exists():
            print(f"skip {run}: no surface_coverage.json"); continue
        rows = json.loads(cov_file.read_text())["rows"]
        by_method = {}
        for r in rows:
            by_method.setdefault(r["method"], []).append(r)
        for m, recs in by_method.items():
            recs.sort(key=lambda r: r["sortie"])
            xs = [r["sortie"] for r in recs]
            ys = [100 * r["surface_coverage"] for r in recs]
            ax.plot(xs, ys, marker="o", linestyle=ls,
                    color=color_per_method.get(m, "gray"),
                    label=f"{m} ({label})")
    ax.set_xlabel("Sortie #")
    ax.set_ylabel("Surface coverage (%)")
    ax.set_title("GT-mesh surface coverage vs sortie count")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    main(a.runs, a.labels, a.out)
