"""Final 5-method comparison summary: combines PSNR, GT-mesh surface
coverage, occupancy-grid stats, and path-length efficiency into a single
CSV + a Markdown summary table.

Run after metrics.jsonl, surface_coverage.json, and occupancy_summary.json
all exist in the benchmark output root.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main(bench_root: Path):
    # Load all three sources
    metrics = {}
    for m_dir in bench_root.iterdir():
        if not m_dir.is_dir(): continue
        f = m_dir / "metrics.jsonl"
        if not f.exists(): continue
        rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        metrics[m_dir.name] = rows

    cov_rows = json.loads((bench_root / "surface_coverage.json").read_text())["rows"]
    cov_by = {}
    for r in cov_rows:
        cov_by.setdefault(r["method"], []).append(r)
    for m in cov_by:
        cov_by[m].sort(key=lambda r: r["sortie"])

    occ_summary = json.loads((bench_root / "occupancy_summary.json").read_text())
    occ_by = {r["method"]: r for r in occ_summary}

    # Method ordering: classical -> GS-based -> ours
    order = ["frontier", "volumetric", "reactive_gs", "deliberative", "gs_completion"]
    methods = [m for m in order if m in metrics]

    # Build final-sortie table
    rows = []
    for m in methods:
        last_m = metrics[m][-1]
        last_c = cov_by[m][-1] if m in cov_by else {"surface_coverage": float("nan")}
        occ = occ_by.get(m, {})
        rows.append({
            "method": m,
            "final_psnr_mean": round(last_m["psnr_mean"], 2),
            "final_psnr_median": round(last_m["psnr_median"], 2),
            "final_cov_25db": round(last_m["coverage"].get("25.0", 0) * 100, 1),
            "final_cov_20db": round(last_m["coverage"].get("20.0", 0) * 100, 1),
            "surface_coverage_pct": round(last_c["surface_coverage"] * 100, 2),
            "path_length_m": round(last_m["path_length"], 2),
            "coverage_per_meter": round(last_c["surface_coverage"] * 100
                                         / max(0.1, last_m["path_length"]), 2),
            "occupied_cells": occ.get("occupied_cells"),
            "free_cells": occ.get("free_cells"),
            "unknown_cells": occ.get("unknown_cells"),
        })

    csv_path = bench_root / "final_comparison.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {csv_path}")

    # Markdown
    md_lines = ["| Method | PSNR | Surf. cov. % | Path (m) | %/m | cov@20dB | Occ cells | Unk cells |",
                "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md_lines.append(
            f"| {r['method']} | {r['final_psnr_mean']} | {r['surface_coverage_pct']} | "
            f"{r['path_length_m']} | {r['coverage_per_meter']} | {r['final_cov_20db']}% | "
            f"{r['occupied_cells']} | {r['unknown_cells']} |"
        )
    md_path = bench_root / "final_comparison.md"
    md_path.write_text("# Final 5-method comparison (K=3, 7k iters, apartment_1)\n\n"
                       + "\n".join(md_lines) + "\n")
    print(f"wrote {md_path}")
    print()
    print("\n".join(md_lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args()
    main(Path(a.out))
