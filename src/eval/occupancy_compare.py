"""Build a side-by-side comparison strip of every method's top-down
occupancy map. Used after `occupancy_export.py` writes the per-method PNGs.

Also emits a compact summary plot of (occupied / free / unknown) cell counts.

CLI::

    python -m src.eval.occupancy_compare \
        --out data/results/apartment_1_K3_7k_v6_hybrid
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np


def build_compare_strip(bench_root: Path, methods: List[str],
                        scale: float = 2.0) -> Path:
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont

    panels = []
    for m in methods:
        p = bench_root / m / "occupancy" / "occupancy_2d.png"
        if not p.exists():
            print(f"[skip] {m}: no occupancy_2d.png")
            continue
        img = imageio.imread(p)[..., :3]
        panels.append((m, img))
    if not panels:
        return None

    H, W = panels[0][1].shape[:2]
    Hs, Ws = int(H * scale), int(W * scale)
    label_h = 28
    pad = 8

    def resize(arr):
        return np.array(Image.fromarray(arr).resize((Ws, Hs), Image.NEAREST))

    n = len(panels)
    grid_w = n * Ws + (n + 1) * pad
    grid_h = Hs + label_h + 2 * pad
    grid = np.full((grid_h, grid_w, 3), 245, dtype=np.uint8)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for i, (name, img) in enumerate(panels):
        x0 = pad + i * (Ws + pad)
        grid[pad+label_h:pad+label_h+Hs, x0:x0+Ws] = resize(img)

    pil = Image.fromarray(grid)
    d = ImageDraw.Draw(pil)
    for i, (name, _) in enumerate(panels):
        x0 = pad + i * (Ws + pad)
        d.text((x0 + 6, pad + 4), name, fill=(0, 0, 0), font=font)
    grid = np.array(pil)

    out = bench_root / "occupancy_compare.png"
    imageio.imwrite(out, grid)
    return out


def stats_bar(bench_root: Path, methods: List[str]) -> Path:
    """Stacked bar chart: per-method (occupied / free / unknown) cell counts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = json.loads((bench_root / "occupancy_summary.json").read_text())
    by_method = {r["method"]: r for r in summary}

    ms = [m for m in methods if m in by_method]
    occ = [by_method[m]["occupied_cells"] for m in ms]
    free = [by_method[m]["free_cells"] for m in ms]
    unk = [by_method[m]["unknown_cells"] for m in ms]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(ms))
    ax.bar(x, occ, label="occupied", color="#d44")
    ax.bar(x, free, bottom=occ, label="free", color="#9e8")
    bottom2 = np.array(occ) + np.array(free)
    ax.bar(x, unk, bottom=bottom2, label="unknown", color="#bbb")
    ax.set_xticks(x); ax.set_xticklabels(ms, rotation=15, ha="right")
    ax.set_ylabel("Voxel count")
    ax.set_title("Occupancy grid breakdown per method (10 cm voxels)")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    out = bench_root / "occupancy_stats.png"
    plt.savefig(out, dpi=120); plt.close()
    return out


def main(bench_root: Path, methods=None):
    if methods is None:
        # Stable order: classical baselines first, then GS-based, then ours
        wanted = ["frontier", "volumetric", "reactive_gs", "deliberative", "gs_completion"]
        methods = [m for m in wanted
                   if (bench_root / m / "occupancy" / "occupancy_2d.png").exists()]
    strip = build_compare_strip(bench_root, methods)
    if strip: print(f"wrote {strip}")
    bar = stats_bar(bench_root, methods)
    if bar: print(f"wrote {bar}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args()
    main(Path(a.out))
