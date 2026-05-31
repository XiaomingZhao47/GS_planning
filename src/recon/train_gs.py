"""Thin wrapper around the Inria 3DGS train.py that points it at a
NeRF-Synthetic-format directory produced by sortie_to_nerf_synthetic.py.

We invoke train.py as a subprocess (rather than importing) because the Inria
script registers ArgParser groups at import time and is awkward to call as a
library. This keeps the wrapper simple and pinned to the upstream behavior.

Usage::

    python -m src.recon.train_gs \
        --source GS_planning/data/sorties_nerf/k1 \
        --model GS_planning/data/gs_models/k1 \
        --iterations 7000
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

GS_REPO_DEFAULT = Path("/home/xiaoming/Desktop/2026_sonar_GS/gaussian-splatting")


def train_gs(source: Path, model_out: Path, iterations: int = 7000,
             gs_repo: Path = GS_REPO_DEFAULT, extra: list[str] | None = None) -> int:
    source = Path(source).resolve()
    model_out = Path(model_out).resolve()
    model_out.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(gs_repo / "train.py"),
        "-s", str(source),
        "-m", str(model_out),
        "--iterations", str(iterations),
        # NeRF-Synthetic reader path triggers when transforms_train.json exists.
        # --disable_viewer is essential when multiple trainings run in parallel
        # (the GUI server binds port 6009 by default and would collide).
        "--disable_viewer",
        # On small per-sortie datasets (8-20 views), the default opacity reset
        # at iter 3000 prunes too aggressively and leaves 0 visible Gaussians
        # from some viewpoints -- the diff_gaussian_rasterizer 0.0.0 backward
        # pass then crashes with a shape mismatch. Push the reset out of range.
        "--opacity_reset_interval", "999999",
        # Cap densification early; otherwise it keeps spawning Gaussians from
        # noisy late-iteration gradients and inflates the field unnecessarily.
        "--densify_until_iter", "5000",
    ]
    if extra:
        cmd.extend(extra)
    print("RUN:", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(gs_repo))


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--iterations", type=int, default=7000)
    p.add_argument("--gs-repo", default=str(GS_REPO_DEFAULT))
    p.add_argument("extra", nargs="*",
                   help="extra args forwarded to Inria train.py (after --)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    rc = train_gs(Path(args.source), Path(args.model), args.iterations,
                  Path(args.gs_repo), args.extra)
    sys.exit(rc)
