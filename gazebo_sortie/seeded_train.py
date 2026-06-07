#!/usr/bin/env python3
"""Wrapper around Inria gaussian-splatting/train.py that overrides its
hardcoded RNG seed (utils.general_utils.safe_state -> torch.manual_seed(0))
with a user-supplied seed via the GS_SEED env var.

Invoke exactly like train.py; the wrapper consumes the same argv. The
shared `gaussian-splatting/` repo is NOT modified.

Usage (as the gs_planning conda env's python):
    GS_SEED=7 python gazebo_sortie/seeded_train.py -s <source> -m <model> [...]

Internally:
  1) Imports utils.general_utils.safe_state and re-binds it so the body
     re-seeds with int(GS_SEED) instead of 0 (random, numpy, torch).
  2) runpy's train.py as __main__ so its argparse + training loop run unchanged.

The wrapper is idempotent at seed 0: omit GS_SEED to reproduce the
upstream behaviour bit-for-bit.
"""
from __future__ import annotations
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

GS_REPO = Path("/home/xiaoming/gaussian-splatting")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _install_safe_state_override(seed: int) -> None:
    """Replace utils.general_utils.safe_state with a version that seeds
    with `seed` instead of 0, while preserving the rest of the original
    behaviour (silent stdout, CUDA device selection)."""
    sys.path.insert(0, str(GS_REPO))
    import utils.general_utils as gu

    original = gu.safe_state

    def patched(silent: bool) -> None:
        # Re-seed first so any later RNG draws use `seed`.
        _seed_everything(seed)
        # Replay the rest of the original safe_state, but skip its own seeding.
        # Inspecting the upstream body, the only non-seeding work is the
        # stdout silencer and torch.cuda.set_device. We call the original and
        # then re-seed once more to defeat the in-body torch.manual_seed(0).
        original(silent)
        _seed_everything(seed)

    gu.safe_state = patched


def main() -> None:
    seed = int(os.environ.get("GS_SEED", "0"))
    print(f"[seeded_train] GS_SEED={seed}")
    _install_safe_state_override(seed)

    # Hand argv to train.py and execute it as __main__ so its argparse runs.
    train_py = GS_REPO / "train.py"
    if not train_py.exists():
        sys.stderr.write(f"[seeded_train] train.py not found at {train_py}\n")
        sys.exit(2)

    # runpy doesn't accept argv; we set sys.argv directly. Replace argv[0]
    # with train.py's path so Inria's argparse messages remain coherent.
    sys.argv = [str(train_py)] + sys.argv[1:]
    # Make the GS repo cwd so relative paths inside train.py resolve.
    os.chdir(str(GS_REPO))
    import runpy
    runpy.run_path(str(train_py), run_name="__main__")


if __name__ == "__main__":
    main()
