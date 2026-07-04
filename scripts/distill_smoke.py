#!/usr/bin/env python
"""Tiny END-TO-END Stage-2 smoke: cache a couple dozen teacher outputs, then run
ONE distillation epoch on them. Fast, CPU/GPU-agnostic, and safe to run (it uses
distill.smoke.* caps). This is the ONLY Stage-2 run intended to execute before a
full run is explicitly approved.

    python scripts/distill_smoke.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = os.path.dirname(__file__)


def run(cmd):
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description="End-to-end Stage-2 distillation smoke.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--cache-dir", default="data/teacher_cache_smoke")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    py = sys.executable
    dev = ["--device", args.device] if args.device else []
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    run([py, os.path.join(HERE, "cache_teacher.py"), "--config", args.config,
         "--smoke", "--cache-dir", args.cache_dir, "--batch-size", "8", *dev])
    run([py, os.path.join(HERE, "train_distill.py"), "--config", args.config,
         "--smoke", "--cache-dir", args.cache_dir, *dev])
    print("\nStage-2 smoke OK: tiny teacher cache + 1 distillation epoch completed.")


if __name__ == "__main__":
    main()
