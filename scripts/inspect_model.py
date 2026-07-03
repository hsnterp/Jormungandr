#!/usr/bin/env python
"""Build the model from config and print param count + MFLOPs + I/O shapes.

Runnable with no data. This is the Phase 2 verification entrypoint.

    python scripts/inspect_model.py --config configs/default.yaml
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from seismic_edge_picker.config import load_config          # noqa: E402
from seismic_edge_picker.model import build_model, model_summary  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = build_model(cfg)

    b, ch, n = 2, cfg.data.n_channels, cfg.data.window_samples
    x = torch.zeros(b, ch, n)
    y = model(x)

    summary = model_summary(model, input_shape=(1, ch, n))
    budget = 300_000
    ok = "OK" if summary["parameters"] < budget else "OVER BUDGET"

    print("=" * 60)
    print("SeismicUNet — Phase 2 model summary")
    print("=" * 60)
    print(f"  input shape         : {tuple(x.shape)}")
    print(f"  output shape        : {tuple(y.shape)}")
    print(f"  output range        : [{y.min().item():.4f}, {y.max().item():.4f}] (sigmoid)")
    print(f"  parameters          : {summary['parameters']:,}  (<300k? {ok})")
    print(f"  MFLOPs / 60s window : {summary['mflops']:.2f}")
    print("=" * 60)

    assert y.shape == (b, cfg.model.out_channels, n), "unexpected output shape"
    print("forward pass shape check: PASS")


if __name__ == "__main__":
    main()
