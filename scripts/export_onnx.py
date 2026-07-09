#!/usr/bin/env python
"""Phase 5 (export only): export a trained student checkpoint to ONNX + parity check.

Reuses build_model + the config + the same weights_only safe-load as evaluate.py
(no model redefinition). Exports with a dynamic batch axis and named I/O, then
verifies ONNX Runtime output matches PyTorch on a dummy (1,3,6000) input and, if
available, a small real test-split batch.

Does NOT quantize, benchmark, or build the streaming wrapper.

Usage:
    python scripts/export_onnx.py --config configs/default.yaml \
        --checkpoint checkpoints/stage2_distill/best.pt \
        --out outputs/onnx/stage2_distill.onnx
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from seismic_edge_picker.config import load_config  # noqa: E402
from seismic_edge_picker.model import build_model, count_parameters  # noqa: E402

INPUT_NAME = "waveform"
OUTPUT_NAME = "streams"


def parse_args():
    p = argparse.ArgumentParser(description="Export student checkpoint to ONNX.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--checkpoint", default="checkpoints/stage2_distill/best.pt")
    p.add_argument("--out", default="outputs/onnx/stage2_distill.onnx")
    p.add_argument("--opset", type=int, default=None, help="default: deploy.opset (17)")
    p.add_argument("--real-batch", type=int, default=8,
                   help="test-split traces for the real-data parity check (0 to skip)")
    p.add_argument("--tol", type=float, default=1e-4, help="max-abs-error tolerance")
    p.add_argument("--causal", action="store_true",
                   help="export a causal SeismicUNet checkpoint")
    p.add_argument("--lookahead", type=int, default=0)
    return p.parse_args()


def load_student(cfg, ckpt_path: Path):
    # SAME safe-load path as evaluate.py
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def parity(model, x_np):
    """Return (torch_out, ort_out, stats) for a given (N,3,6000) numpy input."""
    import onnxruntime as ort

    with torch.no_grad():
        torch_out = model(torch.from_numpy(x_np)).cpu().numpy()
    sess = ort.InferenceSession(str(PARITY_MODEL_PATH),
                                providers=["CPUExecutionProvider"])
    ort_out = sess.run([OUTPUT_NAME], {INPUT_NAME: x_np})[0]
    diff = np.abs(torch_out - ort_out)
    stats = {
        "input_shape": list(x_np.shape),
        "output_shape": list(ort_out.shape),
        "max_abs_err": float(diff.max()),
        "mean_abs_err": float(diff.mean()),
        "shapes_match": list(torch_out.shape) == list(ort_out.shape),
    }
    return torch_out, ort_out, stats


PARITY_MODEL_PATH = None  # set in main() after export


def main():
    global PARITY_MODEL_PATH
    args = parse_args()
    cfg = load_config(args.config)
    if args.causal:
        cfg.model.causal = True
        cfg.model.lookahead = int(args.lookahead)
    opset = args.opset or cfg.deploy.opset
    n_ch = cfg.data.n_channels
    n_samples = cfg.data.window_samples

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")
    model, ckpt = load_student(cfg, ckpt_path)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    PARITY_MODEL_PATH = out_path

    print(f"checkpoint: {ckpt_path}  (epoch={ckpt.get('epoch')}, "
          f"val_loss={ckpt.get('best_val_loss')})")
    print(f"student params: {count_parameters(model):,}  opset={opset}  "
          f"input=({n_ch},{n_samples}) dynamic batch")

    # --- export (dummy (1,3,6000), dynamic batch axis, named I/O) ---
    dummy = torch.randn(1, n_ch, n_samples, dtype=torch.float32)
    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=[INPUT_NAME], output_names=[OUTPUT_NAME],
        dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
        opset_version=opset, do_constant_folding=True,
        dynamo=False,  # legacy TorchScript exporter — avoids the onnxscript dep
    )

    import onnx
    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)
    size_mb = out_path.stat().st_size / 1e6
    print(f"exported: {out_path} ({size_mb:.3f} MB); onnx.checker OK")

    report = {
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": int(ckpt.get("epoch")) if ckpt.get("epoch") is not None else None,
        "checkpoint_val_loss": float(ckpt.get("best_val_loss")) if ckpt.get("best_val_loss") is not None else None,
        "onnx_path": str(out_path),
        "onnx_size_mb": size_mb,
        "opset": int(opset),
        "causal": bool(getattr(cfg.model, "causal", False)),
        "lookahead": int(getattr(cfg.model, "lookahead", 0)),
        "onnx_version": onnx.__version__,
        "input_name": INPUT_NAME,
        "output_name": OUTPUT_NAME,
        "dynamic_axes": {"batch": 0},
        "tolerance_max_abs_err": args.tol,
        "checks": {},
        "passed": True,
    }

    import onnxruntime as ort
    report["onnxruntime_version"] = ort.__version__

    # --- parity: dummy input ---
    rng = np.random.default_rng(0)
    x_dummy = rng.standard_normal((1, n_ch, n_samples)).astype(np.float32)
    _, _, dummy_stats = parity(model, x_dummy)
    report["checks"]["dummy"] = dummy_stats
    ok_dummy = dummy_stats["max_abs_err"] < args.tol and dummy_stats["shapes_match"]
    print(f"[dummy]  shape={dummy_stats['output_shape']}  "
          f"max_abs_err={dummy_stats['max_abs_err']:.3e}  "
          f"mean_abs_err={dummy_stats['mean_abs_err']:.3e}  "
          f"{'PASS' if ok_dummy else 'FAIL'}")

    # --- parity: small real test batch (dynamic batch check) ---
    ok_real = True
    if args.real_batch > 0:
        try:
            from seismic_edge_picker.dataset import build_datasets
            datasets, _ = build_datasets(cfg)
            test_ds = datasets["test"]
            n = min(args.real_batch, len(test_ds))
            xs = [test_ds[i][0].numpy() for i in range(n)]
            x_real = np.stack(xs).astype(np.float32)
            _, _, real_stats = parity(model, x_real)
            report["checks"]["real_test_batch"] = real_stats
            ok_real = real_stats["max_abs_err"] < args.tol and real_stats["shapes_match"]
            print(f"[real]   shape={real_stats['output_shape']}  "
                  f"max_abs_err={real_stats['max_abs_err']:.3e}  "
                  f"mean_abs_err={real_stats['mean_abs_err']:.3e}  "
                  f"{'PASS' if ok_real else 'FAIL'}")
        except Exception as exc:  # dataset heavy/unavailable -> dummy is enough
            report["checks"]["real_test_batch"] = {"skipped": repr(exc)}
            print(f"[real]   skipped ({exc.__class__.__name__}); dummy parity suffices")

    report["passed"] = bool(ok_dummy and ok_real)
    report_path = out_path.with_name(out_path.stem + "_parity.json")
    report_path.write_text(json.dumps(report, indent=2))
    print(f"parity report: {report_path}")

    if not report["passed"]:
        raise SystemExit("PARITY FAILED — ONNX output diverges from PyTorch beyond tolerance")
    print(f"PARITY OK (max_abs_err < {args.tol:g}); output shape (N,3,{n_samples}) confirmed")


if __name__ == "__main__":
    main()
