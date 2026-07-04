#!/usr/bin/env python
"""Phase 5b: INT8 quantization of the FP32 ONNX student + parity / evaluation.

Static INT8 quantization with ONNX Runtime, calibrated on a small validation-split
subset (``deploy.quant.calibration_traces``, default 500). Falls back to dynamic
quantization if static calibration is unavailable (documented in the report).

To make the "keep the head FP32 if picks degrade" decision data-driven (per the
Phase 5 spec), two variants are quantized and evaluated:
  - ``full``      — every eligible op quantized to INT8
  - ``head_fp32`` — the 1x1 head conv (``/head/Conv``) left in FP32, body INT8
The better variant (highest test F1, tie-broken by pick MAE) is shipped as the
INT8 artifact; the report documents both.

Checks performed
----------------
- parity: PyTorch vs FP32 ONNX, and FP32 ONNX vs each INT8 variant, on a dummy
  (1,3,6000) input and a small real test batch (shape / max-abs / mean-abs err)
- lightweight evaluation of FP32 ONNX and each INT8 variant on the test split
  (or a subset): detection P/R/F1 (+FP/FN) at the Stage 2 recommended threshold,
  and P/S pick MAE (ms) within tolerance — reusing evaluate.py helpers

Does NOT benchmark latency, build the streaming wrapper, or retrain.

Usage:
    python scripts/quantize_onnx.py --config configs/default.yaml \
        --checkpoint checkpoints/stage2_distill/best.pt \
        --fp32-onnx outputs/onnx/stage2_distill.onnx \
        --int8-out  outputs/onnx/stage2_distill_int8.onnx \
        --threshold 0.80 --eval-out outputs/stage2_int8_eval
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from seismic_edge_picker.config import load_config  # noqa: E402
from seismic_edge_picker.dataset import build_datasets  # noqa: E402
from seismic_edge_picker.model import build_model, count_parameters  # noqa: E402
from seismic_edge_picker import splits as S  # noqa: E402

# reuse the Phase 4 metric primitives — no re-implementation
from evaluate import (  # noqa: E402
    pick_from_stream, detect_event, prf, residual_stats,
)

INPUT_NAME = "waveform"
OUTPUT_NAME = "streams"
HEAD_NODE = "/head/Conv"  # 1x1 head conv (excluded when keeping the head FP32)


def parse_args():
    p = argparse.ArgumentParser(description="INT8-quantize the FP32 ONNX student.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--checkpoint", default="checkpoints/stage2_distill/best.pt")
    p.add_argument("--fp32-onnx", default="outputs/onnx/stage2_distill.onnx")
    p.add_argument("--int8-out", default="outputs/onnx/stage2_distill_int8.onnx")
    p.add_argument("--report", default="outputs/onnx/quantization_report.json")
    p.add_argument("--eval-out", default="outputs/stage2_int8_eval")
    p.add_argument("--calib-traces", type=int, default=None,
                   help="default: deploy.quant.calibration_traces")
    p.add_argument("--threshold", type=float, default=0.80,
                   help="detection threshold for eval (Stage 2 recommended max-F1 point)")
    p.add_argument("--eval-limit", type=int, default=0,
                   help="test traces to evaluate (0 = full test split)")
    p.add_argument("--eval-batch", type=int, default=32)
    p.add_argument("--tol", type=float, default=1e-4,
                   help="PyTorch/FP32 parity tolerance (reported for reference)")
    return p.parse_args()


def load_student(cfg, ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def diff_stats(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.abs(a - b)
    return {
        "output_shape": list(b.shape),
        "shapes_match": list(a.shape) == list(b.shape),
        "max_abs_err": float(d.max()),
        "mean_abs_err": float(d.mean()),
    }


class ListCalibrationReader:
    """Feeds pre-built {input: (1,3,N)} feed dicts to ORT static quantization."""

    def __init__(self, batches):
        self._batches = batches
        self._iter = iter(batches)

    def get_next(self):
        return next(self._iter, None)

    def rewind(self):
        self._iter = iter(self._batches)


def quantize_variant(model_for_quant, fp32_path, out_path, reader, exclude, quant_api):
    """Static-quantize one variant; fall back to dynamic. Returns (mode, note)."""
    quantize_static, quantize_dynamic, QuantType, QuantFormat, CalibrationMethod = quant_api
    try:
        reader.rewind()
        quantize_static(
            model_input=model_for_quant,
            model_output=str(out_path),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QUInt8,
            calibrate_method=CalibrationMethod.MinMax,
            nodes_to_exclude=exclude,
        )
        return "static", None
    except Exception as exc:
        note = (f"static quantization failed ({exc.__class__.__name__}: {exc}); "
                "fell back to dynamic quantization — activations quantized at runtime "
                "(no calibration), typically less accurate than static")
        print(f"[fallback] {note}")
        quantize_dynamic(model_input=str(fp32_path), model_output=str(out_path),
                         weight_type=QuantType.QInt8, nodes_to_exclude=exclude)
        return "dynamic", note


def preload_test(cfg, test_ds, limit):
    """Read the test split once into memory: inputs (N,3,L) + per-trace ground truth."""
    n = len(test_ds) if not limit else min(limit, len(test_ds))
    meta = test_ds.metadata
    rows = test_ds.rows
    cat = meta[S.COL_CATEGORY].astype(str)
    is_eq = cat.isin(S.EARTHQUAKE_VALUES).to_numpy()
    X = np.empty((n, cfg.data.n_channels, cfg.data.window_samples), dtype=np.float32)
    gt = []
    for i in range(n):
        X[i] = test_ds[i][0].numpy()
        m = meta.iloc[int(rows[i])]
        gt.append({
            "is_eq": bool(is_eq[int(rows[i])]),
            "gt_p": S.parse_scalar(m.get(S.COL_P)),
            "gt_s": S.parse_scalar(m.get(S.COL_S)),
        })
    return X, gt, n


def evaluate_preloaded(session, X, gt, cfg, threshold, batch_size):
    """Run an ORT session over preloaded inputs; return summarized metrics + time."""
    fs = cfg.data.sampling_rate
    ev = cfg.eval
    peak_distance = max(1, int(round(ev.peak_min_distance_s * fs)))
    tol_ms = ev.match_tolerance_s * 1000.0
    n = X.shape[0]
    records = []
    t0 = time.perf_counter()
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        pred = session.run([OUTPUT_NAME], {INPUT_NAME: X[start:stop]})[0]
        for b, i in enumerate(range(start, stop)):
            det, p_str, s_str = pred[b]
            p_pick, _ = pick_from_stream(p_str, ev.peak_height, peak_distance)
            s_pick, _ = pick_from_stream(s_str, ev.peak_height, peak_distance)
            g = gt[i]
            records.append({**g,
                            "detected": detect_event(det, threshold, peak_distance),
                            "p_pick": p_pick, "s_pick": s_pick})
    infer_s = time.perf_counter() - t0

    tp = sum(1 for r in records if r["is_eq"] and r["detected"])
    fn = sum(1 for r in records if r["is_eq"] and not r["detected"])
    fp = sum(1 for r in records if not r["is_eq"] and r["detected"])
    tn = sum(1 for r in records if not r["is_eq"] and not r["detected"])
    precision, recall, f1 = prf(tp, fp, fn)
    summary = {
        "threshold": threshold, "n_traces": len(records),
        "n_earthquake": tp + fn, "n_noise": fp + tn,
        "detection": {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                      "precision": precision, "recall": recall, "f1": f1},
    }
    for phase, gt_key, pick_key in (("p", "gt_p", "p_pick"), ("s", "gt_s", "s_pick")):
        with_gt = [r for r in records if r["is_eq"] and np.isfinite(r[gt_key])]
        picked = [r for r in with_gt if r[pick_key] is not None]
        res = [(r[pick_key] - r[gt_key]) / fs * 1000.0 for r in picked]
        matched = [x for x in res if abs(x) <= tol_ms]
        summary[f"{phase}_picks"] = {
            "n_ground_truth": len(with_gt), "n_picked": len(picked),
            "n_within_tolerance": len(matched),
            "residuals_within_tol": residual_stats(matched),
        }
    return summary, infer_s


def mae(summary, phase):
    return summary[f"{phase}_picks"]["residuals_within_tol"]["mae_ms"]


def fmt_eval(e):
    d = e["detection"]
    pr, sr = mae(e, "p"), mae(e, "s")
    return (f"F1={d['f1']:.4f} P={d['precision']:.4f} R={d['recall']:.4f} "
            f"FP={d['fp']} FN={d['fn']}  P_MAE={pr:.1f}ms S_MAE={sr:.1f}ms")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    calib_n = args.calib_traces or cfg.deploy.quant.calibration_traces

    fp32_path = Path(args.fp32_onnx)
    if not fp32_path.is_file():
        raise SystemExit(f"FP32 ONNX not found: {fp32_path} (run export_onnx.py first)")
    int8_path = Path(args.int8_out)
    int8_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.checkpoint)
    model, ckpt = load_student(cfg, ckpt_path)
    print(f"checkpoint: {ckpt_path}  (epoch={ckpt.get('epoch')}, "
          f"params={count_parameters(model):,})")

    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import (
        quantize_static, quantize_dynamic, QuantType, QuantFormat, CalibrationMethod,
    )
    quant_api = (quantize_static, quantize_dynamic, QuantType, QuantFormat, CalibrationMethod)

    datasets, _ = build_datasets(cfg)
    val_ds, test_ds = datasets["val"], datasets["test"]

    # ---- calibration set from the validation split -------------------------
    n_calib = min(calib_n, len(val_ds))
    print(f"building calibration set: {n_calib} val traces")
    calib_batches = [{INPUT_NAME: val_ds[i][0].numpy()[None].astype(np.float32)}
                     for i in range(n_calib)]
    reader = ListCalibrationReader(calib_batches)

    # ---- pre-process the FP32 graph (shape inference + opt) for static quant -
    prep_path = int8_path.with_name(int8_path.stem + "_prep.onnx")
    model_for_quant = str(fp32_path)
    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process
        quant_pre_process(str(fp32_path), str(prep_path), skip_symbolic_shape=False)
        model_for_quant = str(prep_path)
        print(f"quant pre-process OK -> {prep_path.name}")
    except Exception as exc:
        print(f"quant pre-process skipped ({exc.__class__.__name__}: {exc})")

    # ---- quantize both variants: full INT8, and body-INT8 / head-FP32 -------
    variants = {
        "full": {"exclude": [], "path": int8_path.with_name(int8_path.stem + "_full.onnx")},
        "head_fp32": {"exclude": [HEAD_NODE], "path": int8_path.with_name(int8_path.stem + "_headfp32.onnx")},
    }
    quant_mode, quant_note = "static", None
    for name, v in variants.items():
        t0 = time.perf_counter()
        mode, note = quantize_variant(model_for_quant, fp32_path, v["path"],
                                      reader, v["exclude"], quant_api)
        onnx.checker.check_model(onnx.load(str(v["path"])))
        v["size_mb"] = v["path"].stat().st_size / 1e6
        quant_mode = mode
        quant_note = note or quant_note
        print(f"[{name:10s}] {mode} INT8 in {time.perf_counter()-t0:.1f}s  "
              f"exclude={v['exclude']}  {v['size_mb']:.3f} MB")
    if prep_path.exists():
        prep_path.unlink()

    fp32_mb = fp32_path.stat().st_size / 1e6

    # ---- sessions -----------------------------------------------------------
    def sess(p):
        return ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    sess_fp32 = sess(fp32_path)
    for v in variants.values():
        v["sess"] = sess(v["path"])

    # ---- parity: PyTorch vs FP32, FP32 vs each INT8 variant ------------------
    n_ch, n_samp = cfg.data.n_channels, cfg.data.window_samples
    rng = np.random.default_rng(0)
    inputs = {
        "dummy": rng.standard_normal((1, n_ch, n_samp)).astype(np.float32),
        "real_test_batch": np.stack([test_ds[i][0].numpy() for i in range(8)]).astype(np.float32),
    }
    parity = {}
    for name, x in inputs.items():
        with torch.no_grad():
            torch_out = model(torch.from_numpy(x)).numpy()
        fp32_out = sess_fp32.run([OUTPUT_NAME], {INPUT_NAME: x})[0]
        entry = {"input_shape": list(x.shape),
                 "pytorch_vs_fp32_onnx": diff_stats(torch_out, fp32_out)}
        for vn, v in variants.items():
            int8_out = v["sess"].run([OUTPUT_NAME], {INPUT_NAME: x})[0]
            entry[f"fp32_onnx_vs_int8_{vn}"] = diff_stats(fp32_out, int8_out)
        parity[name] = entry
        print(f"[{name:16s}] pt/fp32 max={entry['pytorch_vs_fp32_onnx']['max_abs_err']:.2e}  "
              + "  ".join(f"fp32/{vn} max={entry[f'fp32_onnx_vs_int8_{vn}']['max_abs_err']:.2e}"
                         for vn in variants))

    # ---- evaluation: FP32 + both INT8 variants on the test split ------------
    thr = args.threshold
    print(f"\npreloading test split for eval (threshold={thr}, "
          f"{'full' if not args.eval_limit else f'{args.eval_limit} traces'})...")
    X, gt, n_eval = preload_test(cfg, test_ds, args.eval_limit)
    print(f"  {n_eval} traces preloaded")

    fp32_eval, fp32_t = evaluate_preloaded(sess_fp32, X, gt, cfg, thr, args.eval_batch)
    print(f"  FP32 ONNX ({fp32_t:.1f}s): {fmt_eval(fp32_eval)}")
    for vn, v in variants.items():
        v["eval"], v["eval_t"] = evaluate_preloaded(v["sess"], X, gt, cfg, thr, args.eval_batch)
        print(f"  INT8 {vn:10s} ({v['eval_t']:.1f}s): {fmt_eval(v['eval'])}")

    # ---- pick the shipped variant: max F1, tie-break by lower (P+S) MAE -----
    def score(v):
        e = v["eval"]
        ps = (mae(e, "p") or 1e9) + (mae(e, "s") or 1e9)
        return (e["detection"]["f1"], -ps)
    chosen_name = max(variants, key=lambda k: score(variants[k]))
    chosen = variants[chosen_name]
    shutil.copyfile(chosen["path"], int8_path)
    int8_mb = int8_path.stat().st_size / 1e6
    print(f"\nshipping variant '{chosen_name}' -> {int8_path} ({int8_mb:.3f} MB)")
    print(f"size: FP32 {fp32_mb:.3f} MB -> INT8 {int8_mb:.3f} MB "
          f"({fp32_mb/int8_mb:.2f}x, -{100*(1-int8_mb/fp32_mb):.1f}%)")

    # ---- degradation verdict (chosen variant vs FP32 ONNX) ------------------
    ce = chosen["eval"]
    f1_drop = fp32_eval["detection"]["f1"] - ce["detection"]["f1"]
    p_delta = None if (mae(ce, "p") is None or mae(fp32_eval, "p") is None) else mae(ce, "p") - mae(fp32_eval, "p")
    s_delta = None if (mae(ce, "s") is None or mae(fp32_eval, "s") is None) else mae(ce, "s") - mae(fp32_eval, "s")
    detection_hurt = f1_drop > 0.005
    picks_hurt = (p_delta is not None and p_delta > 10.0) or (s_delta is not None and s_delta > 10.0)
    verdict = {
        "chosen_variant": chosen_name,
        "f1_fp32": fp32_eval["detection"]["f1"], "f1_int8": ce["detection"]["f1"],
        "f1_drop_fp32_to_int8": f1_drop,
        "p_mae_delta_ms": p_delta, "s_mae_delta_ms": s_delta,
        "detection_meaningfully_hurt": bool(detection_hurt),
        "picks_meaningfully_hurt": bool(picks_hurt),
        "criteria": "detection hurt if F1 drop > 0.005; picks hurt if P or S MAE increase > 10 ms",
    }
    print(f"verdict [{chosen_name}]: F1 {fp32_eval['detection']['f1']:.4f} -> "
          f"{ce['detection']['f1']:.4f} (Δ{f1_drop:+.4f}); "
          f"P MAE Δ{'n/a' if p_delta is None else f'{p_delta:+.1f}'} ms  "
          f"S MAE Δ{'n/a' if s_delta is None else f'{s_delta:+.1f}'} ms  -> "
          f"{'DEGRADED' if (detection_hurt or picks_hurt) else 'OK (no meaningful loss)'}")

    # ---- report -------------------------------------------------------------
    report = {
        "checkpoint": str(ckpt_path),
        "fp32_onnx": str(fp32_path),
        "int8_onnx": str(int8_path),
        "shipped_variant": chosen_name,
        "quantization": {
            "mode": quant_mode,
            "quant_format": "QDQ" if quant_mode == "static" else "dynamic",
            "weight_type": "QInt8",
            "activation_type": "QUInt8" if quant_mode == "static" else "n/a (dynamic)",
            "per_channel": quant_mode == "static",
            "calibrate_method": "MinMax" if quant_mode == "static" else None,
            "calibration_traces": n_calib if quant_mode == "static" else 0,
            "calibration_split": "val",
            "head_node": HEAD_NODE,
            "note": quant_note,
            "onnxruntime_version": ort.__version__,
        },
        "size": {
            "fp32_mb": fp32_mb, "int8_mb": int8_mb,
            "reduction_x": fp32_mb / int8_mb, "reduction_pct": 100 * (1 - int8_mb / fp32_mb),
            "variants_mb": {k: v["size_mb"] for k, v in variants.items()},
        },
        "parity": parity,
        "evaluation": {
            "split": "test", "n_evaluated": n_eval, "threshold": thr,
            "eval_params": {
                "peak_height": cfg.eval.peak_height,
                "peak_min_distance_s": cfg.eval.peak_min_distance_s,
                "match_tolerance_s": cfg.eval.match_tolerance_s,
            },
            "fp32_onnx": fp32_eval, "fp32_infer_seconds": fp32_t,
            "int8_variants": {k: {"exclude": v["exclude"], "eval": v["eval"],
                                  "infer_seconds": v["eval_t"], "size_mb": v["size_mb"]}
                              for k, v in variants.items()},
        },
        "verdict": verdict,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2))
    eval_out = Path(args.eval_out)
    eval_out.mkdir(parents=True, exist_ok=True)
    (eval_out / "int8_eval.json").write_text(json.dumps(report["evaluation"], indent=2))
    print(f"\nwrote: {args.report}\n       {eval_out / 'int8_eval.json'}\n       {int8_path}")


if __name__ == "__main__":
    main()
