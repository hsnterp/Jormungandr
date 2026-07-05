#!/usr/bin/env python
"""Phase 5c: single-window CPU latency benchmark for deployment artifacts.

Benchmarks the Stage 2 PyTorch checkpoint, FP32 ONNX, and INT8 ONNX with the
same fixed float32 input of shape (1, 3, 6000). Warmup calls are excluded from
the measured samples. CPU execution is single-threaded by default for both
PyTorch and ONNX Runtime.

PyTorch is optional: on hosts without it installed (e.g. a Raspberry Pi
ONNX-only deployment), the PyTorch backend is skipped with a printed message
and only the FP32/INT8 ONNX Runtime backends are benchmarked.

This script does not load STEAD, retrain, quantize, or perform streaming work.

Usage:
    python scripts/benchmark_latency.py --config configs/default.yaml \
        --checkpoint checkpoints/stage2_distill/best.pt \
        --fp32-onnx outputs/onnx/stage2_distill.onnx \
        --int8-onnx outputs/onnx/stage2_distill_int8.onnx \
        --out-dir outputs/latency
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

# Set common CPU runtime defaults before importing numerical frameworks.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from seismic_edge_picker.config import load_config  # noqa: E402

INPUT_NAME = "waveform"
OUTPUT_NAME = "streams"


def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark PyTorch, FP32 ONNX, and INT8 ONNX latency on CPU."
    )
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--checkpoint", default="checkpoints/stage2_distill/best.pt")
    p.add_argument("--fp32-onnx", default="outputs/onnx/stage2_distill.onnx")
    p.add_argument("--int8-onnx", default="outputs/onnx/stage2_distill_int8.onnx")
    p.add_argument("--out-dir", default="outputs/latency")
    p.add_argument("--warmup-runs", type=int, default=None,
                   help="default: deploy.benchmark.warmup_runs (20)")
    p.add_argument("--runs", type=int, default=None,
                   help="default: deploy.benchmark.n_runs (200)")
    p.add_argument("--threads", type=int, default=None,
                   help="default: deploy.benchmark.threads (1)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def positive_int(name, value):
    if value < 1:
        raise SystemExit(f"{name} must be >= 1 (got {value})")
    return value


def load_student(cfg, checkpoint):
    from seismic_edge_picker.model import build_model

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def latency_summary(samples_ms):
    values = np.asarray(samples_ms, dtype=np.float64)
    mean_ms = float(values.mean())
    return {
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "mean_ms": mean_ms,
        "throughput_windows_per_s": 1000.0 / mean_ms,
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
        "samples_ms": [float(x) for x in values],
    }


def benchmark(call, warmup_runs, measured_runs):
    for _ in range(warmup_runs):
        call()

    samples_ms = []
    for _ in range(measured_runs):
        start_ns = time.perf_counter_ns()
        call()
        samples_ms.append((time.perf_counter_ns() - start_ns) / 1e6)
    return latency_summary(samples_ms)


def cpu_name():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor().strip() or "unknown"


def make_ort_session(path, threads, ort):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return ort.InferenceSession(
        str(path), sess_options=opts, providers=["CPUExecutionProvider"]
    )


def markdown_report(report):
    b = report["benchmark"]
    rows = []
    for key in ("pytorch_cpu", "fp32_onnx_cpu", "int8_onnx_cpu"):
        r = report["results"].get(key)
        if r is None:
            continue
        rows.append(
            f"| {r['label']} | {r['p50_ms']:.3f} | {r['p95_ms']:.3f} | "
            f"{r['mean_ms']:.3f} | {r['throughput_windows_per_s']:.1f} |"
        )
    return "\n".join([
        "# Phase 5c latency benchmark",
        "",
        f"- Device: CPU (`{report['environment']['cpu']}`)",
        f"- Input: `{tuple(b['input_shape'])}` float32 (one 60 s window)",
        f"- Threads: {b['threads']} intra-op, 1 inter-op; ORT sequential execution",
        f"- Warmup: {b['warmup_runs']} runs per backend (excluded)",
        f"- Measured: {b['measured_runs']} runs per backend",
        "- Throughput: `1000 / mean latency (ms)`",
        "",
        "| Backend | p50 (ms) | p95 (ms) | mean (ms) | windows/s |",
        "|---|---:|---:|---:|---:|",
        *rows,
        "",
    ])


def main():
    args = parse_args()
    cfg = load_config(args.config)
    bench_cfg = cfg.deploy.benchmark
    warmup_runs = positive_int(
        "--warmup-runs",
        args.warmup_runs if args.warmup_runs is not None
        else getattr(bench_cfg, "warmup_runs", 20),
    )
    measured_runs = positive_int(
        "--runs", args.runs if args.runs is not None else bench_cfg.n_runs
    )
    threads = positive_int(
        "--threads", args.threads if args.threads is not None else bench_cfg.threads
    )

    checkpoint = Path(args.checkpoint)
    fp32_path = Path(args.fp32_onnx)
    int8_path = Path(args.int8_onnx)
    required = [("FP32 ONNX", fp32_path), ("INT8 ONNX", int8_path)]
    if HAS_TORCH:
        required.append(("checkpoint", checkpoint))
    else:
        print(
            "PyTorch is not installed; skipping the PyTorch CPU benchmark and "
            "running ONNX Runtime backends only."
        )
    for label, path in required:
        if not path.is_file():
            raise SystemExit(f"{label} not found: {path}")

    if HAS_TORCH:
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)

    import onnxruntime as ort

    rng = np.random.default_rng(args.seed)
    input_shape = (1, cfg.data.n_channels, cfg.data.window_samples)
    x_np = rng.standard_normal(input_shape).astype(np.float32)

    fp32_session = make_ort_session(fp32_path, threads, ort)
    int8_session = make_ort_session(int8_path, threads, ort)

    def fp32_call():
        return fp32_session.run([OUTPUT_NAME], {INPUT_NAME: x_np})[0]

    def int8_call():
        return int8_session.run([OUTPUT_NAME], {INPUT_NAME: x_np})[0]

    calls = {
        "fp32_onnx_cpu": ("FP32 ONNX Runtime", fp32_call),
        "int8_onnx_cpu": ("INT8 ONNX Runtime", int8_call),
    }

    ckpt = None
    if HAS_TORCH:
        x_torch = torch.from_numpy(x_np)
        model, ckpt = load_student(cfg, checkpoint)

        @torch.inference_mode()
        def pytorch_call():
            return model(x_torch)

        calls["pytorch_cpu"] = ("PyTorch CPU", pytorch_call)

    expected_shape = (1, cfg.model.out_channels, cfg.data.window_samples)
    for _key, (label, call) in calls.items():
        output = call()
        if HAS_TORCH and isinstance(output, torch.Tensor):
            output = output.numpy()
        if output.shape != expected_shape or not np.isfinite(output).all():
            raise SystemExit(
                f"{label} produced invalid output: shape={output.shape}, "
                f"finite={np.isfinite(output).all()}"
            )

    print(
        f"CPU: {cpu_name()}\n"
        f"input={input_shape} threads={threads} warmup={warmup_runs} "
        f"measured={measured_runs}"
    )
    results = {}
    for key, (label, call) in calls.items():
        stats = benchmark(call, warmup_runs, measured_runs)
        results[key] = {"label": label, **stats}
        print(
            f"{label:20s} p50={stats['p50_ms']:.3f} ms  "
            f"p95={stats['p95_ms']:.3f} ms  mean={stats['mean_ms']:.3f} ms  "
            f"throughput={stats['throughput_windows_per_s']:.1f} windows/s"
        )

    report = {
        "checkpoint": str(checkpoint) if HAS_TORCH else None,
        "checkpoint_epoch": ckpt.get("epoch") if ckpt is not None else None,
        "fp32_onnx": str(fp32_path),
        "int8_onnx": str(int8_path),
        "benchmark": {
            "device": "cpu",
            "input_shape": list(input_shape),
            "input_dtype": "float32",
            "input_seed": args.seed,
            "threads": threads,
            "inter_op_threads": 1,
            "ort_execution_mode": "sequential",
            "warmup_runs": warmup_runs,
            "measured_runs": measured_runs,
            "timer": "time.perf_counter_ns",
            "throughput_definition": "1000 / mean_ms (serialized windows/s)",
        },
        "environment": {
            "cpu": cpu_name(),
            "logical_cpu_count": os.cpu_count(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__ if HAS_TORCH else "not installed",
            "onnxruntime": ort.__version__,
            "numpy": np.__version__,
        },
        "results": results,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "latency_report.json"
    md_path = out_dir / "latency_report.md"
    json_path.write_text(json.dumps(report, indent=2))
    md_path.write_text(markdown_report(report))
    print(f"wrote: {json_path}\n       {md_path}")


if __name__ == "__main__":
    main()
