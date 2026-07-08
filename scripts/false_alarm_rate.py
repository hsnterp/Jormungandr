#!/usr/bin/env python
"""Deployment-relevant false-alarm rate: false triggers per hour / per day.

WHY THIS EXISTS
---------------
Per-window false-positive count on a curated noise set (e.g. "11 / 2,824") is the
metric the headline tables report, but it is NOT the number an autonomous,
on-device trigger lives on. That device runs a *continuous* stream and fires with
no coincidence check and no human veto, so what matters is **false triggers over
time**: how often does the detector fire on noise per hour / per day of quiet
ground?

This script converts the measured per-window noise FP rate at a chosen operating
point into an approximate false-triggers-per-hour and per-day figure, given the
streaming hop (``deploy.streaming.hop_s`` — 30 s by default → 120 windows/hour).

THE ESTIMATE AND ITS ASSUMPTIONS (documented, on purpose)
---------------------------------------------------------
false_triggers_per_hour ≈ (FP / n_noise) * (3600 / hop_s)

  1. Each streaming window is treated as one independent Bernoulli trial whose
     false-trigger probability equals the per-window FP rate measured on the
     curated STEAD noise set (real, station-diverse seismic noise).
  2. Windows advance by ``hop_s`` (default 30 s), so windows/hour = 3600/hop_s.
     Consecutive 60 s windows overlap 50 %; the streaming path coalesces
     detections within ``event_merge_gap_s``, which can only MERGE adjacent
     false triggers into one — so ignoring coalescing makes this an UPPER bound.
  3. It assumes field-noise statistics resemble STEAD's curated noise. Real
     station noise (cultural, storm, teleseismic coda) can differ; treat this as
     an order-of-magnitude operating figure, not a field-validated guarantee.

DIRECT MEASUREMENT (optional)
-----------------------------
If a continuous noise array is available, ``--noise-npy path.npy`` (shape
``(3, N)`` or ``(N, 3)``, 100 Hz) runs it through the SAME streaming path
(``streaming.stream_probabilities`` + ``extract_events``) with the deployed INT8
ONNX model and reports the *measured* false triggers per hour directly, instead
of estimating. ``--smoke`` synthesizes a short random-noise array and exercises
that path end to end (expecting ~0 triggers) so the pipeline can be verified
without the STEAD cache.

Outputs: outputs/false_alarm/false_alarm_rate.json + a console summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from seismic_edge_picker.config import load_config  # noqa: E402
from seismic_edge_picker.streaming import (  # noqa: E402
    extract_events,
    stream_probabilities,
)

# Documented operating points and their per-window noise-FP counts on the STEAD
# test split (2,824 noise traces). Provenance is recorded per row so the numbers
# trace back to a committed artifact wherever one exists.
N_NOISE_TEST = 2824
OPERATING_POINTS = [
    {
        "name": "low-false-alarm (thr 0.90 + 500 ms)",
        "threshold": 0.90,
        "min_duration_ms": 500,
        "fp_noise": 11,
        "model": "Stage 2 distilled student (reference)",
        "source": "scripts/threshold_sweep.py low-false-alarm point; deployed "
                  "operating point (INT8 not separately revalidated here)",
        "deployed": True,
    },
    {
        "name": "INT8 max-F1 (thr 0.80)",
        "threshold": 0.80,
        "min_duration_ms": 0,
        "fp_noise": 29,
        "model": "shipped INT8 ONNX",
        "source": "outputs/onnx/quantization_report.json (measured)",
        "deployed": False,
    },
    {
        "name": "FP32 max-F1 (thr 0.80)",
        "threshold": 0.80,
        "min_duration_ms": 0,
        "fp_noise": 45,
        "model": "FP32 ONNX",
        "source": "outputs/onnx/quantization_report.json (measured)",
        "deployed": False,
    },
]


def estimate(fp_noise: int, n_noise: int, hop_s: float) -> dict:
    per_window = fp_noise / n_noise
    windows_per_hour = 3600.0 / hop_s
    per_hour = per_window * windows_per_hour
    return {
        "fp_noise": fp_noise,
        "n_noise": n_noise,
        "per_window_fp_rate": per_window,
        "windows_per_hour": windows_per_hour,
        "false_triggers_per_hour": per_hour,
        "false_triggers_per_day": per_hour * 24.0,
    }


def make_session(model_path: Path, threads: int):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
    )
    name_in = session.get_inputs()[0].name
    name_out = session.get_outputs()[0].name
    return session, name_in, name_out


def measure_stream(signal, cfg, model_path, threshold, min_duration_ms,
                   merge_gap_s, threads, preprocess):
    """Run the deployed INT8 streaming path on a continuous array; count triggers."""
    from seismic_edge_picker.preprocessing import preprocess_waveform

    fs = cfg.data.sampling_rate
    window = cfg.data.window_samples
    hop = int(round(cfg.deploy.streaming.hop_s * fs))
    session, name_in, name_out = make_session(model_path, threads)

    def predictor(batch):
        out = np.empty_like(batch)
        for i in range(batch.shape[0]):
            w = batch[i]
            if preprocess:
                w = preprocess_waveform(w, cfg)
            out[i] = session.run([name_out], {name_in: w[None].astype(np.float32)})[0][0]
        return out

    result = stream_probabilities(signal, predictor, window, hop)
    events = extract_events(
        result.probabilities[0], fs, threshold=threshold,
        min_duration_ms=min_duration_ms, merge_gap_s=merge_gap_s,
    )
    duration_s = signal.shape[1] / fs
    per_hour = len(events) / (duration_s / 3600.0)
    return {
        "n_windows": len(result.window_starts),
        "duration_s": duration_s,
        "false_triggers": len(events),
        "false_triggers_per_hour": per_hour,
        "false_triggers_per_day": per_hour * 24.0,
    }


def load_signal(path: Path) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f"expected 2-D array, got {arr.shape}")
    if arr.shape[0] != 3 and arr.shape[1] == 3:
        arr = arr.T
    if arr.shape[0] != 3:
        raise ValueError(f"expected a 3-channel array, got {arr.shape}")
    return arr


def parse_args():
    p = argparse.ArgumentParser(description="False triggers per hour/day for the deployed model.")
    p.add_argument("--config", default=str(REPO / "configs" / "default.yaml"))
    p.add_argument("--hop-s", type=float, default=None,
                   help="streaming hop seconds (default: deploy.streaming.hop_s)")
    p.add_argument("--int8-onnx", default=str(REPO / "outputs" / "onnx" / "stage2_distill_int8.onnx"))
    p.add_argument("--out", default=str(REPO / "outputs" / "false_alarm"))
    p.add_argument("--noise-npy", default=None,
                   help="continuous noise array (3,N)/(N,3) at 100 Hz for a DIRECT measurement")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--smoke", action="store_true",
                   help="synthesize a short random-noise array and exercise the streaming path")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    st = cfg.deploy.streaming
    hop_s = float(args.hop_s if args.hop_s is not None else st.hop_s)
    windows_per_hour = 3600.0 / hop_s

    estimates = []
    for op in OPERATING_POINTS:
        est = estimate(op["fp_noise"], N_NOISE_TEST, hop_s)
        estimates.append({**{k: op[k] for k in
                             ("name", "threshold", "min_duration_ms", "model",
                              "source", "deployed")}, **est})

    measurement = None
    if args.smoke or args.noise_npy:
        if args.noise_npy:
            signal = load_signal(Path(args.noise_npy))
            preprocess = True
            src = args.noise_npy
        else:
            rng = np.random.default_rng(cfg.seed)
            # 1 hour of 3-channel random noise at 100 Hz
            signal = rng.standard_normal((3, 3600 * cfg.data.sampling_rate)).astype(np.float32)
            preprocess = True
            src = "synthetic gaussian noise (smoke)"
        measurement = measure_stream(
            signal, cfg, Path(args.int8_onnx),
            threshold=st.detection_threshold, min_duration_ms=st.min_duration_ms,
            merge_gap_s=st.event_merge_gap_s, threads=args.threads,
            preprocess=preprocess,
        )
        measurement["source"] = src
        measurement["streaming_threshold"] = st.detection_threshold
        measurement["streaming_min_duration_ms"] = st.min_duration_ms

    deployed = next(e for e in estimates if e["deployed"])
    report = {
        "description": "False triggers per hour/day in continuous operation. This "
                       "TIME-based rate — not per-window FP — is the metric an "
                       "unsupervised on-device actuator lives on.",
        "streaming_hop_s": hop_s,
        "windows_per_hour": windows_per_hour,
        "assumptions": [
            "each streaming window ~ one independent 60 s curated-noise trial",
            "windows/hour = 3600 / hop_s (60 s windows overlap 50% at 30 s hop; "
            "coalescing within event_merge_gap_s only lowers the count -> upper bound)",
            "field noise resembles STEAD's curated noise set",
        ],
        "deployed_operating_point": deployed["name"],
        "headline_false_triggers_per_hour": deployed["false_triggers_per_hour"],
        "headline_false_triggers_per_day": deployed["false_triggers_per_day"],
        "estimates": estimates,
        "direct_measurement": measurement,
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "false_alarm_rate.json").write_text(json.dumps(report, indent=2))

    print(f"streaming hop {hop_s:g} s -> {windows_per_hour:g} windows/hour\n")
    print(f"{'operating point':38s} {'per-window FP':>13s} {'/hour':>8s} {'/day':>8s}")
    for e in estimates:
        tag = "  <-- deployed" if e["deployed"] else ""
        print(f"{e['name']:38s} {e['fp_noise']:>4d}/{e['n_noise']:<4d} "
              f"({e['per_window_fp_rate']*100:4.2f}%) {e['false_triggers_per_hour']:>7.3f} "
              f"{e['false_triggers_per_day']:>7.1f}{tag}")
    if measurement:
        print(f"\nDIRECT streaming measurement ({measurement['source']}):")
        print(f"  {measurement['n_windows']} windows over {measurement['duration_s']:.0f} s "
              f"-> {measurement['false_triggers']} triggers "
              f"= {measurement['false_triggers_per_hour']:.3f}/hour")
    print(f"\nwrote {out_dir / 'false_alarm_rate.json'}")


if __name__ == "__main__":
    main()
