#!/usr/bin/env python
"""Phase 5d: continuous-signal inference with overlapping ONNX windows.

The INT8 Stage 2 ONNX model is the default backend. Continuous 3-channel,
100 Hz input is split into 60 s windows at a 30 s hop, the three probability
streams are uniformly averaged in overlap regions, and relative-time event/P/S
records are emitted.

External ``.npy`` input may be shaped (3, N) or (N, 3). It is assumed raw and
is preprocessed independently per model window unless ``--input-preprocessed``
is supplied. Demo mode concatenates a small balanced selection of already-
preprocessed test traces from the cached STEAD dataset.

Usage (demo):
    python scripts/stream_infer.py --demo-traces 4 --plot --save-probabilities

Usage (array):
    python scripts/stream_infer.py --input continuous.npy \
        --out-dir outputs/streaming_demo
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from seismic_edge_picker.config import load_config  # noqa: E402
from seismic_edge_picker.preprocessing import (  # noqa: E402
    CausalPreprocessor,
    preprocess_waveform,
)
from seismic_edge_picker.streaming import (  # noqa: E402
    causal_stream_probabilities,
    associate_picks,
    extract_events,
    extract_phase_picks,
    stream_probabilities,
)

def parse_args():
    p = argparse.ArgumentParser(
        description="Run overlapping-window INT8 ONNX inference on a continuous signal."
    )
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="continuous .npy array shaped (3,N) or (N,3)")
    source.add_argument(
        "--demo-traces", type=int, metavar="N",
        help="concatenate N representative cached test traces",
    )
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--model", default="outputs/onnx/stage2_distill_int8.onnx")
    p.add_argument("--causal", action="store_true",
                   help="run a torch causal checkpoint with causal preprocessing")
    p.add_argument("--checkpoint", default=None,
                   help="torch checkpoint for --causal (default: stage3 causal, then smoke)")
    p.add_argument("--chunk-samples", type=int, default=500,
                   help="raw samples per causal streaming step (default: 500)")
    p.add_argument("--out-dir", default="outputs/streaming_demo")
    p.add_argument(
        "--input-preprocessed", action="store_true",
        help="skip per-window demean/bandpass/normalization for --input",
    )
    p.add_argument("--hop-s", type=float, default=None,
                   help="default: deploy.streaming.hop_s (30)")
    p.add_argument("--detection-threshold", type=float, default=None,
                   help="default: deploy.streaming.detection_threshold (0.80)")
    p.add_argument("--min-duration-ms", type=float, default=None,
                   help="default: deploy.streaming.min_duration_ms (10)")
    p.add_argument("--event-merge-gap-s", type=float, default=None,
                   help="coalesce qualifying detection runs separated by this gap")
    p.add_argument("--p-threshold", type=float, default=None,
                   help="default: deploy.streaming.p_threshold (0.30)")
    p.add_argument("--s-threshold", type=float, default=None,
                   help="default: deploy.streaming.s_threshold (0.30)")
    p.add_argument("--peak-min-distance-s", type=float, default=None)
    p.add_argument("--association-margin-s", type=float, default=None)
    p.add_argument("--threads", type=int, default=None,
                   help="ORT intra-op threads (default: 1)")
    p.add_argument("--batch-size", type=int, default=1,
                   help="windows per ORT call (default: 1)")
    p.add_argument("--plot", action="store_true",
                   help="save waveform + merged probability plot")
    p.add_argument("--save-probabilities", action="store_true",
                   help="save merged streams and coverage as compressed NPZ")
    p.add_argument("--emit-unassociated-picks", action="store_true",
                   help="retain P/S candidates outside detected event regions")
    return p.parse_args()


def load_npy(path):
    array = np.load(path, allow_pickle=False)
    if array.ndim != 2:
        raise SystemExit(f"input must be 2-D, got shape {array.shape}")
    if array.shape[0] == 3:
        signal = array
    elif array.shape[1] == 3:
        signal = array.T
    else:
        raise SystemExit(f"input must have exactly 3 channels, got shape {array.shape}")
    signal = np.ascontiguousarray(signal, dtype=np.float32)
    if signal.shape[1] == 0 or not np.isfinite(signal).all():
        raise SystemExit("input must contain at least one sample and only finite values")
    return signal


def select_demo_indices(dataset, n_traces):
    """Choose an interleaved earthquake/noise subset without reading waveforms."""
    from seismic_edge_picker import splits as S

    if n_traces < 1:
        raise SystemExit("--demo-traces must be >= 1")
    n_traces = min(n_traces, len(dataset))
    categories = dataset.metadata[S.COL_CATEGORY].astype(str)
    earthquakes, noise = [], []
    for local_index, row in enumerate(dataset.rows):
        category = categories.iloc[int(row)]
        if category in S.EARTHQUAKE_VALUES:
            earthquakes.append(local_index)
        elif category in S.NOISE_VALUES:
            noise.append(local_index)
        if len(earthquakes) >= n_traces and len(noise) >= n_traces:
            break

    selected = []
    for i in range(max(len(earthquakes), len(noise))):
        if i < len(earthquakes) and len(selected) < n_traces:
            selected.append(earthquakes[i])
        if i < len(noise) and len(selected) < n_traces:
            selected.append(noise[i])
        if len(selected) == n_traces:
            break
    if len(selected) < n_traces:
        used = set(selected)
        selected.extend(i for i in range(len(dataset)) if i not in used)
    return selected[:n_traces]


def load_demo(cfg, n_traces):
    from seismic_edge_picker import splits as S
    from seismic_edge_picker.dataset import build_datasets

    datasets, _ = build_datasets(cfg)
    dataset = datasets["test"]
    selected = select_demo_indices(dataset, n_traces)
    waveforms = []
    sources = []
    segment_samples = cfg.data.window_samples
    fs = cfg.data.sampling_rate
    for segment, local_index in enumerate(selected):
        waveform, _ = dataset[local_index]
        row = int(dataset.rows[local_index])
        metadata = dataset.metadata.iloc[row]
        start_sample = segment * segment_samples
        source = {
            "segment": segment,
            "dataset_index": int(local_index),
            "metadata_row": row,
            "category": str(metadata.get(S.COL_CATEGORY, "unknown")),
            "start_time_s": float(start_sample / fs),
            "end_time_s": float((start_sample + segment_samples) / fs),
        }
        for phase, column in (("p", S.COL_P), ("s", S.COL_S)):
            arrival = S.parse_scalar(metadata.get(column))
            source[f"ground_truth_{phase}_time_s"] = (
                None if not np.isfinite(arrival)
                else float((start_sample + arrival) / fs)
            )
        sources.append(source)
        waveforms.append(waveform.numpy())
    return np.concatenate(waveforms, axis=1).astype(np.float32), sources


def make_session(model_path, threads):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
    session = ort.InferenceSession(
        str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session, input_name, output_name, ort.__version__


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_plot(path, signal, probabilities, events, picks, sampling_rate, thresholds):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_samples = signal.shape[1]
    step = max(1, int(np.ceil(n_samples / 100_000)))
    sample_index = np.arange(0, n_samples, step)
    times = sample_index / sampling_rate

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    colors = ("C0", "C1", "C2")
    labels = ("Z", "N", "E")
    for channel, (color, label) in enumerate(zip(colors, labels)):
        channel_data = signal[channel, sample_index]
        scale = max(float(np.max(np.abs(channel_data))), 1e-8)
        axes[0].plot(times, channel_data / scale + channel * 2.5,
                     color=color, lw=0.5, label=label)
    axes[0].set_ylabel("normalized waveform + offset")
    axes[0].legend(loc="upper right", ncol=3)
    axes[0].grid(alpha=0.2)

    for channel, (color, label) in enumerate(zip(colors, ("Detection", "P", "S"))):
        axes[1].plot(times, probabilities[channel, sample_index],
                     color=color, lw=0.8, label=label)
    axes[1].axhline(thresholds["detection"], color="C0", ls="--", alpha=0.5)
    axes[1].axhline(thresholds["p"], color="C1", ls=":", alpha=0.4)
    axes[1].axhline(thresholds["s"], color="C2", ls=":", alpha=0.4)
    for event in events:
        axes[1].axvspan(event["start_time_s"], event["end_time_s"],
                        color="C0", alpha=0.08)
    for pick in picks:
        axes[1].axvline(pick["time_s"], color="C1" if pick["phase"] == "P" else "C2",
                        lw=0.7, alpha=0.5)
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_xlabel("relative time (s)")
    axes[1].set_ylabel("probability")
    axes[1].legend(loc="upper right", ncol=3)
    axes[1].grid(alpha=0.2)
    fig.suptitle("Streaming INT8 ONNX inference: merged overlap predictions")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    fs = float(cfg.data.sampling_rate)
    if fs != 100.0:
        raise SystemExit(f"streaming artifact expects 100 Hz, config has {fs:g} Hz")

    stream_cfg = cfg.deploy.streaming
    window_s = float(stream_cfg.window_s)
    hop_s = float(args.hop_s if args.hop_s is not None else stream_cfg.hop_s)
    window_samples = int(round(window_s * fs))
    hop_samples = int(round(hop_s * fs))
    if window_samples != cfg.data.window_samples:
        raise SystemExit(
            f"ONNX model requires {cfg.data.window_samples} samples, "
            f"streaming window resolves to {window_samples}"
        )
    if hop_samples < 1 or hop_samples > window_samples:
        raise SystemExit("hop must be positive and no longer than the 60 s window")

    detection_threshold = float(
        args.detection_threshold if args.detection_threshold is not None
        else stream_cfg.detection_threshold
    )
    min_duration_ms = float(
        args.min_duration_ms if args.min_duration_ms is not None
        else stream_cfg.min_duration_ms
    )
    event_merge_gap_s = float(
        args.event_merge_gap_s if args.event_merge_gap_s is not None
        else stream_cfg.event_merge_gap_s
    )
    p_threshold = float(
        args.p_threshold if args.p_threshold is not None else stream_cfg.p_threshold
    )
    s_threshold = float(
        args.s_threshold if args.s_threshold is not None else stream_cfg.s_threshold
    )
    peak_distance_s = float(
        args.peak_min_distance_s if args.peak_min_distance_s is not None
        else stream_cfg.peak_min_distance_s
    )
    association_margin_s = float(
        args.association_margin_s if args.association_margin_s is not None
        else stream_cfg.association_margin_s
    )
    threads = int(args.threads if args.threads is not None else stream_cfg.threads)
    if threads < 1 or args.batch_size < 1:
        raise SystemExit("--threads and --batch-size must be >= 1")

    if args.input:
        signal = load_npy(args.input)
        sources = [{"mode": "npy", "path": str(Path(args.input))}]
        preprocessed = bool(args.input_preprocessed)
        input_description = str(Path(args.input))
    else:
        if args.causal:
            raise SystemExit("--causal currently requires raw --input .npy")
        signal, sources = load_demo(cfg, args.demo_traces)
        preprocessed = True
        input_description = f"{len(sources)} concatenated test traces"

    if args.causal:
        if preprocessed:
            raise SystemExit("--causal expects raw input; do not pass --input-preprocessed")
        if args.chunk_samples < 1:
            raise SystemExit("--chunk-samples must be >= 1")
        import torch
        from seismic_edge_picker.model import build_model

        cfg.model.causal = True
        cfg.model.lookahead = 0
        candidates = [
            Path(args.checkpoint) if args.checkpoint else None,
            Path("checkpoints/stage3_causal/best.pt"),
            Path("checkpoints/stage3_causal_smoke/best.pt"),
        ]
        ckpt_path = next((path for path in candidates if path and path.is_file()), None)
        if ckpt_path is None:
            raise SystemExit("causal checkpoint not found; pass --checkpoint")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model = build_model(cfg).eval()
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model_path = ckpt_path
        backend = "Torch causal SeismicUNet CPU"
        ort_version = None
        merge_description = "causal chunk emission; unknown future filled from latest real sample"

        def predict(batch):
            with torch.inference_mode():
                return model(torch.from_numpy(batch.astype(np.float32))).numpy()

        start = time.perf_counter()
        streaming = causal_stream_probabilities(
            signal,
            predict,
            CausalPreprocessor(cfg, warmup_samples=min(int(fs), args.chunk_samples)),
            chunk_samples=args.chunk_samples,
        )
        elapsed_s = time.perf_counter() - start
    else:
        model_path = Path(args.model)
        if not model_path.is_file():
            raise SystemExit(f"ONNX model not found: {model_path}")
        session, input_name, output_name, ort_version = make_session(model_path, threads)
        backend = "ONNX Runtime CPUExecutionProvider"
        merge_description = "uniform mean over overlapping window probabilities"

        def predict(batch):
            model_input = batch
            if not preprocessed:
                model_input = np.stack(
                    [preprocess_waveform(window, cfg) for window in batch]
                ).astype(np.float32)
            return session.run([output_name], {input_name: model_input})[0]

        start = time.perf_counter()
        streaming = stream_probabilities(
            signal,
            predict,
            window_samples=window_samples,
            hop_samples=hop_samples,
            batch_size=args.batch_size,
        )
        elapsed_s = time.perf_counter() - start

    events = extract_events(
        streaming.probabilities[0], fs, detection_threshold, min_duration_ms,
        event_merge_gap_s,
    )
    candidate_picks = extract_phase_picks(
        streaming.probabilities, fs, p_threshold, s_threshold, peak_distance_s
    )
    candidate_picks = associate_picks(
        candidate_picks, events, fs, association_margin_s
    )
    picks = (
        candidate_picks if args.emit_unassociated_picks
        else [pick for pick in candidate_picks if pick["event_id"] is not None]
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / "events.csv"
    picks_path = out_dir / "picks.csv"
    summary_json_path = out_dir / "summary.json"
    summary_text_path = out_dir / "summary.txt"
    write_csv(events_path, events, [
        "event_id", "start_sample", "end_sample_exclusive", "start_time_s",
        "end_time_s", "duration_s", "peak_sample", "peak_time_s",
        "peak_probability",
    ])
    write_csv(picks_path, picks, [
        "phase", "sample", "time_s", "probability", "event_id",
    ])

    n_p = sum(pick["phase"] == "P" for pick in picks)
    n_s = sum(pick["phase"] == "S" for pick in picks)
    summary = {
        "model": str(model_path),
        "backend": backend,
        "onnxruntime_version": ort_version,
        "input": input_description,
        "input_preprocessed": preprocessed,
        "sampling_rate_hz": fs,
        "signal_samples": int(signal.shape[1]),
        "signal_duration_s": float(signal.shape[1] / fs),
        "window_s": window_s,
        "hop_s": hop_s,
        "window_samples": window_samples,
        "hop_samples": hop_samples,
        "window_starts": list(streaming.window_starts),
        "n_windows": len(streaming.window_starts),
        "merge": merge_description,
        "causal_chunk_samples": args.chunk_samples if args.causal else None,
        "threads": threads,
        "batch_size": args.batch_size,
        "inference_wall_seconds": elapsed_s,
        "operating_point": {
            "detection_threshold": detection_threshold,
            "min_duration_ms": min_duration_ms,
            "event_merge_gap_s": event_merge_gap_s,
            "p_threshold": p_threshold,
            "s_threshold": s_threshold,
            "peak_min_distance_s": peak_distance_s,
            "association_margin_s": association_margin_s,
        },
        "counts": {
            "events": len(events),
            "p_picks": n_p,
            "s_picks": n_s,
            "associated_picks": sum(pick["event_id"] is not None for pick in picks),
            "unassociated_candidates_filtered": sum(
                pick["event_id"] is None for pick in candidate_picks
            ) if not args.emit_unassociated_picks else 0,
        },
        "events": events,
        "picks": picks,
        "sources": sources,
        "operating_point_note": (
            "0.80 + 10 ms is the Stage 2 FP32 max-F1 point and was evaluated "
            "for INT8 at threshold 0.80. A separate low-false-alarm INT8 "
            "streaming threshold has not been calibrated."
        ),
    }
    summary_json_path.write_text(json.dumps(summary, indent=2))

    lines = [
        "Phase 5d streaming inference",
        f"model: {model_path} ({backend}, {threads} thread(s))",
        f"input: {input_description}",
        f"signal: {signal.shape[1]} samples / {signal.shape[1] / fs:.1f} s at {fs:g} Hz",
        f"windows/chunks: {len(streaming.window_starts)}; {merge_description}",
        f"operating point: detection >= {detection_threshold:g} for >= "
        f"{min_duration_ms:g} ms; merge gap <= {event_merge_gap_s:g} s; "
        f"P/S >= {p_threshold:g}/{s_threshold:g}",
        f"output: {len(events)} events, {n_p} P picks, {n_s} S picks "
        f"({summary['counts']['associated_picks']} associated)",
        "timestamps are seconds relative to the start of the input array",
        "low-false-alarm note: INT8 streaming has not been separately retuned; "
        "do not assume the FP32 0.90 + 500 ms point transfers unchanged.",
    ]
    summary_text_path.write_text("\n".join(lines) + "\n")

    written = [events_path, picks_path, summary_json_path, summary_text_path]
    if args.save_probabilities:
        probability_path = out_dir / "merged_probabilities.npz"
        np.savez_compressed(
            probability_path,
            probabilities=streaming.probabilities,
            coverage=streaming.coverage,
            window_starts=np.asarray(streaming.window_starts, dtype=np.int64),
            sampling_rate_hz=np.asarray(fs),
        )
        written.append(probability_path)
    if args.plot:
        plot_path = out_dir / "streaming_predictions.png"
        save_plot(
            plot_path, signal, streaming.probabilities, events, picks, fs,
            {"detection": detection_threshold, "p": p_threshold, "s": s_threshold},
        )
        written.append(plot_path)

    print("\n".join(lines))
    print("\nwrote:")
    for output_path in written:
        print(f"  {output_path}")


if __name__ == "__main__":
    main()
