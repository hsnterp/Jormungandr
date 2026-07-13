#!/usr/bin/env python
"""Causal early-firing latency curve and STA/LTA baseline.

Default mode is a synthetic smoke run: no STEAD/PNW access, no training, and all
outputs are marked plumbing-only. Real STEAD evaluation is dataset-gated behind
``--data`` and reuses the repo split helpers. PNW remains a held-out downstream
run via ``scripts/pnw_zeroshot.py``; this script records that gate rather than
silently fabricating OOD numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from seismic_edge_picker.config import load_config  # noqa: E402
from seismic_edge_picker.model import build_model  # noqa: E402
from seismic_edge_picker.preprocessing import (  # noqa: E402
    CausalPreprocessor,
    preprocess_waveform,
)
from seismic_edge_picker.streaming import causal_stream_probabilities  # noqa: E402
from evaluate import prf  # noqa: E402
from signal_utils import arrival_wave, sta_lta_ratio  # noqa: E402

BUDGETS_S = [0.5, 1.0, 2.0, 5.0, 7.0, float("inf")]


@dataclass
class Trace:
    raw: np.ndarray
    is_eq: bool
    p_sample: int | None
    source: str


def parse_args():
    p = argparse.ArgumentParser(description="Build causal recall-vs-latency curves.")
    p.add_argument("--config", default=str(REPO / "configs/default.yaml"))
    p.add_argument("--out", default=str(REPO / "outputs/causal"))
    p.add_argument("--data", action="store_true",
                   help="DATASET-GATED: use cached STEAD instead of synthetic smoke")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--n", type=int, default=64,
                   help="max STEAD traces when --data is used")
    p.add_argument("--n-smoke-events", type=int, default=3)
    p.add_argument("--n-smoke-noise", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--causal-checkpoint", default=None)
    p.add_argument("--shipped-checkpoint", default=str(REPO / "checkpoints/stage2_distill/best.pt"))
    p.add_argument("--chunk-samples", type=int, default=1000)
    p.add_argument("--p-threshold", type=float, default=None)
    return p.parse_args()


def synthetic_traces(cfg, n_events, n_noise, seed):
    fs = float(cfg.data.sampling_rate)
    n = int(cfg.data.window_samples)
    rng = np.random.default_rng(seed)
    traces = []
    for _ in range(n_events):
        p_on = int(rng.integers(int(4 * fs), int(18 * fs)))
        s_on = int(min(n - int(2 * fs), p_on + rng.integers(int(3 * fs), int(9 * fs))))
        raw = 0.12 * rng.standard_normal((3, n)).astype(np.float32)
        for ch in range(3):
            raw[ch] += arrival_wave(n, fs, p_on, 9.0, 2.5, 1.0)
            raw[ch] += arrival_wave(n, fs, s_on, 3.0, 3.5, 3.0)
        traces.append(Trace(raw.astype(np.float32), True, p_on, "synthetic_event"))
    for _ in range(n_noise):
        raw = 0.2 * rng.standard_normal((3, n)).astype(np.float32)
        traces.append(Trace(raw.astype(np.float32), False, None, "synthetic_noise"))
    rng.shuffle(traces)
    return traces


def stead_traces(cfg, split, n):
    # DATASET-GATED: this touches cached STEAD only when --data is explicitly set.
    from seismic_edge_picker import splits as S
    from seismic_edge_picker.dataset import _get_waveform, build_datasets

    datasets, _ = build_datasets(cfg)
    ds = datasets[split]
    traces = []
    cat = ds.metadata[S.COL_CATEGORY].astype(str)
    is_eq_all = cat.isin(S.EARTHQUAKE_VALUES).to_numpy()
    for i in range(min(n, len(ds))):
        row = int(ds.rows[i])
        meta = ds.metadata.iloc[row]
        p = S.parse_scalar(meta.get(S.COL_P))
        raw = _get_waveform(ds.ds, row, cfg.data.n_channels, cfg.data.window_samples)
        traces.append(Trace(
            raw=raw.astype(np.float32),
            is_eq=bool(is_eq_all[row]),
            p_sample=None if not np.isfinite(p) else int(round(p)),
            source="STEAD",
        ))
    return traces


def load_checkpoint_model(cfg, path, causal):
    cfg.model.causal = bool(causal)
    cfg.model.lookahead = 0
    model = build_model(cfg).eval()
    ckpt_path = Path(path) if path else None
    if ckpt_path and ckpt_path.is_file():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        return model, str(ckpt_path), False
    return model, "random-untrained (checkpoint missing)", True


def first_crossing(stream, threshold, start=0):
    hits = np.flatnonzero(stream[int(start):] >= threshold)
    if hits.size == 0:
        return None
    return int(start + hits[0])


def chunk_alarm_sample(crossing, chunk_samples, n_samples):
    if crossing is None:
        return None
    return min(((crossing // chunk_samples) + 1) * chunk_samples - 1, n_samples - 1)


def latency_stats(latencies, n_events):
    detected = [x for x in latencies if x is not None and np.isfinite(x)]
    out = {}
    for budget in BUDGETS_S:
        key = "inf" if math.isinf(budget) else f"{budget:g}"
        out[key] = sum(x <= budget for x in detected) / n_events if n_events else 0.0
    arr = np.asarray(detected, dtype=np.float64)
    out["median_latency_s"] = None if arr.size == 0 else float(np.median(arr))
    out["p90_latency_s"] = None if arr.size == 0 else float(np.percentile(arr, 90))
    out["recall_inf"] = len(detected) / n_events if n_events else 0.0
    return out


def false_triggers_per_hour(trigger_counts, noise_seconds):
    if noise_seconds <= 0:
        return None
    return float(sum(trigger_counts) / (noise_seconds / 3600.0))


def rising_edge_count(stream, threshold):
    mask = np.asarray(stream) >= threshold
    if mask.size == 0:
        return 0
    prev = np.r_[False, mask[:-1]]
    return int(np.count_nonzero(mask & ~prev))


def eval_causal_model(traces, cfg, model, threshold, chunk_samples):
    latencies, residual_ms, noise_triggers = [], [], []
    fs = float(cfg.data.sampling_rate)

    def predict(batch):
        with torch.inference_mode():
            return model(torch.from_numpy(batch.astype(np.float32))).numpy()

    for tr in traces:
        stream = causal_stream_probabilities(
            tr.raw,
            predict,
            CausalPreprocessor(cfg, warmup_samples=min(int(fs), chunk_samples)),
            chunk_samples=chunk_samples,
        ).probabilities[1]
        if tr.is_eq and tr.p_sample is not None:
            cross = first_crossing(stream, threshold, tr.p_sample)
            alarm = chunk_alarm_sample(cross, chunk_samples, tr.raw.shape[1])
            if alarm is None:
                latencies.append(None)
            else:
                lat = (alarm - tr.p_sample) / fs
                latencies.append(lat)
                residual_ms.append((cross - tr.p_sample) / fs * 1000.0)
        elif not tr.is_eq:
            noise_triggers.append(rising_edge_count(stream, threshold))
    return summarize_system("causal_unet", traces, latencies, residual_ms, noise_triggers, fs)


def eval_sta_lta_params(traces, cfg, params):
    fs = float(cfg.data.sampling_rate)
    sta_s, lta_s, on = params
    latencies, residual_ms, noise_triggers = [], [], []
    for tr in traces:
        ratio = sta_lta_ratio(tr.raw, fs, sta_s, lta_s)
        if tr.is_eq and tr.p_sample is not None:
            cross = first_crossing(ratio, on, tr.p_sample)
            if cross is None:
                latencies.append(None)
            else:
                latencies.append((cross - tr.p_sample) / fs)
                residual_ms.append((cross - tr.p_sample) / fs * 1000.0)
        elif not tr.is_eq:
            noise_triggers.append(rising_edge_count(ratio, on))
    return summarize_system("sta_lta", traces, latencies, residual_ms, noise_triggers, fs)


def tune_sta_lta(traces, cfg):
    grid = []
    for sta_s in (0.2, 0.5, 1.0):
        for lta_s in (3.0, 5.0, 10.0):
            if lta_s <= sta_s:
                continue
            for on in (1.5, 2.0, 2.5, 3.0, 4.0):
                grid.append((sta_s, lta_s, on))
    best = None
    for params in grid:
        summary = eval_sta_lta_params(traces, cfg, params)
        score = (summary["f1"], summary["recall"], -summary["false_triggers_per_hour"])
        if best is None or score > best[0]:
            best = (score, params, summary)
    return best[1], best[2]


def eval_shipped_proxy(traces, cfg, model, threshold):
    fs = float(cfg.data.sampling_rate)
    n = int(cfg.data.window_samples)
    by_budget = {budget: [] for budget in BUDGETS_S}
    residual_ms = []
    noise_triggers = []
    for tr in traces:
        x_full = preprocess_waveform(tr.raw, cfg)
        if tr.is_eq and tr.p_sample is not None:
            pre = x_full[:, :max(1, tr.p_sample)]
            sigma = float(pre.std()) if pre.size else float(x_full.std())
            rng = np.random.default_rng(tr.p_sample)
            noise = rng.standard_normal(x_full.shape).astype(np.float32) * sigma
            for budget in BUDGETS_S:
                if math.isinf(budget):
                    x = x_full
                else:
                    cut = min(n - 1, tr.p_sample + int(round(budget * fs)))
                    x = x_full.copy()
                    x[:, cut + 1:] = noise[:, cut + 1:]
                with torch.inference_mode():
                    p_stream = model(torch.from_numpy(x[None].astype(np.float32))).numpy()[0, 1]
                cross = first_crossing(p_stream, threshold, tr.p_sample)
                by_budget[budget].append(cross is not None)
                if math.isinf(budget) and cross is not None:
                    residual_ms.append((cross - tr.p_sample) / fs * 1000.0)
        elif not tr.is_eq:
            with torch.inference_mode():
                p_stream = model(torch.from_numpy(x_full[None].astype(np.float32))).numpy()[0, 1]
            noise_triggers.append(rising_edge_count(p_stream, threshold))
    n_events = sum(1 for tr in traces if tr.is_eq and tr.p_sample is not None)
    latencies = []
    for tr in traces:
        if tr.is_eq and tr.p_sample is not None:
            latencies.append(None)
    summary = summarize_system("shipped_unet_proxy", traces, latencies, residual_ms, noise_triggers, fs)
    for budget, detections in by_budget.items():
        key = "inf" if math.isinf(budget) else f"{budget:g}"
        summary["curve"][key] = sum(detections) / n_events if n_events else 0.0
    summary["measurement"] = "masking proxy, not true streaming"
    return summary


def summarize_system(name, traces, latencies, residual_ms, noise_triggers, fs):
    n_events = sum(1 for tr in traces if tr.is_eq and tr.p_sample is not None)
    n_noise = sum(1 for tr in traces if not tr.is_eq)
    detected = sum(x is not None and np.isfinite(x) for x in latencies)
    fp = sum(1 for x in noise_triggers if x > 0)
    precision, recall, f1 = prf(detected, fp, n_events - detected)
    noise_seconds = n_noise * cfg_window_seconds(traces, fs)
    curve = latency_stats(latencies, n_events)
    arr = np.asarray(residual_ms, dtype=np.float64)
    return {
        "system": name,
        "measurement": "true streaming latency",
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mae_conditional_hit_ms": None if arr.size == 0 else float(np.mean(np.abs(arr))),
        "hit_rate": recall,
        "false_triggers_per_hour": false_triggers_per_hour(noise_triggers, noise_seconds) or 0.0,
        "median_latency_s": curve["median_latency_s"],
        "p90_latency_s": curve["p90_latency_s"],
        "curve": curve,
    }


def cfg_window_seconds(traces, fs):
    if not traces:
        return 0.0
    return traces[0].raw.shape[1] / fs


def write_curve_csv(path, dataset, summaries):
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["dataset", "system", "measurement", "latency_budget_s", "p_recall"])
        for summary in summaries:
            for budget in BUDGETS_S:
                key = "inf" if math.isinf(budget) else f"{budget:g}"
                writer.writerow([dataset, summary["system"], summary["measurement"], key, summary["curve"][key]])


def write_table_csv(path, dataset, summaries):
    fields = [
        "dataset", "system", "measurement", "precision", "recall", "f1",
        "mae_conditional_hit_ms", "hit_rate", "false_triggers_per_hour",
        "median_latency_s", "p90_latency_s", "phase_classification",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for summary in summaries:
            row = {key: summary.get(key) for key in fields if key not in {"dataset", "phase_classification"}}
            row["dataset"] = dataset
            row["phase_classification"] = "N/A" if summary["system"] == "sta_lta" else "P stream only in latency curve"
            writer.writerow(row)


def plot_curve(path, dataset, summaries):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = [b if not math.isinf(b) else 8.5 for b in BUDGETS_S]
    labels = ["0.5", "1", "2", "5", "7", "inf"]
    for summary in summaries:
        y = [summary["curve"]["inf" if math.isinf(b) else f"{b:g}"] for b in BUDGETS_S]
        ax.plot(x, y, marker="o", label=summary["system"].replace("_", " "))
    ax.set_xticks(x, labels)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("onset-to-alarm latency budget (s)")
    ax.set_ylabel("P recall within budget")
    ax.set_title(f"P recall vs onset latency ({dataset})")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    fs = float(cfg.data.sampling_rate)
    threshold = float(args.p_threshold if args.p_threshold is not None else cfg.eval.peak_height)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.data:
        traces = stead_traces(cfg, args.split, args.n)
        dataset = f"STEAD-{args.split}"
        data_note = "DATASET-GATED real STEAD run. PNW OOD remains gated; reuse scripts/pnw_zeroshot.py."
    else:
        print("no data — plumbing only")
        traces = synthetic_traces(cfg, args.n_smoke_events, args.n_smoke_noise, args.seed)
        dataset = "SMOKE"
        data_note = "Synthetic smoke only; no STEAD training/eval and no PNW OOD eval were run."

    causal_ckpt = args.causal_checkpoint
    if causal_ckpt is None:
        for candidate in (REPO / "checkpoints/stage3_causal/best.pt", REPO / "checkpoints/stage3_causal_smoke/best.pt"):
            if candidate.is_file():
                causal_ckpt = str(candidate)
                break
    causal_model, causal_model_note, causal_random = load_checkpoint_model(cfg, causal_ckpt, causal=True)
    causal_summary = eval_causal_model(traces, cfg, causal_model, threshold, args.chunk_samples)

    sta_params, sta_val_summary = tune_sta_lta(traces, cfg)
    sta_summary = eval_sta_lta_params(traces, cfg, sta_params)
    sta_summary["tuned_params"] = {"sta_s": sta_params[0], "lta_s": sta_params[1], "on": sta_params[2]}

    shipped_model, shipped_note, shipped_random = load_checkpoint_model(cfg, args.shipped_checkpoint, causal=False)
    shipped_summary = eval_shipped_proxy(traces, cfg, shipped_model, threshold)

    summaries = [shipped_summary, causal_summary, sta_summary]
    write_curve_csv(out / "latency_curve.csv", dataset, summaries)
    write_table_csv(out / "summary_table.csv", dataset, summaries)
    plot_curve(out / "recall_latency.png", dataset, summaries)

    run = {
        "dataset": dataset,
        "data_note": data_note,
        "p_threshold": threshold,
        "latency_budgets_s": ["inf" if math.isinf(b) else b for b in BUDGETS_S],
        "causal_checkpoint": causal_model_note,
        "causal_checkpoint_missing_random_model": causal_random,
        "shipped_checkpoint": shipped_note,
        "shipped_checkpoint_missing_random_model": shipped_random,
        "chunk_samples": args.chunk_samples,
        "sta_lta_tuned_params": sta_summary["tuned_params"],
        "sta_lta_tuning_summary": sta_val_summary,
        "summaries": summaries,
    }
    (out / "run.json").write_text(json.dumps(run, indent=2))

    lines = [
        "# Causal early-firing latency run",
        "",
        data_note,
        "",
        "Measurement notes:",
        "- causal_unet and sta_lta use true streaming onset-to-alarm latency.",
        "- shipped_unet_proxy uses a right-context masking proxy, not deployable streaming.",
        "- STA/LTA is detector-only; phase-classification metrics are N/A.",
        "- Full STEAD training and PNW OOD evaluation are dataset-gated and were not run in smoke mode.",
        "",
        f"Causal checkpoint: {causal_model_note}",
        f"Shipped checkpoint: {shipped_note}",
        f"STA/LTA tuned params: {sta_summary['tuned_params']}",
        "",
        "Artifacts: latency_curve.csv, summary_table.csv, recall_latency.png, run.json.",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
