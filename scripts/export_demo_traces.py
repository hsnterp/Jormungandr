#!/usr/bin/env python
"""Export the 6 waveform-overlay demo traces to outputs/figures/demo_traces.json.

This mirrors the trace-selection logic in ``outputs/figures/make_eval_figures.py``
EXACTLY (same config, same deterministic DataLoader pass, same selection rules),
so the JSON describes the identical six traces shown in the overlay PNGs. Instead
of plotting, it emits a compact, downsampled JSON payload for the interactive
demo page (docs/demo.html):

  per trace: trace_id, SNR (+ human label), the raw 3-channel waveform, the
  student model's detection/P/S probability streams (all downsampled to
  ~DOWNSAMPLE_POINTS samples per channel for visualization), ground-truth P/S
  arrival times in seconds, EQTransformer teacher P/S pick times in seconds, the
  student's own predicted P/S picks (+ peak confidence), the P-pick error, and a
  measured single-window inference latency.

Student  = checkpoints/stage2_distill/best.pt
Teacher  = SeisBench pretrained EQTransformer ('stead'), NATIVE preprocessing,
           matching the corrected/fair protocol (see scripts/fair_comparison.py).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from seismic_edge_picker.config import load_config  # noqa: E402
from seismic_edge_picker.dataset import build_datasets, _get_waveform  # noqa: E402
from seismic_edge_picker.model import build_model, count_parameters  # noqa: E402
from seismic_edge_picker import splits as S  # noqa: E402
from evaluate import pick_from_stream, detect_event  # noqa: E402

OUT = REPO / "outputs" / "figures" / "demo_traces.json"

DOWNSAMPLE_POINTS = 750  # per channel/stream — for visualization, not analysis

KIND_LABEL = {
    "high_snr_eq": "Clean high-SNR earthquake",
    "low_snr_eq": "Low-SNR / near-threshold earthquake",
    "noise": "Pure noise (true negative)",
    "worst_case": "Failure case",
}


def downsample_block(arr: np.ndarray, n_out: int, mode: str) -> np.ndarray:
    """Reduce a 1-D array to ~n_out points by block reduction.

    mode="peak": keep the max-abs sample in each block (preserves waveform
    amplitude/shape visually); mode="mean": block mean (smooth prob streams).
    Returns the reduced array; the shared time axis uses each block's start.
    """
    n = arr.shape[0]
    step = max(1, n // n_out)
    n_blocks = n // step
    trimmed = arr[: n_blocks * step].reshape(n_blocks, step)
    if mode == "peak":
        idx = np.argmax(np.abs(trimmed), axis=1)
        return trimmed[np.arange(n_blocks), idx]
    return trimmed.mean(axis=1)


def block_time(n: int, n_out: int, fs: float) -> np.ndarray:
    step = max(1, n // n_out)
    n_blocks = n // step
    return (np.arange(n_blocks) * step) / fs


def main():
    cfg = load_config(str(REPO / "configs" / "default.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fs = float(cfg.data.sampling_rate)
    ev = cfg.eval
    peak_distance = max(1, int(round(ev.peak_min_distance_s * fs)))
    tol_ms = ev.match_tolerance_s * 1000.0
    n_samples = cfg.data.window_samples

    # ---- student ----------------------------------------------------------
    ckpt_path = REPO / "checkpoints" / "stage2_distill" / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"student: {ckpt_path} epoch={ckpt.get('epoch')} "
          f"params={count_parameters(model):,} device={device}")

    datasets, _ = build_datasets(cfg)
    ds = datasets["test"]
    meta = ds.metadata
    rows = ds.rows
    cat = meta[S.COL_CATEGORY].astype(str)
    is_eq_all = cat.isin(S.EARTHQUAKE_VALUES).to_numpy()
    print(f"test split: {len(ds)} traces")

    # ---- pass 1: scalar records over the whole test split (== figure script)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=4)
    recs = []
    offset = 0
    with torch.inference_mode():
        for x, _ in loader:
            pred = model(x.to(device)).cpu().numpy()
            for b in range(pred.shape[0]):
                row_idx = int(rows[offset + b])
                m = meta.iloc[row_idx]
                det, p_str, s_str = pred[b]
                p_pick, _ = pick_from_stream(p_str, ev.peak_height, peak_distance)
                s_pick, _ = pick_from_stream(s_str, ev.peak_height, peak_distance)
                gt_p = S.parse_scalar(m.get(S.COL_P))
                gt_s = S.parse_scalar(m.get(S.COL_S))
                p_err = (abs(p_pick - gt_p) / fs * 1000.0
                         if (p_pick is not None and np.isfinite(gt_p)) else None)
                recs.append({
                    "i": offset + b, "row": row_idx,
                    "is_eq": bool(is_eq_all[row_idx]),
                    "snr": S.parse_snr_db(m.get(S.COL_SNR)),
                    "detected": detect_event(det, ev.detection_threshold, peak_distance),
                    "gt_p": gt_p, "gt_s": gt_s,
                    "p_pick": p_pick, "s_pick": s_pick, "p_err": p_err,
                })
            offset += pred.shape[0]
    print(f"pass 1 done: {len(recs)} records")

    # ---- selection (identical to make_eval_figures.py) --------------------
    eq = [r for r in recs if r["is_eq"] and np.isfinite(r["snr"])]
    eq_hi = sorted(eq, key=lambda r: -r["snr"])
    eq_lo = sorted(eq, key=lambda r: r["snr"])
    picked_eq = [r for r in eq if r["p_err"] is not None]

    def take(cands, n, used):
        out = []
        for r in cands:
            if r["i"] in used:
                continue
            out.append(r)
            used.add(r["i"])
            if len(out) == n:
                break
        return out

    used = set()
    hi = take([r for r in eq_hi if r["p_err"] is not None and r["p_err"] <= tol_ms], 2, used)
    lo = take([r for r in eq_lo if r["detected"]], 2, used)
    noise = take([r for r in recs if not r["is_eq"] and not r["detected"]], 1, used)
    worst = take(sorted(picked_eq, key=lambda r: -r["p_err"]), 1, used)
    worst_fp = None
    if not worst:
        worst_fp = take([r for r in recs if not r["is_eq"] and r["detected"]], 1, used)
        worst = worst_fp

    selection = (
        [("high_snr_eq", r) for r in hi]
        + [("low_snr_eq", r) for r in lo]
        + [("noise", r) for r in noise]
        + [("worst_case", r) for r in worst]
    )
    print("selected:", [(k, r["i"], round(r["snr"], 1) if np.isfinite(r["snr"]) else "nan",
                          None if r["p_err"] is None else round(r["p_err"], 0))
                        for k, r in selection])

    # ---- teacher (EQTransformer) on the selected traces only --------------
    import seisbench.models as sbm
    teacher = sbm.EQTransformer.from_pretrained("stead").to(device).eval()

    def student_streams_timed(i):
        x, _ = ds[i]
        xb = x.unsqueeze(0).to(device)
        # measured single-window inference latency (median of a few runs)
        times = []
        with torch.inference_mode():
            for _ in range(7):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                pred = model(xb)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000.0)
        return pred.cpu().numpy()[0], float(np.median(times[1:]))

    def teacher_picks(raw):
        rb = torch.tensor(raw[None], dtype=torch.float32, device=device)
        with torch.inference_mode():
            out = teacher(teacher.annotate_batch_pre(rb, {}))
        p_str = out[1].cpu().numpy()[0]
        s_str = out[2].cpu().numpy()[0]
        p_pick, _ = pick_from_stream(p_str, ev.peak_height, peak_distance)
        s_pick, _ = pick_from_stream(s_str, ev.peak_height, peak_distance)
        return p_pick, s_pick

    def sec(sample):
        if sample is None or not np.isfinite(sample):
            return None
        return round(float(sample) / fs, 3)

    t_axis = block_time(n_samples, DOWNSAMPLE_POINTS, fs)
    t_axis_list = [round(float(v), 3) for v in t_axis]

    traces_out = []
    order = 0
    for kind, r in selection:
        order += 1
        i = r["i"]
        pred, latency_ms = student_streams_timed(i)
        raw = _get_waveform(ds.ds, r["row"], cfg.data.n_channels, n_samples)
        tp_pick, ts_pick = teacher_picks(raw)

        m = meta.iloc[r["row"]]
        trace_name = str(m.get("trace_name_original") or "").strip()
        if not trace_name or trace_name == "nan":
            net = str(m.get(S.COL_NETWORK, "")).strip()
            sta = str(m.get(S.COL_STATION, "")).strip()
            trace_name = f"{net}.{sta}".strip(".") or str(m.get(S.COL_TRACE, f"row{r['row']}"))

        snr = r["snr"]
        snr_txt = f"{snr:.1f} dB" if np.isfinite(snr) else "n/a (noise)"

        # per-channel peak-normalized, downsampled waveform (visual)
        wf = {}
        for ch, name in enumerate(("Z", "N", "E")):
            d = raw[ch].astype(np.float64)
            scale = max(float(np.max(np.abs(d))), 1e-9)
            red = downsample_block(d / scale, DOWNSAMPLE_POINTS, "peak")
            wf[name] = [round(float(v), 3) for v in red]

        # student probability streams, downsampled
        det = [round(float(v), 4) for v in downsample_block(pred[0], DOWNSAMPLE_POINTS, "mean")]
        pph = [round(float(v), 4) for v in downsample_block(pred[1], DOWNSAMPLE_POINTS, "mean")]
        sph = [round(float(v), 4) for v in downsample_block(pred[2], DOWNSAMPLE_POINTS, "mean")]

        # student picks + confidence (full-resolution)
        p_pick_s = sec(r["p_pick"])
        s_pick_s = sec(r["s_pick"])
        p_conf = float(pred[1][int(r["p_pick"])]) if r["p_pick"] is not None else None
        s_conf = float(pred[2][int(r["s_pick"])]) if r["s_pick"] is not None else None
        det_max = float(np.max(pred[0]))

        # human-friendly label; append the miss magnitude for the failure case
        label = KIND_LABEL[kind]
        if kind == "worst_case":
            if worst_fp is not None:
                label = "Failure case (noise false positive)"
            elif r["p_err"] is not None:
                label = f"Failure case ({r['p_err'] / 1000.0:.1f}s P miss)"

        traces_out.append({
            "trace_id": trace_name,
            "kind": kind,
            "label": label,
            "snr_db": None if not np.isfinite(snr) else round(float(snr), 1),
            "snr_label": snr_txt,
            "is_earthquake": bool(r["is_eq"]),
            "detected": bool(r["detected"]),
            "waveform": wf,
            "student": {"detection": det, "p": pph, "s": sph},
            "truth": {"p_s": sec(r["gt_p"]), "s_s": sec(r["gt_s"])},
            "teacher": {"p_s": sec(tp_pick), "s_s": sec(ts_pick)},
            "student_pick": {
                "p_s": p_pick_s, "s_s": s_pick_s,
                "p_conf": None if p_conf is None else round(p_conf, 3),
                "s_conf": None if s_conf is None else round(s_conf, 3),
                "det_max": round(det_max, 3),
            },
            "p_error_ms": None if r["p_err"] is None else round(float(r["p_err"]), 1),
            "inference_latency_ms": round(latency_ms, 3),
        })
        print(f"  packed {order}: {kind:12s} id={trace_name} snr={snr_txt} "
              f"lat={latency_ms:.2f}ms")

    payload = {
        "meta": {
            "project": "Jormungandr",
            "description": "48k-param distilled SeismicUNet for real-time edge "
                           "P/S phase picking, distilled from EQTransformer.",
            "sampling_rate_hz": fs,
            "window_s": n_samples / fs,
            "n_samples_full": n_samples,
            "downsample_points": len(t_axis_list),
            "thresholds": {
                "detection": float(ev.detection_threshold),
                "peak_height": float(ev.peak_height),
                "match_tolerance_s": float(ev.match_tolerance_s),
            },
            "edge_latency_ms": {
                "device": "Raspberry Pi 5 (Cortex-A76, INT8 ONNX Runtime)",
                "int8_p50": 2.392,
                "fp32_p50": 3.095,
                "note": "measured on-device; per-trace inference_latency_ms below "
                        "is this host's fp32 PyTorch forward pass.",
            },
            "time_axis_s": t_axis_list,
        },
        "traces": traces_out,
    }

    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    size_kb = OUT.stat().st_size / 1024.0
    print(f"\nwrote {OUT}  ({size_kb:.1f} KB, {len(traces_out)} traces)")
    if size_kb > 1024:
        print("WARNING: file exceeds 1 MB")


if __name__ == "__main__":
    main()
