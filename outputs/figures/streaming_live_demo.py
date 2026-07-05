#!/usr/bin/env python
"""Task 3 — live streaming inference demo (terminal screen-recording friendly).

Concatenates several cached STEAD test traces (earthquakes) with gaps of pure
noise between them into one long continuous 100 Hz signal, then feeds it through
the project's streaming wrapper (``seismic_edge_picker.streaming``) window by
window using the deployed INT8 ONNX model. For each 60 s / 30 s-hop window it
prints, in real time:

    wall timestamp | window index | per-window inference latency (ms) | live
    detection state, and any newly-finalized events with their P/S pick times.

A small sleep between windows simulates a live acquisition pace so the output
scrolls at a readable rate on a screen recording.

Usage:
    python outputs/figures/streaming_live_demo.py                 # 3 events
    python outputs/figures/streaming_live_demo.py --events 4 --pace 0.35
    python outputs/figures/streaming_live_demo.py --no-color      # plain text
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from seismic_edge_picker.config import load_config  # noqa: E402
from seismic_edge_picker import splits as S  # noqa: E402
from seismic_edge_picker.streaming import (  # noqa: E402
    window_starts,
    extract_events,
    extract_phase_picks,
    associate_picks,
)
from stream_infer import make_session  # reuse the exact ORT session setup  # noqa: E402


class Palette:
    def __init__(self, enabled: bool):
        self.on = enabled

    def _w(self, code, s):
        return f"\033[{code}m{s}\033[0m" if self.on else s

    def dim(self, s):
        return self._w("2", s)

    def bold(self, s):
        return self._w("1", s)

    def blue(self, s):
        return self._w("38;5;39", s)

    def green(self, s):
        return self._w("38;5;35", s)

    def orange(self, s):
        return self._w("38;5;208", s)

    def red(self, s):
        return self._w("38;5;203", s)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--events", type=int, default=3,
                   help="number of earthquake traces to stream (default: 3)")
    p.add_argument("--gap-s", type=float, default=60.0,
                   help="seconds of pure noise between events (default: 60)")
    p.add_argument("--pace", type=float, default=0.4,
                   help="sleep seconds between windows to simulate live pace")
    p.add_argument("--merge-gap-s", type=float, default=15.0,
                   help="coalesce detection bursts within this gap into one event, "
                        "so a P + S pair reads as a single earthquake (default: 15)")
    p.add_argument("--detection-threshold", type=float, default=0.8,
                   help="detection trigger level (default: 0.80, the deploy "
                        "low-false-alarm streaming point)")
    p.add_argument("--model", default=str(REPO / "outputs/onnx/stage2_distill_int8.onnx"))
    p.add_argument("--config", default=str(REPO / "configs/default.yaml"))
    p.add_argument("--no-color", action="store_true")
    return p.parse_args()


def build_signal(cfg, n_events, gap_s, screen, det_thr):
    """Concatenate [noise-gap, eq, noise-gap, eq, ...] into one (3, N) signal.

    Earthquake segments are pre-screened with ``screen`` (a single-window
    predictor): only traces the model already resolves cleanly in isolation —
    detection above ``det_thr`` and both a P and an S pick within the eval
    tolerance of the labelled arrivals — are used, so every injected event
    triggers with a genuine P/S pick. Noise traces supply the quiet gaps.

    Returns the signal plus the ground-truth P/S absolute times (s); the model
    itself is never given the labels.
    """
    from seismic_edge_picker.dataset import build_datasets
    from evaluate import pick_from_stream

    datasets, _ = build_datasets(cfg)
    ds = datasets["test"]
    fs = float(cfg.data.sampling_rate)
    ev = cfg.eval
    tol = ev.match_tolerance_s * fs
    dist = max(1, int(round(ev.peak_min_distance_s * fs)))
    cat = ds.metadata[S.COL_CATEGORY].astype(str)

    eq_cands, noise_local = [], []
    for li, row in enumerate(ds.rows):
        c = cat.iloc[int(row)]
        m = ds.metadata.iloc[int(row)]
        if c in S.EARTHQUAKE_VALUES:
            p = S.parse_scalar(m.get(S.COL_P))
            s = S.parse_scalar(m.get(S.COL_S))
            snr = S.parse_snr_db(m.get(S.COL_SNR))
            # a well-separated P/S pair reads best in a live event card
            if (np.isfinite(p) and np.isfinite(s) and np.isfinite(snr)
                    and 5.0 <= snr <= 60.0 and (s - p) >= 300):
                eq_cands.append((snr, li, int(round(p)), int(round(s))))
        elif c in S.NOISE_VALUES:
            noise_local.append(li)

    # screen highest-SNR-first for traces the model resolves cleanly alone:
    # both P and S must pick within tolerance AND drive detection above the
    # trigger, and detection must be quiet at both edges so codas don't bleed
    # across the concatenation boundaries and fake a trigger.
    def around(stream, idx, half=50):
        lo, hi = max(0, idx - half), min(stream.shape[0], idx + half)
        return float(stream[lo:hi].max())

    eq_local = []
    for snr, li, gp, gs in sorted(eq_cands, reverse=True):
        wf, _ = ds[li]
        det, p_str, s_str = screen(wf.numpy())
        pp, _ = pick_from_stream(p_str, ev.peak_height, dist)
        sp, _ = pick_from_stream(s_str, ev.peak_height, dist)
        if (pp is not None and sp is not None
                and abs(pp - gp) <= tol and abs(sp - gs) <= tol
                and around(det, pp) >= det_thr and around(det, sp) >= det_thr
                and det[:100].max() < det_thr and det[-300:].max() < det_thr):
            eq_local.append(li)
        if len(eq_local) == n_events:
            break
    if len(eq_local) < n_events:
        raise SystemExit("could not screen enough clean earthquake traces")
    gap_samples = int(round(gap_s * fs))

    def taper(seg, ramp=50):
        """Cosine-ramp both edges so concatenation steps don't fake a trigger."""
        seg = seg.copy()
        ramp = min(ramp, seg.shape[1] // 2)
        if ramp > 0:
            w = 0.5 * (1 - np.cos(np.linspace(0, np.pi, ramp)))
            seg[:, :ramp] *= w
            seg[:, -ramp:] *= w[::-1]
        return seg

    def noise_gap(k):
        wf, _ = ds[noise_local[k % len(noise_local)]]
        seg = wf.numpy()
        if seg.shape[1] < gap_samples:
            reps = int(np.ceil(gap_samples / seg.shape[1]))
            seg = np.tile(seg, (1, reps))
        return taper(seg[:, :gap_samples])

    segments, truth, cursor = [], [], 0
    for e, li in enumerate(eq_local):
        g = noise_gap(e)
        segments.append(g)
        cursor += g.shape[1]
        wf, _ = ds[li]
        segments.append(taper(wf.numpy()))
        m = ds.metadata.iloc[int(ds.rows[li])]
        for phase, col in (("P", S.COL_P), ("S", S.COL_S)):
            a = S.parse_scalar(m.get(col))
            if np.isfinite(a):
                truth.append((phase, (cursor + a) / fs))
        cursor += wf.numpy().shape[1]
    segments.append(noise_gap(len(eq_local)))  # trailing quiet
    signal = np.concatenate(segments, axis=1).astype(np.float32)
    return signal, sorted(truth, key=lambda t: t[1]), fs


def hms():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def main():
    args = parse_args()
    C = Palette(enabled=not args.no_color and sys.stdout.isatty())
    cfg = load_config(args.config)
    stream_cfg = cfg.deploy.streaming
    fs = float(cfg.data.sampling_rate)
    window_samples = cfg.data.window_samples
    hop_samples = int(round(stream_cfg.hop_s * fs))
    det_thr = float(args.detection_threshold)
    p_thr = float(stream_cfg.p_threshold)
    s_thr = float(stream_cfg.s_threshold)
    peak_dist_s = float(stream_cfg.peak_min_distance_s)

    session, in_name, out_name, ort_ver = make_session(args.model, threads=1)

    def screen(window):
        """Single 60 s window -> (det, P, S) probability streams."""
        batch = np.zeros((1, 3, window_samples), dtype=np.float32)
        valid = min(window_samples, window.shape[1])
        batch[0, :, :valid] = window[:, :valid]
        return session.run([out_name], {in_name: batch})[0][0]

    print(C.dim("building continuous signal from cached test traces ..."))
    signal, truth, fs = build_signal(cfg, args.events, args.gap_s, screen, det_thr)
    n = signal.shape[1]
    starts = window_starts(n, window_samples, hop_samples)

    bar = C.dim("─" * 68)
    print(bar)
    print(C.bold("  Seismic edge picker — live streaming inference"))
    print(f"  model   : {C.blue(Path(args.model).name)}  (ONNX Runtime {ort_ver}, 1 thread, INT8)")
    print(f"  signal  : {n:,} samples / {n / fs:,.0f} s  ({args.events} events + noise gaps)")
    print(f"  windows : {len(starts)} x {window_samples // int(fs)} s, hop {int(hop_samples / fs)} s")
    print(f"  trigger : detection ≥ {det_thr:g}, P/S peak ≥ {p_thr:g}/{s_thr:g}, "
          f"merge gap {args.merge_gap_s:g} s")
    print(bar)
    header = f"  {'wall clock':<13} {'window':>9}  {'latency':>8}   detection"
    print(C.dim(header))

    total = np.zeros((3, n), dtype=np.float32)
    coverage = np.zeros(n, dtype=np.int32)
    announced = 0
    latencies = []

    for k, start in enumerate(starts):
        valid = min(window_samples, n - start)
        batch = np.zeros((1, 3, window_samples), dtype=np.float32)
        batch[0, :, :valid] = signal[:, start:start + valid]

        t0 = time.perf_counter()
        pred = session.run([out_name], {in_name: batch})[0]
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt_ms)

        total[:, start:start + valid] += pred[0, :, :valid]
        coverage[start:start + valid] += 1

        win_det_max = float(pred[0, 0, :valid].max())
        # sparkline of this window's detection stream (8 buckets)
        buckets = np.array_split(pred[0, 0, :valid], 8)
        blocks = "▁▂▃▄▅▆▇█"
        spark = "".join(blocks[min(7, int(b.max() * 8))] for b in buckets)
        state = (C.green("● ARMED  ") if win_det_max >= det_thr
                 else C.dim("· quiet  "))
        spark_c = C.green(spark) if win_det_max >= det_thr else C.dim(spark)
        lat_c = C.orange(f"{dt_ms:5.2f} ms")
        print(f"  {C.dim(hms()):<13} {C.bold(f'{k + 1:>3}/{len(starts)}'):>9}  "
              f"{lat_c:>8}   {state} {spark_c}  max {win_det_max:0.2f}")

        # finalize everything strictly before this window's start (no later
        # window can touch it) and announce newly-completed events with picks
        finalized_end = n if k == len(starts) - 1 else start
        if finalized_end > 0 and coverage[:finalized_end].min() > 0:
            merged = total[:, :finalized_end] / coverage[:finalized_end][None, :]
            events = extract_events(merged[0], fs, det_thr,
                                    stream_cfg.min_duration_ms,
                                    args.merge_gap_s)
            if len(events) > announced:
                picks = associate_picks(
                    extract_phase_picks(merged, fs, p_thr, s_thr, peak_dist_s),
                    events, fs, stream_cfg.association_margin_s)
                for ev in events[announced:]:
                    p = [pk for pk in picks if pk["event_id"] == ev["event_id"]]
                    ptime = next((pk["time_s"] for pk in p if pk["phase"] == "P"), None)
                    stime = next((pk["time_s"] for pk in p if pk["phase"] == "S"), None)
                    ptxt = C.blue(f"P {ptime:7.2f}s") if ptime is not None else C.dim("P    —   ")
                    stxt = C.orange(f"S {stime:7.2f}s") if stime is not None else C.dim("S    —   ")
                    print(C.green(
                        f"      ┏━ EVENT #{ev['event_id']} @ {ev['start_time_s']:.1f}–"
                        f"{ev['end_time_s']:.1f}s  peak {ev['peak_probability']:.2f}   "
                    ) + f"{ptxt}  {stxt}")
                announced = len(events)

        time.sleep(max(0.0, args.pace))

    print(bar)
    lat = np.array(latencies)
    print(f"  streamed {len(starts)} windows | latency p50 {np.percentile(lat, 50):.2f} ms  "
          f"p95 {np.percentile(lat, 95):.2f} ms  mean {lat.mean():.2f} ms")
    print(f"  detected {announced} event(s) over {n / fs:.0f}s of signal "
          f"({args.events} earthquakes injected)")
    if truth:
        gt = "  ".join(f"{ph} {tt:.1f}s" for ph, tt in truth)
        print(C.dim(f"  ground-truth arrivals: {gt}"))
    print(bar)


if __name__ == "__main__":
    main()
