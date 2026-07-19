#!/usr/bin/env python
"""Export a real replay run into a self-contained, offline HTML demo.

Runs the SAME causal pipeline as ``demo_edge.py --source replay`` over the real
STEAD replay bundle, then serializes everything the browser needs to animate the
transmission-gating story:

  Act 1  each real earthquake trace revealed left->right in real time -- the 3-C
         waveform, the model's detection confidence, and the P / S picks landing
         as the waves arrive (model vs catalog).
  Act 2  the session's byte accounting -- raw-continuous vs STA/LTA-gated vs
         model-gated -- the headline "bytes on the wire" comparison.

Outputs (into outputs/demo/):
  gating_traces.json   the raw data bundle (also handy for other tooling)
  gating_demo.html     self-contained page (data inlined) -- just open it

Everything is reused from demo_edge.py (no reimplemented scoring/gating), so the
numbers match bytes_report.json exactly.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import demo_edge as D  # sets up sys.path (src + scripts) on import

REPO = D.REPO
DEMO_DIR = D.DEMO_DIR
TEMPLATE = REPO / "scripts" / "gating_demo_template.html"


def _decimate_peak(x: np.ndarray, edges: np.ndarray) -> list[float]:
    """Peak-preserving decimation: per bin keep the sample of largest magnitude
    (so a waveform's wiggle envelope survives instead of aliasing away)."""
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        seg = x[a:b]
        if seg.size == 0:
            out.append(0.0)
        else:
            out.append(float(seg[np.argmax(np.abs(seg))]))
    return out


def _decimate_mean(x: np.ndarray, edges: np.ndarray) -> list[float]:
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        seg = x[a:b]
        out.append(float(seg.mean()) if seg.size else 0.0)
    return out


def _round(xs, n):
    return [round(v, n) for v in xs]


# Manual context for traces that carry no STEAD source metadata (noise has no
# lat/lon/magnitude -- these are recording-site notes, not derived from data).
NOISE_LABELS = {
    "MOOW.IW_20180115033742_NO": "Construction Site",
}


def _rough_location(lat, lon):
    """Coarse, human-readable region for a real STEAD source lat/lon. Not a
    geocoder -- just enough bucketing to label the handful of regions STEAD's
    curated test events actually come from."""
    if lat is None or lon is None or not (np.isfinite(lat) and np.isfinite(lon)):
        return None
    if -56 <= lat <= -17 and -76 <= lon <= -66:
        if lat < -33:
            return "Southern Chile"
        if lat < -27:
            return "Central Chile"
        return "Northern Chile"
    if 32 <= lat <= 42 and -125 <= lon <= -114:
        return "Central California" if lat >= 35.5 else "Southern California"
    if 42 <= lat <= 49 and -125 <= lon <= -116:
        return "Pacific Northwest"
    return f"{abs(lat):.1f}°{'N' if lat >= 0 else 'S'}, {abs(lon):.1f}°{'E' if lon >= 0 else 'W'}"


def _trace_label(trace_id, eq, lat, lon):
    if not eq:
        return f"{NOISE_LABELS.get(trace_id, 'Ambient Site')} — Noise"
    return f"{_rough_location(lat, lon) or 'Unknown Location'} — Earthquake"


def build_bundle(cfg, bundle, predict, fs, bins=1100):
    hop = int(round(D.HOP_S * fs))
    clip_n = int((D.CLIP_PRE_S + D.CLIP_POST_S) * fs)

    ids = bundle["trace_id"]
    is_eq = bundle["is_eq"]
    p_samp = bundle["p_sample"]
    s_samp = bundle["s_sample"]
    raws = bundle["raw"]
    n_trace = raws.shape[0]
    lat_arr = bundle["source_latitude_deg"] if "source_latitude_deg" in bundle.files else None
    lon_arr = bundle["source_longitude_deg"] if "source_longitude_deg" in bundle.files else None
    time_arr = bundle["source_origin_time"] if "source_origin_time" in bundle.files else None

    traces = []
    # continuous session counters (Act 2) -- reproduces demo_edge.build_report,
    # which resets both triggers per segment (seg_new) just as we do here.
    raw_bytes = stalta_bytes = model_bytes = 0.0
    stalta_count = model_count = stalta_false = model_false = 0

    for ti in range(n_trace):
        raw = raws[ti].astype(np.float32)
        n = raw.shape[1]
        pre = D.CausalPreprocessor(cfg, warmup_samples=min(int(fs), hop))
        probs = D.causal_stream_probabilities(
            raw, predict, pre, chunk_samples=500).probabilities  # (3, N): det, P, S
        ratio = D._stalta(raw, fs)

        eq = bool(is_eq[ti])
        if eq:
            ew = [((p_samp[ti]) / fs - 1.0, (s_samp[ti]) / fs + 3.0)]
        else:
            ew = []  # pure-noise trace: any trigger is a false transmission
        seg_picks = D._window_picks(probs, fs, 0.0)

        # --- gate both detectors hop-by-hop (identical to demo_edge.run) ------
        m_trig = D.EdgeTrigger(D.DET_THRESHOLD, off=0.6, refractory_s=D.REFRACTORY_S)
        s_trig = D.EdgeTrigger(D.STA_LTA_ON, off=D.STA_LTA_ON, refractory_s=D.REFRACTORY_S)
        fires = []
        for pos in range(0, n, hop):
            end = min(pos + hop, n)
            t_s = end / fs
            raw_bytes += hop * 3 * D.BYTES_PER_SAMPLE
            det = float(probs[0, pos:end].max())
            rat = float(ratio[pos:end].max())
            in_event = any(a <= t_s <= b for a, b in ew)
            clip = raw[:, max(0, end - clip_n):end]
            picks = {
                "P": seg_picks["P"] if (seg_picks["P"] is not None and seg_picks["P"] <= t_s) else None,
                "S": seg_picks["S"] if (seg_picks["S"] is not None and seg_picks["S"] <= t_s) else None,
            }
            if s_trig.update(rat, t_s):
                b = D.serialize_packet(clip, picks, f"REPLAY {ti}", t_s)
                false = (in_event is False)
                stalta_bytes += b; stalta_count += 1; stalta_false += int(false)
                fires.append({"gate": "stalta", "t": round(t_s, 2), "bytes": b, "false": false})
            if m_trig.update(det, t_s):
                b = D.serialize_packet(clip, picks, f"REPLAY {ti}", t_s)
                false = (in_event is False)
                model_bytes += b; model_count += 1; model_false += int(false)
                fires.append({"gate": "model", "t": round(t_s, 2), "bytes": b, "false": false})

        # --- decimate the drawable streams onto one shared time axis ----------
        edges = np.linspace(0, n, bins + 1).astype(int)
        centers = (edges[:-1] + edges[1:]) / 2.0 / fs
        gmax = float(np.abs(raw).max()) or 1.0
        wave = [_round(_decimate_peak(raw[ch] / gmax, edges), 3) for ch in range(3)]

        lat = float(lat_arr[ti]) if lat_arr is not None else None
        lon = float(lon_arr[ti]) if lon_arr is not None else None
        date = None
        if time_arr is not None:
            raw_time = str(time_arr[ti])
            if raw_time and raw_time not in ("NaT", "nan", ""):
                date = raw_time[:10]
        traces.append({
            "id": str(ids[ti]),
            "label": _trace_label(str(ids[ti]), eq, lat, lon),
            "date": date,
            "is_eq": eq,
            "n_seconds": round(n / fs, 2),
            "t": _round(centers.tolist(), 3),
            "wave": wave,
            "det": _round(_decimate_mean(probs[0], edges), 3),
            "pp": _round(_decimate_mean(probs[1], edges), 3),
            "sp": _round(_decimate_mean(probs[2], edges), 3),
            "ratio": _round([min(v, 10.0) for v in _decimate_mean(ratio, edges)], 2),
            "catalog": {
                "p": round(float(p_samp[ti]) / fs, 2) if eq else None,
                "s": round(float(s_samp[ti]) / fs, 2) if eq else None,
            },
            "model": {"p": seg_picks["P"], "s": seg_picks["S"]},
            "fires": fires,
        })

    def rr(a, b):
        return None if b <= 0 else round(a / b, 1)

    session = {
        "elapsed_seconds": round(raw_bytes / (3 * D.BYTES_PER_SAMPLE) / fs, 1),
        "bytes": {
            "raw_continuous": int(raw_bytes),
            "stalta_gated": int(stalta_bytes),
            "model_gated": int(model_bytes),
        },
        "reduction_vs_raw": {"stalta": rr(raw_bytes, stalta_bytes),
                             "model": rr(raw_bytes, model_bytes)},
        "model_vs_stalta_bytes_x": rr(stalta_bytes, model_bytes),
        "transmissions": {
            "stalta": {"count": stalta_count, "false": stalta_false},
            "model": {"count": model_count, "false": model_false},
        },
    }

    data = {
        "meta": {
            "generated": time.strftime("%Y-%m-%d %H:%M"),
            "source": "replay (replay_traces.npz, REAL STEAD)",
            "fs": int(fs),
            "hop_s": D.HOP_S,
            "det_threshold": D.DET_THRESHOLD,
            "stalta_on": D.STA_LTA_ON,
            "bytes_per_sample": D.BYTES_PER_SAMPLE,
            "clip_s": D.CLIP_PRE_S + D.CLIP_POST_S,
        },
        "session": session,
        "traces": traces,
    }
    return data


def main():
    ap = argparse.ArgumentParser(description="Export the replay run to an offline HTML demo.")
    ap.add_argument("--trace", default=str(DEMO_DIR / "replay_traces.npz"))
    ap.add_argument("--onnx", default=str(DEMO_DIR / "causal_stage3_int8.onnx"))
    ap.add_argument("--config", default=str(REPO / "configs/default.yaml"))
    ap.add_argument("--json", default=str(DEMO_DIR / "gating_traces.json"))
    ap.add_argument("--html", default=str(DEMO_DIR / "gating_demo.html"))
    ap.add_argument("--bins", type=int, default=1100, help="drawable points per trace")
    args = ap.parse_args()

    cfg = D.load_config(args.config)
    fs = float(cfg.data.sampling_rate)

    onnx_path = Path(args.onnx)
    if not onnx_path.is_file():
        raise SystemExit(f"causal INT8 ONNX not found: {onnx_path}")
    session, name_in, name_out = D.make_session(onnx_path, 1)
    predict = D._predictor(session, name_in, name_out)

    bundle = np.load(args.trace, allow_pickle=True)
    print(f"[export] scoring {bundle['raw'].shape[0]} real STEAD traces through the causal pipeline...")
    data = build_bundle(cfg, bundle, predict, fs, bins=args.bins)

    s = data["session"]
    print(f"[export] raw {D.fmt_bytes(s['bytes']['raw_continuous'])}  "
          f"STA/LTA {D.fmt_bytes(s['bytes']['stalta_gated'])} "
          f"({s['transmissions']['stalta']['count']} tx)  "
          f"model {D.fmt_bytes(s['bytes']['model_gated'])} "
          f"({s['transmissions']['model']['count']} tx)  "
          f"-> model sends {s['model_vs_stalta_bytes_x']}x fewer bytes")

    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, separators=(",", ":"))
    Path(args.json).write_text(payload)

    if not TEMPLATE.is_file():
        raise SystemExit(f"template not found: {TEMPLATE}")
    html = TEMPLATE.read_text().replace("/*__GATING_DATA__*/null", payload)
    Path(args.html).write_text(html)

    print(f"[export] wrote {args.json}\n"
          f"[export] wrote {args.html}   ({len(html)//1024} KB, self-contained)\n"
          f"[export] open it:  open {args.html}")


if __name__ == "__main__":
    main()
