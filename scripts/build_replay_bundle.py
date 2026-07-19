#!/usr/bin/env python
"""Build the demo REPLAY bundle: real STEAD traces the causal model can pick.

DATASET-GATED. Selects a handful of clean high-SNR earthquakes (plus one noise
trace) from the STEAD test split, streams each through the SAME causal pipeline
the live demo uses (CausalPreprocessor -> causal INT8 ONNX ->
causal_stream_probabilities), and keeps only traces where the model actually
recovers P and S near the catalog arrivals. The survivors are saved RAW (no
preprocessing) with ground-truth P/S sample indices + trace_ids to
``outputs/demo/replay_traces.npz`` so scripts/demo_edge.py --source replay can
reproduce genuine P-then-S picks on a laptop with no STEAD access.

This is the load-bearing artifact for the replay screen recording: it is checked
in, and the selection here is exactly what the demo replays.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from seismic_edge_picker.config import load_config  # noqa: E402
from seismic_edge_picker.preprocessing import CausalPreprocessor  # noqa: E402
from seismic_edge_picker.streaming import (  # noqa: E402
    causal_stream_probabilities,
    extract_phase_picks,
)
from false_alarm_rate import make_session  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Build the demo replay bundle from STEAD.")
    p.add_argument("--config", default=str(REPO / "configs/default.yaml"))
    p.add_argument("--onnx", default=str(REPO / "outputs/demo/causal_stage3_int8.onnx"))
    p.add_argument("--out", default=str(REPO / "outputs/demo/replay_traces.npz"))
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--n-eq", type=int, default=4, help="clean EQ traces to keep")
    p.add_argument("--n-noise", type=int, default=1)
    p.add_argument("--scan", type=int, default=600, help="test traces to scan")
    p.add_argument("--chunk-samples", type=int, default=500)
    p.add_argument("--p-threshold", type=float, default=0.30)
    p.add_argument("--s-threshold", type=float, default=0.30)
    p.add_argument("--match-tol-s", type=float, default=0.6,
                   help="max |pick - catalog| to accept a phase as recovered")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    fs = float(cfg.data.sampling_rate)
    n_samples = int(cfg.data.window_samples)
    n_ch = int(cfg.data.n_channels)
    tol = args.match_tol_s

    from seismic_edge_picker import splits as S
    from seismic_edge_picker.dataset import _get_waveform, build_datasets

    datasets, _ = build_datasets(cfg)
    ds = datasets[args.split]
    meta = ds.metadata
    cat = meta[S.COL_CATEGORY].astype(str)
    is_eq_all = cat.isin(S.EARTHQUAKE_VALUES).to_numpy()
    print(f"{args.split} split: {len(ds)} traces; scanning up to {args.scan}")

    session, name_in, name_out = make_session(Path(args.onnx), 1)

    def predict(x):  # x: (1,3,6000) already causally preprocessed
        return session.run([name_out], {name_in: x.astype(np.float32)})[0]

    def stream_probs(raw):
        pre = CausalPreprocessor(cfg, warmup_samples=min(int(fs), args.chunk_samples))
        return causal_stream_probabilities(
            raw, predict, pre, chunk_samples=args.chunk_samples
        ).probabilities

    # ---- score candidate earthquakes by SNR, keep the cleanly-picked ones ----
    cand = []
    for i in range(min(args.scan, len(ds))):
        row = int(ds.rows[i])
        m = meta.iloc[row]
        if not bool(is_eq_all[row]):
            continue
        gt_p = S.parse_scalar(m.get(S.COL_P))
        gt_s = S.parse_scalar(m.get(S.COL_S))
        snr = S.parse_snr_db(m.get(S.COL_SNR))
        if not (np.isfinite(gt_p) and np.isfinite(gt_s) and np.isfinite(snr)):
            continue
        cand.append((float(snr), i, row, int(round(gt_p)), int(round(gt_s))))
    cand.sort(key=lambda c: -c[0])  # highest SNR first
    print(f"  {len(cand)} EQ candidates with finite P/S/SNR")

    kept_eq = []
    for snr, i, row, gt_p, gt_s in cand:
        if len(kept_eq) >= args.n_eq:
            break
        raw = _get_waveform(ds.ds, row, n_ch, n_samples).astype(np.float32)
        probs = stream_probs(raw)
        picks = extract_phase_picks(
            probs, fs, args.p_threshold, args.s_threshold, cfg.eval.peak_min_distance_s
        )
        p_picks = [pk for pk in picks if pk["phase"] == "P"]
        s_picks = [pk for pk in picks if pk["phase"] == "S"]

        def best(picks_list, gt):
            if not picks_list:
                return None
            return min(picks_list, key=lambda pk: abs(pk["sample"] - gt))

        bp = best(p_picks, gt_p)
        bs = best(s_picks, gt_s)
        p_err = None if bp is None else abs(bp["sample"] - gt_p) / fs
        s_err = None if bs is None else abs(bs["sample"] - gt_s) / fs
        ok = (p_err is not None and p_err <= tol and s_err is not None and s_err <= tol)
        trace_id = str(m_trace_id(meta.iloc[row], S, row))
        print(f"  snr={snr:5.1f}dB id={trace_id:>16s}  "
              f"P_err={'--' if p_err is None else f'{p_err*1000:5.0f}ms'}  "
              f"S_err={'--' if s_err is None else f'{s_err*1000:5.0f}ms'}  "
              f"detmax={probs[0].max():.2f}  {'KEEP' if ok else 'skip'}")
        if ok:
            kept_eq.append({
                "trace_id": trace_id, "raw": raw, "is_eq": True,
                "p_sample": gt_p, "s_sample": gt_s, "snr_db": float(snr),
                "p_pick": bp["sample"], "s_pick": bs["sample"],
            })

    # ---- one clean noise trace (model should stay quiet) ---------------------
    kept_noise = []
    for i in range(min(args.scan, len(ds))):
        if len(kept_noise) >= args.n_noise:
            break
        row = int(ds.rows[i])
        if bool(is_eq_all[row]):
            continue
        m = meta.iloc[row]
        raw = _get_waveform(ds.ds, row, n_ch, n_samples).astype(np.float32)
        probs = stream_probs(raw)
        trace_id = str(m_trace_id(m, S, row))
        detmax = float(probs[0].max())
        print(f"  NOISE id={trace_id:>16s}  detmax={detmax:.2f}")
        kept_noise.append({
            "trace_id": trace_id, "raw": raw, "is_eq": False,
            "p_sample": -1, "s_sample": -1, "snr_db": float("nan"),
            "p_pick": -1, "s_pick": -1,
        })

    kept = kept_eq + kept_noise
    if len(kept_eq) < 1:
        raise SystemExit("no earthquake trace picked cleanly — loosen --match-tol-s")

    # ---- serialize -----------------------------------------------------------
    raws = np.stack([k["raw"] for k in kept]).astype(np.float32)   # (T,3,6000)
    np.savez_compressed(
        args.out,
        raw=raws,
        trace_id=np.array([k["trace_id"] for k in kept]),
        is_eq=np.array([k["is_eq"] for k in kept]),
        p_sample=np.array([k["p_sample"] for k in kept], dtype=np.int64),
        s_sample=np.array([k["s_sample"] for k in kept], dtype=np.int64),
        snr_db=np.array([k["snr_db"] for k in kept], dtype=np.float64),
        sampling_rate=np.float64(fs),
        window_samples=np.int64(n_samples),
    )
    print(f"\nwrote {args.out}: {len(kept_eq)} EQ + {len(kept_noise)} noise, "
          f"{Path(args.out).stat().st_size/1024:.0f} KB")


def m_trace_id(m, S, row):
    tn = str(m.get("trace_name_original") or "").strip()
    if tn and tn != "nan":
        return tn
    net = str(m.get(S.COL_NETWORK, "")).strip()
    sta = str(m.get(S.COL_STATION, "")).strip()
    return f"{net}.{sta}".strip(".") or str(m.get(S.COL_TRACE, f"row{row}"))


if __name__ == "__main__":
    main()
