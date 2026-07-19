#!/usr/bin/env python
"""Find a REAL STEAD noise trace that makes STA/LTA false-trigger while the
causal model correctly stays quiet, and append it to the demo replay bundle.

DATASET-GATED (needs STEAD cached, same as build_replay_bundle.py). The 5-clip
outputs/demo/replay_traces.npz shipped in the repo was curated for clean picks
only (build_replay_bundle.py explicitly keeps "clean high-SNR earthquakes" and
one quiet noise trace) -- it has zero false triggers by construction, which is
why the gating_minimal.mp4 video can't show the false-positive side of the
story. The aggregate proof that STA/LTA really does false-trigger more than
the model lives in outputs/causal/run.json (17.51 vs 3.08 false triggers/hr),
but that's a statistic over the whole STEAD test split -- no single offending
clip is saved anywhere. This script finds one and adds it as a 6th trace.

Scans the STEAD noise split (same tuned STA/LTA params as demo_edge.py: sta=1s,
lta=5s, on=3.0) for a trace where the ratio crosses the trigger threshold at
some point but the causal model's detection probability never does. Appends
it (is_eq=False, like the existing noise trace) to replay_traces.npz so the
existing scripts/demo_edge.py, export_gating_traces.py, and
export_gating_video.py pick it up with no changes.

Run on the VM where STEAD is cached:

    python3 scripts/find_false_positive_clip.py
    # then, back wherever the video is rendered:
    python3 scripts/export_gating_traces.py
    python3 scripts/export_gating_video.py
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
from seismic_edge_picker.streaming import causal_stream_probabilities  # noqa: E402
from false_alarm_rate import make_session  # noqa: E402
from demo_edge import _stalta, STA_LTA_ON, DET_THRESHOLD  # noqa: E402
from build_replay_bundle import m_trace_id  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(REPO / "configs/default.yaml"))
    p.add_argument("--onnx", default=str(REPO / "outputs/demo/causal_stage3_int8.onnx"))
    p.add_argument("--bundle", default=str(REPO / "outputs/demo/replay_traces.npz"))
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--scan", type=int, default=2000, help="noise traces to scan")
    p.add_argument("--chunk-samples", type=int, default=500)
    p.add_argument("--det-margin", type=float, default=0.15,
                    help="require model max-detection to stay this far below "
                         "DET_THRESHOLD, so the contrast reads clearly on video")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    fs = float(cfg.data.sampling_rate)
    n_samples = int(cfg.data.window_samples)
    n_ch = int(cfg.data.n_channels)

    from seismic_edge_picker import splits as S
    from seismic_edge_picker.dataset import _get_waveform, build_datasets

    datasets, _ = build_datasets(cfg)
    ds = datasets[args.split]
    meta = ds.metadata
    cat = meta[S.COL_CATEGORY].astype(str)
    is_eq_all = cat.isin(S.EARTHQUAKE_VALUES).to_numpy()
    print(f"{args.split} split: {len(ds)} traces; scanning up to {args.scan} noise traces")

    session, name_in, name_out = make_session(Path(args.onnx), 1)

    def predict(x):
        return session.run([name_out], {name_in: x.astype(np.float32)})[0]

    def stream_probs(raw):
        pre = CausalPreprocessor(cfg, warmup_samples=min(int(fs), args.chunk_samples))
        return causal_stream_probabilities(
            raw, predict, pre, chunk_samples=args.chunk_samples
        ).probabilities

    existing_ids = set()
    if Path(args.bundle).is_file():
        existing = np.load(args.bundle, allow_pickle=True)
        existing_ids = set(str(t) for t in existing["trace_id"])

    found = None
    n_scanned = 0
    for i in range(len(ds)):
        if n_scanned >= args.scan or found is not None:
            break
        row = int(ds.rows[i])
        if bool(is_eq_all[row]):
            continue
        n_scanned += 1
        m = meta.iloc[row]
        trace_id = str(m_trace_id(m, S, row))
        if trace_id in existing_ids:
            continue

        raw = _get_waveform(ds.ds, row, n_ch, n_samples).astype(np.float32)
        ratio = _stalta(raw, fs)
        stalta_fires = bool((ratio >= STA_LTA_ON).any())
        if not stalta_fires:
            continue

        probs = stream_probs(raw)
        det_max = float(probs[0].max())
        model_quiet = det_max <= (DET_THRESHOLD - args.det_margin)
        print(f"  id={trace_id:>20s}  stalta_max={float(ratio.max()):5.2f}  "
              f"det_max={det_max:.2f}  {'FOUND' if (stalta_fires and model_quiet) else 'skip'}")
        if stalta_fires and model_quiet:
            found = {
                "trace_id": trace_id, "raw": raw, "is_eq": False,
                "p_sample": -1, "s_sample": -1, "snr_db": float("nan"),
            }

    if found is None:
        raise SystemExit(
            f"scanned {n_scanned} noise traces, found no STA/LTA false trigger "
            f"with model staying {args.det_margin} below threshold -- try --scan larger "
            f"or loosen --det-margin"
        )

    print(f"\nfound: {found['trace_id']} -- STA/LTA false-triggers, model stays quiet")

    if Path(args.bundle).is_file():
        existing = np.load(args.bundle, allow_pickle=True)
        raws = np.concatenate([existing["raw"], found["raw"][None]], axis=0).astype(np.float32)
        trace_id = np.array(list(existing["trace_id"]) + [found["trace_id"]])
        is_eq = np.array(list(existing["is_eq"]) + [found["is_eq"]])
        p_sample = np.array(list(existing["p_sample"]) + [found["p_sample"]], dtype=np.int64)
        s_sample = np.array(list(existing["s_sample"]) + [found["s_sample"]], dtype=np.int64)
        snr_db = np.array(list(existing["snr_db"]) + [found["snr_db"]], dtype=np.float64)
        sampling_rate = existing["sampling_rate"]
        window_samples = existing["window_samples"]
    else:
        raws = found["raw"][None].astype(np.float32)
        trace_id = np.array([found["trace_id"]])
        is_eq = np.array([found["is_eq"]])
        p_sample = np.array([found["p_sample"]], dtype=np.int64)
        s_sample = np.array([found["s_sample"]], dtype=np.int64)
        snr_db = np.array([found["snr_db"]], dtype=np.float64)
        sampling_rate = np.float64(fs)
        window_samples = np.int64(n_samples)

    np.savez_compressed(
        args.bundle,
        raw=raws, trace_id=trace_id, is_eq=is_eq,
        p_sample=p_sample, s_sample=s_sample, snr_db=snr_db,
        sampling_rate=sampling_rate, window_samples=window_samples,
    )
    print(f"appended to {args.bundle}: now {len(trace_id)} traces total, "
          f"{Path(args.bundle).stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()
