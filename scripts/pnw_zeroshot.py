#!/usr/bin/env python
"""Zero-shot cross-dataset test: the STEAD-trained student on the PNW dataset.

NO training / fine-tuning. We load the existing distilled student
(checkpoints/stage2_distill/best.pt) and run it, unchanged, on a subset of the
Pacific Northwest (PNW) benchmark — a different network, region, and instrument
mix than STEAD — to measure out-of-distribution generalization.

Data access is deliberately lightweight: PNW's event waveforms are 67 GB and
PNWNoise 18.5 GB, so instead of downloading them we stream ONLY the selected
traces directly from the SeisBench remote HDF5 via fsspec + h5py range reads
(the metadata CSVs, ~58 MB total, are cached locally). Disk stays < ~1 GB.

Key PNW facts handled here (verified against the metadata + a live read):
  * 100 Hz, 3-C, 15,001-sample (150 s) traces; every event has P and S picks.
  * Component order is ENZ (PNW) vs ZNE (what the STEAD-trained model expects),
    so channels are reordered E,N,Z -> Z,N,E before preprocessing.
  * source_type is earthquake / explosion (both are positive "events" with
    picks); noise comes from the separate PNWNoise set for detection negatives.
  * trace_snr_db is pipe-separated per component ("a|b|c"); reduced to max.

Each 150 s trace is cropped to a 6,000-sample (60 s) window placed so the P
arrival lands at ~sample 600 — matching STEAD's training distribution (median P
sample ~601) — with the window clamped to the trace and the ground-truth P/S
sample indices recomputed in window coordinates. Preprocessing (demean +
bandpass 1-45 Hz + per-trace std norm), the detection rule, the pick peak
finder, and the ±500 ms tolerances are IDENTICAL to the STEAD eval (imported
from evaluate.py) so STEAD vs PNW is apples-to-apples.

Outputs (outputs/pnw_zeroshot/): pnw_metrics.json, summary.txt, pick_residuals.png,
and (with --plot-examples) a few overlay_*.png sanity panels.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from seismic_edge_picker.config import load_config  # noqa: E402
from seismic_edge_picker.model import build_model, count_parameters  # noqa: E402
from seismic_edge_picker.preprocessing import preprocess_waveform  # noqa: E402
from evaluate import (  # noqa: E402
    pick_from_stream, detect_event, prf, residual_stats, snr_bucket_label,
)

REMOTE = "https://hifis-storage.desy.de/Helmholtz/HelmholtzAI/SeisBench/datasets"
TRACE_LEN = 15001            # PNW native trace length (samples)
EVENT_TYPES = ("earthquake", "explosion")


def parse_trace_name(name: str):
    """'bucket4$0,:3,:15001' -> ('bucket4', 0)."""
    bucket, rest = name.split("$", 1)
    idx = int(rest.split(",", 1)[0])
    return bucket, idx


def parse_snr(value):
    """PNW per-component 'a|b|c' (may contain 'nan') -> max finite, else nan."""
    if not isinstance(value, str):
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")
    vals = []
    for tok in value.split("|"):
        try:
            v = float(tok)
            if np.isfinite(v):
                vals.append(v)
        except ValueError:
            pass
    return max(vals) if vals else float("nan")


def cache_metadata(name: str, cache_dir: Path) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    local = cache_dir / f"{name}_metadata.csv"
    if not local.exists():
        import urllib.request
        url = f"{REMOTE}/{name}/metadata.csv"
        print(f"  downloading {name} metadata -> {local} ...")
        urllib.request.urlretrieve(url, local)
    return pd.read_csv(local, low_memory=False)


def open_remote(name: str):
    import fsspec
    import h5py
    url = f"{REMOTE}/{name}/waveforms.hdf5"
    f = fsspec.open(url, block_size=2 ** 20).open()
    return h5py.File(f, "r")


def read_waveforms(h5, names, want_len=TRACE_LEN):
    """Stream the given trace_names from an open remote HDF5, grouped by bucket.

    Returns a dict trace_name -> (3, want_len) float32 array in the FILE's native
    component order (ENZ for PNW). One indexed read per bucket keeps round-trips
    down. h5py list-indexing needs strictly-increasing unique indices per bucket.
    """
    data = h5["data"]
    by_bucket = {}
    for nm in names:
        b, i = parse_trace_name(nm)
        by_bucket.setdefault(b, []).append((i, nm))
    out = {}
    n_done = 0
    for b, items in by_bucket.items():
        idx_to_names = {}
        for i, nm in items:
            idx_to_names.setdefault(i, []).append(nm)
        uniq = sorted(idx_to_names)
        block = data[b][uniq, :3, :want_len]  # (len(uniq), 3, want_len)
        for k, i in enumerate(uniq):
            arr = np.asarray(block[k], dtype=np.float32)
            for nm in idx_to_names[i]:
                out[nm] = arr
        n_done += len(uniq)
        print(f"    read {n_done}/{len(names)} traces "
              f"({b}: {len(uniq)})", end="\r", flush=True)
    print()
    return out


def enz_to_zne(wf):
    """Reorder PNW E,N,Z channels -> Z,N,E for the STEAD-trained model."""
    return wf[[2, 1, 0], :]


def window_event(wf_zne, p_samp, s_samp, p_offset, win):
    """Crop a `win`-sample window placing P at ~p_offset; recompute picks.

    Returns (window(3,win), gt_p_in_win or None, gt_s_in_win or None).
    """
    start = int(round(p_samp - p_offset))
    start = max(0, min(start, wf_zne.shape[1] - win))
    seg = wf_zne[:, start:start + win]
    gp = p_samp - start
    gs = s_samp - start
    gp = gp if 0 <= gp < win else None
    gs = gs if 0 <= gs < win else None
    return seg, gp, gs


def parse_args():
    p = argparse.ArgumentParser(description="Zero-shot student eval on PNW.")
    p.add_argument("--n-events", type=int, default=5000)
    p.add_argument("--n-noise", type=int, default=2500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--p-offset", type=int, default=600,
                   help="target P position in the window (STEAD median ~601)")
    p.add_argument("--smoke", action="store_true",
                   help="tiny run (64 events / 32 noise) to validate the pipeline")
    p.add_argument("--plot-examples", type=int, default=0,
                   help="save N overlay sanity panels")
    p.add_argument("--out", default=str(REPO / "outputs" / "pnw_zeroshot"))
    p.add_argument("--cache", default=str(REPO / "data" / "pnw_cache"))
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.n_events, args.n_noise = 64, 32
        if args.plot_examples == 0:
            args.plot_examples = 4
    cfg = load_config(str(REPO / "configs" / "default.yaml"))
    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    fs = cfg.data.sampling_rate
    win = cfg.data.window_samples
    ev = cfg.eval
    peak_dist = max(1, int(round(ev.peak_min_distance_s * fs)))
    tol_ms = ev.match_tolerance_s * 1000.0
    edges = list(ev.snr_buckets)

    # ---- student ----------------------------------------------------------
    ckpt = torch.load(REPO / "checkpoints/stage2_distill/best.pt",
                      map_location="cpu", weights_only=True)
    model = build_model(cfg).to(dev).eval()
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"student: stage2_distill/best.pt epoch={ckpt.get('epoch')} "
          f"params={count_parameters(model):,} device={dev}")

    # ---- select subset (reproducible) -------------------------------------
    cache = Path(args.cache)
    print("metadata:")
    ev_meta = cache_metadata("pnw", cache)
    no_meta = cache_metadata("pnwnoise", cache)
    ev_pool = ev_meta[ev_meta["source_type"].isin(EVENT_TYPES)].copy()
    rng = np.random.default_rng(args.seed)
    ev_sel = ev_pool.iloc[rng.choice(len(ev_pool), min(args.n_events, len(ev_pool)),
                                     replace=False)].reset_index(drop=True)
    no_sel = no_meta.iloc[rng.choice(len(no_meta), min(args.n_noise, len(no_meta)),
                                     replace=False)].reset_index(drop=True)
    print(f"selected {len(ev_sel)} events (of {len(ev_pool)}) + "
          f"{len(no_sel)} noise (of {len(no_meta)})")

    # ---- stream waveforms -------------------------------------------------
    print("streaming event waveforms (remote):")
    t0 = time.perf_counter()
    ev_h5 = open_remote("pnw")
    ev_wf = read_waveforms(ev_h5, ev_sel["trace_name"].tolist())
    print("streaming noise waveforms (remote):")
    no_h5 = open_remote("pnwnoise")
    no_wf = read_waveforms(no_h5, no_sel["trace_name"].tolist())
    stream_s = time.perf_counter() - t0
    print(f"streamed {len(ev_wf)+len(no_wf)} traces in {stream_s:.0f}s")

    # ---- build windows + run model ----------------------------------------
    recs = []
    batch_x, batch_meta = [], []
    examples = []

    def flush():
        if not batch_x:
            return
        x = torch.from_numpy(np.stack(batch_x)).to(dev)
        with torch.inference_mode():
            out = model(x).cpu().numpy()
        for k, meta in enumerate(batch_meta):
            det, pstr, sstr = out[k]
            p_pick = pick_from_stream(pstr, ev.peak_height, peak_dist)[0]
            s_pick = pick_from_stream(sstr, ev.peak_height, peak_dist)[0]
            rec = {**meta,
                   "detected": detect_event(det, ev.detection_threshold, peak_dist),
                   "det_max": float(det.max()),
                   "p_pick": p_pick, "s_pick": s_pick}
            recs.append(rec)
            if len(examples) < args.plot_examples and meta["is_eq"]:
                examples.append((batch_x[k].copy(), det, pstr, sstr, rec))
        batch_x.clear(); batch_meta.clear()

    # events
    for _, row in ev_sel.iterrows():
        wf = ev_wf.get(row["trace_name"])
        if wf is None:
            continue
        wf = enz_to_zne(wf)
        p_samp = float(row["trace_P_arrival_sample"])
        s_samp = float(row["trace_S_arrival_sample"])
        if not (np.isfinite(p_samp) and np.isfinite(s_samp)):
            continue
        seg, gp, gs = window_event(wf, p_samp, s_samp, args.p_offset, win)
        xp = preprocess_waveform(seg, cfg)
        batch_x.append(xp)
        batch_meta.append({"is_eq": True, "snr": parse_snr(row.get("trace_snr_db")),
                           "src": row["source_type"], "gt_p": gp, "gt_s": gs})
        if len(batch_x) >= 128:
            flush()
    flush()
    # noise
    for _, row in no_sel.iterrows():
        wf = no_wf.get(row["trace_name"])
        if wf is None:
            continue
        wf = enz_to_zne(wf)
        seg = wf[:, :win]
        xp = preprocess_waveform(seg, cfg)
        batch_x.append(xp)
        batch_meta.append({"is_eq": False, "snr": float("nan"),
                           "src": "noise", "gt_p": None, "gt_s": None})
        if len(batch_x) >= 128:
            flush()
    flush()
    n_eq = sum(r["is_eq"] for r in recs)
    print(f"scored {len(recs)} traces ({n_eq} events / {len(recs)-n_eq} noise)")

    # ---- metrics ----------------------------------------------------------
    def detection_at(thr):
        tp = sum(1 for r in recs if r["is_eq"] and r["det_max"] >= thr)
        fn = sum(1 for r in recs if r["is_eq"] and r["det_max"] < thr)
        fp = sum(1 for r in recs if not r["is_eq"] and r["det_max"] >= thr)
        tn = sum(1 for r in recs if not r["is_eq"] and r["det_max"] < thr)
        precision, recall, f1 = prf(tp, fp, fn)
        return {"threshold": thr, "precision": precision, "recall": recall,
                "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}

    def picks(subset, phase):
        gt_key, pk_key = f"gt_{phase}", f"{phase}_pick"
        with_gt = [r for r in subset if r["is_eq"] and r[gt_key] is not None]
        picked = [r for r in with_gt if r[pk_key] is not None]
        res = [(r[pk_key] - r[gt_key]) / fs * 1000.0 for r in picked]
        matched = [x for x in res if abs(x) <= tol_ms]
        return {"n_ground_truth": len(with_gt), "n_picked": len(picked),
                "pick_rate": len(picked) / len(with_gt) if with_gt else None,
                "n_within_tolerance": len(matched),
                "hit_rate_within_tol": len(matched) / len(with_gt) if with_gt else None,
                "residuals_within_tol": residual_stats(matched),
                "residuals_all_picks": residual_stats(res)}

    # deployed STEAD operating point (val-selected student threshold) for a fair
    # "would you re-tune on the new dataset? no." comparison, plus fixed 0.5.
    try:
        fair = json.loads((REPO / "outputs/fair_eval/comparison.json").read_text())
        stead_thr = fair["val_selected_thresholds"]["student"]
    except Exception:
        stead_thr = 0.89

    # cheap sweep for the PNW-best operating point (informational only)
    sweep = [detection_at(round(t, 2)) for t in np.arange(0.05, 0.951, 0.02)]
    best = max(sweep, key=lambda d: (d["f1"], d["recall"]))

    buckets = {}
    for r in recs:
        buckets.setdefault(snr_bucket_label(r["snr"], edges), []).append(r)
    by_bucket = {k: {"n": len(v), "p": picks(v, "p"), "s": picks(v, "s")}
                 for k, v in sorted(buckets.items())}

    results = {
        "model": "SeismicUNet student (STEAD-trained), zero-shot",
        "eval_dataset": "PNW (events: earthquake+explosion) + PNWNoise",
        "params": count_parameters(model),
        "n_traces": len(recs), "n_events": n_eq, "n_noise": len(recs) - n_eq,
        "subset": {"n_events": args.n_events, "n_noise": args.n_noise,
                   "seed": args.seed, "p_offset": args.p_offset},
        "device": str(dev), "stream_seconds": stream_s,
        "input_pipeline": "ENZ->ZNE reorder; 150s->60s window (P at ~p_offset); "
                          "demean+bandpass 1-45Hz+std norm (identical to STEAD eval)",
        "eval_params": {"peak_height": ev.peak_height,
                        "peak_min_distance_s": ev.peak_min_distance_s,
                        "match_tolerance_s": ev.match_tolerance_s,
                        "snr_buckets_db": edges},
        "detection_fixed_0.5": detection_at(0.5),
        "detection_at_stead_deployed_thr": detection_at(stead_thr),
        "detection_pnw_best": best,
        "p_picks": picks(recs, "p"),
        "s_picks": picks(recs, "s"),
        "by_snr_bucket": by_bucket,
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pnw_metrics.json").write_text(json.dumps(results, indent=2))

    # ---- residual histograms ----------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, phase, color in ((axes[0], "p", "C0"), (axes[1], "s", "C1")):
        gt_key, pk_key = f"gt_{phase}", f"{phase}_pick"
        res = [(r[pk_key] - r[gt_key]) / fs * 1000.0 for r in recs
               if r["is_eq"] and r[gt_key] is not None and r[pk_key] is not None]
        res = [x for x in res if abs(x) <= tol_ms]
        ax.hist(res, bins=50, range=(-tol_ms, tol_ms), color=color)
        st = residual_stats(res)
        ax.set_title(f"{phase.upper()} residuals (n={st['n']}, "
                     f"MAE {st['mae_ms']:.1f} ms, std {st['std_ms']:.1f} ms)")
        ax.set_xlabel("pick − ground truth (ms)"); ax.grid(alpha=0.25)
    axes[0].set_ylabel("count")
    fig.suptitle(f"Student zero-shot on PNW — pick residuals within ±{tol_ms:g} ms")
    fig.tight_layout()
    fig.savefig(out_dir / "pick_residuals.png", dpi=120)
    plt.close(fig)

    # ---- optional overlay sanity panels -----------------------------------
    for n, (xp, det, pstr, sstr, rec) in enumerate(examples, 1):
        t = np.arange(win) / fs
        fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
        for ch, name in enumerate(("Z", "N", "E")):
            ax[0].plot(t, xp[ch] + ch * 6, lw=0.4, color="C0")
        ax[0].set_title(f"PNW {rec['src']} SNR {rec['snr']:.1f} dB "
                        f"(preprocessed, ZNE stacked)")
        ax[1].plot(t, det, label="detection", color="0.3")
        ax[1].plot(t, pstr, label="P", color="C0")
        ax[1].plot(t, sstr, label="S", color="C1")
        for gt, c, lab in ((rec["gt_p"], "C0", "GT P"), (rec["gt_s"], "C1", "GT S")):
            if gt is not None:
                ax[1].axvline(gt / fs, color=c, ls="--", lw=1, label=lab)
        for pk, c, lab in ((rec["p_pick"], "C0", "P̂"), (rec["s_pick"], "C1", "Ŝ")):
            if pk is not None:
                ax[1].axvline(pk / fs, color=c, ls=":", lw=1.5, label=lab)
        ax[1].set_ylim(0, 1.05); ax[1].legend(ncol=4, fontsize=8)
        ax[1].set_xlabel("time (s)")
        fig.tight_layout()
        fig.savefig(out_dir / f"overlay_{n}.png", dpi=110)
        plt.close(fig)

    # ---- summary ----------------------------------------------------------
    d5 = results["detection_fixed_0.5"]; dd = results["detection_at_stead_deployed_thr"]
    pp = results["p_picks"]["residuals_within_tol"]; ss = results["s_picks"]["residuals_within_tol"]
    lines = [
        f"Student (STEAD-trained) ZERO-SHOT on PNW: {len(recs)} traces "
        f"({n_eq} events / {len(recs)-n_eq} noise), seed {args.seed}",
        f"detection @ fixed 0.50:            P={d5['precision']:.4f} R={d5['recall']:.4f} "
        f"F1={d5['f1']:.4f} (TP={d5['tp']} FP={d5['fp']} FN={d5['fn']} TN={d5['tn']})",
        f"detection @ STEAD-deployed {stead_thr:.2f}:   P={dd['precision']:.4f} "
        f"R={dd['recall']:.4f} F1={dd['f1']:.4f}",
        f"detection @ PNW-best {best['threshold']:.2f} (info): F1={best['f1']:.4f}",
        f"P picks: rate {results['p_picks']['pick_rate']:.3f}, hit {results['p_picks']['hit_rate_within_tol']:.3f}, "
        f"within-tol MAE={pp['mae_ms']:.1f} ms std={pp['std_ms']:.1f} ms",
        f"S picks: rate {results['s_picks']['pick_rate']:.3f}, hit {results['s_picks']['hit_rate_within_tol']:.3f}, "
        f"within-tol MAE={ss['mae_ms']:.1f} ms std={ss['std_ms']:.1f} ms",
        "reference (STEAD in-distribution): F1 0.9925, P-MAE 36.9 ms, S-MAE 71.4 ms",
    ]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nwrote artifacts to {out_dir}/")


if __name__ == "__main__":
    main()
