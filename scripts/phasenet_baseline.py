#!/usr/bin/env python
"""Baseline: pretrained PhaseNet (SeisBench 'stead') on the SAME test split.

PhaseNet is the other standard lightweight P/S phase picker (a 3-output U-Net,
~270k params — comparable in size to this project's student). It is NOT a
detector: its three softmax streams are P / S / Noise, so there is no dedicated
event-detection head. We derive an event-present probability as ``1 - Noise``
(equivalently P+S, since the outputs sum to 1) and score detection under the
same rule and tolerances as the student / EQTransformer evals.

Protocol matches scripts/fair_comparison.py (the repo's corrected protocol):

  * Native preprocessing. PhaseNet is fed its documented conditioning (demean +
    per-trace peak norm, no bandpass) via the model's own ``annotate_batch_pre``
    on the RAW waveform — NOT the student's demean+bandpass+std pipeline — so it
    is scored at its own best, exactly as the teacher is in fair_comparison.
  * Threshold selection on validation. The detection threshold is swept on the
    VAL split only, fixed at its val-max-F1 point, and applied ONCE to TEST.
    We also report the a-priori fixed-0.5 operating point for reference.
  * P/S picks use the shared fixed peak_height (0.3) and within-tolerance MAE
    from ``eval:`` in the config (imported from evaluate.py so they can't drift).

Outputs (outputs/phasenet_baseline/):
  pn_metrics.json   full metrics: fixed-0.5 + val-selected, overall + per bucket
  summary.txt       human-readable summary with caveats
  pick_residuals.png  P/S residual histograms (same style as the EQT baseline)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from seismic_edge_picker.config import load_config  # noqa: E402
from seismic_edge_picker.dataset import build_datasets, _get_waveform  # noqa: E402
from seismic_edge_picker import splits as S  # noqa: E402
from evaluate import pick_from_stream, prf, residual_stats, snr_bucket_label  # noqa: E402

THR_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)


def det_peak(stream, distance):
    """Max interior local-peak height (reproduces detect_event for all thr)."""
    peaks, _ = find_peaks(stream, distance=distance, plateau_size=1)
    return float(stream[peaks].max()) if len(peaks) else float("-inf")


def collect(rows, ds, model, dev, cfg):
    """Per-trace scalar records for one split under PhaseNet native preprocessing."""
    fs = cfg.data.sampling_rate
    dist = max(1, int(round(cfg.eval.peak_min_distance_s * fs)))
    ph = cfg.eval.peak_height
    n_ch, n_samp = cfg.data.n_channels, cfg.data.window_samples
    meta = ds.metadata
    cat = meta[S.COL_CATEGORY].astype(str)
    is_eq = cat.isin(S.EARTHQUAKE_VALUES).to_numpy()

    recs = []
    B = 128
    for start in range(0, len(rows), B):
        chunk = rows[start:start + B]
        raw = torch.tensor(
            np.stack([_get_waveform(ds.ds, int(ds.rows[li]), n_ch, n_samp)
                      for li in chunk]), dtype=torch.float32, device=dev)
        with torch.inference_mode():
            out = model(model.annotate_batch_pre(raw, {}))  # (B, 3, samp) = P,S,N
        out = out.cpu().numpy()
        for b, li in enumerate(chunk):
            row = int(ds.rows[li])
            m = meta.iloc[row]
            p_stream, s_stream, n_stream = out[b, 0], out[b, 1], out[b, 2]
            det_stream = 1.0 - n_stream  # event-present proxy (== P+S)
            p_pick = pick_from_stream(p_stream, ph, dist)[0]
            s_pick = pick_from_stream(s_stream, ph, dist)[0]
            recs.append({
                "is_eq": bool(is_eq[row]),
                "snr": S.parse_snr_db(m.get(S.COL_SNR)),
                "gt_p": S.parse_scalar(m.get(S.COL_P)),
                "gt_s": S.parse_scalar(m.get(S.COL_S)),
                "det_peak": det_peak(det_stream, dist),
                "p_pick": p_pick, "s_pick": s_pick,
            })
        print(f"  {min(start + B, len(rows))}/{len(rows)}", end="\r", flush=True)
    print()
    return recs


def f1_at(recs, thr):
    tp = fp = fn = tn = 0
    for r in recs:
        det = r["det_peak"] >= thr
        if r["is_eq"]:
            tp += det; fn += not det
        else:
            fp += det; tn += not det
    precision, recall, f1 = prf(tp, fp, fn)
    return {"threshold": float(thr), "precision": precision, "recall": recall,
            "f1": f1, "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def best_threshold(val_recs):
    sweep = [f1_at(val_recs, float(t)) for t in THR_GRID]
    best = max(sweep, key=lambda d: (d["f1"], d["recall"]))
    return best["threshold"], best


def pick_metrics(recs, phase, cfg):
    """Pick rate + within-tolerance MAE/std for one phase; threshold-independent."""
    fs = cfg.data.sampling_rate
    tol_ms = cfg.eval.match_tolerance_s * 1000.0
    gt_key, pick_key = f"gt_{phase}", f"{phase}_pick"
    with_gt = [r for r in recs if r["is_eq"] and np.isfinite(r[gt_key])]
    picked = [r for r in with_gt if r[pick_key] is not None]
    res = [(r[pick_key] - r[gt_key]) / fs * 1000.0 for r in picked]
    matched = [x for x in res if abs(x) <= tol_ms]
    return {
        "n_ground_truth": len(with_gt),
        "n_picked": len(picked),
        "pick_rate": len(picked) / len(with_gt) if with_gt else None,
        "n_within_tolerance": len(matched),
        "hit_rate_within_tol": len(matched) / len(with_gt) if with_gt else None,
        "residuals_within_tol": residual_stats(matched),
        "residuals_all_picks": residual_stats(res),
    }


def overall(recs, thr, cfg):
    return {
        "detection": f1_at(recs, thr),
        "p_picks": pick_metrics(recs, "p", cfg),
        "s_picks": pick_metrics(recs, "s", cfg),
    }


def parse_args():
    p = argparse.ArgumentParser(description="PhaseNet baseline on the test split.")
    p.add_argument("--config", default=str(REPO / "configs" / "default.yaml"))
    p.add_argument("--pretrained", default="stead")
    p.add_argument("--out", default=str(REPO / "outputs" / "phasenet_baseline"))
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    fs = cfg.data.sampling_rate
    edges = list(cfg.eval.snr_buckets)

    import seisbench.models as sbm
    model = sbm.PhaseNet.from_pretrained(args.pretrained).to(dev).eval()
    n_params = sum(p.numel() for p in model.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    assert model.sampling_rate == fs, f"PhaseNet is {model.sampling_rate}Hz, cfg {fs}Hz"
    print(f"PhaseNet('{args.pretrained}')  params={n_params:,}  "
          f"fp32 size={param_bytes/1e6:.2f} MB  device={dev}")
    print(f"native conditioning: demean + {model.norm}-norm "
          f"(norm_detrend={model.norm_detrend}); labels={model.labels}; "
          f"detection proxy = 1 - Noise")

    datasets, _ = build_datasets(cfg)
    val = list(range(len(datasets["val"])))
    test = list(range(len(datasets["test"])))
    print(f"val={len(val)}  test={len(test)}")

    print("collecting val records...")
    t0 = time.perf_counter()
    val_recs = collect(val, datasets["val"], model, dev, cfg)
    print("collecting test records...")
    test_recs = collect(test, datasets["test"], model, dev, cfg)
    infer_s = time.perf_counter() - t0

    thr, val_best = best_threshold(val_recs)
    print(f"val-selected detection threshold = {thr:.2f} (val F1 {val_best['f1']:.4f})")

    fixed = overall(test_recs, 0.5, cfg)
    selected = overall(test_recs, thr, cfg)

    # per-SNR-bucket at the val-selected threshold
    buckets = {}
    for r in test_recs:
        buckets.setdefault(snr_bucket_label(r["snr"], edges), []).append(r)
    per_bucket = {
        k: {"n": len(v), **overall(v, thr, cfg)}
        for k, v in sorted(buckets.items(),
                           key=lambda kv: (kv[0] == "noise/unknown", kv[0]))
    }

    n_eq = sum(r["is_eq"] for r in test_recs)
    results = {
        "model": f"PhaseNet[{args.pretrained}]",
        "params": n_params,
        "fp32_size_mb": param_bytes / 1e6,
        "split": "test",
        "n_traces": len(test_recs),
        "n_earthquake": int(n_eq),
        "n_noise": int(len(test_recs) - n_eq),
        "device": str(dev),
        "inference_seconds": infer_s,
        "input_pipeline": "PhaseNet native preprocessing (demean + peak norm, "
                          "no bandpass) via annotate_batch_pre on the raw waveform",
        "detection_note": "PhaseNet has no detection head; event-present proxy = "
                          "1 - Noise stream (== P+S), scored with the shared rule",
        "eval_params": {
            "peak_height": cfg.eval.peak_height,
            "peak_min_distance_s": cfg.eval.peak_min_distance_s,
            "match_tolerance_s": cfg.eval.match_tolerance_s,
            "snr_buckets_db": edges,
        },
        "val_selected_threshold": thr,
        "val_best": val_best,
        "overall_fixed_0.5": fixed,
        "overall_val_selected": selected,
        "by_snr_bucket": per_bucket,
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pn_metrics.json").write_text(json.dumps(results, indent=2))

    # ---- residual histograms (same style as eqtransformer_baseline) --------
    tol_ms = cfg.eval.match_tolerance_s * 1000.0
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, phase, color in ((axes[0], "p", "C0"), (axes[1], "s", "C1")):
        gt_key, pick_key = f"gt_{phase}", f"{phase}_pick"
        res = [(r[pick_key] - r[gt_key]) / fs * 1000.0 for r in test_recs
               if r["is_eq"] and np.isfinite(r[gt_key]) and r[pick_key] is not None]
        res = [x for x in res if abs(x) <= tol_ms]
        ax.hist(res, bins=50, range=(-tol_ms, tol_ms), color=color)
        st = residual_stats(res)
        ax.set_title(f"{phase.upper()} residuals (n={st['n']}, "
                     f"MAE {st['mae_ms']:.1f} ms, std {st['std_ms']:.1f} ms)")
        ax.set_xlabel("pick − ground truth (ms)"); ax.grid(alpha=0.25)
    axes[0].set_ylabel("count")
    fig.suptitle(f"PhaseNet[{args.pretrained}] pick residuals within ±{tol_ms:g} ms "
                 "(native preprocessing)")
    fig.tight_layout()
    fig.savefig(out_dir / "pick_residuals.png", dpi=120)
    plt.close(fig)

    # ---- summary.txt -------------------------------------------------------
    d = selected["detection"]
    lines = [
        f"PhaseNet[{args.pretrained}] baseline on split 'test' "
        f"({len(test_recs)} traces: {n_eq} eq / {len(test_recs) - n_eq} noise)",
        f"params={n_params:,}  fp32 size={param_bytes/1e6:.2f} MB  device={dev}",
        f"native preprocessing (demean + peak norm, no bandpass); "
        f"detection proxy = 1 - Noise stream",
        f"detection @ val-selected thr {thr:.2f}: P={d['precision']:.4f} "
        f"R={d['recall']:.4f} F1={d['f1']:.4f} "
        f"(TP={d['tp']} FP={d['fp']} FN={d['fn']} TN={d['tn']})",
    ]
    d0 = fixed["detection"]
    lines.append(f"detection @ fixed 0.50:        P={d0['precision']:.4f} "
                 f"R={d0['recall']:.4f} F1={d0['f1']:.4f}")
    for phase in ("p", "s"):
        pk = selected[f"{phase}_picks"]
        rt = pk["residuals_within_tol"]
        lines.append(
            f"{phase.upper()} picks: {pk['n_picked']}/{pk['n_ground_truth']} picked "
            f"({pk['pick_rate']:.3f}), {pk['n_within_tolerance']} within ±{tol_ms:g} ms "
            f"(hit rate {pk['hit_rate_within_tol']:.3f}); within-tol "
            f"MAE={rt['mae_ms']:.1f} ms std={rt['std_ms']:.1f} ms")
    lines.append("NOTE: PhaseNet fed its NATIVE preprocessing (its own best), matching "
                 "the corrected protocol used for the EQTransformer teacher.")
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nwrote artifacts to {out_dir}/")


if __name__ == "__main__":
    main()
