#!/usr/bin/env python
"""Phase-4 baseline: pretrained EQTransformer (teacher) on the SAME test split.

NOT training / distillation / export. Loads SeisBench's pretrained EQTransformer
('stead' weights) and evaluates it on the identical STEAD test traces used for
the Stage-1 student eval, with the IDENTICAL detection / pick-matching tolerances
from ``eval:`` in the config (imported from scripts/evaluate.py so they cannot
drift). Produces the same metric set for a fair side-by-side.

Fairness note: both models are fed byte-identical inputs — the project pipeline's
demean + bandpass(1-45 Hz) + per-trace std normalization (the student's test
dataset). This differs from EQTransformer's native preprocessing, so EQT numbers
here are "EQT on this project's pipeline", not EQT at its published best. See the
caveats printed at the end and in the docs.

Usage:
    python scripts/eqtransformer_baseline.py --config configs/default.yaml \
        --out outputs/eqtransformer_baseline
"""

from __future__ import annotations

import argparse
import csv
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
from torch.utils.data import DataLoader  # noqa: E402

SCRIPTS_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(SCRIPTS_DIR, "..", "src"))
sys.path.insert(0, SCRIPTS_DIR)

from seismic_edge_picker.config import load_config  # noqa: E402
from seismic_edge_picker.dataset import build_datasets  # noqa: E402
from seismic_edge_picker import splits as S  # noqa: E402
# reuse the EXACT helper logic (tolerances, peak finding) from the student eval
from evaluate import (  # noqa: E402
    pick_from_stream, detect_event, prf, residual_stats, snr_bucket_label,
)


def parse_args():
    p = argparse.ArgumentParser(description="EQTransformer baseline on the test split.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--pretrained", default="stead")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--out", default="outputs/eqtransformer_baseline")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--sweep", action="store_true", default=True,
                   help="also sweep detection thresholds (cheap)")
    return p.parse_args()


def summarize(records, fs, tol_samples, edges=None):
    """Identical detection / pick metric logic to evaluate.py's inner summarize."""
    recs = records
    tp = sum(1 for r in recs if r["is_eq"] and r["detected"])
    fn = sum(1 for r in recs if r["is_eq"] and not r["detected"])
    fp = sum(1 for r in recs if not r["is_eq"] and r["detected"])
    tn = sum(1 for r in recs if not r["is_eq"] and not r["detected"])
    precision, recall, f1 = prf(tp, fp, fn)
    out = {
        "n_traces": len(recs),
        "n_earthquake": tp + fn,
        "n_noise": fp + tn,
        "detection": {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                      "precision": precision, "recall": recall, "f1": f1},
    }
    tol_ms = tol_samples / fs * 1000.0
    for phase, gt_key, pick_key in (("p", "gt_p", "p_pick"), ("s", "gt_s", "s_pick")):
        with_gt = [r for r in recs if r["is_eq"] and np.isfinite(r[gt_key])]
        picked = [r for r in with_gt if r[pick_key] is not None]
        res = [(r[pick_key] - r[gt_key]) / fs * 1000.0 for r in picked]
        matched = [x for x in res if abs(x) <= tol_ms]
        out[f"{phase}_picks"] = {
            "n_ground_truth": len(with_gt),
            "n_picked": len(picked),
            "pick_rate": len(picked) / len(with_gt) if with_gt else None,
            "n_within_tolerance": len(matched),
            "hit_rate_within_tol": len(matched) / len(with_gt) if with_gt else None,
            "residuals_all_picks": residual_stats(res),
            "residuals_within_tol": residual_stats(matched),
        }
    return out


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    fs = cfg.data.sampling_rate
    ev = cfg.eval
    peak_distance = max(1, int(round(ev.peak_min_distance_s * fs)))
    tol_samples = ev.match_tolerance_s * fs
    edges = list(ev.snr_buckets)

    import seisbench.models as sbm
    model = sbm.EQTransformer.from_pretrained(args.pretrained)
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"EQTransformer('{args.pretrained}')  params={n_params:,}  "
          f"fp32 size={param_bytes/1e6:.2f} MB  device={device}")
    assert model.sampling_rate == fs and model.in_samples == cfg.data.window_samples, (
        f"EQT expects {model.sampling_rate}Hz/{model.in_samples} samples; "
        f"config is {fs}Hz/{cfg.data.window_samples}"
    )

    datasets, _ = build_datasets(cfg)
    ds = datasets[args.split]
    meta = ds.metadata
    rows = ds.rows
    cat = meta[S.COL_CATEGORY].astype(str)
    is_eq_all = cat.isin(S.EARTHQUAKE_VALUES).to_numpy()
    print(f"split '{args.split}': {len(ds)} traces (identical to student eval)")

    batch_size = args.batch_size or cfg.train.batch_size
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=cfg.train.num_workers,
                        pin_memory=device.type == "cuda")

    thresholds = np.round(np.arange(0.10, 0.9001, 0.05), 2)
    n = len(ds)
    is_eq = np.zeros(n, dtype=bool)
    runs = np.zeros((n, len(thresholds)), dtype=np.int32)  # for the sweep

    records = []
    offset = 0
    t0 = time.perf_counter()
    with torch.inference_mode():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            out = model(x)  # tuple (Detection, P, S), each (B, 6000)
            det = out[0].detach().cpu().numpy()
            p_s = out[1].detach().cpu().numpy()
            s_s = out[2].detach().cpu().numpy()
            for b in range(det.shape[0]):
                row_idx = int(rows[offset + b])
                m = meta.iloc[row_idx]
                gt_p = S.parse_scalar(m.get(S.COL_P))
                gt_s = S.parse_scalar(m.get(S.COL_S))
                snr = S.parse_snr_db(m.get(S.COL_SNR))
                eq = bool(is_eq_all[row_idx])
                is_eq[offset + b] = eq
                p_pick, p_prob = pick_from_stream(p_s[b], ev.peak_height, peak_distance)
                s_pick, s_prob = pick_from_stream(s_s[b], ev.peak_height, peak_distance)
                records.append({
                    "row": row_idx, "is_eq": eq, "snr_db": snr,
                    "detected": detect_event(det[b], ev.detection_threshold, peak_distance),
                    "det_max": float(det[b].max()),
                    "gt_p": gt_p, "gt_s": gt_s,
                    "p_pick": p_pick, "s_pick": s_pick,
                })
                # sweep cache: longest run above each threshold
                stream = det[b]
                for ti, thr in enumerate(thresholds):
                    mask = stream >= float(thr)
                    if mask.any():
                        idx = np.flatnonzero(
                            np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
                        runs[offset + b, ti] = int(np.max(idx[1::2] - idx[0::2]))
            offset += det.shape[0]
    infer_s = time.perf_counter() - t0
    tput = n / infer_s
    print(f"inference in {infer_s:.1f}s  ({tput:.0f} traces/s on {device})")

    global_summary = summarize(records, fs, tol_samples)
    buckets = {}
    for r in records:
        buckets.setdefault(snr_bucket_label(r["snr_db"], edges), []).append(r)
    bucket_summaries = {
        k: summarize(v, fs, tol_samples)
        for k, v in sorted(buckets.items(),
                           key=lambda kv: (kv[0] == "noise/unknown", kv[0]))
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "model": f"EQTransformer[{args.pretrained}]",
        "params": n_params,
        "fp32_size_mb": param_bytes / 1e6,
        "split": args.split,
        "n_traces": n,
        "device": str(device),
        "inference_seconds": infer_s,
        "throughput_traces_per_s": tput,
        "input_pipeline": "project preprocessing (demean+bandpass 1-45Hz+std norm), "
                          "identical tensors to the student eval",
        "eval_params": {
            "detection_threshold": ev.detection_threshold,
            "peak_height": ev.peak_height,
            "peak_min_distance_s": ev.peak_min_distance_s,
            "match_tolerance_s": ev.match_tolerance_s,
            "snr_buckets_db": edges,
        },
        "overall": global_summary,
        "by_snr_bucket": bucket_summaries,
    }

    # optional threshold sweep (cheap, from the cached run lengths)
    if args.sweep:
        eq_mask, noise_mask = is_eq, ~is_eq
        sweep = []
        for ti, thr in enumerate(thresholds):
            detected = runs[:, ti] >= peak_distance  # >=1 sample really; use 1
            # NOTE: detect_event used peak height; for the sweep we use a simple
            # "any sample above threshold" rule (min_duration=1 sample) so the
            # curve is monotone and comparable to threshold_sweep.py's baseline.
            detected = runs[:, ti] >= 1
            tp = int(np.sum(detected & eq_mask)); fn = int(np.sum(~detected & eq_mask))
            fp = int(np.sum(detected & noise_mask)); tn = int(np.sum(~detected & noise_mask))
            precision, recall, f1 = prf(tp, fp, fn)
            sweep.append({"threshold": float(thr), "precision": precision,
                          "recall": recall, "f1": f1, "fp_noise": fp,
                          "fn_earthquake": fn, "tp": tp, "tn": tn})
        results["threshold_sweep"] = sweep
        best = max(sweep, key=lambda r: (r["f1"], r["recall"]))
        results["max_f1_operating_point"] = best

        with (out_dir / "threshold_sweep.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(sweep[0].keys()))
            w.writeheader()
            for r in sweep:
                w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                            for k, v in r.items()})

        fig, ax = plt.subplots(figsize=(7, 4.5))
        thr_x = [r["threshold"] for r in sweep]
        ax.plot(thr_x, [r["precision"] for r in sweep], marker="o", label="precision", color="C0")
        ax.plot(thr_x, [r["recall"] for r in sweep], marker="o", label="recall", color="C1")
        ax.plot(thr_x, [r["f1"] for r in sweep], marker="o", label="F1", color="C2")
        ax.set_xlabel("detection threshold"); ax.set_ylabel("score")
        ax.set_ylim(0, 1.02); ax.grid(alpha=0.25); ax.legend()
        ax.set_title(f"EQTransformer[{args.pretrained}] detection P/R/F1 vs threshold")
        fig.tight_layout()
        fig.savefig(out_dir / "threshold_sweep.png", dpi=120)
        plt.close(fig)

    (out_dir / "eqt_metrics.json").write_text(json.dumps(results, indent=2))

    # residual histogram (same style as student)
    tol_ms = ev.match_tolerance_s * 1000.0
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, phase, color in ((axes[0], "p", "C0"), (axes[1], "s", "C1")):
        res = [(r[f"{phase}_pick"] - r[f"gt_{phase}"]) / fs * 1000.0 for r in records
               if r["is_eq"] and np.isfinite(r[f"gt_{phase}"]) and r[f"{phase}_pick"] is not None]
        res = [x for x in res if abs(x) <= tol_ms]
        ax.hist(res, bins=50, range=(-tol_ms, tol_ms), color=color)
        st = residual_stats(res)
        ax.set_title(f"{phase.upper()} residuals (n={st['n']}, "
                     f"MAE {st['mae_ms']:.1f} ms, std {st['std_ms']:.1f} ms)")
        ax.set_xlabel("pick − ground truth (ms)"); ax.grid(alpha=0.25)
    axes[0].set_ylabel("count")
    fig.suptitle(f"EQTransformer[{args.pretrained}] pick residuals within ±{tol_ms:g} ms")
    fig.tight_layout()
    fig.savefig(out_dir / "pick_residuals.png", dpi=120)
    plt.close(fig)

    det = global_summary["detection"]
    lines = [
        f"EQTransformer[{args.pretrained}] baseline on split '{args.split}' "
        f"({n} traces: {global_summary['n_earthquake']} eq / {global_summary['n_noise']} noise)",
        f"params={n_params:,}  fp32 size={param_bytes/1e6:.2f} MB  "
        f"throughput={tput:.0f} traces/s on {device}",
        f"detection (threshold={ev.detection_threshold}): P={det['precision']:.4f} "
        f"R={det['recall']:.4f} F1={det['f1']:.4f} "
        f"(TP={det['tp']} FP={det['fp']} FN={det['fn']} TN={det['tn']})",
    ]
    for phase in ("p", "s"):
        pk = global_summary[f"{phase}_picks"]
        rt = pk["residuals_within_tol"]
        lines.append(
            f"{phase.upper()} picks: {pk['n_picked']}/{pk['n_ground_truth']} picked, "
            f"{pk['n_within_tolerance']} within ±{tol_ms:g} ms "
            f"(hit rate {pk['hit_rate_within_tol']:.3f}); "
            f"residuals within tol MAE={rt['mae_ms']:.1f} ms std={rt['std_ms']:.1f} ms")
    lines.append("CAVEAT: EQT run on the project's demean+bandpass(1-45)+std pipeline "
                 "(identical inputs to the student), NOT EQT's native preprocessing.")
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote artifacts to {out_dir}/")


if __name__ == "__main__":
    main()
