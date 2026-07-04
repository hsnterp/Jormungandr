#!/usr/bin/env python
"""Phase 4 evaluation: run a trained checkpoint on one split (default: test).

Reuses the existing config / dataset / model / loss modules — no re-implementation
of preprocessing, splitting, or the network.

Metrics
-------
- weighted BCE test loss (same loss as training, for continuity)
- event detection precision / recall / F1 (thresholded detection stream with
  scipy peak finding; threshold from ``eval.detection_threshold``)
- P / S pick residuals vs ground-truth arrival samples: MAE and std in ms
  (peaks on the P/S streams via ``eval.peak_height`` / ``eval.peak_min_distance_s``)
- everything broken down by SNR buckets (``eval.snr_buckets``, dB)

Usage:
    python scripts/evaluate.py --config configs/default.yaml \
        --checkpoint checkpoints/stage1/best.pt --out outputs/stage1_eval
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
from scipy.signal import find_peaks  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from seismic_edge_picker.config import load_config  # noqa: E402
from seismic_edge_picker.dataset import build_datasets  # noqa: E402
from seismic_edge_picker.losses import stream_weight_tensor, weighted_bce_loss  # noqa: E402
from seismic_edge_picker.model import build_model, count_parameters  # noqa: E402
from seismic_edge_picker import splits as S  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on a split.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/stage1/best.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out", default="outputs/stage1_eval")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--device", default=None, help="override (default: cuda if available, else cpu)"
    )
    return parser.parse_args()


def pick_from_stream(stream: np.ndarray, height: float, distance: int):
    """Highest peak above ``height`` -> (sample_index, prob), or (None, None)."""
    peaks, props = find_peaks(stream, height=height, distance=distance)
    if len(peaks) == 0:
        return None, None
    best = int(np.argmax(props["peak_heights"]))
    return int(peaks[best]), float(props["peak_heights"][best])


def detect_event(stream: np.ndarray, threshold: float, distance: int) -> bool:
    """Thresholded detection stream + peak finding -> event present?"""
    peaks, _ = find_peaks(stream, height=threshold, distance=distance, plateau_size=1)
    return len(peaks) > 0


def prf(tp: int, fp: int, fn: int):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def residual_stats(residuals_ms: list[float]):
    if not residuals_ms:
        return {"n": 0, "mae_ms": None, "std_ms": None, "mean_ms": None}
    r = np.asarray(residuals_ms, dtype=np.float64)
    return {
        "n": int(r.size),
        "mae_ms": float(np.mean(np.abs(r))),
        "std_ms": float(np.std(r)),
        "mean_ms": float(np.mean(r)),
    }


def snr_bucket_label(snr: float, edges) -> str:
    if not np.isfinite(snr):
        return "noise/unknown"
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo <= snr < hi:
            return f"[{lo:g},{hi:g}) dB"
    return "noise/unknown"


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    fs = cfg.data.sampling_rate
    ev = cfg.eval
    peak_distance = max(1, int(round(ev.peak_min_distance_s * fs)))
    tol_samples = ev.match_tolerance_s * fs
    edges = list(ev.snr_buckets)

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    print(
        f"checkpoint: {ckpt_path}  (epoch={ckpt.get('epoch')}, "
        f"best_val_loss={ckpt.get('best_val_loss'):.6f})"
    )

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"model parameters: {count_parameters(model):,}  device: {device}")

    datasets, split_indices = build_datasets(cfg)
    ds = datasets[args.split]
    print(f"evaluating split '{args.split}': {len(ds)} traces")

    batch_size = args.batch_size or cfg.train.batch_size
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=cfg.train.num_workers, pin_memory=device.type == "cuda",
    )

    # ground truth from metadata (eval split has no augmentation, so raw
    # arrival samples are the truth for the fixed 60 s window)
    meta = ds.metadata
    rows = ds.rows
    cat = meta[S.COL_CATEGORY].astype(str)
    is_eq = cat.isin(S.EARTHQUAKE_VALUES).to_numpy()

    weights = stream_weight_tensor(cfg, device)
    total_loss, total_n = 0.0, 0
    stream_loss = np.zeros(3, dtype=np.float64)

    records = []
    t0 = time.perf_counter()
    offset = 0
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x)
            loss, per_stream = weighted_bce_loss(pred, y, weights)
            n = x.shape[0]
            total_loss += loss.item() * n
            stream_loss += per_stream.detach().cpu().numpy() * n
            total_n += n

            pred_np = pred.detach().cpu().numpy()
            for b in range(n):
                row_idx = int(rows[offset + b])
                m = meta.iloc[row_idx]
                gt_p = S.parse_scalar(m.get(S.COL_P))
                gt_s = S.parse_scalar(m.get(S.COL_S))
                snr = S.parse_snr_db(m.get(S.COL_SNR))
                det, p_str, s_str = pred_np[b]
                p_pick, p_prob = pick_from_stream(p_str, ev.peak_height, peak_distance)
                s_pick, s_prob = pick_from_stream(s_str, ev.peak_height, peak_distance)
                records.append({
                    "row": row_idx,
                    "is_eq": bool(is_eq[row_idx]),
                    "snr_db": snr,
                    "detected": detect_event(det, ev.detection_threshold, peak_distance),
                    "det_max": float(det.max()),
                    "gt_p": gt_p, "gt_s": gt_s,
                    "p_pick": p_pick, "p_prob": p_prob,
                    "s_pick": s_pick, "s_prob": s_prob,
                })
            offset += n
    infer_s = time.perf_counter() - t0
    test_loss = total_loss / total_n
    stream_loss /= total_n
    print(
        f"inference done in {infer_s:.1f}s  weighted BCE={test_loss:.5f} "
        f"[det={stream_loss[0]:.4f} P={stream_loss[1]:.4f} S={stream_loss[2]:.4f}]"
    )

    # ---------------- detection P/R/F1 + pick residuals, global & per bucket ---
    def summarize(recs):
        tp = sum(1 for r in recs if r["is_eq"] and r["detected"])
        fn = sum(1 for r in recs if r["is_eq"] and not r["detected"])
        fp = sum(1 for r in recs if not r["is_eq"] and r["detected"])
        tn = sum(1 for r in recs if not r["is_eq"] and not r["detected"])
        precision, recall, f1 = prf(tp, fp, fn)
        out = {
            "n_traces": len(recs),
            "n_earthquake": tp + fn,
            "n_noise": fp + tn,
            "detection": {
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": precision, "recall": recall, "f1": f1,
            },
        }
        for phase, gt_key, pick_key in (("p", "gt_p", "p_pick"), ("s", "gt_s", "s_pick")):
            with_gt = [r for r in recs if r["is_eq"] and np.isfinite(r[gt_key])]
            picked = [r for r in with_gt if r[pick_key] is not None]
            res = [(r[pick_key] - r[gt_key]) / fs * 1000.0 for r in picked]
            matched = [x for x in res if abs(x) <= tol_samples / fs * 1000.0]
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

    global_summary = summarize(records)
    buckets = {}
    for r in records:
        buckets.setdefault(snr_bucket_label(r["snr_db"], edges), []).append(r)
    bucket_summaries = {
        k: summarize(v)
        for k, v in sorted(buckets.items(), key=lambda kv: (kv[0] == "noise/unknown", kv[0]))
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": int(ckpt.get("epoch")),
        "checkpoint_val_loss": float(ckpt.get("best_val_loss")),
        "split": args.split,
        "n_traces": len(ds),
        "device": str(device),
        "inference_seconds": infer_s,
        "eval_params": {
            "detection_threshold": ev.detection_threshold,
            "peak_height": ev.peak_height,
            "peak_min_distance_s": ev.peak_min_distance_s,
            "match_tolerance_s": ev.match_tolerance_s,
            "snr_buckets_db": edges,
        },
        "weighted_bce": {
            "total": test_loss,
            "detection": float(stream_loss[0]),
            "p": float(stream_loss[1]),
            "s": float(stream_loss[2]),
        },
        "overall": global_summary,
        "by_snr_bucket": bucket_summaries,
    }
    metrics_json = out_dir / "test_metrics.json"
    metrics_json.write_text(json.dumps(results, indent=2))

    # CSV: per-SNR-bucket table
    csv_path = out_dir / "snr_breakdown.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "bucket", "n_traces", "n_eq", "n_noise",
            "det_precision", "det_recall", "det_f1",
            "p_mae_ms", "p_std_ms", "p_hit_rate", "s_mae_ms", "s_std_ms", "s_hit_rate",
        ])
        for name, s in [("ALL", global_summary)] + list(bucket_summaries.items()):
            det = s["detection"]
            p_res = s["p_picks"]["residuals_within_tol"]
            s_res = s["s_picks"]["residuals_within_tol"]
            writer.writerow([
                name, s["n_traces"], s["n_earthquake"], s["n_noise"],
                f"{det['precision']:.4f}", f"{det['recall']:.4f}", f"{det['f1']:.4f}",
                "" if p_res["mae_ms"] is None else f"{p_res['mae_ms']:.1f}",
                "" if p_res["std_ms"] is None else f"{p_res['std_ms']:.1f}",
                "" if s["p_picks"]["hit_rate_within_tol"] is None
                else f"{s['p_picks']['hit_rate_within_tol']:.4f}",
                "" if s_res["mae_ms"] is None else f"{s_res['mae_ms']:.1f}",
                "" if s_res["std_ms"] is None else f"{s_res['std_ms']:.1f}",
                "" if s["s_picks"]["hit_rate_within_tol"] is None
                else f"{s['s_picks']['hit_rate_within_tol']:.4f}",
            ])

    # residual histograms (P and S small multiples, repo plot conventions)
    tol_ms = ev.match_tolerance_s * 1000.0
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, phase, color in ((axes[0], "p", "C0"), (axes[1], "s", "C1")):
        recs = [
            (r[f"{phase}_pick"] - r[f"gt_{phase}"]) / fs * 1000.0
            for r in records
            if r["is_eq"] and np.isfinite(r[f"gt_{phase}"]) and r[f"{phase}_pick"] is not None
        ]
        recs = [x for x in recs if abs(x) <= tol_ms]
        ax.hist(recs, bins=50, range=(-tol_ms, tol_ms), color=color)
        st = residual_stats(recs)
        ax.set_title(
            f"{phase.upper()} residuals (n={st['n']}, "
            f"MAE {st['mae_ms']:.1f} ms, std {st['std_ms']:.1f} ms)"
        )
        ax.set_xlabel("pick − ground truth (ms)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("count")
    fig.suptitle(f"Pick residuals within ±{tol_ms:g} ms — split={args.split}")
    fig.tight_layout()
    hist_path = out_dir / "pick_residuals.png"
    fig.savefig(hist_path, dpi=120)
    plt.close(fig)

    # text summary
    det = global_summary["detection"]
    lines = [
        f"Evaluation of {ckpt_path} (epoch {ckpt.get('epoch')}, "
        f"val loss {ckpt.get('best_val_loss'):.5f}) on split '{args.split}' "
        f"({len(ds)} traces: {global_summary['n_earthquake']} eq / "
        f"{global_summary['n_noise']} noise)",
        f"weighted BCE: {test_loss:.5f} "
        f"[det={stream_loss[0]:.4f} P={stream_loss[1]:.4f} S={stream_loss[2]:.4f}]",
        f"detection (threshold={ev.detection_threshold}): "
        f"precision={det['precision']:.4f} recall={det['recall']:.4f} F1={det['f1']:.4f} "
        f"(TP={det['tp']} FP={det['fp']} FN={det['fn']} TN={det['tn']})",
    ]
    for phase in ("p", "s"):
        pk = global_summary[f"{phase}_picks"]
        rt = pk["residuals_within_tol"]
        ra = pk["residuals_all_picks"]
        lines.append(
            f"{phase.upper()} picks: {pk['n_picked']}/{pk['n_ground_truth']} picked "
            f"({pk['pick_rate']:.3f}), {pk['n_within_tolerance']} within ±{tol_ms:g} ms "
            f"(hit rate {pk['hit_rate_within_tol']:.3f}); "
            f"residuals within tol: MAE={rt['mae_ms']:.1f} ms std={rt['std_ms']:.1f} ms; "
            f"all picks: MAE={ra['mae_ms']:.1f} ms std={ra['std_ms']:.1f} ms"
        )
    summary_path = out_dir / "summary.txt"
    summary_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote: {metrics_json}\n       {csv_path}\n       {hist_path}\n       {summary_path}")


if __name__ == "__main__":
    main()
