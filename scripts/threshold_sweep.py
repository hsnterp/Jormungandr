#!/usr/bin/env python
"""Phase-4 postprocessing tuning: detection-threshold + min-duration sweep.

NOT a model change and NOT a retrain. Runs inference on the test split ONCE
(reusing build_datasets / build_model), then evaluates a whole grid of
(threshold, min_duration) operating points from a compact per-trace summary.

Detection rule (postprocessing):
    a trace is "detected" iff the detection stream stays >= threshold for a run
    of at least ``min_duration`` consecutive samples.

For each threshold we cache, per trace, the longest above-threshold run length.
Detection(threshold, N) == (max_run(threshold) >= N), so every N is free after
the single inference pass.

Usage:
    python scripts/threshold_sweep.py --config configs/default.yaml \
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
from torch.utils.data import DataLoader  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from seismic_edge_picker.config import load_config  # noqa: E402
from seismic_edge_picker.dataset import build_datasets  # noqa: E402
from seismic_edge_picker.model import build_model, count_parameters  # noqa: E402
from seismic_edge_picker import splits as S  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Detection threshold + min-duration sweep.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--checkpoint", default="checkpoints/stage1/best.pt")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--out", default="outputs/stage1_eval")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--smoke", type=int, default=0, metavar="N",
                   help="sweep only the first N traces of the split (quick check "
                        "when the full STEAD split is unavailable)")
    return p.parse_args()


def max_run_above(stream: np.ndarray, threshold: float) -> int:
    """Longest run of consecutive samples with stream >= threshold."""
    mask = stream >= threshold
    if not mask.any():
        return 0
    # run lengths via diff of the cumulative reset trick
    idx = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
    # idx pairs mark (start, end) of runs
    return int(np.max(idx[1::2] - idx[0::2]))


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    fs = cfg.data.sampling_rate

    thresholds = np.round(np.arange(0.10, 0.9001, 0.05), 2)
    # min-duration values (samples) with human-readable ms; 1 sample == "no rule"
    min_dur_samples = [1, 5, 10, 20, 50]
    min_dur_ms = [int(round(n / fs * 1000)) for n in min_dur_samples]

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(
        f"checkpoint: {ckpt_path} (epoch={ckpt.get('epoch')}, "
        f"val={ckpt.get('best_val_loss'):.5f})  params={count_parameters(model):,}"
    )

    datasets, _ = build_datasets(cfg)
    ds = datasets[args.split]
    meta = ds.metadata
    rows = ds.rows
    cat = meta[S.COL_CATEGORY].astype(str)
    is_eq_all = cat.isin(S.EARTHQUAKE_VALUES).to_numpy()

    # Subset(ds, range(N)) preserves order, so rows[offset+b] stays aligned.
    loader_ds = ds
    if args.smoke:
        from torch.utils.data import Subset
        loader_ds = Subset(ds, list(range(min(args.smoke, len(ds)))))
        print(f"[smoke] sweeping first {len(loader_ds)} of {len(ds)} traces")

    batch_size = args.batch_size or cfg.train.batch_size
    loader = DataLoader(
        loader_ds, batch_size=batch_size, shuffle=False,
        num_workers=cfg.train.num_workers, pin_memory=device.type == "cuda",
    )

    n = len(loader_ds)
    is_eq = np.zeros(n, dtype=bool)
    # per-trace longest above-threshold run for each threshold
    runs = np.zeros((n, len(thresholds)), dtype=np.int32)

    t0 = time.perf_counter()
    offset = 0
    with torch.inference_mode():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            det = model(x)[:, 0].detach().cpu().numpy()  # (B, 6000) detection stream
            for b in range(det.shape[0]):
                stream = det[b]
                is_eq[offset + b] = is_eq_all[int(rows[offset + b])]
                for ti, thr in enumerate(thresholds):
                    runs[offset + b, ti] = max_run_above(stream, float(thr))
            offset += det.shape[0]
    print(f"inference + run-length summary in {time.perf_counter() - t0:.1f}s "
          f"({n} traces)")

    eq_mask = is_eq
    noise_mask = ~is_eq
    n_eq = int(eq_mask.sum())
    n_noise = int(noise_mask.sum())

    # ---- sweep ------------------------------------------------------------
    sweep_rows = []
    for ndur, ndur_ms in zip(min_dur_samples, min_dur_ms):
        for ti, thr in enumerate(thresholds):
            detected = runs[:, ti] >= ndur
            tp = int(np.sum(detected & eq_mask))
            fn = int(np.sum(~detected & eq_mask))
            fp = int(np.sum(detected & noise_mask))
            tn = int(np.sum(~detected & noise_mask))
            precision, recall, f1 = prf(tp, fp, fn)
            sweep_rows.append({
                "threshold": float(thr),
                "min_duration_samples": ndur,
                "min_duration_ms": ndur_ms,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "fp_noise": fp,          # false positives (all on noise traces)
                "fn_earthquake": fn,     # missed earthquakes
            })

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "threshold_sweep.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(sweep_rows[0].keys()))
        writer.writeheader()
        for r in sweep_rows:
            writer.writerow({
                k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()
            })

    # ---- recommendations --------------------------------------------------
    max_f1 = max(sweep_rows, key=lambda r: (r["f1"], r["recall"]))
    # low-false-alarm: minimize FP while keeping recall usable (>=0.98)
    lowfa_candidates = [r for r in sweep_rows if r["recall"] >= 0.98]
    low_fa = min(lowfa_candidates, key=lambda r: (r["fp_noise"], -r["f1"]))
    # recall-preserving: fewest missed earthquakes, then fewest FP, then best F1
    recall_pref = min(sweep_rows, key=lambda r: (r["fn_earthquake"], r["fp_noise"], -r["f1"]))

    recs = {
        "max_f1": max_f1,
        "low_false_alarm": low_fa,
        "recall_preserving": recall_pref,
        "grid": {
            "thresholds": [float(t) for t in thresholds],
            "min_duration_samples": min_dur_samples,
            "min_duration_ms": min_dur_ms,
        },
        "n_traces": n, "n_earthquake": n_eq, "n_noise": n_noise,
        "checkpoint": str(ckpt_path), "split": args.split,
    }
    (out_dir / "threshold_recommendations.json").write_text(json.dumps(recs, indent=2))

    # ---- plots ------------------------------------------------------------
    # Panel A: P/R/F1 vs threshold at the no-duration-rule baseline (N=1 sample)
    # Panel B: min-duration effect on F1 and FP across thresholds
    base = [r for r in sweep_rows if r["min_duration_samples"] == 1]
    base.sort(key=lambda r: r["threshold"])
    thr_x = [r["threshold"] for r in base]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.plot(thr_x, [r["precision"] for r in base], marker="o", label="precision", color="C0")
    ax.plot(thr_x, [r["recall"] for r in base], marker="o", label="recall", color="C1")
    ax.plot(thr_x, [r["f1"] for r in base], marker="o", label="F1", color="C2")
    ax.axvline(max_f1["threshold"], color="C2", ls="--", alpha=0.5)
    ax.set_xlabel("detection threshold")
    ax.set_ylabel("score")
    ax.set_title("Detection P/R/F1 vs threshold (no min-duration)")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1]
    for ndur, ndur_ms in zip(min_dur_samples, min_dur_ms):
        sub = [r for r in sweep_rows if r["min_duration_samples"] == ndur]
        sub.sort(key=lambda r: r["threshold"])
        label = "no rule" if ndur == 1 else f"{ndur_ms} ms"
        ax.plot([r["threshold"] for r in sub], [r["fp_noise"] for r in sub],
                marker="o", label=label)
    ax.set_xlabel("detection threshold")
    ax.set_ylabel("false positives on noise")
    ax.set_title("Min-duration effect on noise false alarms")
    ax.grid(alpha=0.25)
    ax.legend(title="min-duration")
    fig.tight_layout()
    plot_path = out_dir / "threshold_sweep.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)

    # ---- console summary --------------------------------------------------
    def fmt(r):
        return (f"thr={r['threshold']:.2f} min_dur={r['min_duration_ms']}ms "
                f"P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1']:.4f} "
                f"FP_noise={r['fp_noise']} FN_eq={r['fn_earthquake']}")

    print(f"\nsplit={args.split}  eq={n_eq}  noise={n_noise}")
    print("MAX-F1          :", fmt(max_f1))
    print("LOW-FALSE-ALARM :", fmt(low_fa))
    print("RECALL-PRESERVE :", fmt(recall_pref))
    print(f"\nwrote: {csv_path}\n       {out_dir / 'threshold_recommendations.json'}"
          f"\n       {plot_path}")


if __name__ == "__main__":
    main()
