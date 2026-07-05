#!/usr/bin/env python
"""Generate evaluation-visualization artifacts into outputs/figures/.

Produces:
  1. Six waveform + student-prediction overlay PNGs (2 high-SNR eq, 2 low-SNR
     eq, 1 noise, 1 worst case), each with ground-truth and EQTransformer
     teacher P/S picks for comparison.
  2. snr_bucketed_performance.png  — student vs teacher F1 and P-pick MAE per
     SNR bucket, read from the existing eval JSONs.

The streaming demo (task 3) and latency table (task 4) are separate files.

Reuses the project's own config / dataset / model / eval helpers so nothing is
re-implemented. Student = checkpoints/stage2_distill/best.pt. Teacher picks are
computed by running SeisBench's pretrained EQTransformer on the SAME
preprocessed test tensors (identical inputs to the student), matching the
methodology in scripts/eqtransformer_baseline.py. The per-trace teacher outputs
are NOT taken from data/teacher_cache/ because that cache covers the TRAIN
split only; here we need the TEST traces we plot.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from seismic_edge_picker.config import load_config  # noqa: E402
from seismic_edge_picker.dataset import build_datasets  # noqa: E402
from seismic_edge_picker.model import build_model, count_parameters  # noqa: E402
from seismic_edge_picker import splits as S  # noqa: E402
from evaluate import pick_from_stream, detect_event  # noqa: E402

OUT = REPO / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---- palette (validated dataviz light-surface slots) -----------------------
C_STUDENT = "#2a78d6"   # slot 1 blue
C_TEACHER = "#1baf7a"   # slot 2 aqua
C_P = "#2a78d6"         # P stream / P truth (blue)
C_S = "#eb6834"         # S stream / S truth (orange)
C_DET = "#52514e"       # detection stream (neutral ink)
C_TEACH_P = "#4a3aa7"   # teacher P pick (violet)
C_TEACH_S = "#e87ba4"   # teacher S pick (magenta)
C_WF = ("#2a78d6", "#8a8a86", "#86b6ef")  # Z / N / E waveform lines
SURFACE = "#ffffff"
plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": "#c9c8c3",
    "axes.labelcolor": "#0b0b0b",
    "text.color": "#0b0b0b",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "font.size": 10,
})


def raw_waveform(ds, row_idx, n_channels, n_samples):
    """Raw (pre-filter) 3-channel waveform straight from the STEAD cache."""
    from seismic_edge_picker.dataset import _get_waveform
    return _get_waveform(ds.ds, row_idx, n_channels, n_samples)


def main():
    cfg = load_config(str(REPO / "configs" / "default.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fs = float(cfg.data.sampling_rate)
    ev = cfg.eval
    peak_distance = max(1, int(round(ev.peak_min_distance_s * fs)))
    tol_ms = ev.match_tolerance_s * 1000.0

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

    # ---- pass 1: scalar records over the whole test split -----------------
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

    # ---- selection --------------------------------------------------------
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
    # 2 high-SNR earthquakes that the model picks well (clean showcase)
    hi = take([r for r in eq_hi if r["p_err"] is not None and r["p_err"] <= tol_ms],
              2, used)
    # 2 low-SNR earthquakes (detected, so the overlay is meaningful)
    lo = take([r for r in eq_lo if r["detected"]], 2, used)
    # 1 pure-noise trace the model correctly ignores (true negative)
    noise = take([r for r in recs if not r["is_eq"] and not r["detected"]], 1, used)
    # 1 worst case: largest P pick error among earthquakes
    worst = take(sorted(picked_eq, key=lambda r: -r["p_err"]), 1, used)
    worst_fp = None
    if not worst:  # fall back to a false positive on noise
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

    def student_streams(i):
        x, _ = ds[i]
        with torch.inference_mode():
            return model(x.unsqueeze(0).to(device)).cpu().numpy()[0], x.numpy()

    def teacher_picks(raw):
        # EQTransformer NATIVE preprocessing (demean + peak-norm + taper), via
        # the model's own annotate_batch_pre on the RAW waveform — the fair
        # per-model conditioning used in the corrected comparison.
        rb = torch.tensor(raw[None], dtype=torch.float32, device=device)
        with torch.inference_mode():
            out = teacher(teacher.annotate_batch_pre(rb, {}))
        det = out[0].cpu().numpy()[0]
        p_str = out[1].cpu().numpy()[0]
        s_str = out[2].cpu().numpy()[0]
        p_pick, _ = pick_from_stream(p_str, ev.peak_height, peak_distance)
        s_pick, _ = pick_from_stream(s_str, ev.peak_height, peak_distance)
        return p_pick, s_pick, det

    t = np.arange(cfg.data.window_samples) / fs

    def sec(sample):
        return None if sample is None or not np.isfinite(sample) else sample / fs

    order = 0
    for kind, r in selection:
        order += 1
        i = r["i"]
        pred, _xin = student_streams(i)
        raw = raw_waveform(ds, r["row"], cfg.data.n_channels, cfg.data.window_samples)
        tp_pick, ts_pick, _tdet = teacher_picks(raw)

        m = meta.iloc[r["row"]]
        trace_name = str(m.get("trace_name_original") or "").strip()
        if not trace_name or trace_name == "nan":
            net = str(m.get(S.COL_NETWORK, "")).strip()
            sta = str(m.get(S.COL_STATION, "")).strip()
            trace_name = f"{net}.{sta}".strip(".") or str(m.get(S.COL_TRACE, f"row{r['row']}"))
        snr = r["snr"]
        snr_txt = f"{snr:.1f} dB" if np.isfinite(snr) else "n/a (noise)"

        fig, (ax0, ax1) = plt.subplots(
            2, 1, figsize=(12, 6), sharex=True,
            gridspec_kw={"height_ratios": [1.15, 1.0], "hspace": 0.12},
        )

        # top: raw 3-channel waveform, per-channel normalized + offset
        labels = ("Z", "N", "E")
        for ch in range(3):
            d = raw[ch]
            scale = max(float(np.max(np.abs(d))), 1e-9)
            ax0.plot(t, d / scale + (2 - ch) * 2.4, color=C_WF[ch], lw=0.6,
                     alpha=0.9)
            ax0.text(0.15, (2 - ch) * 2.4, labels[ch], va="center", ha="right",
                     color=C_WF[ch], fontsize=11, fontweight="bold")
        ax0.set_yticks([])
        ax0.set_ylabel("raw waveform\n(per-channel norm.)")
        ax0.margins(x=0)
        ax0.spines[["top", "right", "left"]].set_visible(False)

        # bottom: student probability streams
        ax1.plot(t, pred[0], color=C_DET, lw=1.3, label="Detection (student)")
        ax1.plot(t, pred[1], color=C_P, lw=1.3, label="P prob (student)")
        ax1.plot(t, pred[2], color=C_S, lw=1.3, label="S prob (student)")
        ax1.axhline(ev.detection_threshold, color=C_DET, ls="-", lw=0.6, alpha=0.3)
        ax1.set_ylim(-0.03, 1.05)
        ax1.set_ylabel("probability")
        ax1.set_xlabel("time (s)")
        ax1.margins(x=0)
        ax1.spines[["top", "right"]].set_visible(False)

        # ground-truth arrivals (dashed) + teacher picks (dotted), on both axes
        def vlines(sample, color, style, axes):
            s = sec(sample)
            if s is None:
                return
            for ax in axes:
                ax.axvline(s, color=color, ls=style, lw=1.4, alpha=0.9)

        vlines(r["gt_p"], C_P, (0, (6, 4)), (ax0, ax1))
        vlines(r["gt_s"], C_S, (0, (6, 4)), (ax0, ax1))
        vlines(tp_pick, C_TEACH_P, (0, (1, 2)), (ax0, ax1))
        vlines(ts_pick, C_TEACH_S, (0, (1, 2)), (ax0, ax1))

        # legend for the vertical-line semantics (dashed = truth, dotted = EQT)
        handles = [
            Line2D([0], [0], color=C_DET, lw=1.3, label="Detection (student)"),
            Line2D([0], [0], color=C_P, lw=1.3, label="P prob (student)"),
            Line2D([0], [0], color=C_S, lw=1.3, label="S prob (student)"),
            Line2D([0], [0], color=C_P, lw=1.4, ls=(0, (6, 4)), label="GT P (truth)"),
            Line2D([0], [0], color=C_S, lw=1.4, ls=(0, (6, 4)), label="GT S (truth)"),
            Line2D([0], [0], color=C_TEACH_P, lw=1.4, ls=(0, (1, 2)), label="EQT P (teacher)"),
            Line2D([0], [0], color=C_TEACH_S, lw=1.4, ls=(0, (1, 2)), label="EQT S (teacher)"),
        ]
        ax1.legend(handles=handles, loc="upper right", ncol=2, fontsize=8,
                   framealpha=0.9, edgecolor="#c9c8c3")

        # title
        kind_label = {
            "high_snr_eq": "High-SNR earthquake",
            "low_snr_eq": "Low-SNR earthquake",
            "noise": "Pure noise",
            "worst_case": "Worst case (largest P error)"
                          if worst_fp is None else "Worst case (noise false positive)",
        }[kind]
        extra = ""
        if r["p_err"] is not None:
            extra = f"  |  student P error {r['p_err']:.0f} ms"
        fig.suptitle(f"{kind_label}  —  {trace_name}  |  SNR {snr_txt}{extra}",
                     fontsize=12, fontweight="bold", y=0.97)

        fname = OUT / f"overlay_{order}_{kind}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {fname.name}")

    # ---- task 2: SNR-bucketed performance chart ---------------------------
    make_snr_chart()


def make_snr_chart():
    # corrected-protocol numbers: threshold selected on validation, teacher
    # scored under its native SeisBench preprocessing (see fair_comparison.py)
    comp = json.loads((REPO / "outputs" / "fair_eval" / "comparison.json").read_text())
    buckets = ["[0,10) dB", "[10,20) dB", "[20,100) dB"]
    labels = ["0–10", "10–20", "20–100"]
    thr = comp["val_selected_thresholds"]
    s_thr, t_thr = thr["student"], thr["teacher_native"]
    ov = comp["protocols"]["C_after"]
    s_ov_f1, t_ov_f1 = ov["student"]["f1"], ov["teacher"]["f1"]

    ov_s, ov_t = ov["student"], ov["teacher"]

    def col(model, metric):
        return [comp["per_bucket"][b]["after"][f"{model}_{metric}"] for b in buckets]

    s_f1, t_f1 = col("student", "f1"), col("teacher", "f1")
    s_pmae, t_pmae = col("student", "p_mae_ms"), col("teacher", "p_mae_ms")
    s_smae, t_smae = col("student", "s_mae_ms"), col("teacher", "s_mae_ms")

    x = np.arange(len(buckets))
    w = 0.38
    fig, (axF, axP, axS) = plt.subplots(1, 3, figsize=(16.5, 5))

    def grouped(ax, s_vals, t_vals, ylabel, title, fmt, ylim=None):
        b1 = ax.bar(x - w / 2, s_vals, w,
                    label=f"Student (distilled), thr {s_thr:.2f}",
                    color=C_STUDENT, zorder=3)
        b2 = ax.bar(x + w / 2, t_vals, w,
                    label=f"Teacher (EQT native), thr {t_thr:.2f}",
                    color=C_TEACHER, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("SNR bucket (dB)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e6e5e1", zorder=0)
        ax.set_axisbelow(True)
        if ylim:
            ax.set_ylim(*ylim)
        for bars, vals in ((b1, s_vals), (b2, t_vals)):
            for rect, v in zip(bars, vals):
                if np.isfinite(v):
                    ax.annotate(fmt(v), (rect.get_x() + rect.get_width() / 2,
                                         rect.get_height()),
                                ha="center", va="bottom", fontsize=8,
                                color="#52514e", xytext=(0, 2),
                                textcoords="offset points")
        ax.legend(loc="best", fontsize=9, framealpha=0.9, edgecolor="#c9c8c3")

    mae_top = max(t_pmae + t_smae + s_pmae + s_smae) * 1.18
    grouped(axF, s_f1, t_f1, "Detection F1 (earthquake-only buckets)",
            "Detection F1 by SNR", lambda v: f"{v:.3f}", ylim=(0.9, 1.005))
    grouped(axP, s_pmae, t_pmae, "P-pick MAE (ms, within ±500 ms)",
            "P-pick MAE by SNR", lambda v: f"{v:.0f}", ylim=(0, mae_top))
    grouped(axS, s_smae, t_smae, "S-pick MAE (ms, within ±500 ms)",
            "S-pick MAE by SNR", lambda v: f"{v:.0f}", ylim=(0, mae_top))

    fig.suptitle("Student vs EQTransformer teacher across SNR — corrected protocol "
                 "(STEAD test split)", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.055,
             "Thresholds selected on validation; teacher scored under native SeisBench "
             "preprocessing (peak-norm, no bandpass).",
             ha="center", fontsize=8, color="#52514e")
    fig.text(0.5, 0.02,
             f"Overall (incl. noise): detection F1 student {s_ov_f1:.4f} / teacher "
             f"{t_ov_f1:.4f};  P-MAE {ov_s['p_mae_ms']:.1f}/{ov_t['p_mae_ms']:.1f} ms;  "
             f"S-MAE {ov_s['s_mae_ms']:.1f}/{ov_t['s_mae_ms']:.1f} ms.  "
             "Earthquake-only buckets → per-bucket F1 tracks recall.",
             ha="center", fontsize=8, color="#52514e")
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))
    fname = OUT / "snr_bucketed_performance.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"wrote {fname.name}")


if __name__ == "__main__":
    main()
