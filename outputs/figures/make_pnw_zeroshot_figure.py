#!/usr/bin/env python
"""STEAD (in-distribution) vs PNW (zero-shot) generalization figure.

Small-multiple bars contrasting the SAME student model on the data it was
trained/evaluated on (STEAD) versus a different network/region it has never
seen (PNW), scored identically. Reads outputs/pnw_zeroshot/pnw_metrics.json at
runtime; STEAD student numbers are the published in-distribution values.

Writes outputs/figures/pnw_zeroshot.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

STEAD_BLUE, PNW_ORANGE = "#2a78d6", "#eb6834"   # validated pair (CVD ΔE 96.7)
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

# STEAD student, in-distribution (deployed val-selected threshold 0.89):
STEAD = {"f1": 0.9925, "p_mae": 36.9, "s_mae": 71.4, "p_hit": 0.981, "s_hit": 0.961}


def load_pnw():
    d = json.loads((REPO / "outputs/pnw_zeroshot/pnw_metrics.json").read_text())
    return {
        "f1": d["detection_at_stead_deployed_thr"]["f1"],
        "p_mae": d["p_picks"]["residuals_within_tol"]["mae_ms"],
        "s_mae": d["s_picks"]["residuals_within_tol"]["mae_ms"],
        "p_hit": d["p_picks"]["hit_rate_within_tol"],
        "s_hit": d["s_picks"]["hit_rate_within_tol"],
        "n_events": d["n_events"], "n_noise": d["n_noise"],
    }


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0, labelcolor=MUTED)
    ax.set_yticklabels([])


def pair_panel(ax, s_val, p_val, fmt, title, subtitle, lower_better):
    vals = [s_val, p_val]
    bars = ax.bar([0, 1], vals, width=0.6, color=[STEAD_BLUE, PNW_ORANGE],
                  zorder=3, edgecolor=SURF, linewidth=1.5)
    top = max(vals)
    ax.set_ylim(0, top * 1.20)
    for x, v in zip([0, 1], vals):
        ax.text(x, v + top * 0.03, fmt(v), ha="center", va="bottom",
                fontsize=12, fontweight="bold", color=INK, zorder=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["STEAD\n(in-dist.)", "PNW\n(zero-shot)"], fontsize=9.5, color=MUTED)
    ax.text(0, 1.135, title, transform=ax.transAxes, fontsize=13,
            fontweight="bold", color=INK, ha="left", va="bottom")
    ax.text(0, 1.045, subtitle, transform=ax.transAxes, fontsize=9, color=MUTED,
            ha="left", va="bottom")
    style(ax)


def hit_panel(ax, pnw):
    import numpy as np
    x = np.arange(2)
    w = 0.36
    ax.bar(x - w / 2, [STEAD["p_hit"], STEAD["s_hit"]], w, color=STEAD_BLUE,
           zorder=3, edgecolor=SURF, linewidth=1.5, label="STEAD (in-dist.)")
    ax.bar(x + w / 2, [pnw["p_hit"], pnw["s_hit"]], w, color=PNW_ORANGE,
           zorder=3, edgecolor=SURF, linewidth=1.5, label="PNW (zero-shot)")
    ax.set_ylim(0, 1.15)
    for xi, (a, b) in zip(x, [(STEAD["p_hit"], pnw["p_hit"]),
                              (STEAD["s_hit"], pnw["s_hit"])]):
        ax.text(xi - w / 2, a + 0.03, f"{a:.2f}", ha="center", va="bottom",
                fontsize=10.5, fontweight="bold", color=INK)
        ax.text(xi + w / 2, b + 0.03, f"{b:.2f}", ha="center", va="bottom",
                fontsize=10.5, fontweight="bold", color=INK)
    ax.set_xticks(x); ax.set_xticklabels(["P", "S"], fontsize=10, color=MUTED)
    ax.text(0, 1.135, "Pick hit-rate", transform=ax.transAxes, fontsize=13,
            fontweight="bold", color=INK, ha="left", va="bottom")
    ax.text(0, 1.045, "fraction within ±500 ms · higher is better",
            transform=ax.transAxes, fontsize=9, color=MUTED, ha="left", va="bottom")
    style(ax)
    ax.legend(loc="lower center", frameon=False, fontsize=9, ncol=1,
              bbox_to_anchor=(0.5, -0.02), labelcolor=MUTED)


def main():
    pnw = load_pnw()
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 4.6))
    fig.patch.set_facecolor(SURF)
    for ax in axes:
        ax.set_facecolor(SURF)

    pair_panel(axes[0], STEAD["f1"], pnw["f1"], lambda v: f"{v:.4f}",
               "Detection F1", "higher is better · deployed threshold 0.89", False)
    pair_panel(axes[1], STEAD["p_mae"], pnw["p_mae"], lambda v: f"{v:.1f}",
               "P pick MAE", "ms within ±500 ms · lower is better", True)
    pair_panel(axes[2], STEAD["s_mae"], pnw["s_mae"], lambda v: f"{v:.1f}",
               "S pick MAE", "ms within ±500 ms · lower is better", True)
    hit_panel(axes[3], pnw)

    fig.suptitle("Zero-shot generalization: 48k student, STEAD-trained, evaluated on PNW",
                 fontsize=15.5, fontweight="bold", color=INK, x=0.007, ha="left", y=0.99)
    fig.text(0.007, 0.925,
             f"Same model, no fine-tuning · PNW test subset: {pnw['n_events']:,} events "
             f"(earthquake+explosion) + {pnw['n_noise']:,} noise · different network / "
             "region / instruments · scored identically to STEAD",
             fontsize=9.5, color=MUTED, ha="left", va="top")

    fig.tight_layout(rect=(0, 0, 1, 0.86), w_pad=2.5)
    fig.subplots_adjust(top=0.74)
    out = REPO / "outputs/figures/pnw_zeroshot.png"
    fig.savefig(out, dpi=150, facecolor=SURF, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    print(f"  STEAD  F1 {STEAD['f1']:.4f}  P {STEAD['p_mae']:.1f}  S {STEAD['s_mae']:.1f}  "
          f"Phit {STEAD['p_hit']:.3f}  Shit {STEAD['s_hit']:.3f}")
    print(f"  PNW    F1 {pnw['f1']:.4f}  P {pnw['p_mae']:.1f}  S {pnw['s_mae']:.1f}  "
          f"Phit {pnw['p_hit']:.3f}  Shit {pnw['s_hit']:.3f}")


if __name__ == "__main__":
    main()
