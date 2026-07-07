#!/usr/bin/env python
"""Headline baseline comparison: student vs PhaseNet vs EQTransformer.

Small-multiples bar panels (one metric per panel, since the metrics live on
different scales — never a shared/dual axis). All three models are scored under
the SAME corrected protocol:

  * detection F1 at a threshold selected on the validation split and applied once
    to test (student & teacher from outputs/fair_eval/comparison.json 'C_after';
    PhaseNet from outputs/phasenet_baseline/pn_metrics.json 'val_selected'),
  * within-tolerance P / S pick MAE on the identical STEAD test traces,
  * the two SeisBench baselines are fed their NATIVE preprocessing.

Reads the metric JSONs at runtime so the figure can't drift from the numbers.
Writes outputs/figures/baseline_comparison.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

# categorical slots (validated: min adjacent CVD ΔE 73.6). Student is the hero
# blue; PhaseNet the aqua baseline; EQTransformer the violet teacher (matches
# the demo's teacher color --tp).
COLORS = {"student": "#2a78d6", "phasenet": "#1baf7a", "teacher": "#4a3aa7"}
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

ORDER = ["student", "phasenet", "teacher"]
NAMES = {
    "student": "Jormungandr\n(student, 48k)",
    "phasenet": "PhaseNet\n(268k)",
    "teacher": "EQTransformer\n(377k)",
}


def load_numbers():
    fair = json.loads((REPO / "outputs/fair_eval/comparison.json").read_text())
    c = fair["protocols"]["C_after"]
    pn = json.loads((REPO / "outputs/phasenet_baseline/pn_metrics.json").read_text())
    pv = pn["overall_val_selected"]

    def hit(picks):
        return picks["hit_rate_within_tol"]

    return {
        "student": {
            "f1": c["student"]["f1"], "thr": c["student"]["threshold"],
            "p_mae": c["student"]["p_mae_ms"], "s_mae": c["student"]["s_mae_ms"],
            "params": 48051,
        },
        "phasenet": {
            "f1": pv["detection"]["f1"], "thr": pn["val_selected_threshold"],
            "p_mae": pv["p_picks"]["residuals_within_tol"]["mae_ms"],
            "s_mae": pv["s_picks"]["residuals_within_tol"]["mae_ms"],
            "p_hit": hit(pv["p_picks"]), "s_hit": hit(pv["s_picks"]),
            "params": pn["params"],
        },
        "teacher": {
            "f1": c["teacher"]["f1"], "thr": c["teacher"]["threshold"],
            "p_mae": c["teacher"]["p_mae_ms"], "s_mae": c["teacher"]["s_mae_ms"],
            "params": 376935,
        },
    }


def panel(ax, vals, fmt, title, subtitle, invert_good=False):
    xs = range(len(ORDER))
    heights = [vals[m] for m in ORDER]
    bars = ax.bar(xs, heights, width=0.62,
                  color=[COLORS[m] for m in ORDER], zorder=3,
                  edgecolor=SURF, linewidth=1.5)
    top = max(heights)
    ax.set_ylim(0, top * 1.20)
    for x, h, m in zip(xs, heights, ORDER):
        ax.text(x, h + top * 0.03, fmt(h), ha="center", va="bottom",
                fontsize=11.5, fontweight="bold", color=INK, zorder=4)
    # mark the best performer
    best = min(heights) if invert_good else max(heights)
    bi = heights.index(best)
    ax.text(bi, best * 0.5, "best", ha="center", va="center",
            fontsize=8.5, color="white", fontweight="bold", rotation=90, zorder=5)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([NAMES[m] for m in ORDER], fontsize=9, color=MUTED)
    ax.text(0, 1.135, title, transform=ax.transAxes, fontsize=13,
            fontweight="bold", color=INK, ha="left", va="bottom")
    ax.text(0, 1.045, subtitle, transform=ax.transAxes, fontsize=9,
            color=MUTED, ha="left", va="bottom")
    ax.grid(axis="y", color=GRID, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0, labelcolor=MUTED)
    ax.set_yticklabels([])


def main():
    d = load_numbers()
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 4.5))
    fig.patch.set_facecolor(SURF)
    for ax in axes:
        ax.set_facecolor(SURF)

    panel(axes[0], {m: d[m]["f1"] for m in ORDER}, lambda v: f"{v:.4f}",
          "Detection F1", "higher is better · val-selected threshold")
    panel(axes[1], {m: d[m]["p_mae"] for m in ORDER}, lambda v: f"{v:.1f}",
          "P pick MAE", "milliseconds · lower is better", invert_good=True)
    panel(axes[2], {m: d[m]["s_mae"] for m in ORDER}, lambda v: f"{v:.1f}",
          "S pick MAE", "milliseconds · lower is better", invert_good=True)
    panel(axes[3], {m: d[m]["params"] / 1000 for m in ORDER},
          lambda v: f"{v:.0f}k", "Parameters", "thousands · lower is lighter",
          invert_good=True)

    fig.suptitle("Jormungandr vs. lightweight P/S baselines on the STEAD test split",
                 fontsize=15.5, fontweight="bold", color=INK, x=0.007, ha="left", y=0.99)
    fig.text(0.007, 0.925,
             "7,781 test traces (4,957 eq / 2,824 noise) · SeisBench PhaseNet & "
             "EQTransformer 'stead' weights, native preprocessing · pick MAE within "
             "±500 ms tolerance",
             fontsize=9.5, color=MUTED, ha="left", va="top")

    fig.tight_layout(rect=(0, 0, 1, 0.86), w_pad=2.5)
    fig.subplots_adjust(top=0.74)
    out = REPO / "outputs/figures/baseline_comparison.png"
    fig.savefig(out, dpi=150, facecolor=SURF, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    # console echo of the exact numbers plotted
    for m in ORDER:
        x = d[m]
        print(f"  {m:9s} F1={x['f1']:.4f}@{x['thr']:.2f} "
              f"P={x['p_mae']:.1f}ms S={x['s_mae']:.1f}ms params={x['params']:,}")


if __name__ == "__main__":
    main()
