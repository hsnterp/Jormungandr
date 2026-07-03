#!/usr/bin/env python
"""Phase 1 verification: plot example traces with their label masks overlaid.

Saves a PNG so correctness can be eyeballed (preprocessing, arrival alignment,
detection window, Gaussian pick bumps).

    python scripts/sanity_check_data.py --config configs/default.yaml \
        --n 5 --split train --out outputs/sanity_labels.png

NOTE: this loads STEAD via SeisBench and therefore requires the dataset cache
to be present (do not run while the initial download is still in progress).
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from seismic_edge_picker.config import load_config          # noqa: E402
from seismic_edge_picker.dataset import build_datasets      # noqa: E402
from seismic_edge_picker import splits as S                 # noqa: E402


STREAM_NAMES = ["detection", "P pick", "S pick"]
STREAM_COLORS = ["tab:green", "tab:blue", "tab:red"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--out", default="outputs/sanity_labels.png")
    args = ap.parse_args()

    cfg = load_config(args.config)
    datasets, split_idx = build_datasets(cfg)
    ds = datasets[args.split]
    sr = cfg.data.sampling_rate

    n = min(args.n, len(ds))
    rng = np.random.default_rng(cfg.seed)
    picks = rng.choice(len(ds), size=n, replace=False)

    fig, axes = plt.subplots(n, 1, figsize=(12, 2.6 * n), squeeze=False)
    t = np.arange(cfg.data.window_samples) / sr

    for ax, di in zip(axes[:, 0], picks):
        x, y = ds[int(di)]
        x = x.numpy()
        y = y.numpy()
        row = int(ds.rows[int(di)])
        meta = ds.metadata.iloc[row]
        cat = str(meta.get(S.COL_CATEGORY))
        snr = S.parse_snr_db(meta.get(S.COL_SNR))

        # plot vertical channel (component 0) waveform, scaled into view
        wav = x[0]
        wav = wav / (np.abs(wav).max() + 1e-8)
        ax.plot(t, wav, color="0.5", lw=0.6, label="waveform (ch0)")

        for s in range(3):
            ax.plot(t, y[s], color=STREAM_COLORS[s], lw=1.4, label=STREAM_NAMES[s])

        ax.set_ylim(-1.1, 1.1)
        ax.set_xlim(t[0], t[-1])
        ax.set_ylabel("amp / prob")
        ax.set_title(f"{cat}  |  snr~{snr:.1f} dB  |  row {row}", fontsize=9)

    axes[-1, 0].set_xlabel("time (s)")
    axes[0, 0].legend(loc="upper right", fontsize=7, ncol=4)
    fig.suptitle(f"Phase 1 sanity check — {args.split} split ({n} traces)")
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"saved {args.out}")
    print("split sizes:", {k: len(v) for k, v in split_idx.items()})


if __name__ == "__main__":
    main()
