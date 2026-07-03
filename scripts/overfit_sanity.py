#!/usr/bin/env python
"""Phase 3 sanity check: overfit the model on a tiny fixed set (default 100
traces) and show the loss curve. This confirms the model + loss can drive the
weighted BCE down toward its target-entropy floor before committing to full
training.

Augmentation is DISABLED here (fixed targets) so memorization is possible.

    python scripts/overfit_sanity.py --config configs/default.yaml

Outputs:
    outputs/overfit_loss.png      loss curve (total + per-stream vs floor)
    outputs/overfit_example.png   predicted vs target masks for one trace
"""

import argparse
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from seismic_edge_picker.config import load_config             # noqa: E402
from seismic_edge_picker.dataset import build_datasets, SeismicDataset  # noqa: E402
from seismic_edge_picker.model import build_model, count_parameters      # noqa: E402
from seismic_edge_picker.losses import (                       # noqa: E402
    stream_weight_tensor, weighted_bce_loss, bce_floor,
)

STREAM_NAMES = ["detection", "P", "S"]
STREAM_COLORS = ["tab:green", "tab:blue", "tab:red"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n", type=int, default=None, help="override overfit.n_traces")
    ap.add_argument("--epochs", type=int, default=None, help="override overfit.epochs")
    ap.add_argument("--lr", type=float, default=None, help="override overfit.lr")
    ap.add_argument("--out-loss", default="outputs/overfit_loss.png")
    ap.add_argument("--out-example", default="outputs/overfit_example.png")
    args = ap.parse_args()

    cfg = load_config(args.config)
    n = args.n or cfg.train.overfit.n_traces
    epochs = args.epochs or cfg.train.overfit.epochs
    lr = args.lr or getattr(cfg.train.overfit, "lr", None) or cfg.train.lr

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = "cuda" if (cfg.train.device == "cuda" and torch.cuda.is_available()) else "cpu"

    # Build datasets (loads STEAD), then a NO-AUG dataset over the first n train rows.
    datasets, split_idx = build_datasets(cfg)
    ds = datasets["train"].ds
    rows = datasets["train"].rows[:n]
    fixed = SeismicDataset(ds, rows, cfg, train=False, seed=cfg.seed)

    print(f"device={device}  overfit n={len(fixed)}  epochs={epochs}")
    t0 = time.time()
    X = torch.stack([fixed[i][0] for i in range(len(fixed))]).to(device)
    Y = torch.stack([fixed[i][1] for i in range(len(fixed))]).to(device)
    print(f"preloaded {tuple(X.shape)} in {time.time()-t0:.1f}s")

    model = build_model(cfg).to(device)
    print(f"params: {count_parameters(model):,}")
    weights = stream_weight_tensor(cfg, device)
    floor, floor_ps = bce_floor(Y, weights)
    print(f"weighted BCE floor (target entropy): {floor.item():.5f}")

    # weight_decay 0 for a clean capacity test
    print(f"lr={lr}")
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    clip = cfg.train.grad_clip

    hist_total, hist_ps = [], []
    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        pred = model(X)
        loss, per = weighted_bce_loss(pred, Y, weights)
        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        hist_total.append(loss.item())
        hist_ps.append(per.detach().cpu().numpy())
        if ep % max(1, epochs // 10) == 0 or ep == epochs - 1:
            print(f"  epoch {ep:4d}  loss {loss.item():.5f}  "
                  f"[det {per[0]:.4f} P {per[1]:.4f} S {per[2]:.4f}]")

    hist_ps = np.array(hist_ps)
    print(f"final loss {hist_total[-1]:.5f}  (floor {floor.item():.5f})  "
          f"in {time.time()-t0:.1f}s")

    # ---- loss curve ----
    os.makedirs(os.path.dirname(args.out_loss) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(hist_total, color="k", lw=2, label="weighted total")
    for s in range(3):
        ax.plot(hist_ps[:, s], color=STREAM_COLORS[s], lw=1,
                label=f"{STREAM_NAMES[s]} BCE")
    ax.axhline(floor.item(), color="0.6", ls="--", lw=1, label="target-entropy floor")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("BCE loss (log)")
    ax.set_title(f"Overfit sanity — {len(fixed)} traces, {epochs} epochs")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out_loss, dpi=120)
    print(f"saved {args.out_loss}")

    # ---- predicted vs target for one earthquake example ----
    model.eval()
    with torch.no_grad():
        pred = model(X).cpu().numpy()
    Yc = Y.cpu().numpy()
    # pick an example that actually has picks (non-zero target)
    ex = next((i for i in range(len(fixed)) if Yc[i].max() > 0.5), 0)
    t = np.arange(cfg.data.window_samples) / cfg.data.sampling_rate
    fig, axes = plt.subplots(3, 1, figsize=(11, 6), sharex=True)
    for s, ax in enumerate(axes):
        ax.plot(t, Yc[ex, s], color=STREAM_COLORS[s], lw=1.6, label="target")
        ax.plot(t, pred[ex, s], color="k", lw=1.0, ls="--", label="pred")
        ax.set_ylim(-0.05, 1.1)
        ax.set_ylabel(STREAM_NAMES[s])
        ax.legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("time (s)")
    axes[0].set_title(f"Overfit result — example trace {ex} (pred vs target)")
    fig.tight_layout()
    fig.savefig(args.out_example, dpi=120)
    print(f"saved {args.out_example}")


if __name__ == "__main__":
    main()
