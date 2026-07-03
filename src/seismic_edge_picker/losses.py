"""Losses for the 3-stream segmentation target.

Per-stream binary cross-entropy with configurable weights (P and S weighted
~5x relative to detection). Targets are soft (Gaussian pick bumps), so BCE has a
non-zero floor equal to the target's binary entropy — ``bce_floor`` computes it
so overfit/training curves can be read against the achievable minimum.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def stream_weight_tensor(cfg, device="cpu") -> torch.Tensor:
    sw = cfg.train.loss.stream_weights
    return torch.tensor([sw.detection, sw.p, sw.s], dtype=torch.float32, device=device)


def weighted_bce_loss(pred, target, weights, eps: float = 1e-6):
    """Return (total_loss, per_stream_loss[3]).

    ``pred``/``target`` are (B, 3, N) in [0, 1]. BCE is averaged over batch and
    time per stream, then combined as a weighted mean across the 3 streams.
    """
    pred = pred.clamp(eps, 1.0 - eps)
    bce = F.binary_cross_entropy(pred, target, reduction="none")  # (B,3,N)
    per_stream = bce.mean(dim=(0, 2))                             # (3,)
    total = (per_stream * weights).sum() / weights.sum()
    return total, per_stream


def bce_floor(target, weights, eps: float = 1e-6):
    """Minimum achievable weighted BCE for these soft targets (= target entropy)."""
    t = target.clamp(eps, 1.0 - eps)
    return weighted_bce_loss(t, target, weights, eps)
