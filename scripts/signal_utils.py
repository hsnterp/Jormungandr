#!/usr/bin/env python
"""Dependency-light signal helpers shared by the edge demo and the latency curve.

These are pure-numpy and deliberately live outside ``causal_latency_curve.py`` so
that importing them (e.g. from ``demo_edge.py`` on a Raspberry Pi) does NOT pull in
torch/matplotlib. Keep this module free of heavy imports.
"""

from __future__ import annotations

import numpy as np


def arrival_wave(n, fs, onset, freq, amp, tau):
    """Damped sinusoid onset: a decaying tone starting at sample `onset`."""
    t = (np.arange(n) - onset) / fs
    env = np.where(t >= 0, np.exp(-t / tau), 0.0)
    return amp * env * np.sin(2 * np.pi * freq * t)


def sta_lta_ratio(raw, fs, sta_s, lta_s):
    """Classic short-term/long-term average energy ratio over multichannel `raw`."""
    energy = np.mean(np.asarray(raw, dtype=np.float64) ** 2, axis=0)
    nsta = max(1, int(round(sta_s * fs)))
    nlta = max(nsta + 1, int(round(lta_s * fs)))
    c = np.r_[0.0, np.cumsum(energy)]
    sta = np.zeros_like(energy)
    lta = np.zeros_like(energy)
    idx = np.arange(energy.size)
    sta_ok = idx + 1 >= nsta
    lta_ok = idx + 1 >= nlta
    sta[sta_ok] = (c[idx[sta_ok] + 1] - c[idx[sta_ok] + 1 - nsta]) / nsta
    lta[lta_ok] = (c[idx[lta_ok] + 1] - c[idx[lta_ok] + 1 - nlta]) / nlta
    ratio = sta / (lta + 1e-12)
    ratio[~lta_ok] = 0.0
    return ratio.astype(np.float32)
