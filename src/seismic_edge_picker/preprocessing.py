"""Waveform preprocessing: demean, bandpass, per-trace normalization.

All functions operate on arrays of shape ``(n_channels, n_samples)`` and return
``float32`` arrays of the same shape. They are pure (no global state) so they
can be unit-tested without any dataset.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def demean(x: np.ndarray) -> np.ndarray:
    """Remove per-channel mean."""
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean(axis=-1, keepdims=True)).astype(np.float32)


def bandpass(
    x: np.ndarray,
    freqmin: float,
    freqmax: float,
    sampling_rate: float,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass along the time axis.

    ``freqmax`` is clamped just below Nyquist to keep the filter stable.
    """
    x = np.asarray(x, dtype=np.float64)
    nyq = 0.5 * sampling_rate
    fmax = min(freqmax, nyq * 0.999)
    fmin = max(freqmin, 1e-6)
    sos = butter(order, [fmin, fmax], btype="band", fs=sampling_rate, output="sos")
    # sosfiltfilt needs signal length > padlen; STEAD windows (6000) are ample.
    y = sosfiltfilt(sos, x, axis=-1)
    return y.astype(np.float32)


def normalize(x: np.ndarray, mode: str = "std", eps: float = 1e-8) -> np.ndarray:
    """Per-trace normalization.

    - ``std``: divide the whole 3-channel trace by its global standard deviation.
    - ``max``: divide by the global max absolute value.

    A single scalar is used across channels so relative channel amplitudes
    (which carry P/S polarization information) are preserved.
    """
    x = np.asarray(x, dtype=np.float64)
    if mode == "std":
        scale = x.std()
    elif mode == "max":
        scale = np.abs(x).max()
    else:
        raise ValueError(f"unknown normalization mode: {mode!r}")
    return (x / (scale + eps)).astype(np.float32)


def preprocess_waveform(x: np.ndarray, cfg) -> np.ndarray:
    """Apply the full preprocessing chain per ``cfg.data.preprocess``.

    NOTE on ordering: this performs demean + bandpass + normalize. When
    training augmentations that depend on physical amplitude (noise mixing) are
    used, the dataset applies demean+bandpass first, augments, then normalizes
    last — see ``dataset.py``. This convenience function is the eval-time path.
    """
    pp = cfg.data.preprocess
    if getattr(pp, "demean", True):
        x = demean(x)
    if pp.bandpass.apply:
        x = bandpass(
            x,
            pp.bandpass.freqmin,
            pp.bandpass.freqmax,
            cfg.data.sampling_rate,
            pp.bandpass.order,
        )
    x = normalize(x, pp.normalization, pp.eps)
    return x


def filter_only(x: np.ndarray, cfg) -> np.ndarray:
    """Demean + bandpass, WITHOUT normalization (train-time pre-augment path)."""
    pp = cfg.data.preprocess
    if getattr(pp, "demean", True):
        x = demean(x)
    if pp.bandpass.apply:
        x = bandpass(
            x,
            pp.bandpass.freqmin,
            pp.bandpass.freqmax,
            cfg.data.sampling_rate,
            pp.bandpass.order,
        )
    return np.asarray(x, dtype=np.float32)
