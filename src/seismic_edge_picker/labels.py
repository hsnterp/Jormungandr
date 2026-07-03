"""Label-mask generation from P/S arrival sample indices.

Target tensor is shape ``(3, n_samples)``:
    channel 0 — detection: 1.0 across the event window (P -> coda), else 0.
    channel 1 — P pick:    Gaussian bump centered on the P arrival.
    channel 2 — S pick:    Gaussian bump centered on the S arrival.

Noise traces (no arrivals) produce an all-zero target.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _clean_sample(v) -> Optional[int]:
    """Return an int sample index, or None if missing/NaN."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if np.isnan(f):
        return None
    return int(round(f))


def gaussian_bump(length: int, center: float, sigma_samples: float) -> np.ndarray:
    """Unit-height Gaussian centered at ``center`` (in samples)."""
    t = np.arange(length, dtype=np.float64)
    bump = np.exp(-0.5 * ((t - center) / max(sigma_samples, 1e-6)) ** 2)
    return bump.astype(np.float32)


def build_label_mask(
    length: int,
    p_sample: Optional[float],
    s_sample: Optional[float],
    coda_end_sample: Optional[float] = None,
    sampling_rate: float = 100.0,
    pick_sigma_s: float = 0.25,
    detection_source: str = "coda",
    detection_fallback_s: float = 10.0,
    detection_pad_s: float = 0.0,
) -> np.ndarray:
    """Build the (3, length) supervised target.

    Parameters mirror ``cfg.data.labels``. Any arrival given as ``None`` or NaN
    is simply omitted (its stream stays zero), which yields the correct
    all-zero target for noise traces.
    """
    p = _clean_sample(p_sample)
    s = _clean_sample(s_sample)
    coda = _clean_sample(coda_end_sample)

    labels = np.zeros((3, length), dtype=np.float32)
    sigma_samples = pick_sigma_s * sampling_rate
    pad = int(round(detection_pad_s * sampling_rate))
    fallback = int(round(detection_fallback_s * sampling_rate))

    # --- detection window (channel 0) ---
    if p is not None:
        start = p
        if detection_source == "coda" and coda is not None:
            end = coda
        elif s is not None:
            end = s + fallback
        else:
            end = p + fallback
        start = max(0, start - pad)
        end = min(length, end + pad)
        if end > start:
            labels[0, start:end] = 1.0

    # --- P pick (channel 1) ---
    if p is not None and 0 <= p < length:
        labels[1] = gaussian_bump(length, p, sigma_samples)

    # --- S pick (channel 2) ---
    if s is not None and 0 <= s < length:
        labels[2] = gaussian_bump(length, s, sigma_samples)

    return labels


def build_label_mask_from_cfg(
    length: int,
    p_sample: Optional[float],
    s_sample: Optional[float],
    coda_end_sample: Optional[float],
    cfg,
) -> np.ndarray:
    """Convenience wrapper reading parameters from a config namespace."""
    lb = cfg.data.labels
    return build_label_mask(
        length=length,
        p_sample=p_sample,
        s_sample=s_sample,
        coda_end_sample=coda_end_sample,
        sampling_rate=cfg.data.sampling_rate,
        pick_sigma_s=lb.pick_sigma_s,
        detection_source=lb.detection_source,
        detection_fallback_s=lb.detection_fallback_s,
        detection_pad_s=lb.detection_pad_s,
    )
