"""Training-only augmentations.

Each augmentation operates on a waveform ``(n_channels, n_samples)`` plus an
``arrivals`` dict ``{"p": sample_or_None, "s": sample_or_None,
"coda": sample_or_None}``. Augmentations that move the signal in time return
updated arrival indices so labels can be regenerated afterwards.

Design: augment the *filtered but not-yet-normalized* waveform, then normalize
last, so that noise-mixing SNR is physically meaningful.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

Arrivals = Dict[str, Optional[int]]


def _shift_arrivals(arrivals: Arrivals, delta: int, length: int) -> Arrivals:
    out: Arrivals = {}
    for k, v in arrivals.items():
        if v is None:
            out[k] = None
            continue
        nv = int(v) + delta
        out[k] = nv if 0 <= nv < length else None
    return out


def random_window_shift(
    x: np.ndarray, arrivals: Arrivals, max_shift_s: float, sampling_rate: float, rng
) -> tuple[np.ndarray, Arrivals]:
    """Translate the trace by a random integer offset (zero-filled), so
    arrivals can land anywhere in the window.

    Positive delta shifts the signal to the right (later in time). The shift is
    bounded so that P/S arrivals stay inside the window (they should move
    *within* the window, not off it — otherwise we'd zero the detection stream
    while a pick remains, an inconsistent target). Coda may still shift out and
    is handled by the detection fallback.
    """
    length = x.shape[-1]
    max_shift = int(round(max_shift_s * sampling_rate))
    if max_shift <= 0:
        return x, arrivals
    lo, hi = -max_shift, max_shift
    picks = [arrivals[k] for k in ("p", "s") if arrivals.get(k) is not None]
    if picks:
        lo = max(lo, -min(picks))                 # keep earliest pick >= 0
        hi = min(hi, (length - 1) - max(picks))   # keep latest pick <= length-1
    if hi < lo:
        return x, arrivals
    delta = int(rng.integers(lo, hi + 1))
    if delta == 0:
        return x, arrivals
    out = np.zeros_like(x)
    if delta > 0:
        out[:, delta:] = x[:, : length - delta]
    else:
        out[:, : length + delta] = x[:, -delta:]
    return out, _shift_arrivals(arrivals, delta, length)


def mix_noise(
    signal: np.ndarray, noise: np.ndarray, target_snr_db: float, eps: float = 1e-12
) -> np.ndarray:
    """Additively mix a real noise trace into ``signal`` at ``target_snr_db``.

    SNR is defined over full-trace power. The noise is scaled so that
    ``10*log10(P_signal / P_noise_scaled) == target_snr_db``.
    """
    sig_p = float(np.mean(signal.astype(np.float64) ** 2))
    noise_p = float(np.mean(noise.astype(np.float64) ** 2))
    if noise_p <= eps or sig_p <= eps:
        return signal.astype(np.float32)
    snr_lin = 10.0 ** (target_snr_db / 10.0)
    scale = np.sqrt(sig_p / (snr_lin * noise_p))
    return (signal + scale * noise).astype(np.float32)


def channel_dropout(
    x: np.ndarray, max_channels: int, rng
) -> np.ndarray:
    """Zero out up to ``max_channels`` randomly chosen channels."""
    x = x.copy()
    n_ch = x.shape[0]
    k = int(rng.integers(1, max_channels + 1)) if max_channels >= 1 else 0
    k = min(k, n_ch - 1)  # never drop every channel
    if k <= 0:
        return x
    drop = rng.choice(n_ch, size=k, replace=False)
    x[drop] = 0.0
    return x


def apply_augmentations(
    x: np.ndarray,
    arrivals: Arrivals,
    cfg,
    rng,
    noise_sampler=None,
) -> tuple[np.ndarray, Arrivals]:
    """Apply the configured augmentation stack (train only).

    ``noise_sampler`` is an optional callable returning a filtered noise
    waveform ``(n_channels, n_samples)`` for additive mixing. If ``None``,
    noise mixing is skipped.
    """
    aug = cfg.augment
    sr = cfg.data.sampling_rate

    if aug.window_shift.enabled:
        x, arrivals = random_window_shift(
            x, arrivals, aug.window_shift.max_shift_s, sr, rng
        )

    if aug.noise_mixing.enabled and noise_sampler is not None:
        if rng.random() < aug.noise_mixing.prob:
            lo, hi = aug.noise_mixing.target_snr_db
            snr = float(rng.uniform(lo, hi))
            noise = noise_sampler()
            if noise is not None:
                x = mix_noise(x, noise, snr)

    if aug.channel_dropout.enabled:
        if rng.random() < aug.channel_dropout.prob:
            x = channel_dropout(x, aug.channel_dropout.max_channels, rng)

    return x, arrivals
