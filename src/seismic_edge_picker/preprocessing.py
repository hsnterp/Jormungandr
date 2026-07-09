"""Waveform preprocessing: demean, bandpass, per-trace normalization.

All functions operate on arrays of shape ``(n_channels, n_samples)`` and return
``float32`` arrays of the same shape. They are pure (no global state) so they
can be unit-tested without any dataset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi, sosfiltfilt


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


# ---------------------------------------------------------------------------
# CAUSAL (streaming) preprocessing
# ---------------------------------------------------------------------------
# The functions above read the whole window: `sosfiltfilt` is zero-phase
# (forward-backward) and `normalize`/`demean` use whole-window statistics -- all
# acausal. The functions below are strictly causal: output[t] uses only input
# samples at time <= t. They are sample-recursive, so a streaming runner that
# feeds a rolling buffer reproduces them exactly (see streaming.py + the
# streaming-vs-batch equivalence test). Per the lookahead-ablation OOD finding
# (zero tails collapse detection), the recursions are warm-started from real
# background statistics, never zeros.


def causal_demean(x: np.ndarray) -> np.ndarray:
    """Subtract the expanding running mean per channel (causal demean)."""
    x = np.asarray(x, dtype=np.float64)
    cnt = np.arange(1, x.shape[-1] + 1)
    run_mean = np.cumsum(x, axis=-1) / cnt
    return (x - run_mean).astype(np.float32)


def causal_bandpass(
    x: np.ndarray,
    freqmin: float,
    freqmax: float,
    sampling_rate: float,
    order: int = 4,
    warmup_samples: int | None = None,
) -> np.ndarray:
    """Forward-only Butterworth bandpass (single `sosfilt` pass, NOT zero-phase).

    The filter state ``zi`` is warm-started from the mean of the first
    ``warmup_samples`` (real background, default ~1 s) per channel so the initial
    transient reflects background level rather than a jump from zero.
    """
    x = np.asarray(x, dtype=np.float64)
    nyq = 0.5 * sampling_rate
    fmax = min(freqmax, nyq * 0.999)
    fmin = max(freqmin, 1e-6)
    sos = butter(order, [fmin, fmax], btype="band", fs=sampling_rate, output="sos")
    if warmup_samples is None:
        warmup_samples = int(sampling_rate)  # ~1 s of background
    warmup_samples = max(1, min(warmup_samples, x.shape[-1]))
    zi0 = sosfilt_zi(sos)  # (n_sections, 2), steady-state for unit step
    y = np.empty_like(x)
    for ch in range(x.shape[0]):
        bg = x[ch, :warmup_samples].mean()
        y[ch], _ = sosfilt(sos, x[ch], zi=zi0 * bg)
    return y.astype(np.float32)


def causal_normalize(x: np.ndarray, mode: str = "std", eps: float = 1e-8) -> np.ndarray:
    """Causal running normalization: at time t divide by the statistic of all
    samples (across channels) at time <= t. Mirrors `normalize` (single scalar
    across channels) but expanding rather than whole-window, so it is streamable.
    """
    x = np.asarray(x, dtype=np.float64)
    c = x.shape[0]
    if mode == "std":
        s = np.cumsum(x.sum(axis=0))
        q = np.cumsum((x ** 2).sum(axis=0))
        cnt = c * np.arange(1, x.shape[-1] + 1)
        mean = s / cnt
        var = np.clip(q / cnt - mean ** 2, 0.0, None)
        scale = np.sqrt(var)
    elif mode == "max":
        scale = np.maximum.accumulate(np.abs(x).max(axis=0))
    else:
        raise ValueError(f"unknown normalization mode: {mode!r}")
    return (x / (scale[None, :] + eps)).astype(np.float32)


@dataclass
class CausalPreprocessor:
    """Stateful forward-only preprocessor for chunked streaming.

    The state is initialized from real samples from the beginning of the stream
    (or an explicit real background buffer supplied by the caller). It never
    fabricates a zero-filled warmup, matching the right-context ablation finding
    that zero tails are out of distribution for this picker.
    """

    cfg: object
    warmup_samples: int | None = None
    initial_background: np.ndarray | None = None

    def __post_init__(self):
        pp = self.cfg.data.preprocess
        fs = float(self.cfg.data.sampling_rate)
        bp = pp.bandpass
        nyq = 0.5 * fs
        fmax = min(float(bp.freqmax), nyq * 0.999)
        fmin = max(float(bp.freqmin), 1e-6)
        self._sos = (
            butter(int(bp.order), [fmin, fmax], btype="band", fs=fs, output="sos")
            if bool(bp.apply)
            else None
        )
        self._zi_unit = sosfilt_zi(self._sos) if self._sos is not None else None
        self._zi = None
        self._demean_count = 0
        self._demean_sum = None
        self._norm_count = 0
        self._norm_sum = 0.0
        self._norm_sumsq = 0.0
        self._norm_absmax = 0.0
        if self.warmup_samples is None:
            self.warmup_samples = int(fs)
        if self.initial_background is not None:
            bg = np.asarray(self.initial_background, dtype=np.float64)
            if bg.ndim != 2 or bg.shape[0] != int(self.cfg.data.n_channels):
                raise ValueError(
                    "initial_background must have shape "
                    f"({self.cfg.data.n_channels}, samples)"
                )
            if bg.shape[1] < 1:
                raise ValueError("initial_background must contain real samples")

    def _causal_demean_chunk(self, x: np.ndarray) -> np.ndarray:
        if self._demean_sum is None:
            self._demean_sum = np.zeros(x.shape[0], dtype=np.float64)
        y = np.empty_like(x, dtype=np.float64)
        for t in range(x.shape[1]):
            self._demean_sum += x[:, t]
            self._demean_count += 1
            y[:, t] = x[:, t] - self._demean_sum / self._demean_count
        return y

    def _init_bandpass_state(self, x: np.ndarray) -> None:
        if self._sos is None or self._zi is not None:
            return
        warm = (
            np.asarray(self.initial_background, dtype=np.float64)
            if self.initial_background is not None
            else x
        )
        n = max(1, min(int(self.warmup_samples), warm.shape[1]))
        bg = warm[:, :n].mean(axis=1)
        self._zi = np.stack([self._zi_unit * b for b in bg], axis=0)

    def _causal_bandpass_chunk(self, x: np.ndarray) -> np.ndarray:
        if self._sos is None:
            return x
        self._init_bandpass_state(x)
        y = np.empty_like(x, dtype=np.float64)
        for ch in range(x.shape[0]):
            y[ch], self._zi[ch] = sosfilt(self._sos, x[ch], zi=self._zi[ch])
        return y

    def _causal_normalize_chunk(self, x: np.ndarray) -> np.ndarray:
        pp = self.cfg.data.preprocess
        mode = pp.normalization
        eps = float(pp.eps)
        y = np.empty_like(x, dtype=np.float64)
        for t in range(x.shape[1]):
            col = x[:, t]
            if mode == "std":
                self._norm_count += col.size
                self._norm_sum += float(col.sum())
                self._norm_sumsq += float((col ** 2).sum())
                mean = self._norm_sum / self._norm_count
                var = max(self._norm_sumsq / self._norm_count - mean ** 2, 0.0)
                scale = np.sqrt(var)
            elif mode == "max":
                self._norm_absmax = max(self._norm_absmax, float(np.abs(col).max()))
                scale = self._norm_absmax
            else:
                raise ValueError(f"unknown normalization mode: {mode!r}")
            y[:, t] = col / (scale + eps)
        return y

    def process(self, x: np.ndarray) -> np.ndarray:
        """Process one chunk shaped ``(channels, samples)`` and update state."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"chunk must have shape (channels, samples), got {x.shape}")
        if x.shape[0] != int(self.cfg.data.n_channels):
            raise ValueError(f"expected {self.cfg.data.n_channels} channels, got {x.shape[0]}")
        if x.shape[1] == 0:
            return x.astype(np.float32)
        pp = self.cfg.data.preprocess
        if getattr(pp, "demean", True):
            x = self._causal_demean_chunk(x)
        x = self._causal_bandpass_chunk(x)
        x = self._causal_normalize_chunk(x)
        return x.astype(np.float32)


def preprocess_waveform_causal(x: np.ndarray, cfg, warmup_samples: int | None = None) -> np.ndarray:
    """Strictly-causal counterpart of `preprocess_waveform` (offline scoring).

    demean -> bandpass -> normalize, each replaced by its causal running variant.
    Used to score the causal model on fixed windows on the same footing that the
    streaming runner (streaming.py) produces online.
    """
    return CausalPreprocessor(cfg, warmup_samples=warmup_samples).process(x)


def causal_filter_only(x: np.ndarray, cfg, warmup_samples: int | None = None) -> np.ndarray:
    """Causal demean + bandpass, WITHOUT normalization (train-time path)."""
    pp = cfg.data.preprocess
    if getattr(pp, "demean", True):
        x = causal_demean(x)
    if pp.bandpass.apply:
        x = causal_bandpass(
            x,
            pp.bandpass.freqmin,
            pp.bandpass.freqmax,
            cfg.data.sampling_rate,
            pp.bandpass.order,
            warmup_samples=warmup_samples,
        )
    return np.asarray(x, dtype=np.float32)


def causal_normalize_only(x: np.ndarray, cfg) -> np.ndarray:
    """Causal normalization helper matching ``normalize``'s call site."""
    return causal_normalize(
        x,
        cfg.data.preprocess.normalization,
        cfg.data.preprocess.eps,
    )


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
