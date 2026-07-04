"""Reusable continuous-signal windowing, overlap merge, and postprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.signal import find_peaks


@dataclass(frozen=True)
class StreamingOutput:
    probabilities: np.ndarray
    window_starts: tuple[int, ...]
    coverage: np.ndarray


def window_starts(n_samples: int, window_samples: int, hop_samples: int) -> list[int]:
    """Return fixed-hop starts whose padded windows cover the complete signal."""
    if n_samples < 1:
        raise ValueError("signal must contain at least one sample")
    if window_samples < 1 or hop_samples < 1:
        raise ValueError("window_samples and hop_samples must be positive")
    if hop_samples > window_samples:
        raise ValueError("hop_samples cannot exceed window_samples (would leave gaps)")

    starts = [0]
    while starts[-1] + window_samples < n_samples:
        starts.append(starts[-1] + hop_samples)
    return starts


def stream_probabilities(
    signal: np.ndarray,
    predictor: Callable[[np.ndarray], np.ndarray],
    window_samples: int,
    hop_samples: int,
    batch_size: int = 1,
) -> StreamingOutput:
    """Infer fixed windows and uniformly average predictions in overlap regions."""
    x = np.asarray(signal, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"signal must have shape (channels, samples), got {x.shape}")
    if x.shape[0] != 3:
        raise ValueError(f"expected 3 channels, got {x.shape[0]}")
    if x.shape[1] < 1:
        raise ValueError("signal must contain at least one sample")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    starts = window_starts(x.shape[1], window_samples, hop_samples)
    total = np.zeros((3, x.shape[1]), dtype=np.float32)
    coverage = np.zeros(x.shape[1], dtype=np.int32)

    for offset in range(0, len(starts), batch_size):
        batch_starts = starts[offset:offset + batch_size]
        batch = np.zeros((len(batch_starts), 3, window_samples), dtype=np.float32)
        valid_lengths = []
        for i, start in enumerate(batch_starts):
            valid = min(window_samples, x.shape[1] - start)
            batch[i, :, :valid] = x[:, start:start + valid]
            valid_lengths.append(valid)

        prediction = np.asarray(predictor(batch), dtype=np.float32)
        expected = (len(batch_starts), 3, window_samples)
        if prediction.shape != expected:
            raise ValueError(
                f"predictor returned shape {prediction.shape}, expected {expected}"
            )
        if not np.isfinite(prediction).all():
            raise ValueError("predictor returned non-finite probabilities")

        for i, (start, valid) in enumerate(zip(batch_starts, valid_lengths)):
            total[:, start:start + valid] += prediction[i, :, :valid]
            coverage[start:start + valid] += 1

    if np.any(coverage == 0):
        raise RuntimeError("internal error: streaming windows left uncovered samples")
    probabilities = total / coverage[None, :]
    return StreamingOutput(probabilities, tuple(starts), coverage)


def extract_events(
    detection: np.ndarray,
    sampling_rate: float,
    threshold: float = 0.80,
    min_duration_ms: float = 10.0,
    merge_gap_s: float = 0.5,
) -> list[dict]:
    """Extract and coalesce above-threshold event regions with relative times."""
    stream = np.asarray(detection, dtype=np.float32)
    if stream.ndim != 1:
        raise ValueError("detection stream must be one-dimensional")
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    if min_duration_ms < 0:
        raise ValueError("min_duration_ms cannot be negative")
    if merge_gap_s < 0:
        raise ValueError("merge_gap_s cannot be negative")

    min_samples = max(
        1, int(np.ceil(min_duration_ms * sampling_rate / 1000.0 - 1e-12))
    )
    mask = stream >= threshold
    changes = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)

    regions = [
        (int(start), int(end))
        for start, end in zip(starts, ends)
        if end - start >= min_samples
    ]
    merge_gap_samples = int(round(merge_gap_s * sampling_rate))
    merged = []
    for start, end in regions:
        if merged and start - merged[-1][1] <= merge_gap_samples:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    events = []
    for start, end in merged:
        peak_sample = int(start + np.argmax(stream[start:end]))
        events.append({
            "event_id": len(events) + 1,
            "start_sample": start,
            "end_sample_exclusive": end,
            "start_time_s": float(start / sampling_rate),
            "end_time_s": float(end / sampling_rate),
            "duration_s": float((end - start) / sampling_rate),
            "peak_sample": peak_sample,
            "peak_time_s": float(peak_sample / sampling_rate),
            "peak_probability": float(stream[peak_sample]),
        })
    return events


def extract_phase_picks(
    probabilities: np.ndarray,
    sampling_rate: float,
    p_threshold: float = 0.30,
    s_threshold: float = 0.30,
    min_distance_s: float = 1.0,
) -> list[dict]:
    """Extract all P/S probability peaks above their configurable thresholds."""
    streams = np.asarray(probabilities, dtype=np.float32)
    if streams.ndim != 2 or streams.shape[0] != 3:
        raise ValueError("probabilities must have shape (3, samples)")
    if sampling_rate <= 0 or min_distance_s < 0:
        raise ValueError("sampling_rate must be positive and min_distance_s non-negative")
    for name, value in (("p_threshold", p_threshold), ("s_threshold", s_threshold)):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1]")

    distance = max(1, int(round(min_distance_s * sampling_rate)))
    picks = []
    for phase, channel, threshold in (
        ("P", 1, p_threshold),
        ("S", 2, s_threshold),
    ):
        peaks, props = find_peaks(
            streams[channel], height=threshold, distance=distance
        )
        for sample, probability in zip(peaks, props["peak_heights"]):
            picks.append({
                "phase": phase,
                "sample": int(sample),
                "time_s": float(sample / sampling_rate),
                "probability": float(probability),
                "event_id": None,
            })
    return sorted(picks, key=lambda pick: (pick["sample"], pick["phase"]))


def associate_picks(
    picks: list[dict],
    events: list[dict],
    sampling_rate: float,
    margin_s: float = 1.0,
) -> list[dict]:
    """Associate each pick with the nearest compatible event, if one exists."""
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive")
    if margin_s < 0:
        raise ValueError("margin_s cannot be negative")
    margin = int(round(margin_s * sampling_rate))
    associated = []
    for original in picks:
        pick = dict(original)
        sample = pick["sample"]
        candidates = []
        for event in events:
            start = event["start_sample"]
            end = event["end_sample_exclusive"]
            if start - margin <= sample < end + margin:
                interval_distance = max(start - sample, 0, sample - (end - 1))
                peak_distance = abs(sample - event["peak_sample"])
                candidates.append(
                    (interval_distance, peak_distance, event["event_id"])
                )
        if candidates:
            pick["event_id"] = min(candidates)[2]
        associated.append(pick)
    return associated
