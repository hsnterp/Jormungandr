import os

import numpy as np
import torch

from seismic_edge_picker.streaming import (
    causal_stream_probabilities,
    associate_picks,
    extract_events,
    extract_phase_picks,
    stream_probabilities,
    window_starts,
)


def test_window_starts_fixed_hop_and_padded_tail():
    assert window_starts(6000, 6000, 3000) == [0]
    assert window_starts(7000, 6000, 3000) == [0, 3000]
    assert window_starts(12000, 6000, 3000) == [0, 3000, 6000]


def test_overlap_merge_reconstructs_identity_predictor():
    rng = np.random.default_rng(0)
    signal = rng.standard_normal((3, 137)).astype(np.float32)
    result = stream_probabilities(
        signal, lambda batch: batch, window_samples=60, hop_samples=30,
        batch_size=2,
    )
    assert result.probabilities.shape == signal.shape
    assert np.allclose(result.probabilities, signal)
    assert np.all(result.coverage >= 1)


def test_short_signal_is_padded_and_trimmed():
    signal = np.ones((3, 20), dtype=np.float32)
    result = stream_probabilities(
        signal, lambda batch: batch, window_samples=60, hop_samples=30
    )
    assert result.window_starts == (0,)
    assert result.probabilities.shape == (3, 20)
    assert np.array_equal(result.probabilities, signal)


def test_event_min_duration_and_timestamps():
    detection = np.zeros(100, dtype=np.float32)
    detection[10:11] = 0.95
    detection[30:33] = [0.81, 0.99, 0.85]
    events = extract_events(
        detection, sampling_rate=100, threshold=0.8, min_duration_ms=20
    )
    assert len(events) == 1
    assert events[0]["start_sample"] == 30
    assert events[0]["end_sample_exclusive"] == 33
    assert events[0]["peak_sample"] == 31
    assert events[0]["peak_time_s"] == 0.31


def test_nearby_event_fragments_are_coalesced():
    detection = np.zeros(100, dtype=np.float32)
    detection[10:20] = 0.9
    detection[22:30] = 0.95
    events = extract_events(
        detection, sampling_rate=100, threshold=0.8, min_duration_ms=10,
        merge_gap_s=0.05,
    )
    assert len(events) == 1
    assert events[0]["start_sample"] == 10
    assert events[0]["end_sample_exclusive"] == 30


def test_phase_picks_and_event_association():
    probabilities = np.zeros((3, 200), dtype=np.float32)
    probabilities[1, 45] = 0.9
    probabilities[2, 70] = 0.8
    events = extract_events(
        np.r_[np.zeros(40), np.ones(50), np.zeros(110)],
        sampling_rate=100, threshold=0.8, min_duration_ms=10,
    )
    picks = extract_phase_picks(
        probabilities, sampling_rate=100, p_threshold=0.3, s_threshold=0.3
    )
    picks = associate_picks(picks, events, sampling_rate=100, margin_s=0)
    assert [(p["phase"], p["sample"], p["event_id"]) for p in picks] == [
        ("P", 45, 1),
        ("S", 70, 1),
    ]


def test_causal_streaming_matches_fixed_batch_outputs():
    from seismic_edge_picker.config import load_config
    from seismic_edge_picker.model import SeismicUNet
    from seismic_edge_picker.preprocessing import (
        CausalPreprocessor,
        preprocess_waveform_causal,
    )

    cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"))
    cfg.model.causal = True
    rng = np.random.default_rng(123)
    signal = rng.standard_normal((3, cfg.data.window_samples)).astype(np.float32)
    model = SeismicUNet(causal=True).eval()

    def predict(batch):
        with torch.inference_mode():
            return model(torch.from_numpy(batch)).numpy()

    batch_x = preprocess_waveform_causal(signal, cfg, warmup_samples=100)
    expected = predict(batch_x[None])[0]
    streaming = causal_stream_probabilities(
        signal,
        predict,
        CausalPreprocessor(cfg, warmup_samples=100),
        chunk_samples=500,
    )
    assert streaming.probabilities.shape == expected.shape
    assert np.allclose(streaming.probabilities, expected, atol=1e-6)
    assert streaming.window_starts[0] == 0
    assert np.all(streaming.coverage == 1)
