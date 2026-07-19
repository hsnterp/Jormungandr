"""Causality of the forward-only (streaming) preprocessing."""

import numpy as np
import pytest

from seismic_edge_picker.preprocessing import (
    CausalPreprocessor,
    causal_bandpass,
    causal_demean,
    causal_normalize,
    preprocess_waveform_causal,
)
from seismic_edge_picker.config import load_config
import os

CFG = load_config(os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"))
FS = 100.0
N = 3000


def _rand(seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((3, N)).astype(np.float64)


@pytest.mark.parametrize(
    "fn",
    [
        lambda x: causal_demean(x),
        lambda x: causal_bandpass(x, 1.0, 45.0, FS, 4),
        lambda x: causal_normalize(x, "std"),
        lambda x: causal_normalize(x, "max"),
    ],
)
@pytest.mark.parametrize("i", [500, 1500, 2500])
def test_preprocessing_is_causal(fn, i):
    a = _rand(0)
    b = a.copy()
    b[:, i:] += _rand(1)[:, i:]  # perturb only t >= i
    ya, yb = fn(a), fn(b)
    # output before i must be unchanged by future input
    assert np.allclose(ya[:, :i], yb[:, :i], atol=1e-6), "future leaked into past"


def test_no_nans_and_shape():
    x = _rand(2)
    y = preprocess_waveform_causal(x, CFG)
    assert y.shape == x.shape
    assert np.isfinite(y).all()


def test_warmstart_not_zero_jump():
    """First filtered sample should sit near background, not spike from zero."""
    x = _rand(3) * 0.1
    x += 5.0  # strong DC offset / background level
    y = causal_bandpass(x, 1.0, 45.0, FS, 4)
    # warm-started state => bounded first sample (no huge zero->offset transient)
    assert np.abs(y[:, 0]).max() < 3.0


def test_stateful_preprocessor_matches_one_shot_with_chunks():
    x = _rand(4)
    expected = preprocess_waveform_causal(x, CFG, warmup_samples=100)
    pre = CausalPreprocessor(CFG, warmup_samples=100)
    chunks = []
    pos = 0
    for size in [100, 17, 83, 211, 509, 997, 1183]:
        if pos >= x.shape[1]:
            break
        nxt = min(pos + size, x.shape[1])
        chunks.append(pre.process(x[:, pos:nxt]))
        pos = nxt
    if pos < x.shape[1]:
        chunks.append(pre.process(x[:, pos:]))
    actual = np.concatenate(chunks, axis=1)
    assert np.allclose(actual, expected, atol=1e-5)
