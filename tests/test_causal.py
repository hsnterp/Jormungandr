"""Causality guarantees for the causal SeismicUNet variant.

The core contract: with ``causal=True, lookahead=L`` an output sample at time t
must depend ONLY on input samples at time <= t + L. We test this the hard way:
perturb the input from index i onward and assert the output is bit-identical for
every t < i - L. Any future leakage fails loudly.

We also guard that (a) the acausal default is unchanged and (b) an acausal model
DOES leak (so the test is meaningful), and (c) an acausal checkpoint warm-starts
a causal model with strict=True (padding is not a learned parameter).
"""

import os

import pytest
import torch

from seismic_edge_picker.model import SeismicUNet, build_model, count_parameters
from seismic_edge_picker.config import load_config

CFG = load_config(os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"))
T = 6000


def _max_future_leak(model, L, i, seed=0):
    """Max abs difference over the must-be-identical region t < i - L when the
    input is perturbed for t >= i. Returns None if the region is empty."""
    model.eval()
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(1, 3, T, generator=g)
    b = a.clone()
    b[..., i:] = torch.randn(1, 3, T - i, generator=g)  # differ only for t >= i
    with torch.no_grad():
        oa, ob = model(a), model(b)
    cut = i - L
    if cut <= 0:
        return None
    return (oa[..., :cut] - ob[..., :cut]).abs().max().item()


@pytest.mark.parametrize("L", [0, 25, 50])
@pytest.mark.parametrize("i", [1000, 2500, 4000, 5000, 5900])
def test_no_future_leakage(L, i):
    model = SeismicUNet(causal=True, lookahead=L).eval()
    leak = _max_future_leak(model, L, i)
    if leak is None:
        pytest.skip("region t < i-L is empty")
    assert leak == 0.0, f"future leakage {leak:.3e} at i={i}, L={L}"


def test_acausal_model_leaks():
    """Sanity: the shipped (acausal) model must leak, else the test is vacuous."""
    model = SeismicUNet(causal=False).eval()
    leak = _max_future_leak(model, 0, 2500)
    assert leak is not None and leak > 0.0


def test_causal_output_contract():
    model = SeismicUNet(causal=True).eval()
    x = torch.randn(2, 3, T)
    y = model(x)
    assert y.shape == (2, 3, T)
    assert y.min() >= 0.0 and y.max() <= 1.0


def test_causal_param_count_matches_acausal():
    """Causal flag adds no parameters (only changes padding)."""
    assert count_parameters(SeismicUNet(causal=True)) == count_parameters(
        SeismicUNet(causal=False)
    )


def test_acausal_checkpoint_warmstarts_causal():
    """An acausal checkpoint loads into a causal model with strict=True."""
    ckpt_path = os.path.join(
        os.path.dirname(__file__), "..", "checkpoints", "stage2_distill", "best.pt"
    )
    if not os.path.exists(ckpt_path):
        pytest.skip("stage2_distill checkpoint not present")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = SeismicUNet(causal=True, lookahead=0)
    res = model.load_state_dict(ck["model_state_dict"], strict=True)
    assert not res.missing_keys and not res.unexpected_keys


def test_build_model_causal_flag():
    """build_model honors cfg.model.causal / lookahead via getattr defaults."""
    from types import SimpleNamespace

    cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"))
    cfg.model.causal = True
    cfg.model.lookahead = 0
    model = build_model(cfg)
    assert model.causal is True
    leak = _max_future_leak(model, 0, 3000)
    assert leak == 0.0
