import torch

from seismic_edge_picker.config import load_config
from seismic_edge_picker.model import (
    build_model,
    count_parameters,
    count_macs,
    model_summary,
)

import os

CFG = load_config(os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"))


def test_forward_shape():
    model = build_model(CFG).eval()
    x = torch.zeros(2, 3, 6000)
    y = model(x)
    assert y.shape == (2, 3, 6000)


def test_output_is_probability():
    model = build_model(CFG).eval()
    x = torch.randn(2, 3, 6000)
    y = model(x)
    assert y.min() >= 0.0 and y.max() <= 1.0


def test_param_budget_under_300k():
    model = build_model(CFG)
    n = count_parameters(model)
    assert n < 300_000, f"model has {n} params, over 300k budget"


def test_no_forbidden_ops():
    model = build_model(CFG)
    for m in model.modules():
        name = type(m).__name__
        assert "LSTM" not in name and "GRU" not in name
        assert "Attention" not in name
        assert "ConvTranspose" not in name


def test_macs_positive():
    model = build_model(CFG)
    summ = model_summary(model, input_shape=(1, 3, 6000))
    assert summ["macs"] > 0
    assert summ["mflops"] > 0


def test_batch_independence():
    # per-sample segmentation: sample outputs must not depend on batch peers
    torch.manual_seed(0)
    model = build_model(CFG).eval()
    x = torch.randn(4, 3, 6000)
    with torch.no_grad():
        full = model(x)
        single = model(x[1:2])
    assert torch.allclose(full[1:2], single, atol=1e-5)
