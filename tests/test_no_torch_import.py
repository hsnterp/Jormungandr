"""seismic_edge_picker.config/streaming must import on ONNX-only hosts
(e.g. Raspberry Pi) with no PyTorch installed. Simulate that by making
``import torch`` fail and reloading the package fresh, rather than requiring
an actual torch-free environment to run this test."""

import sys

import pytest


@pytest.fixture
def no_torch(monkeypatch):
    for name in list(sys.modules):
        if name == "torch" or name.startswith("torch.") \
                or name == "seismic_edge_picker" or name.startswith("seismic_edge_picker."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "torch", None)
    yield
    monkeypatch.delitem(sys.modules, "seismic_edge_picker", raising=False)


def test_package_import_without_torch(no_torch):
    import seismic_edge_picker

    assert seismic_edge_picker._has_torch is False
    assert seismic_edge_picker.SeismicUNet is None
    assert seismic_edge_picker.build_model is None


def test_config_load_without_torch(no_torch):
    import os

    from seismic_edge_picker.config import load_config

    config_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "default.yaml"
    )
    cfg = load_config(config_path)
    assert cfg.data.sampling_rate == 100


def test_streaming_without_torch(no_torch):
    from seismic_edge_picker.streaming import window_starts

    assert window_starts(6000, 6000, 3000) == [0]


def test_model_import_without_torch_raises(no_torch):
    with pytest.raises(ImportError):
        import seismic_edge_picker.model  # noqa: F401
