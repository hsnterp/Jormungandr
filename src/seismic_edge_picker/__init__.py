"""Jormungandr: compact, edge-deployable seismic event detector and
phase picker (STEAD-trained, EQTransformer-distilled, INT8-ready).

Public API surface used across phases. Heavy / data-dependent modules
(``dataset``) are imported lazily so that ``import seismic_edge_picker`` never
triggers a SeisBench dataset download. PyTorch-dependent model utilities are
also optional: on ONNX-only inference hosts (e.g. Raspberry Pi) without
PyTorch installed, ``SeismicUNet``/``build_model``/``count_parameters``/
``count_macs`` are ``None`` instead of raising ImportError. Import
``seismic_edge_picker.model`` directly when PyTorch is available and training
utilities are needed.
"""

from .config import load_config
from .preprocessing import demean, bandpass, normalize, preprocess_waveform
from .labels import gaussian_bump, build_label_mask

try:
    from .model import SeismicUNet, build_model, count_parameters, count_macs
    _has_torch = True
except ImportError:
    SeismicUNet = None
    build_model = None
    count_parameters = None
    count_macs = None
    _has_torch = False

__all__ = [
    "load_config",
    "demean",
    "bandpass",
    "normalize",
    "preprocess_waveform",
    "gaussian_bump",
    "build_label_mask",
    "SeismicUNet",
    "build_model",
    "count_parameters",
    "count_macs",
]

__version__ = "0.1.0"
