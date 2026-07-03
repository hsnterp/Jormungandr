"""seismic-edge-picker: compact, edge-deployable seismic event detector and
phase picker (STEAD-trained, EQTransformer-distilled, INT8-ready).

Public API surface used across phases. Heavy / data-dependent modules
(``dataset``) are imported lazily so that ``import seismic_edge_picker`` never
triggers a SeisBench dataset download.
"""

from .config import load_config
from .preprocessing import demean, bandpass, normalize, preprocess_waveform
from .labels import gaussian_bump, build_label_mask
from .model import SeismicUNet, build_model, count_parameters, count_macs

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
