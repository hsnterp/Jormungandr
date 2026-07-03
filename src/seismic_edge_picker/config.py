"""Config loading.

Loads the YAML config into a nested attribute-accessible namespace while also
retaining the raw dict (``cfg._raw``) for serialization / logging.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import yaml


def _to_ns(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj


def _to_dict(obj: Any) -> Any:
    if isinstance(obj, SimpleNamespace):
        return {k: _to_dict(v) for k, v in vars(obj).items() if not k.startswith("_")}
    if isinstance(obj, list):
        return [_to_dict(v) for v in obj]
    return obj


def load_config(path: str) -> SimpleNamespace:
    """Load a YAML config file into a nested namespace.

    Access nested values with dot notation, e.g. ``cfg.model.encoder_channels``.
    The original dict is available as ``cfg._raw``.
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    ns = _to_ns(raw)
    ns._raw = raw
    return ns


def config_to_dict(cfg: SimpleNamespace) -> dict:
    """Recursively convert a config namespace back to a plain dict."""
    return _to_dict(cfg)
