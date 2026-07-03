import os

import pytest

from seismic_edge_picker.config import load_config

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")


@pytest.fixture
def cfg():
    return load_config(CONFIG_PATH)
