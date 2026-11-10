import pytest

from src.common.config import load_config
from src.validation.samples import NOW, clean_sample, dirty_sample  # noqa: F401  (NOW re-exported)


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture
def clean_raw():
    return clean_sample()


@pytest.fixture
def dirty_raw():
    return dirty_sample()
