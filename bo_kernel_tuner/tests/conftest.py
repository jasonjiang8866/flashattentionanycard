import pytest

from bo_kernel_tuner.tuner.hardware import Hardware


@pytest.fixture
def hardware():
    return Hardware("CPU test double; not a GPU measurement", (8, 6), 82, 24 * 1024**3, 99 * 1024)
