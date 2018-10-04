import pytest
import numpy as np
from cogload import spatial_transform

def test_rad2deg_with_valid_input():
    deg = rad2deg(2 * np.pi)
    assert deg == pytest.approx(180.0, 0.1)
