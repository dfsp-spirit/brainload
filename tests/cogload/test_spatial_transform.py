import pytest
import numpy as np
import cogload.spatial_transform as st

def test_rad2deg_with_valid_input():
    deg = st.rad2deg(2 * np.pi)
    assert deg == pytest.approx(360.0, 0.01)
