import pytest
import numpy as np
import cogload as cl
import cogload.spatial_transform as st

def test_rad2deg_with_positive_input():
    deg = st.rad2deg(2 * np.pi)
    assert deg == pytest.approx(360.0, 0.01)

def test_deg2rad_with_positive_input():
    rad = st.deg2rad(180)
    assert rad == pytest.approx(np.pi, 0.01)

def test_deg2rad_with_negative_input():
    rad = st.deg2rad(-90)
    assert rad == pytest.approx(np.pi * 1.5, 0.01)
