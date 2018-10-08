import pytest
import numpy as np
from numpy.testing import assert_array_equal
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

def test_coords_a2s_single_values():
    coords = np.array([[5, 7, 9]])
    x, y, z = st.coords_a2s(coords)
    assert x == 5
    assert y == 7
    assert z == 9

def test_coords_a2s_arrays():
    coords = np.array([[5, 7, 9], [6, 8, 10]])
    x, y, z = st.coords_a2s(coords)
    assert x[0] == 5
    assert y[0] == 7
    assert z[0] == 9
    assert x[1] == 6
    assert y[1] == 8
    assert z[1] == 10


def test_coords_s2a_single_values():
    x = 5
    y = 7
    z = 9
    coords = st.coords_s2a(x, y, z)
    assert coords[0][0] == 5
    assert coords[0][1] == 7
    assert coords[0][2] == 9

def test_coords_s2a_arrays():
    x = np.array([5, 6])
    y = np.array([7, 8])
    z = np.array([9, 10])
    coords = st.coords_s2a(x, y, z)
    assert coords[0][0] == 5
    assert coords[0][1] == 7
    assert coords[0][2] == 9
    assert coords[1][0] == 6
    assert coords[1][1] == 8
    assert coords[1][2] == 10

def test_applying_s2a_followed_by_a2s_is_identical_to_start_value():
    coords = np.array([[5, 7, 9], [6, 8, 10]])
    x, y, z = st.coords_a2s(coords)
    coords_new = st.coords_s2a(x, y, z)
    assert_array_equal(coords, coords_new)


def test_rotate_3D_coordinates_around_axis_with_x_axis():
    coords = np.array([])
    #rotate_3D_coordinates_around_axis()
