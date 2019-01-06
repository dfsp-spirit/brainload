import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import brainload as bl
import brainload.spatial as st

def test_rad2deg_with_positive_input():
    deg = st.rad2deg(2 * np.pi)
    assert deg == pytest.approx(360.0, 0.01)


def test_rad2deg_with_negative_input():
    deg = st.rad2deg(- 0.5 * np.pi)
    assert deg == pytest.approx(270.0, 0.01)


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


def test_rotate_3D_coordinates_around_axes_with_x_axis():
    x = np.array([5, 6])
    y = np.array([7, 8])
    z = np.array([9, 10])
    xr, yr, zr = st.rotate_3D_coordinates_around_axes(x, y, z, np.pi, 0, 0)
    expected_xr = np.array([5, 6])
    expected_yr = np.array([-7, -8])
    expected_zr = np.array([-9, -10])
    assert_allclose(xr, expected_xr)
    assert_allclose(yr, expected_yr)
    assert_allclose(zr, expected_zr)


def test_rotate_3D_coordinates_around_axes_with_y_axis():
    x = np.array([5, 6])
    y = np.array([7, 8])
    z = np.array([9, 10])
    xr, yr, zr = st.rotate_3D_coordinates_around_axes(x, y, z, 0, np.pi, 0)
    expected_xr = np.array([-5, -6])
    expected_yr = np.array([7, 8])
    expected_zr = np.array([-9, -10])
    assert_allclose(xr, expected_xr)
    assert_allclose(yr, expected_yr)
    assert_allclose(zr, expected_zr)

def test_rotate_3D_coordinates_around_axes_with_z_axis():
    x = np.array([5, 6])
    y = np.array([7, 8])
    z = np.array([9, 10])
    xr, yr, zr = st.rotate_3D_coordinates_around_axes(x, y, z, 0, 0, np.pi)
    expected_xr = np.array([-5, -6])
    expected_yr = np.array([-7, -8])
    expected_zr = np.array([9, 10])
    assert_allclose(xr, expected_xr)
    assert_allclose(yr, expected_yr)
    assert_allclose(zr, expected_zr)

def test_mirror_3D_coordinates_at_axis_with_x_axis_no_exlicit_value():
    x = np.array([5, 6])
    y = np.array([7, 8])
    z = np.array([9, 10])
    xm, ym, zm = st.mirror_3D_coordinates_at_axis(x, y, z, 'x')
    expected_xm = np.array([5, 4])
    expected_ym = np.array([7, 8])
    expected_zm = np.array([9, 10])
    assert_allclose(xm, expected_xm)
    assert_allclose(ym, expected_ym)
    assert_allclose(zm, expected_zm)

def test_mirror_3D_coordinates_at_axis_with_x_axis_with_exlicit_value():
    x = np.array([5, 6])
    y = np.array([7, 8])
    z = np.array([9, 10])
    xm, ym, zm = st.mirror_3D_coordinates_at_axis(x, y, z, 'x', 0)
    expected_xm = np.array([-5, -6])
    expected_ym = np.array([7, 8])
    expected_zm = np.array([9, 10])
    assert_allclose(xm, expected_xm)
    assert_allclose(ym, expected_ym)
    assert_allclose(zm, expected_zm)

def test_mirror_3D_coordinates_at_axis_with_y_axis_no_exlicit_value():
    x = np.array([5, 6])
    y = np.array([7, 8])
    z = np.array([9, 10])
    xm, ym, zm = st.mirror_3D_coordinates_at_axis(x, y, z, 'y')
    expected_xm = np.array([5, 6])
    expected_ym = np.array([7, 6])
    expected_zm = np.array([9, 10])
    assert_allclose(xm, expected_xm)
    assert_allclose(ym, expected_ym)
    assert_allclose(zm, expected_zm)

def test_mirror_3D_coordinates_at_axis_with_y_axis_with_exlicit_value():
    x = np.array([5, 6])
    y = np.array([7, 8])
    z = np.array([9, 10])
    xm, ym, zm = st.mirror_3D_coordinates_at_axis(x, y, z, 'y', 0)
    expected_xm = np.array([5, 6])
    expected_ym = np.array([-7, -8])
    expected_zm = np.array([9, 10])
    assert_allclose(xm, expected_xm)
    assert_allclose(ym, expected_ym)
    assert_allclose(zm, expected_zm)

def test_mirror_3D_coordinates_at_axis_with_z_axis_no_exlicit_value():
    x = np.array([5, 6])
    y = np.array([7, 8])
    z = np.array([9, 10])
    xm, ym, zm = st.mirror_3D_coordinates_at_axis(x, y, z, 'z')
    expected_xm = np.array([5, 6])
    expected_ym = np.array([7, 8])
    expected_zm = np.array([9, 8])
    assert_allclose(xm, expected_xm)
    assert_allclose(ym, expected_ym)
    assert_allclose(zm, expected_zm)

def test_mirror_3D_coordinates_at_axis_with_z_axis_with_exlicit_value():
    x = np.array([5, 6])
    y = np.array([7, 8])
    z = np.array([9, 10])
    xm, ym, zm = st.mirror_3D_coordinates_at_axis(x, y, z, 'z', -2)
    expected_xm = np.array([5, 6])
    expected_ym = np.array([7, 8])
    expected_zm = np.array([-13, -14])
    assert_allclose(xm, expected_xm)
    assert_allclose(ym, expected_ym)
    assert_allclose(zm, expected_zm)

def test_point_mirror_3D_coordinates():
    x = np.array([5, 6])
    y = np.array([7, 8])
    z = np.array([9, 10])
    xm, ym, zm = st.point_mirror_3D_coordinates(x, y, z, 0, 0, 0)
    expected_xm = np.array([-5, -6])
    expected_ym = np.array([-7, -8])
    expected_zm = np.array([-9, -10])
    assert_allclose(xm, expected_xm)
    assert_allclose(ym, expected_ym)
    assert_allclose(zm, expected_zm)

def test_translate_3D_coordinates_along_axes():
    x = np.array([5, 6])
    y = np.array([7, 8])
    z = np.array([9, 10])
    xt, yt, zt = st.translate_3D_coordinates_along_axes(x, y, z, 2, -4, 0)
    expected_xt = np.array([7, 8])
    expected_yt = np.array([3, 4])
    expected_zt = np.array([9, 10])
    assert_allclose(xt, expected_xt)
    assert_allclose(yt, expected_yt)
    assert_allclose(zt, expected_zt)

def test_scale_3D_coordinates():
    x = np.array([5, 6])
    y = np.array([7, 8])
    z = np.array([9, 10])
    xs, ys, zs = st.scale_3D_coordinates(x, y, z, 3.0)
    expected_xs = np.array([15, 18])
    expected_ys = np.array([21, 24])
    expected_zs = np.array([27, 30])
    assert_allclose(xs, expected_xs)
    assert_allclose(ys, expected_ys)
    assert_allclose(zs, expected_zs)


def test_parse_registration_matrix():
    matrix_str="""1.000000000000000e+00 0.000000000000000e+00 0.000000000000000e+00 0.000000000000000e+00
0.000000000000000e+00 0.000000000000000e+00 1.000000000000000e+00 0.000000000000000e+00
0.000000000000000e+00 -1.000000000000000e+00 0.000000000000000e+00 0.000000000000000e+00
0 0 0 1"""
    matrix = st.parse_registration_matrix(matrix_str.splitlines())
    assert matrix.shape == (4, 4)


def test_apply_affine_from_MNI305_to_MIN152():
    MNI305_x = 10
    MNI305_y = -20
    MNI305_z = 35
    affine_matrix = st.get_affine_matrix_MNI305_to_MNI152()
    v = st.apply_affine(MNI305_x, MNI305_y, MNI305_z, affine_matrix)
    assert v[0] == pytest.approx(10.695, 0.01)            # see http://freesurfer.net/fswiki/CoordinateSystems
    assert v[1] == pytest.approx(-18.409, 0.01)
    assert v[2] == pytest.approx(36.137, 0.01)

def test_apply_affine_from_MNI305_to_MIN152_and_back():
    MNI305_x = 10
    MNI305_y = -20
    MNI305_z = 35
    v = st.apply_affine(MNI305_x, MNI305_y, MNI305_z, st.get_affine_matrix_MNI305_to_MNI152())
    w = st.apply_affine(v[0], v[1], v[2], st.get_affine_matrix_MNI152_to_MNI305())
    assert w[0] == pytest.approx(MNI305_x, 0.01)
    assert w[1] == pytest.approx(MNI305_y, 0.01)
    assert w[2] == pytest.approx(MNI305_z, 0.01)


def test_project_fsaverage_voxel_to_surface_coord():
    m = st.get_matrix_voxel_MNI305_orig_to_vertex_surface()
    
