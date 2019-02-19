import pytest
import numpy as np
import os
from numpy.testing import assert_array_equal, assert_allclose
import brainload as bl
import brainload.spatial as st

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(THIS_DIR, os.pardir, 'test_data')

# Respect the environment variable BRAINLOAD_TEST_DATA_DIR if it is set. If not, fall back to default.
TEST_DATA_DIR = os.getenv('BRAINLOAD_TEST_DATA_DIR', TEST_DATA_DIR)


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
    assert v.shape == (3, )
    assert v[0] == pytest.approx(10.695, 0.01)            # see http://freesurfer.net/fswiki/CoordinateSystems for this example. The source and target values used here are from the example given on that page.
    assert v[1] == pytest.approx(-18.409, 0.01)
    assert v[2] == pytest.approx(36.137, 0.01)


def test_apply_affine_from_MNI305_to_MIN152_array():
    MNI305_x = np.array([10, -5])
    MNI305_y = np.array([-20, -20])
    MNI305_z = np.array([35, 72])
    affine_matrix = st.get_affine_matrix_MNI305_to_MNI152()
    v = st.apply_affine(MNI305_x, MNI305_y, MNI305_z, affine_matrix)
    assert v.shape == (3, 2)
    assert_allclose(v[0,:], np.array([10.6941, -3.6172]))
    assert_allclose(v[1,:], np.array([-18.4064, -18.7142]))
    assert_allclose(v[2,:], np.array([36.1385, 73.2262]))


def test_apply_affine_3D_from_MNI305_to_MIN152_array():
    coords_305 = np.array([[10, -20, 35], [10, -20, 35]])
    affine_matrix = st.get_affine_matrix_MNI305_to_MNI152()
    coords_152 = st.apply_affine_3D(coords_305, affine_matrix)
    assert coords_152.shape == (2, 3)
    expected = np.array([[10.6941, -18.4064, 36.1385], [10.6941, -18.4064, 36.1385]])
    assert_allclose(coords_152, expected)


def test_apply_affine_3D_from_MNI305_to_MIN152_array():
    coords_305 = np.array([[10, -20, 35], [-5, -20, 72]])
    affine_matrix = st.get_affine_matrix_MNI305_to_MNI152()
    coords_152 = st.apply_affine_3D(coords_305, affine_matrix)
    assert coords_152.shape == (2, 3)
    expected = np.array([[10.6941, -18.4064, 36.1385], [-3.6172, -18.7142, 73.2262]])
    assert_allclose(coords_152, expected)


def test_apply_affine_from_MNI305_to_MIN152_and_back():
    MNI305_x = 10
    MNI305_y = -20
    MNI305_z = 35
    v = st.apply_affine(MNI305_x, MNI305_y, MNI305_z, st.get_affine_matrix_MNI305_to_MNI152())
    w = st.apply_affine(v[0], v[1], v[2], st.get_affine_matrix_MNI152_to_MNI305())
    assert w[0] == pytest.approx(MNI305_x, 0.01)
    assert w[1] == pytest.approx(MNI305_y, 0.01)
    assert w[2] == pytest.approx(MNI305_z, 0.01)


def test_project_fsaverage_voxel_index_to_RAS_coord():
    #Cortex structure 'paracentral lobule, anterior part, left' with id 4072: found MIN152 coordinates (-5, -20, 72)
    coords_mni152 = np.array([[-5, -20, 72]])
    coords_mni305 = st.apply_affine_3D(coords_mni152, st.get_affine_matrix_MNI152_to_MNI305())
    coords_mni305_surface = st.apply_affine_3D(coords_mni305, st.get_freesurfer_matrix_vox2ras())
    assert coords_mni305_surface.shape == (1, 3)
    assert_allclose(coords_mni305_surface, np.array([[134.37332 , -57.259495, 149.267631]]))


def test_get_freesurfer_matrix_ras2vox():
    expected = np.array([[-1.00000, 0.00000, 0.00000, 128.00000],
   [0.00000, 0.00000, -1.00000, 128.00000],
   [0.00000, 1.00000, 0.00000, 128.00000],
   [0.00000, 0.00000, 0.00000, 1.00000]])
    m = st.get_freesurfer_matrix_ras2vox()
    assert_allclose(expected, m)
    # now apply it: the coordinate (0.0, 0.0, 0.0) should give us voxel index (128, 128, 128)
    query_coord = np.array([[0., 0., 0.]])
    vox_idx = np.rint(st.apply_affine_3D(query_coord, m)).astype(int)
    expected = np.array([[128, 128, 128]])
    assert_array_equal(vox_idx, expected)


def test_get_freesurfer_matrix_vox2ras_for_vertex_0():
    # Tests that the vertex at index (128, 128, 128) has a RAS coordinate close to the origin (0., 0., 0.).
    m = st.get_freesurfer_matrix_vox2ras()
    voxel_origin = np.array([[128, 128, 128]])
    ras_coords_near_origin = st.apply_affine_3D(voxel_origin, m)
    assert ras_coords_near_origin.shape == (1, 3)
    assert_allclose(ras_coords_near_origin, np.array([[0., 0., 0.]]))


def test_get_n_neighborhood_start_stop_indices_3D_point_itself():
    # test with 0-neighborhood should return only index of the point itself
    volume = np.zeros((3, 3, 3))
    point = [1, 1, 1]   # x, y, z
    xstart, xend, ystart, yend, zstart, zend = st.get_n_neighborhood_start_stop_indices_3D(volume.shape, point, 0)
    assert xstart == 1
    assert xend == 2 # target index is non-inclusive
    assert ystart == 1
    assert yend == 2
    assert zstart == 1
    assert zend == 2


def test_get_n_neighborhood_start_stop_indices_3D_point_n1():
    # test with 1-neighborhood should return the indices of the entire 3x3 volume
    volume = np.zeros((3, 3, 3))
    point = [1, 1, 1]
    xstart, xend, ystart, yend, zstart, zend = st.get_n_neighborhood_start_stop_indices_3D(volume.shape, point, 1)    # 1-neighborhood
    assert xstart == 0
    assert xend == 3 # target index is non-inclusive
    assert ystart == 0
    assert yend == 3
    assert zstart == 0
    assert zend == 3


def test_get_n_neighborhood_start_stop_indices_3D_point_n1_in_corner():
    # test with 1-neighborhood, but a point in the corner of the volume
    volume = np.zeros((3, 3, 3))
    point = [0, 0, 0]
    xstart, xend, ystart, yend, zstart, zend = st.get_n_neighborhood_start_stop_indices_3D(volume.shape, point, 1)    # 1-neighborhood
    assert xstart == 0
    assert xend == 2 # target index is non-inclusive
    assert ystart == 0
    assert yend == 2
    assert zstart == 0
    assert zend == 2


def test_get_n_neighborhood_start_stop_indices_3D_point_n1_in_opposite_corner():
    # test with 1-neighborhood, but a point in the opposite corner of the volume
    volume = np.zeros((3, 3, 3))
    point = [2, 2, 2]
    xstart, xend, ystart, yend, zstart, zend = st.get_n_neighborhood_start_stop_indices_3D(volume.shape, point, 1)    # 1-neighborhood
    assert xstart == 1
    assert xend == 3 # target index is non-inclusive
    assert ystart == 1
    assert yend == 3
    assert zstart == 1
    assert zend == 3


def test_get_n_neighborhood_start_stop_indices_3D_point_n2():
    # test with 2-neighborhood. for this volume, should return the indices of the entire 3x3 volume, but nothing more (the volume is 3x3), so the same values as for 1n in this case.
    volume = np.zeros((3, 3, 3))
    point = [1, 1, 1]
    xstart, xend, ystart, yend, zstart, zend = st.get_n_neighborhood_start_stop_indices_3D(volume.shape, point, 2)    # 2-neighborhood
    assert xstart == 0
    assert xend == 3 # target index is non-inclusive
    assert ystart == 0
    assert yend == 3
    assert zstart == 0
    assert zend == 3


def test_get_n_neighborhood_start_stop_indices_3D_point_n5():
    # test with 5-neighborhood. for this volume, should return the indices of the entire 3x3 volume, but nothing more (the volume is 3x3), so the same values as for 1n in this case.
    volume = np.zeros((3, 3, 3))
    point = [1, 1, 1]
    xstart, xend, ystart, yend, zstart, zend = st.get_n_neighborhood_start_stop_indices_3D(volume.shape, point, 5)    # 5-neighborhood
    assert xstart == 0
    assert xend == 3 # target index is non-inclusive
    assert ystart == 0
    assert yend == 3
    assert zstart == 0
    assert zend == 3


def test_get_n_neighborhood_start_stop_indices_3D_points():
    volume = np.zeros((3, 3, 3))
    points = np.array([[1, 1, 1], [2,2,2]])
    xstart, xend, ystart, yend, zstart, zend = st.get_n_neighborhood_start_stop_indices_3D_points(volume.shape, points, 0)
    assert xstart.shape == (2, )
    assert ystart.shape == (2, )
    assert zstart.shape == (2, )

    assert xstart[0] == 1
    assert xend[0] == 2
    assert ystart[0] == 1
    assert yend[0] == 2
    assert zstart[0] == 1
    assert zend[0] == 2

    assert xstart[1] == 2
    assert xend[1] == 3
    assert ystart[1] == 2
    assert yend[1] == 3
    assert zstart[1] == 2
    assert zend[1] == 3


def test_get_n_neighborhood_indices_3D_n0_neighborhood():
    volume = np.zeros((3, 3, 3))
    point = [1, 1, 1]
    indices = st.get_n_neighborhood_indices_3D(volume.shape, point, 0)
    #assert_array_equal(indices[0], np.array([0, 0, 0, 0, 0, 0, 0], dtype=int))
    assert len(indices) == 3
    assert len(indices[0]) == 1 # only the point itself
    assert len(indices[1]) == 1
    assert len(indices[2]) == 1
    assert indices[0] == np.array([1], dtype=int)
    assert indices[1] == np.array([1], dtype=int)
    assert indices[2] == np.array([1], dtype=int)
    assert_array_equal(indices, np.array([[1], [1], [1]]))


def test_get_n_neighborhood_indices_3D_n1_neighborhood():
    volume = np.zeros((3, 3, 3))
    point = [1, 1, 1]
    indices = st.get_n_neighborhood_indices_3D(volume.shape, point, 1)
    assert len(indices) == 3
    assert len(indices[0]) == 27    # 3**3
    assert len(indices[1]) == 27
    assert len(indices[2]) == 27
    assert_array_equal(indices[0], np.array([0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2], dtype=int))
    assert_array_equal(indices[1], np.array([0,0,0,1,1,1,2,2,2,0,0,0,1,1,1,2,2,2,0,0,0,1,1,1,2,2,2], dtype=int))
    assert_array_equal(indices[2], np.array([0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2], dtype=int))


def test_get_n_neighborhood_indices_3D_n2_neighborhood():
    volume = np.zeros((5, 5, 5))        # 5x5x5 volume this time!
    point = [2, 2, 2]
    indices = st.get_n_neighborhood_indices_3D(volume.shape, point, 2)
    assert len(indices) == 3
    assert len(indices[0]) == 125       # 5**3
    assert len(indices[1]) == 125
    assert len(indices[2]) == 125


def test_get_n_neighborhood_indices_3D_n2_neighborhood():
    volume = np.zeros((5, 5, 5))        # 5x5x5 volume this time!
    point = [2, 2, 2]
    indices = st.get_n_neighborhood_indices_3D(volume.shape, point, 3)
    assert len(indices) == 3
    assert len(indices[0]) == 125       # it would be 7**3 = 343, but the volume is only 5x5, so not possible
    assert len(indices[1]) == 125
    assert len(indices[2]) == 125


def test_get_n_neighborhood_indices_3D_n2_neighborhood():
    volume = np.zeros((7, 7, 7))        # 5x5x5 volume this time!
    point = [3, 3, 3]
    indices = st.get_n_neighborhood_indices_3D(volume.shape, point, 3)
    assert len(indices) == 3
    assert len(indices[0]) == 343
    assert len(indices[1]) == 343
    assert len(indices[2]) == 343


def test_get_n_neighborhood_indices_3D_n1_neighborhood_points():
    volume = np.zeros((3, 3, 3))
    points = np.array([[0, 0, 0], [2,2,2]])
    indices = st.get_n_neighborhood_indices_3D_points(volume.shape, points, 0)
    assert len(indices) == 3
    assert len(indices[0]) == 2
    assert len(indices[1]) == 2
    assert len(indices[2]) == 2
    assert_array_equal(indices[0], np.array([0,2], dtype=int))
    assert_array_equal(indices[1], np.array([0,2], dtype=int))
    assert_array_equal(indices[2], np.array([0,2], dtype=int))

# This test requires files which are not part of the repo and will be removed soon.
#def test_get_equivalent_voxel_of_raw_volume_in_conformed_volume():
#    raw_volume_file = os.path.join(os.getenv('HOME'), 'data', 'allan_brain_atlas', 'final', 'H03511009.mgz')
#    conformed_volume_file = os.path.join(os.getenv('HOME'), 'data', 'allan_brain_atlas', 'final', 'H03511009_conformed.mgz')
#    raw_volume_query_voxels_crs = np.array([[111,41,137]], dtype=int)
#    conf_volume_voxels_crs = st.get_equivalent_voxel_of_raw_volume_in_conformed_volume(raw_volume_file, conformed_volume_file, raw_volume_query_voxels_crs)
#    expected = np.array([[148, 78, 100]], dtype=int)
#    assert_array_equal(conf_volume_voxels_crs, expected)

def test_get_equivalent_voxel_of_raw_volume_in_conformed_volume_2():
    raw_volume_file = os.path.join(TEST_DATA_DIR, 'subject1', 'mri', 'rawavg.mgz')
    conformed_volume_file = os.path.join(TEST_DATA_DIR, 'subject1', 'mri', 'orig.mgz')
    raw_volume_query_voxels_crs = np.array([[184, 127, 46]], dtype=int)
    conf_volume_voxels_crs = st.get_equivalent_voxel_of_raw_volume_in_conformed_volume(raw_volume_file, conformed_volume_file, raw_volume_query_voxels_crs)
    expected = np.array([[77, 127, 73]], dtype=int)
    assert_array_equal(conf_volume_voxels_crs, expected)
    # now test whether the computation can be inverted and results in the source CRS values that we queried for initially
    raw_volume_voxels_crs_restored = st.get_equivalent_voxel_of_raw_volume_in_conformed_volume(conformed_volume_file, raw_volume_file, conf_volume_voxels_crs)
    assert_array_equal(raw_volume_voxels_crs_restored, raw_volume_query_voxels_crs)
