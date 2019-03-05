import os
import pytest
import numpy as np
from numpy.testing import assert_raises, assert_array_equal, assert_allclose
import brainload.brainwrite as bw
import brainload as bl

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(THIS_DIR, os.pardir, 'test_data')

# Respect the environment variable BRAINLOAD_TEST_DATA_DIR if it is set. If not, fall back to default.
TEST_DATA_DIR = os.getenv('BRAINLOAD_TEST_DATA_DIR', TEST_DATA_DIR)

def test_get_volume_data_with_custom_marks():
    voxel_mark_list = [(np.array([[1, 1, 1], [2, 2, 2]], dtype=int), 40), (np.array([[0, 1, 2], [0, 2, 2]], dtype=int), 160)]
    vol_data = bw.get_volume_data_with_custom_marks(voxel_mark_list, background_voxel_value=0, shape=(3, 3, 3))
    assert vol_data.shape == (3, 3, 3)
    assert vol_data.dtype == np.uint8
    assert vol_data[0, 0, 0] == 0
    assert vol_data[0, 0, 1] == 0
    assert vol_data[0, 0, 2] == 0
    assert vol_data[0, 2, 0] == 0
    assert vol_data[0, 2, 1] == 0
    assert vol_data[0, 1, 0] == 0
    assert vol_data[0, 1, 1] == 0
    assert vol_data[1, 1, 1] == 40
    assert vol_data[2, 2, 2] == 40
    assert vol_data[0, 1, 2] == 160
    assert vol_data[0, 2, 2] == 160
    assert vol_data[1, 2, 2] == 0


def test_get_volume_data_with_custom_marks_no_shape_given_custom_background_value():
    voxel_mark_list = [(np.array([[1, 1, 1], [2, 2, 2]], dtype=int), 40), (np.array([[0, 1, 2], [0, 2, 2]], dtype=int), 160)]
    vol_data = bw.get_volume_data_with_custom_marks(voxel_mark_list, background_voxel_value=10, dtype=np.int16)
    assert vol_data.shape == (256, 256, 256)
    assert vol_data.dtype == np.int16
    assert vol_data[0, 0, 0] == 10
    assert vol_data[0, 0, 1] == 10
    assert vol_data[0, 0, 2] == 10
    assert vol_data[0, 2, 0] == 10
    assert vol_data[0, 2, 1] == 10
    assert vol_data[0, 1, 0] == 10
    assert vol_data[0, 1, 1] == 10
    assert vol_data[1, 1, 1] == 40
    assert vol_data[2, 2, 2] == 40
    assert vol_data[0, 1, 2] == 160
    assert vol_data[0, 2, 2] == 160
    assert vol_data[1, 2, 2] == 10


def test_get_surface_vertices_overlay_volume_data():
    num_verts = 10
    vertex_mark_list = [(np.array([0, 2, 4], dtype=int), [20, 20, 20]), (np.array([1, 3, 5, 7], dtype=int), [40, 40, 40])]
    vol_data = bw.get_surface_vertices_overlay_volume_data(num_verts, vertex_mark_list, background_rgb=[200, 200, 200])
    assert vol_data.shape == (10, 3, 1)
    assert_array_equal(vol_data[0,:,0], [20, 20, 20])
    assert_array_equal(vol_data[2,:,0], [20, 20, 20])
    assert_array_equal(vol_data[4,:,0], [20, 20, 20])
    assert_array_equal(vol_data[1,:,0], [40, 40, 40])
    assert_array_equal(vol_data[3,:,0], [40, 40, 40])
    assert_array_equal(vol_data[5,:,0], [40, 40, 40])
    assert_array_equal(vol_data[7,:,0], [40, 40, 40])
    assert_array_equal(vol_data[6,:,0], [200, 200, 200])
    assert_array_equal(vol_data[8,:,0], [200, 200, 200])
    assert_array_equal(vol_data[9,:,0], [200, 200, 200])


def test_get_surface_vertices_overlay_text_file_lines():
    num_verts = 10
    vertex_mark_list = [(np.array([0, 2, 4], dtype=int), [20, 20, 20]), (np.array([1, 3, 5, 7], dtype=int), [40, 40, 40])]
    overlay_lines = bw.get_surface_vertices_overlay_text_file_lines(num_verts, vertex_mark_list)
    assert len(overlay_lines) == 10
    assert overlay_lines[0] == "20, 20, 20"
    assert overlay_lines[2] == "20, 20, 20"
    assert overlay_lines[4] == "20, 20, 20"
    assert overlay_lines[1] == "40, 40, 40"
    assert overlay_lines[7] == "40, 40, 40"
    assert overlay_lines[6] == "200, 200, 200"


def test_get_surface_vertices_overlay_volume_data_1color():
    num_verts = 10
    vertex_mark_list = [(np.array([0, 2, 4], dtype=int), 20), (np.array([1, 3, 5, 7], dtype=int), 40)]
    vol_data = bw.get_surface_vertices_overlay_volume_data_1color(num_verts, vertex_mark_list, background_value=0)
    assert vol_data.shape == (10, 1, 1)
    assert vol_data[0,0,0] == 20
    assert vol_data[2,0,0] == 20
    assert vol_data[4,0,0] == 20
    assert vol_data[1,0,0] == 40
    assert vol_data[7,0,0] == 40
    assert vol_data[6,0,0] == 0
