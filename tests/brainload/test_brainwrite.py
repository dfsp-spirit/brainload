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
