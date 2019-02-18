"""
Functions for testing brainvoxlocate.
"""

import pytest
import numpy as np
import os
from numpy.testing import assert_array_equal, assert_allclose
import brainload as bl
import brainload.brainvoxlocate as vloc


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(THIS_DIR, os.pardir, 'test_data')

# Respect the environment variable BRAINLOAD_TEST_DATA_DIR if it is set. If not, fall back to default.
TEST_DATA_DIR = os.getenv('BRAINLOAD_TEST_DATA_DIR', TEST_DATA_DIR)


def test_get_vox_crs_at_ras_coords():
    volume_file = os.path.join(TEST_DATA_DIR, 'subject1', 'mri', 'aseg.mgz')
    lookup_file = os.path.join(TEST_DATA_DIR, 'fs', 'FreeSurferColorLUT.txt')
    locator = vloc.BrainVoxLocate(volume_file, lookup_file)
    query_ras_coords = np.array([[40.1, 20.4, 46.3], [33., -32., 25.5]])
    vox_crs = locator.get_voxel_crs_at_ras_coords(query_ras_coords)
    assert vox_crs.shape == (2, 3)


def test_get_ras_coords_at_vox_crs():
    volume_file = os.path.join(TEST_DATA_DIR, 'subject1', 'mri', 'aseg.mgz')
    lookup_file = os.path.join(TEST_DATA_DIR, 'fs', 'FreeSurferColorLUT.txt')
    locator = vloc.BrainVoxLocate(volume_file, lookup_file)
    query_vox_rcs = np.array([[24, 28, 20], [64, 64, 45]], dtype=int)
    vox_ras_coords = locator.get_ras_coords_at_voxel_crs(query_vox_rcs)
    assert vox_ras_coords.shape == (2, 3)


def test_get_voxel_segmentation_labels():
    volume_file = os.path.join(TEST_DATA_DIR, 'subject1', 'mri', 'aseg.mgz')
    lookup_file = os.path.join(TEST_DATA_DIR, 'fs', 'FreeSurferColorLUT.txt')
    locator = vloc.BrainVoxLocate(volume_file, lookup_file)
    query_vox_rcs = np.array([[24, 28, 20], [64, 64, 45], [90, 90, 90]], dtype=int)
    seg_code, seg_data = locator.get_voxel_segmentation_labels(query_vox_rcs)
    assert seg_code.shape == (3, )
    assert seg_code[0] == 0     # 0 is 'Unknown', see FreeSurferColorLUT.txt
    assert seg_code[1] == 0
    assert seg_code[2] == 41
    assert seg_data.shape == (3, )
    assert seg_data[0] == "Unknown"
    assert seg_data[1] == "Unknown"
    assert seg_data[2] == "Right-Cerebral-White-Matter"


def test_closest_vertex_to_very_close_point_known_dist():
    pass
