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
    query_vox_crs = np.array([[24, 28, 20], [64, 64, 45]], dtype=int)
    vox_ras_coords = locator.get_ras_coords_at_voxel_crs(query_vox_crs)
    assert vox_ras_coords.shape == (2, 3)


def test_get_voxel_segmentation_labels():
    volume_file = os.path.join(TEST_DATA_DIR, 'subject1', 'mri', 'aseg.mgz')
    lookup_file = os.path.join(TEST_DATA_DIR, 'fs', 'FreeSurferColorLUT.txt')
    locator = vloc.BrainVoxLocate(volume_file, lookup_file)
    query_vox_crs = np.array([[24, 28, 20], [64, 64, 45], [90, 90, 90], [95, 127, 45]], dtype=int)
    seg_code, seg_data = locator.get_voxel_segmentation_labels(query_vox_crs)
    assert seg_code.shape == (4, )
    assert seg_code[0] == 0     # 0 is 'Unknown', see FreeSurferColorLUT.txt
    assert seg_code[1] == 0
    assert seg_code[2] == 41
    assert seg_code[3] == 42
    assert seg_data.shape == (4, )
    assert seg_data[0] == "Unknown"
    assert seg_data[1] == "Unknown"
    assert seg_data[2] == "Right-Cerebral-White-Matter"         # Note: you can check that these values are correct by loading the aseg.mgz file in FreeView.
    assert seg_data[3] == "Right-Cerebral-Cortex"


def test_closest_not_unknown_neighborhood_default_10():
    try:
        from scipy.spatial.distance import cdist
    except ImportError:
        pytest.skip("Optional dependency scipy not installed, skipping tests which require scipy.")
    volume_file = os.path.join(TEST_DATA_DIR, 'subject1', 'mri', 'aseg.mgz')
    lookup_file = os.path.join(TEST_DATA_DIR, 'fs', 'FreeSurferColorLUT.txt')
    locator = vloc.BrainVoxLocate(volume_file, lookup_file)
    query_vox_crs = np.array([[24, 28, 20], [64, 64, 45], [90, 90, 90], [95, 127, 45]], dtype=int)
    voxels, codes, distances, closest_voxels_ras_coords = locator.get_closest_not_unknown(query_vox_crs)
    assert voxels.shape == (4, 3)
    assert codes.shape == (4, )
    assert codes[0] == -1     # 0 is 'Unknown', see FreeSurferColorLUT.txt. If you increase neighborhood_size further, you will see that the closest label is 42, but it is quite far away.
    assert codes[1] == 42
    assert codes[2] == 41
    assert codes[3] == 42
    assert distances.shape == (4, )
    assert closest_voxels_ras_coords.shape == (4, 3)


def test_closest_not_unknown_neighborhood_15_near_brain_stem():
    try:
        from scipy.spatial.distance import cdist
    except ImportError:
        pytest.skip("Optional dependency scipy not installed, skipping tests which require scipy.")
    volume_file = os.path.join(TEST_DATA_DIR, 'subject1', 'mri', 'aseg.mgz')
    lookup_file = os.path.join(TEST_DATA_DIR, 'fs', 'FreeSurferColorLUT.txt')
    locator = vloc.BrainVoxLocate(volume_file, lookup_file)
    query_vox_crs = np.array([[127, 136, 103]], dtype=int)
    voxels, codes, distances, closest_voxels_ras_coords = locator.get_closest_not_unknown(query_vox_crs, neighborhood_size=15)
    assert voxels.shape == (1, 3)
    assert codes.shape == (1, )
    assert codes[0] == 16     # 16 is 'Brain-Stem', see FreeSurferColorLUT.txt. See the voxel location in freeview to see that this is correct.
    assert distances.shape == (1, )
    assert closest_voxels_ras_coords.shape == (1, 3)
