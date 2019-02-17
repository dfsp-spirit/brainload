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


def test_closest_vertex_to_very_close_point_known_dist():
    try:
        from scipy.spatial.distance import cdist
    except ImportError:
        pytest.skip("Optional dependency scipy not installed, skipping tests which require scipy.")
    query_vox_rcs = np.array([[24, 28, 20], [64, 64, 45]], dtype=int)
    locator = vloc.BrainVoxLocate()
    vox_ras_coords = locator.get_voxel_coords(query_vox_rcs)
