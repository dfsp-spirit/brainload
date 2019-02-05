"""
Functions for testing brainlocate.
"""

import pytest
import numpy as np
import os
from numpy.testing import assert_array_equal, assert_allclose
import brainload as bl
import brainload.brainlocate as loc


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(THIS_DIR, os.pardir, 'test_data')

# Respect the environment variable BRAINLOAD_TEST_DATA_DIR if it is set. If not, fall back to default.
TEST_DATA_DIR = os.getenv('BRAINLOAD_TEST_DATA_DIR', TEST_DATA_DIR)


def test_closest_vertex():
    try:
        from scipy.spatial.distance import cdist
    except ImportError:
        pytest.skip("Optional dependency scipy not installed, skipping tests which require scipy.")
    vert_coords, faces, _ = bl.subject_mesh('subject1', TEST_DATA_DIR, surf='white', hemi='both')
    locator = loc.BrainLocate(vert_coords, faces)
    query_coords = np.array([[134.37332 , -57.259495, 149.267631], [134.37332 , -57.259495, 149.267631]])
    res = locator.get_closest_vertex(query_coords)
    assert res.shape == (2, )
    assert res[0] == 209519     # the vertex index in the mesh
    assert res[1] == 209519
    assert_allclose(vert_coords[209519], np.array((58.258751, -45.213722,  74.348068)))
    dist = cdist(np.array([[58.258751, -45.213722,  74.348068]]), np.array([[134.37332 , -57.259495, 149.267631]]))
    assert dist[0][0] == pytest.approx(107.47776133, 0.001)


def test_get_closest_vertex_and_distance():
    try:
        from scipy.spatial.distance import cdist
    except ImportError:
        pytest.skip("Optional dependency scipy not installed, skipping tests which require scipy.")
    vert_coords, faces, _ = bl.subject_mesh('subject1', TEST_DATA_DIR, surf='white', hemi='both')
    locator = loc.BrainLocate(vert_coords, faces)
    query_coords = np.array([[134.37332 , -57.259495, 149.267631], [134.37332 , -57.259495, 149.267631], [134.37332 , -57.259495, 149.267631]])
    res = locator.get_closest_vertex_and_distance(query_coords)
    assert res.shape == (3, 2)
    assert res[0,0] == 209519              # the vertex index in the mesh
    assert res[1,0] == 209519
    assert res[2,0] == 209519
    assert res[0,1] == 107.47776120258028  # the distance
    assert res[1,1] == 107.47776120258028
    assert res[2,1] == 107.47776120258028
