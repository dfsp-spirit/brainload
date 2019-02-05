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


def test_closest_vertex_to_very_close_point_known_dist():
    try:
        from scipy.spatial.distance import cdist
    except ImportError:
        pytest.skip("Optional dependency scipy not installed, skipping tests which require scipy.")
    vert_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    faces = np.array([0, 1, 2])
    locator = loc.BrainLocate(vert_coords, faces)
    query_coords = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.1, 0.0], [0.1, 0.1, 0.1]])
    res = locator.get_closest_vertex(query_coords)
    assert res.shape == (4, )
    assert res[0] == 1     # the closest vertex index in the mesh for query coordinate at index 0
    assert res[1] == 2     # the closest vertex index in the mesh for query coordinate at index 1
    assert res[2] == 2     # the closest vertex index in the mesh for query coordinate at index 2
    assert res[3] == 0     # the closest vertex index in the mesh for query coordinate at index 3
    dist_matrix = cdist(vert_coords, query_coords)
    assert dist_matrix.shape == (3, 4)
    assert dist_matrix[0][0] == pytest.approx(1.0, 0.00001)
    assert dist_matrix[0][1] == pytest.approx(1.4142135623730951, 0.00001)   # This is sqrt(2)
    assert dist_matrix[1][0] == pytest.approx(0.0, 0.00001)
    min_index = np.argmin(dist_matrix, axis=0)
    assert min_index.shape == (4, )    # we queried for 4 coordinates.
    assert min_index[0] == res[0]
    assert min_index[1] == res[1]
    assert min_index[2] == res[2]
    assert min_index[3] == res[3]


def test_closest_vertex_to_very_close_point():
    try:
        from scipy.spatial.distance import cdist
    except ImportError:
        pytest.skip("Optional dependency scipy not installed, skipping tests which require scipy.")
    vert_coords, faces, _ = bl.subject_mesh('subject1', TEST_DATA_DIR, surf='white', hemi='both')
    locator = loc.BrainLocate(vert_coords, faces)
    query_coords = np.array([[58.0 , -45.0, 75.0]])
    res = locator.get_closest_vertex(query_coords)
    assert res.shape == (1, )
    assert res[0] == 210683     # the vertex index in the mesh
    expected_vert_210683_coords = (58.005173, -44.736935,  74.418076)
    assert_allclose(vert_coords[210683], np.array(expected_vert_210683_coords))
    dist = cdist(np.array([expected_vert_210683_coords]), query_coords)
    assert dist[0][0] == pytest.approx(0.6386434810831467, 0.001)


def test_closest_vertex_to_far_away_point():
    try:
        from scipy.spatial.distance import cdist
    except ImportError:
        pytest.skip("Optional dependency scipy not installed, skipping tests which require scipy.")
    vert_coords, faces, _ = bl.subject_mesh('subject1', TEST_DATA_DIR, surf='white', hemi='both')
    locator = loc.BrainLocate(vert_coords, faces)
    query_coords = np.array([[134.37332 , -57.259495, 149.267631], [134.37332 , -57.259495, 149.267631], [58.0 , -45.0, 75.0]])
    res = locator.get_closest_vertex(query_coords)
    assert res.shape == (3, )
    assert res[0] == 209519     # the vertex index in the mesh
    assert res[1] == 209519
    assert res[2] == 210683
    assert_allclose(vert_coords[209519], np.array((58.258751, -45.213722,  74.348068)))
    dist = cdist(np.array([[58.258751, -45.213722,  74.348068]]), np.array([[134.37332 , -57.259495, 149.267631]]))
    assert dist[0][0] == pytest.approx(107.47776133, 0.001)


def test_get_closest_vertex_and_distance_to_far_away_point():
    try:
        from scipy.spatial.distance import cdist
    except ImportError:
        pytest.skip("Optional dependency scipy not installed, skipping tests which require scipy.")
    vert_coords, faces, _ = bl.subject_mesh('subject1', TEST_DATA_DIR, surf='white', hemi='both')
    locator = loc.BrainLocate(vert_coords, faces)
    query_coords = np.array([[134.37332 , -57.259495, 149.267631], [134.37332 , -57.259495, 149.267631], [134.37332 , -57.259495, 149.267631]])  # just query 3 times for the same coord to see whether results and consistent
    res = locator.get_closest_vertex_and_distance(query_coords)
    assert res.shape == (3, 2)
    assert res[0,0] == 209519              # the vertex index in the mesh
    assert res[1,0] == 209519
    assert res[2,0] == 209519
    assert res[0,1] == 107.47776120258028  # the distance
    assert res[1,1] == 107.47776120258028
    assert res[2,1] == 107.47776120258028


def test_get_closest_vertex_to_vertex_0_coordinate():
    try:
        from scipy.spatial.distance import cdist
    except ImportError:
        pytest.skip("Optional dependency scipy not installed, skipping tests which require scipy.")
    vert_coords, faces, _ = bl.subject_mesh('subject1', TEST_DATA_DIR, surf='white', hemi='both')
    locator = loc.BrainLocate(vert_coords, faces)
    known_vertex_0_coord = (-1.85223234, -107.98274994, 22.76972961)    # coordinate for vertex 0 in the test data.
    assert_allclose(vert_coords[0], np.array(known_vertex_0_coord))
    query_coords = np.array([known_vertex_0_coord])
    res = locator.get_closest_vertex_and_distance(query_coords)
    assert res.shape == (1, 2)
    assert res[0, 0] == 0    # vertex index of closest vertex. The query coordinate is the known coordinate of vertex 0, so this must be 0.
    assert abs(res[0, 1]) <= 0.001   # distance to closest vertex. The query coordinate is the known coordinate of vertex 0, so this must be very close to 0.0
