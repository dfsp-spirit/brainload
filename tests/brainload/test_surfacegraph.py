import os
import pytest
import numpy as np
from numpy.testing import assert_raises, assert_array_equal, assert_allclose
import brainload as bl

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(THIS_DIR, os.pardir, 'test_data')

# Respect the environment variable BRAINLOAD_TEST_DATA_DIR if it is set. If not, fall back to default.
TEST_DATA_DIR = os.getenv('BRAINLOAD_TEST_DATA_DIR', TEST_DATA_DIR)

SUBJECT1_SURF_LH_WHITE_NUM_VERTICES = 149244

def test_surface_graph():
    try:
        import networkx as nx
        import brainload.surfacegraph as sg
    except ImportError:
        pytest.skip("Optional dependency networkx not installed, skipping tests which require it.")
    vert_coords, faces, meta_data = bl.subject_mesh('subject1', TEST_DATA_DIR, surf='white', hemi='lh')
    surface_graph = sg.SurfaceGraph(vert_coords, faces)
    g = surface_graph.graph
    assert len(g) == SUBJECT1_SURF_LH_WHITE_NUM_VERTICES
    assert g.number_of_nodes() == SUBJECT1_SURF_LH_WHITE_NUM_VERTICES
    # now for some neighborhood queries
    source_vertex = 100
    neighbors_dist_1 = surface_graph.get_neighbors_up_to_dist(source_vertex, 1)
    assert len(neighbors_dist_1) == 9
    neighbors_dist_2 = surface_graph.get_neighbors_up_to_dist(source_vertex, 2)
    assert len(neighbors_dist_2) == 25
    neighbors_dist_3 = surface_graph.get_neighbors_up_to_dist(source_vertex, 3)
    assert len(neighbors_dist_3) == 48
