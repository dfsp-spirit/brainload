import os
import pytest
import numpy as np
from numpy.testing import assert_raises, assert_array_equal, assert_allclose
import brainload as bl
import brainload.surfacegraph as sg

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(THIS_DIR, os.pardir, 'test_data')

# Respect the environment variable BRAINLOAD_TEST_DATA_DIR if it is set. If not, fall back to default.
TEST_DATA_DIR = os.getenv('BRAINLOAD_TEST_DATA_DIR', TEST_DATA_DIR)

SUBJECT1_SURF_LH_WHITE_NUM_VERTICES = 149244

def test_surface_graph():
    vert_coords, faces, meta_data = bl.subject_mesh('subject1', TEST_DATA_DIR, surf='white', hemi='lh')
    surface_graph = sg.SurfaceGraph(vert_coords, faces)
    g = surface_graph.graph
    assert len(g) == SUBJECT1_SURF_LH_WHITE_NUM_VERTICES
    assert g.number_of_nodes() == SUBJECT1_SURF_LH_WHITE_NUM_VERTICES
