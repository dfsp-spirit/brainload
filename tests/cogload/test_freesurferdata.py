
import os
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import cogload as cl
import cogload.freesurferdata as fsd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(THIS_DIR, os.pardir, 'test_data')

def test_get_morphology_data_suffix_for_surface_with_surf_white():
    suffix = fsd.get_morphology_data_suffix_for_surface('white')
    assert suffix == ''

def test_get_morphology_data_suffix_for_surface_with_surf_other():
    suffix = fsd.get_morphology_data_suffix_for_surface('pial')
    assert suffix == '.pial'

def test_read_mgh_file_with_valid_file():
    mgh_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'rh.area.fsaverage.mgh')
    mgh_data, mgh_meta_data = fsd.read_mgh_file(mgh_file)
    assert mgh_meta_data['data_bytes_per_voxel'] == 4

def test_merge_meshes():
    m1_vertex_coords = np.array([[0, 0, 0], [5, -5, 0], [5, 5, 0], [10, 5, 0]])
    m1_faces = np.array([[0, 1, 2], [1, 2, 3]])

    m2_vertex_coords = np.array([[0, 0, 0], [10, -10, 0], [10, 10, 0], [15, 10, 0]])
    m2_faces = np.array([[0, 2, 1], [1, 3, 2]])

    merged_verts, merged_faces = fsd.merge_meshes(np.array([[m1_vertex_coords, m1_faces], [m2_vertex_coords, m2_faces]]))
    assert merged_verts.shape[0] == 8
    assert merged_verts.shape[1] == 3

    assert merged_faces.shape[0] == 4
    assert merged_faces.shape[1] == 3

    # test vertices
    assert_allclose(np.array([0, 0, 0]), merged_verts[0])
    assert_allclose(np.array([5, -5, 0]), merged_verts[1])
    assert_allclose(np.array([0, 0, 0]), merged_verts[4])
    assert_allclose(np.array([10, -10, 0]), merged_verts[5])

    # test faces without vertex index shift
    assert_allclose(np.array([0, 1, 2]), merged_faces[0])
    assert_allclose(np.array([1, 2, 3]), merged_faces[1])

    # test faces WITH vertex index shift (shift should be +4 because m1 has 4 vertices)
    assert_allclose(np.array([4, 6, 5]), merged_faces[2])
    assert_allclose(np.array([5, 7, 6]), merged_faces[3])
