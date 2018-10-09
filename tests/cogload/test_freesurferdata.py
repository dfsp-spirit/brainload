
import os
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import cogload as cl
import cogload.freesurferdata as fsd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(THIS_DIR, os.pardir, 'test_data')
FSAVERAGE_NUM_VERTS_PER_HEMISPHERE = 163842         # number of vertices of the 'fsaverage' subject from FreeSurfer 6.0


def test_get_morphology_data_suffix_for_surface_with_surf_white():
    suffix = fsd.get_morphology_data_suffix_for_surface('white')
    assert suffix == ''


def test_get_morphology_data_suffix_for_surface_with_surf_other():
    suffix = fsd.get_morphology_data_suffix_for_surface('pial')
    assert suffix == '.pial'


def test_read_mgh_file_with_valid_fsaverage_file():
    mgh_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'rh.area.fsaverage.mgh')
    mgh_data, mgh_meta_data = fsd.read_mgh_file(mgh_file)
    assert mgh_meta_data['data_bytes_per_voxel'] == 4
    assert mgh_data.shape == (FSAVERAGE_NUM_VERTS_PER_HEMISPHERE, 1, 1)


def test_merge_meshes():
    m1_vertex_coords = np.array([[0, 0, 0], [5, -5, 0], [5, 5, 0], [10, 5, 0]])
    m1_faces = np.array([[0, 1, 2], [1, 2, 3]])

    m2_vertex_coords = np.array([[0, 0, 0], [10, -10, 0], [10, 10, 0], [15, 10, 0]])
    m2_faces = np.array([[0, 2, 1], [1, 3, 2]])

    merged_verts, merged_faces = fsd.merge_meshes(np.array([[m1_vertex_coords, m1_faces], [m2_vertex_coords, m2_faces]]))
    assert merged_verts.shape == (8, 3)
    assert merged_faces.shape == (4, 3)

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


def test_merge_morphology_data():
    morph_data1 = np.array([0.0, 0.1, 0.2, 0.3])
    morph_data2 = np.array([0.4])
    morph_data3 = np.array([0.5, 0.6])
    merged_data = fsd.merge_morphology_data(np.array([morph_data1, morph_data2, morph_data3]))
    assert merged_data.shape == (7,)
    assert merged_data[0] == pytest.approx(0.0, 0.0001)
    assert merged_data[4] == pytest.approx(0.4, 0.0001)
    assert merged_data[6] == pytest.approx(0.6, 0.0001)


def test_read_fs_surface_file_and_record_meta_data_without_existing_metadata():
    surf_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.white')
    vert_coords, faces, meta_data = fsd.read_fs_surface_file_and_record_meta_data(surf_file, 'lh')
    assert meta_data['lh.num_vertices'] == 149244     # the number is quite arbitrary: the number of vertices is specific for this subject and surface.
    assert meta_data['lh.num_faces'] == 298484    # the number is quite arbitrary: the number of vertices is specific for this subject and surface.
    assert vert_coords.shape == (149244, 3)
    assert faces.shape == (298484, 3)
    assert len(meta_data) == 2


def test_read_fs_surface_file_and_record_meta_data_with_existing_metadata():
    surf_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.white')
    vert_coords, faces, meta_data = fsd.read_fs_surface_file_and_record_meta_data(surf_file, 'lh', meta_data={'this_boy': 'still_exists'})
    assert len(meta_data) == 3
    assert meta_data['this_boy'] == 'still_exists'


def test_read_fs_surface_file_and_record_meta_data_raises_on_wrong_hemisphere_value():
    surf_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.white')
    with pytest.raises(ValueError) as exc_info:
        vert_coords, faces, meta_data = fsd.read_fs_surface_file_and_record_meta_data(surf_file, 'invalid_hemisphere')
    assert 'hemisphere_label must be one of' in str(exc_info.value)

def test_read_fs_morphology_data_file_and_record_meta_data_with_curv_file():
    curv_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.area')
    per_vertex_data, meta_data = fsd.read_fs_morphology_data_file_and_record_meta_data(curv_file, 'lh')
