
import os
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import cogload as cl
import cogload.freesurferdata as fsd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(THIS_DIR, os.pardir, 'test_data')
FSAVERAGE_NUM_VERTS_PER_HEMISPHERE = 163842         # number of vertices of the 'fsaverage' subject from FreeSurfer 6.0

SUBJECT1_SURF_LH_WHITE_NUM_VERTICES = 149244        # this number is quite arbitrary: the number of vertices is specific for this subject and surface.
SUBJECT1_SURF_LH_WHITE_NUM_FACES = 298484           # this number is quite arbitrary: the number of faces is specific for this subject and surface.

SUBJECT1_SURF_RH_WHITE_NUM_VERTICES = 153333        # this number is quite arbitrary: the number of vertices is specific for this subject and surface.
SUBJECT1_SURF_RH_WHITE_NUM_FACES = 306662           # this number is quite arbitrary: the number of faces is specific for this subject and surface.

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
    assert meta_data['lh.num_vertices'] == SUBJECT1_SURF_LH_WHITE_NUM_VERTICES
    assert meta_data['lh.num_faces'] == SUBJECT1_SURF_LH_WHITE_NUM_FACES
    assert meta_data['lh.surf_file'] == surf_file
    assert vert_coords.shape == (SUBJECT1_SURF_LH_WHITE_NUM_VERTICES, 3)
    assert faces.shape == (SUBJECT1_SURF_LH_WHITE_NUM_FACES, 3)
    assert len(meta_data) == 3


def test_read_fs_surface_file_and_record_meta_data_with_existing_metadata():
    surf_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.white')
    vert_coords, faces, meta_data = fsd.read_fs_surface_file_and_record_meta_data(surf_file, 'lh', meta_data={'this_boy': 'still_exists'})
    assert vert_coords.shape == (SUBJECT1_SURF_LH_WHITE_NUM_VERTICES, 3)
    assert faces.shape == (SUBJECT1_SURF_LH_WHITE_NUM_FACES, 3)
    assert len(meta_data) == 4
    assert meta_data['this_boy'] == 'still_exists'


def test_read_fs_surface_file_and_record_meta_data_raises_on_wrong_hemisphere_value():
    surf_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.white')
    with pytest.raises(ValueError) as exc_info:
        vert_coords, faces, meta_data = fsd.read_fs_surface_file_and_record_meta_data(surf_file, 'invalid_hemisphere')
    assert 'hemisphere_label must be one of' in str(exc_info.value)
    assert 'invalid_hemisphere' in str(exc_info.value)


def test_read_fs_morphology_data_file_and_record_meta_data_with_subj1_curv_file_without_existing_metadata():
    morphology_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.area')
    per_vertex_data, meta_data = fsd.read_fs_morphology_data_file_and_record_meta_data(morphology_file, 'lh')
    assert len(meta_data) == 3
    assert meta_data['lh.morphology_file'] == morphology_file
    assert meta_data['lh.morphology_file_format'] == 'curv'
    assert meta_data['lh.num_data_points'] == SUBJECT1_SURF_LH_WHITE_NUM_VERTICES
    assert per_vertex_data.shape == (SUBJECT1_SURF_LH_WHITE_NUM_VERTICES, )


def test_read_fs_morphology_data_file_and_record_meta_data_with_subj1_curv_file_with_existing_metadata():
    morphology_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.area')
    per_vertex_data, meta_data = fsd.read_fs_morphology_data_file_and_record_meta_data(morphology_file, 'lh', meta_data={'this_boy': 'still_exists'})
    assert len(meta_data) == 4
    assert meta_data['this_boy'] == 'still_exists'
    assert per_vertex_data.shape == (SUBJECT1_SURF_LH_WHITE_NUM_VERTICES, )


def test_read_fs_morphology_data_file_and_record_meta_data_with_fsavg_mgh_file_with_existing_metadata():
    morphology_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.area.fsaverage.mgh')
    per_vertex_data, meta_data = fsd.read_fs_morphology_data_file_and_record_meta_data(morphology_file, 'lh', format='mgh', meta_data={'this_boy': 'still_exists'})
    assert len(meta_data) == 4
    assert meta_data['this_boy'] == 'still_exists'
    assert meta_data['lh.morphology_file'] == morphology_file
    assert meta_data['lh.morphology_file_format'] == 'mgh'
    assert meta_data['lh.num_data_points'] == FSAVERAGE_NUM_VERTS_PER_HEMISPHERE
    assert per_vertex_data.shape == (FSAVERAGE_NUM_VERTS_PER_HEMISPHERE, )


def test_read_fs_morphology_data_file_and_record_meta_data_raises_on_wrong_hemisphere_value():
    morphology_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.area')
    with pytest.raises(ValueError) as exc_info:
        per_vertex_data, meta_data = fsd.read_fs_morphology_data_file_and_record_meta_data(morphology_file, 'invalid_hemisphere')
    assert 'hemisphere_label must be one of' in str(exc_info.value)
    assert 'invalid_hemisphere' in str(exc_info.value)


def test_read_fs_morphology_data_file_and_record_meta_data_raises_on_wrong_format_value():
    morphology_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.area')
    with pytest.raises(ValueError) as exc_info:
        per_vertex_data, meta_data = fsd.read_fs_morphology_data_file_and_record_meta_data(morphology_file, 'lh', format='invalid_format')
    assert 'format must be one of' in str(exc_info.value)
    assert 'invalid_format' in str(exc_info.value)

def test_load_subject_mesh_files():
    lh_surf_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.white')
    rh_surf_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'rh.white')
    vert_coords, faces, meta_data = fsd.load_subject_mesh_files(lh_surf_file, rh_surf_file)
    assert meta_data['lh.num_vertices'] == SUBJECT1_SURF_LH_WHITE_NUM_VERTICES
    assert meta_data['lh.num_faces'] == SUBJECT1_SURF_LH_WHITE_NUM_FACES
    assert meta_data['lh.surf_file'] == lh_surf_file
    assert meta_data['rh.num_vertices'] == SUBJECT1_SURF_RH_WHITE_NUM_VERTICES
    assert meta_data['rh.num_faces'] == SUBJECT1_SURF_RH_WHITE_NUM_FACES
    assert meta_data['rh.surf_file'] == rh_surf_file
    assert vert_coords.shape == (SUBJECT1_SURF_LH_WHITE_NUM_VERTICES + SUBJECT1_SURF_RH_WHITE_NUM_VERTICES, 3)
    assert faces.shape == (SUBJECT1_SURF_LH_WHITE_NUM_FACES + SUBJECT1_SURF_RH_WHITE_NUM_FACES, 3)
    assert len(meta_data) == 6

def test_load_subject_mesh_files_preserves_existing_meta_data():
    lh_surf_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.white')
    rh_surf_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'rh.white')
    vert_coords, faces, meta_data = fsd.load_subject_mesh_files(lh_surf_file, rh_surf_file, hemi='both', meta_data={'this_boy': 'still_exists'})
    assert vert_coords.shape == (SUBJECT1_SURF_LH_WHITE_NUM_VERTICES + SUBJECT1_SURF_RH_WHITE_NUM_VERTICES, 3)
    assert faces.shape == (SUBJECT1_SURF_LH_WHITE_NUM_FACES + SUBJECT1_SURF_RH_WHITE_NUM_FACES, 3)
    assert meta_data['this_boy'] == 'still_exists'
    assert len(meta_data) == 7

def test_load_subject_mesh_files_works_with_left_hemisphere_only():
    lh_surf_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.white')
    vert_coords, faces, meta_data = fsd.load_subject_mesh_files(lh_surf_file, None, hemi='lh')
    assert meta_data['lh.num_vertices'] == SUBJECT1_SURF_LH_WHITE_NUM_VERTICES
    assert meta_data['lh.num_faces'] == SUBJECT1_SURF_LH_WHITE_NUM_FACES
    assert meta_data['lh.surf_file'] == lh_surf_file
    assert vert_coords.shape == (SUBJECT1_SURF_LH_WHITE_NUM_VERTICES, 3)
    assert faces.shape == (SUBJECT1_SURF_LH_WHITE_NUM_FACES, 3)
    assert len(meta_data) == 3

def test_load_subject_mesh_files_works_with_right_hemisphere_only():
    rh_surf_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'rh.white')
    vert_coords, faces, meta_data = fsd.load_subject_mesh_files(None, rh_surf_file, hemi='rh')
    assert meta_data['rh.num_vertices'] == SUBJECT1_SURF_RH_WHITE_NUM_VERTICES
    assert meta_data['rh.num_faces'] == SUBJECT1_SURF_RH_WHITE_NUM_FACES
    assert meta_data['rh.surf_file'] == rh_surf_file
    assert vert_coords.shape == (SUBJECT1_SURF_RH_WHITE_NUM_VERTICES, 3)
    assert faces.shape == (SUBJECT1_SURF_RH_WHITE_NUM_FACES, 3)
    assert len(meta_data) == 3

def test_load_subject_morphology_data_files():
    lh_morphology_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.area')
    rh_morphology_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'rh.area')
    morphology_data, meta_data = fsd.load_subject_morphology_data_files(lh_morphology_file, rh_morphology_file)
    assert meta_data['lh.morphology_file'] == lh_morphology_file
    assert meta_data['lh.morphology_file_format'] == 'curv'
    assert meta_data['lh.num_data_points'] == SUBJECT1_SURF_LH_WHITE_NUM_VERTICES
    assert meta_data['rh.morphology_file'] == rh_morphology_file
    assert meta_data['rh.morphology_file_format'] == 'curv'
    assert meta_data['rh.num_data_points'] == SUBJECT1_SURF_RH_WHITE_NUM_VERTICES
    assert len(meta_data) == 6
    assert morphology_data.shape == (SUBJECT1_SURF_LH_WHITE_NUM_VERTICES + SUBJECT1_SURF_RH_WHITE_NUM_VERTICES, )

def test_load_subject_morphology_data_files_preserves_existing_meta_data():
    lh_morphology_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.area')
    rh_morphology_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'rh.area')
    morphology_data, meta_data = fsd.load_subject_morphology_data_files(lh_morphology_file, rh_morphology_file, meta_data={'this_boy': 'still_exists'})
    assert meta_data['this_boy'] == 'still_exists'
    assert len(meta_data) == 7
    assert morphology_data.shape == (SUBJECT1_SURF_LH_WHITE_NUM_VERTICES + SUBJECT1_SURF_RH_WHITE_NUM_VERTICES, )


def test_load_subject_morphology_data_files_works_with_left_hemisphere_only():
    lh_morphology_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.area')
    morphology_data, meta_data = fsd.load_subject_morphology_data_files(lh_morphology_file, None, hemi='lh')
    assert meta_data['lh.morphology_file'] == lh_morphology_file
    assert meta_data['lh.morphology_file_format'] == 'curv'
    assert meta_data['lh.num_data_points'] == SUBJECT1_SURF_LH_WHITE_NUM_VERTICES
    assert len(meta_data) == 3
    assert morphology_data.shape == (SUBJECT1_SURF_LH_WHITE_NUM_VERTICES, )

def test_load_subject_morphology_data_files_works_with_right_hemisphere_only():
    rh_morphology_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'rh.area')
    morphology_data, meta_data = fsd.load_subject_morphology_data_files(None, rh_morphology_file, hemi='rh')
    assert meta_data['rh.morphology_file'] == rh_morphology_file
    assert meta_data['rh.morphology_file_format'] == 'curv'
    assert meta_data['rh.num_data_points'] == SUBJECT1_SURF_RH_WHITE_NUM_VERTICES
    assert len(meta_data) == 3
    assert morphology_data.shape == (SUBJECT1_SURF_RH_WHITE_NUM_VERTICES, )

def test_load_subject_morphology_data_files_raises_on_invalid_format():
    with pytest.raises(ValueError) as exc_info:
        morphology_data, meta_data = fsd.load_subject_morphology_data_files('some_file', 'some_other_file', format='invalid_format')
    assert 'format must be one of' in str(exc_info.value)
    assert 'invalid_format' in str(exc_info.value)

def test_load_subject_morphology_data_files_raises_on_invalid_hemisphere():
    with pytest.raises(ValueError) as exc_info:
        morphology_data, meta_data = fsd.load_subject_morphology_data_files('some_file', 'some_other_file', hemi='invalid_hemisphere')
    assert 'hemi must be one of' in str(exc_info.value)
    assert 'invalid_hemisphere' in str(exc_info.value)

def test_parse_subject():
    vert_coords, faces, morphology_data, meta_data = fsd.parse_subject('subject1', subjects_dir=TEST_DATA_DIR)
    assert len(meta_data) == 18
    expected_lh_morphology_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.area')
    expected_rh_morphology_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'rh.area')
    #assert meta_data['lh.num_vertices'] == SUBJECT1_SURF_LH_WHITE_NUM_VERTICES
    #assert meta_data['lh.num_faces'] == SUBJECT1_SURF_LH_WHITE_NUM_FACES
    #assert meta_data['lh.surf_file'] == lh_surf_file
    #assert meta_data['rh.num_vertices'] == SUBJECT1_SURF_RH_WHITE_NUM_VERTICES
    #assert meta_data['rh.num_faces'] == SUBJECT1_SURF_RH_WHITE_NUM_FACES
    #assert meta_data['rh.surf_file'] == rh_surf_file
