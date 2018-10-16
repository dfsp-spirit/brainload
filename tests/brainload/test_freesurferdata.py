import os
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import brainload as bl
import brainload.freesurferdata as fsd


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(THIS_DIR, os.pardir, 'test_data')

# Respect the environment variable BRAINLOAD_TEST_DATA_DIR if it is set. If not, fall back to default.
TEST_DATA_DIR = os.getenv('BRAINLOAD_TEST_DATA_DIR', TEST_DATA_DIR)

FSAVERAGE_NUM_VERTS_PER_HEMISPHERE = 163842         # number of vertices of the 'fsaverage' subject from FreeSurfer 6.0
FSAVERAGE_NUM_FACES_PER_HEMISPHERE = 327680

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
    assert mgh_meta_data['data_bytespervox'] == 4
    assert len(mgh_meta_data) == 13
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
    assert len(meta_data) == 20
    expected_subjects_dir = TEST_DATA_DIR
    expected_lh_surf_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.white')
    expected_rh_surf_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'rh.white')
    expected_lh_morphology_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.area')
    expected_rh_morphology_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'rh.area')
    assert meta_data['lh.num_vertices'] == SUBJECT1_SURF_LH_WHITE_NUM_VERTICES
    assert meta_data['lh.num_faces'] == SUBJECT1_SURF_LH_WHITE_NUM_FACES
    assert meta_data['lh.surf_file'] == expected_lh_surf_file
    assert meta_data['rh.num_vertices'] == SUBJECT1_SURF_RH_WHITE_NUM_VERTICES
    assert meta_data['rh.num_faces'] == SUBJECT1_SURF_RH_WHITE_NUM_FACES
    assert meta_data['rh.surf_file'] == expected_rh_surf_file

    assert meta_data['lh.morphology_file'] == expected_lh_morphology_file
    assert meta_data['lh.morphology_file_format'] == 'curv'
    assert meta_data['lh.num_data_points'] == SUBJECT1_SURF_LH_WHITE_NUM_VERTICES
    assert meta_data['rh.morphology_file'] == expected_rh_morphology_file
    assert meta_data['rh.morphology_file_format'] == 'curv'
    assert meta_data['rh.num_data_points'] == SUBJECT1_SURF_RH_WHITE_NUM_VERTICES

    assert meta_data['subject_id'] == 'subject1'
    assert meta_data['display_subject'] == 'subject1'
    assert meta_data['subjects_dir'] == expected_subjects_dir
    assert meta_data['surf'] == 'white'
    assert meta_data['display_surf'] == 'white'
    assert meta_data['measure'] == 'area'
    assert meta_data['space'] == 'native_space'
    assert meta_data['hemi'] == 'both'

    assert vert_coords.shape == (SUBJECT1_SURF_LH_WHITE_NUM_VERTICES + SUBJECT1_SURF_RH_WHITE_NUM_VERTICES, 3)
    assert faces.shape == (SUBJECT1_SURF_LH_WHITE_NUM_FACES + SUBJECT1_SURF_RH_WHITE_NUM_FACES, 3)
    assert morphology_data.shape == (SUBJECT1_SURF_LH_WHITE_NUM_VERTICES + SUBJECT1_SURF_RH_WHITE_NUM_VERTICES, )


def test_parse_subject_preserves_existing_meta_data():
    vert_coords, faces, morphology_data, meta_data = fsd.parse_subject('subject1', subjects_dir=TEST_DATA_DIR, meta_data={'this_boy': 'still_exists'})
    assert len(meta_data) == 21
    assert meta_data['this_boy'] == 'still_exists'


def test_parse_subject_raises_on_invalid_hemisphere():
    with pytest.raises(ValueError) as exc_info:
        vert_coords, faces, morphology_data, meta_data = fsd.parse_subject('subject1', subjects_dir=TEST_DATA_DIR, hemi='invalid_hemisphere')
    assert 'hemi must be one of' in str(exc_info.value)
    assert 'invalid_hemisphere' in str(exc_info.value)


def test_parse_subject_works_with_left_hemisphere_only():
    vert_coords, faces, morphology_data, meta_data = fsd.parse_subject('subject1', subjects_dir=TEST_DATA_DIR, hemi='lh')
    assert len(meta_data) == 14
    expected_subjects_dir = TEST_DATA_DIR
    expected_lh_surf_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.white')
    expected_lh_morphology_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.area')
    assert meta_data['lh.num_vertices'] == SUBJECT1_SURF_LH_WHITE_NUM_VERTICES
    assert meta_data['lh.num_faces'] == SUBJECT1_SURF_LH_WHITE_NUM_FACES
    assert meta_data['lh.surf_file'] == expected_lh_surf_file

    assert meta_data['lh.morphology_file'] == expected_lh_morphology_file
    assert meta_data['lh.morphology_file_format'] == 'curv'
    assert meta_data['lh.num_data_points'] == SUBJECT1_SURF_LH_WHITE_NUM_VERTICES

    assert meta_data['subject_id'] == 'subject1'
    assert meta_data['subjects_dir'] == expected_subjects_dir
    assert meta_data['surf'] == 'white'
    assert meta_data['measure'] == 'area'
    assert meta_data['space'] == 'native_space'
    assert meta_data['hemi'] == 'lh'
    assert meta_data['display_subject'] == 'subject1'
    assert meta_data['display_surf'] == 'white'

    assert vert_coords.shape == (SUBJECT1_SURF_LH_WHITE_NUM_VERTICES, 3)
    assert faces.shape == (SUBJECT1_SURF_LH_WHITE_NUM_FACES, 3)
    assert morphology_data.shape == (SUBJECT1_SURF_LH_WHITE_NUM_VERTICES, )


def test_parse_subject_works_with_right_hemisphere_only():
    vert_coords, faces, morphology_data, meta_data = fsd.parse_subject('subject1', subjects_dir=TEST_DATA_DIR, hemi='rh')
    assert len(meta_data) == 14
    expected_subjects_dir = TEST_DATA_DIR
    expected_rh_surf_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'rh.white')
    expected_rh_morphology_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'rh.area')
    assert meta_data['rh.num_vertices'] == SUBJECT1_SURF_RH_WHITE_NUM_VERTICES
    assert meta_data['rh.num_faces'] == SUBJECT1_SURF_RH_WHITE_NUM_FACES
    assert meta_data['rh.surf_file'] == expected_rh_surf_file

    assert meta_data['rh.morphology_file'] == expected_rh_morphology_file
    assert meta_data['rh.morphology_file_format'] == 'curv'
    assert meta_data['rh.num_data_points'] == SUBJECT1_SURF_RH_WHITE_NUM_VERTICES

    assert meta_data['subject_id'] == 'subject1'
    assert meta_data['subjects_dir'] == expected_subjects_dir
    assert meta_data['surf'] == 'white'
    assert meta_data['measure'] == 'area'
    assert meta_data['space'] == 'native_space'
    assert meta_data['hemi'] == 'rh'
    assert meta_data['display_subject'] == 'subject1'
    assert meta_data['display_surf'] == 'white'

    assert vert_coords.shape == (SUBJECT1_SURF_RH_WHITE_NUM_VERTICES, 3)
    assert faces.shape == (SUBJECT1_SURF_RH_WHITE_NUM_FACES, 3)
    assert morphology_data.shape == (SUBJECT1_SURF_RH_WHITE_NUM_VERTICES, )


def test_parse_subject_does_not_load_surface_when_asked_not_to():
    vert_coords, faces, morphology_data, meta_data = fsd.parse_subject('subject1', subjects_dir=TEST_DATA_DIR, load_surface_files=False)
    assert len(meta_data) == 14
    expected_subjects_dir = TEST_DATA_DIR
    expected_lh_morphology_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.area')
    expected_rh_morphology_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'rh.area')

    assert meta_data['lh.morphology_file'] == expected_lh_morphology_file
    assert meta_data['lh.morphology_file_format'] == 'curv'
    assert meta_data['lh.num_data_points'] == SUBJECT1_SURF_LH_WHITE_NUM_VERTICES
    assert meta_data['rh.morphology_file'] == expected_rh_morphology_file
    assert meta_data['rh.morphology_file_format'] == 'curv'
    assert meta_data['rh.num_data_points'] == SUBJECT1_SURF_RH_WHITE_NUM_VERTICES

    assert meta_data['subject_id'] == 'subject1'
    assert meta_data['subjects_dir'] == expected_subjects_dir
    assert meta_data['surf'] == 'white'
    assert meta_data['measure'] == 'area'
    assert meta_data['space'] == 'native_space'
    assert meta_data['hemi'] == 'both'
    assert meta_data['display_subject'] is None
    assert meta_data['display_surf'] is None

    assert vert_coords == None
    assert faces == None
    assert morphology_data.shape == (SUBJECT1_SURF_LH_WHITE_NUM_VERTICES + SUBJECT1_SURF_RH_WHITE_NUM_VERTICES, )


def test_parse_subject_does_not_load_morphology_data_when_asked_not_to():
    vert_coords, faces, morphology_data, meta_data = fsd.parse_subject('subject1', subjects_dir=TEST_DATA_DIR, load_morhology_data=False)
    assert len(meta_data) == 14
    expected_subjects_dir = TEST_DATA_DIR
    expected_lh_surf_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.white')
    expected_rh_surf_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'rh.white')
    assert meta_data['lh.num_vertices'] == SUBJECT1_SURF_LH_WHITE_NUM_VERTICES
    assert meta_data['lh.num_faces'] == SUBJECT1_SURF_LH_WHITE_NUM_FACES
    assert meta_data['lh.surf_file'] == expected_lh_surf_file
    assert meta_data['rh.num_vertices'] == SUBJECT1_SURF_RH_WHITE_NUM_VERTICES
    assert meta_data['rh.num_faces'] == SUBJECT1_SURF_RH_WHITE_NUM_FACES
    assert meta_data['rh.surf_file'] == expected_rh_surf_file

    assert meta_data['subject_id'] == 'subject1'
    assert meta_data['subjects_dir'] == expected_subjects_dir
    assert meta_data['surf'] == 'white'
    assert meta_data['measure'] == None
    assert meta_data['space'] == 'native_space'
    assert meta_data['hemi'] == 'both'
    assert meta_data['display_subject'] == 'subject1'
    assert meta_data['display_surf'] == 'white'

    assert vert_coords.shape == (SUBJECT1_SURF_LH_WHITE_NUM_VERTICES + SUBJECT1_SURF_RH_WHITE_NUM_VERTICES, 3)
    assert faces.shape == (SUBJECT1_SURF_LH_WHITE_NUM_FACES + SUBJECT1_SURF_RH_WHITE_NUM_FACES, 3)
    assert morphology_data is None


def test_parse_subject_standard_space_data():
    expected_subjects_dir = TEST_DATA_DIR
    expected_fsaverage_surf_dir = os.path.join(TEST_DATA_DIR, 'fsaverage', 'surf')
    if not os.path.isdir(expected_fsaverage_surf_dir):
        pytest.skip("Test data for average subject not available: directory '%s' does not exist. You can get it by running './develop/get_test_data_fsaverage.bash' in the repo root." % expected_fsaverage_surf_dir)

    vert_coords, faces, morphology_data, meta_data = fsd.parse_subject_standard_space_data('subject1', subjects_dir=TEST_DATA_DIR)
    assert len(meta_data) == 24
    expected_lh_surf_file = os.path.join(TEST_DATA_DIR, 'fsaverage', 'surf', 'lh.white')
    expected_rh_surf_file = os.path.join(TEST_DATA_DIR, 'fsaverage', 'surf', 'rh.white')
    expected_lh_morphology_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.area.fwhm10.fsaverage.mgh')
    expected_rh_morphology_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'rh.area.fwhm10.fsaverage.mgh')
    assert meta_data['lh.num_vertices'] == FSAVERAGE_NUM_VERTS_PER_HEMISPHERE
    assert meta_data['lh.num_faces'] == FSAVERAGE_NUM_FACES_PER_HEMISPHERE
    assert meta_data['lh.surf_file'] == expected_lh_surf_file
    assert meta_data['rh.num_vertices'] == FSAVERAGE_NUM_VERTS_PER_HEMISPHERE
    assert meta_data['rh.num_faces'] == FSAVERAGE_NUM_FACES_PER_HEMISPHERE
    assert meta_data['rh.surf_file'] == expected_rh_surf_file

    assert meta_data['lh.morphology_file'] == expected_lh_morphology_file
    assert meta_data['lh.morphology_file_format'] == 'mgh'
    assert meta_data['lh.num_data_points'] == FSAVERAGE_NUM_VERTS_PER_HEMISPHERE
    assert meta_data['rh.morphology_file'] == expected_rh_morphology_file
    assert meta_data['rh.morphology_file_format'] == 'mgh'
    assert meta_data['rh.num_data_points'] == FSAVERAGE_NUM_VERTS_PER_HEMISPHERE

    assert meta_data['subject_id'] == 'subject1'
    assert meta_data['subjects_dir'] == expected_subjects_dir
    assert meta_data['average_subject'] == 'fsaverage'
    assert meta_data['average_subjects_dir'] == expected_subjects_dir
    assert meta_data['display_surf'] == 'white'
    assert meta_data['display_subject'] == 'fsaverage'
    assert meta_data['surf'] == 'white'
    assert meta_data['measure'] == 'area'
    assert meta_data['space'] == 'standard_space'
    assert meta_data['hemi'] == 'both'
    assert meta_data['fwhm'] == '10'

    assert vert_coords.shape == (FSAVERAGE_NUM_VERTS_PER_HEMISPHERE * 2, 3)
    assert faces.shape == (FSAVERAGE_NUM_FACES_PER_HEMISPHERE * 2, 3)
    assert morphology_data.shape == (FSAVERAGE_NUM_VERTS_PER_HEMISPHERE * 2, )


def test_parse_subject_standard_space_data_raises_on_invalid_hemisphere():
    with pytest.raises(ValueError) as exc_info:
        vert_coords, faces, morphology_data, meta_data = fsd.parse_subject('subject1', subjects_dir=TEST_DATA_DIR, hemi='invalid_hemisphere')
    assert 'hemi must be one of' in str(exc_info.value)
    assert 'invalid_hemisphere' in str(exc_info.value)


def test_parse_subject_standard_space_data_works_with_left_hemisphere_only():
    expected_subjects_dir = TEST_DATA_DIR
    expected_fsaverage_surf_dir = os.path.join(TEST_DATA_DIR, 'fsaverage', 'surf')
    if not os.path.isdir(expected_fsaverage_surf_dir):
        pytest.skip("Test data for average subject not available: directory '%s' does not exist. You can get it by running './develop/get_test_data_fsaverage.bash' in the repo root." % expected_fsaverage_surf_dir)

    vert_coords, faces, morphology_data, meta_data = fsd.parse_subject_standard_space_data('subject1', subjects_dir=TEST_DATA_DIR, hemi='lh')
    assert len(meta_data) == 18
    expected_lh_surf_file = os.path.join(TEST_DATA_DIR, 'fsaverage', 'surf', 'lh.white')
    expected_lh_morphology_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.area.fwhm10.fsaverage.mgh')
    assert meta_data['lh.num_vertices'] == FSAVERAGE_NUM_VERTS_PER_HEMISPHERE
    assert meta_data['lh.num_faces'] == FSAVERAGE_NUM_FACES_PER_HEMISPHERE
    assert meta_data['lh.surf_file'] == expected_lh_surf_file

    assert meta_data['lh.morphology_file'] == expected_lh_morphology_file
    assert meta_data['lh.morphology_file_format'] == 'mgh'
    assert meta_data['lh.num_data_points'] == FSAVERAGE_NUM_VERTS_PER_HEMISPHERE

    assert meta_data['subject_id'] == 'subject1'
    assert meta_data['subjects_dir'] == expected_subjects_dir
    assert meta_data['surf'] == 'white'
    assert meta_data['measure'] == 'area'
    assert meta_data['space'] == 'standard_space'
    assert meta_data['hemi'] == 'lh'
    assert meta_data['fwhm'] == '10'

    assert vert_coords.shape == (FSAVERAGE_NUM_VERTS_PER_HEMISPHERE, 3)
    assert faces.shape == (FSAVERAGE_NUM_FACES_PER_HEMISPHERE, 3)
    assert morphology_data.shape == (FSAVERAGE_NUM_VERTS_PER_HEMISPHERE, )


def test_parse_subject_standard_space_data_works_with_right_hemisphere_only():
    expected_subjects_dir = TEST_DATA_DIR
    expected_fsaverage_surf_dir = os.path.join(TEST_DATA_DIR, 'fsaverage', 'surf')
    if not os.path.isdir(expected_fsaverage_surf_dir):
        pytest.skip("Test data for average subject not available: directory '%s' does not exist. You can get it by running './develop/get_test_data_fsaverage.bash' in the repo root." % expected_fsaverage_surf_dir)

    vert_coords, faces, morphology_data, meta_data = fsd.parse_subject_standard_space_data('subject1', subjects_dir=TEST_DATA_DIR, hemi='rh')
    assert len(meta_data) == 18
    expected_rh_surf_file = os.path.join(TEST_DATA_DIR, 'fsaverage', 'surf', 'rh.white')
    expected_rh_morphology_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'rh.area.fwhm10.fsaverage.mgh')
    assert meta_data['rh.num_vertices'] == FSAVERAGE_NUM_VERTS_PER_HEMISPHERE
    assert meta_data['rh.num_faces'] == FSAVERAGE_NUM_FACES_PER_HEMISPHERE
    assert meta_data['rh.surf_file'] == expected_rh_surf_file

    assert meta_data['rh.morphology_file'] == expected_rh_morphology_file
    assert meta_data['rh.morphology_file_format'] == 'mgh'
    assert meta_data['rh.num_data_points'] == FSAVERAGE_NUM_VERTS_PER_HEMISPHERE

    assert meta_data['subject_id'] == 'subject1'
    assert meta_data['subjects_dir'] == expected_subjects_dir
    assert meta_data['surf'] == 'white'
    assert meta_data['measure'] == 'area'
    assert meta_data['space'] == 'standard_space'
    assert meta_data['hemi'] == 'rh'

    assert vert_coords.shape == (FSAVERAGE_NUM_VERTS_PER_HEMISPHERE, 3)
    assert faces.shape == (FSAVERAGE_NUM_FACES_PER_HEMISPHERE, 3)
    assert morphology_data.shape == (FSAVERAGE_NUM_VERTS_PER_HEMISPHERE, )


def test_parse_subject_standard_space_data_respects_fwhm_setting_none():
    expected_subjects_dir = TEST_DATA_DIR
    expected_fsaverage_surf_dir = os.path.join(TEST_DATA_DIR, 'fsaverage', 'surf')
    if not os.path.isdir(expected_fsaverage_surf_dir):
        pytest.skip("Test data for average subject not available: directory '%s' does not exist. You can get it by running './develop/get_test_data_fsaverage.bash' in the repo root." % expected_fsaverage_surf_dir)

    vert_coords, faces, morphology_data, meta_data = fsd.parse_subject_standard_space_data('subject1', subjects_dir=TEST_DATA_DIR, fwhm=None)
    assert len(meta_data) == 24
    expected_lh_morphology_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.area.fsaverage.mgh')    # No 'fhwmX' in here!
    expected_rh_morphology_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'rh.area.fsaverage.mgh')

    assert meta_data['lh.morphology_file'] == expected_lh_morphology_file
    assert meta_data['lh.morphology_file_format'] == 'mgh'
    assert meta_data['lh.num_data_points'] == FSAVERAGE_NUM_VERTS_PER_HEMISPHERE
    assert meta_data['rh.morphology_file'] == expected_rh_morphology_file
    assert meta_data['rh.morphology_file_format'] == 'mgh'
    assert meta_data['rh.num_data_points'] == FSAVERAGE_NUM_VERTS_PER_HEMISPHERE


    assert meta_data['fwhm'] == None

    assert vert_coords.shape == (FSAVERAGE_NUM_VERTS_PER_HEMISPHERE * 2, 3)
    assert faces.shape == (FSAVERAGE_NUM_FACES_PER_HEMISPHERE * 2, 3)
    assert morphology_data.shape == (FSAVERAGE_NUM_VERTS_PER_HEMISPHERE * 2, )

def test_parse_subject_standard_space_data_does_not_load_surface_when_asked_not_to():
    expected_subjects_dir = TEST_DATA_DIR
    expected_fsaverage_surf_dir = os.path.join(TEST_DATA_DIR, 'fsaverage', 'surf')
    if not os.path.isdir(expected_fsaverage_surf_dir):
        pytest.skip("Test data for average subject not available: directory '%s' does not exist. You can get it by running './develop/get_test_data_fsaverage.bash' in the repo root." % expected_fsaverage_surf_dir)

    vert_coords, faces, morphology_data, meta_data = fsd.parse_subject_standard_space_data('subject1', subjects_dir=TEST_DATA_DIR, load_surface_files=False)
    assert len(meta_data) == 18
    expected_lh_morphology_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.area.fwhm10.fsaverage.mgh')
    expected_rh_morphology_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'rh.area.fwhm10.fsaverage.mgh')

    assert meta_data['lh.morphology_file'] == expected_lh_morphology_file
    assert meta_data['lh.morphology_file_format'] == 'mgh'
    assert meta_data['lh.num_data_points'] == FSAVERAGE_NUM_VERTS_PER_HEMISPHERE
    assert meta_data['rh.morphology_file'] == expected_rh_morphology_file
    assert meta_data['rh.morphology_file_format'] == 'mgh'
    assert meta_data['rh.num_data_points'] == FSAVERAGE_NUM_VERTS_PER_HEMISPHERE

    assert meta_data['display_subject'] is None
    assert meta_data['display_surf'] is None
    assert meta_data['measure'] == 'area'
    assert meta_data['custom_morphology_files_used'] == False

    assert vert_coords is None
    assert faces is None
    assert morphology_data.shape == (FSAVERAGE_NUM_VERTS_PER_HEMISPHERE * 2, )


def test_parse_subject_standard_space_data_does_not_load_morphology_data_when_asked_not_to():
    expected_subjects_dir = TEST_DATA_DIR
    expected_fsaverage_surf_dir = os.path.join(TEST_DATA_DIR, 'fsaverage', 'surf')
    if not os.path.isdir(expected_fsaverage_surf_dir):
        pytest.skip("Test data for average subject not available: directory '%s' does not exist. You can get it by running './develop/get_test_data_fsaverage.bash' in the repo root." % expected_fsaverage_surf_dir)

    vert_coords, faces, morphology_data, meta_data = fsd.parse_subject_standard_space_data('subject1', subjects_dir=TEST_DATA_DIR, load_morhology_data=False)
    assert len(meta_data) == 17
    expected_lh_surf_file = os.path.join(TEST_DATA_DIR, 'fsaverage', 'surf', 'lh.white')
    expected_rh_surf_file = os.path.join(TEST_DATA_DIR, 'fsaverage', 'surf', 'rh.white')

    assert meta_data['lh.num_vertices'] == FSAVERAGE_NUM_VERTS_PER_HEMISPHERE
    assert meta_data['lh.num_faces'] == FSAVERAGE_NUM_FACES_PER_HEMISPHERE
    assert meta_data['lh.surf_file'] == expected_lh_surf_file
    assert meta_data['rh.num_vertices'] == FSAVERAGE_NUM_VERTS_PER_HEMISPHERE
    assert meta_data['rh.num_faces'] == FSAVERAGE_NUM_FACES_PER_HEMISPHERE
    assert meta_data['rh.surf_file'] == expected_rh_surf_file

    assert meta_data['display_subject'] == 'fsaverage'
    assert meta_data['display_surf'] == 'white'
    assert meta_data['measure'] is None

    assert vert_coords.shape == (FSAVERAGE_NUM_VERTS_PER_HEMISPHERE * 2, 3)
    assert faces.shape == (FSAVERAGE_NUM_FACES_PER_HEMISPHERE * 2, 3)
    assert morphology_data is None


def test_parse_subject_standard_space_data_accepts_custom_morphology_files():
    expected_subjects_dir = TEST_DATA_DIR
    expected_fsaverage_surf_dir = os.path.join(TEST_DATA_DIR, 'fsaverage', 'surf')
    if not os.path.isdir(expected_fsaverage_surf_dir):
        pytest.skip("Test data for average subject not available: directory '%s' does not exist. You can get it by running './develop/get_test_data_fsaverage.bash' in the repo root." % expected_fsaverage_surf_dir)

    custom_morphology_files = { 'lh': 'lh.area.fsaverage.mgh', 'rh': 'rh.area.fsaverage.mgh' }  # You could access these files without the custom_morphology_files argument (by setting fwhm to None explicitely), but using this custom name is convenient because we already have test data named like this.
    vert_coords, faces, morphology_data, meta_data = fsd.parse_subject_standard_space_data('subject1', subjects_dir=TEST_DATA_DIR, custom_morphology_files=custom_morphology_files)
    assert len(meta_data) == 24
    expected_lh_morphology_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.area.fsaverage.mgh')
    expected_rh_morphology_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'rh.area.fsaverage.mgh')
    assert meta_data['lh.morphology_file'] == expected_lh_morphology_file
    assert meta_data['lh.morphology_file_format'] == 'mgh'
    assert meta_data['rh.morphology_file'] == expected_rh_morphology_file
    assert meta_data['rh.morphology_file_format'] == 'mgh'
    assert meta_data['custom_morphology_files_used'] == True


def test_load_group_data():
    expected_subject2_dir = os.path.join(TEST_DATA_DIR, 'subject2')
    if not os.path.isdir(expected_subject2_dir):
        pytest.skip("Test data for subject2 .. subject5 not available: e.g., directory '%s' does not exist. You can get it by running './develop/get_group_data.bash' in the repo root." % expected_subject2_dir)

    group_data, group_meta_data = fsd.load_group_data('area', subjects_dir=TEST_DATA_DIR)

    expected_lh_morphology_file_subject1 = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.area.fwhm10.fsaverage.mgh')
    expected_rh_morphology_file_subject1 = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'rh.area.fwhm10.fsaverage.mgh')
    expected_lh_morphology_file_subject5 = os.path.join(TEST_DATA_DIR, 'subject5', 'surf', 'lh.area.fwhm10.fsaverage.mgh')
    expected_rh_morphology_file_subject5 = os.path.join(TEST_DATA_DIR, 'subject5', 'surf', 'rh.area.fwhm10.fsaverage.mgh')

    assert group_data.shape == (5, FSAVERAGE_NUM_VERTS_PER_HEMISPHERE * 2)   # We have 5 subjects in the subjects.txt file in the test data dir
    assert len(group_meta_data) == 5
    assert len(group_meta_data['subject1']) == 18
    assert group_meta_data['subject1']['lh.morphology_file'] == expected_lh_morphology_file_subject1
    assert group_meta_data['subject1']['rh.morphology_file'] == expected_rh_morphology_file_subject1

    assert group_meta_data['subject1']['display_subject'] is None
    assert group_meta_data['subject1']['display_surf'] is None
    assert group_meta_data['subject1']['measure'] == 'area'

    assert len(group_meta_data['subject5']) == 18
    assert group_meta_data['subject5']['lh.morphology_file'] == expected_lh_morphology_file_subject5
    assert group_meta_data['subject5']['rh.morphology_file'] == expected_rh_morphology_file_subject5


def test_load_group_data_works_with_left_hemisphere_only():
    expected_subject2_dir = os.path.join(TEST_DATA_DIR, 'subject2')
    if not os.path.isdir(expected_subject2_dir):
        pytest.skip("Test data for subject2 .. subject5 not available: e.g., directory '%s' does not exist. You can get it by running './develop/get_group_data.bash' in the repo root." % expected_subject2_dir)

    group_data, group_meta_data = fsd.load_group_data('area', hemi='lh', subjects_dir=TEST_DATA_DIR)

    expected_lh_morphology_file_subject1 = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.area.fwhm10.fsaverage.mgh')
    expected_lh_morphology_file_subject5 = os.path.join(TEST_DATA_DIR, 'subject5', 'surf', 'lh.area.fwhm10.fsaverage.mgh')

    assert group_data.shape == (5, FSAVERAGE_NUM_VERTS_PER_HEMISPHERE)   # We have 5 subjects in the subjects.txt file in the test data dir
    assert len(group_meta_data) == 5
    assert len(group_meta_data['subject1']) == 15
    assert group_meta_data['subject1']['lh.morphology_file'] == expected_lh_morphology_file_subject1
    assert group_meta_data['subject5']['lh.morphology_file'] == expected_lh_morphology_file_subject5
    assert not 'rh.morphology_file' in group_meta_data['subject1']
    assert not 'rh.morphology_file' in group_meta_data['subject5']


def test_load_group_data_works_with_right_hemisphere_only():
    expected_subject2_dir = os.path.join(TEST_DATA_DIR, 'subject2')
    if not os.path.isdir(expected_subject2_dir):
        pytest.skip("Test data for subject2 .. subject5 not available: e.g., directory '%s' does not exist. You can get it by running './develop/get_group_data.bash' in the repo root." % expected_subject2_dir)

    group_data, group_meta_data = fsd.load_group_data('area', hemi='rh', subjects_dir=TEST_DATA_DIR)

    expected_rh_morphology_file_subject1 = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'rh.area.fwhm10.fsaverage.mgh')
    expected_rh_morphology_file_subject5 = os.path.join(TEST_DATA_DIR, 'subject5', 'surf', 'rh.area.fwhm10.fsaverage.mgh')

    assert group_data.shape == (5, FSAVERAGE_NUM_VERTS_PER_HEMISPHERE)   # We have 5 subjects in the subjects.txt file in the test data dir
    assert len(group_meta_data) == 5
    assert len(group_meta_data['subject1']) == 15
    assert group_meta_data['subject1']['rh.morphology_file'] == expected_rh_morphology_file_subject1
    assert group_meta_data['subject5']['rh.morphology_file'] == expected_rh_morphology_file_subject5
    assert not 'lh.morphology_file' in group_meta_data['subject1']
    assert not 'lh.morphology_file' in group_meta_data['subject5']


def test_load_group_data_works_with_custom_morphology_file_templates_using_variables():
    expected_subject2_dir = os.path.join(TEST_DATA_DIR, 'subject2')
    if not os.path.isdir(expected_subject2_dir):
        pytest.skip("Test data for subject2 .. subject5 not available: e.g., directory '%s' does not exist. You can get it by running './develop/get_group_data.bash' in the repo root." % expected_subject2_dir)

    morphology_template = '${HEMI}.${MEASURE}.${AVERAGE_SUBJECT}.mgh'
    custom_morphology_file_templates = {'lh': morphology_template, 'rh': morphology_template}
    group_data, group_meta_data = fsd.load_group_data('area', hemi='both', subjects_dir=TEST_DATA_DIR, custom_morphology_file_templates=custom_morphology_file_templates)

    expected_lh_morphology_file_subject1 = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.area.fsaverage.mgh')
    expected_rh_morphology_file_subject1 = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'rh.area.fsaverage.mgh')
    expected_lh_morphology_file_subject5 = os.path.join(TEST_DATA_DIR, 'subject5', 'surf', 'lh.area.fsaverage.mgh')
    expected_rh_morphology_file_subject5 = os.path.join(TEST_DATA_DIR, 'subject5', 'surf', 'rh.area.fsaverage.mgh')

    assert group_data.shape == (5, FSAVERAGE_NUM_VERTS_PER_HEMISPHERE * 2)   # We have 5 subjects in the subjects.txt file in the test data dir
    assert len(group_meta_data) == 5
    assert len(group_meta_data['subject1']) == 20
    assert group_meta_data['subject1']['lh.morphology_file'] == expected_lh_morphology_file_subject1
    assert group_meta_data['subject1']['rh.morphology_file'] == expected_rh_morphology_file_subject1
    assert group_meta_data['subject5']['lh.morphology_file'] == expected_lh_morphology_file_subject5
    assert group_meta_data['subject5']['rh.morphology_file'] == expected_rh_morphology_file_subject5


def test_load_group_data_works_with_custom_morphology_file_templates_using_hardcoded_filenames():
    expected_subject2_dir = os.path.join(TEST_DATA_DIR, 'subject2')
    if not os.path.isdir(expected_subject2_dir):
        pytest.skip("Test data for subject2 .. subject5 not available: e.g., directory '%s' does not exist. You can get it by running './develop/get_group_data.bash' in the repo root." % expected_subject2_dir)

    custom_morphology_file_templates = {'lh': 'lh.area.fsaverage.mgh', 'rh': 'rh.area.fsaverage.mgh'}     # nobody forces you to use any variables
    group_data, group_meta_data = fsd.load_group_data('area', hemi='both', subjects_dir=TEST_DATA_DIR, custom_morphology_file_templates=custom_morphology_file_templates)

    expected_lh_morphology_file_subject1 = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.area.fsaverage.mgh')
    expected_rh_morphology_file_subject1 = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'rh.area.fsaverage.mgh')
    expected_lh_morphology_file_subject5 = os.path.join(TEST_DATA_DIR, 'subject5', 'surf', 'lh.area.fsaverage.mgh')
    expected_rh_morphology_file_subject5 = os.path.join(TEST_DATA_DIR, 'subject5', 'surf', 'rh.area.fsaverage.mgh')

    assert group_data.shape == (5, FSAVERAGE_NUM_VERTS_PER_HEMISPHERE * 2)   # We have 5 subjects in the subjects.txt file in the test data dir
    assert len(group_meta_data) == 5
    assert len(group_meta_data['subject1']) == 20
    assert group_meta_data['subject1']['lh.morphology_file'] == expected_lh_morphology_file_subject1
    assert group_meta_data['subject1']['rh.morphology_file'] == expected_rh_morphology_file_subject1
    assert group_meta_data['subject5']['lh.morphology_file'] == expected_lh_morphology_file_subject5
    assert group_meta_data['subject5']['rh.morphology_file'] == expected_rh_morphology_file_subject5
