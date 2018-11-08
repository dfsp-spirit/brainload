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

def test_get_morphometry_data_suffix_for_surface_with_surf_white():
    suffix = fsd._get_morphometry_data_suffix_for_surface('white')
    assert suffix == ''


def test_get_morphometry_data_suffix_for_surface_with_surf_other():
    suffix = fsd._get_morphometry_data_suffix_for_surface('pial')
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

    merged_verts, merged_faces = fsd._merge_meshes(np.array([[m1_vertex_coords, m1_faces], [m2_vertex_coords, m2_faces]]))
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


def test_merge_morphometry_data():
    morph_data1 = np.array([0.0, 0.1, 0.2, 0.3])
    morph_data2 = np.array([0.4])
    morph_data3 = np.array([0.5, 0.6])
    merged_data = fsd.merge_morphometry_data(np.array([morph_data1, morph_data2, morph_data3]))
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


def test_read_fs_morphometry_data_file_and_record_meta_data_with_subj1_curv_file_without_existing_metadata():
    morphometry_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.area')
    per_vertex_data, meta_data = fsd.read_fs_morphometry_data_file_and_record_meta_data(morphometry_file, 'lh')
    assert len(meta_data) == 3
    assert meta_data['lh.morphometry_file'] == morphometry_file
    assert meta_data['lh.morphometry_file_format'] == 'curv'
    assert meta_data['lh.num_data_points'] == SUBJECT1_SURF_LH_WHITE_NUM_VERTICES
    assert per_vertex_data.shape == (SUBJECT1_SURF_LH_WHITE_NUM_VERTICES, )


def test_read_fs_morphometry_data_file_and_record_meta_data_with_subj1_curv_file_with_existing_metadata():
    morphometry_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.area')
    per_vertex_data, meta_data = fsd.read_fs_morphometry_data_file_and_record_meta_data(morphometry_file, 'lh', meta_data={'this_boy': 'still_exists'})
    assert len(meta_data) == 4
    assert meta_data['this_boy'] == 'still_exists'
    assert per_vertex_data.shape == (SUBJECT1_SURF_LH_WHITE_NUM_VERTICES, )


def test_read_fs_morphometry_data_file_and_record_meta_data_with_fsavg_mgh_file_with_existing_metadata():
    morphometry_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.area.fsaverage.mgh')
    per_vertex_data, meta_data = fsd.read_fs_morphometry_data_file_and_record_meta_data(morphometry_file, 'lh', format='mgh', meta_data={'this_boy': 'still_exists'})
    assert len(meta_data) == 4
    assert meta_data['this_boy'] == 'still_exists'
    assert meta_data['lh.morphometry_file'] == morphometry_file
    assert meta_data['lh.morphometry_file_format'] == 'mgh'
    assert meta_data['lh.num_data_points'] == FSAVERAGE_NUM_VERTS_PER_HEMISPHERE
    assert per_vertex_data.shape == (FSAVERAGE_NUM_VERTS_PER_HEMISPHERE, )


def test_read_fs_morphometry_data_file_and_record_meta_data_raises_on_wrong_hemisphere_value():
    morphometry_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.area')
    with pytest.raises(ValueError) as exc_info:
        per_vertex_data, meta_data = fsd.read_fs_morphometry_data_file_and_record_meta_data(morphometry_file, 'invalid_hemisphere')
    assert 'hemisphere_label must be one of' in str(exc_info.value)
    assert 'invalid_hemisphere' in str(exc_info.value)


def test_read_fs_morphometry_data_file_and_record_meta_data_raises_on_wrong_format_value():
    morphometry_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.area')
    with pytest.raises(ValueError) as exc_info:
        per_vertex_data, meta_data = fsd.read_fs_morphometry_data_file_and_record_meta_data(morphometry_file, 'lh', format='invalid_format')
    assert 'format must be one of' in str(exc_info.value)
    assert 'invalid_format' in str(exc_info.value)


def test_load_subject_mesh_files_raises_on_invalid_hemi():
    lh_surf_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.white')
    rh_surf_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'rh.white')
    with pytest.raises(ValueError) as exc_info:
        vert_coords, faces, meta_data = fsd.load_subject_mesh_files(lh_surf_file, rh_surf_file, hemi='invalid_hemisphere')
    assert 'hemi must be one of' in str(exc_info.value)
    assert 'invalid_hemisphere' in str(exc_info.value)


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


def test_load_subject_morphometry_data_files():
    lh_morphometry_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.area')
    rh_morphometry_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'rh.area')
    morphometry_data, meta_data = fsd.load_subject_morphometry_data_files(lh_morphometry_file, rh_morphometry_file)
    assert meta_data['lh.morphometry_file'] == lh_morphometry_file
    assert meta_data['lh.morphometry_file_format'] == 'curv'
    assert meta_data['lh.num_data_points'] == SUBJECT1_SURF_LH_WHITE_NUM_VERTICES
    assert meta_data['rh.morphometry_file'] == rh_morphometry_file
    assert meta_data['rh.morphometry_file_format'] == 'curv'
    assert meta_data['rh.num_data_points'] == SUBJECT1_SURF_RH_WHITE_NUM_VERTICES
    assert len(meta_data) == 6
    assert morphometry_data.shape == (SUBJECT1_SURF_LH_WHITE_NUM_VERTICES + SUBJECT1_SURF_RH_WHITE_NUM_VERTICES, )


def test_load_subject_morphometry_data_files_preserves_existing_meta_data():
    lh_morphometry_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.area')
    rh_morphometry_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'rh.area')
    morphometry_data, meta_data = fsd.load_subject_morphometry_data_files(lh_morphometry_file, rh_morphometry_file, meta_data={'this_boy': 'still_exists'})
    assert meta_data['this_boy'] == 'still_exists'
    assert len(meta_data) == 7
    assert morphometry_data.shape == (SUBJECT1_SURF_LH_WHITE_NUM_VERTICES + SUBJECT1_SURF_RH_WHITE_NUM_VERTICES, )


def test_load_subject_morphometry_data_files_works_with_left_hemisphere_only():
    lh_morphometry_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.area')
    morphometry_data, meta_data = fsd.load_subject_morphometry_data_files(lh_morphometry_file, None, hemi='lh')
    assert meta_data['lh.morphometry_file'] == lh_morphometry_file
    assert meta_data['lh.morphometry_file_format'] == 'curv'
    assert meta_data['lh.num_data_points'] == SUBJECT1_SURF_LH_WHITE_NUM_VERTICES
    assert len(meta_data) == 3
    assert morphometry_data.shape == (SUBJECT1_SURF_LH_WHITE_NUM_VERTICES, )


def test_load_subject_morphometry_data_files_works_with_right_hemisphere_only():
    rh_morphometry_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'rh.area')
    morphometry_data, meta_data = fsd.load_subject_morphometry_data_files(None, rh_morphometry_file, hemi='rh')
    assert meta_data['rh.morphometry_file'] == rh_morphometry_file
    assert meta_data['rh.morphometry_file_format'] == 'curv'
    assert meta_data['rh.num_data_points'] == SUBJECT1_SURF_RH_WHITE_NUM_VERTICES
    assert len(meta_data) == 3
    assert morphometry_data.shape == (SUBJECT1_SURF_RH_WHITE_NUM_VERTICES, )

def test_load_subject_morphometry_data_files_raises_on_invalid_format():
    with pytest.raises(ValueError) as exc_info:
        morphometry_data, meta_data = fsd.load_subject_morphometry_data_files('some_file', 'some_other_file', format='invalid_format')
    assert 'format must be one of' in str(exc_info.value)
    assert 'invalid_format' in str(exc_info.value)


def test_load_subject_morphometry_data_files_raises_on_invalid_hemisphere():
    with pytest.raises(ValueError) as exc_info:
        morphometry_data, meta_data = fsd.load_subject_morphometry_data_files('some_file', 'some_other_file', hemi='invalid_hemisphere')
    assert 'hemi must be one of' in str(exc_info.value)
    assert 'invalid_hemisphere' in str(exc_info.value)


def test_parse_subject():
    vert_coords, faces, morphometry_data, meta_data = bl.subject('subject1', subjects_dir=TEST_DATA_DIR)
    assert len(meta_data) == 20
    expected_subjects_dir = TEST_DATA_DIR
    expected_lh_surf_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.white')
    expected_rh_surf_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'rh.white')
    expected_lh_morphometry_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.area')
    expected_rh_morphometry_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'rh.area')
    assert meta_data['lh.num_vertices'] == SUBJECT1_SURF_LH_WHITE_NUM_VERTICES
    assert meta_data['lh.num_faces'] == SUBJECT1_SURF_LH_WHITE_NUM_FACES
    assert meta_data['lh.surf_file'] == expected_lh_surf_file
    assert meta_data['rh.num_vertices'] == SUBJECT1_SURF_RH_WHITE_NUM_VERTICES
    assert meta_data['rh.num_faces'] == SUBJECT1_SURF_RH_WHITE_NUM_FACES
    assert meta_data['rh.surf_file'] == expected_rh_surf_file

    assert meta_data['lh.morphometry_file'] == expected_lh_morphometry_file
    assert meta_data['lh.morphometry_file_format'] == 'curv'
    assert meta_data['lh.num_data_points'] == SUBJECT1_SURF_LH_WHITE_NUM_VERTICES
    assert meta_data['rh.morphometry_file'] == expected_rh_morphometry_file
    assert meta_data['rh.morphometry_file_format'] == 'curv'
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
    assert morphometry_data.shape == (SUBJECT1_SURF_LH_WHITE_NUM_VERTICES + SUBJECT1_SURF_RH_WHITE_NUM_VERTICES, )


def test_parse_subject_preserves_existing_meta_data():
    vert_coords, faces, morphometry_data, meta_data = bl.subject('subject1', subjects_dir=TEST_DATA_DIR, meta_data={'this_boy': 'still_exists'})
    assert len(meta_data) == 21
    assert meta_data['this_boy'] == 'still_exists'


def test_parse_subject_raises_on_invalid_hemisphere():
    with pytest.raises(ValueError) as exc_info:
        vert_coords, faces, morphometry_data, meta_data = bl.subject('subject1', subjects_dir=TEST_DATA_DIR, hemi='invalid_hemisphere')
    assert 'hemi must be one of' in str(exc_info.value)
    assert 'invalid_hemisphere' in str(exc_info.value)


def test_parse_subject_works_with_left_hemisphere_only():
    vert_coords, faces, morphometry_data, meta_data = bl.subject('subject1', subjects_dir=TEST_DATA_DIR, hemi='lh')
    assert len(meta_data) == 14
    expected_subjects_dir = TEST_DATA_DIR
    expected_lh_surf_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.white')
    expected_lh_morphometry_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.area')
    assert meta_data['lh.num_vertices'] == SUBJECT1_SURF_LH_WHITE_NUM_VERTICES
    assert meta_data['lh.num_faces'] == SUBJECT1_SURF_LH_WHITE_NUM_FACES
    assert meta_data['lh.surf_file'] == expected_lh_surf_file

    assert meta_data['lh.morphometry_file'] == expected_lh_morphometry_file
    assert meta_data['lh.morphometry_file_format'] == 'curv'
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
    assert morphometry_data.shape == (SUBJECT1_SURF_LH_WHITE_NUM_VERTICES, )


def test_parse_subject_works_with_single_hemispheres_data_correct():
    lh_vert_coords, lh_faces, lh_morphometry_data, lh_meta_data = bl.subject('subject1', subjects_dir=TEST_DATA_DIR, hemi='lh')
    rh_vert_coords, rh_faces, rh_morphometry_data, rh_meta_data = bl.subject('subject1', subjects_dir=TEST_DATA_DIR, hemi='rh')
    both_vert_coords, both_faces, both_morphometry_data, both_meta_data = bl.subject('subject1', subjects_dir=TEST_DATA_DIR, hemi='both')

    assert lh_vert_coords.shape == (SUBJECT1_SURF_LH_WHITE_NUM_VERTICES, 3)
    assert lh_faces.shape == (SUBJECT1_SURF_LH_WHITE_NUM_FACES, 3)
    assert lh_morphometry_data.shape == (SUBJECT1_SURF_LH_WHITE_NUM_VERTICES, )

    assert rh_vert_coords.shape == (SUBJECT1_SURF_RH_WHITE_NUM_VERTICES, 3)
    assert rh_faces.shape == (SUBJECT1_SURF_RH_WHITE_NUM_FACES, 3)
    assert rh_morphometry_data.shape == (SUBJECT1_SURF_RH_WHITE_NUM_VERTICES, )

    assert both_vert_coords.shape == (SUBJECT1_SURF_LH_WHITE_NUM_VERTICES + SUBJECT1_SURF_RH_WHITE_NUM_VERTICES, 3)
    assert both_faces.shape == (SUBJECT1_SURF_LH_WHITE_NUM_FACES + SUBJECT1_SURF_RH_WHITE_NUM_FACES, 3)
    assert both_morphometry_data.shape == (SUBJECT1_SURF_LH_WHITE_NUM_VERTICES + SUBJECT1_SURF_RH_WHITE_NUM_VERTICES, )

    # Test vertices (i.e., their consistency between both, lh and rh data)
    for vert_idx in range(5000, 5100):
        assert lh_vert_coords[vert_idx][0] == pytest.approx(both_vert_coords[vert_idx][0], 0.01)   # x coord
        assert lh_vert_coords[vert_idx][1] == pytest.approx(both_vert_coords[vert_idx][1], 0.01)   # y coord
        assert lh_vert_coords[vert_idx][2] == pytest.approx(both_vert_coords[vert_idx][2], 0.01)   # z coord

    rh_vertex_offset = SUBJECT1_SURF_LH_WHITE_NUM_VERTICES
    for vert_idx in range(5000, 5100):
        assert rh_vert_coords[vert_idx][0] == pytest.approx(both_vert_coords[vert_idx + rh_vertex_offset][0], 0.01)
        assert rh_vert_coords[vert_idx][1] == pytest.approx(both_vert_coords[vert_idx + rh_vertex_offset][1], 0.01)
        assert rh_vert_coords[vert_idx][2] == pytest.approx(both_vert_coords[vert_idx + rh_vertex_offset][2], 0.01)

    # Test faces (i.e., their consistency between both, lh and rh data)
    for face_idx in range(5000, 5100):
        assert lh_faces[face_idx][0] == both_faces[face_idx][0]   # first vertex index of 3-face. Theses are integers, so no need for approx.
        assert lh_faces[face_idx][1] == both_faces[face_idx][1]   # second vertex index of 3-face
        assert lh_faces[face_idx][2] == both_faces[face_idx][2]   # third vertex index of 3-face

    rh_face_offset = SUBJECT1_SURF_LH_WHITE_NUM_FACES
    for face_idx in range(5000, 5100):
        assert rh_faces[face_idx][0]+SUBJECT1_SURF_LH_WHITE_NUM_VERTICES == both_faces[face_idx + rh_face_offset][0]
        assert rh_faces[face_idx][1]+SUBJECT1_SURF_LH_WHITE_NUM_VERTICES == both_faces[face_idx + rh_face_offset][1]
        assert rh_faces[face_idx][2]+SUBJECT1_SURF_LH_WHITE_NUM_VERTICES == both_faces[face_idx + rh_face_offset][2]

    # Test morphometry data (i.e., their consistency between both, lh and rh data)
    for vert_data_idx in range(5000, 5100):
        assert lh_morphometry_data[vert_data_idx] == pytest.approx(both_morphometry_data[vert_data_idx], 0.01)

    rh_data_offset = SUBJECT1_SURF_LH_WHITE_NUM_VERTICES
    for vert_data_idx in range(5000, 5100):
        assert rh_morphometry_data[vert_data_idx] == both_morphometry_data[vert_data_idx + rh_data_offset]


def test_parse_subject_works_with_right_hemisphere_only():
    vert_coords, faces, morphometry_data, meta_data = bl.subject('subject1', subjects_dir=TEST_DATA_DIR, hemi='rh')
    assert len(meta_data) == 14
    expected_subjects_dir = TEST_DATA_DIR
    expected_rh_surf_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'rh.white')
    expected_rh_morphometry_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'rh.area')
    assert meta_data['rh.num_vertices'] == SUBJECT1_SURF_RH_WHITE_NUM_VERTICES
    assert meta_data['rh.num_faces'] == SUBJECT1_SURF_RH_WHITE_NUM_FACES
    assert meta_data['rh.surf_file'] == expected_rh_surf_file

    assert meta_data['rh.morphometry_file'] == expected_rh_morphometry_file
    assert meta_data['rh.morphometry_file_format'] == 'curv'
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
    assert morphometry_data.shape == (SUBJECT1_SURF_RH_WHITE_NUM_VERTICES, )


def test_parse_subject_does_not_load_surface_when_asked_not_to():
    vert_coords, faces, morphometry_data, meta_data = bl.subject('subject1', subjects_dir=TEST_DATA_DIR, load_surface_files=False)
    assert len(meta_data) == 14
    expected_subjects_dir = TEST_DATA_DIR
    expected_lh_morphometry_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.area')
    expected_rh_morphometry_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'rh.area')

    assert meta_data['lh.morphometry_file'] == expected_lh_morphometry_file
    assert meta_data['lh.morphometry_file_format'] == 'curv'
    assert meta_data['lh.num_data_points'] == SUBJECT1_SURF_LH_WHITE_NUM_VERTICES
    assert meta_data['rh.morphometry_file'] == expected_rh_morphometry_file
    assert meta_data['rh.morphometry_file_format'] == 'curv'
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
    assert morphometry_data.shape == (SUBJECT1_SURF_LH_WHITE_NUM_VERTICES + SUBJECT1_SURF_RH_WHITE_NUM_VERTICES, )


def test_parse_subject_does_not_load_morphometry_data_when_asked_not_to():
    vert_coords, faces, morphometry_data, meta_data = bl.subject('subject1', subjects_dir=TEST_DATA_DIR, load_morphometry_data=False)
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
    assert morphometry_data is None


def test_parse_subject_standard_space_data():
    expected_subjects_dir = TEST_DATA_DIR
    expected_fsaverage_surf_dir = os.path.join(TEST_DATA_DIR, 'fsaverage', 'surf')
    if not os.path.isdir(expected_fsaverage_surf_dir):
        pytest.skip("Test data missing: e.g., directory '%s' does not exist. You can get all test data by running './develop/get_test_data_all.bash' in the repo root." % expected_fsaverage_surf_dir)

    vert_coords, faces, morphometry_data, meta_data = bl.subject_avg('subject1', subjects_dir=TEST_DATA_DIR)
    assert len(meta_data) == 24
    expected_lh_surf_file = os.path.join(TEST_DATA_DIR, 'fsaverage', 'surf', 'lh.white')
    expected_rh_surf_file = os.path.join(TEST_DATA_DIR, 'fsaverage', 'surf', 'rh.white')
    expected_lh_morphometry_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.area.fwhm10.fsaverage.mgh')
    expected_rh_morphometry_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'rh.area.fwhm10.fsaverage.mgh')
    assert meta_data['lh.num_vertices'] == FSAVERAGE_NUM_VERTS_PER_HEMISPHERE
    assert meta_data['lh.num_faces'] == FSAVERAGE_NUM_FACES_PER_HEMISPHERE
    assert meta_data['lh.surf_file'] == expected_lh_surf_file
    assert meta_data['rh.num_vertices'] == FSAVERAGE_NUM_VERTS_PER_HEMISPHERE
    assert meta_data['rh.num_faces'] == FSAVERAGE_NUM_FACES_PER_HEMISPHERE
    assert meta_data['rh.surf_file'] == expected_rh_surf_file

    assert meta_data['lh.morphometry_file'] == expected_lh_morphometry_file
    assert meta_data['lh.morphometry_file_format'] == 'mgh'
    assert meta_data['lh.num_data_points'] == FSAVERAGE_NUM_VERTS_PER_HEMISPHERE
    assert meta_data['rh.morphometry_file'] == expected_rh_morphometry_file
    assert meta_data['rh.morphometry_file_format'] == 'mgh'
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
    assert morphometry_data.shape == (FSAVERAGE_NUM_VERTS_PER_HEMISPHERE * 2, )


def test_parse_subject_standard_space_data_raises_on_invalid_hemisphere():
    with pytest.raises(ValueError) as exc_info:
        vert_coords, faces, morphometry_data, meta_data = bl.subject('subject1', subjects_dir=TEST_DATA_DIR, hemi='invalid_hemisphere')
    assert 'hemi must be one of' in str(exc_info.value)
    assert 'invalid_hemisphere' in str(exc_info.value)


def test_parse_subject_standard_space_data_works_with_left_hemisphere_only():
    expected_subjects_dir = TEST_DATA_DIR
    expected_fsaverage_surf_dir = os.path.join(TEST_DATA_DIR, 'fsaverage', 'surf')
    if not os.path.isdir(expected_fsaverage_surf_dir):
        pytest.skip("Test data missing: e.g., directory '%s' does not exist. You can get all test data by running './develop/get_test_data_all.bash' in the repo root." % expected_fsaverage_surf_dir)

    vert_coords, faces, morphometry_data, meta_data = bl.subject_avg('subject1', subjects_dir=TEST_DATA_DIR, hemi='lh')
    assert len(meta_data) == 18
    expected_lh_surf_file = os.path.join(TEST_DATA_DIR, 'fsaverage', 'surf', 'lh.white')
    expected_lh_morphometry_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.area.fwhm10.fsaverage.mgh')
    assert meta_data['lh.num_vertices'] == FSAVERAGE_NUM_VERTS_PER_HEMISPHERE
    assert meta_data['lh.num_faces'] == FSAVERAGE_NUM_FACES_PER_HEMISPHERE
    assert meta_data['lh.surf_file'] == expected_lh_surf_file

    assert meta_data['lh.morphometry_file'] == expected_lh_morphometry_file
    assert meta_data['lh.morphometry_file_format'] == 'mgh'
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
    assert morphometry_data.shape == (FSAVERAGE_NUM_VERTS_PER_HEMISPHERE, )


def test_parse_subject_standard_space_data_works_with_right_hemisphere_only():
    expected_subjects_dir = TEST_DATA_DIR
    expected_fsaverage_surf_dir = os.path.join(TEST_DATA_DIR, 'fsaverage', 'surf')
    if not os.path.isdir(expected_fsaverage_surf_dir):
        pytest.skip("Test data missing: e.g., directory '%s' does not exist. You can get all test data by running './develop/get_test_data_all.bash' in the repo root." % expected_fsaverage_surf_dir)

    vert_coords, faces, morphometry_data, meta_data = bl.subject_avg('subject1', subjects_dir=TEST_DATA_DIR, hemi='rh')
    assert len(meta_data) == 18
    expected_rh_surf_file = os.path.join(TEST_DATA_DIR, 'fsaverage', 'surf', 'rh.white')
    expected_rh_morphometry_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'rh.area.fwhm10.fsaverage.mgh')
    assert meta_data['rh.num_vertices'] == FSAVERAGE_NUM_VERTS_PER_HEMISPHERE
    assert meta_data['rh.num_faces'] == FSAVERAGE_NUM_FACES_PER_HEMISPHERE
    assert meta_data['rh.surf_file'] == expected_rh_surf_file

    assert meta_data['rh.morphometry_file'] == expected_rh_morphometry_file
    assert meta_data['rh.morphometry_file_format'] == 'mgh'
    assert meta_data['rh.num_data_points'] == FSAVERAGE_NUM_VERTS_PER_HEMISPHERE

    assert meta_data['subject_id'] == 'subject1'
    assert meta_data['subjects_dir'] == expected_subjects_dir
    assert meta_data['surf'] == 'white'
    assert meta_data['measure'] == 'area'
    assert meta_data['space'] == 'standard_space'
    assert meta_data['hemi'] == 'rh'

    assert vert_coords.shape == (FSAVERAGE_NUM_VERTS_PER_HEMISPHERE, 3)
    assert faces.shape == (FSAVERAGE_NUM_FACES_PER_HEMISPHERE, 3)
    assert morphometry_data.shape == (FSAVERAGE_NUM_VERTS_PER_HEMISPHERE, )


def test_parse_subject_standard_space_data_respects_fwhm_setting_none():
    expected_subjects_dir = TEST_DATA_DIR
    expected_fsaverage_surf_dir = os.path.join(TEST_DATA_DIR, 'fsaverage', 'surf')
    if not os.path.isdir(expected_fsaverage_surf_dir):
        pytest.skip("Test data missing: e.g., directory '%s' does not exist. You can get all test data by running './develop/get_test_data_all.bash' in the repo root." % expected_fsaverage_surf_dir)

    vert_coords, faces, morphometry_data, meta_data = bl.subject_avg('subject1', subjects_dir=TEST_DATA_DIR, fwhm=None)
    assert len(meta_data) == 24
    expected_lh_morphometry_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.area.fsaverage.mgh')    # No 'fhwmX' in here!
    expected_rh_morphometry_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'rh.area.fsaverage.mgh')

    assert meta_data['lh.morphometry_file'] == expected_lh_morphometry_file
    assert meta_data['lh.morphometry_file_format'] == 'mgh'
    assert meta_data['lh.num_data_points'] == FSAVERAGE_NUM_VERTS_PER_HEMISPHERE
    assert meta_data['rh.morphometry_file'] == expected_rh_morphometry_file
    assert meta_data['rh.morphometry_file_format'] == 'mgh'
    assert meta_data['rh.num_data_points'] == FSAVERAGE_NUM_VERTS_PER_HEMISPHERE


    assert meta_data['fwhm'] == None

    assert vert_coords.shape == (FSAVERAGE_NUM_VERTS_PER_HEMISPHERE * 2, 3)
    assert faces.shape == (FSAVERAGE_NUM_FACES_PER_HEMISPHERE * 2, 3)
    assert morphometry_data.shape == (FSAVERAGE_NUM_VERTS_PER_HEMISPHERE * 2, )

def test_parse_subject_standard_space_data_does_not_load_surface_when_asked_not_to():
    expected_subjects_dir = TEST_DATA_DIR
    expected_fsaverage_surf_dir = os.path.join(TEST_DATA_DIR, 'fsaverage', 'surf')
    if not os.path.isdir(expected_fsaverage_surf_dir):
        pytest.skip("Test data missing: e.g., directory '%s' does not exist. You can get all test data by running './develop/get_test_data_all.bash' in the repo root." % expected_fsaverage_surf_dir)

    vert_coords, faces, morphometry_data, meta_data = bl.subject_avg('subject1', subjects_dir=TEST_DATA_DIR, load_surface_files=False)
    assert len(meta_data) == 18
    expected_lh_morphometry_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.area.fwhm10.fsaverage.mgh')
    expected_rh_morphometry_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'rh.area.fwhm10.fsaverage.mgh')

    assert meta_data['lh.morphometry_file'] == expected_lh_morphometry_file
    assert meta_data['lh.morphometry_file_format'] == 'mgh'
    assert meta_data['lh.num_data_points'] == FSAVERAGE_NUM_VERTS_PER_HEMISPHERE
    assert meta_data['rh.morphometry_file'] == expected_rh_morphometry_file
    assert meta_data['rh.morphometry_file_format'] == 'mgh'
    assert meta_data['rh.num_data_points'] == FSAVERAGE_NUM_VERTS_PER_HEMISPHERE

    assert meta_data['display_subject'] is None
    assert meta_data['display_surf'] is None
    assert meta_data['measure'] == 'area'
    assert meta_data['custom_morphometry_files_used'] == False

    assert vert_coords is None
    assert faces is None
    assert morphometry_data.shape == (FSAVERAGE_NUM_VERTS_PER_HEMISPHERE * 2, )


def test_parse_subject_standard_space_data_does_not_load_morphometry_data_when_asked_not_to():
    expected_subjects_dir = TEST_DATA_DIR
    expected_fsaverage_surf_dir = os.path.join(TEST_DATA_DIR, 'fsaverage', 'surf')
    if not os.path.isdir(expected_fsaverage_surf_dir):
        pytest.skip("Test data missing: e.g., directory '%s' does not exist. You can get all test data by running './develop/get_test_data_all.bash' in the repo root." % expected_fsaverage_surf_dir)

    vert_coords, faces, morphometry_data, meta_data = bl.subject_avg('subject1', subjects_dir=TEST_DATA_DIR, load_morphometry_data=False)
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
    assert morphometry_data is None


def test_parse_subject_standard_space_data_accepts_custom_morphometry_files():
    expected_subjects_dir = TEST_DATA_DIR
    expected_fsaverage_surf_dir = os.path.join(TEST_DATA_DIR, 'fsaverage', 'surf')
    if not os.path.isdir(expected_fsaverage_surf_dir):
        pytest.skip("Test data missing: e.g., directory '%s' does not exist. You can get all test data by running './develop/get_test_data_all.bash' in the repo root." % expected_fsaverage_surf_dir)

    custom_morphometry_files = { 'lh': 'lh.area.fsaverage.mgh', 'rh': 'rh.area.fsaverage.mgh' }  # You could access these files without the custom_morphometry_files argument (by setting fwhm to None explicitely), but using this custom name is convenient because we already have test data named like this.
    vert_coords, faces, morphometry_data, meta_data = bl.subject_avg('subject1', subjects_dir=TEST_DATA_DIR, custom_morphometry_files=custom_morphometry_files)
    assert len(meta_data) == 24
    expected_lh_morphometry_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.area.fsaverage.mgh')
    expected_rh_morphometry_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'rh.area.fsaverage.mgh')
    assert meta_data['lh.morphometry_file'] == expected_lh_morphometry_file
    assert meta_data['lh.morphometry_file_format'] == 'mgh'
    assert meta_data['rh.morphometry_file'] == expected_rh_morphometry_file
    assert meta_data['rh.morphometry_file_format'] == 'mgh'
    assert meta_data['custom_morphometry_files_used'] == True


def test_load_group_data():
    expected_subject2_dir = os.path.join(TEST_DATA_DIR, 'subject2')
    if not os.path.isdir(expected_subject2_dir):
        pytest.skip("Test data missing: e.g., directory '%s' does not exist. You can get all test data by running './develop/get_test_data_all.bash' in the repo root." % expected_subject2_dir)

    group_data, group_data_subjects, group_meta_data, run_meta_data = bl.group('area', subjects_dir=TEST_DATA_DIR)

    expected_lh_morphometry_file_subject1 = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.area.fwhm10.fsaverage.mgh')
    expected_rh_morphometry_file_subject1 = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'rh.area.fwhm10.fsaverage.mgh')
    expected_lh_morphometry_file_subject5 = os.path.join(TEST_DATA_DIR, 'subject5', 'surf', 'lh.area.fwhm10.fsaverage.mgh')
    expected_rh_morphometry_file_subject5 = os.path.join(TEST_DATA_DIR, 'subject5', 'surf', 'rh.area.fwhm10.fsaverage.mgh')

    expected_subjects_file = os.path.join(TEST_DATA_DIR, 'subjects.txt')

    assert group_data.shape == (5, FSAVERAGE_NUM_VERTS_PER_HEMISPHERE * 2)   # We have 5 subjects in the subjects.txt file in the test data dir

    assert len(run_meta_data) == 5
    assert run_meta_data['custom_morphometry_file_templates_used'] == False
    assert run_meta_data['subjects_file_used'] == True
    assert run_meta_data['subjects_file'] == expected_subjects_file
    assert run_meta_data['subjects_detection_mode'] == 'auto'
    assert run_meta_data['subjects_detection_mode_auto_used_method'] == 'file'

    assert len(group_meta_data) == 5
    assert len(group_meta_data) == len(group_data_subjects)

    assert len(group_meta_data['subject1']) == 18
    assert group_meta_data['subject1']['lh.morphometry_file'] == expected_lh_morphometry_file_subject1
    assert group_meta_data['subject1']['rh.morphometry_file'] == expected_rh_morphometry_file_subject1

    assert group_meta_data['subject1']['display_subject'] is None
    assert group_meta_data['subject1']['display_surf'] is None
    assert group_meta_data['subject1']['measure'] == 'area'

    assert len(group_meta_data['subject5']) == 18
    assert group_meta_data['subject5']['lh.morphometry_file'] == expected_lh_morphometry_file_subject5
    assert group_meta_data['subject5']['rh.morphometry_file'] == expected_rh_morphometry_file_subject5


def test_load_group_data_works_with_subjects_file_in_custom_dir():
    expected_subject2_dir = os.path.join(TEST_DATA_DIR, 'subject2')
    if not os.path.isdir(expected_subject2_dir):
        pytest.skip("Test data missing: e.g., directory '%s' does not exist. You can get all test data by running './develop/get_test_data_all.bash' in the repo root." % expected_subject2_dir)

    custom_subjects_file_dir = os.path.join(TEST_DATA_DIR, 'subject_files_in_extra_dir')
    group_data, group_data_subjects, group_meta_data, run_meta_data = bl.group('area', subjects_dir=TEST_DATA_DIR, subjects_file='subjects_including_s6_in_subdir.csv', subjects_file_dir=custom_subjects_file_dir)

    expected_lh_morphometry_file_subject1 = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.area.fwhm10.fsaverage.mgh')
    expected_rh_morphometry_file_subject1 = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'rh.area.fwhm10.fsaverage.mgh')
    expected_lh_morphometry_file_subject5 = os.path.join(TEST_DATA_DIR, 'subject5', 'surf', 'lh.area.fwhm10.fsaverage.mgh')
    expected_rh_morphometry_file_subject5 = os.path.join(TEST_DATA_DIR, 'subject5', 'surf', 'rh.area.fwhm10.fsaverage.mgh')

    expected_subjects_file = os.path.join(TEST_DATA_DIR, 'subjects.txt')

    assert group_data.shape == (6, FSAVERAGE_NUM_VERTS_PER_HEMISPHERE * 2)   # We have 5 subjects in the subjects.txt file in the test data dir
    assert len(group_meta_data) == 6
    assert len(group_meta_data) == len(group_data_subjects)


def test_load_group_data_works_with_left_hemisphere_only():
    expected_subject2_dir = os.path.join(TEST_DATA_DIR, 'subject2')
    if not os.path.isdir(expected_subject2_dir):
        pytest.skip("Test data missing: e.g., directory '%s' does not exist. You can get all test data by running './develop/get_test_data_all.bash' in the repo root." % expected_subject2_dir)

    group_data, group_data_subjects, group_meta_data, run_meta_data = bl.group('area', hemi='lh', subjects_dir=TEST_DATA_DIR)

    expected_lh_morphometry_file_subject1 = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.area.fwhm10.fsaverage.mgh')
    expected_lh_morphometry_file_subject5 = os.path.join(TEST_DATA_DIR, 'subject5', 'surf', 'lh.area.fwhm10.fsaverage.mgh')

    assert len(run_meta_data) == 5
    assert run_meta_data['subjects_detection_mode'] == 'auto'
    assert run_meta_data['subjects_detection_mode_auto_used_method'] == 'file'
    assert group_data.shape == (5, FSAVERAGE_NUM_VERTS_PER_HEMISPHERE)   # We have 5 subjects in the subjects.txt file in the test data dir
    assert len(group_meta_data) == 5
    assert len(group_meta_data) == len(group_data_subjects)
    assert len(group_meta_data['subject1']) == 15
    assert group_meta_data['subject1']['lh.morphometry_file'] == expected_lh_morphometry_file_subject1
    assert group_meta_data['subject5']['lh.morphometry_file'] == expected_lh_morphometry_file_subject5
    assert not 'rh.morphometry_file' in group_meta_data['subject1']
    assert not 'rh.morphometry_file' in group_meta_data['subject5']


def test_load_group_data_works_with_right_hemisphere_only():
    expected_subject2_dir = os.path.join(TEST_DATA_DIR, 'subject2')
    if not os.path.isdir(expected_subject2_dir):
        pytest.skip("Test data missing: e.g., directory '%s' does not exist. You can get all test data by running './develop/get_test_data_all.bash' in the repo root." % expected_subject2_dir)

    group_data, group_data_subjects, group_meta_data, run_meta_data = bl.group('area', hemi='rh', subjects_dir=TEST_DATA_DIR)

    expected_rh_morphometry_file_subject1 = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'rh.area.fwhm10.fsaverage.mgh')
    expected_rh_morphometry_file_subject5 = os.path.join(TEST_DATA_DIR, 'subject5', 'surf', 'rh.area.fwhm10.fsaverage.mgh')

    assert len(run_meta_data) == 5
    assert run_meta_data['subjects_detection_mode'] == 'auto'
    assert run_meta_data['subjects_detection_mode_auto_used_method'] == 'file'
    assert group_data.shape == (5, FSAVERAGE_NUM_VERTS_PER_HEMISPHERE)   # We have 5 subjects in the subjects.txt file in the test data dir
    assert len(group_meta_data) == 5
    assert len(group_meta_data) == len(group_data_subjects)
    assert len(group_meta_data['subject1']) == 15
    assert group_meta_data['subject1']['rh.morphometry_file'] == expected_rh_morphometry_file_subject1
    assert group_meta_data['subject5']['rh.morphometry_file'] == expected_rh_morphometry_file_subject5
    assert not 'lh.morphometry_file' in group_meta_data['subject1']
    assert not 'lh.morphometry_file' in group_meta_data['subject5']


def test_load_group_data_works_with_custom_morphometry_file_templates_using_variables_surf_white():
    expected_subject2_dir = os.path.join(TEST_DATA_DIR, 'subject2')
    if not os.path.isdir(expected_subject2_dir):
        pytest.skip("Test data missing: e.g., directory '%s' does not exist. You can get all test data by running './develop/get_test_data_all.bash' in the repo root." % expected_subject2_dir)

    morphometry_template = '${HEMI}.${SURF}${MEASURE}.${AVERAGE_SUBJECT}.mgh'
    custom_morphometry_file_templates = {'lh': morphometry_template, 'rh': morphometry_template}
    group_data, group_data_subjects, group_meta_data, run_meta_data = bl.group('area', hemi='both', surf='white', subjects_dir=TEST_DATA_DIR, custom_morphometry_file_templates=custom_morphometry_file_templates)

    expected_lh_morphometry_file_subject1 = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.area.fsaverage.mgh')     # for surface 'white', the surface must NOT show up in the result.
    expected_rh_morphometry_file_subject1 = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'rh.area.fsaverage.mgh')
    expected_lh_morphometry_file_subject5 = os.path.join(TEST_DATA_DIR, 'subject5', 'surf', 'lh.area.fsaverage.mgh')
    expected_rh_morphometry_file_subject5 = os.path.join(TEST_DATA_DIR, 'subject5', 'surf', 'rh.area.fsaverage.mgh')
    expected_subjects_file = os.path.join(TEST_DATA_DIR, 'subjects.txt')

    assert len(run_meta_data) == 7
    assert run_meta_data['custom_morphometry_file_templates_used'] == True
    assert run_meta_data['subjects_file_used'] == True
    assert run_meta_data['subjects_file'] == expected_subjects_file
    assert run_meta_data['lh.custom_morphometry_file_template'] == morphometry_template
    assert run_meta_data['rh.custom_morphometry_file_template'] == morphometry_template
    assert run_meta_data['subjects_detection_mode'] == 'auto'
    assert run_meta_data['subjects_detection_mode_auto_used_method'] == 'file'

    assert group_data.shape == (5, FSAVERAGE_NUM_VERTS_PER_HEMISPHERE * 2)   # We have 5 subjects in the subjects.txt file in the test data dir
    assert len(group_meta_data) == 5
    assert len(group_meta_data) == len(group_data_subjects)
    assert len(group_meta_data['subject1']) == 18
    assert group_meta_data['subject1']['lh.morphometry_file'] == expected_lh_morphometry_file_subject1
    assert group_meta_data['subject1']['rh.morphometry_file'] == expected_rh_morphometry_file_subject1
    assert group_meta_data['subject5']['lh.morphometry_file'] == expected_lh_morphometry_file_subject5
    assert group_meta_data['subject5']['rh.morphometry_file'] == expected_rh_morphometry_file_subject5


def test_load_group_data_works_with_custom_morphometry_file_templates_using_hardcoded_filenames():
    expected_subject2_dir = os.path.join(TEST_DATA_DIR, 'subject2')
    if not os.path.isdir(expected_subject2_dir):
        pytest.skip("Test data missing: e.g., directory '%s' does not exist. You can get all test data by running './develop/get_test_data_all.bash' in the repo root." % expected_subject2_dir)

    template_lh = 'lh.area.fsaverage.mgh'   # nobody forces you to use any variables
    template_rh = 'rh.area.fsaverage.mgh'
    custom_morphometry_file_templates = {'lh': template_lh, 'rh': template_rh}
    group_data, group_data_subjects, group_meta_data, run_meta_data = bl.group('area', hemi='both', subjects_dir=TEST_DATA_DIR, custom_morphometry_file_templates=custom_morphometry_file_templates)

    expected_lh_morphometry_file_subject1 = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.area.fsaverage.mgh')
    expected_rh_morphometry_file_subject1 = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'rh.area.fsaverage.mgh')
    expected_lh_morphometry_file_subject5 = os.path.join(TEST_DATA_DIR, 'subject5', 'surf', 'lh.area.fsaverage.mgh')
    expected_rh_morphometry_file_subject5 = os.path.join(TEST_DATA_DIR, 'subject5', 'surf', 'rh.area.fsaverage.mgh')
    expected_subjects_file = os.path.join(TEST_DATA_DIR, 'subjects.txt')

    assert len(run_meta_data) == 7
    assert run_meta_data['custom_morphometry_file_templates_used'] == True
    assert run_meta_data['subjects_file_used'] == True
    assert run_meta_data['subjects_file'] == expected_subjects_file
    assert run_meta_data['lh.custom_morphometry_file_template'] == template_lh
    assert run_meta_data['rh.custom_morphometry_file_template'] == template_rh
    assert run_meta_data['subjects_detection_mode'] == 'auto'
    assert run_meta_data['subjects_detection_mode_auto_used_method'] == 'file'

    assert group_data.shape == (5, FSAVERAGE_NUM_VERTS_PER_HEMISPHERE * 2)   # We have 5 subjects in the subjects.txt file in the test data dir
    assert len(group_meta_data) == 5
    assert len(group_meta_data) == len(group_data_subjects)
    assert len(group_meta_data['subject1']) == 18
    assert group_meta_data['subject1']['lh.morphometry_file'] == expected_lh_morphometry_file_subject1
    assert group_meta_data['subject1']['rh.morphometry_file'] == expected_rh_morphometry_file_subject1
    assert group_meta_data['subject5']['lh.morphometry_file'] == expected_lh_morphometry_file_subject5
    assert group_meta_data['subject5']['rh.morphometry_file'] == expected_rh_morphometry_file_subject5


def test_load_group_data_works_with_subjects_list():
    expected_subject2_dir = os.path.join(TEST_DATA_DIR, 'subject2')
    if not os.path.isdir(expected_subject2_dir):
        pytest.skip("Test data missing: e.g., directory '%s' does not exist. You can get all test data by running './develop/get_test_data_all.bash' in the repo root." % expected_subject2_dir)

    subjects_list = [ 'subject1', 'subject3' ]
    group_data, group_data_subjects, group_meta_data, run_meta_data = bl.group('area', subjects_dir=TEST_DATA_DIR, subjects_list=subjects_list)

    expected_lh_morphometry_file_subject1 = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.area.fwhm10.fsaverage.mgh')
    expected_rh_morphometry_file_subject1 = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'rh.area.fwhm10.fsaverage.mgh')
    expected_lh_morphometry_file_subject3 = os.path.join(TEST_DATA_DIR, 'subject3', 'surf', 'lh.area.fwhm10.fsaverage.mgh')
    expected_rh_morphometry_file_subject3 = os.path.join(TEST_DATA_DIR, 'subject3', 'surf', 'rh.area.fwhm10.fsaverage.mgh')

    assert len(run_meta_data) == 4
    assert run_meta_data['custom_morphometry_file_templates_used'] == False
    assert run_meta_data['subjects_file_used'] == False
    assert run_meta_data['subjects_detection_mode'] == 'auto'
    assert run_meta_data['subjects_detection_mode_auto_used_method'] == 'list'

    assert group_data.shape == (2, FSAVERAGE_NUM_VERTS_PER_HEMISPHERE * 2)
    assert len(group_meta_data) == 2
    assert len(group_meta_data) == len(group_data_subjects)
    assert not 'subject2' in group_meta_data
    assert len(group_meta_data['subject1']) == 18
    assert group_meta_data['subject1']['lh.morphometry_file'] == expected_lh_morphometry_file_subject1
    assert group_meta_data['subject1']['rh.morphometry_file'] == expected_rh_morphometry_file_subject1

    assert group_meta_data['subject1']['display_subject'] is None
    assert group_meta_data['subject1']['display_surf'] is None
    assert group_meta_data['subject1']['measure'] == 'area'

    assert len(group_meta_data['subject3']) == 18
    assert group_meta_data['subject3']['lh.morphometry_file'] == expected_lh_morphometry_file_subject3
    assert group_meta_data['subject3']['rh.morphometry_file'] == expected_rh_morphometry_file_subject3


def test_load_group_data_subject_order_in_data_is_correct_from_subjects_file():
    expected_subject2_dir = os.path.join(TEST_DATA_DIR, 'subject2')
    if not os.path.isdir(expected_subject2_dir):
        pytest.skip("Test data missing: e.g., directory '%s' does not exist. You can get all test data by running './develop/get_test_data_all.bash' in the repo root." % expected_subject2_dir)

    group_data, group_data_subjects, group_meta_data, run_meta_data = bl.group('area', subjects_dir=TEST_DATA_DIR, subjects_file='subjects_including_s6.csv')

    assert len(group_meta_data) == 6
    assert len(group_meta_data) == len(group_data_subjects)

    assert group_data_subjects[0] == 'subject1'         # This is the order in which the subjects appear in our test data subjects file. So we can use group_data_subjects to access the correct data.
    assert group_data_subjects[1] == 'subject2'
    assert group_data_subjects[2] == 'subject3'
    assert group_data_subjects[3] == 'subject4'
    assert group_data_subjects[4] == 'subject5'
    assert group_data_subjects[5] == 'subject6'

    assert group_data[0][100000] == pytest.approx(0.74, 0.1)    # We know this value from loading the MGH file manually.
    assert group_data[1][100000] == pytest.approx(0.74, 0.1)    # the value is identical for this subject2 because it is a copy of subject1 (see how test data is created)
    assert group_data[2][100000] == pytest.approx(0.74, 0.1)    # the value is identical for this subject3 because it is a copy of subject1 (see how test data is created)
    assert group_data[3][100000] == pytest.approx(0.74, 0.1)    # the value is identical for this subject4 because it is a copy of subject1 (see how test data is created)
    assert group_data[4][100000] == pytest.approx(0.74, 0.1)    # the value is identical for this subject5 because it is a copy of subject1 (see how test data is created)
    assert group_data[5][100000] == pytest.approx(0.20, 0.1)    # the value is NOT identical for subject6 because we manually edited it in the test data file in the repo. See also the function test_test_data_lh_is_as_expected() in this file. Note that this shows that subject 6 shows up in the correct position.


def test_load_group_data_subject_order_in_data_is_correct_from_subjects_list():
    expected_subject2_dir = os.path.join(TEST_DATA_DIR, 'subject2')
    if not os.path.isdir(expected_subject2_dir):
        pytest.skip("Test data for subject2 .. subject5 not available: e.g., directory '%s' does not exist. You can get it by running './develop/get_group_data.bash' in the repo root." % expected_subject2_dir)

    subjects_list = [ 'subject1', 'subject6', 'subject3' ]
    group_data, group_data_subjects, group_meta_data, run_meta_data = bl.group('area', fwhm='10', subjects_dir=TEST_DATA_DIR, subjects_list=subjects_list)

    assert len(group_meta_data) == 3
    assert len(group_meta_data) == len(group_data_subjects)
    assert len(group_data_subjects) == len(subjects_list)

    assert group_data_subjects[0] == 'subject1'
    assert group_data_subjects[1] == 'subject6'
    assert group_data_subjects[2] == 'subject3'

    # This is most likely close to real-world usage of the 'group_data_subjects' list:
    subject6_idx = group_data_subjects.index('subject6')
    assert group_data[subject6_idx][100000] == pytest.approx(0.20, 0.1)         # The modified value. So subject6 is in the expected position.

    # This is a unit test, so test some more stuff
    assert subject6_idx == 1
    assert group_data[0][100000] == pytest.approx(0.74, 0.1)
    assert group_data[2][100000] == pytest.approx(0.74, 0.1)


def test_test_data_lh_is_as_expected():
    # The file lh.area.fwhm11.fsaverage.mgh is an edited version of lh.area.fwhm10.fsaverage.mgh. The only change is that the data value at index 100,000 (with indexing starting at 0), 0.74, is replaced with the value 0.2.
    # The MGH file edits were done with the FreeSurfer matlab functions MRIread and MRIwrite.
    morphometry_file_value_orig = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.area.fwhm10.fsaverage.mgh')
    morphometry_file_value_mod = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'lh.area.fwhm11.fsaverage.mgh')

    mgh_data_orig, mgh_meta_data_orig = fsd.read_mgh_file(morphometry_file_value_orig)
    assert mgh_data_orig.shape == (FSAVERAGE_NUM_VERTS_PER_HEMISPHERE, 1, 1)
    relevant_data_inner_array_orig = mgh_data_orig[:,0]
    assert relevant_data_inner_array_orig.shape == (FSAVERAGE_NUM_VERTS_PER_HEMISPHERE, 1)
    per_vertex_data_orig = relevant_data_inner_array_orig[:,0]
    assert per_vertex_data_orig.shape == (FSAVERAGE_NUM_VERTS_PER_HEMISPHERE, )
    assert per_vertex_data_orig[100000] == pytest.approx(0.74, 0.1)                     # lh original value at index 100,000

    mgh_data_mod, mgh_meta_data_mod = fsd.read_mgh_file(morphometry_file_value_mod)
    assert mgh_data_mod.shape == (FSAVERAGE_NUM_VERTS_PER_HEMISPHERE, 1, 1)
    relevant_data_inner_array_mod = mgh_data_mod[:,0]
    assert relevant_data_inner_array_mod.shape == (FSAVERAGE_NUM_VERTS_PER_HEMISPHERE, 1)
    per_vertex_data_mod = relevant_data_inner_array_mod[:,0]
    assert per_vertex_data_mod.shape == (FSAVERAGE_NUM_VERTS_PER_HEMISPHERE, )
    assert per_vertex_data_mod[100000] == pytest.approx(0.20, 0.1)                      # edited value

    # test random other data for equality
    assert per_vertex_data_orig[50] == pytest.approx(per_vertex_data_mod[50], 0.1)
    assert per_vertex_data_orig[5000] == pytest.approx(per_vertex_data_mod[5000], 0.1)
    assert per_vertex_data_orig[9000] == pytest.approx(per_vertex_data_mod[9000], 0.1)
    assert per_vertex_data_orig[123000] == pytest.approx(per_vertex_data_mod[123000], 0.1)


def test_test_data_rh_is_as_expected():
    # The file rh.area.fwhm11.fsaverage.mgh is an edited version of rh.area.fwhm10.fsaverage.mgh. The only change is that the data value at index 100,000 (with indexing starting at 0), 0.60, is replaced with the value 0.2.
    # The MGH file edits were done with the FreeSurfer matlab functions MRIread and MRIwrite.
    morphometry_file_value_orig = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'rh.area.fwhm10.fsaverage.mgh')
    morphometry_file_value_mod = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'rh.area.fwhm11.fsaverage.mgh')

    mgh_data_orig, mgh_meta_data_orig = fsd.read_mgh_file(morphometry_file_value_orig)
    assert mgh_data_orig.shape == (FSAVERAGE_NUM_VERTS_PER_HEMISPHERE, 1, 1)
    relevant_data_inner_array_orig = mgh_data_orig[:,0]
    assert relevant_data_inner_array_orig.shape == (FSAVERAGE_NUM_VERTS_PER_HEMISPHERE, 1)
    per_vertex_data_orig = relevant_data_inner_array_orig[:,0]
    assert per_vertex_data_orig.shape == (FSAVERAGE_NUM_VERTS_PER_HEMISPHERE, )
    assert per_vertex_data_orig[100000] == pytest.approx(0.60, 0.1)                     # rh original value at index 100,000

    mgh_data_mod, mgh_meta_data_mod = fsd.read_mgh_file(morphometry_file_value_mod)
    assert mgh_data_mod.shape == (FSAVERAGE_NUM_VERTS_PER_HEMISPHERE, 1, 1)
    relevant_data_inner_array_mod = mgh_data_mod[:,0]
    assert relevant_data_inner_array_mod.shape == (FSAVERAGE_NUM_VERTS_PER_HEMISPHERE, 1)
    per_vertex_data_mod = relevant_data_inner_array_mod[:,0]
    assert per_vertex_data_mod.shape == (FSAVERAGE_NUM_VERTS_PER_HEMISPHERE, )
    assert per_vertex_data_mod[100000] == pytest.approx(0.20, 0.1)                      # edited value

    # test random other data for equality
    assert per_vertex_data_orig[50] == pytest.approx(per_vertex_data_mod[50], 0.1)
    assert per_vertex_data_orig[5000] == pytest.approx(per_vertex_data_mod[5000], 0.1)
    assert per_vertex_data_orig[9000] == pytest.approx(per_vertex_data_mod[9000], 0.1)
    assert per_vertex_data_orig[123000] == pytest.approx(per_vertex_data_mod[123000], 0.1)


def test_load_group_data_raises_on_invalid_hemisphere():
    with pytest.raises(ValueError) as exc_info:
        group_data, group_data_subjects, group_meta_data, run_meta_data = bl.group('area', hemi='invalid_hemisphere', subjects_dir=TEST_DATA_DIR)
    assert 'hemi must be one of' in str(exc_info.value)
    assert 'invalid_hemisphere' in str(exc_info.value)


def test_load_group_data_raises_on_invalid_subjects_detection_mode():
    with pytest.raises(ValueError) as exc_info:
        group_data, group_data_subjects, group_meta_data, run_meta_data = bl.group('area', subjects_detection_mode='invalid_subjects_detection_mode', subjects_dir=TEST_DATA_DIR)
    assert 'subjects_detection_mode must be one of' in str(exc_info.value)
    assert 'invalid_subjects_detection_mode' in str(exc_info.value)


def test_load_group_data_raises_with_subjects_list_in_mode_file():
    with pytest.raises(ValueError) as exc_info:
        group_data, group_data_subjects, group_meta_data, run_meta_data = bl.group('area', subjects_detection_mode='file', subjects_list=['bert', 'tim'], subjects_dir=TEST_DATA_DIR)
    assert 'subjects_detection_mode is set to \'file\'' in str(exc_info.value)
    assert 'but a subjects_list was given' in str(exc_info.value)


def test_load_group_data_raises_without_subjects_list_in_mode_list():
    with pytest.raises(ValueError) as exc_info:
        group_data, group_data_subjects, group_meta_data, run_meta_data = bl.group('area', subjects_detection_mode='list', subjects_dir=TEST_DATA_DIR)
    assert 'subjects_detection_mode is set to \'list\'' in str(exc_info.value)
    assert 'but the subjects_list parameter was not given' in str(exc_info.value)


def test_load_group_data_raises_with_subjects_list_in_mode_search_dir():
    with pytest.raises(ValueError) as exc_info:
        group_data, group_data_subjects, group_meta_data, run_meta_data = bl.group('area', subjects_detection_mode='search_dir', subjects_list=['bert', 'tim'], subjects_dir=TEST_DATA_DIR)
    assert 'subjects_detection_mode is set to \'search_dir\'' in str(exc_info.value)
    assert 'but a subjects_list was given' in str(exc_info.value)


def test_load_group_data_raises_with_nonexistant_subjects_file_in_mode_file():
    with pytest.raises(ValueError) as exc_info:
        group_data, group_data_subjects, group_meta_data, run_meta_data = bl.group('area', subjects_detection_mode='file', subjects_file='no_such_file', subjects_dir=TEST_DATA_DIR)
    assert 'no_such_file' in str(exc_info.value)
    assert 'subjects_detection_mode is set to \'file\' but the subjects_file' in str(exc_info.value)


def test_load_group_data_auto_mode_prefers_list_over_explicitely_given_subjects_file():
    expected_subject2_dir = os.path.join(TEST_DATA_DIR, 'subject2')
    if not os.path.isdir(expected_subject2_dir):
        pytest.skip("Test data missing: e.g., directory '%s' does not exist. You can get all test data by running './develop/get_test_data_all.bash' in the repo root." % expected_subject2_dir)

    subjects_list = [ 'subject1', 'subject6', 'subject3' ]
    group_data, group_data_subjects, group_meta_data, run_meta_data = bl.group('area', fwhm='10', subjects_dir=TEST_DATA_DIR, subjects_list=subjects_list, subjects_file='no_such_file')

    assert len(group_meta_data) == 3
    assert len(group_meta_data) == len(group_data_subjects)
    assert len(group_data_subjects) == len(subjects_list)

    assert run_meta_data['subjects_detection_mode'] == 'auto'
    assert run_meta_data['subjects_detection_mode_auto_used_method'] == 'list'


def test_load_group_data_auto_mode_prefers_list_over_default_subjects_file_and_search_dir():
    expected_subject2_dir = os.path.join(TEST_DATA_DIR, 'subject2')
    if not os.path.isdir(expected_subject2_dir):
        pytest.skip("Test data missing: e.g., directory '%s' does not exist. You can get all test data by running './develop/get_test_data_all.bash' in the repo root." % expected_subject2_dir)

    subjects_list = [ 'subject1', 'subject6', 'subject3' ]
    group_data, group_data_subjects, group_meta_data, run_meta_data = bl.group('area', fwhm='10', subjects_dir=TEST_DATA_DIR, subjects_list=subjects_list)

    assert len(group_meta_data) == 3
    assert len(group_meta_data) == len(group_data_subjects)
    assert len(group_data_subjects) == len(subjects_list)

    assert run_meta_data['subjects_detection_mode'] == 'auto'
    assert run_meta_data['subjects_detection_mode_auto_used_method'] == 'list'


def test_load_group_data_auto_mode_searches_dir_as_last_resort():
    extra_subjects_dir = os.path.join(TEST_DATA_DIR, 'extra_subjects')
    if not os.path.isdir(extra_subjects_dir):
        pytest.skip("Test data missing: e.g., directory '%s' does not exist. You can get all test data by running './develop/get_test_data_all.bash' in the repo root." % extra_subjects_dir)

    group_data, group_data_subjects, group_meta_data, run_meta_data = bl.group('area', fwhm='10', subjects_dir=extra_subjects_dir)

    assert len(group_meta_data) == 2
    assert len(group_meta_data) == len(group_data_subjects)
    assert 'subject2x' in group_data_subjects       # relying on order would be dangerous when the files are read from a file system
    assert 'subject3x' in group_data_subjects

    assert run_meta_data['subjects_detection_mode'] == 'auto'
    assert run_meta_data['subjects_detection_mode_auto_used_method'] == 'search_dir'


def test_load_group_data_search_dir_mode_works():
    extra_subjects_dir = os.path.join(TEST_DATA_DIR, 'extra_subjects')
    if not os.path.isdir(extra_subjects_dir):
        pytest.skip("Test data missing: e.g., directory '%s' does not exist. You can get all test data by running './develop/get_test_data_all.bash' in the repo root." % extra_subjects_dir)

    group_data, group_data_subjects, group_meta_data, run_meta_data = bl.group('area', fwhm='10', subjects_dir=extra_subjects_dir, subjects_detection_mode='search_dir')

    assert len(group_meta_data) == 2
    assert len(group_meta_data) == len(group_data_subjects)
    assert 'subject2x' in group_data_subjects       # relying on order would be dangerous when the files are read from a file system
    assert 'subject3x' in group_data_subjects

    assert run_meta_data['subjects_detection_mode'] == 'search_dir'
    assert run_meta_data['subjects_file_used'] == False
    assert not 'subjects_detection_mode_auto_used_method' in run_meta_data


def test_load_group_data_file_mode_works():
    expected_subject2_dir = os.path.join(TEST_DATA_DIR, 'subject2')
    if not os.path.isdir(expected_subject2_dir):
        pytest.skip("Test data missing: e.g., directory '%s' does not exist. You can get all test data by running './develop/get_test_data_all.bash' in the repo root." % expected_subject2_dir)

    group_data, group_data_subjects, group_meta_data, run_meta_data = bl.group('area', subjects_dir=TEST_DATA_DIR, subjects_detection_mode='file')

    assert len(group_meta_data) == 5
    assert len(group_meta_data) == len(group_data_subjects)

    assert run_meta_data['subjects_detection_mode'] == 'file'
    assert run_meta_data['subjects_file_used'] == True
    assert not 'subjects_detection_mode_auto_used_method' in run_meta_data


def test_load_group_data_list_mode_works():
    expected_subject2_dir = os.path.join(TEST_DATA_DIR, 'subject2')
    if not os.path.isdir(expected_subject2_dir):
        pytest.skip("Test data missing: e.g., directory '%s' does not exist. You can get all test data by running './develop/get_test_data_all.bash' in the repo root." % expected_subject2_dir)

    subjects_list = [ 'subject1', 'subject6', 'subject3' ]
    group_data, group_data_subjects, group_meta_data, run_meta_data = bl.group('area', subjects_dir=TEST_DATA_DIR, subjects_list=subjects_list, subjects_detection_mode='list')

    assert len(group_meta_data) == 3
    assert len(group_meta_data) == len(group_data_subjects)
    assert len(group_meta_data) == len(subjects_list)

    assert run_meta_data['subjects_detection_mode'] == 'list'
    assert run_meta_data['subjects_file_used'] == False
    assert not 'subjects_detection_mode_auto_used_method' in run_meta_data


def test_rhi_all_fine():
    morphometry_data_lh, meta_data_lh = bl.subject('subject1', hemi='lh', subjects_dir=TEST_DATA_DIR, load_surface_files=False)[2:4]
    morphometry_data_rh, meta_data_rh = bl.subject('subject1', hemi='rh', subjects_dir=TEST_DATA_DIR, load_surface_files=False)[2:4]
    morphometry_data_both, meta_data_both = bl.subject('subject1', hemi='both', subjects_dir=TEST_DATA_DIR, load_surface_files=False)[2:4]
    assert meta_data_both['lh.num_data_points'] == len(morphometry_data_lh)
    assert meta_data_both['rh.num_data_points'] == len(morphometry_data_rh)
    assert meta_data_lh['lh.num_data_points'] == meta_data_both['lh.num_data_points']
    assert meta_data_rh['rh.num_data_points'] == meta_data_both['rh.num_data_points']
    abs_rh_start = bl.rhi(0, meta_data_both)
    assert morphometry_data_both[abs_rh_start] == pytest.approx(morphometry_data_rh[0], 0.1)
    assert bl.rhv(0, morphometry_data_both, meta_data_both) == pytest.approx(morphometry_data_rh[0], 0.1)
    abs_rh_second_to_last = bl.rhi(-1, meta_data_both)
    assert abs_rh_second_to_last == len(morphometry_data_lh) + len(morphometry_data_rh) -2
    assert morphometry_data_both[abs_rh_second_to_last] == pytest.approx(morphometry_data_rh[len(morphometry_data_rh)-2], 0.1)


def test_rhi_raises_on_invalid_metadata():
    meta_data_both = 5
    with pytest.raises(ValueError) as exc_info:
        abs_rh_start = bl.rhi(0, meta_data_both)
    assert 'must be a meta data dictionary containing the keys' in str(exc_info.value)


def test_rhi_raises_on_missing_metadata_keys():
    meta_data_both = {'a': 'test', 'b': 'test2'}
    with pytest.raises(ValueError) as exc_info:
        abs_rh_start = bl.rhi(0, meta_data_both)
    assert 'must be a meta data dictionary containing the keys' in str(exc_info.value)


def test_rhi_raises_on_index_out_of_bounds():
    morphometry_data_both, meta_data_both = bl.subject('subject1', hemi='both', subjects_dir=TEST_DATA_DIR, load_surface_files=False)[2:4]
    with pytest.raises(ValueError) as exc_info:
        abs_rh_start = bl.rhi(500000, meta_data_both)
    assert 'out of bounds: right hemisphere has' in str(exc_info.value)
    assert '500000' in str(exc_info.value)

def test_fsaverage_mesh():
    verts, faces, meta_data = bl.fsaverage_mesh(subjects_dir=TEST_DATA_DIR, use_freesurfer_home_if_missing=True)
    assert verts.shape == (FSAVERAGE_NUM_VERTS_PER_HEMISPHERE * 2, 3)
