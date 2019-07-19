import os
import pytest
import numpy as np
from numpy.testing import assert_raises, assert_array_equal, assert_allclose
import brainload.nitools as nit
import brainload.annotations as an
import brainload as bl

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

SUBJECT1_NUM_VERTICES_IN_LABEL_CORTEX_LH = 140891
SUBJECT1_NUM_VERTICES_IN_LABEL_CORTEX_RH =144884

NUM_LABELS_APARC = 36
NUM_LABELS_APARC_A2009S = 76
NUM_LABELS_APARC_DKTATLAS = 36


def test_read_annotation_md_lh():
    annotation_file = os.path.join(TEST_DATA_DIR, 'subject1', 'label', 'lh.aparc.annot')
    labels, ctab, names, meta_data = an.read_annotation_md(annotation_file, 'lh', meta_data=None)
    assert len(meta_data) == 1
    assert meta_data['lh.annotation_file'] == annotation_file
    assert labels.shape == (SUBJECT1_SURF_LH_WHITE_NUM_VERTICES, )
    assert ctab.shape == (NUM_LABELS_APARC, 5)
    assert len(names) == NUM_LABELS_APARC
    assert names[0] == "unknown"    # The first label is known to be 'unknown'. This also tests whether the object really is a string, i.e., whether the bytes have been coverted to string properly for Python 3. This is the real goal.


def test_read_annotation_md_rh():
    annotation_file = os.path.join(TEST_DATA_DIR, 'subject1', 'label', 'rh.aparc.annot')
    labels, ctab, names, meta_data = an.read_annotation_md(annotation_file, 'rh', meta_data=None)
    assert len(meta_data) == 1
    assert meta_data['rh.annotation_file'] == annotation_file
    assert labels.shape == (SUBJECT1_SURF_RH_WHITE_NUM_VERTICES, )
    assert ctab.shape == (NUM_LABELS_APARC, 5)
    assert len(names) == NUM_LABELS_APARC
    assert names[0] == "unknown"    # The first label is known to be 'unknown'. This also tests whether the object really is a string, i.e., whether the bytes have been coverted to string properly for Python 3. This is the real goal.


def test_read_annotation_md_raises_on_invalid_hemisphere_label():
    annotation_file = os.path.join(TEST_DATA_DIR, 'subject1', 'label', 'rh.aparc.annot')
    with pytest.raises(ValueError) as exc_info:
        labels, ctab, names, meta_data = an.read_annotation_md(annotation_file, 'invalid_hemisphere_label', meta_data=None)
    assert 'hemisphere_label must be one of' in str(exc_info.value)
    assert 'invalid_hemisphere_label' in str(exc_info.value)


def test_annot_raises_on_invalid_hemisphere():
    with pytest.raises(ValueError) as exc_info:
        vertex_labels, label_colors, label_names, meta_data = an.annot('subject1', TEST_DATA_DIR, 'aparc', hemi='invalid_hemisphere')
    assert 'hemi must be one of' in str(exc_info.value)
    assert 'invalid_hemisphere' in str(exc_info.value)


def test_annot_metadata_both_hemispheres():
    vertex_labels, label_colors, label_names, meta_data = an.annot('subject1', TEST_DATA_DIR, 'aparc', hemi='both', orig_ids=True)
    assert len(meta_data) == 2
    assert len(label_names) == NUM_LABELS_APARC
    assert vertex_labels.shape == (SUBJECT1_SURF_LH_WHITE_NUM_VERTICES + SUBJECT1_SURF_RH_WHITE_NUM_VERTICES, )
    assert label_colors.shape == (NUM_LABELS_APARC, 5)


def test_annot_metadata_single_hemi_lh():
    vertex_labels, label_colors, label_names, meta_data = an.annot('subject1', TEST_DATA_DIR, 'aparc', hemi='lh')
    assert len(meta_data) == 1
    assert len(label_names) == NUM_LABELS_APARC
    assert vertex_labels.shape == (SUBJECT1_SURF_LH_WHITE_NUM_VERTICES, )
    assert label_colors.shape == (NUM_LABELS_APARC, 5)


def test_annot_metadata_single_hemi_rh_and_keep_metadata():
    vertex_labels, label_colors, label_names, meta_data = an.annot('subject1', TEST_DATA_DIR, 'aparc', hemi='rh', meta_data={'todo': 'keep_this'})
    expected_annot_file = os.path.join(TEST_DATA_DIR, 'subject1', 'label', 'rh.aparc.annot')
    assert len(meta_data) == 2
    assert meta_data['rh.annotation_file'] == expected_annot_file
    assert meta_data['todo'] == 'keep_this'
    assert len(label_names) == NUM_LABELS_APARC
    assert vertex_labels.shape == (SUBJECT1_SURF_RH_WHITE_NUM_VERTICES, )
    assert label_colors.shape == (NUM_LABELS_APARC, 5)


def test_annot_aparc():
    vertex_labels, label_colors, label_names, meta_data = an.annot('subject1', TEST_DATA_DIR, 'aparc', hemi='both')
    assert label_colors.shape == (NUM_LABELS_APARC, 5)


def test_annot_aparc_orig_ids():
    vertex_labels_mod, label_colors_mod, label_names_mod, meta_data_mod = an.annot('subject1', TEST_DATA_DIR, 'aparc', hemi='both')
    vertex_labels_orig, label_colors_orig, label_names_orig, meta_data_orig = an.annot('subject1', TEST_DATA_DIR, 'aparc', hemi='both', orig_ids=True)
    # these should not diff
    assert label_colors_mod.shape == (NUM_LABELS_APARC, 5)
    assert label_colors_mod.shape == label_colors_orig.shape
    assert label_names_mod == label_names_orig
    assert meta_data_mod == meta_data_orig
    assert vertex_labels_mod.shape == vertex_labels_orig.shape
    # now for the parts that should be different between the two
    assert_raises(AssertionError, assert_array_equal, vertex_labels_mod, vertex_labels_orig)


def test_annot_aparc_data_makes_sense():
    vertex_labels, label_colors, label_names, meta_data = an.annot('subject1', TEST_DATA_DIR, 'aparc', hemi='both', orig_ids=False)
    assert len(np.unique(vertex_labels)) == NUM_LABELS_APARC - 1
    index_for_first_vertex = vertex_labels[0]
    assert index_for_first_vertex > 0     # make sure it has a valid ID (-1 means has no label and color)
    assert index_for_first_vertex == 11
    color = (label_colors[index_for_first_vertex, 0], label_colors[index_for_first_vertex, 1], label_colors[index_for_first_vertex, 2], label_colors[index_for_first_vertex, 3])
    assert color == (20, 30, 140, 0)


def test_annot_get_label_index():
    vertex_labels, label_colors, label_names, meta_data = an.annot('subject1', TEST_DATA_DIR, 'aparc', hemi='both', orig_ids=True)
    assert vertex_labels[0] == 9182740
    idx = an._get_annot_label_index(vertex_labels[0], label_colors)
    assert idx == 11
    # this index can now be used to retrieve the color and the label name:
    color_rgbt = (label_colors[idx,0], label_colors[idx, 1], label_colors[idx, 2], label_colors[idx, 3])
    assert color_rgbt == (20, 30, 140, 0)
    label_name = label_names[idx]
    assert label_name == "lateraloccipital"


def test_color_rgbt_to_rgba():
    c1 = (20, 30, 140, 0)
    c2 = (1, 2, 3, 100)
    c3 = (240, 240, 240, 240)
    assert an.color_rgbt_to_rgba(c1) == (20, 30, 140, 255)
    assert an.color_rgbt_to_rgba(c2) == (1, 2, 3, 155)
    assert an.color_rgbt_to_rgba(c3) == (240, 240, 240, 15)


def test_annot_with_different_orig_ids_settings():
    vertex_labels_orig, label_colors_orig, label_names_orig, meta_data_orig = an.annot('subject1', TEST_DATA_DIR, 'aparc', hemi='both', orig_ids=True)
    vertex_labels, label_colors, label_names, meta_data = an.annot('subject1', TEST_DATA_DIR, 'aparc', hemi='both', orig_ids=False)
    assert len(vertex_labels_orig) <= len(vertex_labels)
    assert label_colors_orig.shape == label_colors.shape
    assert len(label_names_orig) == len(label_names)
    assert_array_equal(label_names_orig, label_names)
    assert_array_equal(label_colors_orig, label_colors)
    assert vertex_labels[0] == 11      # the proper index into the label_colors and label_names datastructures for this vertex, pre-computed for us.
    assert vertex_labels_orig[0] == 9182740      # the original ID


def test_annot_get_label_indices():
    vertex_labels, label_colors, label_names, meta_data = an.annot('subject1', TEST_DATA_DIR, 'aparc', hemi='both', orig_ids=True)
    assert vertex_labels[0] == 9182740
    idx_map = an._get_indices_for_unique_vertex_labels(vertex_labels, label_colors)
    assert len(idx_map) == len(label_colors) - 1
    assert len(idx_map) == len(label_names) - 1
    assert idx_map[9182740] == 11


def test_annot_aparc_a2009s():
    vertex_labels, label_colors, label_names, meta_data = an.annot('subject1', TEST_DATA_DIR, 'aparc.a2009s', hemi='both')
    assert label_colors.shape == (NUM_LABELS_APARC_A2009S, 5)


def test_annot_aparc():
    vertex_labels, label_colors, label_names, meta_data = an.annot('subject1', TEST_DATA_DIR, 'aparc.DKTatlas', hemi='both')
    assert label_colors.shape == (NUM_LABELS_APARC_DKTATLAS, 5)


def test_read_label_md_raises_on_invalid_hemisphere_label():
    label_file = os.path.join(TEST_DATA_DIR, 'subject1', 'label', 'rh.cortex.label')
    with pytest.raises(ValueError) as exc_info:
        verts_in_label, meta_data = an.read_label_md(label_file, 'invalid_hemisphere_label')
    assert 'hemisphere_label must be one of' in str(exc_info.value)
    assert 'invalid_hemisphere_label' in str(exc_info.value)


def test_read_label_md_metadata_lh():
    label_file = os.path.join(TEST_DATA_DIR, 'subject1', 'label', 'lh.cortex.label')
    verts_in_label, meta_data = an.read_label_md(label_file, 'lh')
    assert len(meta_data) == 1
    assert meta_data['lh.label_file'] == label_file
    assert verts_in_label.shape == (SUBJECT1_NUM_VERTICES_IN_LABEL_CORTEX_LH, )
    assert len(verts_in_label) < SUBJECT1_SURF_LH_WHITE_NUM_VERTICES
    assert len(verts_in_label) == len(set(verts_in_label)) # Test for duplicate entries


def test_read_label_md_metadata_rh():
    label_file = os.path.join(TEST_DATA_DIR, 'subject1', 'label', 'rh.cortex.label')
    verts_in_label, meta_data = an.read_label_md(label_file, 'rh')
    assert len(meta_data) == 1
    assert meta_data['rh.label_file'] == label_file
    assert verts_in_label.shape == (SUBJECT1_NUM_VERTICES_IN_LABEL_CORTEX_RH, )
    assert len(verts_in_label) < SUBJECT1_SURF_RH_WHITE_NUM_VERTICES
    assert len(verts_in_label) == len(set(verts_in_label))


def test_label_cortex_both_with_meta_data_entry_lh_num_vertices():
    expected_lh_label_file = os.path.join(TEST_DATA_DIR, 'subject1', 'label', 'lh.cortex.label')
    expected_rh_label_file = os.path.join(TEST_DATA_DIR, 'subject1', 'label', 'rh.cortex.label')
    meta_data = {'lh.num_vertices': SUBJECT1_SURF_LH_WHITE_NUM_VERTICES}
    verts_in_label, meta_data = an.label('subject1', TEST_DATA_DIR, 'cortex', hemi='both', meta_data=meta_data)
    assert len(meta_data) == 3
    assert meta_data['lh.label_file'] == expected_lh_label_file
    assert meta_data['rh.label_file'] == expected_rh_label_file
    assert meta_data['lh.num_vertices'] == SUBJECT1_SURF_LH_WHITE_NUM_VERTICES # should be conserved
    assert verts_in_label.shape == (SUBJECT1_NUM_VERTICES_IN_LABEL_CORTEX_LH + SUBJECT1_NUM_VERTICES_IN_LABEL_CORTEX_RH, )
    # Test whether the list contains duplicate entries, this must NOT be the case as the vertex ids should be merged properly:
    assert len(verts_in_label) == len(set(verts_in_label))


def test_label_cortex_both_with_meta_data_entry_lh_num_data_points():
    expected_lh_label_file = os.path.join(TEST_DATA_DIR, 'subject1', 'label', 'lh.cortex.label')
    expected_rh_label_file = os.path.join(TEST_DATA_DIR, 'subject1', 'label', 'rh.cortex.label')
    meta_data = {'lh.num_data_points': SUBJECT1_SURF_LH_WHITE_NUM_VERTICES}
    verts_in_label, meta_data = an.label('subject1', TEST_DATA_DIR, 'cortex', hemi='both', meta_data=meta_data)
    assert len(meta_data) == 3
    assert meta_data['lh.label_file'] == expected_lh_label_file
    assert meta_data['rh.label_file'] == expected_rh_label_file
    assert meta_data['lh.num_data_points'] == SUBJECT1_SURF_LH_WHITE_NUM_VERTICES # should be conserved
    assert verts_in_label.shape == (SUBJECT1_NUM_VERTICES_IN_LABEL_CORTEX_LH + SUBJECT1_NUM_VERTICES_IN_LABEL_CORTEX_RH, )
    # Test whether the list contains duplicate entries, this must NOT be the case as the vertex ids should be merged properly:
    assert len(verts_in_label) == len(set(verts_in_label))


def test_label_raises_on_invalid_hemisphere():
    with pytest.raises(ValueError) as exc_info:
        verts_in_label, meta_data = an.label('subject1', TEST_DATA_DIR, 'cortex', hemi='invalid_hemisphere')
    assert 'hemi must be one of' in str(exc_info.value)
    assert 'invalid_hemisphere' in str(exc_info.value)


def test_label_raises_on_missing_meta_data_if_hemi_is_both():
    with pytest.raises(ValueError) as exc_info:
        verts_in_label, meta_data = an.label('subject1', TEST_DATA_DIR, 'cortex')
    assert 'the meta_data argument is required' in str(exc_info.value)


def test_label_raises_on_missing_meta_data_content_if_hemi_is_both():
    with pytest.raises(ValueError) as exc_info:
        verts_in_label, meta_data = an.label('subject1', TEST_DATA_DIR, 'cortex', meta_data={'not_the': 'right_keys'})
    assert 'the meta_data argument is required' in str(exc_info.value)
    assert 'must contain the key' in str(exc_info.value)


def test_label_is_ok_with_missing_meta_data_if_hemi_is_not_both():
    verts_in_label_lh, meta_data_lh = an.label('subject1', TEST_DATA_DIR, 'cortex', hemi='lh')
    verts_in_label_rh, meta_data_rh = an.label('subject1', TEST_DATA_DIR, 'cortex', hemi='rh')
    assert len(verts_in_label_lh) == SUBJECT1_NUM_VERTICES_IN_LABEL_CORTEX_LH
    assert len(verts_in_label_rh) == SUBJECT1_NUM_VERTICES_IN_LABEL_CORTEX_RH


def test_label_to_mask_normal():
    verts_in_label = [3, 4, 6, 9]
    num_verts_total = 11
    mask = an.label_to_mask(verts_in_label, num_verts_total)
    assert len(mask) == num_verts_total
    assert mask[0] == False
    assert mask[1] == False
    assert mask[2] == False
    assert mask[3] == True
    assert mask[4] == True
    assert mask[5] == False
    assert mask[6] == True
    assert mask[7] == False
    assert mask[8] == False
    assert mask[9] == True
    assert mask[10] == False


def test_label_to_mask_invert():
    verts_in_label = [3, 4, 6, 9]
    num_verts_total = 11
    mask = an.label_to_mask(verts_in_label, num_verts_total, invert=True)
    assert len(mask) == num_verts_total
    assert mask[0] == True
    assert mask[1] == True
    assert mask[2] == True
    assert mask[3] == False
    assert mask[4] == False
    assert mask[5] == True
    assert mask[6] == False
    assert mask[7] == True
    assert mask[8] == True
    assert mask[9] == False
    assert mask[10] == True


def test_label_to_mask_raises_on_wrong_input():
    verts_in_label = [3, 4, 6, 9]
    num_verts_total = 3
    with pytest.raises(ValueError) as exc_info:
        mask = an.label_to_mask(verts_in_label, num_verts_total, invert=True)
    assert 'Argument num_verts_total is 3' in str(exc_info.value)
    assert 'must be at least the length of verts_in_label, which is 4' in str(exc_info.value)


def test_create_and_use_binary_mask_example():
    data = np.array([.1, 3.0, 2.1, 7.8, 6.34, 3.0], dtype=float)
    verts_in_label = [0, 4, 5]
    mask = an.label_to_mask(verts_in_label, len(data))
    assert len(mask) == len(data)
    assert mask[0] == True
    assert mask[1] == False
    assert mask[2] == False
    assert mask[3] == False
    assert mask[4] == True
    assert mask[5] == True
    data[mask == False] = np.nan
    assert data[0] == pytest.approx(.1, 0.01)
    assert np.isnan(data[1])
    assert np.isnan(data[2])
    assert np.isnan(data[3])
    assert data[4] == pytest.approx(6.34, 0.01)
    assert data[5] == pytest.approx(3.0, 0.01)


def test_mask_data_using_label():
    data = np.array([.1, 3.0, 2.1, 7.8, 6.34, 3.0], dtype=float)
    verts_in_label = [0, 4, 5]
    masked_data = an.mask_data_using_label(data, verts_in_label)
    assert masked_data[0] == pytest.approx(.1, 0.01)
    assert np.isnan(masked_data[1])
    assert np.isnan(masked_data[2])
    assert np.isnan(masked_data[3])
    assert masked_data[4] == pytest.approx(6.34, 0.01)
    assert masked_data[5] == pytest.approx(3.0, 0.01)
    # make sure the original data was not changed
    assert_allclose(data, np.array([.1, 3.0, 2.1, 7.8, 6.34, 3.0], dtype=float))


def test_mask_data_using_label_with_invert():
    data = np.array([.1, 3.0, 2.1, 7.8, 6.34, 3.0], dtype=float)
    verts_in_label = [0, 4, 5]
    masked_data = an.mask_data_using_label(data, verts_in_label, invert=True)
    assert np.isnan(masked_data[0])
    assert masked_data[1] == pytest.approx(3.0, 0.01)
    assert masked_data[2] == pytest.approx(2.1, 0.01)
    assert masked_data[3] == pytest.approx(7.8, 0.01)
    assert np.isnan(masked_data[4])
    assert np.isnan(masked_data[5])
    # make sure the original data was not changed
    assert_allclose(data, np.array([.1, 3.0, 2.1, 7.8, 6.34, 3.0], dtype=float))


def test_read_vertex_list_file():
    file_name = os.path.join(TEST_DATA_DIR, 'subject1', 'label', 'some_verts.vlabel')
    vert_indices = an.read_vertex_list_file(file_name, sep=' ')
    assert vert_indices.shape == (10, )


def test_vertices_to_label():
    vert_coords, faces, meta_data = bl.subject_mesh('subject1', TEST_DATA_DIR, surf='white', hemi='both')
    selected_vert_indices = [0, 1, 2, 4]
    label_str = an.vertices_to_label(selected_vert_indices, vert_coords)
    expected = """#!ascii label
4
0 -1.852232 -107.982750 22.769730 0.0000000000
1 -2.139076 -108.102333 22.826056 0.0000000000
2 -3.120585 -108.130928 22.830750 0.0000000000
4 -4.798367 -108.059784 22.777563 0.0000000000"""
    assert label_str == expected


def test_annotquery():
    vertex_lookup_indices = np.array([-1, 1, 0, -1, 2])
    label_colors = np.array([[255, 0, 0, 0], [0, 255, 0, 0], [0, 0, 255, 0]])
    label_names = np.array(["red", "green", "blue"])
    aq = an.AnnotQuery(vertex_lookup_indices, label_colors, label_names)
    # now perform a query
    query_vert_indices = np.array([0, 1, 0, 2])
    names = aq.get_vertex_label_names(query_vert_indices)
    colors = aq.get_vertex_label_colors(query_vert_indices)
    assert names.shape == (4, )
    assert names[0] == "None"
    assert names[1] == "green"
    assert names[2] == "None"
    assert names[3] == "red"
    assert colors.shape == (4, 4)
    assert np.array_equal(colors[0], np.array([0,0,0,0], dtype=int))
    assert np.array_equal(colors[1], np.array([0,255,0,0], dtype=int))
    assert np.array_equal(colors[2], np.array([0,0,0,0], dtype=int))
    assert np.array_equal(colors[3], np.array([255,0,0,0], dtype=int))


def test_region_data_native():
    hemi = 'lh'
    morphometry_data, morphometry_meta_data = bl.subject_data_native('subject1', TEST_DATA_DIR, 'area', hemi)
    region_data, label_names = an.region_data_native('subject1', TEST_DATA_DIR, 'aparc', hemi, morphometry_data, morphometry_meta_data)
    assert len(label_names) == 36
    assert len(region_data) == 1    # one hemi only requested
    assert len(region_data[hemi]) == 36
    some_region_name = label_names[5]
    assert some_region_name == "cuneus"
    some_region_data = region_data[hemi][some_region_name]
    assert some_region_data.shape == (2183, )
    mean_thickness_in_region = np.mean(some_region_data)
    assert mean_thickness_in_region == pytest.approx(0.687, 0.1)


def test_get_atlas_region_names_hardcoded():
    region_names = an.get_atlas_region_names_hardcoded('aseg')
    assert len(region_names) == 44

    region_names = an.get_atlas_region_names_hardcoded('aparc')
    assert len(region_names) == 34

    region_names = an.get_atlas_region_names_hardcoded('aparc.a2009s')
    assert len(region_names) == 74

    region_names = an.get_atlas_region_names_hardcoded('aparc.DKTatlas')
    assert len(region_names) == 31
