import os
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import brainload.nitools as nit
import brainload.annotations as an

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


def test_annot_metadata_both_hemispheres():
    vertex_labels, label_colors, label_names, meta_data = an.annot('subject1', TEST_DATA_DIR, 'aparc', hemi='both')
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


def test_annot_metadata_single_hemi_rh():
    vertex_labels, label_colors, label_names, meta_data = an.annot('subject1', TEST_DATA_DIR, 'aparc', hemi='rh')
    assert len(meta_data) == 1
    assert len(label_names) == NUM_LABELS_APARC
    assert vertex_labels.shape == (SUBJECT1_SURF_RH_WHITE_NUM_VERTICES, )
    assert label_colors.shape == (NUM_LABELS_APARC, 5)


def test_annot_aparc():
    vertex_labels, label_colors, label_names, meta_data = an.annot('subject1', TEST_DATA_DIR, 'aparc', hemi='both')
    assert label_colors.shape == (NUM_LABELS_APARC, 5)


def test_annot_aparc_a2009s():
    vertex_labels, label_colors, label_names, meta_data = an.annot('subject1', TEST_DATA_DIR, 'aparc.a2009s', hemi='both')
    assert label_colors.shape == (NUM_LABELS_APARC_A2009S, 5)


def test_annot_aparc():
    vertex_labels, label_colors, label_names, meta_data = an.annot('subject1', TEST_DATA_DIR, 'aparc.DKTatlas', hemi='both')
    assert label_colors.shape == (NUM_LABELS_APARC_DKTATLAS, 5)


def test_are_label_names_identical_with_identical():
    res = an._are_label_names_identical(["unknown", "same"], ["unknown", "same"])
    assert res == True


def test_are_label_names_identical_with_not_identical():
    res = an._are_label_names_identical(["unknown", "same"], ["same", "same"])
    assert res == False


def test_are_label_names_identical_with_diff_sizes():
    res = an._are_label_names_identical(["same"], ["same", "same"])
    assert res == False


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


def test_label_cortex_both():
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
