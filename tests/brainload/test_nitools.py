import os
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import brainload.nitools as nit

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(THIS_DIR, os.pardir, 'test_data')

# Respect the environment variable BRAINLOAD_TEST_DATA_DIR if it is set. If not, fall back to default.
TEST_DATA_DIR = os.getenv('BRAINLOAD_TEST_DATA_DIR', TEST_DATA_DIR)


def test_read_subjects_file_five_subjects():
    subjects_file = os.path.join(TEST_DATA_DIR, 'subjects.txt')
    subject_ids = nit.read_subjects_file(subjects_file)
    assert len(subject_ids) == 5
    assert 'subject1' in subject_ids
    assert 'subject2' in subject_ids
    assert 'subject3' in subject_ids
    assert 'subject4' in subject_ids
    assert 'subject5' in subject_ids


def test_read_subjects_file_one_subject():
    subjects_file = os.path.join(TEST_DATA_DIR, 'subjects_only1.txt')
    subject_ids = nit.read_subjects_file(subjects_file)
    assert len(subject_ids) == 1
    assert 'subject1' in subject_ids


def test_detect_subjects_in_directory_with_five_subjects():
    expected_subject2_dir = os.path.join(TEST_DATA_DIR, 'subject2')
    if not os.path.isdir(expected_subject2_dir):
        pytest.skip("Test data for subject2 .. subject5 not available: e.g., directory '%s' does not exist. You can get it by running the 'get_test_data_all.bash' script." % expected_subject2_dir)
    subject_ids = nit.detect_subjects_in_directory(TEST_DATA_DIR)
    assert len(subject_ids) == 5
    assert 'subject1' in subject_ids
    assert 'subject2' in subject_ids
    assert 'subject3' in subject_ids
    assert 'subject4' in subject_ids
    assert 'subject5' in subject_ids


def test_detect_subjects_in_directory_without_any_subjects():
    subject1_dir = os.path.join(TEST_DATA_DIR, 'subject1')  # This contains the data for a single subject, so none of its sub directories are valid subject dirs.
    subject_ids = nit.detect_subjects_in_directory(subject1_dir)
    assert len(subject_ids) == 0


def test_fill_template_filename():
    template_string = '${hemi}${surf}.${measure}'
    substitution_dict = {'hemi': 'lh', 'surf': '.pial', 'measure': 'area'}
    result = nit.fill_template_filename(template_string, substitution_dict)
    assert result == 'lh.pial.area'
