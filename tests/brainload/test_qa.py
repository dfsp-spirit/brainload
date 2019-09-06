import os
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import brainload as bl
import brainload.freesurferdata as fsd
import brainload.qa as bqa


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(THIS_DIR, os.pardir, 'test_data')

# Respect the environment variable BRAINLOAD_TEST_DATA_DIR if it is set. If not, fall back to default.
TEST_DATA_DIR = os.getenv('BRAINLOAD_TEST_DATA_DIR', TEST_DATA_DIR)


def test_braindataconsistency_init():
    subjects_list = ['subject1', 'subject2']
    bdc = bqa.BrainDataConsistency(TEST_DATA_DIR, subjects_list)
    assert bdc.hemis == ['lh', 'rh']
    assert len(bdc.subject_issues) == 2
    assert 'subject1' in bdc.subject_issues
    assert 'subject2' in bdc.subject_issues
    assert len(bdc.subject_issues['subject1']) == 0
    assert len(bdc.subject_issues['subject2']) == 0

def test_braindataconsistency_init_invalid_hemi():
    subjects_list = ['subject1', 'subject2']
    with pytest.raises(ValueError) as exc_info:
        bdc = bqa.BrainDataConsistency(TEST_DATA_DIR, subjects_list, hemi='invalid_hemi')
    assert 'hemi must be one of' in str(exc_info.value)
    assert 'invalid_hemi' in str(exc_info.value)


def test_braindataconsistency_check_essentials_runs():
    expected_subject2_testdata_dir = os.path.join(TEST_DATA_DIR, 'subject2')
    if not os.path.isdir(expected_subject2_testdata_dir):
        pytest.skip("Test data missing: e.g., directory '%s' does not exist. You can get all test data by running './develop/get_test_data_all.bash' in the repo root." % expected_subject2_testdata_dir)
    subjects_list = ['subject1', 'subject2']
    bdc = bqa.BrainDataConsistency(TEST_DATA_DIR, subjects_list)
    bdc.check_essentials()
    bdc.check_essentials()  # run again, this time no vertices should be counted


def test_braindataconsistency_check_custom_runs():
    expected_subject2_testdata_dir = os.path.join(TEST_DATA_DIR, 'subject2')
    if not os.path.isdir(expected_subject2_testdata_dir):
        pytest.skip("Test data missing: e.g., directory '%s' does not exist. You can get all test data by running './develop/get_test_data_all.bash' in the repo root." % expected_subject2_testdata_dir)
    subjects_list = ['subject1', 'subject2']
    bdc = bqa.BrainDataConsistency(TEST_DATA_DIR, subjects_list, hemi='lh')
    bdc.check_file_modification_times = False
    bdc.check_custom(['area'], ['area'])


def test_braindataconsistency_pts():
    subjects_list = ['subject1', 'subject2']
    bdc = bqa.BrainDataConsistency(TEST_DATA_DIR, subjects_list)
    assert bdc._pts(1235235455) == "2009-02-21 16:57:35"


def test_braindataconsistency_ptd():
    subjects_list = ['subject1', 'subject2']
    bdc = bqa.BrainDataConsistency(TEST_DATA_DIR, subjects_list)
    assert bdc._ptd(25) == "0:00:25 later"
    assert bdc._ptd(-70) == "0:01:10 earlier"
