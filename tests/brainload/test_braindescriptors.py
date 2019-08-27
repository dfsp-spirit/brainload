import os
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import brainload as bl
import brainload.freesurferdata as fsd
import brainload.braindescriptors as bd


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(THIS_DIR, os.pardir, 'test_data')

# Respect the environment variable BRAINLOAD_TEST_DATA_DIR if it is set. If not, fall back to default.
TEST_DATA_DIR = os.getenv('BRAINLOAD_TEST_DATA_DIR', TEST_DATA_DIR)

def test_braindescriptors_init_nonempty():
    expected_subject2_testdata_dir = os.path.join(TEST_DATA_DIR, 'subject2')
    if not os.path.isdir(expected_subject2_testdata_dir):
        pytest.skip("Test data missing: e.g., directory '%s' does not exist. You can get all test data by running './develop/get_test_data_all.bash' in the repo root." % expected_subject2_testdata_dir)
    subjects_list = ['subject1', 'subject2']
    bdi = bd.BrainDescriptors(TEST_DATA_DIR, subjects_list)
    assert len(bdi.subjects_list) == 2
    assert len(bdi.descriptor_names) == 0
    assert len(bdi.hemis) == 2
    assert bdi.descriptor_values.shape == (2, 0)


def test_braindescriptors_init_with_hemi():
    expected_subject2_testdata_dir = os.path.join(TEST_DATA_DIR, 'subject2')
    if not os.path.isdir(expected_subject2_testdata_dir):
        pytest.skip("Test data missing: e.g., directory '%s' does not exist. You can get all test data by running './develop/get_test_data_all.bash' in the repo root." % expected_subject2_testdata_dir)
    subjects_list = ['subject1', 'subject2']
    bdi = bd.BrainDescriptors(TEST_DATA_DIR, subjects_list, hemi='lh')
    bdi.report_descriptors()
    bdi.check_for_hemi_dependent_file([])
    assert len(bdi.subjects_list) == 2
    assert len(bdi.descriptor_names) == 0
    assert bdi.descriptor_values.shape == (2, 0)
    assert len(bdi.hemis) == 1


def test_check_for_NaNs_no_descriptors_yet():
    expected_subject2_testdata_dir = os.path.join(TEST_DATA_DIR, 'subject2')
    if not os.path.isdir(expected_subject2_testdata_dir):
        pytest.skip("Test data missing: e.g., directory '%s' does not exist. You can get all test data by running './develop/get_test_data_all.bash' in the repo root." % expected_subject2_testdata_dir)
    subjects_list = ['subject1', 'subject2']
    bdi = bd.BrainDescriptors(TEST_DATA_DIR, subjects_list, hemi='lh')
    subjects_over_threshold, descriptors_over_threshold, nan_share_per_subject, nan_share_per_descriptor = bdi.check_for_NaNs()
    assert len(subjects_over_threshold) == 0
    assert len(descriptors_over_threshold) == 0


def test_check_for_NaNs_with_curv_descriptors():
    expected_subject2_testdata_dir = os.path.join(TEST_DATA_DIR, 'subject2')
    if not os.path.isdir(expected_subject2_testdata_dir):
        pytest.skip("Test data missing: e.g., directory '%s' does not exist. You can get all test data by running './develop/get_test_data_all.bash' in the repo root." % expected_subject2_testdata_dir)
    subjects_list = ['subject1', 'subject2']
    bdi = bd.BrainDescriptors(TEST_DATA_DIR, subjects_list, hemi='lh')
    bdi.add_curv_stats()
    subjects_over_threshold, descriptors_over_threshold, nan_share_per_subject, nan_share_per_descriptor = bdi.check_for_NaNs()
    assert len(subjects_over_threshold) == 0
    assert len(descriptors_over_threshold) == 0


def test_check_for_custom_measure_stats_files_invalid_format():
    expected_subject2_testdata_dir = os.path.join(TEST_DATA_DIR, 'subject2')
    if not os.path.isdir(expected_subject2_testdata_dir):
        pytest.skip("Test data missing: e.g., directory '%s' does not exist. You can get all test data by running './develop/get_test_data_all.bash' in the repo root." % expected_subject2_testdata_dir)
    subjects_list = ['subject1', 'subject2']
    bdi = bd.BrainDescriptors(TEST_DATA_DIR, subjects_list, hemi='rh')
    with pytest.raises(ValueError) as exc_info:
        bdi.check_for_custom_measure_stats_files(["aparc"], ["area"], morph_file_format="nosuchformat")
    assert "nosuchformat" in str(exc_info.value)
    assert "morph_file_format must be one of" in str(exc_info.value)


def test_check_for_custom_measure_stats_files_curv_format():
    expected_subject2_testdata_dir = os.path.join(TEST_DATA_DIR, 'subject2')
    if not os.path.isdir(expected_subject2_testdata_dir):
        pytest.skip("Test data missing: e.g., directory '%s' does not exist. You can get all test data by running './develop/get_test_data_all.bash' in the repo root." % expected_subject2_testdata_dir)
    subjects_list = ['subject1', 'subject2']
    bdi = bd.BrainDescriptors(TEST_DATA_DIR, subjects_list, hemi='rh')
    bdi.check_for_custom_measure_stats_files(["aparc"], ["area"], morph_file_format="curv")


def test_check_for_custom_measure_stats_files_mgh_format():
    expected_subject2_testdata_dir = os.path.join(TEST_DATA_DIR, 'subject2')
    if not os.path.isdir(expected_subject2_testdata_dir):
        pytest.skip("Test data missing: e.g., directory '%s' does not exist. You can get all test data by running './develop/get_test_data_all.bash' in the repo root." % expected_subject2_testdata_dir)
    subjects_list = ['subject1', 'subject2']
    bdi = bd.BrainDescriptors(TEST_DATA_DIR, subjects_list, hemi='rh')
    bdi.check_for_custom_measure_stats_files(["aparc"], ["area"], morph_file_format="mgh")


def test_braindescriptors_init_with_invalid_hemi():
    expected_subject2_testdata_dir = os.path.join(TEST_DATA_DIR, 'subject2')
    if not os.path.isdir(expected_subject2_testdata_dir):
        pytest.skip("Test data missing: e.g., directory '%s' does not exist. You can get all test data by running './develop/get_test_data_all.bash' in the repo root." % expected_subject2_testdata_dir)
    subjects_list = ['subject1', 'subject2']
    with pytest.raises(ValueError) as exc_info:
            bdi = bd.BrainDescriptors(TEST_DATA_DIR, subjects_list, hemi='nosuchhemi')
    assert "hemi must be one of {'lh', 'rh', 'both'} but is" in str(exc_info.value)
    assert "nosuchhemi" in str(exc_info.value)


def test_braindescriptors_parcellation_stats():
    expected_subject2_testdata_dir = os.path.join(TEST_DATA_DIR, 'subject2')
    if not os.path.isdir(expected_subject2_testdata_dir):
        pytest.skip("Test data missing: e.g., directory '%s' does not exist. You can get all test data by running './develop/get_test_data_all.bash' in the repo root." % expected_subject2_testdata_dir)
    subjects_list = ['subject1', 'subject2']
    bdi = bd.BrainDescriptors(TEST_DATA_DIR, subjects_list)
    bdi.add_parcellation_stats(['aparc', 'aparc.a2009s'])
    bdi.add_segmentation_stats(['aseg'])
    bdi.add_custom_measure_stats(['aparc'], ['area'])
    bdi.add_curv_stats()
    assert len(bdi.descriptor_names) == 3089
    assert bdi.descriptor_values.shape == (2, 3089)


def test_braindescriptors_add_standard_stats():
    expected_subject2_testdata_dir = os.path.join(TEST_DATA_DIR, 'subject2')
    if not os.path.isdir(expected_subject2_testdata_dir):
        pytest.skip("Test data missing: e.g., directory '%s' does not exist. You can get all test data by running './develop/get_test_data_all.bash' in the repo root." % expected_subject2_testdata_dir)
    subjects_list = ['subject1', 'subject2']
    bdi = bd.BrainDescriptors(TEST_DATA_DIR, subjects_list)
    bdi.add_standard_stats()
    assert len(bdi.descriptor_names) == 3426
    assert bdi.descriptor_values.shape == (2, 3426)


def test_braindescriptors_standard_stats_have_unique_names():
    expected_subject2_testdata_dir = os.path.join(TEST_DATA_DIR, 'subject2')
    if not os.path.isdir(expected_subject2_testdata_dir):
        pytest.skip("Test data missing: e.g., directory '%s' does not exist. You can get all test data by running './develop/get_test_data_all.bash' in the repo root." % expected_subject2_testdata_dir)
    subjects_list = ['subject1', 'subject2']
    bdi = bd.BrainDescriptors(TEST_DATA_DIR, subjects_list)
    bdi.add_standard_stats()
    assert len(bdi.descriptor_names) == 3426
    assert bdi.descriptor_values.shape == (2, 3426)
    assert len(bdi.descriptor_names) == len(list(set(bdi.descriptor_names)))
    dup_list = bdi._check_for_duplicate_descriptor_names()
    assert not dup_list


def test_braindescriptors_file_checks():
    expected_subject2_testdata_dir = os.path.join(TEST_DATA_DIR, 'subject2')
    if not os.path.isdir(expected_subject2_testdata_dir):
        pytest.skip("Test data missing: e.g., directory '%s' does not exist. You can get all test data by running './develop/get_test_data_all.bash' in the repo root." % expected_subject2_testdata_dir)
    subjects_list = ['subject1', 'subject2']
    bdi = bd.BrainDescriptors(TEST_DATA_DIR, subjects_list)
    bdi.check_for_parcellation_stats_files(['aparc', 'aparc.a2009s'])
    bdi.check_for_segmentation_stats_files(['aseg', 'wmparc'])
    bdi.check_for_custom_measure_stats_files(['aparc'], ['area'])
    bdi.check_for_curv_stats_files()
    assert len(bdi.subjects_list) == 2
    assert len(bdi.descriptor_names) == 0
    assert bdi.descriptor_values.shape == (2, 0)
