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


def test_read_subjects_file_csv_format_comma_separated_no_header():
    subjects_file = os.path.join(TEST_DATA_DIR, 'subjects_including_s6.csv')
    subject_ids = nit.read_subjects_file(subjects_file)
    assert len(subject_ids) == 6
    assert 'subject1' in subject_ids
    assert 'subject2' in subject_ids
    assert 'subject3' in subject_ids
    assert 'subject4' in subject_ids
    assert 'subject5' in subject_ids
    assert 'subject6' in subject_ids


def test_read_subjects_file_csv_format_tab_separated_with_header():
    subjects_file = os.path.join(TEST_DATA_DIR, 'subject_files_tab_separated', 'subjects_including_s6_tab_hdr.csv')
    subject_ids = nit.read_subjects_file(subjects_file, has_header_line=True, delimiter='\t')  # the name arg 'delimiter' should be passed on to csv.reader by the function. This is tested here.
    assert len(subject_ids) == 6
    assert 'subject1' in subject_ids
    assert 'subject2' in subject_ids
    assert 'subject3' in subject_ids
    assert 'subject4' in subject_ids
    assert 'subject5' in subject_ids
    assert 'subject6' in subject_ids



def test_detect_subjects_in_directory_with_six_subjects():
    expected_subject2_dir = os.path.join(TEST_DATA_DIR, 'subject2')
    if not os.path.isdir(expected_subject2_dir):
        pytest.skip("Test data for subject2 .. subject5 not available: e.g., directory '%s' does not exist. You can get it by running the 'get_test_data_all.bash' script." % expected_subject2_dir)
    subject_ids = nit.detect_subjects_in_directory(TEST_DATA_DIR)
    assert len(subject_ids) == 6
    assert 'subject1' in subject_ids
    assert 'subject2' in subject_ids
    assert 'subject3' in subject_ids
    assert 'subject4' in subject_ids
    assert 'subject5' in subject_ids
    assert 'subject6' in subject_ids
    assert not 'fsaverage' in subject_ids                       # the default setting is to ignore 'fsaverage'
    assert not 'subject_files_tab_separated' in subject_ids     # this dir exists, but it has no sub dir 'surf' and should thus not be listed


def test_detect_subjects_in_directory_does_not_ignore_fsaverage_when_ignore_list_is_empty():
    expected_subject2_dir = os.path.join(TEST_DATA_DIR, 'subject2')
    if not os.path.isdir(expected_subject2_dir):
        pytest.skip("Test data for subject2 .. subject5 not available: e.g., directory '%s' does not exist. You can get it by running the 'get_test_data_all.bash' script." % expected_subject2_dir)

    subject_ids = nit.detect_subjects_in_directory(TEST_DATA_DIR, ignore_dir_names=[])
    assert len(subject_ids) == 7
    assert 'subject1' in subject_ids
    assert 'subject2' in subject_ids
    assert 'subject3' in subject_ids
    assert 'subject4' in subject_ids
    assert 'subject5' in subject_ids
    assert 'subject6' in subject_ids
    assert 'fsaverage' in subject_ids
    assert not 'subject_files_tab_separated' in subject_ids     # this dir exists, but it has no sub dir 'surf' and should thus not be listed


def test_detect_subjects_in_directory_does_not_ignores_subjects_when_asked_to_via_ignore_listy():
    expected_subject2_dir = os.path.join(TEST_DATA_DIR, 'subject2')
    if not os.path.isdir(expected_subject2_dir):
        pytest.skip("Test data for subject2 .. subject5 not available: e.g., directory '%s' does not exist. You can get it by running the 'get_test_data_all.bash' script." % expected_subject2_dir)

    subject_ids = nit.detect_subjects_in_directory(TEST_DATA_DIR, ignore_dir_names=['subject5', 'subject3'])
    assert len(subject_ids) == 5
    assert 'subject1' in subject_ids
    assert 'subject2' in subject_ids
    assert not 'subject3' in subject_ids
    assert 'subject4' in subject_ids
    assert not 'subject5' in subject_ids
    assert 'subject6' in subject_ids
    assert 'fsaverage' in subject_ids
    assert not 'subject_files_tab_separated' in subject_ids     # this dir exists, but it has no sub dir 'surf' and should thus not be listed


def test_detect_subjects_in_directory_does_not_match_anything_when_adding_nonexistant_required_sub_dirs():
    expected_subject2_dir = os.path.join(TEST_DATA_DIR, 'subject2')
    if not os.path.isdir(expected_subject2_dir):
        pytest.skip("Test data for subject2 .. subject5 not available: e.g., directory '%s' does not exist. You can get it by running the 'get_test_data_all.bash' script." % expected_subject2_dir)

    subject_ids = nit.detect_subjects_in_directory(TEST_DATA_DIR, required_subdirs_for_hits=['not_there'])
    assert len(subject_ids) == 0


def test_detect_subjects_in_directory_does_match_when_adding_existant_required_sub_dirs():
    expected_subject2_dir = os.path.join(TEST_DATA_DIR, 'subject2')
    if not os.path.isdir(expected_subject2_dir):
        pytest.skip("Test data for subject2 .. subject5 not available: e.g., directory '%s' does not exist. You can get it by running the 'get_test_data_all.bash' script." % expected_subject2_dir)

    subject_ids = nit.detect_subjects_in_directory(TEST_DATA_DIR, required_subdirs_for_hits=['this_dir_exists_for_a_unit_test'])
    assert len(subject_ids) == 1
    assert 'subject6' in subject_ids


def test_detect_subjects_in_directory_without_any_subjects():
    subject1_dir = os.path.join(TEST_DATA_DIR, 'subject1')  # This contains the data for a single subject, so none of its sub directories are valid subject dirs.
    subject_ids = nit.detect_subjects_in_directory(subject1_dir)
    assert len(subject_ids) == 0


def test_fill_template_filename():
    template_string = '${hemi}${surf}.${measure}'
    substitution_dict = {'hemi': 'lh', 'surf': '.pial', 'measure': 'area'}
    result = nit.fill_template_filename(template_string, substitution_dict)
    assert result == 'lh.pial.area'


def test_check_hemi_dict_ok_with_both():
    hemi_dict = {'lh': 'string', 'rh': 'string'}
    assert nit._check_hemi_dict(hemi_dict) == True


def test_check_hemi_dict_ok_with_both_None():
    hemi_dict = {'lh': None, 'rh': None}
    assert nit._check_hemi_dict(hemi_dict) == True


def test_check_hemi_dict_fail_with_left():
    hemi_dict = {'lh': None }
    assert nit._check_hemi_dict(hemi_dict) == False


def test_check_hemi_dict_fail_with_right():
    hemi_dict = {'rh': None }
    assert nit._check_hemi_dict(hemi_dict) == False


def test_check_hemi_dict_ok_with_left_not_both_needed():
    hemi_dict = {'lh': None }
    assert nit._check_hemi_dict(hemi_dict, both_required=False) == True


def test_check_hemi_dict_fail_with_right_not_both_needed():
    hemi_dict = {'rh': None }
    assert nit._check_hemi_dict(hemi_dict, both_required=False) == True


def test_do_subject_files_exist_lh_area():
    expected_subjects_dir = TEST_DATA_DIR
    expected_fsaverage_surf_dir = os.path.join(TEST_DATA_DIR, 'fsaverage', 'surf')
    if not os.path.isdir(expected_fsaverage_surf_dir):
        pytest.skip("Test data missing: e.g., directory '%s' does not exist. You can get all test data by running './develop/get_test_data_all.bash' in the repo root." % expected_fsaverage_surf_dir)

    subjects_dir = TEST_DATA_DIR
    subjects_file = os.path.join(subjects_dir, 'subjects.txt')
    subjects_list = nit.read_subjects_file(subjects_file)
    assert len(subjects_list) == 5
    missing = nit.do_subject_files_exist(subjects_list, subjects_dir, filename='lh.area')
    assert len(missing) == 0


def test_do_subject_files_exist_not_there():
    subjects_dir = TEST_DATA_DIR
    subjects_file = os.path.join(subjects_dir, 'subjects.txt')
    subjects_list = nit.read_subjects_file(subjects_file)
    assert len(subjects_list) == 5
    missing = nit.do_subject_files_exist(subjects_list, subjects_dir, filename='not_there')
    assert len(missing) == 5
    assert 'subject1' in missing
    assert 'subject2' in missing
    assert 'subject3' in missing
    assert 'subject4' in missing
    assert 'subject5' in missing


def test_do_subject_files_exist_raises_on_no_fileinfo():
    subjects_dir = TEST_DATA_DIR
    subjects_file = os.path.join(subjects_dir, 'subjects.txt')
    subjects_list = nit.read_subjects_file(subjects_file)
    assert len(subjects_list) == 5
    with pytest.raises(ValueError) as exc_info:
        missing = nit.do_subject_files_exist(subjects_list, subjects_dir)
    assert 'Exactly one of' in str(exc_info.value)


def test_do_subject_files_exist_raises_on_too_much_fileinfo():
    subjects_dir = TEST_DATA_DIR
    subjects_file = os.path.join(subjects_dir, 'subjects.txt')
    subjects_list = nit.read_subjects_file(subjects_file)
    assert len(subjects_list) == 5
    with pytest.raises(ValueError) as exc_info:
        missing = nit.do_subject_files_exist(subjects_list, subjects_dir, filename='not_there', filename_template='not_there')
    assert 'Exactly one of' in str(exc_info.value)


def test_do_subject_files_exist_template_for_existing_files():
    subjects_dir = os.path.join(TEST_DATA_DIR, 'empty_subjects')
    subjects_file = os.path.join(subjects_dir, 'subjects_empty.txt')
    subjects_list = nit.read_subjects_file(subjects_file)
    assert len(subjects_list) == 3
    expected_missing_file1 = os.path.join(subjects_dir, 'empty1', 'surf', 'empty1_not_there.txt')
    expected_missing_file2 = os.path.join(subjects_dir, 'empty2', 'surf', 'empty2_not_there.txt')
    expected_missing_file3 = os.path.join(subjects_dir, 'empty3', 'surf', 'empty3_not_there.txt')
    missing = nit.do_subject_files_exist(subjects_list, subjects_dir, filename_template='${SUBJECT_ID}_not_there.txt')
    assert len(missing) == 3
    assert missing['empty1'] == expected_missing_file1
    assert missing['empty2'] == expected_missing_file2
    assert missing['empty3'] == expected_missing_file3


def test_do_subject_files_exist_template_for_existing_files_exist():
    subjects_dir = os.path.join(TEST_DATA_DIR, 'empty_subjects')
    subjects_file = os.path.join(subjects_dir, 'subjects_empty.txt')
    subjects_list = nit.read_subjects_file(subjects_file)
    assert len(subjects_list) == 3
    missing = nit.do_subject_files_exist(subjects_list, subjects_dir, filename_template='${SUBJECT_ID}.txt')
    assert len(missing) == 0


def test_do_subject_files_exist_with_custom_dir_surf():
    expected_subjects_dir = os.path.join(TEST_DATA_DIR, 'extra_subjects')
    if not os.path.isdir(expected_subjects_dir):
        pytest.skip("Test data missing: e.g., directory '%s' does not exist. You can get all test data by running './develop/get_test_data_all.bash' in the repo root." % expected_subjects_dir)

    subjects_dir = expected_subjects_dir
    subjects_list = ['subject2x', 'subject3x']
    missing = nit.do_subject_files_exist(subjects_list, subjects_dir, filename='lh.area', sub_dir='surf')
    assert len(missing) == 0


def test_do_subject_files_exist_with_custom_dir_mri():
    expected_subjects_dir = os.path.join(TEST_DATA_DIR, 'extra_subjects')
    if not os.path.isdir(expected_subjects_dir):
        pytest.skip("Test data missing: e.g., directory '%s' does not exist. You can get all test data by running './develop/get_test_data_all.bash' in the repo root." % expected_subjects_dir)

    subjects_dir = expected_subjects_dir
    subjects_list = ['subject2x', 'subject3x']
    missing = nit.do_subject_files_exist(subjects_list, subjects_dir, filename='subject2x_mri.txt', sub_dir='mri')
    assert len(missing) == 1
    assert 'subject3x' in missing


def test_do_subject_files_exist_with_custom_dir_None():
    expected_subjects_dir = os.path.join(TEST_DATA_DIR, 'extra_subjects')
    if not os.path.isdir(expected_subjects_dir):
        pytest.skip("Test data missing: e.g., directory '%s' does not exist. You can get all test data by running './develop/get_test_data_all.bash' in the repo root." % expected_subjects_dir)

    subjects_dir = expected_subjects_dir
    subjects_list = ['subject2x', 'subject3x']
    missing = nit.do_subject_files_exist(subjects_list, subjects_dir, filename='subject2x.txt', sub_dir=None)
    assert len(missing) == 1
    assert 'subject3x' in missing
