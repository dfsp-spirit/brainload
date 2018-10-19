import os
import csv
import string
import collections
import numpy as np


def read_subjects_file(subjects_file, has_header_line=False, **kwargs):
    """
    Read a subjects file in CSV format that has the subject id as the first entry on each line. Arbitrary data may follow in the consecutive fields on each line, and will be ignored. Having nothing but the subject id on the line is also fine, of course.

    Any additional named arguments you pass will be passed on to the csv.reader call.
    """
    subject_ids = []
    with open(subjects_file, 'r') as sfh:
        reader = csv.reader(sfh, **kwargs)
        if has_header_line:
            next(reader)
        for row in reader:
            subject_ids.append(row[0])  # we assume that the subject id is always the first field on each line. This is a requirement of the function.
    return subject_ids


def detect_subjects_in_directory(subjects_dir, ignore_dir_names=None, required_subdirs_for_hits=None):
    """
    Search for directories containing FreeSurfer output in a directory and return the subject names.

    Given a directory, search its sub directories for FreeSurfer data and return the directory names of all directories in which such data was found. The resulting list can be used
    to create a subjects.txt file. This method searches all direct sub directories of the given subjects_dir for the existance of the typical FreeSurfer output directory structure.
    """
    detected_subjects = []

    if ignore_dir_names is None:
        ignore_dir_names = [ 'fsaverage' ]

    if required_subdirs_for_hits is None:
        required_subdirs_for_hits = [ 'surf' ]          # If you add more here, e.g., 'mri' and 'label', be sure to update the test data.

    direct_sub_dirs = [os.path.join(subjects_dir, direct_child) for direct_child in os.listdir(subjects_dir) if os.path.isdir(os.path.join(subjects_dir, direct_child))]

    for potential_subject_dir in direct_sub_dirs:

        potential_subject_id = os.path.basename(potential_subject_dir)
        if potential_subject_id in ignore_dir_names:
            continue

        is_missing_cruical_subdir = False           # Yes, we are using a programming language which cannot break out of nested for loops. ><
        for required_subdir in required_subdirs_for_hits:
            if not os.path.isdir(os.path.join(potential_subject_dir, required_subdir)):
                is_missing_cruical_subdir = True
                break

        if is_missing_cruical_subdir:
            continue

        detected_subjects.append(potential_subject_id)
    return detected_subjects


def fill_template_filename(template_string, substitution_dict):
    """
    Replace variables in the template with the respective substitution dict entries.

    Checks the `template_string` for variables (i.e., something like '${VAR_NAME}') that are listed as keys in `substitution_dict`. If such entries are found, they are replaced with the respective values in the `substitution_dict`.
    """
    return string.Template(template_string).substitute(substitution_dict)


def _check_hemi_dict(hemi_dict, both_required=True):
    """
    Checks whether the given `hemi_dict` variable is a dictionary with the required format.

    If `both_required` is True, checks whether `hemi_dict` is a dictionary containing exactly 2 keys named 'lh' and 'rh'. Otherwise, checks whether at least one of these two keys exists and the length is exactly 1. It does not check the values in the dictionary in any way.

    Returns
    -------
    bool
        Whether the dict is well-formed accordin to the definition given above.
    """
    if not isinstance(hemi_dict, collections.Mapping):
        return False
    if both_required:
        if not len(hemi_dict) == 2 or not ( 'lh' in hemi_dict and 'rh' in hemi_dict ):
            return False
    else:
        if not len(hemi_dict) == 1 or not ( 'lh' in hemi_dict or 'rh' in hemi_dict ):
            return False
    return True


def do_subject_files_exist(subjects_list, subjects_dir, filename=None, filename_template=None):
    """
    Checks for the existance of files in each subject directory.

    Checks for the existance of files in each subject directory. This is useful to see whether data you intend to work on exists for all subjects you are interested in.

    Returns
    -------
    dictionary
        A dictionary. The keys are subjects that are missing the respective file, and the value is the absolute path of the file that is missing. If no files are missing, the dictionary is empty. If none of the subjects have the file, the length of the dictionary is equal to the length of the input `subjects_list`.
    """
    if filename is None and filename_template is None:
        raise ValueError("Exactly one of 'filename' or 'filename_template' must be given.")

    if filename is not None and filename_template is not None:
        raise ValueError("Exactly one of 'filename' or 'filename_template' must be given.")

    missing_files_by_subject = {}
    for subject_id in subjects_list:
        if filename_template is not None:
            substitution_dict = {'SUBJECT_ID': subject_id}
            filename = fill_template_filename(filename_template, substitution_dict)

        full_file = os.path.join(subjects_dir, subject_id, 'surf', filename)
        if not os.path.isfile(full_file):
            missing_files_by_subject[subject_id] = full_file
    return missing_files_by_subject
