import os
import csv
import numpy as np


def read_subjects_file(subjects_file, **kwargs):
    """
    Read a subjects file in CSV format that has the subject id as the first entry on each line. Arbitrary data may follow in the consecutive fields on each line, and will be ignored. Having nothing but the subject id on the line is also fine, of course.

    Any additional named arguments you pass will be passed on to the csv.reader call.
    """
    subject_ids = []
    with open(subjects_file, 'r') as sfh:
        reader = csv.reader(sfh, **kwargs)
        for row in reader:
            subject_ids.append(row[0])
    return subject_ids


def detect_subjects_in_directory(subjects_dir, ignore_dir_names=None, required_subdirs_for_hits=None):
    """
    Search for directories containing FreeSurfer output in a directory.

    Given a directory, search its sub directories for FreeSurfer data and return the directory names of all directories in which such data was found. The resulting list can be used
    to create a subjects.txt file. This method searches all direct sub directories of the given subjects_dir for the existance of the typical FreeSurfer output directory structure.
    """
    detected_subjects = []

    if ignore_dir_names is None:
        ignore_dir_names = [ 'fsaverage' ]

    if required_subdirs_for_hits is None:
        required_subdirs_for_hits = [ 'surf', 'mri', 'label' ]

    direct_sub_dirs = [os.path.join(subjects_dir, direct_child) for direct_child in os.listdir(subjects_dir) if os.path.isdir(os.path.join(subjects_dir, direct_child))]

    for potential_subject_dir in direct_sub_dirs:

        potential_subject_id = os.path.basename(potential_subject_dir)
        if potential_subject_id in ignore_dir_names:
            print "ignored dir name: '%s' " % potential_subject_id
            continue

        is_missing_cruical_subdir = False           # Yes, we are using a programming language which cannot break out of nested for loops. ><
        for required_subdir in required_subdirs_for_hits:
            if not os.path.isdir(os.path.join(potential_subject_dir, required_subdir)):
                print "missing sub dir name: '%s' " % os.path.join(potential_subject_dir, required_subdir)
                is_missing_cruical_subdir = True
                break

        if is_missing_cruical_subdir:
            continue

        detected_subjects.append(potential_subject_id)
    return detected_subjects
