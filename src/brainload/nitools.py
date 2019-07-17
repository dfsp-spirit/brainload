"""
Utility functions for loading neuroimaging data.

Most of these functions interact with the filesystem to find data.
"""

import os
import csv
import string
import collections
import numpy as np


def _read_text_file_lines(file_name):
    """
    Open a text file and read it, return a list of lines.

    Open a text file and read it, return a list of lines.
    """
    with open(file_name, 'r') as fh:
        lines = [line.rstrip('\n') for line in fh]
    return lines



def read_subjects_file(subjects_file, has_header_line=False, index_of_subject_id_field=0, **kwargs):
    """
    Read a subjects file in CSV format that has the subject id as the first entry on each line. Arbitrary data may follow in the consecutive fields on each line, and will be ignored. Having nothing but the subject id on the line is also fine, of course.

    The file can be a simple text file that contains one `subject_id` per line. It can also be a CSV file that has other data following, but the `subject_id` has to be the first item on each line and the separator must be a comma. So a line is allowed to look like this: `subject1, 35, center1, 147`. No header is allowed. If you have a different format, consider reading the file yourself and pass the result as `subjects_list` instead.

    Parameters
    ----------
    subjects_file: string
        Path to a subjects file (see above for format details).

    has_header_line: boolean, optional
        Whether the first line is a header line and should be skipped. Defaults to 'False'.

    index_of_subject_id_field: integer, optional
        The column index of the field that contains the subject id in each row. Defaults to '0'. Changing this only makes sense for CSV files.

    **kwargs: any
        Any other named arguments will be passed on to the call to the call to the `csv.reader` constructor. That is a class from Python's standard `csv` module. Example: pass `delimiter='\t'` if your CSV file is limited by tabs.

    Returns
    -------
    list of strings
        A list of subject identifiers.

    Examples
    --------
    Load a list of subjects from a simple text file that contains one subject per line.

    >>> import brainload.nitools as nit
    >>> subjects_ids = nit.read_subjects_file('/home/myuser/data/study5/subjects.txt')
    """
    subject_ids = []
    with open(subjects_file, 'r') as sfh:
        reader = csv.reader(sfh, **kwargs)
        if has_header_line:
            next(reader)
        for row in reader:
            subject_ids.append(row[index_of_subject_id_field])
    return subject_ids


def detect_subjects_in_directory(subjects_dir, ignore_dir_names=None, required_subdirs_for_hits=None):
    """
    Search for directories containing FreeSurfer output in a directory and return the subject names.

    Given a directory, search its sub directories for FreeSurfer data and return the directory names of all directories in which such data was found. The resulting list can be used
    to create a subjects.txt file. This method searches all direct sub directories of the given subjects_dir for the existance of the typical FreeSurfer output directory structure.

    Parameters
    ----------
    subjects_dir: string
        Path to a subjects directory.

    ignore_dir_names: list of strings | None, optional
        A list of directory names that should be ignored, even if they have the required sub directories. This is useful if you do not want to load certain subjects. It is often used to avoid loading the average subject 'fsaverage'. Defaults to a list with the single element 'fsaverage'. You can explicitely pass an empty list if you want to include all subjects.

    required_subdirs_for_hits: list of strings | None
        A sub directory of the given `subjects_dir` is considered a subject if it contains the typical FreeSurfer directory structure. Which sub directories are required is determined by this argument. If all of them are found under a dir, that dir is added tp the output list. This list defaults to a list with the single element 'surf'. If that leads to false positives in your case, you could pass something like `['surf', 'mri', 'label']`.

    Returns
    -------
    list of strings
        A list of the subject identifiers (or directories that were considered as such).

    Examples
    --------
    Guess which directories under the current SUBJECTS_DIR contain subject data:

    >>> import brainload.nitools as nit
    >>> import os
    >>> my_subject_dir = os.getenv('SUBJECTS_DIR')
    >>> subjects_ids = nit.detect_subjects_in_directory(my_subject_dir, ignore_dir_names=['fsaverage', 'Copy of subject4'])
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

    Checks the `template_string` for variables (i.e., something like '${VAR_NAME}') that are listed as keys in `substitution_dict`. If such entries are found, they are replaced with the respective values in the `substitution_dict`. This function only calls `ting.Template().substitute()` in the background.

    Parameters
    ----------
    template_string: string
        A template string, see the `string.Template` constructor in the standard Python `string` module. Variable names must be enclosed in `${}`. Example: `${SUBJECT_ID}_hardcoded_text`.

    substitution_dict: dictionary string, string
        The keys are variable names, values are the replacements. See `string.Template.substitute` in the standard Python `string` module. Example: `{ 'SUBJECT_ID' : 'subject3' }`.

    Returns
    -------
    string
        The result of the replacement.

    Examples
    --------
    Fill in a template string:

    >>> import brainload.nitools as nit
    >>> template_str = '${HEMI}.white'
    >>> substitution_dict = {'HEMI' : 'lh'}
    >>> print nit.fill_template_filename(template_str, substitution_dict)
    lh.white
    """
    return string.Template(template_string).substitute(substitution_dict)


def _check_hemi_dict(hemi_dict, both_required=True):
    """
    Checks whether the given `hemi_dict` variable is a dictionary with the required format.

    If `both_required` is True, checks whether `hemi_dict` is a dictionary containing exactly 2 keys named 'lh' and 'rh'. Otherwise, checks whether at least one of these two keys exists and the length is exactly 1. It does not check the values in the dictionary in any way.

    Parameters
    ----------
    hemi_dict: dictionary

    both_required: boolean, optional
        If set to True, this function only returns True if (only) the two keys 'lh' and 'rh' are contained in the dictionary. Otherwise, only one of them is enough.

    Returns
    -------
    bool
        Whether the dict is well-formed according to the definition given above.

    Examples
    --------
    Check whether a dict is fine:

    >>> import brainload.nitools as nit
    >>> my_dict = {'lh' : 'lh.area', 'rh': 'rh.area'}
    >>> print nit._check_hemi_dict(my_dict)
    True
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


def do_subject_files_exist(subjects_list, subjects_dir, filename=None, filename_template=None, sub_dir='surf'):
    """
    Checks for the existance of certain files in each subject directory for a group of subjects.

    Checks for the existance of certain files in the each subject directory for a group of subjects. This is useful to see whether data you intend to work on exists for all subjects you are interested in.

    Parameters
    ----------
    subjects_list: list of strings
        List of subject ids.

    subjects_dir: string
        Path to a directory that contains the subject data.

    filename: string
        A string representing the file name within the `sub_dir` sub directory of each subject, hardcoded. You must supply this or a `filename_template`.

    filename_template: string
        A string representing the file name within the 'surf' sub directory of each subject as a template. You must supply this or a `filename`, but not both. You can use the variable `${SUBJECT_ID}` in the template.

    sub_dir: string | None, optional
        The sub directory to look in. You could set any value, but the typical ones are the default FreeSurfer directories, e.g., 'surf', 'mri', 'scripts' and so on. You can set this to `None` if you want to look directly in the subjct's dir, but FreeSurfer does not seem to store any data there by default. Defaults to 'surf'.

    Returns
    -------
    dictionary
        A dictionary. The keys are subjects that are missing the respective file, and the value is the absolute path of the file that is missing. If no files are missing, the dictionary is empty. If none of the subjects have the file, the length of the dictionary is equal to the length of the input `subjects_list`.

    Examples
    --------
    Check whether a file exists for all subjects:

    >>> import brainload.nitools as nit
    >>> subjects_list = ['subject1', 'subject4', 'subject7']
    >>> subjects_dir = subjects_dir = os.path.join(os.getenv('HOME'), 'data', 'my_study_x')
    >>> searched_file = 'lh.area'
    >>> missing = nit.do_subject_files_exist(subjects_list, subjects_dir, filename=searched_file)
    >>> print "The file '%s' exists for %d of the %d subjects." % (searched_file, len(missing), len(subjects_list))
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

        if sub_dir is None:
            full_file = os.path.join(subjects_dir, subject_id, filename)
        else:
            full_file = os.path.join(subjects_dir, subject_id, sub_dir, filename)
        if not os.path.isfile(full_file):
            missing_files_by_subject[subject_id] = full_file
    return missing_files_by_subject


def write_lines_to_text_file(lines, file_name, line_sep="\n"):
    """
    Write the lines to a text file.

    Write the lines to a text file, overwriting it in case it exists.

    Parameters
    ----------
    lines: list of str
        The lines, must not contain line ending.

    file_name: str
        Path to new text file to create (or overwrite if it exists).

    line_sep: str, optional
        Line separator. Defaults to "\n".
        
    """
    with open(file_name, 'w') as f:
        for l in lines:
            f.write("%s%s" % (l, line_sep))


def write_subjects_file(file_name, subjects_list):
    lines = ["%s" % (s) for s in subjects_list]
    write_lines_to_text_file(lines, file_name)


def load_vertex_indices(vertex_indices_file):
    return np.loadtxt(vertex_indices_file, dtype=np.uint32, delimiter=",")


def save_vertex_indices(vertex_indices_file, vertex_indices):
    np.savetxt(vertex_indices_file, vertex_indices, delimiter=",")

def load_voxel_indices(vertex_indices_file):
    return np.loadtxt(vertex_indices_file, dtype=np.uint32, delimiter=",")
