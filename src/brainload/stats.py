"""
Functions for parsing FreeSurfer brain stat files.

You can use these to read files like `subject/stats/aseg.stats`. Note that these functions read the stats files for a single subject, typically from the 'stats' sub directory of that subject.

Notes
-----
    - You could also use the FreeSurfer command 'aparcstats2table' to merge data from many subjects for a measure (like 'thickness') into a single file and parse that file. This is NOT what is done by these functions though, they are designed for the stats files that are generated by default.
    - The information in the `aseg.stats` file is of interest because many brain properties interact with brain volume or cortical thickness, so people often use these as covariates in their models.
    - The atlas stats files (e.g., `lh.aparc.stats`) contain information on the different brain regions, based on registering the respective subject to a brain atlas.
"""

import warnings
import os
import numpy as np

def stat(file_name):
    """
    Read information from a FreeSurfer stats file.

    Read information from a FreeSurfer stats file, e.g., `subject/stats/lh.aparc.stats` or `aseg.stats`. A stats file is a text file that contains a data table and various meta data.

    Parameters
    ----------
    file_name: string
        The path to the stats file.

    Returns
    -------
    dictionary of strings (includes nested sub dicts)
        The result dictionary, containing the following 4 keys:
            - 'ignored_lines': list of strings. The list of lines that were not parsed in a special way. This is raw data.
            - 'measures': string list of dimension (n, m) if there are n measures with m properties each stored in the stats file.
            - 'table_data': string list of dimension (i, j) when there are i lines containing j values each in the table stored in the stats file. You may want to convert the columns to the proper data types and put the result into several numpy arrays or a single Pandas data frame.
            - 'table_column_headers': string list. The names for the columns for the table_data. This information is parsed from the table_meta_data and given here for convenience.
            - 'table_meta_data': dictionary. The full table_meta_data. Stores properties in key, value sub dictionaries. For simple table properties, the dictionaries are keys of the returned dictionary. The only exception is the information on the table columns (header data). This information can be found under the key `column_info_`, which contains one dictionary for each column. In these dictionaries, data is stored as explained for simple table properties.

    Examples
    --------
    Read the `aseg.stats` file for a subject:

    >>> import brainload as bl
    >>> stats = bl.stats('/path/to/study/subject1/stats/aseg.stats')

    Collect some data, just to show the data structures.

    >>> print(len(stats['measures']))    # Will print the number of measures.
    >>> print("|".join(stats['measures'][0]))  #  Print all data on the first measure.

    Now lets print the table_data:

    >>> num_data_rows = len(stats['table_data'])
    >>> num_entries_per_row = len(stats['table_data'][0])

    And get some information on the table columns (the table header):

    >>> print stats['table_meta_data']['NTableCols']   # will print "10" (from a simple table property stored directly in the dictionary).

    Get the names of all the data columns:

    >>> print ",".join(stats['table_column_headers'])

    Get the name of the first column:

    >>> first_column_name = stats['table_column_headers'][0]

    More detailed information on the individual columns can be found under the special `column_info_` key if needed:

    >>> column2_info_dict = stats['table_meta_data']['column_info_']['2']
    >>> print(column2_info_dict['some_key'])          # will print the value

    Note that all data is returned as string type, you will need to covert it to float (or whatever) yourself.
    """
    with open(file_name) as fh:
        lines = [line.rstrip('\n') for line in fh]
    return _parse_stats_lines(lines)


def _parse_stats_lines(lines):
    """
    Parse all lines from a stats file. See read_stats_file for details.
    """
    ignored_lines = []
    measures = []
    table_rows = []
    table_meta_data = {}
    for line in lines:
        if line.startswith('#'):
            if line.startswith('# Measure '):
                measures.append(_measure(line))
            elif line.startswith('# TableCol') or line.startswith('# NRows') or line.startswith('# NTableCols') or line.startswith('# ColHeaders'):
                table_meta_data = _table_meta_data(line, table_meta_data)
            else:
                ignored_lines.append(line)
        else:
            table_rows.append(_table_row(line))
    results = {}
    results['ignored_lines'] = ignored_lines
    results['measures'] = measures
    results['table_data'] = table_rows
    results['table_meta_data'] = table_meta_data
    results['table_column_headers'] = _header_line_elements(table_meta_data)
    return results


def _measure(line):
    """
    Parse a measure line.

    Parse a measure line. Looks similar to the following: '# Measure BrainSeg, BrainSegVol, Brain Segmentation Volume, 1243340.000000, mm^3'

    Returns
    -------
    list of strings
        A list of strings, containing the data on the line. The prefix, '# Measure', is discarded.
    """
    return line[10:].split(', ')    # ignore first 10 characters, the '# Measure' prefix.


def _table_row(line):
    """
    Return all elements of a data line.

    Return all elements of a data line. Simply splits it.
    """
    return line.split()


def _table_meta_data(line, table_meta_data):
    """
    Parse a table meta data line.

    Parse a table meta data line, and add the result to the given table_meta_data dictionary.

    Parameters
    ----------
    line: string
        a table meta data line from the stats file

    table_meta_data: dictionary
        a dictionary which may already contain some data. The result of parsing the line is added to it.

    Returns
    -------
    dictionary
        The result of parsing the line, as a key value pair of strings. If the line is a TableCol line, the information gets stored into a sub dictionary structure called 'column_info_[index]'. Otherwise, it gets stored directly in the result dictionary.
    """
    key_string = _get_column_info_keystring()
    if line.startswith('# TableCol'):
        if not key_string in table_meta_data.keys():
            table_meta_data[key_string] = {}
        line_entries = line.split(None, 4)
        column_index = line_entries[2]
        header_keyword = line_entries[3]
        header_value = line_entries[4].rstrip()
        if not column_index in table_meta_data[key_string].keys():
            table_meta_data[key_string][column_index] = {}
        table_meta_data[key_string][column_index][header_keyword] = header_value
    else:
        line_entries = line.split(None, 2)
        table_meta_data[line_entries[1]] = line_entries[2].rstrip()
    return table_meta_data


def _get_column_info_keystring():
    """
    Define the special key name within the table_meta_data dictionary for columns.

    Define the special key name within the table_meta_data dictionary that is used to hold information on all columns (parsed from the '# TableCol' lines). Could be any string, as long as it does not clash with other meta data entries in the stats file.
    """
    return 'column_info_'


def _sorted_header_indices(table_meta_data):
    """
    Return the header indices from table_meta_data['column_info_'] sorted.

    Return the header indices from table_meta_data['column_info_'] sorted. The keys of that dictionary are strings originating from the column labels in the stats file ('1', '2', ...). This functions sorts them as if they were integers.
    """
    key_string = _get_column_info_keystring()
    sorted_field_list = table_meta_data[key_string].keys()
    sorted_field_list = sorted(sorted_field_list, key=int)
    return sorted_field_list


def _header_line(table_meta_data, field_separator='\t'):
    """
    Return the table header line in a csv format.

    Return the table header line in a csv format, using the given field_separator.
    """
    return field_separator.join(_header_line_elements(table_meta_data))


def _header_line_elements(table_meta_data):
    """
    Return a list of the table header entries.

    Return a list of the table header entries in the correct order.
    """
    key_string = _get_column_info_keystring()

    # If possible, retrieve the column headers from the 'ColHeaders' line
    header_elements_from_col_headers = None
    if 'ColHeaders' in table_meta_data:
        try:
            header_elements_from_col_headers = table_meta_data['ColHeaders'].split()
        except:
            pass

    # The same information can be retrieved from all the individual 'TableCol' lines
    header_elements_from_table_col = None
    try:
        column_indices = _sorted_header_indices(table_meta_data)
        header_elements_from_table_col = []
        for column_index in column_indices:
            header_elements_from_table_col.append(table_meta_data[key_string][column_index]['ColHeader'])
    except:
        pass

    if header_elements_from_col_headers is None and header_elements_from_table_col is None:
        raise ValueError('Could not determine header line: stats file contains no table header information. Broken stats file?')
    if header_elements_from_col_headers is None or header_elements_from_table_col is None:
        warnings.warn('Stats data is missing some header data. Incomplete stats file? Could still parse requested data, but you may want to check the stats file.', UserWarning)
        if header_elements_from_col_headers is None:
            return header_elements_from_table_col
        else:
            return header_elements_from_col_headers
    else:
        if not header_elements_from_table_col == header_elements_from_col_headers:
            warnings.warn('Stats data regarding table header is inconsistent between ColHeaders and TableCol->ColHeader entries. Returning data based on TableCol->ColHeader entries.', UserWarning)
        return header_elements_from_table_col


def typelist_for_aseg_stats():
    """
    Determine list of numpy data types for the table in an `aseg.stats` file.

    Determine list of numpy data types for the table in an `aseg.stats` file. The 10 columns in this file are: Index SegId NVoxels Volume_mm3 StructName normMean normStdDev normMin normMax normRange.

    Returns
    -------
    list of numpy data types
        List of the proper numpy data types to use for each data column in the file.
    """
    f = np.float32
    i = np.int32
    s = np.string_
    return [i, i, i, f, s, f, f, f, f, f]


def typelist_for_aparc_atlas_stats():
    """
    Determine list of numpy data types for the table in an aparc atlas file.

    Determine list of numpy data types for the table in an aparc atlas file. The type list is identical for the files `?h.aparc.stats`, `?h.aparc.2009s.stats`, and `?h.aparc.DKTatlas.stats`. The 10 columns in each file are: StructName NumVert SurfArea GrayVol ThickAvg ThickStd MeanCurv GausCurv FoldInd CurvInd

    Returns
    -------
    list of numpy data types
        List of the proper numpy data types to use for each data column in the file.
    """
    f = np.float32
    i = np.int32
    s = np.string_
    return [s, i, i, i, f, f, f, f, i, f]


def measures_to_numpy(measures, requested_measures=None, dtype=np.float_):
    """
    Convert the measures list of lists to a 2D numpy array of the given type.

    Convert the measures list of lists to a 2D numpy array of the given type. If only some of the measures are compatible with the type, you can give the names of all requested measures as a dictionary.

    Parameters
    ----------
    measures: list of str lists
        measures as returned by the stat() function: each element of the outer list represents a row in a stats file, and the inner string list contains the tokens of the line

    requested_measures: list of string 2-tuples, optional
        If given, only the measures listed in here are used. Each measure is identified by 2 strings, which must match the first and second token on the measure line. For a line like '# Measure Cortex, NumVert, Number of Vertices, 140843, unitless', the 2 strings of a tuple would be 'Cortex' and 'NumVert'. If omitted, all measures will be used.

    dtype: numpy data type, optional
        The data type that should be used for the returned numpy array. Defaults to ```np.float_``` if omitted.

    Returns
    -------
    measures_data: numpy 1D array
        The measure values, with the requested data type. The shape is (n, ) for n (requested) measures. The order is as given in the parameter measures. (If requested_measures is set, only those are included.)

    measure_names: list of string 2-tuples
        The names of the measures (same order as the data). The order is guaranteed to be identical to the order of measures in the input argument. (If requested_measures is set, only those are included.)
    """
    measure_values = []
    measure_names = []
    for line_tokens in measures:
        measure_unique_tuple = (line_tokens[0], line_tokens[1])
        measure_value = line_tokens[3]
        if requested_measures is not None:
            if measure_unique_tuple in requested_measures:
                measure_values.append(measure_value)
                measure_names.append(measure_unique_tuple)
        else:
            measure_values.append(measure_value)
            measure_names.append(measure_unique_tuple)
    np_measures = np.array(measure_values, dtype=dtype)
    return np_measures, measure_names


def stats_table_to_numpy(stat, type_list):
    """
    Given types, convert the string matrix to a dictionary of numpy arrays.

    Given types, convert the string matrix to a dictionary of numpy arrays. The stat dictionary is returned by the stat function, and you have to specify a list of numpy types, one for each column, to convert this. The type list is specific for the file that has been parsed, i.e., it differs between asge.stats and lh.aparc.stats. Determine it by looking at the file data. See the `typelist_for_*` functions in this module for pre-defined type lists for commonly parsed FreeSurfer stats files.

    Parameters
    ----------
    stats: dictionary
        The data returned by the stat() function. Must contain the keys 'table_data' (2D list of strings, dimension n x m for n rows with m columns each) and 'table_column_headers' (1D list of m strings).

    type_list: list of numpy types
        List of numpy types with length m. Types must be listed in the order in which they should be applied to the columns.

    Returns
    -------
    dictionary of string : numpy array
        Each key is a column name, and each value is a numpy column array containing the typed data with shape (n, ) for n data rows in the table for the subject.
    """
    table = stat['table_data']
    header = stat['table_column_headers']
    if not (len(type_list) == len(header)):
        raise ValueError('Length of type_list (%d) must match number of stat[table_column_headers] (%d).' % (len(type_list), len(header)))
    result = {}
    numpy_string_matrix = np.array(table)
    for column_index, column_name in enumerate(header):
        numpy_string_array_column = numpy_string_matrix[:,column_index]
        numpy_typed_array_column = numpy_string_array_column.astype(type_list[column_index])
        result[column_name] = numpy_typed_array_column
    return result


def _measure_names_from_tuples(measure_name_tuples):
    return [name_tuple[0]+","+name_tuple[1] for name_tuple in measure_name_tuples]        # join the 2 fields with ",". E.g., turn "Cortex" and "NumVert" into a unique name "Cortex,NumVert". The comma is a good choice as it cannot appear in the strings: it is the field separator in the source file. This gives us a key name for the measure.

def _stats_measures_to_dict(numpy_measures, measure_name_tuples):
    num_measures = numpy_measures.shape[0]
    num_names = len(measure_name_tuples)
    if num_measures != num_names:
        raise ValueError("Length mismatch: expected same number of measures and names, got %d and %d." % (num_measures, num_names))
    measures_dict = {}
    measure_names = _measure_names_from_tuples(measure_name_tuples)
    for idx, measure_name in enumerate(measure_names):
        print("_stats_measures_to_dict: measure_names %s value %f idx %d" % (measure_name, numpy_measures[idx], idx))
        measures_dict[measure_name] = np.array([numpy_measures[idx]])
    return measures_dict


def _append_stats_measures_to_dict(measures_dict, numpy_measures, measure_name_tuples):
    """
    Append the new measures to measures_dict.

    Append the new measures to measures_dict. The new measure values are given by numpy_measures, and their names by measure_name_tuples.

    Parameters
    ----------
    measures_dict: dictionary string: numpy array
        The existing data, the new data will be appended to this.

    numpy_measures: np array
        The new measure values.

    measure_name_tuples: list of str 2-tuples
        The measure names (the 2 tokens identifying a measure in the stats file)

    Returns
    -------
    dictionary string: numpy array
        The merged data
    """
    num_measures = numpy_measures.shape[0]
    num_names = len(measure_name_tuples)
    if num_measures != num_names:
        raise ValueError("Length mismatch: expected same number of measures and names, got %d and %d." % (num_measures, num_names))
    new_measure_names = _measure_names_from_tuples(measure_name_tuples)
    for idx, new_name in enumerate(new_measure_names):
        if new_name in measures_dict:
            existing_measure_data = measures_dict[new_name]
            updated_measure_data = np.append(existing_measure_data, numpy_measures[idx])
            measures_dict[new_name] = updated_measure_data
        else:
            measures_dict[new_name] = np.array([numpy_measures[idx]])
    return measures_dict


def _stats_measures_dict(measures_dict, numpy_measures, measure_name_tuples):
    if measures_dict is None:
        return _stats_measures_to_dict(numpy_measures, measure_name_tuples)
    else:
        return _append_stats_measures_to_dict(measures_dict, numpy_measures, measure_name_tuples)


def _stats_table_dict(all_subjects_table_data, table_data_dict):
    """
    Append the data for a single subject to the data for all subjects).

    Append the data for a single subject (table_data_dict) to the data for all subjects (all_subjects_table_data). Also works if the latter is still None.

    Parameters
    ----------
    all_subjects_table_data: dict string : numpy array (or None)
        The data for all subjects. Each key is a column name, and each dict value is a 2D array of data, containing n values per subject. So the shape is (m, n) for m subjects with n values each. The number of values, n, s the number of values (rows) in the stats file table for the column identified by the dictionary key.

    table_data_dict:
        The data for a single subject. Dictionary of string (column name) to array with n values for 1 subject, so the shape is (1, n).

    Returns
    -------
    The merged data for all subjects, including the new data from table_data_dict. All the arrays are 2D arrays.
    """
    if all_subjects_table_data is None:
        return _make_dict_arrays_2D(table_data_dict)
    else:
        for key in table_data_dict:
            new_data = table_data_dict[key]   # a column array
            if key in all_subjects_table_data:
                existing_data = all_subjects_table_data[key]
                updated_data = np.vstack((existing_data, new_data))
                all_subjects_table_data[key] = updated_data
            else:
                all_subjects_table_data[key] = new_data
    return all_subjects_table_data


def _make_dict_arrays_2D(data_dict):
    for key in data_dict:
        existing_data = data_dict[key]
        new_data = np.array([existing_data])
        data_dict[key] = new_data
    return data_dict


def group_stats(subjects_list, subjects_dir, stats_file_name, stats_table_type_list=None):
    """
    Retrieve stats for a group of subjects.

    Retrieve stats for a group of subjects. The file may be for one hemisphere (files like lh.aparc.stats) or for the entire brain (like aseg.stats). This function does not care about hemispheres.

    Parameters
    ----------
    subjects_list: list of str
        List of subject identifiers (subjects in the subjects_dir).

    subjects_dir: str
        Subjects directory, as defined by the environment variable SUBJECTS_DIR for FreeSurfer.

    stats_file_name: str
        File name of the subjects file including file extension, relative to a subject's ```stats``` directory. Example: 'aseg.stats'.

    stats_table_type_list: list of numpy types, optional.
        A list defining the data types for the columns in the table contained in stats files. See the functions typelist_for_aseg_stats and typelist_for_aparc_atlas_stats for examples. If omitted, the table data will be returned as None. The measures data is unaffected.

    Returns
    -------
    all_subjects_measures_dict: dict of string to numpy 1D array.
        The data from the measure rows in the files. Each key in the dictionary is the name of a measure, and the value is the data for all subjects in a numpy float array. The array shape is (n, ) for n subjects.

    all_subjects_table_data_dict: dict of string to numpy 2D array
        The data for all table columns in the files. Each key in the dictionary is the name of a column in the stats table, and the value is the data for all rows for all subjects in a numpy 2D float array. The array shape is (n, m) for n subjects and a table with m rows.

    See also
    --------
    typelist_for_aseg_stats: pre-defined list of numpy data types for the files aseg.stats, can be used to pass stats_table_type_list
    typelist_for_aparc_atlas_stats: pre-defined list of numpy data types for the files lh.aparc.stats and rh.aparc.stats, can be used to pass stats_table_type_list
    """
    all_subjects_measures_dict = None
    all_subjects_table_data_dict = None
    for subject in subjects_list:
        stats_file = os.path.join(subjects_dir, subject, 'stats', stats_file_name)
        stats = stat(stats_file)
        print("Subject %s, reading file: %s" % (subject, stats_file))
        # Handle measures
        numpy_measures, measure_name_tuples = measures_to_numpy(stats['measures'])
        all_subjects_measures_dict = _stats_measures_dict(all_subjects_measures_dict, numpy_measures, measure_name_tuples)

        # Handle table data if possible (i.e., if stats_table_type_list was given)
        if stats_table_type_list is not None:
            table_data = stats_table_to_numpy(stats, stats_table_type_list)
            all_subjects_table_data_dict = _stats_table_dict(all_subjects_table_data_dict, table_data)
    return all_subjects_measures_dict, all_subjects_table_data_dict


def group_stats_aseg(subjects_list, subjects_dir):
    return group_stats(subjects_list, subjects_dir, 'aseg.stats', stats_table_type_list=typelist_for_aseg_stats())


def group_stats_aparc(subjects_list, subjects_dir, hemi):
    if hemi not in ('lh', 'rh'):
        raise ValueError("ERROR: hemi must be one of {'lh', 'rh'} but is '%s'." % hemi)
    return group_stats(subjects_list, subjects_dir, '%s.aparc.stats' % hemi, stats_table_type_list=typelist_for_aparc_atlas_stats())


def group_stats_aparc_a2009s(subjects_list, subjects_dir, hemi):
    if hemi not in ('lh', 'rh'):
        raise ValueError("ERROR: hemi must be one of {'lh', 'rh'} but is '%s'." % hemi)
    return group_stats(subjects_list, subjects_dir, '%s.aparc.a2009s.stats' % hemi, stats_table_type_list=typelist_for_aparc_atlas_stats())


def group_stats_aparc_DKTatlas(subjects_list, subjects_dir, hemi):
    if hemi not in ('lh', 'rh'):
        raise ValueError("ERROR: hemi must be one of {'lh', 'rh'} but is '%s'." % hemi)
    return group_stats(subjects_list, subjects_dir, '%s.aparc.DKTatlas.stats' % hemi, stats_table_type_list=typelist_for_aparc_atlas_stats())


def register_dat_matrix(file_path):
    """
    Parse the registration matrix from the given file.

    Parse the registration matrix from the given file in register.dat file format. See https://surfer.nmr.mgh.harvard.edu/fswiki/RegisterDat for the file format.

    Parameters
    ----------
    file_path: str
        Path to the file in register.dat format.

    Returns
    -------
    2D numpy array of floats
        The parsed matrix, with dimension (4, 4).
    """
    with open(file_path) as fh:
        lines = [line.rstrip('\n') for line in fh]
    return _parse_register_dat_lines(lines)


def _parse_register_dat_lines(lines):
    """
    Parse all lines of the register.dat file.

    Parse all lines of the register.dat file and return the parsed registration matrix. See https://surfer.nmr.mgh.harvard.edu/fswiki/RegisterDat for the file format.

    Parameters
    ----------
    lines: list of str
        The lines of a file in register.dat format. The file contains 8 or 9 lines.

    Returns
    -------
    2D numpy array of floats
        The parsed matrix, with dimension (4, 4).
    """
    if len(lines) == 8:     # no subject line included
        return _parse_registration_matrix(lines[3:7])
    elif len(lines) == 9:     # first line holds subject id. (This line is optional.)
        return _parse_registration_matrix(lines[4:8])
    else:
        raise ValueError("Registration matrix file has wrong line count. Expected 8 or 9 lines, got %d." % len(lines))


def _parse_registration_matrix(matrix_lines):
    """
    Parse a registration matrix.

    Parse a registration matrix from the 4 lines representing it in a register.dat file. See https://surfer.nmr.mgh.harvard.edu/fswiki/RegisterDat for the file format. This function expects only the 4 matrix lines.

    Parameters
    ----------
    matrix_lines: list of str
        The 4 matrix lines of a file in register.dat format. Each line muyt contain 4 floats, separated by spaces.

    Returns
    -------
    2D numpy array of floats
        The parsed matrix, with dimension (4, 4).
    """
    if len(matrix_lines) != 4:
        raise ValueError("Registration matrix has wrong line count. Expected exactly 4 lines, got %d." % len(matrix_lines))
    reg_matrix = np.zeros((4, 4))
    for idx, line in enumerate(matrix_lines):
        reg_matrix[idx] = np.fromstring(line, dtype=np.float_, sep=' ')
    return reg_matrix
