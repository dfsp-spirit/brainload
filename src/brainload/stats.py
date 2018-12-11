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
    dictionary of string, numpy array
        Each key is a column name, and each value is a numpy column array containing the typed data.
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
