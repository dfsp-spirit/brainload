"""
Functions for parsing FreeSurfer brain stat files.

You can use these to read files like `subject/stats/aseg.stats`.
"""


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
            - 'table_data': string list of dimension (i, j) when there are i lines containing j values each in the table stored in the stats file.
            - 'table_meta_data': dictionary. Stores properties in key, value sub dictionaries. For simple table properties, the dictionaries are keys of the returned dictionary. The only exception is the information on the table columns (header data). This information can be found under the key 'column_info', which contains one dictionary for each column. In these dictionaries, data is stored as explained for simple table properties.

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

    Information on the individual columns can be found under the special `column_info` key:

    >>> column2_info_dict = stats['table_meta_data']['column_info']['2']
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
    return line[10:].split(', ')


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
        The result of parsing the line, as a key value pair of strings. If the line is a TableCol line, the information gets stored into a sub dictionary structure called 'column_info[index]'. Otherwise, it gets stored directly in the result dictionary.
    """
    if line.startswith('# TableCol'):
        if not 'column_info' in table_meta_data.keys():
            table_meta_data['column_info'] = {}
        line_entries = line.split(None, 4)
        column_index = line_entries[2]
        header_keyword = line_entries[3]
        header_value = line_entries[4].rstrip()
        if not column_index in table_meta_data['column_info'].keys():
            table_meta_data['column_info'][column_index] = {}
        table_meta_data['column_info'][column_index][header_keyword] = header_value
    else:
        line_entries = line.split(None, 2)
        table_meta_data[line_entries[1]] = line_entries[2].rstrip()
    return table_meta_data
