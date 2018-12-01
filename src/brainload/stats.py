"""
Functions for parsing FreeSurfer brain stat files.

You can use these to read files like `subject/stats/aseg.stats`.
"""


def read_stats_file(file_name):
    with open(file_name) as fh:
        lines = [line.rstrip('\n') for line in fh]
    return parse_stats_lines(lines)


def parse_stats_lines(lines):
    ignored_lines = []
    measures = {}
    table_rows = []
    table_meta_data = {}
    for line in lines:
        if line.startswith('#'):
            if line.startswith('# Measure '):
                _measure(line, measures)
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


def _measure(line, measures):
    line_data = line[10:].split(', ')
    measure = line_data.pop(0)
    measures[measure] = line_data


def _table_row(line):
    return line.split()


def _table_meta_data(line, table_meta_data):
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
