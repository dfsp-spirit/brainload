import os
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import brainload.stats as st

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(THIS_DIR, os.pardir, 'test_data')

# Respect the environment variable BRAINLOAD_TEST_DATA_DIR if it is set. If not, fall back to default.
TEST_DATA_DIR = os.getenv('BRAINLOAD_TEST_DATA_DIR', TEST_DATA_DIR)


def test_measure():
    line = '# Measure BrainSeg, BrainSegVol, Brain Segmentation Volume, 1243340.000000, mm^3'
    measure = st._measure(line)
    assert len(measure) == 5
    assert measure[0] == 'BrainSeg'
    assert measure[1] == 'BrainSegVol'
    assert measure[2] == 'Brain Segmentation Volume'
    assert measure[3] == '1243340.000000'
    assert measure[4] == 'mm^3'


def test_table_row():
    line = ' 10  15      2099     2209.1  4th-Ventricle                     25.1806    10.7109     8.0000    74.0000    66.0000'
    row10 = st._table_row(line)
    assert row10[0] == "10"
    assert row10[1] == "15"
    assert row10[2] == "2099"
    assert row10[3] == "2209.1"
    assert row10[4] == "4th-Ventricle"
    assert row10[5] == "25.1806"
    assert row10[6] == "10.7109"
    assert row10[7] == "8.0000"
    assert row10[8] == "74.0000"
    assert row10[9] == "66.0000"


def test_table_meta_data_TableCol():
    line = '# TableCol  6 ColHeader normMean'
    table_meta_data = {}
    md = st._table_meta_data(line, table_meta_data)
    assert len(md) == 1
    assert 'column_info' in md.keys()
    assert md['column_info']['6']['ColHeader'] == 'normMean'


def test_table_meta_data_other():
    line = '# NRows 45'
    table_meta_data = {}
    md = st._table_meta_data(line, table_meta_data)
    assert len(md) == 1
    assert 'NRows' in md.keys()
    assert md['NRows'] == '45'


def test_read_stats_file_aseg():
    stats_file = os.path.join(TEST_DATA_DIR, 'subject1', 'stats', 'aseg.stats')
    stats = st.read_stats_file(stats_file)
    assert len(stats) == 4
    assert 'measures' in stats
    assert 'table_data' in stats
    assert 'table_meta_data' in stats
    assert 'ignored_lines' in stats
    # check ignored lines
    ignored_lines = stats['ignored_lines']
    assert len(ignored_lines) == 25

    # check measures
    measures = stats['measures']
    assert len(measures) == 22
    expected_measures = ['BrainSeg', 'BrainSegNotVent', 'BrainSegNotVentSurf', 'VentricleChoroidVol', 'lhCortex', 'rhCortex', 'Cortex', 'lhCerebralWhiteMatter', 'rhCerebralWhiteMatter', 'CerebralWhiteMatter', 'SubCortGray', 'TotalGray', 'SupraTentorial', 'SupraTentorialNotVent', 'SupraTentorialNotVentVox', 'Mask', 'BrainSegVol-to-eTIV', 'MaskVol-to-eTIV', 'lhSurfaceHoles', 'rhSurfaceHoles', 'SurfaceHoles', 'EstimatedTotalIntraCranialVol']
    for measure_data_list in measures:
        assert len(measure_data_list) == 5
        measure_name = measure_data_list[0]
        assert measure_name in expected_measures

    # fully test a single measure
    assert measures[0][0] == 'BrainSeg'
    assert measures[0][1] == 'BrainSegVol'
    assert measures[0][2] == 'Brain Segmentation Volume'
    assert measures[0][3] == '1243340.000000'
    assert measures[0][4] == 'mm^3'

    # check table_data
    table_data = stats['table_data']
    assert len(table_data) == 45
    for index, row in enumerate(table_data):
        assert len(row) == 10
        assert int(row[0]) == index + 1 # first element should be the row index, and it starts at 1 (and lines should be in the order in which they appear in the file)
    # fully test a row
    row10 = table_data[9]
    assert row10[0] == "10"
    assert row10[1] == "15"
    assert row10[2] == "2099"
    assert row10[3] == "2209.1"
    assert row10[4] == "4th-Ventricle"
    assert row10[5] == "25.1806"
    assert row10[6] == "10.7109"
    assert row10[7] == "8.0000"
    assert row10[8] == "74.0000"
    assert row10[9] == "66.0000"

    # check table meta data
    table_meta_data = stats['table_meta_data']
    assert table_meta_data['NRows'] == "45"
    assert table_meta_data['NTableCols'] == "10"
    assert 'column_info' in table_meta_data
    assert len(table_meta_data['column_info']) == 10
    # fully test a single column info
    second_column_header = table_meta_data['column_info']['2']
    assert len(second_column_header) == 3
    for expected_key in ['ColHeader', 'FieldName', 'Units']:
        assert expected_key in second_column_header.keys()
    assert second_column_header['ColHeader'] == "SegId"
    assert second_column_header['FieldName'] == "Segmentation Id"
    assert second_column_header['Units'] == "NA"


def test_read_stats_file_lh_aparc():
    stats_file = os.path.join(TEST_DATA_DIR, 'subject1', 'stats', 'lh.aparc.stats')
    stats = st.read_stats_file(stats_file)
    assert len(stats) == 4
    assert len(stats['ignored_lines']) == 18
    assert len(stats['measures']) == 10
    assert len(stats['table_data']) == 34
    assert len(stats['table_meta_data']) == 3
    assert 'ColHeaders' in stats['table_meta_data']
    assert 'NTableCols' in stats['table_meta_data']
    assert 'column_info' in stats['table_meta_data']
    assert len(stats['table_meta_data']['column_info']) == 10


def test_read_stats_file_rh_aparc():
    stats_file = os.path.join(TEST_DATA_DIR, 'subject1', 'stats', 'rh.aparc.stats')
    stats = st.read_stats_file(stats_file)
    assert len(stats) == 4
    assert len(stats['ignored_lines']) == 18
    assert len(stats['measures']) == 10
    assert len(stats['table_data']) == 34
    assert len(stats['table_meta_data']) == 3
    assert 'ColHeaders' in stats['table_meta_data']
    assert 'NTableCols' in stats['table_meta_data']
    assert 'column_info' in stats['table_meta_data']
    assert len(stats['table_meta_data']['column_info']) == 10


def test_read_stats_file_lh_aparc_a2009s():
    stats_file = os.path.join(TEST_DATA_DIR, 'subject1', 'stats', 'lh.aparc.a2009s.stats')
    stats = st.read_stats_file(stats_file)
    assert len(stats) == 4
    assert len(stats['ignored_lines']) == 18
    assert len(stats['measures']) == 10
    assert len(stats['table_data']) == 74
    assert len(stats['table_meta_data']) == 3
    assert 'ColHeaders' in stats['table_meta_data']
    assert 'NTableCols' in stats['table_meta_data']
    assert 'column_info' in stats['table_meta_data']
    assert len(stats['table_meta_data']['column_info']) == 10


def test_read_stats_file_rh_aparc_a2009s():
    stats_file = os.path.join(TEST_DATA_DIR, 'subject1', 'stats', 'lh.aparc.a2009s.stats')
    stats = st.read_stats_file(stats_file)
    assert len(stats) == 4
    assert len(stats['ignored_lines']) == 18
    assert len(stats['measures']) == 10
    assert len(stats['table_data']) == 74
    assert len(stats['table_meta_data']) == 3
    assert 'ColHeaders' in stats['table_meta_data']
    assert 'NTableCols' in stats['table_meta_data']
    assert 'column_info' in stats['table_meta_data']
    assert len(stats['table_meta_data']['column_info']) == 10


def test_read_stats_file_lh_aparc_DKTatlas():
    stats_file = os.path.join(TEST_DATA_DIR, 'subject1', 'stats', 'lh.aparc.DKTatlas.stats')
    stats = st.read_stats_file(stats_file)
    assert len(stats) == 4
    assert len(stats['ignored_lines']) == 18
    assert len(stats['measures']) == 10
    assert len(stats['table_data']) == 31
    assert len(stats['table_meta_data']) == 3
    assert 'ColHeaders' in stats['table_meta_data']
    assert 'NTableCols' in stats['table_meta_data']
    assert 'column_info' in stats['table_meta_data']
    assert len(stats['table_meta_data']['column_info']) == 10


def test_read_stats_file_rh_aparc_DKTatlas():
    stats_file = os.path.join(TEST_DATA_DIR, 'subject1', 'stats', 'rh.aparc.DKTatlas.stats')
    stats = st.read_stats_file(stats_file)
    assert len(stats) == 4
    assert len(stats['ignored_lines']) == 18
    assert len(stats['measures']) == 10
    assert len(stats['table_data']) == 31
    assert len(stats['table_meta_data']) == 3
    assert 'ColHeaders' in stats['table_meta_data']
    assert 'NTableCols' in stats['table_meta_data']
    assert 'column_info' in stats['table_meta_data']
    assert len(stats['table_meta_data']['column_info']) == 10
