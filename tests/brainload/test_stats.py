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
    measures = {}
    st._measure(line, measures)
    assert len(measures) == 1
    assert 'BrainSeg' in measures.keys()
    assert measures['BrainSeg'][0] == 'BrainSegVol'
    assert measures['BrainSeg'][1] == 'Brain Segmentation Volume'
    assert measures['BrainSeg'][2] == '1243340.000000'
    assert measures['BrainSeg'][3] == 'mm^3'


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


def test_read_stats_file():
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
    for measure_data_list in measures.values():
        assert len(measure_data_list) == 4
    for expected_measure in ['BrainSeg', 'BrainSegNotVent', 'BrainSegNotVentSurf', 'VentricleChoroidVol', 'lhCortex', 'rhCortex', 'Cortex', 'lhCerebralWhiteMatter', 'rhCerebralWhiteMatter', 'CerebralWhiteMatter', 'SubCortGray', 'TotalGray', 'SupraTentorial', 'SupraTentorialNotVent', 'SupraTentorialNotVentVox', 'Mask', 'BrainSegVol-to-eTIV', 'MaskVol-to-eTIV', 'lhSurfaceHoles', 'rhSurfaceHoles', 'SurfaceHoles', 'EstimatedTotalIntraCranialVol']:
        assert expected_measure in measures.keys()
    # fully test a single measure
    assert measures['BrainSeg'][0] == 'BrainSegVol'
    assert measures['BrainSeg'][1] == 'Brain Segmentation Volume'
    assert measures['BrainSeg'][2] == '1243340.000000'
    assert measures['BrainSeg'][3] == 'mm^3'

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
