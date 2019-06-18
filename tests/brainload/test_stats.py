import os
import pytest
import warnings
import brainload as bl
import brainload.stats as st
import numpy as np
from numpy.testing import assert_allclose

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
    assert 'column_info_' in md.keys()
    assert md['column_info_']['6']['ColHeader'] == 'normMean'


def test_table_meta_data_other():
    line = '# NRows 45'
    table_meta_data = {}
    md = st._table_meta_data(line, table_meta_data)
    assert len(md) == 1
    assert 'NRows' in md.keys()
    assert md['NRows'] == '45'


def test_read_stats_file_aseg():
    stats_file = os.path.join(TEST_DATA_DIR, 'subject1', 'stats', 'aseg.stats')
    stats = st.stat(stats_file)
    assert len(stats) == 5
    assert 'measures' in stats
    assert 'table_data' in stats
    assert 'table_meta_data' in stats
    assert 'ignored_lines' in stats
    assert 'table_column_headers' in stats
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
    assert 'column_info_' in table_meta_data
    assert len(table_meta_data['column_info_']) == 10
    # fully test a single column info
    second_column_header = table_meta_data['column_info_']['2']
    assert len(second_column_header) == 3
    for expected_key in ['ColHeader', 'FieldName', 'Units']:
        assert expected_key in second_column_header.keys()
    assert second_column_header['ColHeader'] == "SegId"
    assert second_column_header['FieldName'] == "Segmentation Id"
    assert second_column_header['Units'] == "NA"


def test_read_stats_file_lh_aparc():
    stats_file = os.path.join(TEST_DATA_DIR, 'subject1', 'stats', 'lh.aparc.stats')
    stats = st.stat(stats_file)
    assert len(stats) == 5
    assert len(stats['ignored_lines']) == 18
    assert len(stats['measures']) == 10
    assert len(stats['table_data']) == 34
    assert len(stats['table_meta_data']) == 3
    assert len(stats['table_column_headers']) == 10
    assert 'ColHeaders' in stats['table_meta_data']
    assert 'NTableCols' in stats['table_meta_data']
    assert 'column_info_' in stats['table_meta_data']
    assert len(stats['table_meta_data']['column_info_']) == 10


def test_read_stats_file_rh_aparc():
    stats_file = os.path.join(TEST_DATA_DIR, 'subject1', 'stats', 'rh.aparc.stats')
    stats = st.stat(stats_file)
    assert len(stats) == 5
    assert len(stats['ignored_lines']) == 18
    assert len(stats['measures']) == 10
    assert len(stats['table_data']) == 34
    assert len(stats['table_meta_data']) == 3
    assert len(stats['table_column_headers']) == 10
    assert 'ColHeaders' in stats['table_meta_data']
    assert 'NTableCols' in stats['table_meta_data']
    assert 'column_info_' in stats['table_meta_data']
    assert len(stats['table_meta_data']['column_info_']) == 10


def test_read_stats_file_lh_aparc_a2009s():
    stats_file = os.path.join(TEST_DATA_DIR, 'subject1', 'stats', 'lh.aparc.a2009s.stats')
    stats = st.stat(stats_file)
    assert len(stats) == 5
    assert len(stats['ignored_lines']) == 18
    assert len(stats['measures']) == 10
    assert len(stats['table_data']) == 74
    assert len(stats['table_meta_data']) == 3
    assert len(stats['table_column_headers']) == 10
    assert 'ColHeaders' in stats['table_meta_data']
    assert 'NTableCols' in stats['table_meta_data']
    assert 'column_info_' in stats['table_meta_data']
    assert len(stats['table_meta_data']['column_info_']) == 10


def test_read_stats_file_rh_aparc_a2009s():
    stats_file = os.path.join(TEST_DATA_DIR, 'subject1', 'stats', 'lh.aparc.a2009s.stats')
    stats = st.stat(stats_file)
    assert len(stats) == 5
    assert len(stats['ignored_lines']) == 18
    assert len(stats['measures']) == 10
    assert len(stats['table_data']) == 74
    assert len(stats['table_meta_data']) == 3
    assert len(stats['table_column_headers']) == 10
    assert 'ColHeaders' in stats['table_meta_data']
    assert 'NTableCols' in stats['table_meta_data']
    assert 'column_info_' in stats['table_meta_data']
    assert len(stats['table_meta_data']['column_info_']) == 10


def test_read_stats_file_lh_aparc_DKTatlas():
    stats_file = os.path.join(TEST_DATA_DIR, 'subject1', 'stats', 'lh.aparc.DKTatlas.stats')
    stats = st.stat(stats_file)
    assert len(stats) == 5
    assert len(stats['ignored_lines']) == 18
    assert len(stats['measures']) == 10
    assert len(stats['table_data']) == 31
    assert len(stats['table_meta_data']) == 3
    assert len(stats['table_column_headers']) == 10
    assert 'ColHeaders' in stats['table_meta_data']
    assert 'NTableCols' in stats['table_meta_data']
    assert 'column_info_' in stats['table_meta_data']
    assert len(stats['table_meta_data']['column_info_']) == 10


def test_read_stats_file_rh_aparc_DKTatlas():
    stats_file = os.path.join(TEST_DATA_DIR, 'subject1', 'stats', 'rh.aparc.DKTatlas.stats')
    stats = st.stat(stats_file)
    assert len(stats) == 5
    assert len(stats['ignored_lines']) == 18
    assert len(stats['measures']) == 10
    assert len(stats['table_data']) == 31
    assert len(stats['table_meta_data']) == 3
    assert len(stats['table_column_headers']) == 10
    assert stats['table_column_headers'] == ['StructName', 'NumVert', 'SurfArea', 'GrayVol', 'ThickAvg', 'ThickStd', 'MeanCurv', 'GausCurv', 'FoldInd', 'CurvInd']
    assert 'ColHeaders' in stats['table_meta_data']
    assert 'NTableCols' in stats['table_meta_data']
    assert 'column_info_' in stats['table_meta_data']
    assert len(stats['table_meta_data']['column_info_']) == 10


def test_sorted_header_indices():
    stats_file = os.path.join(TEST_DATA_DIR, 'subject1', 'stats', 'lh.aparc.DKTatlas.stats')
    stats = bl.stat(stats_file)
    header_indices = st._sorted_header_indices(stats['table_meta_data'])
    assert len(header_indices) == 10
    assert header_indices[0] == '1'
    assert header_indices[5] == '6'
    assert header_indices[9] == '10'


def test_header_line():
    stats_file = os.path.join(TEST_DATA_DIR, 'subject1', 'stats', 'lh.aparc.DKTatlas.stats')
    stats = bl.stat(stats_file)
    hdr_string = st._header_line(stats['table_meta_data'])
    assert hdr_string == '\t'.join(['StructName', 'NumVert', 'SurfArea', 'GrayVol', 'ThickAvg', 'ThickStd', 'MeanCurv', 'GausCurv', 'FoldInd', 'CurvInd'])


def test_header_line_missing_col_headers_warns():
    stats_file = os.path.join(TEST_DATA_DIR, 'subject1', 'stats', 'lh.aparc.DKTatlas.stats')
    stats = bl.stat(stats_file)
    del stats['table_meta_data']['ColHeaders']
    with pytest.warns(UserWarning, match='Stats data is missing some header data'):
        hdr_string = st._header_line(stats['table_meta_data'])
    assert hdr_string == '\t'.join(['StructName', 'NumVert', 'SurfArea', 'GrayVol', 'ThickAvg', 'ThickStd', 'MeanCurv', 'GausCurv', 'FoldInd', 'CurvInd'])


def test_header_line_missing_column_info_warns():
    stats_file = os.path.join(TEST_DATA_DIR, 'subject1', 'stats', 'lh.aparc.DKTatlas.stats')
    stats = bl.stat(stats_file)
    del stats['table_meta_data']['column_info_']
    with pytest.warns(UserWarning, match='Stats data is missing some header data'):
        hdr_string = st._header_line(stats['table_meta_data'])
    assert hdr_string == '\t'.join(['StructName', 'NumVert', 'SurfArea', 'GrayVol', 'ThickAvg', 'ThickStd', 'MeanCurv', 'GausCurv', 'FoldInd', 'CurvInd'])


def test_header_line_raises_on_both_missing():
    stats_file = os.path.join(TEST_DATA_DIR, 'subject1', 'stats', 'lh.aparc.DKTatlas.stats')
    stats = bl.stat(stats_file)
    del stats['table_meta_data']['column_info_']
    del stats['table_meta_data']['ColHeaders']
    with pytest.raises(ValueError) as exc_info:
        hdr_string = st._header_line(stats['table_meta_data'])
    assert 'Could not determine header line' in str(exc_info.value)


def test_header_line_inconsistent_warns():
    stats_file = os.path.join(TEST_DATA_DIR, 'subject1', 'stats', 'lh.aparc.DKTatlas.stats')
    stats = bl.stat(stats_file)
    stats['table_meta_data']['ColHeaders'] = 'StructName NumVertBROKEN SurfArea GrayVol ThickAvg ThickStd MeanCurv GausCurv FoldInd CurvInd' # mess with data, add 'BROKEN' to 2nd header column
    with pytest.warns(UserWarning, match='Stats data regarding table header is inconsistent between ColHeaders and TableCol->ColHeader entries'):
        hdr_string = st._header_line(stats['table_meta_data'])
    assert hdr_string == '\t'.join(['StructName', 'NumVert', 'SurfArea', 'GrayVol', 'ThickAvg', 'ThickStd', 'MeanCurv', 'GausCurv', 'FoldInd', 'CurvInd'])


def test_header_line_does_not_raise_on_broken_col_headers():
    stats_file = os.path.join(TEST_DATA_DIR, 'subject1', 'stats', 'lh.aparc.DKTatlas.stats')
    stats = bl.stat(stats_file)
    stats['table_meta_data']['ColHeaders'] = 1  # ruin data: cannot split an int
    with pytest.warns(UserWarning, match='Stats data is missing some header data'):
        hdr_string = st._header_line(stats['table_meta_data'])
    assert hdr_string == '\t'.join(['StructName', 'NumVert', 'SurfArea', 'GrayVol', 'ThickAvg', 'ThickStd', 'MeanCurv', 'GausCurv', 'FoldInd', 'CurvInd'])


def test_stats_table_to_numpy_aparc_dktatlas_stats():
    stats_file = os.path.join(TEST_DATA_DIR, 'subject1', 'stats', 'lh.aparc.DKTatlas.stats')
    stats = bl.stat(stats_file)
    types = st.typelist_for_aparc_atlas_stats()
    numpy_data = st.stats_table_to_numpy(stats, types)
    assert len(numpy_data) == 10        # 10 columns
    assert 'StructName' in numpy_data
    assert 'NumVert' in numpy_data
    for column_name in numpy_data:
        assert len(numpy_data[column_name]) == 31


def test_stats_table_to_numpy_aseg_stats():
    stats_file = os.path.join(TEST_DATA_DIR, 'subject1', 'stats', 'aseg.stats')
    stats = bl.stat(stats_file)
    types = st.typelist_for_aseg_stats()
    numpy_data = st.stats_table_to_numpy(stats, types)
    assert len(numpy_data) == 10        # 10 columns
    assert 'Index' in numpy_data
    assert 'SegId' in numpy_data
    for column_name in numpy_data:
        assert len(numpy_data[column_name]) == 45       # 45 rows


def test_stats_table_to_numpy_raises_on_invalied_type_count():
    stats_file = os.path.join(TEST_DATA_DIR, 'subject1', 'stats', 'aseg.stats')
    stats = bl.stat(stats_file)
    types = [np.string_, np.string_]        # wrong number of types!
    with pytest.raises(ValueError) as exc_info:
        numpy_data = st.stats_table_to_numpy(stats, types)
    assert 'Length of type_list' in str(exc_info.value)
    assert 'must match number of' in str(exc_info.value)


def test_measures_to_numpy():
    stats_file = os.path.join(TEST_DATA_DIR, 'subject1', 'stats', 'aseg.stats')
    stats = bl.stat(stats_file)
    numpy_measures, measure_names = st.measures_to_numpy(stats['measures'])
    assert numpy_measures.shape == (22, )
    assert numpy_measures.shape[0] == len(stats['measures'])
    assert measure_names[0] == ('BrainSeg', 'BrainSegVol')
    assert numpy_measures[0] == pytest.approx(1243340.0, 0.01)
    assert np.dtype(numpy_measures[0]) == np.float_


def test_measures_to_numpy_with_custom_dtype():
    stats_file = os.path.join(TEST_DATA_DIR, 'subject1', 'stats', 'aseg.stats')
    stats = bl.stat(stats_file)
    numpy_measures, measure_names = st.measures_to_numpy(stats['measures'], dtype=np.float32)
    assert numpy_measures.shape == (22, )
    assert numpy_measures.shape[0] == len(stats['measures'])
    assert measure_names[0] == ('BrainSeg', 'BrainSegVol')
    assert numpy_measures[0] == pytest.approx(1243340.0, 0.01)
    assert np.dtype(numpy_measures[0]) == np.float32


def test_measures_to_numpy_subset():
    stats_file = os.path.join(TEST_DATA_DIR, 'subject1', 'stats', 'aseg.stats')
    stats = bl.stat(stats_file)
    requested_measures = [('BrainSeg', 'BrainSegVol'), ('SurfaceHoles', 'SurfaceHoles')]
    numpy_measures, measure_names = st.measures_to_numpy(stats['measures'], requested_measures=requested_measures)
    assert numpy_measures.shape == (2, )
    assert numpy_measures.shape[0] == len(requested_measures)
    assert measure_names[0] == ('BrainSeg', 'BrainSegVol')
    assert measure_names[1] == ('SurfaceHoles', 'SurfaceHoles')
    assert numpy_measures[0] == pytest.approx(1243340.0, 0.01)
    assert numpy_measures[1] == pytest.approx(29, 0.01)


def test_measure_names_from_tuples():
    measure_name_tuples = [('Cortex', 'NumVerts'), ('Cortex', 'NumFaces')]
    names = st._measure_names_from_tuples(measure_name_tuples)
    assert names == ['Cortex,NumVerts', 'Cortex,NumFaces']


def test_stats_measures_to_dict():
    numpy_measures = np.array([0.74, 0.34])
    numpy_measures = np.transpose(numpy_measures)
    assert numpy_measures.shape == (2, )
    measure_name_tuples = [('Cortex', 'NumVerts'), ('Cortex', 'NumFaces')]
    measures_dict = st._stats_measures_to_dict(numpy_measures, measure_name_tuples)
    assert len(measures_dict) == 2
    assert 'Cortex,NumVerts' in measures_dict
    assert 'Cortex,NumFaces' in measures_dict
    num_verts = measures_dict['Cortex,NumVerts']
    num_faces = measures_dict['Cortex,NumFaces']
    assert num_verts.shape == (1, )
    assert num_faces.shape == (1, )
    assert num_verts[0] == pytest.approx(0.74, 0.01)
    assert num_faces[0] == pytest.approx(0.34, 0.01)


def test_stats_table_dict_alldata_empty():
    new_subject_data = {'NVoxels': np.array([65535])}
    merged = st._stats_table_dict(None, new_subject_data)
    assert merged == new_subject_data


def test_stats_table_dict_alldata_contains_data_already():
    existing_all_subjects_data = {'NVoxels': np.array([[1111, 2222], [3333, 4444]])} # data for 2 subjects, originating from 2 rows in the tables each (i.e., the table has only 2 rows.)
    new_subject_data = {'NVoxels': np.array([65535, 33333]), 'NewStat': np.array([5555, 6666])}
    merged = st._stats_table_dict(existing_all_subjects_data, new_subject_data)
    assert len(merged) == 2
    assert 'NVoxels' in merged
    assert 'NewStat' in merged
    assert np.array_equal(np.array([[1111, 2222], [3333, 4444], [65535, 33333]]), merged['NVoxels'])
    assert np.array_equal(np.array([5555, 6666]), merged['NewStat'])


def test_append_stats_measures_to_dict():
    measures_dict = { 'BrainSeg,BrainSegVol': np.array([34234.1, 55555.5]), 'BrainSegNotVent,BrainSegVolNotVent': np.array([343.3, 2355.6]) }
    numpy_measures = np.array([44.4, 66.6, 77.7])
    measure_name_tuples = [('BrainSeg', 'BrainSegVol'), ('BrainSegNotVent', 'BrainSegVolNotVent'), ('New', 'Measure')]
    result_measures_dict = st._append_stats_measures_to_dict(measures_dict, numpy_measures, measure_name_tuples)
    assert len(result_measures_dict) == 3
    assert 'BrainSeg,BrainSegVol' in result_measures_dict
    assert 'BrainSegNotVent,BrainSegVolNotVent' in result_measures_dict
    assert 'New,Measure' in result_measures_dict
    assert len(result_measures_dict['BrainSeg,BrainSegVol']) == 3
    assert len(result_measures_dict['BrainSegNotVent,BrainSegVolNotVent']) == 3
    assert len(result_measures_dict['New,Measure']) == 1


def test_append_stats_measures_to_dict_raises_on_unmachted_lengths():
    measures_dict = { 'BrainSeg,BrainSegVol': np.array([34234.1, 55555.5]), 'BrainSegNotVent,BrainSegVolNotVent': np.array([343.3, 2355.6]) }
    numpy_measures = np.array([44.4, 66.6, 77.7])  # 3 values....
    measure_name_tuples = [('BrainSeg', 'BrainSegVol'), ('BrainSegNotVent', 'BrainSegVolNotVent')]    # ... but only 2 names for them.
    with pytest.raises(ValueError) as exc_info:
        result_measures_dict = st._append_stats_measures_to_dict(measures_dict, numpy_measures, measure_name_tuples)
    assert 'Length mismatch: expected same number of measures and names' in str(exc_info.value)


def test_group_stats_measures_only_asegstats():
    expected_stats_file = os.path.join(TEST_DATA_DIR, 'subject2', 'stats', 'aseg.stats')
    if not os.path.isfile(expected_stats_file):
        pytest.skip("Test data missing: stats file '%s' does not exist. You can get all test data by running './develop/get_test_data_all.bash' in the repo root." % expected_stats_file)
    subjects_list = ['subject1', 'subject2']
    all_subjects_measures_dict, all_subjects_table_data_dict = st.group_stats(subjects_list, TEST_DATA_DIR, 'aseg.stats')
    assert all_subjects_table_data_dict is None
    assert len(all_subjects_measures_dict) == 22
    assert 'BrainSeg,BrainSegVol' in all_subjects_measures_dict
    brainsegvol_data = all_subjects_measures_dict['BrainSeg,BrainSegVol']
    assert brainsegvol_data.shape == (2, )
    assert brainsegvol_data[0] == pytest.approx(1243340.0, 0.01)
    assert brainsegvol_data[1] == pytest.approx(1243340.0, 0.01)    # test data for subject2 is copied from subject1


def test_group_stats_measures_and_table_asegstats():
    expected_stats_file = os.path.join(TEST_DATA_DIR, 'subject2', 'stats', 'aseg.stats')
    if not os.path.isfile(expected_stats_file):
        pytest.skip("Test data missing: stats file '%s' does not exist. You can get all test data by running './develop/get_test_data_all.bash' in the repo root." % expected_stats_file)
    subjects_list = ['subject1', 'subject2']
    stats_table_type_list = st.typelist_for_aseg_stats()
    all_subjects_measures_dict, all_subjects_table_data_dict = st.group_stats(subjects_list, TEST_DATA_DIR, 'aseg.stats', stats_table_type_list=stats_table_type_list)
    assert len(all_subjects_measures_dict) == 22
    assert 'BrainSeg,BrainSegVol' in all_subjects_measures_dict
    brainsegvol_data = all_subjects_measures_dict['BrainSeg,BrainSegVol']
    assert brainsegvol_data.shape == (2, )
    assert brainsegvol_data[0] == pytest.approx(1243340.0, 0.01)
    assert brainsegvol_data[1] == pytest.approx(1243340.0, 0.01)    # test data for subject2 is copied from subject1
    assert all_subjects_table_data_dict is not None
    assert len(all_subjects_table_data_dict) == 10
    expected_table_column_names_aseg = ['Index', 'SegId', 'NVoxels', 'Volume_mm3', 'StructName', 'normMean', 'normStdDev', 'normMin', 'normMax', 'normRange']
    for name in expected_table_column_names_aseg:
        assert name in all_subjects_table_data_dict
        column_data = all_subjects_table_data_dict[name]
        assert column_data.shape == (2, 45)  # 2 subjects, each has a table in aseg.stats with 45 rows


def test_group_stats_aseg():
    expected_stats_file = os.path.join(TEST_DATA_DIR, 'subject2', 'stats', 'aseg.stats')
    if not os.path.isfile(expected_stats_file):
        pytest.skip("Test data missing: stats file '%s' does not exist. You can get all test data by running './develop/get_test_data_all.bash' in the repo root." % expected_stats_file)
    subjects_list = ['subject1', 'subject2']
    all_subjects_measures_dict, all_subjects_table_data_dict = st.group_stats_aseg(subjects_list, TEST_DATA_DIR)
    assert len(all_subjects_measures_dict) == 22
    assert 'BrainSeg,BrainSegVol' in all_subjects_measures_dict
    brainsegvol_data = all_subjects_measures_dict['BrainSeg,BrainSegVol']
    assert brainsegvol_data.shape == (2, )
    assert brainsegvol_data[0] == pytest.approx(1243340.0, 0.01)
    assert brainsegvol_data[1] == pytest.approx(1243340.0, 0.01)    # test data for subject2 is copied from subject1
    assert all_subjects_table_data_dict is not None
    assert len(all_subjects_table_data_dict) == 10
    expected_table_column_names_aseg = ['Index', 'SegId', 'NVoxels', 'Volume_mm3', 'StructName', 'normMean', 'normStdDev', 'normMin', 'normMax', 'normRange']
    for name in expected_table_column_names_aseg:
        assert name in all_subjects_table_data_dict
        column_data = all_subjects_table_data_dict[name]
        assert column_data.shape == (2, 45)  # 2 subjects, each has a table in aseg.stats with 45 rows


def test_group_stats_aparc(capsys):
    expected_stats_file = os.path.join(TEST_DATA_DIR, 'subject2', 'stats', 'lh.aparc.stats')
    if not os.path.isfile(expected_stats_file):
        pytest.skip("Test data missing: stats file '%s' does not exist. You can get all test data by running './develop/get_test_data_all.bash' in the repo root." % expected_stats_file)
    subjects_list = ['subject1', 'subject2']
    with capsys.disabled():
        all_subjects_measures_dict, all_subjects_table_data_dict = st.group_stats_aparc(subjects_list, TEST_DATA_DIR, 'lh')
    assert len(all_subjects_measures_dict) == 10
    assert 'BrainSeg,BrainSegVol' in all_subjects_measures_dict
    brainsegvol_data = all_subjects_measures_dict['BrainSeg,BrainSegVol']
    assert brainsegvol_data.shape == (2, )
    assert brainsegvol_data[0] == pytest.approx(1243340.0, 0.01)
    assert brainsegvol_data[1] == pytest.approx(1243340.0, 0.01)    # test data for subject2 is copied from subject1
    assert all_subjects_table_data_dict is not None
    assert len(all_subjects_table_data_dict) == 10
    expected_table_column_names_aparc = ['StructName', 'NumVert', 'SurfArea', 'GrayVol', 'ThickAvg', 'ThickStd', 'MeanCurv', 'GausCurv', 'FoldInd', 'CurvInd']
    for name in expected_table_column_names_aparc:
        assert name in all_subjects_table_data_dict
        column_data = all_subjects_table_data_dict[name]
        assert column_data.shape == (2, 34)  # 2 subjects, each has a table in lh.aparc.stats with 34 rows


def test_group_stats_aparc_a2009s(capsys):
    expected_stats_file = os.path.join(TEST_DATA_DIR, 'subject2', 'stats', 'lh.aparc.a2009s.stats')
    if not os.path.isfile(expected_stats_file):
        pytest.skip("Test data missing: stats file '%s' does not exist. You can get all test data by running './develop/get_test_data_all.bash' in the repo root." % expected_stats_file)
    subjects_list = ['subject1', 'subject2']
    with capsys.disabled():
        all_subjects_measures_dict, all_subjects_table_data_dict = st.group_stats_aparc_a2009s(subjects_list, TEST_DATA_DIR, 'lh')
    assert len(all_subjects_measures_dict) == 10
    assert 'BrainSeg,BrainSegVol' in all_subjects_measures_dict
    brainsegvol_data = all_subjects_measures_dict['BrainSeg,BrainSegVol']
    assert brainsegvol_data.shape == (2, )
    assert brainsegvol_data[0] == pytest.approx(1243340.0, 0.01)
    assert brainsegvol_data[1] == pytest.approx(1243340.0, 0.01)    # test data for subject2 is copied from subject1
    assert all_subjects_table_data_dict is not None
    assert len(all_subjects_table_data_dict) == 10
    expected_table_column_names_aparc_a2009s = ['StructName', 'NumVert', 'SurfArea', 'GrayVol', 'ThickAvg', 'ThickStd', 'MeanCurv', 'GausCurv', 'FoldInd', 'CurvInd']
    for name in expected_table_column_names_aparc_a2009s:
        assert name in all_subjects_table_data_dict
        column_data = all_subjects_table_data_dict[name]
        assert column_data.shape == (2, 74)  # 2 subjects, each has a table in lh.aparc.a2009s.stats with 74 rows


def test_group_stats_aparc_DKTatlas(capsys):
    expected_stats_file = os.path.join(TEST_DATA_DIR, 'subject2', 'stats', 'lh.aparc.DKTatlas.stats')
    if not os.path.isfile(expected_stats_file):
        pytest.skip("Test data missing: stats file '%s' does not exist. You can get all test data by running './develop/get_test_data_all.bash' in the repo root." % expected_stats_file)
    subjects_list = ['subject1', 'subject2']
    with capsys.disabled():
        all_subjects_measures_dict, all_subjects_table_data_dict = st.group_stats_aparc_DKTatlas(subjects_list, TEST_DATA_DIR, 'lh')
    assert len(all_subjects_measures_dict) == 10
    assert 'BrainSeg,BrainSegVol' in all_subjects_measures_dict
    brainsegvol_data = all_subjects_measures_dict['BrainSeg,BrainSegVol']
    assert brainsegvol_data.shape == (2, )
    assert brainsegvol_data[0] == pytest.approx(1243340.0, 0.01)
    assert brainsegvol_data[1] == pytest.approx(1243340.0, 0.01)    # test data for subject2 is copied from subject1
    assert all_subjects_table_data_dict is not None
    assert len(all_subjects_table_data_dict) == 10
    expected_table_column_names_aparc_DKTatlas = ['StructName', 'NumVert', 'SurfArea', 'GrayVol', 'ThickAvg', 'ThickStd', 'MeanCurv', 'GausCurv', 'FoldInd', 'CurvInd']
    for name in expected_table_column_names_aparc_DKTatlas:
        assert name in all_subjects_table_data_dict
        column_data = all_subjects_table_data_dict[name]
        assert column_data.shape == (2, 31)  # 2 subjects, each has a table in lh.aparc.DKTatlas.stats with 31 rows



def test_group_stats_aparc_raises_on_invalid_hemi():
    subjects_list = ['subject1', 'subject2']
    with pytest.raises(ValueError) as exc_info:
        all_subjects_measures_dict, all_subjects_table_data_dict = st.group_stats_aparc(subjects_list, TEST_DATA_DIR, 'invalid_hemi')
    assert 'hemi must be one of' in str(exc_info.value)
    assert 'invalid_hemi' in str(exc_info.value)


def test_group_stats_aparc_a2009s_raises_on_invalid_hemi():
    subjects_list = ['subject1', 'subject2']
    with pytest.raises(ValueError) as exc_info:
        all_subjects_measures_dict, all_subjects_table_data_dict = st.group_stats_aparc_a2009s(subjects_list, TEST_DATA_DIR, 'invalid_hemi')
    assert 'hemi must be one of' in str(exc_info.value)
    assert 'invalid_hemi' in str(exc_info.value)


def test_group_stats_aparc_DKTatlas_raises_on_invalid_hemi():
    subjects_list = ['subject1', 'subject2']
    with pytest.raises(ValueError) as exc_info:
        all_subjects_measures_dict, all_subjects_table_data_dict = st.group_stats_aparc_DKTatlas(subjects_list, TEST_DATA_DIR, 'invalid_hemi')
    assert 'hemi must be one of' in str(exc_info.value)
    assert 'invalid_hemi' in str(exc_info.value)

def test_parse_register_dat_lines_correct_linecount_with_subject():
    register_dat_contents = """fsaverage
1.000000
1.000000
0.150000
9.975314e-01 -7.324822e-03 1.760415e-02 9.570923e-01
-1.296475e-02 -9.262221e-03 9.970638e-01 -1.781596e+01
-1.459537e-02 -1.000945e+00 2.444772e-03 -1.854964e+01
0 0 0 1
tkregister"""
    registration_matrix = st._parse_register_dat_lines(register_dat_contents.splitlines())
    assert registration_matrix.shape == (4, 4)

def test_parse_register_dat_lines_correct_linecount_without_subject():
    register_dat_contents = """1.000000
1.000000
0.150000
9.975314e-01 -7.324822e-03 1.760415e-02 9.570923e-01
-1.296475e-02 -9.262221e-03 9.970638e-01 -1.781596e+01
-1.459537e-02 -1.000945e+00 2.444772e-03 -1.854964e+01
0 0 0 1
tkregister"""
    registration_matrix = st._parse_register_dat_lines(register_dat_contents.splitlines())
    assert registration_matrix.shape == (4, 4)
    expected = np.array([[0.9975314, -0.007324822, 0.01760415, 0.9570923], [-0.01296475, -0.009262221, 0.9970638, -17.81596], [-0.01459537, -1.000945, 0.002444772, -18.54964], [0., 0., 0., 1.]])
    assert_allclose(registration_matrix, expected)


def test_parse_register_dat_lines_raises_on_incorrect_linecount():
    register_dat_contents = """1.000000
1.000000
0.150000"""
    with pytest.raises(ValueError) as exc_info:
        registration_matrix = st._parse_register_dat_lines(register_dat_contents.splitlines())
    assert 'Registration matrix file has wrong line count' in str(exc_info.value)


def test_parse_registration_matrix_raises_on_incorrect_linecount():
    matrix_contents = """1.000000
1.000000
0.150000"""
    with pytest.raises(ValueError) as exc_info:
        registration_matrix = st._parse_registration_matrix(matrix_contents.splitlines())
    assert 'Registration matrix has wrong line count' in str(exc_info.value)


def test_extract_table_data_indices_where():
    expected_stats_file = os.path.join(TEST_DATA_DIR, 'subject2', 'stats', 'aseg.stats')
    if not os.path.isfile(expected_stats_file):
        pytest.skip("Test data missing: stats file '%s' does not exist. You can get all test data by running './develop/get_test_data_all.bash' in the repo root." % expected_stats_file)
    subjects_list = ['subject1', 'subject2']
    _, all_subjects_table_data_dict = st.group_stats_aseg(subjects_list, TEST_DATA_DIR)
    struct_name_data = all_subjects_table_data_dict['StructName']
    assert struct_name_data.shape == (2, 45)
    assert struct_name_data[0][12] == b'Left-Amygdala'
    row_indices = st.extract_table_data_indices_where('StructName', b'Left-Amygdala', all_subjects_table_data_dict)
    assert len(row_indices) == 1
    assert row_indices[0] == 12        # see aseg.stats table: amygdala is row 12


def test_extract_field_from_table_data():
    expected_stats_file = os.path.join(TEST_DATA_DIR, 'subject2', 'stats', 'aseg.stats')
    if not os.path.isfile(expected_stats_file):
        pytest.skip("Test data missing: stats file '%s' does not exist. You can get all test data by running './develop/get_test_data_all.bash' in the repo root." % expected_stats_file)
    subjects_list = ['subject1', 'subject2']
    _, all_subjects_table_data_dict = st.group_stats_aseg(subjects_list, TEST_DATA_DIR)
    row_index_amygdala = st.extract_table_data_indices_where('StructName', b'Left-Amygdala', all_subjects_table_data_dict)
    assert not isinstance(row_index_amygdala, tuple)    # must be np.array
    assert row_index_amygdala.shape == (1, )
    amygdala_column_all_subjects = st.extract_field_from_table_data('Volume_mm3', row_index_amygdala, all_subjects_table_data_dict)
    assert amygdala_column_all_subjects.shape == (2, )


def test_extract_column_from_table_data():
    expected_stats_file = os.path.join(TEST_DATA_DIR, 'subject2', 'stats', 'aseg.stats')
    if not os.path.isfile(expected_stats_file):
        pytest.skip("Test data missing: stats file '%s' does not exist. You can get all test data by running './develop/get_test_data_all.bash' in the repo root." % expected_stats_file)
    subjects_list = ['subject1', 'subject2']
    _, all_subjects_table_data_dict = st.group_stats_aseg(subjects_list, TEST_DATA_DIR)
    res = st.extract_column_from_table_data(all_subjects_table_data_dict, 'StructName', 'NVoxels')
    assert len(res) == 45
    assert b'Left-Lateral-Ventricle' in res
    for struct_name in res:
        num_voxels_of_structure = res[struct_name]
        assert num_voxels_of_structure.shape == (2, )
    assert res[b'Left-Lateral-Ventricle'][0] == pytest.approx(12159, 0.01)
    assert res[b'Left-Lateral-Ventricle'][1] == pytest.approx(12159, 0.01)    # subject2 is copied from subject1


def test_extract_column_from_table_data_invalid_label_name():
    all_subjects_table_data_dict = {'StructName': np.array([['3rd-ventricle', '4th-ventricle'], ['3rd-ventricle', '4th-ventricle']]), 'NVoxels': np.array([['333', '444'], ['333', '444']])}
    with pytest.raises(ValueError) as exc_info:
        res = st.extract_column_from_table_data(all_subjects_table_data_dict, 'noSuchLabelName', 'NVoxels')
    assert 'Given column_name_for_dict_keys' in str(exc_info.value)
    assert 'noSuchLabelName' in str(exc_info.value)


def test_extract_column_from_table_data_invalid_value_name():
    all_subjects_table_data_dict = {'StructName': np.array([['3rd-ventricle', '4th-ventricle'], ['3rd-ventricle', '4th-ventricle']]), 'NVoxels': np.array([['333', '444'], ['333', '444']])}
    with pytest.raises(ValueError) as exc_info:
        res = st.extract_column_from_table_data(all_subjects_table_data_dict, 'StructName', 'noSuchValueName')
    assert 'Given column_name_of_values' in str(exc_info.value)
    assert 'noSuchValueName' in str(exc_info.value)

def test_extract_column_from_table_data_empty_names_matrix():
    all_subjects_table_data_dict = {'StructName': np.array([]), 'NVoxels': np.array([['333', '444'], ['333', '444']])}
    with pytest.raises(ValueError) as exc_info:
        res = st.extract_column_from_table_data(all_subjects_table_data_dict, 'StructName', 'NVoxels')
    assert 'Expected non-empty 2D matrix of strings' in str(exc_info.value)
