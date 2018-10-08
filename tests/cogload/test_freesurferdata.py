
import os
import pytest
import numpy as np
import cogload as cl
import cogload.freesurferdata as fsd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(THIS_DIR, os.pardir, 'test_data')

def test_get_morphology_data_suffix_for_surface_with_surf_white():
    suffix = fsd.get_morphology_data_suffix_for_surface('white')
    assert suffix == ''

def test_get_morphology_data_suffix_for_surface_with_surf_other():
    suffix = fsd.get_morphology_data_suffix_for_surface('pial')
    assert suffix == '.pial'

def test_read_mgh_file_with_valid_file():
    mgh_file = os.path.join(TEST_DATA_DIR, 'subject1', 'surf', 'rh.area.fsaverage.mgh')
    mgh_data, mgh_meta_data = fsd.read_mgh_file(mgh_file)
    assert mgh_meta_data['data_bytes_per_voxel'] == 4
