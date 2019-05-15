# Tests for the brain_mesh_info script.
#
# These tests require the package `pytest-console-scripts`.

import os
import pytest
import tempfile
import shutil

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(THIS_DIR, os.pardir, 'test_data')
TEST_VOL_FILE = os.path.join(TEST_DATA_DIR, 'subject1', 'mri', 'orig.mgz')
TEST_VOL_INDEX_FILE = os.path.join(TEST_DATA_DIR, 'subject1', 'mri', 'voxel_roi.txt')

def test_brain_mesh_info_help(script_runner):
    ret = script_runner.run('brain_vol_info', '--help')
    assert ret.success
    assert 'usage' in ret.stdout
    assert ret.stderr == ''


def test_brain_mesh_info_by_single_index(script_runner):
    ret = script_runner.run('brain_vol_info', TEST_VOL_FILE, '-i', '10', '10', '10', '-v')
    assert ret.success
    assert not 'usage' in ret.stdout
    assert '---Brain Mesh Info---' in ret.stdout
    assert 'for the 1 vertex' in ret.stdout
    assert 'has 149244 vertices and 298484 faces' in ret.stdout
    assert "Coords of vertices # ['10'] are:" in ret.stdout
    assert ret.stderr == ''
