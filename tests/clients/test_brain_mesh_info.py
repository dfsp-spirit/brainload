# Tests for the visualize_verts script.
#
# These tests require the package `pytest-console-scripts`.

import os
import pytest
import tempfile
import shutil

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(THIS_DIR, os.pardir, 'test_data')

def test_visualize_verts_help(script_runner):
    ret = script_runner.run('brain_mesh_info', '--help')
    assert ret.success
    assert 'usage' in ret.stdout
    assert ret.stderr == ''
