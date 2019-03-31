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
    ret = script_runner.run('visualize_verts', '--help')
    assert ret.success
    assert 'usage' in ret.stdout
    assert ret.stderr == ''


def test_visualize_verts_num_verts_color_on_commandline_one_foreground_vert(script_runner):
    ret = script_runner.run('visualize_verts', '-n' , '10', '-i',  '5', '-v')
    assert ret.success
    assert 'Verbosity' in ret.stdout
    assert ret.stderr == ''


def test_visualize_verts_num_verts_color_on_commandline_several_foreground_verts(script_runner):
    ret = script_runner.run('visualize_verts', '-n' , '15', '-i',  '10,11,12', '-v')
    assert ret.success
    assert 'Verbosity' in ret.stdout
    assert ret.stderr == ''


def test_visualize_verts_wrong_vertex_list_command_line_format(script_runner):
    ret = script_runner.run('visualize_verts', '-n' , '15', '-i',  '10 11 12', '-v')
    assert not ret.success
    assert ret.stderr != ''
