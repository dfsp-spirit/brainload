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
    assert 'Surface has 10 vertices (with indices 0 to 9).' in ret.stdout
    assert 'Using the 1 vertex indices from the command line.' in ret.stdout
    assert 'No foreground color given on the command line (-c) and vertex index file contains no color values' in ret.stdout
    assert ret.stderr == ''


def test_visualize_verts_num_verts_color_on_commandline_several_foreground_verts(script_runner):
    ret = script_runner.run('visualize_verts', '-n' , '15', '-i',  '10,11,12', '-v')
    assert ret.success
    assert 'Verbosity' in ret.stdout
    assert 'Surface has 15 vertices (with indices 0 to 14).' in ret.stdout
    assert 'Using the 3 vertex indices from the command line.' in ret.stdout
    assert 'No foreground color given on the command line (-c) and vertex index file contains no color values' in ret.stdout
    assert ret.stderr == ''


def test_visualize_verts_wrong_vertex_list_command_line_format(script_runner):
    ret = script_runner.run('visualize_verts', '-n' , '15', '-i',  '10 11 12', '-v')
    assert not ret.success
    assert ret.stderr != ''


def test_visualize_verts_query_index_too_small(script_runner):
    ret = script_runner.run('visualize_verts', '-n' , '15', '-i',  '10,-11', '-v')
    assert not ret.success
    assert ret.stderr != ''
    assert 'Using the 2 vertex indices from the command line.' in ret.stdout
    assert "ERROR: All query indices must be >= 0, but encountered negative index '-11'. Exiting." in ret.stderr


def test_visualize_verts_query_index_too_large(script_runner):
    ret = script_runner.run('visualize_verts', '-n' , '15', '-i',  '10,200', '-v')
    assert not ret.success
    assert ret.stderr != ''
    assert 'Using the 2 vertex indices from the command line.' in ret.stdout
    assert "ERROR: All query indices must be < 15 (i.e., the number of vertices in the mesh), but encountered larger index '200'. Exiting." in ret.stderr


def test_visualize_verts_num_verts_color_on_commandline_several_foreground_verts_with_color(script_runner):
    ret = script_runner.run('visualize_verts', '-n' , '15', '-i',  '10,11,12', '-v', '-c', '0', '255', '0')
    assert ret.success
    assert 'Verbosity' in ret.stdout
    assert 'Surface has 15 vertices (with indices 0 to 14).' in ret.stdout
    assert 'Using the 3 vertex indices from the command line.' in ret.stdout
    assert not 'No foreground color given on the command line (-c) and vertex index file contains no color values' in ret.stdout
    assert "Using foreground color '0 255 0' from command line for all 3 foreground vertices."
    assert "Using background color '128 128 128'." in ret.stdout
    assert 'Resulting surface RGB map contains 3 marked vertices (3 unique).' in ret.stdout
    assert ret.stderr == ''


def test_visualize_verts_num_verts_color_on_commandline_several_nonunique_foreground_verts_with_color(script_runner):
    ret = script_runner.run('visualize_verts', '-n' , '15', '-i',  '10,11,11', '-v', '-c', '0', '255', '0')
    assert ret.success
    assert 'Verbosity' in ret.stdout
    assert 'Surface has 15 vertices (with indices 0 to 14).' in ret.stdout
    assert 'Using the 3 vertex indices from the command line.' in ret.stdout
    assert not 'No foreground color given on the command line (-c) and vertex index file contains no color values' in ret.stdout
    assert "Using foreground color '0 255 0' from command line for all 3 foreground vertices."
    assert "Using background color '128 128 128'." in ret.stdout
    assert 'Resulting surface RGB map contains 3 marked vertices (2 unique).' in ret.stdout      # Note that 2 are unique only!
    assert ret.stderr == ''

def test_visualize_verts_num_verts_color_on_commandline_several_nonunique_foreground_verts_with_color_and_bgcolor(script_runner):
    ret = script_runner.run('visualize_verts', '-n' , '15', '-i',  '10,11,11', '-v', '-c', '0', '255', '0', '-b', '10', '10', '10')
    assert ret.success
    assert 'Verbosity' in ret.stdout
    assert 'Surface has 15 vertices (with indices 0 to 14).' in ret.stdout
    assert 'Using the 3 vertex indices from the command line.' in ret.stdout
    assert not 'No foreground color given on the command line (-c) and vertex index file contains no color values' in ret.stdout
    assert "Using foreground color '0 255 0' from command line for all 3 foreground vertices."
    assert "Using background color '10 10 10'." in ret.stdout
    assert 'Resulting surface RGB map contains 3 marked vertices (2 unique).' in ret.stdout      # Note that 2 are unique only!
    assert ret.stderr == ''
