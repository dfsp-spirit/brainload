import os
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import brainload.nitools as nit
import brainload.annotations as an

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(THIS_DIR, os.pardir, 'test_data')

# Respect the environment variable BRAINLOAD_TEST_DATA_DIR if it is set. If not, fall back to default.
TEST_DATA_DIR = os.getenv('BRAINLOAD_TEST_DATA_DIR', TEST_DATA_DIR)


def test_read_annotation_md():
    annotation_file = os.path.join(TEST_DATA_DIR, 'subject1', 'label', 'lh.aparc.annot')
    labels, ctab, names, meta_data = an.read_annotation_md(annotation_file, 'lh', meta_data=None)
    assert len(meta_data) == 1
    assert meta_data['lh.annotation_file'] == annotation_file
