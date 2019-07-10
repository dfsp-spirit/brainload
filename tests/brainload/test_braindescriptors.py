import os
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import brainload as bl
import brainload.freesurferdata as fsd
import brainload.braindescriptors as bd


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(THIS_DIR, os.pardir, 'test_data')

# Respect the environment variable BRAINLOAD_TEST_DATA_DIR if it is set. If not, fall back to default.
TEST_DATA_DIR = os.getenv('BRAINLOAD_TEST_DATA_DIR', TEST_DATA_DIR)

def test_braindescriptors_init_nonempty():
    subjects_list = ['subject1', 'subject2']
    bdi = bd.BrainDescriptors(TEST_DATA_DIR, subjects_list)


def test_braindescriptors_parcellation_stats():
    subjects_list = ['subject1', 'subject2']
    bdi = bd.BrainDescriptors(TEST_DATA_DIR, subjects_list)
    bdi.add_parcellation_stats(['aparc', 'aparc.a2009s'])
