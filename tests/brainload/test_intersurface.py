import pytest
import numpy as np
import os
from numpy.testing import assert_array_equal, assert_allclose
import brainload as bl
import brainload.intersurface as blis


def test_face_area():
    a = np.array([0, 0, 0])
    b = np.array([0, 1, 0])
    c = np.array([0, 0, 1])
    areas = blis.face_area(a, b, c)
    assert areas.shape == (3,)
