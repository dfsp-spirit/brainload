import pytest
import numpy as np
import os
from numpy.testing import assert_array_equal, assert_allclose
import brainload as bl


def test_face_area_single_face():
    try:
        import brainload.intersurface as blis
    except ImportError:
        pytest.skip("Optional dependencies scipy, networkx not installed, skipping tests which require it.")
    points_x = np.array([0, 5, 5])
    points_y = np.array([0, 0, 5])
    points_z = np.array([0, 0, 0])
    area = blis.face_area(points_x, points_y, points_z)
    assert area == pytest.approx(12.5, 0.00001)


def test_get_mesh_face_areas():
    try:
        import brainload.intersurface as blis
    except ImportError:
        pytest.skip("Optional dependencies scipy, networkx not installed, skipping tests which require it.")
    vert_coords = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.0, 5.0, 0.0], [10.0, 0.0, 0.0]])
    faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=int)
    areas = blis.get_mesh_face_areas(vert_coords, faces)
    assert areas.shape == (2,)
    assert areas[0] == pytest.approx(12.5, 0.00001)
    assert areas[1] == pytest.approx(12.5, 0.00001)


def test_get_convex_polygon_volumes():
    try:
        import brainload.intersurface as blis
    except ImportError:
        pytest.skip("Optional dependencies scipy, networkx not installed, skipping tests which require it.")
    # define a 3D box with size 2x2x2
    polygon_points = np.array([[0,0,0], [2,0,0], [2,2,0], [0,2,0], [0,0,2], [2,0,2], [2,2,2], [0,2,2]], dtype=float)
    vol = blis.get_convex_polygon_volume(polygon_points)
    assert vol == pytest.approx(8, 0.000001)
