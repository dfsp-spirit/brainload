# -*- coding: utf-8 -*-
"""
Functions for spatial tranformation of 3-dimensional coordinates.

These functions are helpful if you want to rotate, translate, mirror, or scale (brain) meshes.
"""

import numpy as np

def rotate_3D_coordinates_around_axes(x, y, z, rotx, roty, rotz):
    """
    Rotate coordinates around the 3 axes.

    Rotate coordinates around the x, y, and z axes. The rotation values must be given in radians.

    Parameters
    ----------
    x: Numpy array of numbers
        A 1D array representing x axis coordinates. Must have the same length as the `y` and `z` arrays. (See `coords_a2s` if you have a single 2D array containing all 3.)

    y: Numpy array of numbers
        A 1D array representing y axis coordinates. Must have the same length as the `x` and `z` arrays. (See `coords_a2s` if you have a single 2D array containing all 3.)

    z: Numpy array of numbers
        A 1D array, representing z axis coordinates. Must have the same length as the `x` and `y` arrays. (See `coords_a2s` if you have a single 2D array containing all 3.)

    rotx: number
        A single number, representing the rotation in radians around the x axis.

    roty: number
        A single number, representing the rotation in radians around the y axis.

    rotz: number
        A single number, representing the rotation in radians around the z axis.

    Returns
    -------
    xr: Numpy array of numbers
        The rotated x coordinates.

    yr: Numpy array of numbers
        The rotated y coordinates.

    zr: Numpy array of numbers
        The rotated z coordinates.

    Examples
    --------
    >>> import brainload.spatial as st; import numpy as np;
    >>> x = np.array([5, 6])
    >>> y = np.array([7, 8])
    >>> z = np.array([9, 10])
    >>> xr, yr, zr = st.rotate_3D_coordinates_around_axes(x, y, z, np.pi, 0, 0)
    """
    xr, yr, zr = _rotate_3D_coordinates_around_x_axis(x, y, z, rotx)
    xr, yr, zr = _rotate_3D_coordinates_around_y_axis(xr, yr, zr, roty)
    xr, yr, zr = _rotate_3D_coordinates_around_z_axis(xr, yr, zr, rotz)
    return xr, yr, zr


def _rotate_3D_coordinates_around_x_axis(x, y, z, rot):
    """
    Rotate coordinates around the x axis. Rotation must be given in radians.
    """
    y_rotated  = np.cos(rot) * y - np.sin(rot) * z
    z_rotated  = np.sin(rot) * y + np.cos(rot) * z
    x_rotated  = x
    return x_rotated, y_rotated, z_rotated


def _rotate_3D_coordinates_around_y_axis(x, y, z, rot):
    """
    Rotate coordinates around the y axis. Rotation must be given in radians.
    """
    z_rotated = np.cos(rot) * z - np.sin(rot) * x
    x_rotated = np.sin(rot) * z + np.cos(rot) * x
    y_rotated = y
    return x_rotated, y_rotated, z_rotated


def _rotate_3D_coordinates_around_z_axis(x, y, z, rot):
    """
    Rotate coordinates around the z axis. Rotation must be given in radians.
    """
    x_rotated = np.cos(rot) * x - np.sin(rot) * y
    y_rotated = np.sin(rot) * x + np.cos(rot) * y
    z_rotated = z
    return x_rotated, y_rotated, z_rotated


def coords_a2s(coords):
    """
    Split single array for all 3 coords into 3 separate ones.

    Split a 2D array with shape (3, n) of coordinates (x, y, z values) into 3 separate 1D arrays of length n.

    Parameters
    ----------
    coords: Numpy 2D array of numbers
        The merged coordinate array.

    Returns
    -------
    x: Numpy array of numbers
        A 1D array representing x axis coordinates. Has the same length as the `y` and `z` arrays.

    y: Numpy array of numbers
        A 1D array representing y axis coordinates. Has the same length as the `x` and `z` arrays.

    z: Numpy array of numbers
        A 1D array, representing z axis coordinates. Has the same length as the `x` and `y` arrays.

    Examples
    --------
    >>> import brainload.spatial as st; import numpy as np;
    >>> coords = np.array([[5, 7, 9], [6, 8, 10]])
    >>> x, y, z = st.coords_a2s(coords)
    >>> print y[1]
    8
    """
    x = coords[:,0]
    y = coords[:,1]
    z = coords[:,2]
    return np.asarray(x), np.asarray(y), np.asarray(z)


def coords_s2a(x, y, z):
    """
    Separate a single xyz coordinate array into x, y and z arrays.

    Merge 3 arrays of length n with coordinates (x, y, z values) into a single 2D coordinate array of shape (3, n).

    Parameters
    ----------
    x: Numpy array of numbers
        A 1D array representing x axis coordinates. Must have the same length as the `y` and `z` arrays.

    y: Numpy array of numbers
        A 1D array representing y axis coordinates. Must have the same length as the `x` and `z` arrays.

    z: Numpy array of numbers
        A 1D array, representing z axis coordinates. Must have the same length as the `x` and `y` arrays.

    Returns
    -------
    Numpy 2D array of numbers
        The merged coordinate array.

    Examples
    --------
    >>> import brainload.spatial as st; import numpy as np;
    >>> x = np.array([5, 6])
    >>> y = np.array([7, 8])
    >>> z = np.array([9, 10])
    >>> coords = st.coords_s2a(x, y, z)
    >>> print coords[1][2]
    10
    """
    if np.isscalar(x) and np.isscalar(y) and np.isscalar(z):
        x = np.array([x])
        y = np.array([y])
        z = np.array([z])
    return np.column_stack((x, y, z))


def translate_3D_coordinates_along_axes(x, y, z, shift_x, shift_y, shift_z):
    """
    Translate coordinates along one or more axes.

    Translate or shift coordinates along one or more axes.

    Parameters
    ----------
    x: Numpy array of numbers
        A 1D array representing x axis coordinates. Must have the same length as the `y` and `z` arrays. (See `coords_a2s` if you have a single 2D array containing all 3.)

    y: Numpy array of numbers
        A 1D array representing y axis coordinates. Must have the same length as the `x` and `z` arrays. (See `coords_a2s` if you have a single 2D array containing all 3.)

    z: Numpy array of numbers
        A 1D array, representing z axis coordinates. Must have the same length as the `x` and `y` arrays. (See `coords_a2s` if you have a single 2D array containing all 3.)

    shift_x: number
        A single number, representing the shift along the x axis.

    shift_y: number
        A single number, representing the shift along the y axis.

    shift_z: number
        A single number, representing the shift along the z axis.

    Returns
    -------
    x_shifted: Numpy array of numbers
        The shifted x coordinates.

    y_shifted: Numpy array of numbers
        The shifted y coordinates.

    z_shifted: Numpy array of numbers
        The shifted z coordinates.

    Examples
    --------
    >>> import brainload.spatial as st; import numpy as np;
    """
    x_shifted = x + shift_x
    y_shifted = y + shift_y
    z_shifted = z + shift_z
    return x_shifted, y_shifted, z_shifted


def scale_3D_coordinates(x, y, z, x_scale_factor, y_scale_factor=None, z_scale_factor=None):
    """
    Scale the given coordinates by the given scale factor.
    """
    if y_scale_factor is None:
        y_scale_factor = x_scale_factor
    if z_scale_factor is None:
        z_scale_factor = x_scale_factor
    x_scaled = x * x_scale_factor
    y_scaled = y * y_scale_factor
    z_scaled = z * z_scale_factor
    return x_scaled, y_scaled, z_scaled

def mirror_3D_coordinates_at_axis(x, y, z, axis, mirror_at_axis_coordinate=None):
    """
    Mirror the given 3D coordinates on the given mirror plane.

    Mirror or reflect the given 3D coordinates on a plane (perpendicular to the axis) at axis coordinate `mirror_at_axis_coordinate` at the given axis. If `mirror_at_axis_coordinate` is not given, the smallest coordinate along the mirror axis in the data is used.
    """
    if axis not in ('x', 'y', 'z'):
        raise ValueError("ERROR: axis must be one of {'x', 'y', 'z'}")

    if axis == 'x':
        return _mirror_coordinates_at_axis(x, mirror_at_axis_coordinate), np.copy(y), np.copy(z)
    elif axis == 'y':
        return np.copy(x), _mirror_coordinates_at_axis(y, mirror_at_axis_coordinate), np.copy(z)
    else:
        return np.copy(x), np.copy(y), _mirror_coordinates_at_axis(z, mirror_at_axis_coordinate)

def _mirror_coordinates_at_axis(c, mirror_at_axis_coordinate=None):
    """
    Mirror or reflect a 1-dimensional array of coordinates on a mirror plane.

    Mirror or reflect a 1-dimensional array of coordinates on a plane (perpendicular to the axis) at the given axis coordinate. If no coordinate is given, the minimum value of the coordinates is used.
    """
    if mirror_at_axis_coordinate is None:
        mirror_at_axis_coordinate = np.min(c)
    c_mirrored = mirror_at_axis_coordinate - (c - mirror_at_axis_coordinate)
    return c_mirrored


def point_mirror_3D_coordinates(x, y, z, point_x, point_y, point_z):
    """
    Point-mirror or reflect the given coordinates at the given point.
    """
    return _mirror_coordinates_at_axis(x, point_x), _mirror_coordinates_at_axis(y, point_y), _mirror_coordinates_at_axis(z, point_z)


def rad2deg(rad):
    """
    Convert an angle given in radians to degrees.

    Convert an angle given in radians to degrees. 2 Pi radians are 360 degrees. If negative values or values larger than 2 Pi are passed, use the modulo operation to bring them to a suitable range first. In other words, passing -0.5 * Pi will be transformed to 2 - 0.5 = 1.5 Pi, and will thus return 270 degrees.

    Parameters
    ----------
    rad : float
        The angle in radians.

    Returns
    -------
    float
        The angle in degrees.

    Examples
    --------
    >>> import brainload.spatial as st
    >>> deg = st.rad2deg(2 * np.pi)   # will be 360
    """
    if rad < 0 or rad > 2 * np.pi:
        adjusted = rad % (2 * np.pi)
        rad = adjusted
    return rad * 180 / np.pi


def deg2rad(degrees):
    """
    Convert an angle given in degrees to radians.

    Convert an angle given in degrees to radians. 360 degrees are 2 Pi radians. If negative values or values larger than 360 are passed, use the modulo operation to bring them to a suitable range first. In other words, passing -90 will be transformed to 360 - 90 = 270 degrees, and will thus return 1.5 Pi radians.

    Parameters
    ----------
    degrees : float
        The angle in degrees.

    Returns
    -------
    float
        The angle in radians.

    Examples
    --------
    >>> import brainload.spatial as st
    >>> rad = st.deg2rad(180)   # will be Pi
    """
    if degrees < 0 or degrees > 360:
        adjusted = degrees % 360
        degrees = adjusted
    return degrees * np.pi / 180
