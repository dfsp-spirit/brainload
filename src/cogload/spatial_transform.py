import numpy as np

def rotate_3D_coordinates_around_axis(x, y, z, rot, axis):
    '''Rotate coordinates around the given axis. Rotation must be given in radians.'''
    if axis not in ('x', 'y', 'z'):
        raise ValueError("ERROR: axis must be one of {'x', 'y', 'z'}")

    if axis == 'x':
        return rotate_3D_coordinates_around_x_axis(x, y, z, rot)
    elif axis == 'y':
        return rotate_3D_coordinates_around_y_axis(x, y, z, rot)
    else:
        return rotate_3D_coordinates_around_y_axis(x, y, z, rot)

def rotate_3D_coordinates_around_axes(x, y, z, rotx, roty, rotz):
    '''Rotate coordinates around the axes. Rotation must be given in radians.'''
    xr, yr, zr = rotate_3D_coordinates_around_x_axis(x, y, z, rotx)
    xr, yr, zr = rotate_3D_coordinates_around_y_axis(xr, yr, zr, roty)
    xr, yr, zr = rotate_3D_coordinates_around_z_axis(xr, yr, zr, rotz)
    return xr, yr, zr


def rotate_3D_coordinates_around_x_axis(x, y, z, rot):
    y_rotated  = np.cos(rot) * y - np.sin(rot) * z
    z_rotated  = np.sin(rot) * y + np.cos(rot) * z
    x_rotated  = x
    return x_rotated, y_rotated, z_rotated


def rotate_3D_coordinates_around_y_axis(x, y, z, rot):
    z_rotated = np.cos(rot) * z - np.sin(rot) * x
    x_rotated = np.sin(rot) * z + np.cos(rot) * x
    y_rotated = y
    return x_rotated, y_rotated, z_rotated

def translate_3D_coordinates_along_axes(x, y, z, shift_x, shift_y, shift_z):
    '''Translates coordinates along one or more axes'''
    x_shifted = x + shift_x
    y_shifted = y + shift_y
    z_shifted = z + shift_z
    return x_shifted, y_shifted, z_shifted

def scale_3D_coordinates(x, y, z, x_scale_factor, y_scale_factor=None, z_scale_factor=None):
    '''Scales the given coordinates by the given scale factor.'''
    if y_scale_factor is None:
        y_scale_factor = x_scale_factor
    if z_scale_factor is None:
        z_scale_factor = x_scale_factor
    x_scaled = x * x_scale_factor
    y_scaled = y * y_scale_factor
    z_scaled = z * z_scale_factor
    return x_scaled, y_scaled, z_scaled

def mirror_3D_coordinates_at_axis(x, y, z, axis, mirror_at_axis_coordinate=None):
    '''Mirrors or reflects the given 3D coordinates on a plane (perpendicular to the axis) at axis coordinate x at the given axis. If mirror_at_axis_coordinate is not given, the smallest coordinate along the mirror axis is used.'''
    if axis not in ('x', 'y', 'z'):
        raise ValueError("ERROR: axis must be one of {'x', 'y', 'z'}")

    if axis == 'x':
        return mirror_coordinates_at_axis(x, mirror_at_axis_coordinate), np.copy(y), np.copy(z)
    elif axis == 'y':
        return np.copy(x), mirror_coordinates_at_axis(y, mirror_at_axis_coordinate), np.copy(z)
    else:
        return np.copy(x), np.copy(y), mirror_coordinates_at_axis(z, mirror_at_axis_coordinate)

def mirror_coordinates_at_axis(c, mirror_at_axis_coordinate=None):
    '''Mirrors or reflects a 1-dimensional array of coordinates on a plane (perpendicular to the axis) at the given axis coordinate. If no coordinate is given, the minimum value of the coordinates is used.'''
    if mirror_at_axis_coordinate is None:
        mirror_at_axis_coordinate = np.min(c)
    c_mirrored = mirror_at_axis_coordinate - (c - mirror_at_axis_coordinate)
    return c_mirrored

def point_mirror_3D_coordinates(x, y, z, point_x, point_y, point_z):
    '''Point-mirrors or reflects the given coordinates at the given point.'''
    return mirror_coordinates_at_axis(x, point_x), mirror_coordinates_at_axis(y, point_y), mirror_coordinates_at_axis(z, point_z)





def rad2deg(rad):
    '''Convert an angle given in radians to degree.'''
    if rad < 0 or rad > 2 * np.pi:
        adjusted = rad % (2 * np.pi)
        print "WARNING: rad2deg: called with unusual rad value '%d'. Adjusted to '%d'." % (rad, adjusted)
        rad = adjusted
    return rad * 180 / np.pi

def deg2rad(degree):
    '''Convert an angle given in degree to radians.'''
    if degree < 0 or degree > 360:
        adjusted = degree % 360
        print "WARNING: deg2rad: called with unusual degree value '%d'. Adjusted to '%d'." % (degree, adjusted)
        degree = adjusted
    return degree * np.pi / 180

def rotate_3D_coordinates_around_z_axis(x, y, z, rot):
    x_rotated = np.cos(rot) * x - np.sin(rot) * y
    y_rotated = np.sin(rot) * x + np.cos(rot) * y
    z_rotated = z
    return x_rotated, y_rotated, z_rotated
