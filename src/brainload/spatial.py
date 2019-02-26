# -*- coding: utf-8 -*-
"""
Simple functions for spatial tranformation of 3-dimensional coordinates.

These functions are helpful if you want to rotate, translate, mirror, or scale (brain) meshes. In general, you would use them roughly like this:

>>> import brainload as bl
>>> vert_coords, faces = bl.subject('bert')[0:2]
>>> x, y, z = bl.spatial.coords_a2s(vert_coords)

Now you have the coordinates of the mesh vertices in the required format and can call any function from this module:

>>> xt, yt, zt = bl.spatial.translate_3D_coordinates_along_axes(x, y, z, 5, 0, 0)   # or some other function


"""

import numpy as np
import brainload.freesurferdata as blfsd
import numpy.linalg as npl  # for matrix inversion

def rotate_3D_coordinates_around_axes(x, y, z, radians_x, radians_y, radians_z):
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

    radians_x: number
        A single number, representing the rotation in radians around the x axis.

    radians_y: number
        A single number, representing the rotation in radians around the y axis.

    radians_z: number
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
    xr, yr, zr = _rotate_3D_coordinates_around_x_axis(x, y, z, radians_x)
    xr, yr, zr = _rotate_3D_coordinates_around_y_axis(xr, yr, zr, radians_y)
    xr, yr, zr = _rotate_3D_coordinates_around_z_axis(xr, yr, zr, radians_z)
    return xr, yr, zr


def _rotate_3D_coordinates_around_x_axis(x, y, z, radians):
    """
    Rotate coordinates around the x axis. Rotation must be given in radians.

    Rotate coordinates around the x axis. See the documentation for `rotate_3D_coordinates_around_axes` for details.
    """
    y_rotated  = np.cos(radians) * y - np.sin(radians) * z
    z_rotated  = np.sin(radians) * y + np.cos(radians) * z
    x_rotated  = x
    return x_rotated, y_rotated, z_rotated


def _rotate_3D_coordinates_around_y_axis(x, y, z, radians):
    """
    Rotate coordinates around the y axis. Rotation must be given in radians.

    Rotate coordinates around the y axis. See the documentation for `rotate_3D_coordinates_around_axes` for details.
    """
    z_rotated = np.cos(radians) * z - np.sin(radians) * x
    x_rotated = np.sin(radians) * z + np.cos(radians) * x
    y_rotated = y
    return x_rotated, y_rotated, z_rotated


def _rotate_3D_coordinates_around_z_axis(x, y, z, radians):
    """
    Rotate coordinates around the z axis. Rotation must be given in radians.

    Rotate coordinates around the z axis. See the documentation for `rotate_3D_coordinates_around_axes` for details.
    """
    x_rotated = np.cos(radians) * x - np.sin(radians) * y
    y_rotated = np.sin(radians) * x + np.cos(radians) * y
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
    >>> import brainload.spatial as st; import numpy as np
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
        A 1D array representing x axis coordinates. Must have the same length as the `y` and `z` arrays.

    y: Numpy array of numbers
        A 1D array representing y axis coordinates. Must have the same length as the `x` and `z` arrays.

    z: Numpy array of numbers
        A 1D array, representing z axis coordinates. Must have the same length as the `x` and `y` arrays.

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
    >>> import brainload.spatial as st; import numpy as np
    >>> x = np.array([5, 6])
    >>> y = np.array([7, 8])
    >>> z = np.array([9, 10])
    >>> xt, yt, zt = st.translate_3D_coordinates_along_axes(x, y, z, 2, -4, 0)
    >>> print "%d %d %d" % (xt[0], yt[0], zt[0])     # 7 3 9
    >>> print "%d %d %d" % (xt[1], yt[1], zt[1])     # 8 4 10
    """
    x_shifted = x + shift_x
    y_shifted = y + shift_y
    z_shifted = z + shift_z
    return x_shifted, y_shifted, z_shifted


def scale_3D_coordinates(x, y, z, x_scale_factor, y_scale_factor=None, z_scale_factor=None):
    """
    Scale coordinates by factors.

    Scale the given coordinates by the given scale factor or factors.

    Parameters
    ----------
    x: Numpy array of numbers
        A 1D array representing x axis coordinates. Must have the same length as the `y` and `z` arrays.

    y: Numpy array of numbers
        A 1D array representing y axis coordinates. Must have the same length as the `x` and `z` arrays.

    z: Numpy array of numbers
        A 1D array, representing z axis coordinates. Must have the same length as the `x` and `y` arrays.

    x_scale_factor: number
        A single number, representing the scaling factor along the x axis. If the other values are not given, this counts for all axes.

    y_scale_factor: number | None
        A single number, representing the scaling factor along the y axis. If this is `None`, the value given for `x_scale_factor` is used.

    z_scale_factor: number | None
        A single number, representing the scaling factor along the z axis. If this is `None`, the value given for `x_scale_factor` is used.

    Returns
    -------
    x_scaled: Numpy array of numbers
        The scaled x coordinates.

    y_scaled: Numpy array of numbers
        The scaled y coordinates.

    z_scaled: Numpy array of numbers
        The scaled z coordinates.

    Examples
    --------
    >>> import brainload.spatial as st; import numpy as np
    >>> x = np.array([5, 6])
    >>> y = np.array([7, 8])
    >>> z = np.array([9, 10])
    >>> xs, ys, zs = st.scale_3D_coordinates(x, y, z, 3.0)
    >>> print "%d %d %d" % (xs[0], ys[0], zs[0])     # 15 21 27
    >>> print "%d %d %d" % (xs[1], ys[1], zs[1])     # 18 24 30
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

    Parameters
    ----------
    x: Numpy array of numbers
        A 1D array representing x axis coordinates. Must have the same length as the `y` and `z` arrays.

    y: Numpy array of numbers
        A 1D array representing y axis coordinates. Must have the same length as the `x` and `z` arrays.

    z: Numpy array of numbers
        A 1D array, representing z axis coordinates. Must have the same length as the `x` and `y` arrays.

    axis: string, one of {'x', 'y', 'z'}
        An axis identifier.

    mirror_at_axis_coordinate: number | None
        The coordinate along the axis `axis` at which the mirror plane should be created. If you set `axis` to 'x' and specify `5` for this, a yz-plane will be used at x coordinate 5. If not given, it defaults to the minimal axis coordinate for the respective axis in the data.

    Returns
    -------
    x_mirrored: Numpy array of numbers
        The mirrored x coordinates.

    y_mirrored: Numpy array of numbers
        The mirrored y coordinates.

    z_mirrored: Numpy array of numbers
        The mirrored z coordinates.

    Examples
    --------
    Mirror at the origin of the x axis:

    >>> import brainload.spatial as st; import numpy as np
    >>> x = np.array([5, 6])
    >>> y = np.array([7, 8])
    >>> z = np.array([9, 10])
    >>> xm, ym, zm = st.mirror_3D_coordinates_at_axis(x, y, z, 'x', 0)
    >>> print "%d %d %d" % (xm[0], ym[0], zm[0])     # -5 7 9
    >>> print "%d %d %d" % (xm[1], ym[1], zm[1])     # -6 8 10
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

    Parameters
    ----------
    x: Numpy array of numbers
        A 1D array representing x axis coordinates. Must have the same length as the `y` and `z` arrays.

    y: Numpy array of numbers
        A 1D array representing y axis coordinates. Must have the same length as the `x` and `z` arrays.

    z: Numpy array of numbers
        A 1D array, representing z axis coordinates. Must have the same length as the `x` and `y` arrays.

    point_x: number
        The x coordinate of the point used for mirroring.

    point_y: number
        The y coordinate of the point used for mirroring.

    point_z: number
        The z coordinate of the point used for mirroring.

    Returns
    -------
    xm: Numpy array of numbers
        The mirrored x coordinates.

    ym: Numpy array of numbers
        The mirrored y coordinates.

    zm: Numpy array of numbers
        The mirrored z coordinates.

    Examples
    --------
    Mirror at the origin:

    >>> import brainload.spatial as st; import numpy as np
    >>> x = np.array([5, 6])
    >>> y = np.array([7, 8])
    >>> z = np.array([9, 10])
    >>> xm, ym, zm = st.point_mirror_3D_coordinates(x, y, z, 0, 0, 0)
    >>> print "%d %d %d" % (xm[0], ym[0], zm[0])     # -5 -7 -9
    >>> print "%d %d %d" % (xm[1], ym[1], zm[1])     # -6 -8 -10
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


def get_affine_matrix_MNI305_to_MNI152():
    """
    Get the transformation matrix from fsaverage space (=MNI305 space) to MNI152 space.

    Get the transformation matrix from fsaverage space (=MNI305 space) to MNI152 space. This matrix was retrieved from http://freesurfer.net/fswiki/CoordinateSystems, see use case 8 at the very bottom of the page.
    Quoting the linked page: 'The above matrix is V152*inv(T152)*R*T305*inv(V305), where V152 and V305 are the vox2ras matrices from the 152 and 305 spaces, T152 and T305 are the tkregister-vox2ras matrices from the 152 and 305 spaces, and R is from $FREESURFER_HOME/average/mni152.register.dat'
    """
    return np.array([[0.9975, -0.0073, 0.0176, -0.0429], [0.0146, 1.0009, -0.0024, 1.5496], [-0.0130, -0.0093, 0.9971, 1.1840], [0, 0, 0, 1]])


def get_affine_matrix_MNI152_to_MNI305():
    """
    Get the transformation matrix from MNI152 sapce to fsaverage space (=MNI305 space).

    This is the inverse of the MNI305 to MNI152 matrix.

    Returns
    -------
    2D numpy array
        The affine transformation matrix, a float matrix with shape (4, 4).
    """
    return npl.inv(get_affine_matrix_MNI305_to_MNI152())


def parse_registration_matrix(matrix_lines):
    """
    Parse a registration matrix.

    Parse a registration matrix from the 4 lines representing it in a register.dat file. See https://surfer.nmr.mgh.harvard.edu/fswiki/RegisterDat for the file format. This function expects only the 4 matrix lines.

    Parameters
    ----------
    matrix_lines: list of str
        The 4 matrix lines of a file in register.dat format. Each line muyt contain 4 floats, separated by spaces.

    Returns
    -------
    2D numpy array of floats
        The parsed matrix, with dimension (4, 4).
    """
    if len(matrix_lines) != 4:
        raise ValueError("Registration matrix has wrong line count. Expected exactly 4 lines, got %d." % len(matrix_lines))
    reg_matrix = np.zeros((4, 4))
    for idx, line in enumerate(matrix_lines):
        reg_matrix[idx] = np.fromstring(line, dtype=np.float_, sep=' ')
    return reg_matrix


def apply_affine(i, j, k, affine_matrix):
    """
    Applies an affine matrix to the given coordinates. The affine matrix consists of a 3x3 rotation matrix and a 3x1 transposition matrix (plus the last row).

    Parameters
    ----------
    i, j, k: numeric scalars or array-likes
        The source coordinates.

    affine_matrix: numpy 2D float array with shape (4, 4)
        The affine matrix

    Returns
    -------
    The coordinate vector after applying the matrix. A 1D array (vector) of shape (3, ) if the inputs were scalar, a 2D array with shape (n, 3) otherwise, were n is the length of the input array-likes.

    See also
    --------
    ```apply_affine_3D``` can handle a 2D matrix of coordinates, e.g., with shape (n, 3) for n 3D coordinates.
    """
    if np.isscalar(i):
        return _apply_affine_scalar(i, j, k, affine_matrix)
    else:
        return _apply_affine_arraylike(i, j, k, affine_matrix)


def _apply_affine_arraylike(i, j, k, affine_matrix):
    """
    Applies an affine matrix to the given coordinates. The 3 values i, j and k must be array-likes (usually numpy 1D arrays) with identical shape. The affine matrix consists of a 3x3 rotation matrix and a 3x1 transposition matrix (plus the last row).
    """
    rotation = affine_matrix[:3, :3]
    translation = affine_matrix[:3, 3]
    res = np.zeros((i.size, 3))
    for idx, row in enumerate(i):
        res[idx,:] = rotation.dot([i[idx], j[idx], k[idx]]) + translation
    return np.transpose(res)


def _apply_affine_scalar(i, j, k, affine_matrix):
    """
    Applies an affine matrix to the given coordinates. The 3 values i, j and k must be scalars. The affine matrix consists of a 3x3 rotation matrix and a 3x1 transposition matrix (plus the last row).

    Parameters
    ----------
    i, j, k: numeric scalars
        The source coordinates.

    affine_matrix: numpy 2D float array with shape (4, 4)
        The affine matrix

    Returns
    -------
    The coordinate vector after applying the matrix.
    """
    rotation = affine_matrix[:3, :3]
    translation = affine_matrix[:3, 3]
    return rotation.dot([i, j, k]) + translation


def apply_affine_3D(coords_3d, affine_matrix):
    """
    Apply an affine transformation to all coordinates.

    Parameters
    ----------
    coords_3d: numpy 2D array
        The source coordinates, given as a 2D numpy array with shape (n, 3). Each of the n rows represents a point in space, given by its x, y and z coordinates.

    affine_matrix: numpy 2D float array with shape (4, 4)
        The affine matrix

    Returns
    -------
    The coordinates after applying the matrix, 2D numpy array with shape (n, 3). Same shape as the input coords.
    """
    rotation = affine_matrix[:3, :3]
    translation = affine_matrix[:3, 3]
    res = np.zeros((coords_3d.shape[0], 3))
    for idx, row in enumerate(coords_3d):
        res[idx,:] = rotation.dot(row) + translation
    return res


def get_freesurfer_matrix_vox2ras():
    """
    The FreeSurfer vox2ras matrix, this is identical to the vox2ras-tkr matrix.

    Retrieved from http://freesurfer.net/fswiki/CoordinateSystems, use case 4. Note that fsaverage is in MNI305 space. Generated by running: `mri_info --vox2ras-tkr $FREESURFER_HOME/subjects/fsaverage/mri/orig.mgz`. See also http://eeg.sourceforge.net/doc_m2html/bioelectromagnetism/freesurfer_surf2voxels.html. Quoting from that page: 'FreeSurfer MRI volumes are 256^3 voxels, 1mm^3 each. The MRI volume index has an origin at the left, posterior, inferior voxel, such that:
       - Sagital increases from left to right (+X Right)
       - Coronal increases from posterior to anterior (+Y Anterior)
       - Axial   increases from inferior to superior (+Z Superior).

    The MRI RAS values have an origin at the middle of the volume, in approximately voxel 128, 128, 128.'

    Returns
    -------
    2D numpy array
        The affine transformation matrix, a float matrix with shape (4, 4).
    """
    return np.array([[-1.00000, 0.00000, 0.00000, 128.00000],
   [0.00000, 0.00000, 1.00000, -128.00000],
   [0.00000, -1.00000, 0.00000, 128.00000],
   [0.00000, 0.00000, 0.00000, 1.00000]])


def get_freesurfer_matrix_ras2vox():
    """
    Get standard matrix to convert RAS coordinate to voxel index for Freesurfer conformed space volumes.

    Get matrix to convert RAS coordinate to voxel index for Freesurfer conformed space volumes. See the documentation for get_freesurfer_matrix_vox2ras for background information.

    Returns
    -------
    2D numpy array
        The affine transformation matrix, a float matrix with shape (4, 4).
    """
    return npl.inv(get_freesurfer_matrix_vox2ras())


def get_n_neighborhood_start_stop_indices_3D(volume_shape, point, n):
    """
    Compute the start and stop indices along the 3 dimensions for the n-neighborhood of the point within the 3D volume.

    Note that this returns an index range where the end is *non-inclusive*! So for a point at x,y,z with 0-neighborhood (only the point itself), you will get x,x+1,y,y+1,z,z+1 as return values.

    Parameters
    ----------
    volume_shape: 3-tuple of int
        The shape of the volume (e.g., whole 3D image)

    point: 1D array of length 3
        The x, y, and z coordinates of the query point. Must lie within the volume. This is the point around which the neighborhood will be computed.

    n: int >= 0
        The neighborhood size (in every direction, the neighborhood is always square). For 0, only the index of the point itself will be returned. For 1, the 26 neighbors in distance 1 plus the index of the point itself (so 27 indices) will be returned. If the point is close to the border of the volume, only the valid subset will be returned of course. For n=2 you get (up to) 125 indices.

    Returns
    -------
    xstart: int
            The x start index, inclusive.

    xend: int
            The x end index, exclusive.

    ystart: int
            The y start index, inclusive.

    yend: int
            The y end index, exclusive.

    zstart: int
            The z start index, inclusive.

    zend: int
            The z end index, exclusive.

    Examples
    --------
    >>> volume = np.zeros((3, 3, 3))
    >>> point = [2, 2, 2]
    >>> xstart, xend, ystart, yend, zstart, zend = st.get_n_neighborhood_start_stop_indices_3D(volume.shape, point, 1)    # 1-neighborhood
    """
    vx = volume_shape[0]
    vy = volume_shape[1]
    vz = volume_shape[2]

    # now set valid ones to 1
    xstart = max(0, point[0]-n)
    xend = min(point[0]+1+n, vx)
    ystart = max(0, point[1]-n)
    yend = min(point[1]+1+n, vy)
    zstart= max(0, point[2]-n)
    zend = min(point[2]+1+n, vz)
    return xstart, xend, ystart, yend, zstart, zend


def get_n_neighborhood_start_stop_indices_3D_points(volume_shape, points, n):
    """
    Compute the start and stop indices along the 3 dimensions for the n-neighborhood of the points within the 3D volume.

    Note that this returns an index range where the end is *non-inclusive*! So for a point at x,y,z with 0-neighborhood (only the point itself), you will get x,x+1,y,y+1,z,z+1 as return values.

    Parameters
    ----------
    volume_shape: 3-tuple of int
        The shape of the volume (e.g., whole 3D image)

    points: 2D array of shape (n, 3) for n points
        The x, y, and z coordinates of the query points. Must lie within the volume. These are the points around which the neighborhoods will be computed.

    n: int >= 0
        The neighborhood size (in every direction, the neighborhood is always square). For 0, only the index of the point itself will be returned. For 1, the 26 neighbors in distance 1 plus the index of the point itself (so 27 indices) will be returned. If the point is close to the border of the volume, only the valid subset will be returned of course. For n=2 you get (up to) 125 indices.

    Returns
    -------
    xstart: 1D numpy int array
            The x start indices, inclusive.

    xend: 1D numpy int array
            The x end indices, exclusive.

    ystart: 1D numpy int array
            The y start indices, inclusive.

    yend: 1D numpy int array
            The y end indices, exclusive.

    zstart: 1D numpy int array
            The z start indices, inclusive.

    zend: 1D numpy int array
            The z end indices, exclusive.

    Examples
    --------
    >>> volume = np.zeros((3, 3, 3))
    >>> points = np.array([[1, 1, 1], [2,2,2]])
    >>> xstart, xend, ystart, yend, zstart, zend = st.get_n_neighborhood_start_stop_indices_3D_points(volume.shape, points, 1)    # 1-neighborhood
    """
    vx = volume_shape[0]
    vy = volume_shape[1]
    vz = volume_shape[2]

    num_points = points.shape[0]
    # now set valid ones to 1
    zeros = np.zeros((num_points), dtype=int)

    xstart = np.maximum(zeros, points[:,0]-n)
    xend = np.minimum(points[:,0]+1+n, np.ones((num_points), dtype=int) * vx)
    ystart = np.maximum(zeros, points[:,1]-n)
    yend = np.minimum(points[:,1]+1+n, np.ones((num_points), dtype=int) * vy)
    zstart= np.maximum(zeros, points[:,2]-n)
    zend = np.minimum(points[:,2]+1+n, np.ones((num_points), dtype=int) * vz)
    return xstart, xend, ystart, yend, zstart, zend


def get_n_neighborhood_indices_3D(volume_shape, point, n):
    """
    Compute the indices of the n-neighborhood of the point within the volume.

    Compute the indices of the square n-neighborhood of the point within the volume, including the point itself. This function returns only valid indices in the volume.

    Parameters
    ----------
    volume_shape: 3-tuple of int
        The shape of the volume (e.g., whole 3D image)

    point: 1D array of length 3
        The x, y, and z coordinates of the query point. Must lie within the volume. This is the point around which the neighborhood will be computed.

    n: int >= 0
        The neighborhood size (in every direction, the neighborhood is always square). For 0, only the index of the point itself will be returned. For 1, the 26 neighbors in distance 1 plus the index of the point itself (so 27 indices) will be returned. If the point is close to the border of the volume, only the valid subset will be returned of course. For n=2 you get (up to) 125 indices.

    Returns
    -------
    indices: tuple of 3 numpy 1D arrays
            The 3 arrays have identical size and contain the x, y, and z indices of the neighborhood.

    Examples
    --------
    >>> volume = np.zeros((3, 3, 3))
    >>> point = [1, 1, 1]
    >>> indices = st.get_n_neighborhood_indices_3D(volume.shape, point, 1)
    """
    point = np.array(point)
    if point.shape != (3, ):
        raise ValueError("Point must be a 1D array and have 3 entries (shape=(3, )), but it has shape %s." % str(point.shape))
    xstart, xend, ystart, yend, zstart, zend = get_n_neighborhood_start_stop_indices_3D(volume_shape, point, n)
    M = np.zeros(volume_shape, dtype=int)   # all disabled
    M[xstart : xend, ystart : yend, zstart : zend] = 1
    indices = np.nonzero(M)
    return indices


def get_n_neighborhood_indices_3D_points(volume_shape, points, n):
    """
    Compute the indices of the n-neighborhood of the point within the volume.

    Compute the indices of the square n-neighborhood of the point within the volume, including the point itself. This function returns only valid indices in the volume.

    Parameters
    ----------
    volume_shape: 3-tuple of int
        The shape of the volume (e.g., whole 3D image)

    points: 2D array of shape (n, 3) for n points
        The x, y, and z coordinates of the query points. Must lie within the volume. These are the points around which the neighborhoods will be computed.

    n: int >= 0
        The neighborhood size (in every direction, the neighborhood is always square). For 0, only the index of the point itself will be returned. For 1, the 26 neighbors in distance 1 plus the index of the point itself (so 27 indices) will be returned. If the point is close to the border of the volume, only the valid subset will be returned of course. For n=2 you get (up to) 125 indices.

    Returns
    -------
    indices: tuple of 3 numpy 1D arrays
            The 3 arrays have identical size and contain the x, y, and z indices of the neighborhood.

    Examples
    --------
    >>> volume = np.zeros((3, 3, 3))
    >>> points = np.array([[1, 1, 1], [2, 2, 2]])
    >>> indices = st.get_n_neighborhood_indices_3D_points(volume.shape, points, 1)
    """
    xstart, xend, ystart, yend, zstart, zend = get_n_neighborhood_start_stop_indices_3D_points(volume_shape, points, n)
    M = np.zeros(volume_shape, dtype=int)   # all disabled
    for idx, value in enumerate(xstart):
        M[xstart[idx] : xend[idx], ystart[idx] : yend[idx], zstart[idx] : zend[idx]] = 1
    indices = np.nonzero(M)
    return indices


def get_equivalent_voxel_of_raw_volume_in_conformed_volume(raw_volume_file, conformed_volume_file, raw_volume_query_voxels_crs):
    """
    Find the position of a voxel in the conformed volume.

    Find the voxel CRS in the conformed volume that is equivalent to the query voxel CRS in the raw volume. Note that this always returns a single voxel for a query voxel. It does not compute all voxel that may represent the source voxel (e.g., if the voxel size is much smaller in the destination volume). Note that this function also works in reverse (simply exchange the order of the 2 volumes and give query voxels in the first one). In fact, it will work between any pair of volume files as long as they within the same space (i.e., the RAS coordinates of the same spot in both files are identical). The function works based on the vox2ras matrix in the source file and the ras2vox matrix in the destination file.

    Parameters
    ----------
    raw_volume_file: string
        Path to the raw volume, i.e., the volume that has not yet been processed using FreeSurfer. You could also use ```rawavg.mgz``` in the ```mri``` sub directory of a subject. Must be in mgh or mgz format. You can use the ```mri_convert``` binary that comes with FreeSurfer to convert from nifti to mgz.

    conformed_volume_file: string
        Path to a conformed volume, i.e., a volume that has been conformed to 256x256x256 voxels with 1mm^3 voxel volume. This applies to all subject volumes in the ```mri``` directory of a subject, e.g., ```brain.mgz``` or ```orig.mgz```. Must be in mgh or mgz format. You can use the ```mri_convert``` binary that comes with FreeSurfer to convert from nifti to mgz. Hint: You can create a conformed version from a raw volume using the ```--conform``` option of ```mri_convert```.

    raw_volume_query_voxels_crs: numpy 2D array of int
        The column, row, slice indices of the query voxels in the raw volume. The array must have dimension (n, 3) for n query voxels.

    Returns
    -------
    conf_volume_voxels_crs: numpy 2D array of int
        The column, row, slice indices of the query voxels in the conformed volume. The array has dimension (n, 3) for n query voxels.
    """
    raw_ras2vox, raw_vox2ras, raw_vox2ras_tkr = blfsd.read_mgh_header_matrices(raw_volume_file)
    conf_ras2vox, conf_vox2ras, conf_vox2ras_tkr = blfsd.read_mgh_header_matrices(conformed_volume_file)
    query_voxels_ras = apply_affine_3D(raw_volume_query_voxels_crs, raw_vox2ras)
    conf_volume_voxels_crs = np.rint(apply_affine_3D(query_voxels_ras, conf_ras2vox)).astype(int)
    return conf_volume_voxels_crs
