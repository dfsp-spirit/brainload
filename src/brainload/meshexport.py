"""
Functions for exporting meshes.

Functions for exporting brain meshes to formats used by common 3D modeling software.
"""

import os
import sys
import numpy as np

def mesh_to_obj(vertex_coords, faces):
    """
    Write an OBJ format string of a mesh.

    Write an OBJ PLY format string of a mesh. The format is the Wavefront object format, see `https://en.wikipedia.org/wiki/Wavefront_.obj_file` for details. This exporter only writes the geometry, vertex colors are not a standard OBJ feature and are not included. Use mesh_to_ply to get vertex colors.

    Parameters
    ----------
    vertex_coords: numpy array of floats
        A 2D array containing 3 coordinates for each vertex. Dimension is (n, 3) for n vertices.

    faces: numpy array of integers
        A 2D array containing 3 vertex indices per face. Dimension is (m, 3) for m faces.

    Returns
    -------
    string
        The OBJ format string for the mesh.
    """
    verts_rep = _obj_verts(vertex_coords)
    faces_rep = _obj_faces(faces)
    hdr = "# Generated by Brainload\n"
    return ''.join([hdr, verts_rep, faces_rep])


def mesh_to_off(vertex_coords, faces):
    """
    Write an OFF format string of a mesh.

    Write an OFF format string of a mesh. The format is the Object File Format, see `https://en.wikipedia.org/wiki/OFF_(file_format)` for details.


    Parameters
    ----------
    vertex_coords: numpy array of floats
        A 2D array containing 3 coordinates for each vertex. Dimension is (n, 3) for n vertices.

    faces: numpy array of integers
        A 2D array containing 3 vertex indices per face. Dimension is (m, 3) for m faces.

    Returns
    -------
    string
        The OFF string for the mesh.
    """
    verts_rep = _off_verts(vertex_coords)
    faces_rep = _off_faces(faces)
    hdr = "OFF\n"
    counts =  "%d %d %d\n" % (vertex_coords.shape[0], faces.shape[0], 0)
    return ''.join([hdr, counts, verts_rep, faces_rep])


def _obj_verts(vertex_coords):
    """
    Return a string representing the vertices in OBJ format.

    Parameters
    ----------
    vertex_coords: numpy array of floats
        A 2D array containing 3 coordinates for each vertex. Dimension is (n, 3) for n vertices.

    Returns
    -------
    string
        The OBJ format string for the vertices.
    """
    if vertex_coords.shape[0] == 0:
        return ''
    vert_reps = ["v %f %f %f" % (v[0], v[1], v[2]) for v in vertex_coords]                      # x, y, z coords
    return '\n'.join(vert_reps) + '\n'


def _obj_faces(faces):
    """
    Return a string representing the faces in OBJ format.

    Parameters
    ----------
    faces: numpy array of integers
        A 2D array containing 3 vertex indices per face. Dimension is (m, 3) for m faces.

    Returns
    -------
    string
        The OBJ format string for the faces.
    """
    if faces.shape[0] == 0:
        return ''
    face_reps = ["f %d %d %d" % (f[0]+1, f[1]+1, f[2]+1) for f in faces]                    # the 3 vertex indices defining the face, starting at 1
    return '\n'.join(face_reps) + '\n'


def _off_verts(vertex_coords):
    """
    Return a string representing the vertices in OFF format.

    Parameters
    ----------
    vertex_coords: numpy array of floats
        A 2D array containing 3 coordinates for each vertex. Dimension is (n, 3) for n vertices.

    Returns
    -------
    string
        The OFF format string for the vertices.
    """
    if vertex_coords.shape[0] == 0:
        return ''
    vert_reps = ["%f %f %f" % (v[0], v[1], v[2]) for v in vertex_coords]                      # x, y, z coords
    return '\n'.join(vert_reps) + '\n'


def _off_faces(faces):
    """
    Return a string representing the faces in OFF format.

    Parameters
    ----------
    faces: numpy array of integers
        A 2D array containing 3 vertex indices per face. Dimension is (m, 3) for m faces.

    Returns
    -------
    string
        The OFF format string for the faces.
    """
    if faces.shape[0] == 0:
        return ''
    face_reps = ["3 %d %d %d" % (f[0], f[1], f[2]) for f in faces]         # vertex count and the 3 vertex indices defining the face, starting at 0
    return '\n'.join(face_reps) + '\n'



def mesh_to_ply(vertex_coords, faces, vertex_colors=None):
    """
    Write a PLY format string of a mesh.

    Write a PLY format string of a mesh. See http://paulbourke.net/dataformats/ply/ for details on the format.

    Parameters
    ----------
    vertex_coords: numpy array of floats
        A 2D array containing 3 coordinates for each vertex. Dimension is (n, 3) for n vertices.

    faces: numpy array of integers
        A 2D array containing 3 vertex indices per face. Dimension is (m, 3) for m faces.

    vertex_colors: numpy array or None, optional
        A 2D array with shape (n, 4) assigning a color to each vertex (for the n vertices in vertex_coords). The 4 values in each column define the 4 channels of an RGBA color. Channel values should be given as integers in range 0..255. If omitted, no vertex colors will be included in the PLY format string.

    Returns
    -------
    string
        The PLY format string for the mesh.
    """
    use_vertex_colors = vertex_colors is not None
    num_vertices = vertex_coords.shape[0]
    num_faces = faces.shape[0]
    hdr = _ply_header(num_vertices, num_faces, use_vertex_colors=use_vertex_colors)
    verts_rep = _ply_verts(vertex_coords, vertex_colors=vertex_colors)
    faces_rep = _ply_faces(faces)
    return ''.join([hdr, verts_rep, faces_rep])


def _ply_header(num_vertices, num_faces, use_vertex_colors=False):
    """
    Return a string representing the PLY format header for the given data properties.
    """
    hdr_top = """ply
format ascii 1.0
comment Generated by Brainload
"""

    hdr_verts = """element vertex %d
property float x
property float y
property float z
""" % num_vertices

    hdr_vertex_colors = """property uchar red
property uchar green
property uchar blue
property uchar alpha
"""

    hdr_face = "element face %d\nproperty list uchar int vertex_indices\n" % num_faces
    hdr_end = "end_header\n"

    if use_vertex_colors:
        hdr_elements = [hdr_top, hdr_verts, hdr_vertex_colors, hdr_face, hdr_end]
    else:
        hdr_elements = [hdr_top, hdr_verts, hdr_face, hdr_end]
    return ''.join(hdr_elements)


def _ply_verts(vertex_coords, vertex_colors=None):
    """
    Return a string representing the vertices in PLY format. Vertex colors are optional.
    """
    if vertex_coords.shape[0] == 0:
        return ''
    vert_reps = ["%f %f %f" % (v[0], v[1], v[2]) for v in vertex_coords]                      # x, y, z coords
    if vertex_colors is not None:
        col_reps = [" %d %d %d %d" % (c[0], c[1], c[2], c[3]) for c in vertex_colors]      # RGBA values of color
        vert_col_reps = [i+j for i,j in zip(vert_reps, col_reps)]
        return '\n'.join(vert_col_reps) + '\n'
    else:
        return '\n'.join(vert_reps) + '\n'


def _ply_faces(faces):
    """
    Return a string representing the faces in PLY format.
    """
    if faces.shape[0] == 0:
        return ''
    face_reps = ["3 %d %d %d" % (f[0], f[1], f[2]) for f in faces]                    # the 3 vertex indices defining the face
    return '\n'.join(face_reps) + '\n'


def scalars_to_colors_matplotlib(data, matplotlib_cmap_name='viridis', data_normalization='linear', custom_cmap=None, scale=True):
    """
    Assign colors to scalars using a colormap from matplotlib.

    Assign colors to scalars using functions and a colormap from matplotlib. This requires matplotlib to be installed, which is NOT a hard dependency of brainload. If you want to use this function, you need to install matplotlib.

    Parameters
    ----------
    data: 1D numpy array of numerical data, length n.
        The scalars data, each data point will be assigned a color.

    matplotlib_cmap_name: string
        A valid name of a matplotlib colormap. Example: 'Spectral'. Note that it is important to chose the color map based on the data and your application. For sequential data, try 'viridis' or 'plasma'. For diverging data, try 'Spectral' or 'coolwarm'. For qualitative color maps, try 'tab10' or 'tab20'. See https://matplotlib.org/users/colormaps.html for details. If the parameter custom_cmap is given, this can be a freeform name for your that colormap. Defaults to 'viridis'.

    data_normalization: string, one of ('linear', 'log'), optional
        How the data should be normalized to match the range of the color map. Defaults to 'linear'.

    custom_cmap: matplotlib colormap instance, optional
        A custom matplotlib colormap, e.g., one created using LinearSegmentedColormap.from_list() or other matplotlib functions. Optional. If given, takes precedence over matplotlib_cmap_name.

    scale: boolean
        Whether to scale the returned color values to the range 1..255. If False, the colors will be in range 0..1. Defaults to TRUE.

    Returns
    -------
    numpy float array of shape (n, 4)
        An array that assigns one RGBA color to each value from the scalars parameter (use the index). A color is given as 4 floats (RGBA), each in range 0.0 to 1.0 or 1 to 255, depending on the parameter 'scale'.
    """
    if data_normalization not in ('linear', 'log'):
        raise ValueError("ERROR: data_normalization must be one of {'linear', 'log'} but is '%s'." % data_normalization)

    data_min = np.min(data)
    data_max = np.max(data)

    try:
        import matplotlib.cm as mpl_cm
        import matplotlib.colors as mpl_colors
    except:
        raise ImportError('The package matplotlib is not installed. While matplotlib is not a hard dependency of Brainload, you need to have it installed if you use the scalars_to_colors_matplotlib function.')

    if custom_cmap is None:
        cmap = mpl_cm.get_cmap(name=matplotlib_cmap_name)
    else:
        cmap = custom_cmap

    if data_normalization == 'linear':
        norm = mpl_colors.Normalize(vmin=data_min, vmax=data_max)
    else:
        norm = mpl_colors.LogNorm(vmin=data_min, vmax=data_max)

    num_scalars = data.shape[0]
    assigned_colors = np.zeros((num_scalars, 4), dtype=np.float_)

    assigned_colors = cmap(norm(data))
    if scale:
        assigned_colors = np.round(assigned_colors * 255.0)   # matplotlib colors are in range 0. to 1., transform to 0 to 255
    return assigned_colors


def _normalize_to_range_zero_one(data):
    """
    Normalize the given data to the range [0, 1].

    Normalize the given data to the range [0, 1] in a linear fashion. If the given data is constant (i.e, all values in the array are identical), maps all values to 1.0.

    Parameters
    ----------
    numpy array
        1D numpy array of numerical data, length n.

    Returns
    -------
    numpy array
        The normalized data: 1D numpy array of floats in range [0, 1] with length n. If the given data is constant (i.e, all values in the array are identical), all values in the array are 1.0.
    """
    data=np.array(data)
    if np.unique(data).shape[0]==1:
        return np.ones(data.shape)
    else:
        return (data - np.min(data)) / np.ptp(data)


def scalars_to_colors_clist(scalars, color_list):
    """
    Given scalar values and a color list, assign a color to each scalar value.

    Given scalar values and a color list, assign a color to each scalar value. This is useful for exporting vertex colored brain meshes.

    Parameters
    ----------
    scalars: scalar numpy array of shape (i, ).
        1D array of i numerical scalar values, usually floats.

    cmap: numpy array of shape (n, m)
        Array containing n colors, each of which is defined by m values (e.g., m=3 for RGB colors, m=4 for RGBA colors, but this function does not care for the meaning in any way).

    Returns
    -------
    numpy int array of shape (i, m)
        An array that assigns one color to each value from the scalars parameter (use the index).
    """
    num_colors = color_list.shape[0]
    num_color_channels = color_list.shape[1]      # 3 or 4, depending on whether an alpha channel is included
    scalars = np.array(scalars)
    norm_scalars = _normalize_to_range_zero_one(scalars)
    min_scalar = np.min(norm_scalars) # 0.0
    max_scalar = np.max(norm_scalars) # 1.0
    num_scalars = norm_scalars.shape[0]
    assigned_colors = np.zeros((num_scalars, num_color_channels))

    it = np.nditer(norm_scalars, flags=['f_index'])
    while not it.finished:
        assigned_colors[it.index][:] = _color_from_clist(it[0], color_list)
        it.iternext()
    return assigned_colors


def _color_from_clist(scalar_in_range_zero_to_one, color_list):
    """
    Given a scalar value between 0.0 and 1.0, return its entry from a color list.

    Given a scalar value between 0.0 and 1.0, return its entry from a color list. Note that you can normalize your data to the range [0, 1] in whatever way you see fit (linear, log) before calling this function.

    Parameters
    ----------
    scalar_in_range_zero_to_one: float
        A scalar value in range 0.0 to 1.0. It is assumed that your data array (the source of this single value) has been normalized to the range already.

    color_list: numpy array of shape (n, m)
        An array defining n colors. Each color is given by m channel values. Note: `m` could be 4 and refer to the RGBA channels, but this function does not care.

    """
    num_colors = color_list.shape[0]
    color_list_index = _color_index_from_clist(scalar_in_range_zero_to_one, num_colors)
    return color_list[color_list_index][:]


def _get_example_colorlist(n=256):
    """
    Return an example color list.

    Return an example color list with n values. Each colors is given as 4 integers in range 0 - 255, representing the four RGBA channels. This is a toy map that manipulates colors in RGB space, which is not a good idea and will not produce perceptually linear colors. You may want to look into other color spaces to create your own map.

    Parameters
    ----------
    int
        The number of colors you want.

    Returns
    -------
    numpy array of ints, dimension (n, 4)
        The color list.
    """
    cmap = np.zeros((n, 4), dtype=int)
    cmap[:,0] = np.arange(n)
    cmap[:,1] = np.arange(n)
    cmap[:][2] = 150
    cmap[:][3] = 255    # always full alpha
    return cmap


def _color_index_from_clist(scalar_in_range_zero_to_one, num_colors):
    """
    Given a scalar value between 0.0 and 1.0, return the index of the color in the color list for that value.

    Parameters
    ----------
    scalar_in_range_zero_to_one: float
        A scalar value in range 0.0 to 1.0. It is assumed that your data array (the source of this single value) has been normalized to the range already.

    num_colors: int
        The number of colors in your color list.

    Returns
    -------
    int: the index into the color map
    """
    min_scalar = 0.0
    max_scalar = 1.0
    if scalar_in_range_zero_to_one < min_scalar:
        return 0       # assign first color in color map
    elif scalar_in_range_zero_to_one > max_scalar:
        return num_colors - 1      # assign last color in color map
    else:
        color_list_index = int(np.floor(num_colors * ((scalar_in_range_zero_to_one - min_scalar) / (max_scalar - min_scalar))))
        if color_list_index >= num_colors:
            color_list_index = num_colors - 1
        return color_list_index
