"""
Export functions for brainload. In contrast to the brainview functions, these do not support color.

These functions allow one to export brain meshes, e.g., for loading into standard 3D modeling software.
"""

import brainload as bl
import os
import brainload.meshexport as me
import numpy as np


def export_mesh_nocolor_to_file(filename, vertex_coords, faces):
    export_format, matched = _mesh_export_nc_format_from_filename(filename)
    export_string = _get_export_nc_string(export_format, vertex_coords, faces)

    with open(filename, "w") as text_file:
        text_file.write(export_string)


def _get_export_nc_string(export_format, vertex_coords, faces):
    if export_format not in ('obj', 'ply'):
        raise ValueError("ERROR: export_format must be one of {'obj', 'ply'} but is '%s'." % export_format)

    if export_format == 'obj':
        return bl.mesh_to_obj(vertex_coords, faces)
    else:
        return bl.mesh_to_ply(vertex_coords, faces)



def _mesh_export_nc_format_from_filename(filename):
    """
    Determine a mesh output format based on a file name.

    Determine a mesh output format based on a file name. This inspects the file extension.

    Parameters
    ----------
    filename: string
        A file name, may start with a full path. Examples: 'brain.obj' or '/home/myuser/export/brain.ply'. If the file extension is not a recognized extension for a supported format, the default format 'obj' is returned.

    Returns
    -------
    format: string
        A string defining a supported mesh output format. One of ('ply', 'obj').

    matched: Boolean
        Whether the file name ended with a known extension. If not, the returned format was chosen because it is the default format.
    """
    if filename.endswith('.ply'):
        return 'ply', True
    elif filename.endswith('.obj'):
        return 'obj', True
    else:
        return 'obj', False
