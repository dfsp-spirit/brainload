#!/usr/bin/env python
from __future__ import print_function
import sys
import numpy as np
import nibabel.freesurfer.io as fsio
import brainload.nitools as nit
import brainload.freesurferdata as fsd
import brainload.brainwrite as brw
import argparse

# To run this in dev mode (in virtual env, pip -e install of brainload active) from REPO_ROOT:
# PYTHONPATH=./src/brainload python src/brainload/clients/visualize_verts.py tests/test_data/subject1/surf/lh.area -i 10 -v

def visualize_verts():
    """
    Visualize brain surface vertices.

    Generate a text file overlay that can be read by freeview that colors the given vertices.
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate files to visualize vertices in Freeview.")
    num_vert_group = parser.add_mutually_exclusive_group(required=True)     # we need some way to determine the number of vertices to use for the overlay
    num_vert_group.add_argument("-n", "--num-verts", help="The number of vertices in the target surface, an integer. Defaults to 163842, the number for the Freesurfer fsaverage subject.", default="163842")
    num_vert_group.add_argument("-t", "--target-surface-file", help="The target surface file to use for determining the number of vertices automatically.")
    index_group = parser.add_mutually_exclusive_group(required=True)
    index_group.add_argument("-i", "--index", help="The index of the vertex to query. A single integer or several integers separated by commata (no spaces allowed).")
    index_group.add_argument("-f", "--index-file", help="A file containing a list of vertex indices.")
    parser.add_argument("-v", "--verbose", help="Increase output verbosity.", action="store_true")
    parser.add_argument('-c', '--color', nargs=3, help="The color to use for the vertices as 3 RGB values between 0 and 255, e.g., '255 0 0' for red. Must be given unless an index file is used that contains the color values.", default=None)
    parser.add_argument('-b', '--background-color', nargs=3, help="The background color to use for all the vertices which are NOT listed on the command line or in the index file. 3 RGB values between 0 and 255. Defaults to '128 128 128', a gray.", default="128 128 128")
    args = parser.parse_args()

    verbose = args.verbose

    if verbose:
        print("---Brain Surface Vertices Visualization---")

    if args.index:
        query_indices = np.array([int(s) for s in args.index.split(',')], dtype=int)
        if verbose:
            print("Using the %d vertex indices from the command line." % (query_indices.shape[0]))
    else:
        query_indices = nit.load_vertex_indices(args.index_file)
        if verbose:
            print("Using the %d vertex indices from file '%s'. (File should contain indices separated by '%s'.)" % (query_indices.shape[0], args.index_file, args.separator))

    if args.num_verts:
        num_verts = int(args.num_verts)
    else:
        mesh_file = args.target_surface_file
        vert_coords, faces = fsio.read_geometry(mesh_file)
        num_verts = vert_coords.shape[0]

    background_color = [int(x) for x in args.background_color]

    if args.color:
        color_all = [int(x) for x in args.color]
        if verbose:
            print("Using color %s for all %d foreground vertices." % (" ".join(args.color), query_indices.shape[0]))
        vertex_mark_list = [(query_indices, color_all)]
    else:
        if verbose:
            print("No color given on command line, parsing it from vertex index file.")

    lines = brw.get_surface_vertices_overlay_text_file_lines(num_verts, vertex_mark_list, background_rgb=background_color)



    sys.exit(0)


if __name__ == "__main__":
    visualize_verts()
