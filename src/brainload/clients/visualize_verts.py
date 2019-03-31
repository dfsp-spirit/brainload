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
    num_vert_group.add_argument("-n", "--num-verts", help="The number of vertices in the target surface, an integer. Hint: the number for the Freesurfer fsaverage subject is 163842.")
    num_vert_group.add_argument("-t", "--target-surface-file", help="A target surface file to determine the number of vertices from. E.g., '<your_study>/subject1/surf/lh.white'.")
    index_group = parser.add_mutually_exclusive_group(required=True)
    index_group.add_argument("-i", "--index", help="The index of the vertex to query. A single integer or several integers separated by commata (no spaces allowed).")
    index_group.add_argument("-f", "--index-file", help="A file containing a list of vertex indices, one integer per line. Can optionally contain colors per vertex, then add 3 more integers per line, separated by commata. Example line: '0,255,0,0'")
    parser.add_argument("-v", "--verbose", help="Increase output verbosity.", action="store_true")
    parser.add_argument('-c', '--color', nargs=3, help="The color to use for the vertices as 3 RGB values between 0 and 255, e.g., '255 0 0' for red. Must be given unless an index file is used that contains the color values.", default=None)
    parser.add_argument('-b', '--background-color', nargs=3, help="The background color to use for all the vertices which are NOT listed on the command line or in the index file. 3 RGB values between 0 and 255. Defaults to '128 128 128', a gray.", default=[128, 128, 128])
    parser.add_argument('-e', '--extend-neighborhood', nargs=2, help="In addition to the given vertices, also color their neighbors in the given mesh file, up to the given graph distance in hops.")
    parser.add_argument('-o', '--output-file', help="Ouput file. The format is an RGB overlay that can be loaded into Freeview.", default="surface_RGB_map.txt")
    args = parser.parse_args()

    verbose = args.verbose

    if verbose:
        print("---Brain Surface Vertices Visualization---")


    colors = None

    if args.index:
        query_indices = np.array([int(s) for s in args.index.split(',')], dtype=int)
        if verbose:
            print("Using the %d vertex indices from the command line." % (query_indices.shape[0]))
    else:
        query_indices = nit.load_vertex_indices(args.index_file)
        # The vertex index file may not only contain either only a vertex index per line, or 3 additional RGB values. Parse colors if it does have them:
        if len(query_indices.shape) == 2:
            num_fields_per_row = query_indices.shape[1]
            if num_fields_per_row < 4:
                print("ERROR: Lines in the vertex index must must contain either a single integer (the vertex index), or at least four (the index and the 3 RGB color values). Found %d." % (num_fields_per_row))
            colors = query_indices[:,1:4]
            query_indices = query_indices[:,0]
            if verbose:
                if args.color:
                    print("Using the %d vertex indices but ignoring the colors from file '%s'. Command line -c takes precedence for color." % (query_indices.shape[0], args.index_file))
                else:
                    print("Using the %d vertex indices and colors from file '%s'." % (query_indices.shape[0], args.index_file))
        else:
            if verbose:
                print("Using the %d vertex indices from file '%s'." % (query_indices.shape[0], args.index_file))

    if not args.color and colors is None:
        print("ERROR: No foreground color given on the command line (-c) and vertex index file contains no color values. Use -c or add colors to vertex index file.")

    if args.num_verts:
        num_verts = int(args.num_verts)
    else:
        mesh_file = args.target_surface_file
        vert_coords, faces = fsio.read_geometry(mesh_file)
        num_verts = vert_coords.shape[0]

    background_color = [int(x) for x in args.background_color]

    if args.extend_neighborhood:
        extend_mesh_file = args.extend_neighborhood[0]
        extend_num_hops = int(args.extend_neighborhood[1])
        print("Extension of neighborhood by %d requested based on surface mesh file '%s'. Computing surface graph." % (extend_num_hops, extend_mesh_file))
        import brainload.surfacegraph as sg
        hemi_label, is_default = fsd._deduce_hemisphere_label_from_file_path(extend_mesh_file)
        print("Hemi label is '%s' based on file '%s'." % (hemi_label, extend_mesh_file))
        vert_coords, faces, meta_data = fsd.read_fs_surface_file_and_record_meta_data(extend_mesh_file, hemi_label)
        mesh_graph = sg.SurfaceGraph(vert_coords, faces)


    if args.color:
        color_all = [int(x) for x in args.color]
        if verbose:
            print("Using foreground color %s for all %d foreground vertices." % (" ".join(args.color), query_indices.shape[0]))
        vertex_mark_list = []
        for i, vertex in enumerate(query_indices):
            vertex_mark_list.append(([query_indices[i]], color_all))

    else:
        if verbose:
            print("No foreground color given on command line, using per-vertex colors from vertex index file.")
        vertex_mark_list = []
        for i, vertex in enumerate(query_indices):
            vertex_mark_list.append(([query_indices[i]], colors[i]))

    if args.extend_neighborhood:
        for t_idx, mark_tuple in enumerate(vertex_mark_list):
            tuple_vert_indices = mark_tuple[0]
            tuple_colors = mark_tuple[1]
            for marked_vertex_index in tuple_vert_indices:
                neighbors = mesh_graph.get_neighbors_up_to_dist(marked_vertex_index, extend_num_hops)
                new_mark_tuple_verts = np.concatenate((np.array(mark_tuple[0], dtype=int), np.array(neighbors, dtype=int)))
                new_mark_tuple = (new_mark_tuple_verts, tuple_colors)
                vertex_mark_list[t_idx] = new_mark_tuple

    if verbose:
        all_foreground_verts = np.empty((0,), dtype=int)
        for mark_tuple in vertex_mark_list:
            all_foreground_verts = np.concatenate((all_foreground_verts, mark_tuple[0]))
        num_foreground_verts = all_foreground_verts.shape[0]
        num_unique_foreground_verts = np.unique(all_foreground_verts).shape[0]
        print("Resulting surface RGB map contains %d marked vertices (%d unique)." % (num_foreground_verts, num_unique_foreground_verts))

    lines = brw.get_surface_vertices_overlay_text_file_lines(num_verts, vertex_mark_list, background_rgb=background_color)

    print("Writing surface RGB map to output file '%s'. To use it, load and select a surface in Freeview, then click 'Color -> Load RGB Map'." % (args.output_file))
    nit.write_lines_to_text_file(lines, args.output_file)
    sys.exit(0)


if __name__ == "__main__":
    visualize_verts()
