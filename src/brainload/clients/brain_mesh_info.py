#!/usr/bin/env python
from __future__ import print_function
import sys
import numpy as np
import nibabel.freesurfer.io as fsio
import brainload.nitools as nit
import argparse

# To run this in dev mode (in virtual env, pip -e install of brainload active) from REPO_ROOT:
# PYTHONPATH=./src/brainload python src/brainload/clients/brain_mesh_info.py tests/test_data/subject1/surf/lh.white -i 10 -v

def brain_mesh_info():
    """
    Brain surface mesh information.

    Simple script to query data from a brain surface mesh.
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Query brain surface mesh data.")
    parser.add_argument("mesh", help="The surface mesh file to load. Must be in Freesurfer geometry format. Example files are 'lh.white' or 'rh.white'.")
    index_group = parser.add_mutually_exclusive_group(required=True)
    index_group.add_argument("-i", "--index", help="The index of the element to query (vertex index or face index, depending on mode). A single integer or several integers separated by commata (no spaces allowed).")
    index_group.add_argument("-f", "--index-file", help="A file containing a list of the indices of the element to query (vertex indices or face indices, depending on mode). All indices on a single line, separated by nothing but ',' (no spaces allowed).")
    parser.add_argument("-m", "--mode", help="The query mode, i.e., whether you want to query information on a 'vertex' or a 'face'.", default="vertex", choices=['vertex', 'face'])
    parser.add_argument("-s", "--separator", help="Output separator (between vertex coords / indices). Defaults to ','.", default=",")
    parser.add_argument("-v", "--verbose", help="Increase output verbosity.", action="store_true")
    args = parser.parse_args()

    mesh_file = args.mesh
    query_mode = args.mode
    verbose = args.verbose
    sep = args.separator

    if verbose:
        print("---Brain Mesh Info---")

    if args.index:
        query_indices = np.array([int(s) for s in args.index.split(',')], dtype=int)
        if verbose:
            print("Querying mesh for the %d %s indices from the command line." % (query_indices.shape[0], args.mode))
    else:
        query_indices = nit.load_vertex_indices(args.index_file)
        if verbose:
            print("Querying mesh for the %d %s indices from file '%s'. (File should contain indices separated by '%s'.)" % (query_indices.shape[0], args.mode, args.index_file, args.separator))


    vert_coords, faces = fsio.read_geometry(mesh_file)
    if verbose:
        print("Mesh from file '%s' has %d vertices and %d faces." % (mesh_file, vert_coords.shape[0], faces.shape[0]))

    if query_mode == "vertex":
        res = sep.join(str(x) for x in vert_coords[query_indices,:])
        if verbose:
            print("Coords of vertices # %s are: %s" % ([str(x) for x in query_indices], res))
        else:
            print(res)
    else:
        res = sep.join(str(x) for x in faces[query_indices,:])
        if verbose:
            print("Vertices forming faces # %s are: %s" % ([str(x) for x in query_indices], res))
        else:
            print(res)

    sys.exit(0)


if __name__ == "__main__":
    brain_mesh_info()
