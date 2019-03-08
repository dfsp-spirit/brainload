#!/usr/bin/env python
from __future__ import print_function
import sys
import numpy as np
import nibabel.freesurfer.io as fsio
import argparse

# To run this in dev mode (in virtual env, pip -e install of brainload active) from REPO_ROOT:
# PYTHONPATH=./src/brainload python src/brainload/brain_mesh_info.py tests/test_data/subject1/surf/lh.white 10 -v

def brain_mesh_info():
    """
    Brain surface mesh information.

    Simple script to query data from a brain surface mesh.
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Query brain surface mesh data.")
    parser.add_argument("mesh", help="The surface mesh file to load. Must be in Freesurfer geometry format. Example files are 'lh.white' or 'rh.white'.")
    parser.add_argument("index", help="The index of the element to query (vertex index or face index, depending on mode).")
    parser.add_argument("-m", "--mode", help="The query mode, i.e., whether you want to query information on a 'vertex' or a 'face'.", default="vertex", choices=['vertex', 'face'])
    parser.add_argument("-s", "--separator", help="Output separator (between vertex coords / indices).", default=" ")
    parser.add_argument("-v", "--verbose", help="Increase output verbosity.", action="store_true")
    args = parser.parse_args()

    mesh_file = args.mesh
    query_index = int(args.index)
    query_mode = args.mode
    verbose = args.verbose
    sep = args.separator

    vert_coords, faces = fsio.read_geometry(mesh_file)
    if verbose:
        print("Mesh has %d vertices and %d faces. Mode is '%s'." % (vert_coords.shape[0], faces.shape[0], query_mode))

    if query_mode == "vertex":
        res = sep.join(str(x) for x in vert_coords[query_index,:])
        if verbose:
            print("Coords of vertex # %d are: %s" % (query_index, res))
        else:
            print(res)
    else:
        res = sep.join(str(x) for x in faces[query_index,:])
        if verbose:
            print("Vertices forming face # %d are: %s" % (query_index, res))
        else:
            print(res)

    sys.exit(0)


if __name__ == "__main__":
    brain_mesh_info()
