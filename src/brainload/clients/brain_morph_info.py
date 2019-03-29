#!/usr/bin/env python
from __future__ import print_function
import sys
import numpy as np
import nibabel.freesurfer.io as fsio
import brainload.nitools as nit
import brainload.freesurferdata as fsd
import argparse

# To run this in dev mode (in virtual env, pip -e install of brainload active) from REPO_ROOT:
# PYTHONPATH=./src/brainload python src/brainload/clients/brain_morph_info.py tests/test_data/subject1/surf/lh.area -i 10 -v

def brain_morph_info():
    """
    Brain surface morphometry data or ```curv``` file information.

    Simple script to query morphometry data from a curv file that holds a scalar value (area, thickness, etc) for each surface vertex. An example file would be ```surf/lh.area```, but not ```surf/lh.white```.
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Query brain surface morphometry data from a curv file.")
    parser.add_argument("curv_file", help="The curv file to load. Must be in Freesurfer curv format. Example files are 'lh.area' or 'rh.thickness'.")
    index_group = parser.add_mutually_exclusive_group(required=True)
    index_group.add_argument("-i", "--index", help="The index of the vertex to query. A single integer or several integers separated by commata (no spaces allowed).")
    index_group.add_argument("-f", "--index-file", help="A file containing a list of vertex indices.")
    parser.add_argument("-s", "--separator", help="Output separator (between vertex coords / indices). Defaults to ','.", default=",")
    parser.add_argument("-v", "--verbose", help="Increase output verbosity.", action="store_true")
    args = parser.parse_args()

    curv_file = args.curv_file
    verbose = args.verbose
    sep = args.separator

    if verbose:
        print("---Brain Surface Morphometry Info---")

    if args.index:
        query_indices = np.array([int(s) for s in args.index.split(',')], dtype=int)
        if verbose:
            print("Querying curv file for the %d vertex indices from the command line." % (query_indices.shape[0]))
    else:
        query_indices = nit.load_vertex_indices(args.index_file)
        if verbose:
            print("Querying curv file for the %d vertex indices from file '%s'. (File should contain indices separated by '%s'.)" % (query_indices.shape[0], args.index_file, args.separator))

    hemisphere_label, is_default = fsd._deduce_hemisphere_label_from_file_path(curv_file)
    morphometry_data, meta_data = fsd.read_fs_morphometry_data_file_and_record_meta_data(curv_file, hemisphere_label)
    if verbose:
        print("Morphometry file '%s' contains %d values." % (curv_file, morphometry_data.shape[0]))

    res = sep.join(str(x) for x in morphometry_data[query_indices])
    if verbose:
        print("Morphometry values of vertices # %s are: %s" % ([str(x) for x in query_indices], res))
    else:
        print(res)

    sys.exit(0)


if __name__ == "__main__":
    brain_morph_info()
