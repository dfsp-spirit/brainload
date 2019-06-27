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


    epilog_example_text = '''examples:
 brain_morph_info ~/studyA/subject1/surf/lh.area -i 10 --verbose
 brain_morph_info ~/studyA/subject1/surf/lh.thickness -a -q describe
 brain_morph_info ~/studyA/subject1/surf/rh.curv -f ~/vertices_of_interest.txt -q sortasc'''

    parser = argparse.ArgumentParser(description="Query brain surface morphometry data from a curv file.", epilog=epilog_example_text, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("curv_file", help="The curv file to load. Must be in Freesurfer curv format. Example files are 'lh.area' or 'rh.thickness'.")
    index_group = parser.add_mutually_exclusive_group(required=True)
    index_group.add_argument("-i", "--index", help="The index of the vertex to query. A single integer or several integers separated by commata (no spaces allowed).")
    index_group.add_argument("-f", "--index-file", help="A file containing the list of vertex indices to query.")
    index_group.add_argument("-a", "--all", help="Select all vertices.", action="store_true")
    parser.add_argument("-q", "--query", help="The query action to perform. Defaults to 'values'.", default="values", choices=['values', 'describe', 'sortasc', 'sortdsc'])
    parser.add_argument("-s", "--separator", help="Output separator (between vertex coords / indices). Defaults to ','.", default=",")
    parser.add_argument("-v", "--verbose", help="Increase output verbosity.", action="store_true")
    args = parser.parse_args()

    curv_file = args.curv_file
    verbose = args.verbose
    sep = args.separator

    if verbose:
        print("---Brain Surface Morphometry Info---")

    hemisphere_label, is_default = fsd._deduce_hemisphere_label_from_file_path(curv_file)
    morphometry_data, meta_data = fsd.read_fs_morphometry_data_file_and_record_meta_data(curv_file, hemisphere_label)

    if args.index:
        query_indices = np.array([int(s) for s in args.index.split(',')], dtype=int)
        if verbose:
            print("Querying curv file for the %d vertex indices from the command line." % (query_indices.shape[0]))
    elif args.index_file:
        query_indices = nit.load_vertex_indices(args.index_file)
        if verbose:
            print("Querying curv file for the %d vertex indices from file '%s'. (File should contain indices separated by '%s'.)" % (query_indices.shape[0], args.index_file, args.separator))
    else:
        if verbose:
            print("Querying curv file for all its %d vertex indices." % (morphometry_data.shape[0]))
        query_indices = np.arange(morphometry_data.shape[0])

    if verbose:
        print("Morphometry file '%s' contains %d values." % (curv_file, morphometry_data.shape[0]))

    d = morphometry_data[query_indices]
    if args.query == "values":
        res = sep.join(str(x) for x in d)
        if verbose:
            print("Morphometry values of vertices # %s are: %s" % ([str(x) for x in query_indices], res))
        else:
            print(res)
    elif args.query == "sortasc":
        if verbose:
            print("Printing vertex indices and their data values in ascending order.")
        sorted_indices = np.argsort(d)
        for idx in sorted_indices:
            print("%d, %f" % (idx, d[idx]))
    elif args.query == "sortdsc":
        if verbose:
            print("Printing vertex indices and their data values in descending order.")
        sorted_indices = np.argsort(-d)
        for idx in sorted_indices:
            print("%d, %f" % (idx, d[idx]))
    else:   # descriptive stats
        if verbose:
            print("count, mean, median, .25 quantile, .75 quantile, min, max")
        print("%d, %f, %f, %f, %f, %f, %f" % (d.shape[0], np.mean(d), np.median(d), np.quantile(d, 0.25), np.quantile(d, 0.75), np.min(d), np.max(d)))

    sys.exit(0)


if __name__ == "__main__":
    brain_morph_info()
