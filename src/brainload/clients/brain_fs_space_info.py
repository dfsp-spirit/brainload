#!/usr/bin/env python
from __future__ import print_function
import sys
import numpy as np
import brainload.freesurferdata as fsd
import brainload.spatial as sp
import argparse

# To run this in dev mode (in virtual env, pip -e install of brainload active) from REPO_ROOT:
# PYTHONPATH=./src/brainload python src/brainload/brain_fs_space_info.py tests/test_data/subject1/mri/orig.mgz -l 10 10 10 --apply-vox2ras

def brain_fs_space_info():
    """
    Brain FreeSurfer volume file space information.

    Simple script to query data from a FreeSurfer format brain volume file with transform information in the header.
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Query brain space information from a FreeSurfer volume data file.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-l", "--location", nargs=3, help="The location to which the matrix should be applied. Must be a RAS coord or a voxel CRS in the 3D volume (three digits).")
    input_group.add_argument("-f", "--location-file", help="A file that contains multiple locations (one per line, the three digits separated by spaces within the line).")
    matrix_to_apply_group = parser.add_mutually_exclusive_group(required=True)
    matrix_to_apply_group.add_argument('-r', '--ras2vox-from-vol', help="Use ras2vox matrix from the header of the given mgh or mgz format file.")
    matrix_to_apply_group.add_argument('-o', '--vox2ras-from-vol', help="Use vox2ras matrix from the header of the given mgh or mgz format file.")
    matrix_to_apply_group.add_argument('-t', '--vox2ras-tkr-from-vol', help="Use ras2vox-tkr matrix from the header of the given mgh or mgz format file.")
    parser.add_argument("-s", "--separator", help="Output separator (between vertex coords / indices).", default=" ")
    parser.add_argument("-i", "--inverse-matrix", help="Inverse the matrix before applying it.", action="store_true")
    parser.add_argument("-c", "--round-output", help="Round output to closest integer. (Useful when result is a voxel CRS.)", action="store_true")
    parser.add_argument("-v", "--verbose", help="Increase output verbosity.", action="store_true")
    args = parser.parse_args()

    if args.location:
        location = tuple([float(x) for x in args.location])
        locations = np.array([location])
    else:
        pass

    volume_file = args.volume
    verbose = args.verbose
    sep = args.separator

    vol_data, mgh_meta_data = fsd.read_mgh_file(volume_file)
    m_ras2vox = mgh_meta_data['ras2vox']
    m_vox2ras = mgh_meta_data['vox2ras']
    m_vox2ras_tkr = mgh_meta_data['vox2ras_tkr']

    if verbose:
        print("Volume has %d dimensions, shape %s and data type %s." % (len(vol_data.shape), vol_data.shape, vol_data.dtype))

    if args.apply_ras2vox:
        location = tuple([float(x) for x in args.apply_ras2vox])
        if verbose:
            print("Applying ras2vox to %s." % str(location))
        res_matrix = np.rint(sp.apply_affine_3D(np.array([location]), m_ras2vox)).astype(int)
        for row in res_matrix:
            res = sep.join(str(x) for x in row)
            print(res)

    if args.apply_vox2ras:
        location = tuple([int(x) for x in args.apply_vox2ras])
        if verbose:
            print("Applying vox2ras to %s." % str(location))
        res_matrix = sp.apply_affine_3D(np.array([location]), m_vox2ras)
        for row in res_matrix:
            res = sep.join(str(x) for x in row)
            print(res)

    if args.apply_vox2ras_tkr:
        location = tuple([int(x) for x in args.apply_vox2ras_tkr])
        if verbose:
            print("Applying vox2ras_tkr to %s." % str(location))
        res_matrix = sp.apply_affine_3D(np.array([location]), m_vox2ras_tkr)
        for row in res_matrix:
            res = sep.join(str(x) for x in row)
            print(res)

    sys.exit(0)


if __name__ == "__main__":
    brain_fs_space_info()
