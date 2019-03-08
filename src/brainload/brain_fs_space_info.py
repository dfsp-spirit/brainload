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
    parser.add_argument("volume", help="The volume file to load. Should be in mgh or mgz format.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-r', '--apply-ras2vox', nargs=3, help="Apply ras2vox matrix from the volume file header to the location. The location should be a voxel CRS (three integers) in this case.")
    group.add_argument('-o', '--apply-vox2ras', nargs=3, help="Apply vox2ras matrix from the volume file header to the location. The location should be a RAS coordinate (three floats) in this case.")
    group.add_argument('-t', '--apply-vox2ras-tkr', nargs=3, help="Apply vox2ras-tkr matrix from the volume file header to the location. The location should be a RAS coordinate (three floats) in this case.")
    parser.add_argument("-s", "--separator", help="Output separator (between vertex coords / indices).", default=" ")
    parser.add_argument("-v", "--verbose", help="Increase output verbosity.", action="store_true")
    args = parser.parse_args()

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
        print("Computing ras2vox for %s." % str(location))
        res_matrix = sp.apply_affine_3D(np.array([location]), m_ras2vox)
        for row in res_matrix:
            res = sep.join(str(x) for x in row)
            print(res)

    if args.apply_vox2ras:
        location = tuple([int(x) for x in args.apply_vox2ras])
        print("Computing vox2ras for %s." % str(location))
        res_matrix = sp.apply_affine_3D(np.array([location]), m_vox2ras)
        for row in res_matrix:
            res = sep.join(str(x) for x in row)
            print(res)

    if args.apply_vox2ras_tkr:
        location = tuple([int(x) for x in args.apply_vox2ras_tkr])
        print("Computing vox2ras_tkr for %s." % str(location))
        res_matrix = sp.apply_affine_3D(np.array([location]), m_vox2ras_tkr)
        for row in res_matrix:
            res = sep.join(str(x) for x in row)
            print(res)

    sys.exit(0)


if __name__ == "__main__":
    brain_fs_space_info()
