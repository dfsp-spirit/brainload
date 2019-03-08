#!/usr/bin/env python
from __future__ import print_function
import sys
import numpy as np
import nibabel as nib
import argparse

# To run this in dev mode (in virtual env, pip -e install of brainview active) from REPO_ROOT:
# PYTHONPATH=./src/brainload python src/brainload/vol_info.py tests/test_data/subject1/mri/orig.mgz --crs 10 10 10 -v

def vol_info():
    """
    Brain volume information.

    Simple script to query data from a brain volume, based on voxel CRS.
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Query brain volume data.")
    parser.add_argument("volume", help="The volume file to load. Should be in mgh, mgz or nifti format.")
    parser.add_argument('-c', '--crs', nargs='*', help="The query voxel, defined as a 0-based index into the volume. For a 3D volume, this would be 3 integers which represent the CRS (column, row, slice) of the voxel.")
    parser.add_argument("-v", "--verbose", help="Increase output verbosity.", action="store_true")
    args = parser.parse_args()

    volume_file = args.volume
    voxel_index = tuple([int(x) for x in args.crs])
    verbose = args.verbose

    vol_data = nib.load(volume_file).get_data()
    if verbose:
        print("Volume has %d dimensions, shape %s and data type %s." % (len(vol_data.shape), vol_data.shape, vol_data.dtype))

    voxel_value_print_format = "%f"
    if np.issubdtype(vol_data.dtype, np.integer):
        voxel_value_print_format = "%d"

    print(voxel_value_print_format % (vol_data[voxel_index]))
    sys.exit(0)


if __name__ == "__main__":
    vol_info()
