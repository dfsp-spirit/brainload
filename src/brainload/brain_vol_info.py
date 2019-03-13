#!/usr/bin/env python
from __future__ import print_function
import sys
import numpy as np
import nibabel as nib
import argparse

# To run this in dev mode (in virtual env, pip -e install of brainload active) from REPO_ROOT:
# PYTHONPATH=./src/brainload python src/brainload/brain_vol_info.py tests/test_data/subject1/mri/orig.mgz --crs 10 10 10 -v

def brain_vol_info():
    """
    Brain volume information.

    Simple script to query data from a brain volume, based on voxel CRS.
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Query brain volume data.")
    parser.add_argument("volume", help="The volume file to load. Should be in mgh, mgz or nifti format.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-c', '--crs', nargs='*', help="The query voxel, defined as a 0-based index into the volume. For a 3D volume, this would be 3 integers which represent the CRS (column, row, slice) of the voxel, like 128 128 128.")
    group.add_argument('-a', '--all-values', help="Instead of returning the value for a single voxel, return all voxel values which occur in the volume. Forces integer values (by rounding).", action="store_true")
    group.add_argument('-l', '--all-value-counts', help="Instead of returning the value for a single voxel, return the counts for all voxel values which occur in the volume. The order of the counts is guaranteed to be identical to the order of the output when running with '-a'. Forces integer values (by rounding).", action="store_true")
    parser.add_argument("-v", "--verbose", help="Increase output verbosity.", action="store_true")
    parser.add_argument("-s", "--separator", help="Output separator (between vertex coords / indices).", default=" ")
    args = parser.parse_args()

    volume_file = args.volume
    verbose = args.verbose
    sep = args.separator

    vol_data = nib.load(volume_file).get_data()
    if verbose:
        print("Volume has %d dimensions, shape %s and data type %s. It contains %d voxels." % (len(vol_data.shape), vol_data.shape, vol_data.dtype, len(np.ravel(vol_data))))

    voxel_value_print_format = "%f"
    if np.issubdtype(vol_data.dtype, np.integer):
        voxel_value_print_format = "%d"

    if args.all_values or args.all_value_counts:
        voxel_value_print_format = "%d"
        vol_data = np.rint(vol_data).astype(int)    # Force integer values. For floats, you would get as many values of there are voxels, and this does not make sense.
        vol_data_flat = np.ravel(vol_data)
        occuring_values = dict()
        for value in vol_data_flat:
            if value in occuring_values:
                occuring_values[value] = occuring_values[value] + 1
            else:
                occuring_values[value] = 1
        if args.all_values:
            if verbose:
                print("Printing all %d different intensity values that occur within the volume." % (len(occuring_values)))
            print(sep.join([str(k) for k in sorted(occuring_values.keys())]))
        else:
            if verbose:
                print("Printing the counts for the %d different intensity values that occur within the volume. Sum of counts is %d." % (len(occuring_values), sum(occuring_values.values())))
            print(sep.join([str(pair[1]) for pair in sorted(occuring_values.items(), key=lambda pair: pair[0])]))

    else:
        voxel_index = tuple([int(x) for x in args.crs])
        print(voxel_value_print_format % (vol_data[voxel_index]))

    sys.exit(0)


if __name__ == "__main__":
    brain_vol_info()
