#!/usr/bin/env python
from __future__ import print_function
import sys
import numpy as np
import nibabel as nib
import argparse
import brainload.nitools as nit
import warnings

# To run this in dev mode (in virtual env, pip -e install of brainload active) from REPO_ROOT:
# PYTHONPATH=./src/brainload python src/brainload/clients/brain_vol_info.py tests/test_data/subject1/mri/orig.mgz --crs 10 10 10 -v

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
    group.add_argument("-f", "--crs-file", help="A file containing the voxels to query, one per line. A voxel should be given by zero-based indices into each dimension of the volume, e.g., '0 23 188'.")
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
        print("---Brain Vol Info---")
        print("Volume has %d dimensions, shape %s and data type %s. It contains %d voxels." % (len(vol_data.shape), vol_data.shape, vol_data.dtype, len(np.ravel(vol_data))))

    voxel_value_print_format = "%f"
    if np.issubdtype(vol_data.dtype, np.integer):
        voxel_value_print_format = "%d"

    if args.all_values or args.all_value_counts:
        if verbose:
            print("NOTE: This mode treats the intensity values in the volume as integers. You should only use it if that is suitable for the input volume.")
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
        if args.crs:
            voxel_index = tuple([int(x) for x in args.crs])
            voxel_display_string = " ".join(args.crs)
            if verbose:
                print("Received 1 voxel index (with %d dimensions) from the command line. Printing intensity value of the voxel '%s' in the volume." % (len(voxel_index), voxel_display_string))
            if len(voxel_index) != len(vol_data.shape):
                warnings.warn("Dimension mismatch: Received query voxel with %d dimenions, but the volume has %d." % (len(voxel_index), len(vol_data.shape)))
            print(voxel_value_print_format % (vol_data[voxel_index]))
        else:
            voxel_indices = nit.load_voxel_indices(args.crs_file)
            voxel_values = []
            if voxel_indices.shape[1] != len(vol_data.shape):
                warnings.warn("Dimension mismatch: Received query voxels with %d dimensions, but the volume has %d." % (voxel_indices.shape[1], len(vol_data.shape)))
            if verbose:
                print("Received %d voxel indices (with %d dimensions) from file '%s'. Printing their intensity values in the volume." % (voxel_indices.shape[0], voxel_indices.shape[1], args.crs_file))
            for voxel_index in voxel_indices:
                voxel_index = tuple(voxel_index.tolist())
                voxel_values.append(vol_data[voxel_index])
            print(sep.join([str(v) for v in voxel_values]))

    sys.exit(0)


if __name__ == "__main__":
    brain_vol_info()
