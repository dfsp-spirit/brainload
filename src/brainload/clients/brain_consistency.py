#!/usr/bin/env python
from __future__ import print_function
import sys
import brainload.nitools as nit
import brainload.qa as brainqa
import brainload.nitools as nit
import argparse
import logging

# To run this in dev mode (in virtual env, pip -e install of brainload active) from REPO_ROOT:
# PYTHONPATH=./src/brainload python src/brainload/clients/brain_consistency.py $SUBJECTS_DIR $SUBJECTS_DIR/subjects.txt -m area:volume:thickness:pial_lgi -v

def brain_consistency():
    """
    Brain data consistency checks.

    Simple script to check brain data consistency.
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Check brain data consistency.")
    parser.add_argument("subjects_dir", help="The directory that contains your subjects. Note: in most cases you can pass the environment variable SUBJECTS_DIR.")
    parser.add_argument("subjects_file", help="Text file containing all subjects that should be checked, one subject per line.")
    parser.add_argument("-m", "--measures-native", help="Which vertex-wise native space measure data to check. Colon-separated string. Defaults to 'area:volume:thickness'", default="area:volume:thickness")
    parser.add_argument("-v", "--verbose", help="Increase output verbosity.", action="store_true")
    args = parser.parse_args()


    if args.verbose:
        print("---Brain Data Consistency Checks---")
        print("Using subjects_dir '%s' and subjects file '%s'." % (args.subjects_dir, args.subjects_file))
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARN)

    native_measures = args.measures_native.split(":")
    subjects_list = nit.read_subjects_file(args.subjects_file)

    if args.verbose:
        print("Handling %d subjects." % (len(subjects_list)))
        print("Handling %d native measures: %s" % (len(native_measures), ", ".join(native_measures)))



    bdc = brainqa.BrainDataConsistency(args.subjects_dir, subjects_list)
    bdc.check_custom(native_measures)

    sys.exit(0)


if __name__ == "__main__":
    brain_consistency()
