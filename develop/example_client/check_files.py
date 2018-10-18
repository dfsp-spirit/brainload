#!/usr/bin/env python

import os
import brainload.freesurferdata as fsd
import brainload.errors as ble

def check_files():
    user_home = os.getenv('HOME')
    subjects_dir = os.path.join(user_home, 'data', 'euaims_curvature')
    try:
        data, subjects, gmd, rmd = fsd.load_group_data('principal_curvature_k1_avg0', surf='white', hemi='lh', fwhm='10', subjects_dir=subjects_dir)
    except ble.HemiFileIOError as e:
        print "ERROR: file: " + e.filename + ", hemi:" + e.hemi


if __name__ == '__main__':
    check_files()
