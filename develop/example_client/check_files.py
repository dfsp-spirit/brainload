#!/usr/bin/env python

import os
import brainload.freesurferdata as fsd
import brainload.errors as ble
import brainload.nitools as nit

def gen_result_measures(measures):
    new_measures = []
    for measure in measures:
        new_measures.append(measure + '_avg0')
        new_measures.append(measure + '_avg10')
    return new_measures

def check_files():
    user_home = os.getenv('HOME')
    subjects_dir = os.path.join(user_home, 'data', 'euaims_curvature')
    subjects_file = os.path.join(subjects_dir, 'subjects_analysis.txt')
    subjects = nit.read_subjects_file(subjects_file)

    curv_measures_str = "principal_curvature_k1 principal_curvature_k2 principal_curvature_k_major principal_curvature_k_minor mean_curvature gaussian_curvature intrinsic_curvature_index negative_intrinsic_curvature_index gaussian_l2_norm absolute_intrinsic_curvature_index mean_curvature_index negative_mean_curvature_index mean_l2_norm absolute_mean_curvature_index folding_index curvedness_index shape_index shape_type area_fraction_of_intrinsic_curvature_index area_fraction_of_negative_intrinsic_curvature_index area_fraction_of_mean_curvature_index area_fraction_of_negative_mean_curvature_index sh2sh sk2sk"
    measures = curv_measures_str.split()
    check_subject_morph_files(subjects, subjects_dir, measures)
    check_fsaverage_morph_files(subjects, subjects_dir, measures)


def check_fsaverage_morph_files(subjects, subjects_dir, measures, surf='white', fwhm='10'):
    """
    Checks for the existance of the output files that contain the morph data mapped to fsaverage.
    """
    # generate our special measures
    measures = gen_result_measures(measures)

    for measure in measures:
        missing_lh = {}
        missing_rh = {}
        for subject_id in subjects:
#            try:
#                morph, meta = fsd.parse_subject_standard_space_data(subject_id, measure=measure, surf=surf, hemi='lh', fwhm=fwhm, subjects_dir=subjects_dir, load_surface_files=False)[2:4]
#            except ble.HemiFileIOError as e:
#                missing_lh[subject_id] = e.filename
#
#            try:
#                morph, meta = fsd.parse_subject_standard_space_data(subject_id, measure=measure, surf=surf, hemi='rh', fwhm=fwhm, subjects_dir=subjects_dir, load_surface_files=False)[2:4]
#            except ble.HemiFileIOError as e:
#                missing_rh[subject_id] = e.filename
            lh_file_name = "lh%s.%s" % (surf_file_part, measure)
            lh_file = os.path.join(subjects_dir, subject_id, 'surf', lh_file_name)
            if not os.path.isfile(lh_file):
                print "subject %s: missing %s" % (subject_id, lh_file)
                missing_lh[subject_id] = lh_file

            rh_file_name = "rh%s.%s" % (surf_file_part, measure)
            rh_file = os.path.join(subjects_dir, subject_id, 'surf', rh_file_name)
            if not os.path.isfile(rh_file):
                missing_rh[subject_id] = rh_file
                print "subject %s: missing %s" % (subject_id, rh_file)
        if len(missing_lh) > 0 or len(missing_rh) > 0:
            print "[fsaverage] Measure '%s': There were %d subjects with missing files for lh and %d with missing files for rh." % (measure, len(missing_lh), len(missing_rh))
        else:
            print "[fsaverage] Measure '%s': OK


def check_subject_morph_files(subjects, subjects_dir, measures, surf='white', fwhm='10'):
    """
    Checks for the existance of the output files that contain the morph data of the subject.
    """
    # generate our special measures
    measures = gen_result_measures(measures)

    for measure in measures:
        missing_lh = {}
        missing_rh = {}
        for subject_id in subjects:
            surf_file_part = fsd._get_morphology_data_suffix_for_surface(surf)

            lh_file_name = "lh%s.%s" % (surf_file_part, measure)
            lh_file = os.path.join(subjects_dir, subject_id, 'surf', lh_file_name)
            if not os.path.isfile(lh_file):
                print "subject %s: missing %s" % (subject_id, lh_file)
                missing_lh[subject_id] = lh_file

            rh_file_name = "rh%s.%s" % (surf_file_part, measure)
            rh_file = os.path.join(subjects_dir, subject_id, 'surf', rh_file_name)
            if not os.path.isfile(rh_file):
                missing_rh[subject_id] = rh_file
                print "subject %s: missing %s" % (subject_id, rh_file)

        if len(missing_lh) > 0 or len(missing_rh) > 0:
            print "[subject] Measure '%s': There were %d subjects with missing files for lh and %d with missing files for rh." % (measure, len(missing_lh), len(missing_rh))
        else:
            print "[subject] Measure '%s': OK



if __name__ == '__main__':
    check_files()
