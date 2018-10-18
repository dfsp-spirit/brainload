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
    print "-------------------- Checking ----------------------"
    user_home = os.getenv('HOME')
    subjects_dir = os.path.join(user_home, 'data', 'euaims_curvature')
    subjects_file = os.path.join(subjects_dir, 'subjects_analysis.txt')
    subjects = nit.read_subjects_file(subjects_file)

    curv_measures_str = "principal_curvature_k1 principal_curvature_k2 principal_curvature_k_major principal_curvature_k_minor mean_curvature gaussian_curvature intrinsic_curvature_index negative_intrinsic_curvature_index gaussian_l2_norm absolute_intrinsic_curvature_index mean_curvature_index negative_mean_curvature_index mean_l2_norm absolute_mean_curvature_index folding_index curvedness_index shape_index shape_type area_fraction_of_intrinsic_curvature_index area_fraction_of_negative_intrinsic_curvature_index area_fraction_of_mean_curvature_index area_fraction_of_negative_mean_curvature_index sh2sh sk2sk"
    measures = curv_measures_str.split()
    surfaces = ['white', 'pial']
    check_subject_morph_files(subjects, subjects_dir, measures, surfaces)
    check_fsaverage_morph_files(subjects, subjects_dir, measures, surfaces, fwhm='10')


def check_fsaverage_morph_files(subjects, subjects_dir, measures, surfaces, fwhm='10'):
    """
    Checks for the existance of the output files that contain the morph data mapped to fsaverage.
    """
    # generate our special measures
    measures = gen_result_measures(measures)
    num_missing_files_total = 0
    for surf in surfaces:
        surf_file_part = fsd._get_morphology_data_suffix_for_surface(surf)

        num_missing_files_total_this_surf = 0
        num_measures_ok = 0
        for measure in measures:
            missing_lh = {}
            missing_rh = {}
            for subject_id in subjects:
                lh_file_name = "lh%s.%s.fwhm%s.fsaverage.mgh" % (surf_file_part, measure, fwhm)
                lh_file = os.path.join(subjects_dir, subject_id, 'surf', lh_file_name)
                if not os.path.isfile(lh_file):
                    #print "[fsaverage surf=%s] subject %s lh: missing %s" % (surf, subject_id, lh_file)
                    missing_lh[subject_id] = lh_file

                rh_file_name = "rh%s.%s.fwhm%s.fsaverage.mgh" % (surf_file_part, measure, fwhm)
                rh_file = os.path.join(subjects_dir, subject_id, 'surf', rh_file_name)
                if not os.path.isfile(rh_file):
                    missing_rh[subject_id] = rh_file
                    #print "[fsaverage surf=%s] subject %s rh: missing %s" % (surf, subject_id, rh_file)
            if len(missing_lh) > 0 or len(missing_rh) > 0:
                if len(missing_lh) == len(missing_rh):
                    print "[fsaverage surf=%s] Measure '%s': There were %d subjects which were missing both lh and rh files." % (surf, measure, len(missing_lh))
                else:
                    print "[fsaverage surf=%s] Measure '%s': There were %d subjects with missing files for lh and %d with missing files for rh." % (surf, measure, len(missing_lh), len(missing_rh))
            else:
                num_measures_ok += 1
            num_missing_files_this_measure = len(missing_lh) + len(missing_rh)
            num_missing_files_total_this_surf += num_missing_files_this_measure
        print "[fsaverage surf=%s fwhm=%s] %d of %d measures OK." % (surf, fwhm, num_measures_ok, len(measures))
        num_max = len(subjects) * len(measures) * 2
        print "[fsaverage surf=%s fwhm=%s] Total number of missing files for surface %s: %d of %d" % (surf, fwhm, surf, num_missing_files_total_this_surf, num_max)
        num_missing_files_total += num_missing_files_total_this_surf
    num_max_total = len(subjects) * len(measures) * 2 * len(surfaces)
    print "[fsaverage fwhm=%s] Total number of missing files over all %d surfaces: %d of %d" % (fwhm, len(surfaces), num_missing_files_total, num_max_total)



def check_subject_morph_files(subjects, subjects_dir, measures, surfaces):
    """
    Checks for the existance of the output files that contain the morph data of the subject.
    """
    # generate our special measures
    measures = gen_result_measures(measures)
    num_missing_files_total = 0
    for surf in surfaces:
        surf_file_part = fsd._get_morphology_data_suffix_for_surface(surf)

        num_missing_files_total_this_surf = 0
        num_measures_ok = 0
        for measure in measures:
            missing_lh = {}
            missing_rh = {}
            for subject_id in subjects:

                lh_file_name = "lh%s.%s" % (surf_file_part, measure)
                lh_file = os.path.join(subjects_dir, subject_id, 'surf', lh_file_name)
                if not os.path.isfile(lh_file):
                    #print "subject %s: missing %s" % (subject_id, lh_file)
                    missing_lh[subject_id] = lh_file

                rh_file_name = "rh%s.%s" % (surf_file_part, measure)
                rh_file = os.path.join(subjects_dir, subject_id, 'surf', rh_file_name)
                if not os.path.isfile(rh_file):
                    missing_rh[subject_id] = rh_file
                    #print "subject %s: missing %s" % (subject_id, rh_file)

            if len(missing_lh) > 0 or len(missing_rh) > 0:
                if len(missing_lh) == len(missing_rh):
                    print "[subject surf=%s] Measure '%s': There were %d subjects which were missing both lh and rh files." % (surf, measure, len(missing_lh))
                else:
                    print "[subject surf=%s] Measure '%s': There were %d subjects with missing files for lh and %d with missing files for rh." % (surf, measure, len(missing_lh), len(missing_rh))
            else:
                num_measures_ok += 1
            num_missing_files_this_measure = len(missing_lh) + len(missing_rh)
            num_missing_files_total_this_surf += num_missing_files_this_measure
        print "[subject surf=%s] %d of %d measures OK." % (surf, num_measures_ok, len(measures))
        num_max = len(subjects) * len(measures) * 2
        print "[subject surf=%s] Total number of missing files for surface %s: %d of %d" % (surf, surf, num_missing_files_total_this_surf, num_max)
        num_missing_files_total += num_missing_files_total_this_surf
    num_max_total = len(subjects) * len(measures) * 2 * len(surfaces)
    print "[subject] Total number of missing files over all %d surfaces: %d of %d" % (len(surfaces), num_missing_files_total, num_max_total)



if __name__ == '__main__':
    check_files()
