#!/usr/bin/env python

import os
import brainload.freesurferdata as fsd
import brainload.nitools as nit
from datetime import datetime

def gen_result_measures(measures):
    new_measures = []
    for measure in measures:
        new_measures.append(measure + '_avg0')
        new_measures.append(measure + '_avg10')
    return new_measures

def append_pial(measures):
    new_measures = []
    for measure in measures:
        new_measures.append(measure + '.pial')
    return new_measures

def check_files():
    print "-------------------- Checking ----------------------"
    print datetime.now()
    user_home = os.getenv('HOME')
    subjects_dir = os.path.join(user_home, 'data', 'euaims_curvature')
    subjects_file = os.path.join(subjects_dir, 'subjects_analysis.txt')
    subjects = nit.read_subjects_file(subjects_file)

    curv_measures_str = "principal_curvature_k1 principal_curvature_k2 principal_curvature_k_major principal_curvature_k_minor mean_curvature gaussian_curvature intrinsic_curvature_index negative_intrinsic_curvature_index gaussian_l2_norm absolute_intrinsic_curvature_index mean_curvature_index negative_mean_curvature_index mean_l2_norm absolute_mean_curvature_index folding_index curvedness_index shape_index shape_type area_fraction_of_intrinsic_curvature_index area_fraction_of_negative_intrinsic_curvature_index area_fraction_of_mean_curvature_index area_fraction_of_negative_mean_curvature_index sh2sh sk2sk"
    measures = curv_measures_str.split()
    surfaces = ['white']
    print "------- subject space -------"
    check_subject_morph_files(subjects, subjects_dir, measures, surfaces)
    print "------- common fsaverage space fwhm 0 -------"
    check_fsaverage_morph_files(subjects, subjects_dir, measures, surfaces, fwhm='0', write_missing_subjects_file=True, delete_missing_subjects_files_no_longer_missing=True)
    print "------- common fsaverage space fwhm 10 -------"
    check_fsaverage_morph_files(subjects, subjects_dir, measures, surfaces, fwhm='10')

    # just do it again without details printing for a better overview.
    print "+++++++++++++++++++++ Overview +++++++++++++++++++++++"
    check_subject_morph_files(subjects, subjects_dir, measures, surfaces, print_details=False)
    check_fsaverage_morph_files(subjects, subjects_dir, measures, surfaces, fwhm='0', print_details=False)
    check_fsaverage_morph_files(subjects, subjects_dir, measures, surfaces, fwhm='10', print_details=False)


def check_fsaverage_morph_files(subjects, subjects_dir, measures, surfaces, fwhm='10', print_details=True, write_missing_subjects_file=False, delete_missing_subjects_files_no_longer_missing=False):
    """
    Checks for the existance of the output files that contain the morph data mapped to fsaverage.
    """
    if write_missing_subjects_file and len(surfaces) > 1:
        print "ERROR: More than one surface given, the missing subjects files for the last surface one will overwrite all the others."
        exit(1)

    # generate our special measures
    measures = gen_result_measures(measures)

    # hax, sry
    white_measures = measures
    pial_measures = append_pial(measures)

    num_missing_files_total = 0
    for surf in surfaces:
        if surf not in ('pial', 'white'):
            raise ValueError("ERROR: surf must be 'white' or 'pial'.")
        if surf == 'pial':
            measures = pial_measures
        else:
            measures = white_measures
        surf_file_part = fsd._get_morphology_data_suffix_for_surface(surf)

        num_missing_files_total_this_surf = 0
        num_measures_ok = 0
        for measure in measures:
            missing_lh = {}
            missing_rh = {}
            for subject_id in subjects:
                lh_file_name = "lh.%s.fwhm%s.fsaverage.mgh" % (measure, fwhm)
                lh_file = os.path.join(subjects_dir, subject_id, 'surf', lh_file_name)
                if not os.path.isfile(lh_file):
                    #print "[fsaverage surf=%s] subject %s lh: missing %s" % (surf, subject_id, lh_file)
                    missing_lh[subject_id] = lh_file

                rh_file_name = "rh.%s.fwhm%s.fsaverage.mgh" % (measure, fwhm)
                rh_file = os.path.join(subjects_dir, subject_id, 'surf', rh_file_name)
                if not os.path.isfile(rh_file):
                    missing_rh[subject_id] = rh_file
                    #print "[fsaverage surf=%s] subject %s rh: missing %s" % (surf, subject_id, rh_file)
            missing_subjects_filename = 'missing_%s.txt' % measure
            missing_subjects_file = os.path.join(subjects_dir, missing_subjects_filename)
            if len(missing_lh) > 0 or len(missing_rh) > 0:
                if write_missing_subjects_file:
                    with open(missing_subjects_file, "w") as text_file:
                        # merge lh and rh into set
                        missing_both = set()
                        missing_both = missing_both.union(missing_lh.keys)
                        missing_both = missing_both.union(missing_rh.keys)
                        for subject_id in missing_both:
                            text_file.write("%s\n" % subject_id)
                    #print "NOTE: Created missing subjects file containing %d subjects at location '%s'." % (len(missing_lh), missing_subjects_file)

                if print_details:
                    if len(missing_lh) == len(missing_rh):
                        print "[fsaverage surf=%s fwhm=%s] Measure '%s': There were %d subjects which were missing both lh and rh files." % (surf, fwhm, measure, len(missing_lh))
                    else:
                        print "[fsaverage surf=%s fwhm=%s] Measure '%s': There were %d subjects with missing files for lh and %d with missing files for rh." % (surf, fwhm, measure, len(missing_lh), len(missing_rh))
            else:
                num_measures_ok += 1
                if delete_missing_subjects_files_no_longer_missing:
                    if os.path.isfile(missing_subjects_file):
                        os.remove(missing_subjects_file)
                        print "NOTE: Deleted file '%s', all subjects now have data for that measure." % missing_subjects_file

            num_missing_files_this_measure = len(missing_lh) + len(missing_rh)
            num_missing_files_total_this_surf += num_missing_files_this_measure
        print "[fsaverage surf=%s fwhm=%s] %d of %d measures OK." % (surf, fwhm, num_measures_ok, len(measures))
        num_max = len(subjects) * len(measures) * 2
        print "[fsaverage surf=%s fwhm=%s] Total number of missing files for surface %s: %d of %d" % (surf, fwhm, surf, num_missing_files_total_this_surf, num_max)
        num_missing_files_total += num_missing_files_total_this_surf
    num_max_total = len(subjects) * len(measures) * 2 * len(surfaces)
    print "[fsaverage fwhm=%s] Total number of missing files over all %d surfaces: %d of %d" % (fwhm, len(surfaces), num_missing_files_total, num_max_total)



def check_subject_morph_files(subjects, subjects_dir, measures, surfaces, print_details=True):
    """
    Checks for the existance of the output files that contain the morph data of the subject.
    """

    # generate our special measures
    measures = gen_result_measures(measures)
    num_missing_files_total = 0

    white_measures = measures
    pial_measures = append_pial(measures)

    for surf in surfaces:
        if surf not in ('pial', 'white'):
            raise ValueError("ERROR: surf must be 'white' or 'pial'.")
        if surf == 'pial':
            measures = pial_measures
        else:
            measures = white_measures
        surf_file_part = fsd._get_morphology_data_suffix_for_surface(surf)

        num_missing_files_total_this_surf = 0
        num_measures_ok = 0
        for measure in measures:
            missing_lh = {}
            missing_rh = {}
            for subject_id in subjects:

                lh_file_name = "lh.%s" % measure
                lh_file = os.path.join(subjects_dir, subject_id, 'surf', lh_file_name)
                if not os.path.isfile(lh_file):
                    #print "subject %s: missing %s" % (subject_id, lh_file)
                    missing_lh[subject_id] = lh_file

                rh_file_name = "rh.%s" % measure
                rh_file = os.path.join(subjects_dir, subject_id, 'surf', rh_file_name)
                if not os.path.isfile(rh_file):
                    missing_rh[subject_id] = rh_file
                    #print "subject %s: missing %s" % (subject_id, rh_file)

            if len(missing_lh) > 0 or len(missing_rh) > 0:
                if print_details:
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
