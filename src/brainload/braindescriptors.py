"""
Collect various brain descriptors.

Functions to collect brain descriptors from neuroimaging data preprocessed with FreeSurfer.
"""

import os
import numpy as np
import nibabel.freesurfer.io as fsio
import brainload.nitools as nit
import brainload.freesurferdata as fsd
import brainload.stats
import logging

class BrainDescriptors:
    """
    Collects descriptors for one or more subjects.
    """
    def __init__(self, subjects_dir, subjects_list, hemi='both'):
        self.subjects_dir = subjects_dir
        self.subjects_list = subjects_list
        self.descriptor_names = []
        self.descriptor_values = np.zeros((len(self.subjects_list), 1))

        if hemi not in ('lh', 'rh', 'both'):
            raise ValueError("ERROR: hemi must be one of {'lh', 'rh', 'both'} but is '%s'." % hemi)
        if hemi == 'both':
            self.hemis = ['lh', 'rh']
        else:
            self.hemis = [hemi]


    def add_parcellation_stats(self, atlas_list):
        for hemi in self.hemis:
            for atlas in atlas_list:
                self._add_single_parcellation_stats(atlas, hemi)


    def _add_single_parcellation_stats(self, atlas, hemi):
        """
        Add brain parcellation atlas stats.

        Add brain parcellation atlas stats, e.g., 'aparc' for stats/?h.aparc.annot. (Note that this function is not for segmentation stats like aseg.)
        """
        all_subjects_measures_dict, all_subjects_table_data_dict = brainload.stats.group_stats(self.subjects_list, self.subjects_dir, '%s.%s.stats' % (hemi, atlas), stats_table_type_list=brainload.stats.typelist_for_aparc_atlas_stats())

        self._add_measure_dict_stats(all_subjects_measures_dict, atlas)
        self._add_all_subjects_table_data_stats(all_subjects_table_data_dict, atlas)


    def _add_measure_dict_stats(self, all_subjects_measures_dict, atlas):
        for measure_unique_name in list(all_subjects_measures_dict.keys()):
            measure_data_all_subjects = all_subjects_measures_dict[measure_unique_name]
            self.descriptor_values = np.hstack((self.descriptor_values, np.expand_dims(measure_data_all_subjects, axis=1)))
            self.descriptor_names.append("stats_%s_measure_%s" % (atlas, measure_unique_name))

    def _add_all_subjects_table_data_stats(self, all_subjects_table_data_dict, atlas, ignore_columns=None):
        if ignore_columns is None:
            ignore_columns = []
        region_colum_name = brainload.stats.stats_table_region_label_column_name()
        region_names = [rname.decode('utf-8') for rname in all_subjects_table_data_dict[region_colum_name][0]]

        for table_column_name in all_subjects_table_data_dict.keys():
            if table_column_name not in ignore_columns and table_column_name != region_colum_name:
                all_subjects_all_region_data = all_subjects_table_data_dict[table_column_name]
                self.descriptor_values = np.hstack((self.descriptor_values, all_subjects_all_region_data))
                for region_name in region_names:
                    self.descriptor_names.append("stats_%s_table_%s_%s" % (atlas, table_column_name, region_name))


    def add_standard_stats(self):
        """
        Convenience function to add all descriptors which are computed by default when running Freesurfer v6 recon-all on a subject. WARNING: In the current state, it only adds data we have available for testing.
        """
        self.add_parcellation_stats(['aparc', 'aparc.a2009s'])
        self.add_segmentation_stats()
        self.add_custom_measure_stats(['aparc'], ['area'])


    def report_descriptors(self):
        print("DEBUG: self.descriptor_values has shape %s" % (str(self.descriptor_values.shape)))
        print("DEBUG: Found %d descriptor names." % (len(self.descriptor_names)))

        if len(self.descriptor_names) != self.descriptor_values.shape[1] -1:    # the '-1' is because the subject ID is part of the
            print("Mismatch between descriptor names and values.")

        print("---------------------------------------------------------------------")
        print("subject_id " + " ".join(self.descriptor_names))
        for sidx, subject_id in enumerate(self.subjects_list):
            print("%s " % (subject_id), end="")
            for i in range(len(self.descriptor_names)):
                print("%.2f" % (self.descriptor_values[sidx,i]), end=" ")
            print("")


    def add_segmentation_stats(self):
        """
        Add brain parcellation stats.

        Add brain parcellation stats. This add the data from the stats/aseg.stats file.
        """
        self.add_single_segmentation_stats('aseg')


    def add_curv_stats(self):
        """
        Add surface curvature stats.

        Add brain surface curvature stats. This add the data from the stats/?h.curv.stats files.
        """
        for subject_id in self.subjects_list:
            for hemi in self.hemis:
                curv_stat_names, curv_stat_values = brainload.stats.parse_curve_stats(subject_id, subjects_dir, hemi)
                self.descriptor_values = np.hstack((self.descriptor_values, np.expand_dims(curv_stat_values, axis=1)))
                self.descriptor_names.append(curv_stat_names)


    def add_single_segmentation_stats(self, atlas):
        """
        Add brain parcellation stats.

        Add brain parcellation stats. This add the data from the stats/aseg.stats file.
        """
        all_subjects_measures_dict, all_subjects_table_data_dict = brainload.stats.group_stats_aseg(self.subjects_list, self.subjects_dir)

        self._add_measure_dict_stats(all_subjects_measures_dict, atlas)
        self._add_all_subjects_table_data_stats(all_subjects_table_data_dict, atlas, ignore_columns=['Index', 'SegId'])


    def add_custom_measure_stats(self, atlas_list, measure_list):
        """
        Add custom stats for a measure and atlas.

        Add custom stats for a measure and atlas. E.g., compute descriptive stats (min, max, mean, ...) for a measure like 'pial_lgi' in all regions of an atlas like 'aparc'.
        """
        for hemi in self.hemis:
            for atlas in atlas_list:
                for measure in measure_list:
                    self._add_custom_measure_stats_single(atlas, measure, hemi)


    def _add_custom_measure_stats_single(self, atlas, measure, hemi):
        for subject_id in self.subjects_list:
            morphometry_data, morphometry_meta_data = brainload.freesurferdata.subject_data_native(subject_id, self.subjects_dir, measure, hemi)
            region_data_per_hemi, label_names = brainload.annotations.region_data_native(subject_id, self.subjects_dir, atlas, hemi, morphometry_data, morphometry_meta_data)
            # TODO: compute descriptive stats
