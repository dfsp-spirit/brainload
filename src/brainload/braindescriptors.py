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
        self.descriptor_values = []

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


    def add_segmentation_stats(self):
        """
        Add brain parcellation stats.

        Add brain parcellation stats. This add the data from the stats/aseg.stats file.
        """
        all_subjects_measures_dict, all_subjects_table_data_dict = brainload.stats.group_stats_aseg(self.subjects_list, self.subjects_dir)


    def add_custom_measure_stats(self, atlas_list, measure_list):
        """
        Add custom stats for a measure and atlas.

        Add custom stats for a measure and atlas. E.g., compute descriptive stats (min, max, mean, ...) for a measure like 'lgi_pial' in all regions of an atlas like 'aparc'.
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
