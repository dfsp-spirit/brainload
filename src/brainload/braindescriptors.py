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
        Add atlas stats, e.g., 'aparc' for stats/?h.aparc.annot
        """
        all_subjects_measures_dict, all_subjects_table_data_dict = brainload.stats.group_stats(self.subjects_list, self.subjects_dir, '%s.%s.stats' % (hemi, atlas), stats_table_type_list=brainload.stats.typelist_for_aparc_atlas_stats())
