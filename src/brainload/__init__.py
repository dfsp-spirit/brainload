"""
Brainload high-level API functions.
"""

# The next line makes the listed functions show up in sphinx documentation directly under the package (they also show up under their real sub module, of course)
__all__ = [ 'subject', 'subject_avg', 'group', 'fsaverage_mesh', 'subject_mesh', 'rhi', 'rhv', 'hemi_range', 'annot', 'label', 'stat', 'mesh_to_ply', 'mesh_to_obj', 'read_subjects_file', 'read_subjects_file', 'detect_subjects_in_directory', 'subject_data_native', 'subject_data_standard']

__version__ = '0.3.4'

from .freesurferdata import subject, subject_avg, group, fsaverage_mesh, subject_mesh, rhi, rhv, hemi_range, subject_data_native, subject_data_standard
from .annotations import label, annot
from .stats import stat
from .meshexport import mesh_to_ply, mesh_to_obj
from .nitools import read_subjects_file, detect_subjects_in_directory
