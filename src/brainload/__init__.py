"""
Brainload high-level API functions.
"""

# The next line makes the listed functions show up in sphinx documentation directly under the package (they also show up under their real sub module, of course)
__all__ = [ 'subject', 'subject_avg', 'group', 'fsaverage_mesh', 'rhi', 'rhv', 'annot', 'label', 'stat', 'mesh_to_ply', 'mesh_to_obj' ]

__version__ = '0.3.1'

from .freesurferdata import subject, subject_avg, group, fsaverage_mesh, rhi, rhv
from .annotations import label, annot
from .stats import stat
from .meshexport import mesh_to_ply, mesh_to_obj
