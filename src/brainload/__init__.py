"""
Brainload high-level API functions.
"""

# The next line makes the listed functions show up in sphinx documentation directly under the package (they also show up under their real sub module, of course)
__all__ = ['parse_subject', 'load_group_data']

from .freesurferdata import parse_subject, load_group_data
