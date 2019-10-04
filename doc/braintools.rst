Brainload Command Line Tools
=============================

Brainload comes with some command line tools which demonstrate example usage of the library and make some of its the functionality available to non-programmers.


Available command line tools
---------------------------------

- ``brain_vol_info``: Query voxel values from a volume file, e.g., a structural scan in MGH or MGZ format like ``mri/brain.mgz``.
- ``brain_morph_info``: Query the values from a morphometry file like ``surf/lh.thickness``.
- ``brain_mesh_info``: Query vertex coordinates and faces from a surface file like ``surf/lh.white``.
- ``visualize_verts``: Visualize brain surface vertices. Generate a text file overlay that can be read by freeview that colors the given vertices.
- ``brain_qa``: Tool to check the consistency and availability of pre-processed FreeSurfer data for a study. Checks whether all requested data files are available for all subjects, and whether the data are consistent.

Getting help on command line tools
---------------------------------

All command line tools come with detailed built-in help. Run them with ``-h`` or ``--help`` to access it. Example:


.. code:: shell
    brain_vol_info --help
