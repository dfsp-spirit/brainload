Brainload Example Workflows
===========================

This document illustrates example workflows for common tasks.


Loading data for a single subject
---------------------------------


Load the brain mesh and morphometry data for a single subject in subject space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this example, we will load area data for each vertex of the example subject bert that comes with FreeSurfer. We will not rely on the environment variable SUBJECTS_DIR, but explicitly specify the directory containing the data.

.. code:: python

    import brainload as bl
    import os
    freesurfer_dir = os.path.join('usr', 'local', 'freesurfer')  # or wherever your FREESURFER_HOME is
    subjects_dir = os.path.join(freesurfer_dir, 'subjects')

    vert_coords, faces, per_vertex_data, meta_data = bl.subject('bert', measure='area')


This operation loaded 4 files: the 2 brain mesh files (one for the hemisphere, one for the right hemisphere) and the 2 morphometry data files. The mesh data are in the variables *vert_coords* and *faces*, and the morphometry data can be found in *per_vertex_data*. The *meta_data* contains information on the loaded data. Let's use it to see exactly which files were loaded.

.. code:: python

    print "%s\n%s\n%s\n%s\n" % (meta_data['lh.curv_file'], meta_data['rh.curv_file'], meta_data['lh.morphometry_file'], meta_data['rh.morphometry_file'])
    /usr/local/freesurfer/subjects/bert/surf/lh.white
    /usr/local/freesurfer/subjects/bert/surf/rh.white
    /usr/local/freesurfer/subjects/bert/surf/rh.area
    /usr/local/freesurfer/subjects/bert/surf/rh.area


This way you always know what data you are working with.
