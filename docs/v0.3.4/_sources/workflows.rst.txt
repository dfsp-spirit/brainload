Brainload Example Workflows
===========================

This document illustrates example workflows for common tasks.


Loading data for a single subject
---------------------------------


Load brain mesh and morphometry data for a single subject in subject space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this example, we will load area data for each vertex of the example subject bert that comes with FreeSurfer from the files ``?h.area``. We will not rely on the environment variable SUBJECTS_DIR, but explicitly specify the directory containing the data.

.. code:: python

    import brainload as bl
    import os
    freesurfer_dir = os.path.join('usr', 'local', 'freesurfer')  # or wherever your FREESURFER_HOME is
    subjects_dir = os.path.join(freesurfer_dir, 'subjects')

    vert_coords, faces, per_vertex_data, meta_data = bl.subject('bert', subjects_dir=subjects_dir, measure='area')


This operation loaded 4 files: the 2 brain mesh files (one for the hemisphere, one for the right hemisphere) and the 2 morphometry data files. The mesh data are in the variables *vert_coords* and *faces*, and the morphometry data can be found in *per_vertex_data*. The *meta_data* contains information on the loaded data. Let's use it to see exactly which files were loaded.

.. code:: python

    print "%s\n%s\n%s\n%s\n" % (meta_data['lh.curv_file'], meta_data['rh.curv_file'], meta_data['lh.morphometry_file'], meta_data['rh.morphometry_file'])
    /usr/local/freesurfer/subjects/bert/surf/lh.white
    /usr/local/freesurfer/subjects/bert/surf/rh.white
    /usr/local/freesurfer/subjects/bert/surf/lh.area
    /usr/local/freesurfer/subjects/bert/surf/rh.area


This way you always know what data you are working with. See the API documentation for more options. You can specify a different surface, load only one hemisphere, or not load the mesh at all when using this function.


Load morphometry data for a single subject that has been mapped to a common subject
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this example, we will load morphometry data that have been mapped to a common subject, in this case, the fsaverage subject from FreeSurfer. The data have to be mapped using the ``recon-all ... -qcache`` FreeSurfer command. We assume the data already exist for your subject in files like *?h.area.fwhm20.fsaverage.mgh*.

.. code:: python

    import brainload as bl
    import os
    subjects_dir = os.path.join(os.getenv('HOME'), 'data', 'mystudy')

    vert_coords, faces, morphometry_data, meta_data = bl.subject_avg('subject1', subjects_dir=subjects_dir, measure='area', fwhm='20')

This operation loaded 4 files: the 2 brain mesh files of the fsaverage subject and the 2 morphometry data files of subject1. The mesh data are in the variables *vert_coords* and *faces*, and the morphometry data can be found in *per_vertex_data*. The *meta_data* contains information on the loaded data. Let's use it to see exactly which files were loaded.

.. code:: python

    print "%s\n%s\n%s\n%s\n" % (meta_data['lh.curv_file'], meta_data['rh.curv_file'], meta_data['lh.morphometry_file'], meta_data['rh.morphometry_file'])
    /home/me/data/mystudy/fsaverage/surf/lh.white
    /home/me/data/mystudy/fsaverage/surf/rh.white
    /home/me/data/mystudy/subject1/surf/lh.area.fwhm20.fsaverage.mgh
    /home/me/data/mystudy/subject1/surf/rh.area.fwhm20.fsaverage.mgh


See the API documentation for more options. You can specify a different surface, load only one hemisphere, not load the mesh at all, or chose a custom average subject when using this function.
