Brainload Example Workflows
===========================

This document illustrates example workflows for common tasks.


Loading data for a single subject
---------------------------------


Load brain mesh and morphometry data for a single subject in native space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this example, we will load area data for each vertex of the example subject *bert* that comes with FreeSurfer from the files ``?h.area``. We will not rely on the environment variable SUBJECTS_DIR, but explicitly specify the directory containing the data.

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

Load an atlas or brain parcellation for a subject (e.g., Desikan or Destrieux atlas)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Native space data is often used for region-based analysis using a brain atlas. Here we shoud how to load and use a brain parcellation. In FreeSurfer, the parcellations are saved in annotation files in ``subject/label/``, and the following are available by default:

- Desikan-Killiany Atlas (``aparc``): stored in files ``lh.aparc.annot`` and ``rh.aparc.annot``. (The files are for the two hemispheres, we will replace lh and rh with ?h from now on.)
- Destrieux Atlas  (``aparc.a2009s``): stored in files ``?h.aparc.a2009s.annot``.
- DKT Atlas (``aparc.DKTatlas``): stored in files ``?h.aparc.DKTatlas.annot``

A parcellation contains:

- A colortable, which lists the regions of the atlas and assigns a color, a unique id (actually computed from the color) and a name to each region.
- A list which assigns each vertex of the brain to one of the regions, using the unique id from the colortable.

Here is how to load a parcellation in brainload:

.. code:: python
    import brainload as bl
    subjects_dir = os.path.join(os.getenv('HOME'), 'data', 'study1')
    vertex_labels, label_colors, label_names, meta_data = bl.annot('subject1', subjects_dir, 'aparc', hemi='lh')

The ``vertex_labels`` contain, for each vertex of the brain mesh, an index into the label_names datastructure. So if we want to know how many vertices of this subjects are assigned to the region 'bankssts':

.. code:: python
    np.sum(vertex_labels==label_names.index('bankssts'))

If you want to know the region name for the vertex at index 100000, try this (not that indices are 0-based on Python):

.. code:: python
    label_names[vertex_labels[100000]]

The ``label_colors`` return value contains the colormap as a matrix, and the first 3 columns are the RGB color values. So let's get the color of that vertex now:

.. code:: python
    label_colors[vertex_labels[100000]]

That's it for annotations.

Load morphometry data for a single subject that has been mapped to a common subject (standard space)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this example, we will load morphometry data that have been mapped to a common subject, in this case, the fsaverage subject from FreeSurfer. The data have to be mapped using the ``recon-all ... -qcache`` FreeSurfer command. We assume the data already exist for your subject in files like *?h.area.fwhm20.fsaverage.mgh*.

.. code:: python

    import brainload as bl
    import os
    subjects_dir = os.path.join(os.getenv('HOME'), 'data', 'study1')

    vert_coords, faces, morphometry_data, meta_data = bl.subject_avg('subject1', subjects_dir=subjects_dir, measure='area', fwhm='20')

This operation loaded 4 files: the 2 brain mesh files of the fsaverage subject and the 2 morphometry data files of subject1. The mesh data are in the variables *vert_coords* and *faces*, and the morphometry data can be found in *per_vertex_data*. The *meta_data* contains information on the loaded data. Let's use it to see exactly which files were loaded.

.. code:: python

    print "%s\n%s\n%s\n%s\n" % (meta_data['lh.curv_file'], meta_data['rh.curv_file'], meta_data['lh.morphometry_file'], meta_data['rh.morphometry_file'])
    /home/me/data/study1/fsaverage/surf/lh.white
    /home/me/data/study1/fsaverage/surf/rh.white
    /home/me/data/study1/subject1/surf/lh.area.fwhm20.fsaverage.mgh
    /home/me/data/study1/subject1/surf/rh.area.fwhm20.fsaverage.mgh


See the API documentation for more options. You can specify a different surface, load only one hemisphere, not load the mesh at all, or chose a custom average subject when using this function.


Load brain mesh and morphometry data for a group of subjects in native space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. code:: python
    import os
    import brainload as bl
    import numpy as np
    subjects_dir = os.path.join(os.getenv('HOME'), 'data', 'study1')
    subjects_list = ['subject1', 'subject4', 'subject5']
    morphdata_by_subject, metadata_by_subject = bl.group_native('curv', hemi='lh', subjects_dir=subjects_dir, subjects_list=subjects_list)

This will load the file ``surf/lh.curv`` for each subject.

Continuing the last example, we may want to have a look at the curv value of the vertex at index 100000 of the subject 'subject4':

.. code:: python
    morphdata_by_subject['subject4'][100000]

You may also be interested in the average curvature of subject1:

.. code:: python
    np.mean(morphdata_by_subject['subject1'])


Load brain mesh and morphometry data for a group of subjects in standard space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. code:: python
    import os
    import brainload as bl
    import numpy as np
    subjects_dir = os.path.join(os.getenv('HOME'), 'data', 'study1')
    subjects_list = ['subject1', 'subject4', 'subject5']
    data, subjects, group_md, run_md = bl.group('curv', fwhm='20', hemi='lh', subjects_dir=subjects_dir, subjects_list=subjects_list)

This will load the file ``surf/lh.curv.fwhm20.fsaverage.mgh`` for each subject.

In standard space, all subjects have the same number of vertices, so the data is returned as a matrix instead of dictionaries. Continuing the last example, we may want to have a look at the curv value of the vertex at index 100000 of the subject 'subject4':

.. code:: python
    subject4_idx = subjects.index('subject4')
    print data[subject4_idx][100000]

You may also be interested in the average curvature of subject1:

.. code:: python
    np.mean(data[subjects.index('subject1')])
