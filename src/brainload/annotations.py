"""
Functions for reading FreeSurfer vertex annotation files.

See https://surfer.nmr.mgh.harvard.edu/fswiki/LabelsClutsAnnotationFiles#Annotation and
 http://nipy.org/nibabel/reference/nibabel.freesurfer.html#nibabel.freesurfer.io.read_annotfor more information.
"""

import os
import numpy as np
import nibabel.freesurfer.io as fsio
import brainload.nitools as nit


def annot(subject_id, subjects_dir, annotation, hemi="both", meta_data=None):
    """
    Load annotation for the mesh vertices of a single subject.

    An annotation defines a label string and a color to each vertex, it is typically used to define brain regions, e.g., for cortical parcellation.

    Parameters
    ----------
    subject_id: string
        The subject identifier.

    subject_dir: string
        A string representing the path to the subjects dir.

    annotation: string
        An annotation to load, part of the file name of the respective file in the subjects label directory. E.g., 'aparc', 'aparc.a2009s', or 'aparc.DKTatlas'.

    hemi: {'both', 'lh', 'rh'}, optional
        The hemisphere for which data should actually be loaded. Defaults to 'both'.

    meta_data: dictionary | None, optional
        Meta data to merge into the output `meta_data`. Defaults to the empty dictionary.

    Returns
    -------
    labels: ndarray, shape (n_vertices,)
        Contains an annotation_id for each vertex. If the vertex has no annotation, the annotation_id -1 is returned.

    ctab: ndarray, shape (n_labels, 5)
        RGBT + label id colortable array. The first 4 values encode the label color: RGB is red, green, blue as usual, from 0 to 255 per value. T is the transparency, which is defined as 255 - alpha. The number of labels (n_label) cannot be know in advance.

    names: list of strings
       The names of the labels. The length of the list is n_labels. Note that, contrary to the respective nibabel function, this function will always return this as a list of strings, no matter the Python version used.

    meta_data: dictionary
        Contains detailed information on the data that was loaded. The following keys are available (replace `?h` with the value of the argument `hemisphere_label`, which must be 'lh' or 'rh').
            - `?h.annotation_file` : the file that was loaded
    """
    if hemi not in ('lh', 'rh', 'both'):
        raise ValueError("ERROR: hemi must be one of {'lh', 'rh', 'both'} but is '%s'." % hemi)

    if meta_data is None:
        meta_data = {}

    lh_annotation_file_name = "lh.%s.annot" % annotation
    rh_annotation_file_name = "rh.%s.annot" % annotation
    lh_annotation_file = os.path.join(subjects_dir, subject_id, 'label', lh_annotation_file_name)
    rh_annotation_file = os.path.join(subjects_dir, subject_id, 'label', rh_annotation_file_name)

    if hemi == 'lh':
        vertex_labels, label_colors, label_names, meta_data = read_annotation_md(lh_annotation_file, 'lh', meta_data=meta_data)
    elif hemi == 'rh':
        vertex_labels, label_colors, label_names, meta_data = read_annotation_md(rh_annotation_file, 'rh', meta_data=meta_data)
    else:
        lh_vertex_labels, lh_label_colors, lh_label_names, meta_data = read_annotation_md(lh_annotation_file, 'lh', meta_data=meta_data)
        rh_vertex_labels, rh_label_colors, rh_label_names, meta_data = read_annotation_md(rh_annotation_file, 'rh', meta_data=meta_data)
        #vertex_labels = merge_vertex_labels(np.array([lh_vertex_labels, rh_vertex_labels]))
        #label_colors = merge_label_colors(np.array([lh_label_colors, rh_label_colors]))
        if not _are_label_names_identical(lh_label_names, rh_label_names):
            raise ValueError("The %d labels for the lh and the %d labels for the rh are not identical for annotation '%s'." % (len(lh_label_names), len(rh_label_names), annotation))
        else:
            label_names = lh_label_names    # both are identical, so just pick any

        vertex_labels = np.hstack((lh_vertex_labels, rh_vertex_labels))

        if len(rh_label_colors) != len(lh_label_colors):
            raise ValueError("There are %d colors for the lh labels and %d colors for the rh labels, but they should be identical for annotation '%s'." % (len(lh_label_colors), len(rh_label_colors), annotation))

        label_colors = lh_label_colors    # both are identical, so just pick any
    return vertex_labels, label_colors, label_names, meta_data


def _are_label_names_identical(lh_label_names, rh_label_names):
    """
    Checks whether the two string lists have the same elements in same order.
    """
    if len(lh_label_names) != len(rh_label_names):
        return False
    for idx, label in enumerate(lh_label_names):
        if lh_label_names[idx] != rh_label_names[idx]:
            return False
    return True


def read_annotation_md(annotation_file, hemisphere_label, meta_data=None, encoding="utf-8"):
    """
    Read annotation file and record meta data for it.

    For details on the first three return values, see http://nipy.org/nibabel/reference/nibabel.freesurfer.html#nibabel.freesurfer.io.read_annot as they are the output of that function. An exception is the last parameter (names, names_str in this function) which returns a different data type depending on the Python version for the nibabel function. This function always returns strings, independent of the Python version.

    Parameters
    ----------
    annotation_file: string
        A string representing a path to a FreeSurfer vertex annotation file (e.g., the path to 'lh.aparc.annot').

    hemisphere_label: {'lh' or 'rh'}
        A string representing the hemisphere this file belongs to. This is used to write the correct meta data.

    meta_data: dictionary | None, optional
        Meta data to merge into the output `meta_data`. Defaults to the empty dictionary.

    encoding: string describing an encoding, optional
        The encoding to use when decoding the label strings from binary. Only used in Python 3. Defaults to 'utf-8'.

    Returns
    -------
    labels: ndarray, shape (n_vertices,)
        Contains an annotation_id for each vertex. If the vertex has no annotation, the annotation_id -1 is returned.

    ctab: ndarray, shape (n_labels, 5)
        RGBT + label id colortable array. The first 4 values encode the label color: RGB is red, green, blue as usual, from 0 to 255 per value. T is the transparency, which is defined as 255 - alpha. The number of labels (n_label) cannot be know in advance.

    names: list of strings
       The names of the labels. The length of the list is n_labels. Note that, contrary to the respective nibabel function, this function will always return this as a list of strings, no matter the Python version used.

    meta_data: dictionary
        Contains detailed information on the data that was loaded. The following keys are available (replace `?h` with the value of the argument `hemisphere_label`, which must be 'lh' or 'rh').
            - `?h.annotation_file` : the file that was loaded
    """
    if hemisphere_label not in ('lh', 'rh'):
        raise ValueError("ERROR: hemisphere_label must be one of {'lh', 'rh'} but is '%s'." % hemisphere_label)

    if meta_data is None:
        meta_data = {}

    labels, ctab, names = fsio.read_annot(annotation_file, orig_ids=False)

    label_file = hemisphere_label + '.annotation_file'
    meta_data[label_file] = annotation_file

    # The nibabel read_annot function returns string under Python 2 and bytes under Python 3, see http://nipy.org/nibabel/reference/nibabel.freesurfer.html#nibabel.freesurfer.io.read_annot.
    # We convert this to strings here so we always return strings.
    try:
        names_decoded = []
        for name in names:
            name_str = name.decode(encoding)
            names_decoded.append(name_str)
        names = names_decoded
    except AttributeError:
        pass

    return labels, ctab, names, meta_data
