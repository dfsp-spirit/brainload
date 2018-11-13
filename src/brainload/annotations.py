"""
Functions for reading FreeSurfer vertex annotation files.

"""

import os
import numpy as np
import nibabel.freesurfer.io as fsio
import brainload.nitools as nit


def annot(subject_id, subjects_dir, annotation, hemi="both", meta_data=None, orig_ids=False):
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

    orig_ids: boolean, optional
        Passed on to nibabel.freesurfer.io.read_annot function. From the documentation of that function: 'Whether to return the vertex ids as stored in the annotation file or the positional colortable ids. With orig_ids=False vertices with no id have an id set to -1.' Defaults to False.

    Returns
    -------
    vertex_label_colors: ndarray, shape (n_vertices,)
        Contains an annotation color id for each vertex listed in the annotation file. If orig_ids is False (the default), and some vertex has no annotation, -1 is returned for it. IMPORTANT: The annotation value in here is NOT the label id. It is the color for the vertex, encoded in a weird way! Yes, this is ugly. See https://surfer.nmr.mgh.harvard.edu/fswiki/LabelsClutsAnnotationFiles#Annotation for details, especially the section 'Annotation file design surprise'. The color is encoded as a single number. Quoting the linked document, the numer is the 'RGB value combined into a single 32-bit integer: annotation value = (B * 256^2) + (G * 256) + (R)'. From this it follows that, quoting the doc once more, 'Code that loads an annotation file ... has to compare annotation values to the color values in the ColorLUT part of the annotation file to discover what parcellation label code (ie: structure code) corresponds.'

    label_colors: ndarray, shape (n_labels, 5)
        RGBT + label id colortable array. The first 4 values encode the label color: RGB is red, green, blue as usual, from 0 to 255 per value. T is the transparency, which is defined as 255 - alpha. The last value represents the label id. The number of labels (n_label) cannot be know in advance by this function in the general case (but the user can know based on the Atlas he is loading, e.g., the Desikan-Killiany Atlas has 36 labels).

    label_names: list of strings
       The names of the labels. The length of the list is n_labels. Note that, contrary to the respective nibabel function, this function will always return this as a list of strings, no matter the Python version used.

    meta_data: dictionary
        Contains detailed information on the data that was loaded. The following keys are available (replace `?h` with the value of the argument `hemisphere_label`, which must be 'lh' or 'rh').
            - `?h.annotation_file` : the file that was loaded

    Examples
    --------
    Load cortical parcellation annotations for both hemispheres of a subject from the Desikan-Killiany ('aparc') atlas:

    >>> import brainload as bl; import os
    >>> subjects_dir = os.path.join(os.getenv('HOME'), 'data', 'my_study_x')
    >>> vertex_color_labels, label_colors, label_names, meta_data = bl.annot('subject1', subjects_dir, 'aparc', hemi='both')
    >>> print meta_data['lh.annotation_file']     # will print /home/someuser/data/my_study_x/subject1/label/lh.aparc.annot
    >>> print meta_data['rh.annotation_file']     # will print /home/someuser/data/my_study_x/subject1/label/rh.aparc.annot


    Now load cortical parcellation annotations for the left hemisphere of a subject from the Destrieux ('aparc.a2009s') atlas:

    >>> vertex_color_labels, label_colors, label_names, meta_data = bl.annot('subject1', subjects_dir, 'aparc.a2009s', hemi='lh')
    >>> print meta_data['lh.annotation_file']     # will print /home/someuser/data/my_study_x/subject1/label/lh.aparc.a2009s.annot


    Now load cortical parcellation annotations for the right hemisphere of a subject from the DKT ('aparc.DKTatlas40') atlas:

    >>> vertex_color_labels, label_colors, label_names, meta_data = bl.annot('subject1', subjects_dir, 'aparc.DKTatlas40', hemi='rh')
    >>> print meta_data['rh.annotation_file']     # will print /home/someuser/data/my_study_x/subject1/label/lh.aparc.DKTatlas40.annot

    References
    ----------
    Atlas information is available at https://surfer.nmr.mgh.harvard.edu/fswiki/CorticalParcellation
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
        vertex_label_colors, label_colors, label_names, meta_data = read_annotation_md(lh_annotation_file, 'lh', meta_data=meta_data, orig_ids=orig_ids)
    elif hemi == 'rh':
        vertex_label_colors, label_colors, label_names, meta_data = read_annotation_md(rh_annotation_file, 'rh', meta_data=meta_data, orig_ids=orig_ids)
    else:
        lh_vertex_label_colors, lh_label_colors, lh_label_names, meta_data = read_annotation_md(lh_annotation_file, 'lh', meta_data=meta_data, orig_ids=orig_ids)
        rh_vertex_label_colors, rh_label_colors, rh_label_names, meta_data = read_annotation_md(rh_annotation_file, 'rh', meta_data=meta_data, orig_ids=orig_ids)
        #vertex_labels = merge_vertex_labels(np.array([lh_vertex_labels, rh_vertex_labels]))
        #label_colors = merge_label_colors(np.array([lh_label_colors, rh_label_colors]))
        if not _are_label_names_identical(lh_label_names, rh_label_names):
            raise ValueError("The %d labels for the lh and the %d labels for the rh are not identical for annotation '%s'." % (len(lh_label_names), len(rh_label_names), annotation))
        else:
            label_names = lh_label_names    # both are identical, so just pick any

        vertex_label_colors = np.hstack((lh_vertex_label_colors, rh_vertex_label_colors))

        if len(rh_label_colors) != len(lh_label_colors):
            raise ValueError("There are %d colors for the lh labels and %d colors for the rh labels, but they should be identical for annotation '%s'." % (len(lh_label_colors), len(rh_label_colors), annotation))

        label_colors = lh_label_colors    # both are identical, so just pick any

    return vertex_label_colors, label_colors, label_names, meta_data


def get_int_encoding_for_color(r, g, b):
    return r + (g * 256) + (b * (256**2))


def get_annot_label_and_color_for_vertex_label_color(req_vertex_label_color, label_colors):
    """
    req_vertex_label_color is the vertex_label_color for a single vertex

    label_colors os the cmap or lookup-table
    """
    for row in label_colors:
        r = row[0]
        g = row[1]
        b = row[2]
        t = row[3]
        label_id = row[4]
        #enc = get_int_encoding_for_color(r, g, b)
        if label_id == req_vertex_label_color:
            color_rgbt = (r, g, b, t)
            return label_id, color_rgbt
    return None, None


def get_annot_label_index(req_vertex_label_color, label_colors):
    relevant_row = label_colors[:, 4]
    idx_tpl = np.where(relevant_row == req_vertex_label_color)
    if len(idx_tpl) == 1:
        return idx_tpl[0][0]
    return -1


def get_annot_label_indices(req_vertex_label_colors, label_colors):
    relevant_row = label_colors[:, 4]
    bool_ar = np.isin(relevant_row, label_colors)
    return np.where(bool_ar)[0]


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


def read_annotation_md(annotation_file, hemisphere_label, meta_data=None, encoding="utf-8", orig_ids=False):
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

    orig_ids: boolean, optional
        Passed on to nibabel.freesurfer.io.read_annot function. From the documentation of that function: 'Whether to return the vertex ids as stored in the annotation file or the positional colortable ids. With orig_ids=False vertices with no id have an id set to -1.' Defaults to False.

    Returns
    -------
    vertex_label_colors: ndarray, shape (n_vertices,)
        Contains an annotation color id for each vertex listed in the annotation file. If orig_ids is False (the default), and some vertex has no annotation, -1 is returned for it. IMPORTANT: The annotation value in here is NOT the label id. It is the color for the vertex, encoded in a weird way! Yes, this is ugly. See https://surfer.nmr.mgh.harvard.edu/fswiki/LabelsClutsAnnotationFiles#Annotation for details, especially the section 'Annotation file design surprise'. The color is encoded as a single number. Quoting the linked document, the numer is the 'RGB value combined into a single 32-bit integer: annotation value = (B * 256^2) + (G * 256) + (R)'. From this it follows that, quoting the doc once more, 'Code that loads an annotation file ... has to compare annotation values to the color values in the ColorLUT part of the annotation file to discover what parcellation label code (ie: structure code) corresponds.'

    label_colors: ndarray, shape (n_labels, 5)
        RGBT + label id colortable array. The first 4 values encode the label color: RGB is red, green, blue as usual, from 0 to 255 per value. T is the transparency, which is defined as 255 - alpha. The number of labels (n_label) cannot be know in advance.

    label_names: list of strings
       The names of the labels. The length of the list is n_labels. Note that, contrary to the respective nibabel function, this function will always return this as a list of strings, no matter the Python version used.

    meta_data: dictionary
        Contains detailed information on the data that was loaded. The following keys are available (replace `?h` with the value of the argument `hemisphere_label`, which must be 'lh' or 'rh').
            - `?h.annotation_file` : the file that was loaded
    """
    if hemisphere_label not in ('lh', 'rh'):
        raise ValueError("ERROR: hemisphere_label must be one of {'lh', 'rh'} but is '%s'." % hemisphere_label)

    if meta_data is None:
        meta_data = {}

    vertex_label_colors, label_colors, label_names = fsio.read_annot(annotation_file, orig_ids=orig_ids)

    label_file = hemisphere_label + '.annotation_file'
    meta_data[label_file] = annotation_file

    # The nibabel read_annot function returns string under Python 2 and bytes under Python 3, see http://nipy.org/nibabel/reference/nibabel.freesurfer.html#nibabel.freesurfer.io.read_annot.
    # We convert this to strings here so we always return strings.
    try:
        label_names_decoded = []
        for name in label_names:
            name_str = name.decode(encoding)
            label_names_decoded.append(name_str)
        label_names = label_names_decoded
    except AttributeError:
        pass

    return vertex_label_colors, label_colors, label_names, meta_data


def read_label_md(label_file, hemisphere_label, meta_data=None):
    """
    Read label file and record meta data for it.

    A label file is a FreeSurfer text file like 'subject/label/lh.cortex.label' that contains a list of vertex ids (with RAS coordinates) that are part of the label. It may optionally contain a scalar values for each vertex, but that is currently ignored by this function.

    Parameters
    ----------
    label_file: string
        A string representing a path to a FreeSurfer vertex annotation file (e.g., the path to 'lh.cortex.label').

    hemisphere_label: {'lh' or 'rh'}
        A string representing the hemisphere this file belongs to. This is used to write the correct meta data.

    meta_data: dictionary | None, optional
        Meta data to merge into the output `meta_data`. Defaults to the empty dictionary.

    Returns
    -------
    verts_in_label: ndarray, shape (num_labeled_verts,)
        Contains an array of vertex ids, one id for each vertex that is part of the label.

    meta_data: dictionary
        Contains detailed information on the data that was loaded. The following keys are available (replace `?h` with the value of the argument `hemisphere_label`, which must be 'lh' or 'rh').
            - `?h.label_file` : the file that was loaded
    """
    if hemisphere_label not in ('lh', 'rh'):
        raise ValueError("ERROR: hemisphere_label must be one of {'lh', 'rh'} but is '%s'." % hemisphere_label)

    if meta_data is None:
        meta_data = {}

    verts_in_label = fsio.read_label(label_file, read_scalars=False)

    key_for_label_file = hemisphere_label + '.label_file'
    meta_data[key_for_label_file] = label_file

    return verts_in_label, meta_data


def label(subject_id, subjects_dir, label, hemi="both", meta_data=None):
    """
    Load annotation for the mesh vertices of a single subject.

    An annotation defines a label string and a color to each vertex, it is typically used to define brain regions, e.g., for cortical parcellation.

    Parameters
    ----------
    subject_id: string
        The subject identifier.

    subject_dir: string
        A string representing the path to the subjects dir.

    label: string
        A label to load, part of the file name of the respective file in the subjects label directory. E.g., 'cortex'.

    hemi: {'both', 'lh', 'rh'}, optional
        The hemisphere for which data should actually be loaded. Defaults to 'both'.

    meta_data: dictionary | None, optional if hemi is 'lh' or 'rh'
        Meta data to merge into the output `meta_data`. Defaults to the empty dictionary. If 'hemi' is 'both', this dictionary is required and MUST contain at least one of the keys 'lh.num_vertices' or 'lh.num_data_points', the value of which must contain the number of vertices of the left hemisphere of the subject. Background: If hemi is 'both', the vertex indices of both hemispheres are merged in the return value verts_in_label, and thus we need to know the shift, i.e., the number of vertices in the left hemisphere.

    Returns
    -------
    verts_in_label: ndarray, shape (n_vertices,)
        Contains the ids of all vertices included in the label.

    meta_data: dictionary
        Contains detailed information on the data that was loaded. The following keys are available (replace `?h` with the value of the argument `hemisphere_label`, which must be 'lh' or 'rh').
            - `?h.label_file` : the file that was loaded

    Examples
    --------
    Load the cortex label for the left hemisphere of a subject:

    >>> import brainload as bl; import os
    >>> subjects_dir = os.path.join(os.getenv('HOME'), 'data', 'my_study_x')
    >>> verts_in_label, meta_data = bl.label('subject1', subjects_dir, 'cortex', hemi='lh')
    >>> print meta_data['lh.label_file']     # will print /home/someuser/data/my_study_x/subject1/label/lh.cortex.label

    You could now use the label information to mask your morphology data.

    See also
    --------
    mask_data_using_label: Mask data using a label.
    """
    if hemi not in ('lh', 'rh', 'both'):
        raise ValueError("ERROR: hemi must be one of {'lh', 'rh', 'both'} but is '%s'." % hemi)

    if meta_data is None:
        if hemi == 'both':
            raise ValueError("Argument 'hemi' is set to 'both'. In this case, the meta_data argument is required. See the doc string for details.")
        meta_data = {}

    if hemi == 'both':
        if 'lh.num_vertices' in meta_data:
            rh_shift = meta_data['lh.num_vertices']
        elif 'lh.num_data_points' in meta_data:
            rh_shift = meta_data['lh.num_data_points']
        else:
            raise ValueError("Argument 'hemi' is set to 'both'. In this case, the meta_data argument is required and must contain the key 'lh.num_data_points' or 'lh.num_vertices'. See the doc string for details.")


    lh_label_file_name = "lh.%s.label" % label
    rh_label_file_name = "rh.%s.label" % label
    lh_label_file = os.path.join(subjects_dir, subject_id, 'label', lh_label_file_name)
    rh_label_file = os.path.join(subjects_dir, subject_id, 'label', rh_label_file_name)

    if hemi == 'lh':
        verts_in_label, meta_data = read_label_md(lh_label_file, 'lh', meta_data=meta_data)
    elif hemi == 'rh':
        verts_in_label, meta_data = read_label_md(rh_label_file, 'rh', meta_data=meta_data)
    else:
        lh_verts_in_label, meta_data = read_label_md(lh_label_file, 'lh', meta_data=meta_data)
        rh_verts_in_label, meta_data = read_label_md(rh_label_file, 'rh', meta_data=meta_data)
        rh_verts_in_label_shifted = rh_verts_in_label + rh_shift
        verts_in_label = np.hstack((lh_verts_in_label, rh_verts_in_label_shifted))

    return verts_in_label, meta_data


def label_to_mask(verts_in_label, num_verts_total, invert=False):
    """
    Generate binary mask from vertex indices.

    Generate a binary mask from the list of vertex indices in verts_in_label. The mask contains one entry for each vertex, i.e., it has length num_verts_total.

    Parameters
    ----------
    verts_in_label: 1D numpy array
        Array of vertex indices.

    num_verts_total: int
        The total number of vertices that exist. (Obviously, the highest index in verts_in_label does not need to be the last vertex.)

    invert: boolean, optional
        Whether the mask should be inverted. If inverse is set to False (or not set at all), vertex indices which occur in verts_in_label will be set to True in the mask. If inverse is set to True, vertex indices which occur in the mask will be set to False in the mask instead.  Defaults to False.

    Returns
    -------
    mask: numpy array of booleans
        The mask array, length is num_verts_total.

    See also
    --------
    mask_data_using_label: Mask data using a label.
    """
    if num_verts_total < len(verts_in_label):
        raise ValueError("Argument num_verts_total is %d but must be at least the length of verts_in_label, which is %d." % (num_verts_total, len(verts_in_label)))

    verts_in_label = np.asarray(verts_in_label)

    mask = np.zeros((num_verts_total), dtype=bool)  # all False, as 0 is False in Python when evaluated in Boolean context
    mask[verts_in_label] = True

    if invert:
        mask = np.invert(mask)
    return mask


def mask_data_using_label(data, verts_in_label, invert=False):
    """
    Mask data using a list of vertex indices.

    Set all indices in data which do NOT occur in verts_in_label to np.nan. If invert is True, set all indices which DO occur in verts_in_label to np.nan. In both cases, the other values are not altered.

    Parameters
    ----------
    data: numpy array
        Array of input data.

    verts_in_label: numpy array of int
        Each number in the array represents a vertex index in the data array.

    invert: boolean, optional
        Whether the mask should be inverted. If inverse is set to False (default), vertex indices which occur in verts_in_label will be left unaltered, and all others will be set to np.nan. If inverse is set to True, the opposite happens. Defaults to False.

    meta_data: dictionary | None, optional if hemi is 'lh' or 'rh'
        Meta data to merge into the output `meta_data`. Defaults to the empty dictionary. If 'hemi' is 'both', this dictionary is required and MUST contain at least one of the keys 'lh.num_vertices' or 'lh.num_data_points', the value of which must contain the number of vertices of the left hemisphere of the subject. Background: If hemi is 'both', the vertex indices of both hemispheres are merged in the return value verts_in_label, and thus we need to know the shift, i.e., the number of vertices in the left hemisphere.

    Returns
    -------
    numpy array of booleans
        The masked data. (This is a copy, the input data is not altered.)
    """
    mask = np.zeros((len(data)), dtype=bool)    # all False
    mask[verts_in_label] = True
    if invert:
        mask = np.invert(mask)
    masked_data = np.copy(np.asarray(data))
    masked_data[mask == False] = np.nan
    return masked_data
