"""
Read FreeSurfer vertex label and annotation files.

Functions for reading FreeSurfer vertex annotation files. These are the file in the label sub directory of a subject, with file extensions '.label' and '.annot'. Examples are 'lh.aparc.annot' and 'lh.cortex.label'. A label is a set of vertices. An annotation consists of several sets of vertices, each of which is assigned a label and a color.
"""

import os
import numpy as np
import nibabel.freesurfer.io as fsio
import brainload.nitools as nit
import brainload.freesurferdata as fsd
import logging


class AnnotQuery:
    """
    Convenience class that allows one to query the vertex names and colors for a list of vertex indices. The required parameters for the constructor are what is returned by the annot
    """
    def __init__(self, vertex_lookup_indices, label_colors, label_names, name_dtype='U50', name_null_value="None"):
        self.vertex_lookup_indices = vertex_lookup_indices
        self.label_colors = label_colors
        self.label_names = label_names
        self.name_null_value = name_null_value
        self.name_dtype = name_dtype
        self.compute_labels()

    def compute_labels(self):
        num_verts = len(self.vertex_lookup_indices)
        self.vertex_names = np.array([self.name_null_value] * num_verts, dtype=self.name_dtype)
        self.vertex_colors = np.zeros((num_verts, 4), dtype=int)
        for idx in range(num_verts):
            if self.vertex_lookup_indices[idx] >= 0:
                self.vertex_names[idx] = self.label_names[self.vertex_lookup_indices[idx]]
                self.vertex_colors[idx] = self.label_colors[self.vertex_lookup_indices[idx]][0:4]


    def get_vertex_label_names(self, query_vertex_indices):
        """
        Query the label name for a list of vertex indices.

        Parameters
        ----------
        query_vertex_indices: numpy 1D int array
            The vertex indices (in the mesh) for which you want to query the name.

        Returns
        -------
        numpy 1D string array
            Name array with shape (n, ) for n query vertices.
        """
        names = np.empty((len(query_vertex_indices), ), dtype=self.name_dtype)
        for idx, vert_idx in enumerate(query_vertex_indices):
            names[idx] = self.vertex_names[vert_idx]
        return names


    def get_vertex_label_colors(self, query_vertex_indices):
        """
        Query the label color for a list of vertex indices.

        Parameters
        ----------
        query_vertex_indices: numpy 1D int array
            The vertex indices (in the mesh) for which you want to query the color.

        Returns
        -------
        numpy 2D int array
            Color array with shape (n, 4) for n query vertices. Each color is represented by 4 int values that encode an RGBT color, where T is transparency and equal to T = alpha - 255.
        """
        colors = np.zeros((len(query_vertex_indices), 4), dtype=int)
        for idx, vert_idx in enumerate(query_vertex_indices):
            colors[idx] = self.vertex_colors[vert_idx,:]
        return colors


def get_atlas_region_names(annotation, subjects_dir, subject_id="fsaverage"):
    """
    Get the region names of the label for an annotation from the annot file of a subject.
    """
    annotation_file_lh = os.path.join(subjects_dir, subject_id, 'label', "lh.%s.annot" % (annotation))
    annotation_file_rh = os.path.join(subjects_dir, subject_id, 'label', "rh.%s.annot" % (annotation))

    if os.path.isfile(annotation_file_lh):
        vertex_labels, label_colors, label_names, meta_data = annot(subject_id, subjects_dir, annotation, hemi='lh')
        return label_names
    elif os.path.isfile(annotation_file_rh):
        vertex_labels, label_colors, label_names, meta_data = annot(subject_id, subjects_dir, annotation, hemi='rh')
        return label_names
    else:
        return None



def annot(subject_id, subjects_dir, annotation, hemi="both", meta_data=None, orig_ids=False):
    """
    Load annotation for the mesh vertices of a single subject.

    An annotation defines a label string and a color to each vertex, it is typically used to define brain regions, e.g., for cortical parcellation. An annotation consists of several groups of vertices, each of which is assigned a label and a color.

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
    vertex_labels: ndarray, shape (n_vertices,)
        If orig_ids is False (the default), returns the index (for each vertex) into the label_colors and label_names datastructures to retrieve the color and name. If some vertex has no annotation, -1 is returned for it.

        If orig_ids is True, returns an annotation color id for each vertex listed in the annotation file. IMPORTANT: The annotation value in here is NOT the label id. It is a code based on the color for the vertex. Yes, this is ugly. See https://surfer.nmr.mgh.harvard.edu/fswiki/LabelsClutsAnnotationFiles#Annotation for details, especially the section 'Annotation file design surprise'. The color is encoded as a single number. Quoting the linked document, the numer is the 'RGB value combined into a single 32-bit integer: annotation value = (B * 256^2) + (G * 256) + (R)'. From this it follows that, quoting the doc once more, 'Code that loads an annotation file ... has to compare annotation values to the color values in the ColorLUT part of the annotation file to discover what parcellation label code (ie: structure code) corresponds.' (Basically this has already been done for you if you simply set orig_ids to False.)

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
    >>> vertex_labels, label_colors, label_names, meta_data = bl.annot('subject1', subjects_dir, 'aparc', hemi='both')
    >>> print meta_data['lh.annotation_file']     # will print /home/someuser/data/my_study_x/subject1/label/lh.aparc.annot
    >>> print meta_data['rh.annotation_file']     # will print /home/someuser/data/my_study_x/subject1/label/rh.aparc.annot


    Now load cortical parcellation annotations for the left hemisphere of a subject from the Destrieux ('aparc.a2009s') atlas:

    >>> vertex_labels, label_colors, label_names, meta_data = bl.annot('subject1', subjects_dir, 'aparc.a2009s', hemi='lh')
    >>> print meta_data['lh.annotation_file']     # will print /home/someuser/data/my_study_x/subject1/label/lh.aparc.a2009s.annot


    Now load cortical parcellation annotations for the right hemisphere of a subject from the DKT ('aparc.DKTatlas40') atlas:

    >>> vertex_labels, label_colors, label_names, meta_data = bl.annot('subject1', subjects_dir, 'aparc.DKTatlas40', hemi='rh')
    >>> print meta_data['rh.annotation_file']     # will print /home/someuser/data/my_study_x/subject1/label/lh.aparc.DKTatlas40.annot

    Print the color and the annotation name for an example vertex:

    >>> vert_idx = 0     # We'll take the first vertex as an example.
    >>> if vertex_labels[vert_idx] >= 0:     # it is -1 if the vertex is not assigned any label/color
    >>>     i = vertex_labels[vert_idx]
    >>>     print "label for vertex %d is %s" % (vert_idx, label_names[i])
    >>>     print "color for vertex %d in RGBA is (%d %d %d %d)" % (vert_idx, label_colors[idx, 0], label_colors[idx, 1], label_colors[idx, 2], (255 - label_colors[idx, 3]))

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
        vertex_labels, label_colors, label_names, meta_data = read_annotation_md(lh_annotation_file, 'lh', meta_data=meta_data, orig_ids=orig_ids)
    elif hemi == 'rh':
        vertex_labels, label_colors, label_names, meta_data = read_annotation_md(rh_annotation_file, 'rh', meta_data=meta_data, orig_ids=orig_ids)
    else:
        lh_vertex_labels, lh_label_colors, lh_label_names, meta_data = read_annotation_md(lh_annotation_file, 'lh', meta_data=meta_data, orig_ids=orig_ids)
        rh_vertex_labels, rh_label_colors, rh_label_names, meta_data = read_annotation_md(rh_annotation_file, 'rh', meta_data=meta_data, orig_ids=orig_ids)
        if not lh_label_names == rh_label_names:
            raise ValueError("The %d labels for the lh and the %d labels for the rh are not identical for annotation '%s'." % (len(lh_label_names), len(rh_label_names), annotation))
        else:
            label_names = lh_label_names    # both are identical, so just pick any

        vertex_labels = np.hstack((lh_vertex_labels, rh_vertex_labels))

        if len(rh_label_colors) != len(lh_label_colors):
            raise ValueError("There are %d colors for the lh labels and %d colors for the rh labels, but they should be identical for annotation '%s'." % (len(lh_label_colors), len(rh_label_colors), annotation))

        label_colors = lh_label_colors    # both are identical, so just pick any

    return vertex_labels, label_colors, label_names, meta_data


def region_stats(region_data_per_hemi, label_names):
    """
    Compute descriptive stats for all regions. Return as 2D matrix.
    """
    nan = float('nan')
    descriptor_names = []
    descriptor_data = []
    base_stat_names = ["min", "max", "mean", "std", "median", "25perc", "75perc"]
    for hemi in region_data_per_hemi:
        for region in label_names:
            try:
                rd = region_data_per_hemi[hemi][region]
                desc_stats = [np.nanmin(rd), np.nanmax(rd), np.nanmean(rd), np.nanstd(rd), np.nanmedian(rd), np.percentile(rd, 25), np.nanpercentile(rd, 75)]
            except (ValueError, KeyError):  # raised on empty array or if there is no data on that region
                desc_stats = [nan] * 7
            descriptor_data.extend(desc_stats)
            descriptor_names_this_hemi_and_region = ["%s_%s_%s" % (hemi, region, n) for n in base_stat_names]
            descriptor_names.extend(descriptor_names_this_hemi_and_region)
    return np.array(descriptor_data), descriptor_names


def region_data_native(subject_id, subjects_dir, annotation, hemi, morphometry_data, morphometry_meta_data):
    """
    Get morphometry data for atlas regions.

    Get morphometry data for each region in an annotation file/atlas. You can use these to compute anatomical statistics per region, like average cortical thickness.

    Parameters
    ----------
    subject_id: string
        The subject identifier.

    subjects_dir: string
        A string representing the path to the subjects dir.

    annotation: string
        An annotation to load, part of the file name of the respective file in the subjects label directory. E.g., 'aparc', 'aparc.a2009s', or 'aparc.DKTatlas'.

    hemi: string, one of {'both', 'lh', 'rh'}
        The hemisphere for which data should actually be loaded.

    morphometry_data: numpy array
        morphometry data array, one value per vertex

    morphometry_meta_data: dictionary
        morphometry meta data dictionary, as returned by the functions that load morphometry data

    Returns
    -------
    region_data_per_hemi: nested dictionary
        Has a key for each hemi you requested (only 'lh', only 'rh', or both 'lh' and 'rh'). The inner one has the region names as keys, and a 1D numpy array of values.

    label_names: list of strings
        Names of the atlas regions. The length differs by atlas/annotation file.

    Examples
    --------
    Let us compute the average thickness in region 5 of the aparc atlas for a subject:
    >>> hemi = 'lh'
    >>> morphometry_data, morphometry_meta_data = bl.subject_data_native('subject1', TEST_DATA_DIR, 'thickness', hemi)
    >>> region_data, label_names = an.region_data_native('subject1', TEST_DATA_DIR, 'aparc', hemi, morphometry_data, morphometry_meta_data)
    >>> region_6_name = label_names[5]
    >>> mean_thickness_in_region_6 = np.mean(region_data[hemi][region_6_name])
    >>> print("Avg thickness in region '%s' is: %f" % (region_6_name, mean_thickness_in_region_6))
    """
    if hemi not in ('lh', 'rh', 'both'):
        raise ValueError("ERROR: hemi must be one of {'lh', 'rh', 'both'} but is '%s'." % hemi)

    vertex_labels, label_colors, label_names, annot_meta_data = annot(subject_id, subjects_dir, annotation, hemi=hemi)

    region_data_per_hemi = dict()

    hemis = []
    if hemi == "both":
        hemis.append("lh")
        hemis.append("rh")
    else:
        hemis.append(hemi)

    for h in hemis:
        s, e = fsd.hemi_range(morphometry_meta_data, h)
        hemi_data = morphometry_data[s:e]
        hemi_labels = vertex_labels[s:e]
        region_data = _split_morph_data_into_regions(hemi_data, hemi_labels, label_names)
        region_data_per_hemi[h] = region_data
    return region_data_per_hemi, label_names


def _split_morph_data_into_regions(hemi_data, vertex_labels, label_names):
    """
    Split morphometry data into subsets based on atlas regions.

    Split morphometry data for a single hemisphere of one subject into subsets based on atlas regions.

    Parameters
    ----------
    hemi_data: numpy 1D array
        The morphometry data for a single hemisphere.

    vertex_labels: numpy 1D array
        Vertex labels ONLY for the vertices of the hemisphere. Shape is (n, ), where n is the number of vertices of the hemisphere. Each value is an index into label_names.

    label_names: list of strings
        The atlas region names.

    Returns
    -------
    region_data: dictionary
        Each key is a region name (string), and each value is a 1D numpy array of all morphometry values in the region (a subset of hemi_data).
    """
    data = dict()
    for label_idx, label_name in enumerate(label_names):
        indices = np.where(vertex_labels == label_idx)[0]
        data[label_name] = hemi_data[indices]
    return data


def color_rgbt_to_rgba(rgbt):
    """
    Convert RGBT color to RGBA.

    Convert an RGBT color given as (r, g, b, t) with all values in range [0.255] to the respective color in RGBA. The T is for transparency, an defined as 1 - alpha, where alpha is the RGBA value A.

    Parameters
    ----------
    rgbt: tupel of 4 integers (in range 0..255)
        The color according to RGBT definition, where T is transparency.

    Returns
    -------
    tupel of 4 integers (in range 0..255)
        The color in RGBA notation.

    Examples
    --------
    Convert a color from RGBT to RGBA:

    >>> import brainload.annotations as an
    >>> an.color_rgbt_to_rgba((120, 0, 240, 40))
    (120, 0, 240, 215)
    """
    rgba = (rgbt[0], rgbt[1], rgbt[2], 255 - rgbt[3])
    return rgba


def _get_annot_label_index(vertex_label, label_colors):
    """
    Retrieve relevant index in the label_colors and label_names datastructures for a single vertex.

    Retrieve the relevant index in the label_colors and label_names datastructures for the label carried by a single vertex (in vertex_label_colors). This function can most likely be removed, get_annot_label_indices does the same for all at once.
    """
    relevant_row = label_colors[:, 4]
    idx_tpl = np.where(relevant_row == vertex_label)
    if len(idx_tpl[0]) == 1:
        return idx_tpl[0][0]
    return -1


def _get_indices_for_unique_vertex_labels(all_vertex_labels, label_colors):
    """
    Retrieve relevant indices in the label_colors and label_names datastructures for all vertices.

    Retrieve the relevant index in the label_colors and label_names datastructures for the labels carried by the vertices in vertex_label_colors.

    Returns
    -------
    Dict of int, int
        A dictionary that maps each unique vc_code to an index. The indices can be used to access the corresponding label_color and label_name.
    """
    unique_vlabels = np.unique(all_vertex_labels)
    vlabel_to_idx_map = {}
    for idx, uvl in enumerate(unique_vlabels):
        vlabel_to_idx_map[uvl] = _get_annot_label_index(uvl, label_colors)
    return vlabel_to_idx_map


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

    logging.info("Reading annotation file '%s'." % (annotation_file))
    vertex_label_colors, label_colors, label_names = fsio.read_annot(annotation_file, orig_ids=orig_ids)

    label_file = hemisphere_label + '.annotation_file'
    meta_data[label_file] = annotation_file

    # The nibabel read_annot function returns string under Python 2 and bytes under Python 3, see http://nipy.org/nibabel/reference/nibabel.freesurfer.html#nibabel.freesurfer.io.read_annot.
    # We convert this to strings here so we always return strings.
    try:
        label_names_decoded = [name.decode(encoding) for name in label_names]
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


def read_vertex_list_file(file_name, sep=' '):
    """
    Read a vertex list from a text file.

    Read a vertex list from a text file. The file must contain vertex indices (integers) separated by spaces. It is OK if it has several lines. Lines starting with '#' are considered to be comments and ignored. Note that a vertex is given by its index in a brain mesh, starting at 0. This is used as a way to import a list of vertices, similar to a FreeSurfer label file. This is a brainload custom format, but a very simple one that follows common file format conventions.

    Parameters
    ----------
    file_name: str
        Path to a file to read

    sep: str, optional
        Separator between integer values on a line. Defaults to a space, ' '.

    Returns
    -------
    numpy 1D int array
        The data read from all non-comment lines.
    """
    verts = np.array([], dtype=np.int)
    lines = nit._read_text_file_lines(file_name)
    for line in lines:
        if line.startswith('#'):
            continue
        else:
            line_data = np.fromstring(line, dtype=int, sep=sep)
            verts = np.append(verts, line_data)
    return verts


def vertices_to_label(selected_vert_indices, all_vert_coords, header="#!ascii label"):
    """
    Write a string in FreeSurfer label format from the vertices.

    Write a string in FreeSurfer label format from the vertices. This can be used to create a label from a list of vertices, e.g., for displaying the vertices in Freeview or other tools supporting FreeSurfers label file format.
    """
    res = [header]
    res.append("%d" % len(selected_vert_indices))
    for idx in selected_vert_indices:
        res.append("%d %f %f %f 0.0000000000" % (idx, all_vert_coords[idx][0], all_vert_coords[idx][1], all_vert_coords[idx][2]))
    return '\n'.join(res)


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
