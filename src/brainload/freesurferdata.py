"""
Functions for loading FreeSurfer data on different levels.
"""

import os
import sys
import errno
import numpy as np
import nibabel.freesurfer.io as fsio
import nibabel.freesurfer.mghformat as fsmgh
import brainload.nitools as nit


def read_mgh_file(mgh_file_name, collect_meta_data=True):
    """
    Read data from a FreeSurfer output file in mgh format.

    Read all data from the MGH file and return it as a numpy array. Optionally, collect meta data from the mgh file header.

    Parameters
    ----------
    mgh_file_name: string
        A string representing a full path to a file in FreeSurfer MGH file format.

    collect_meta_data: bool, optional
        Whether or not to collect meta data from the MGH file header. Defaults to True.

    Returns
    -------
    mgh_data: numpy array
        The data from the MGH file, usually one scalar value per voxel.

    mgh_meta_data: dictionary
        The meta data collected from the header, or an empty dictionary if the argument `collect_meta_data` was 'False'. The keys correspond to the names of the respective nibabel function used to retrieve the data. The values are the data as returned by nibabel.

    See also
    --------
        - https://surfer.nmr.mgh.harvard.edu/fswiki/CoordinateSystems
        - https://github.com/nipy/nibabel/blob/master/nibabel/freesurfer/mghformat.py
        - https://surfer.nmr.mgh.harvard.edu/fswiki/FileFormats
    """
    mgh_meta_data = {}
    with open(mgh_file_name, 'r') as mgh_file_handle:
        header = fsmgh.MGHHeader.from_fileobj(mgh_file_handle)

        if collect_meta_data:
            mgh_meta_data['data_shape'] = header.get_data_shape()
            mgh_meta_data['affine'] = header.get_affine()
            mgh_meta_data['best_affine'] = header.get_best_affine()             # identical to get_affine for MGH format
            mgh_meta_data['data_bytespervox'] = header.get_data_bytespervox()
            mgh_meta_data['data_dtype'] = header.get_data_dtype()
            mgh_meta_data['data_offset'] =  header.get_data_offset()            # MGH format has a header, then data, then a footer
            mgh_meta_data['data_shape'] =  header.get_data_shape()
            mgh_meta_data['data_size'] = header.get_data_size()
            mgh_meta_data['footer_offset'] =  header.get_footer_offset()
            mgh_meta_data['ras2vox'] =  header.get_ras2vox()
            mgh_meta_data['slope_inter'] =  header.get_slope_inter()
            mgh_meta_data['vox2ras'] =  header.get_vox2ras()
            mgh_meta_data['vox2ras_tkr'] =  header.get_vox2ras_tkr()
            mgh_meta_data['zooms'] =  header.get_zooms()                        # the voxel dimensions (along all 3 axes in space)

        mgh_data = header.data_from_fileobj(mgh_file_handle)
        mgh_file_handle.close()
    return mgh_data, mgh_meta_data


def merge_morphology_data(morphology_data_arrays, dtype=float):
    """
    Merge morphology data horizontally.

    Merge morphology data read from several meshes of the same subject horizontally. This is used to merge data from the left and right hemispheres.

    Parameters
    ----------
    morphology_data_arrays: 2D array
        An array of arrays, each of which represents morphology data from different hemispheres of the same subject.

    dtype: data type, optional
        Data type for the output numpy array. Defaults to float.

    Returns
    -------
    numpy array
        Horizontally stacked array containing the data from all arrays in the input array.
    """
    merged_data = np.empty((0), dtype=dtype)
    for morphology_data in morphology_data_arrays:
        merged_data = np.hstack((merged_data, morphology_data))
    return merged_data


def _get_morphology_data_suffix_for_surface(surf):
    """
    Determine FreeSurfer surface representation string.

    Determine the substring representing the given surface in a FreeSurfer output curv file. For FreeSurfer's default surface 'white', the surface is not represented in the output file name pattern. For all others, it is represented by a dot followed by the name.

    Parameters
    ----------
    surf: string
        A string representing a FreeSurfer surface, e.g., 'white' or 'pial'.

    Returns
    -------
    string
        The empty string if `surf` is 'white'. A dot followed by the string in the input argument `surf` otherwise.
    """
    if surf == 'white':
        return ''
    return '.' + surf


def read_fs_surface_file_and_record_meta_data(surf_file, hemisphere_label, meta_data=None):
    """
    Read a surface file and record meta data on it.

    Read a surface file and record meta data on it. A surface file is a mesh file in FreeSurfer format, e.g., 'lh.white'. It contains vertices and 3-faces made out of them.

    Parameters
    ----------
    surf_file: string
        A string representing an absolute path to a surface (or 'mesh') file (e.g., the path to 'lh.white').

    hemisphere_label: {'lh' or 'rh'}
        A string representing the hemisphere this file belongs to. This is used to write the correct meta data.

    meta_data: dictionary | None, optional
        Meta data to merge into the output `meta_data`. Defaults to the empty dictionary.

    Returns
    -------
    vert_coords: numpy array
        A 2D array containing 3 coordinates for each vertex in the `surf_file`.

    faces: numpy array
        A 2D array containing 3 vertex indices per face. Look at the respective indices in `vert_coords` to get the vertex coordinates.

    meta_data: dictionary
        Contains detailed information on the data that was loaded.
    """
    if hemisphere_label not in ('lh', 'rh'):
        raise ValueError("ERROR: hemisphere_label must be one of {'lh', 'rh'} but is '%s'." % hemisphere_label)

    if meta_data is None:
        meta_data = {}

    vert_coords, faces = fsio.read_geometry(surf_file)

    label_num_vertices = hemisphere_label + '.num_vertices'
    meta_data[label_num_vertices] = vert_coords.shape[0]

    label_num_faces = hemisphere_label + '.num_faces'
    meta_data[label_num_faces] = faces.shape[0]

    label_surf_file = hemisphere_label + '.surf_file'
    meta_data[label_surf_file] = surf_file

    return vert_coords, faces, meta_data


def read_fs_morphology_data_file_and_record_meta_data(curv_file, hemisphere_label, meta_data=None, format='curv'):
    """
    Read a morphology file and record meta data on it.

    Read a morphology file and record meta data on it. A morphology file is file containing a scalar value for each vertex on the surface of a FreeSurfer mesh. An example is the file 'lh.area', which contains the area values for all vertices of the left hemisphere of the white surface. Such a file can be in two different formats: 'curv' or 'mgh'. The former is used when the data refers to the surface mesh of the original subject, the latter when it has been mapped to a standard subject like fsaverage.

    Parameters
    ----------
    curv_file: string
        A string representing an absolute path to a morphology file (e.g., the path to 'lh.area').

    hemisphere_label: {'lh' or 'rh'}
        A string representing the hemisphere this file belongs to. This is used to write the correct meta data.

    hemi: {'both', 'lh', 'rh'}, optional
        The hemisphere for which data should actually be loaded. Defaults to 'both'.

    meta_data: dictionary | None, optional
        Meta data to merge into the output `meta_data`. Defaults to the empty dictionary.

    format: {'curv', 'mgh'}, optional
        The file format for the files that are to be loaded. Defaults to 'curv'.

    Returns
    -------
    per_vertex_data: numpy array
        A 1D array containing one scalar value per vertex.

    meta_data: dictionary
        Contains detailed information on the data that was loaded.
    """
    if format not in ('curv', 'mgh'):
        raise ValueError("ERROR: format must be one of {'curv', 'mgh'} but is '%s'." % format)

    if hemisphere_label not in ('lh', 'rh'):
        raise ValueError("ERROR: hemisphere_label must be one of {'lh', 'rh'} but is '%s'." % hemisphere_label)

    if meta_data is None:
        meta_data = {}

    if format == 'mgh':
        full_mgh_data, mgh_meta_data = read_mgh_file(curv_file, collect_meta_data=False)
        relevant_data_inner_array = full_mgh_data[:,0]
        per_vertex_data = relevant_data_inner_array[:,0]
    else:
        per_vertex_data = fsio.read_morph_data(curv_file)

    label_num_values = hemisphere_label + '.num_data_points'
    meta_data[label_num_values] = per_vertex_data.shape[0]

    label_file = hemisphere_label + '.morphology_file'
    meta_data[label_file] = curv_file

    label_file_format = hemisphere_label + '.morphology_file_format'
    meta_data[label_file_format] = format

    return per_vertex_data, meta_data


def load_subject_mesh_files(lh_surf_file, rh_surf_file, hemi='both', meta_data=None):
    """
    Load mesh files for a subject.

    Load one or two mesh files for a subject. Which of the two files `lh_surf_file` and `rh_surf_file` are actually loaded is determined by the `hemi` parameter.

    Parameters
    ----------
    lh_surf_file: string | None
        A string representing an absolute path to a mesh file for the left hemisphere (e.g., the path to 'lh.white'). If `hemi` is 'rh', this will be ignored and can thus be None.

    rh_surf_file: string | None
        A string representing an absolute path to a mesh file for the right hemisphere (e.g., the path to 'rh.white'). If `hemi` is 'lh', this will be ignored and can thus be None.

    hemi: {'both', 'lh', 'rh'}, optional
        The hemisphere for which data should actually be loaded. Defaults to 'both'.

    meta_data: dictionary | None, optional
        Meta data to merge into the output `meta_data`. Defaults to the empty dictionary.

    Returns
    -------
    vert_coords: numpy array
        A 2D array containing 3 coordinates for each vertex. If the argument `hemi` was 'both', this includes vertices from several meshes. You can check the `meta_data` return values to get the border between meshes, see `meta_data['lh.num_vertices']` and  `meta_data['rh.num_vertices']`.

    faces: numpy array
        A 2D array containing 3 vertex indices per face. Look at the respective indices in `vert_coords` to get the vertex coordinates. If the argument `hemi` was 'both', this includes faces from several meshes. You can check the `meta_data` return values to get the border between meshes, see `meta_data['lh.num_faces']` and  `meta_data['rh.num_faces']`.

    meta_data: dictionary
        Contains detailed information on the data that was loaded.
    """
    if hemi not in ('lh', 'rh', 'both'):
        raise ValueError("ERROR: hemi must be one of {'lh', 'rh', 'both'} but is '%s'." % hemi)

    if meta_data is None:
        meta_data = {}

    if hemi == 'lh':
        vert_coords, faces, meta_data = read_fs_surface_file_and_record_meta_data(lh_surf_file, 'lh', meta_data=meta_data)
    elif hemi == 'rh':
        vert_coords, faces, meta_data = read_fs_surface_file_and_record_meta_data(rh_surf_file, 'rh', meta_data=meta_data)
    else:
        lh_vert_coords, lh_faces, meta_data = read_fs_surface_file_and_record_meta_data(lh_surf_file, 'lh', meta_data=meta_data)
        rh_vert_coords, rh_faces, meta_data = read_fs_surface_file_and_record_meta_data(rh_surf_file, 'rh', meta_data=meta_data)
        vert_coords, faces = _merge_meshes(np.array([[lh_vert_coords, lh_faces], [rh_vert_coords, rh_faces]]))
    return vert_coords, faces, meta_data


def load_subject_morphology_data_files(lh_morphology_data_file, rh_morphology_data_file, hemi='both', format='curv', meta_data=None):
    """
    Load morphology data files for a subject.

    Load one or two morphology data files for a subject. Which of the two files `lh_morphology_data_file` and `rh_morphology_data_file` are actually loaded is determined by the `hemi` parameter.

    Parameters
    ----------
    lh_morphology_data_file: string | None
        A string representing an absolute path to a morphology data file for the left hemisphere. If `hemi` is 'rh', this will be ignored and can thus be None.

    rh_morphology_data_file: string | None
        A string representing an absolute path to a morphology data file for the right hemisphere. If `hemi` is 'lh', this will be ignored and can thus be None.

    hemi: {'both', 'lh', 'rh'}, optional
        The hemisphere for which data should actually be loaded. Defaults to 'both'.

    format: {'curv', 'mgh'}, optional
        The file format for the files that are to be loaded. Defaults to 'curv'.

    meta_data: dictionary | None, optional
        Meta data to merge into the output `meta_data`. Defaults to the empty dictionary.

    Returns
    -------
    morphology_data: numpy array
        An array containing the scalar per-vertex data loaded from the file(s).

    meta_data: dictionary
        Contains detailed information on the data that was loaded.
    """
    if hemi not in ('lh', 'rh', 'both'):
        raise ValueError("ERROR: hemi must be one of {'lh', 'rh', 'both'} but is '%s'." % hemi)

    if format not in ('curv', 'mgh'):
        raise ValueError("ERROR: format must be one of {'curv', 'mgh'} but is '%s'." % format)

    if meta_data is None:
        meta_data = {}

    if hemi == 'lh':
        morphology_data, meta_data = read_fs_morphology_data_file_and_record_meta_data(lh_morphology_data_file, 'lh', meta_data=meta_data, format=format)
    elif hemi == 'rh':
        morphology_data, meta_data = read_fs_morphology_data_file_and_record_meta_data(rh_morphology_data_file, 'rh', meta_data=meta_data, format=format)
    else:
        lh_morphology_data, meta_data = read_fs_morphology_data_file_and_record_meta_data(lh_morphology_data_file, 'lh', meta_data=meta_data, format=format)
        rh_morphology_data, meta_data = read_fs_morphology_data_file_and_record_meta_data(rh_morphology_data_file, 'rh', meta_data=meta_data, format=format)
        morphology_data = merge_morphology_data(np.array([lh_morphology_data, rh_morphology_data]))
    return morphology_data, meta_data


def parse_subject(subject_id, surf='white', measure='area', hemi='both', subjects_dir=None, meta_data=None, load_surface_files=True, load_morhology_data=True):
    """
    Parse FreeSurfer brain data for a single subject.

    High-level interface to parse FreeSurfer brain data for a single space. This parses the data for the surfaces of this subject. If you want to load data that has been mapped to an average subject like 'fsaverage', use `parse_subject_standard_space_data` instead.

    Parameters
    ----------
    subject_id: string
        The subject identifier of the subject. As always, it is assumed that this is the name of the directory containing the subject's data, relative to `subjects_dir`. Example: 'subject33'.

    measure : string, optional
        The measure to load, e.g., 'area' or 'curv'. Defaults to 'area'.

    surf : string, optional
        The brain surface where the data has been measured, e.g., 'white' or 'pial'. This will become part of the file name that is loaded. Defaults to 'white'.

    hemi : {'both', 'lh', 'rh'}, optional
        The hemisphere that should be loaded. Defaults to 'both'.

    subjects_dir: string, optional
        A string representing the full path to a directory. This should be the directory containing all subjects of your study. Defaults to the environment variable SUBJECTS_DIR if omitted. If that is not set, used the current working directory instead. This is the directory from which the application was executed.

    meta_data: dictionary, optional
        A dictionary that should be merged into the return value `meta_data`. Defaults to the empty dictionary if omitted.

    load_surface_files: boolean, optional
        Whether to load mesh data. If set to `False`, the first return values `vert_coords` and `faces` will be `None`. Defaults to `True`.

    load_morphology_data: boolean, optional
        Whether to load morphology data. If set to `False`, the first return value `morphology_data` will be `None`. Defaults to `True`.

    Returns
    -------
    vert_coords: numpy array
        A 2-dimensional array containing the vertices of the mesh(es) of the subject. Each vertex entry contains 3 coordinates. Each coordinate describes a 3D position in a FreeSurfer surface file (e.g., 'lh.white'), as returned by the `nibabel` function `nibabel.freesurfer.io.read_geometry`.

    faces: numpy array
        A 2-dimensional array containing the 3-faces of the mesh(es) of the subject. Each face entry contains 3 indices. Each index references the respective vertex in the `vert_coords` array.

    morphology_data: numpy array
        A numpy array with as many entries as there are vertices in the subject. If you load two hemispheres instead of one, the length doubles. You can get the start indices for data of the hemispheres in the returned `meta_data`, see `meta_data['lh.num_vertices']` and `meta_data['rh.num_vertices']`. You can be sure that the data for the left hemisphere will always come first (if both were loaded). Indices start at 0, of course. So if the left hemisphere has `n` vertices, the data for them are at indices `0..n-1`, and the data for the right hemisphere start at index `n`. Note that the two hemispheres do in general NOT have the same number of vertices.

    meta_data: dictionary
        A dictionary containing detailed information on all files that were loaded and used settings.

    Raises
    ------
    ValueError
        If one of the parameters with a fixed set of values receives a value that is not allowed.

    Examples
    --------
    Load area data for both hemispheres and white surface of subject1 in the directory defined by the environment variable SUBJECTS_DIR:

    >>> v, f, data, md = parse_subject('subject1')

    Here, we are a bit more explicit about what we want to load:

    >>> import os
    >>> user_home = os.getenv('HOME')
    >>> subjects_dir = os.path.join(user_home, 'data', 'my_study_x')
    >>> v, f, data, md = parse_subject('subject1', hemi='lh', measure='curv', subjects_dir=subjects_dir)

    Sometime we do not care for the mesh, e.g., we only want the morphometry data:

    >>> data, md = parse_subject('subject1', hemi='rh', fwhm='15', load_surface_files=False)[2:4]

    """
    if hemi not in ('lh', 'rh', 'both'):
        raise ValueError("ERROR: hemi must be one of {'lh', 'rh', 'both'} but is '%s'." % hemi)

    if meta_data is None:
        meta_data = {}

    if subjects_dir is None:
        subjects_dir = os.getenv('SUBJECTS_DIR', os.getcwd())
    subject_surf_dir = os.path.join(subjects_dir, subject_id, 'surf')

    vert_coords = None
    faces = None
    if load_surface_files:
        display_subject = subject_id
        display_surf = surf
        lh_surf_file = os.path.join(subject_surf_dir, ('lh.' + surf))
        rh_surf_file = os.path.join(subject_surf_dir, ('rh.' + surf))
        vert_coords, faces, meta_data = load_subject_mesh_files(lh_surf_file, rh_surf_file, hemi=hemi, meta_data=meta_data)
    else:
        display_subject = None
        display_surf = None

    morphology_data = None
    if load_morhology_data:
        lh_morphology_file = os.path.join(subject_surf_dir, ('lh.' + measure + _get_morphology_data_suffix_for_surface(surf)))
        rh_morphology_file = os.path.join(subject_surf_dir, ('rh.' + measure + _get_morphology_data_suffix_for_surface(surf)))
        morphology_data, meta_data = load_subject_morphology_data_files(lh_morphology_file, rh_morphology_file, hemi=hemi, format='curv', meta_data=meta_data)
    else:
        measure = None


    meta_data['subject_id'] = subject_id
    meta_data['display_subject'] = display_subject
    meta_data['subjects_dir'] = subjects_dir
    meta_data['surf'] = surf
    meta_data['display_surf'] = display_surf
    meta_data['measure'] = measure
    meta_data['space'] = 'native_space'
    meta_data['hemi'] = hemi

    return vert_coords, faces, morphology_data, meta_data


def _merge_meshes(meshes):
    """
    Merge several meshes into a single one.

    Merge a list of meshes into a single one. Each mesh is given by a vertex list and a face list. The merged vertex list is just a vstack of the individual lists, and the vertex coordinates are not altered in any way. As a face id defined by the indices of its 3 vertices, these indices get adjusted for all meshes but the first one.

    Parameters
    ----------
    meshes: array-like
        A array of meshes. Each mesh is represented as an array of length 2, where the entry at index 0 is the vertex list, and the one at index 2 is the face list.

    Returns
    -------
    all_vert_coords: numpy array
        An array of vertex coordinates with shape(3, n), where n is the sum of the vertex counts of all input meshes.

    all_faces: numpy_array (2d)
        An array of faces with shape(3, m), where m is the sum of the face counts of all input meshes. For each face, each of its 3 values represent the vertex at the respective index in the `all_vert_coords` array.
    """
    all_vert_coords = np.empty((0, 3), dtype=float)
    all_faces = np.empty((0, 3), dtype=int)

    for mesh in meshes:
        new_vert_coords = mesh[0]
        new_faces = mesh[1]
        # Keep track of the total number of vertices we had *before* adding the new ones. This is the shift we need for the faces.
        vertex_index_shift = all_vert_coords.shape[0]
        all_vert_coords = np.vstack((all_vert_coords, new_vert_coords))

        new_faces_shifted = new_faces + vertex_index_shift
        all_faces = np.vstack((all_faces, new_faces_shifted))
    return all_vert_coords, all_faces


def parse_subject_standard_space_data(subject_id, measure='area', surf='white', display_surf='white', hemi='both', fwhm='10', subjects_dir=None, average_subject='fsaverage', subjects_dir_for_average_subject=None, meta_data=None, load_surface_files=True, load_morhology_data=True, custom_morphology_files=None):
    """
    Load morphometry data for a subjects that has been mapped to an average subject.

    Load data for a single subject that has been mapped to an average subject like the `fsaverage` subject from FreeSurfer. Can also load the mesh of an arbitrary surface for the average subject.

    Parameters
    ----------
    subject_id: string
        The subject identifier of the subject. As always, it is assumed that this is the name of the directory containing the subject's data, relative to `subjects_dir`. Example: 'subject33'.

    measure : string, optional
        The measure to load, e.g., 'area' or 'curv'. Defaults to 'area'.

    surf : string, optional
        The brain surface where the data has been measured, e.g., 'white' or 'pial'. This will become part of the file name that is loaded. Defaults to 'white'.

    hemi : {'both', 'lh', 'rh'}, optional
        The hemisphere that should be loaded. Defaults to 'both'.

    fwhm : string or None, optional
        Which averaging version of the data should be loaded. FreeSurfer usually generates different standard space files with a number of smoothing settings. Defaults to '10'. If None is passed, the `.fwhmX` part is omitted from the file name completely. Set this to '0' to get the unsmoothed version.

    subjects_dir: string, optional
        A string representing the full path to a directory. This should be the directory containing all subjects of your study. Defaults to the environment variable SUBJECTS_DIR if omitted. If that is not set, used the current working directory instead. This is the directory from which the application was executed.

    average_subject: string, optional
        The name of the average subject to which the data was mapped. Defaults to 'fsaverage'.

    display_surf: string, optional
        The surface of the average subject for which the mesh should be loaded, e.g., 'white', 'pial', 'inflated', or 'sphere'. Defaults to 'white'. Ignored if `load_surface_files` is `False`.

    subjects_dir_for_average_subject: string, optional
        A string representing the full path to a directory. This can be used if the average subject is not in the same directory as all your study subjects. Defaults to the setting of `subjects_dir`.

    meta_data: dictionary, optional
        A dictionary that should be merged into the return value `meta_data`. Defaults to the empty dictionary if omitted.

    load_surface_files: boolean, optional
        Whether to load mesh data. If set to `False`, the first return values `vert_coords` and `faces` will be `None`. Defaults to `True`.

    load_morphology_data: boolean, optional
        Whether to load morphology data. If set to `False`, the first return value `morphology_data` will be `None`. Defaults to `True`.

    custom_morphology_files: dictionary, optional
        Cutom filenames for the left and right hemispjere data files that should be loaded. A dictionary of strings with exactly the following two keys: `lh` and `rh`. The value strings must contain hardcoded file names or template strings for them. As always, the files will be loaded relative to the `surf/` directory of the respective subject. Example: `{'lh': 'lefthemi.nonstandard.mymeasure44.mgh', 'rh': 'righthemi.nonstandard.mymeasure44.mgh'}`.

    Returns
    -------
    vert_coords: numpy array
        A 2-dimensional array containing the vertices of the mesh(es) of the average subject. Each vertex entry contains 3 coordinates. Each coordinate describes a 3D position in a FreeSurfer surface file (e.g., 'lh.white'), as returned by the `nibabel` function `nibabel.freesurfer.io.read_geometry`.

    faces: numpy array
        A 2-dimensional array containing the 3-faces of the mesh(es) of the average subject. Each face entry contains 3 indices. Each index references the respective vertex in the `vert_coords` array.

    morphology_data: numpy array
        A numpy array with as many entries as there are vertices in the average subject. If you load two hemispheres instead of one, the length doubles. You can get the start indices for data of the hemispheres in the returned `meta_data`, see `meta_data['lh.num_vertices']` and `meta_data['rh.num_vertices']`. You can be sure that the data for the left hemisphere will always come first (if both were loaded). Indices start at 0, of course. So if the left hemisphere has `n` vertices, the data for them are at indices `0..n-1`, and the data for the right hemisphere start at index `n`. In many cases, your average subject will have the same number of vertices for both hemispheres and you will know this number beforehand, so you may not have to worry about this at all.

    meta_data: dictionary
        A dictionary containing detailed information on all files that were loaded and used settings.

    Raises
    ------
    ValueError
        If one of the parameters with a fixed set of values receives a value that is not allowed.

    Examples
    --------
    Load area data for both hemispheres and white surface of subject1 in the directory defined by the environment variable SUBJECTS_DIR, mapped to fsaverage:

    >>> import brainload.freesurferdata as fsd
    >>> v, f, data, md = fsd.parse_subject_standard_space_data('subject1')

    Here, we are a bit more explicit about what we want to load:

    >>> import os
    >>> import brainload.freesurferdata as fsd
    >>> user_home = os.getenv('HOME')
    >>> subjects_dir = os.path.join(user_home, 'data', 'my_study_x')
    >>> v, f, data, md = fsd.parse_subject_standard_space_data('subject1', hemi='lh', measure='curv', fwhm='15', display_surf='inflated', subjects_dir=subjects_dir)

    Sometime we do not care for the mesh, e.g., we only want the morphometry data:

    >>> import brainload.freesurferdata as fsd
    >>> data, md = fsd.parse_subject_standard_space_data('subject1', hemi='rh', fwhm='15', load_surface_files=False)[2:4]

    """
    if hemi not in ('lh', 'rh', 'both'):
        raise ValueError("ERROR: hemi must be one of {'lh', 'rh', 'both'} but is '%s'." % hemi)

    if subjects_dir is None:
        subjects_dir = os.getenv('SUBJECTS_DIR', os.getcwd())

    if subjects_dir_for_average_subject is None:
        subjects_dir_for_average_subject = subjects_dir

    if meta_data is None:
        meta_data = {}

    # Retrieve the surface mesh on which to display the data. This is the average subject's surface.
    vert_coords = None
    faces = None
    if load_surface_files:
        display_subject = average_subject
        fsaverage_surf_dir = os.path.join(subjects_dir_for_average_subject, average_subject, 'surf')
        lh_surf_file = os.path.join(fsaverage_surf_dir, ('lh.' + display_surf))
        rh_surf_file = os.path.join(fsaverage_surf_dir, ('rh.' + display_surf))
        vert_coords, faces, meta_data = load_subject_mesh_files(lh_surf_file, rh_surf_file, hemi=hemi, meta_data=meta_data)
    else:
        display_surf = None
        display_subject = None

    # Parse the subject's morphology data, mapped to standard space by FreeSurfer's recon-all.
    morphology_data = None
    if load_morhology_data:
        subject_surf_dir = os.path.join(subjects_dir, subject_id, 'surf')
        if fwhm is None:    # If the uses explicitely sets fwmh to None, we use the file without any 'fwhmX' part. This data in this file should be identical to the data on the fwhm='0' case, so we expect that this will be rarely used.
            fhwm_tag = ''
        else:
            fhwm_tag = '.fwhm' + fwhm

        if custom_morphology_files is None:
            meta_data['custom_morphology_files_used'] = False
            lh_morphology_data_mapped_to_fsaverage = os.path.join(subject_surf_dir, ('lh.' + measure + _get_morphology_data_suffix_for_surface(surf) + fhwm_tag + '.' + average_subject + '.mgh'))
            rh_morphology_data_mapped_to_fsaverage = os.path.join(subject_surf_dir, ('rh.' + measure + _get_morphology_data_suffix_for_surface(surf) + fhwm_tag + '.' + average_subject + '.mgh'))
        else:
            meta_data['custom_morphology_files_used'] = True
            lh_morphology_data_mapped_to_fsaverage = os.path.join(subject_surf_dir, custom_morphology_files['lh'])
            rh_morphology_data_mapped_to_fsaverage = os.path.join(subject_surf_dir, custom_morphology_files['rh'])

        morphology_data, meta_data = load_subject_morphology_data_files(lh_morphology_data_mapped_to_fsaverage, rh_morphology_data_mapped_to_fsaverage, hemi=hemi, format='mgh', meta_data=meta_data)
    else:
        measure = None


    meta_data['measure'] = measure
    meta_data['subject_id'] = subject_id
    meta_data['subjects_dir'] = subjects_dir
    meta_data['display_surf'] = display_surf
    meta_data['display_subject'] = display_subject
    meta_data['average_subjects_dir'] = subjects_dir_for_average_subject
    meta_data['surf'] = surf
    meta_data['space'] = 'standard_space'
    meta_data['average_subject'] = average_subject
    meta_data['fwhm'] = fwhm
    meta_data['hemi'] = hemi

    return vert_coords, faces, morphology_data, meta_data



def load_group_data(measure, surf='white', hemi='both', fwhm='10', subjects_dir=None, average_subject='fsaverage', group_meta_data=None, subjects_list=None, subjects_file='subjects.txt', subjects_file_dir=None, custom_morphology_file_templates=None, subjects_detection_mode='auto'):
    """
    Load morphometry data for a number of subjects.

    Load group data, i.e., morphometry data for all subjects in a study that has already been mapped to standard space and is ready for group analysis.
    The information given in the parameters `measure`, `surf`, `hemi`, and `fwhm` are used to construct the file name that will be loaded by default.

    Parameters
    ----------
    measure : string
        The measure to load, e.g., 'area' or 'curv'. Data files for this measure have to exist for all subjects.

    surf : string, optional
        The brain surface where the data has been measured, e.g., 'white' or 'pial'. Defaults to 'white'.

    hemi : {'both', 'lh', 'rh'}, optional
        The hemisphere that should be loaded. Defaults to 'both'.

    fwhm : string or None, optional
        Which averaging version of the data should be loaded. FreeSurfer usually generates different standard space files with a number of smoothing settings. Defaults to '10'. If None is passed, the `.fwhmX` part is omitted from the file name completely. Set this to '0' to get the unsmoothed version.

    subjects_dir: string, optional
        A string representing the full path to a directory. Defaults to the environment variable SUBJECTS_DIR if omitted. If that is not set, used the current working directory instead. This is the directory from which the application was executed.

    average_subject: string, optional
        The name of the average subject to which the data was mapped. Defaults to 'fsaverage'.

    group_meta_data: dictionary, optional
        A dictionary that should be merged into the return value `group_meta_data`. Defaults to the empty dictionary if omitted.

    subjects_list: list of strings, optional (unless `subjects_detection_mode` is set to `list`)
        A list of subject identifiers or directory names that should be loaded from the `subjects_dir`. Example list: `['subject1', 'subject2']`. Defaults to None. Only allowed if `subjects_detection_mode` is `auto` or `list`. In `auto` mode, this takes
        precedence over all other options, i.e., if a `subjects_list` *and* the (default or custom) `subjects_file` are given, the `subjects_list` will be used.

    subjects_file_dir: string, optional
        A string representing the full path to a directory. This directory must contain the `subjects_file` (see below). Defaults to the `subjects_dir`.

    subjects_file: string, optional
        The name of the subjects file, relative to the `subjects_file_dir`. Defaults to 'subjects.txt'. The file must be a simple text file that contains one `subject_id` per line. It can be a CSV file that has other data following, but the `subject_id` has to be the first item on each line and the separator must be a comma. So a line is allowed to look like this: `subject1, 35, center1, 147`. No header is allowed. If you have a different format, consider reading the file yourself and pass the result as `subjects_list` instead.

    custom_morphology_file_templates: dictionary, optional
        Cutom filenames for the left and right hemisphere data files that should be loaded. A dictionary of strings with exactly the following two keys: `lh` and `rh`. The value strings can contain hardcoded file names or template strings for them. As always, the files will be loaded relative to the `surf/` directory of the respective subject. Example for hard-coded files: `{'lh': 'lefthemi.nonstandard.mymeasure44.mgh', 'rh': 'righthemi.nonstandard.mymeasure44.mgh'}`. The strings may contain any of the following variabes, which will be replaced by what you supplied to the other arguments of this function:
            - `${MEASURE}` will be replaced with the value of `measure`.
            - `${SURF}` will be replaced with the FreeSurfer file name part for the surface `surf`. This is the empty string if `surf` is 'white', and a dot followed by the value of `surf` for all other settings of surf. Examples: when `surf` is 'pial', this will be replaced with '.pial' (Note the dot!). If `surf` is 'white', this will be replaced with the empty string.
            - `${SURF_RAW}` will be replaced with the value of `surf`.
            - `${HEMI}` will be replaced with 'lh' for the left hemisphere, and with 'rh' for the right hemisphere.
            - `${FWHM}` will be replaced with the value of `fwhm`, so something like '10'.
            - `${SUBJECT_ID}` will be replaced by the id of the subject that is being loaded, e.g., 'subject3'.
            - `${AVERAGE_SUBJECT}` will be replaced by the value of `average_subject`.
            
            Note that only `${SURF}` and `${HEMI}` are usually needed, everything else can be hardcoded (or is not part of typical FreeSurfer file names at all, like `${SUBJECT_ID}`).
            Example template string: `subj_${SUBJECT_ID}_hemi_${HEMI}.alsononstandard.mgh`. Complete example for template strings in dictionary: `{'lh': 'subj_${SUBJECT_ID}_hemi_${HEMI}.alsononstandard.mgh', 'rh': 'subj_${SUBJECT_ID}_hemi_${HEMI}.alsononstandard.mgh'}`.

    subjects_detection_mode: {'auto', 'list', 'file', 'search_dir'}, optional
        The method used to determine the subjects that should be loaded. Defaults to 'auto'. You can always see which mode was used by looking at the returned `run_meta_data`, see `run_meta_data['subjects_detection_mode']`.
            - 'auto': In this mode, all available methods will be tried in the following order: If a `subjects_list` is given, it is used. Then, the `subjects_file` is used if it exists. Note that this may be the default file, '$SUBJECTS_DIR/subject_surf_dir.txt', or another if one has explicitely been defined by setting `subjects_file` and/or `subjects_file_dir`. If the file does not exist, the directory is searched for directories containing FreeSurfer data as defined in the section for 'search_dir' mode below. You can always see which method was used in auto mode by looking at the returned `run_meta_data`, see `run_meta_data['subjects_detection_mode_auto_used_method']`.

            - 'list': In this mode, the given `subjects_list` is used, and you have to supply one. If not, an error is raised. You are not allowed to supply a `subjects_file` in this mode, or an error will be raised.

            - 'file': In this mode, the subjects file is used. Note that this may be the default file, '$SUBJECTS_DIR/subjects.txt', or another if one has explicitely been defined by setting `subjects_file` and/or `subjects_file_dir`. If the file does not exist, an error is raised. You can see which file was used by looking at the returned `run_meta_data`, see `run_meta_data['subjects_file']`. You are not allowed to supply a `subjects_list` in this mode, or an error will be raised.

            - 'search_dir': In this mode, the `subjects_dir` (default or explicitely given) is searched for sub directories which look as if they could contain FreeSurfer data. The latter means that they contain a sub directory named 'surf'. There is one exception though: if the name of one such directory equals the name of the `average_subject`, the directory is skipped. You are not allowed to supply a `subjects_list` in this mode, or an error will be raised.

    Returns
    -------
    group_morphology_data: numpy array
        An array filled with the morphology data for the subjects. The array has shape `(n, m)` where `n` is the number of subjects, and `m` is the number of vertices of the standard subject. (If you load both hemispheres instead of one, m doubles.) To get the subject id for the entries, look at the respective index in the returned `subjects_list`.

    subjects_list: list of strings
        A list containing the subject identifiers in the same order as the data in `group_morphology_data`. (If `subjects_detection_mode` is 'list' or 'file', the order in these is guaranteed to be preserved. But in mode 'search_dir' or 'auto' which may have chosen to fall back to 'search_dir' as a last resort, this is helpful: You can use the index of a subject in this list to find its data in `group_morphology_data`, as it will have the same index. See the examples below.)

    group_meta_data: dictionary
        A dictionary containing detailed information on all subjects and files that were loaded. Each of its keys is a subject identifier. The data value is another dictionary that contains all meta data for this subject as returned by the `parse_subject_standard_space_data` function.

    run_meta_data: dictionary
        A dictionary containing general information on the settings used when executing the function and determining which subjects to load.

    Raises
    ------
    ValueError
        If one of the parameters with a fixed set of values receives a value that is not allowed.

    Examples
    --------
    Load area data for all subjects in the directory defined by the environment variable SUBJECTS_DIR:

    >>> import brainload.freesurferdata as fsd
    >>> data, subjects, group_md, run_md = fsd.load_group_data('area')

    Here, we load curv data for the right hemisphere, computed on the pial surface with smooting of 20:

    >>> import brainload.freesurferdata as fsd
    >>> data, subjects, group_md, run_md = fsd.load_group_data('curv', hemi='rh', surf='pial', fwhm='20')

    We may want to be a but more explicit on which subjects are loaded from where:

    >>> import os
    >>> import brainload.freesurferdata as fsd
    >>> subjects_dir = os.path.join(os.getenv('HOME'), 'data', 'my_study_x')
    >>> subjects_list = ['subject1', 'subject4', 'subject8']
    >>> data, subjects, group_md, run_md = fsd.load_group_data('curv', fwhm='20', subjects_dir=subjects_dir, subjects_list=subjects_list)

    Continuing the last example, we may want to have a look at the curv value of the vertex at index 100000 of the subject 'subject4':

    >>> subject4_idx = subjects.index('subject4')   # Will be 1, as the order of the subjects_list is preserved. But if you load from a file s
    >>> print group_data[subject4_idx][100000]

    """
    if hemi not in ('lh', 'rh', 'both'):
        raise ValueError("ERROR: hemi must be one of {'lh', 'rh', 'both'} but is '%s'." % hemi)

    run_meta_data = {}

    if subjects_detection_mode not in ('auto', 'file', 'list', 'search_dir'):
        raise ValueError("ERROR: subjects_detection_mode must be one of {'auto', 'file', 'list', 'search_dir'} but is '%s'." % subjects_detection_mode)
    else:
        run_meta_data['subjects_detection_mode'] = subjects_detection_mode

    # supplying a list is only valid in modes 'auto' and 'list'
    if subjects_detection_mode in ('file', 'search_dir') and subjects_list is not None:
        raise ValueError("ERROR: subjects_detection_mode is set to '%s' but a subjects_list was given. Not supported in subjects_detection_mode 'file' and 'search_dir'." % subjects_detection_mode)

    if subjects_dir is None:
        subjects_dir = os.getenv('SUBJECTS_DIR', os.getcwd())

    if subjects_file_dir is None:
        subjects_file_dir = subjects_dir

    if group_meta_data is None:
        group_meta_data = {}

    run_meta_data['subjects_file_used'] = False
    if subjects_detection_mode == 'auto' and subjects_list is not None:
        run_meta_data['subjects_detection_mode_auto_used_method'] = 'list'  # just use the existing list
    elif subjects_detection_mode == 'list':
        if subjects_list is None:
            raise ValueError("ERROR: subjects_detection_mode is set to 'list' but the subjects_list parameter was not given.")
    elif subjects_detection_mode == 'search_dir':
        subjects_list = nit.detect_subjects_in_directory(subjects_dir, ignore_dir_names=[average_subject])
    else:
        # we are in modes 'auto' or 'file'
        subjects_file_with_path = os.path.join(subjects_file_dir, subjects_file)
        assumed_subjects_file_exists = os.path.isfile(subjects_file_with_path)

        # in file mode, the file has to exist.
        if subjects_detection_mode == 'file' and not assumed_subjects_file_exists:
            raise ValueError("ERROR: subjects_detection_mode is set to 'file' but the subjects_file '%s' does not exist." % subjects_file_with_path)

        auto_mode_done = False
        if assumed_subjects_file_exists:    # we are still in modes 'auto' or 'file', and the file exists.
            subjects_list = nit.read_subjects_file(subjects_file_with_path)
            run_meta_data['subjects_file_used'] = True
            run_meta_data['subjects_file'] = subjects_file_with_path
            if subjects_detection_mode == 'auto':   # in auto mode, we prefer/use the subject file if it exists. If it does not exist, we try to guess the list from the directory later.
                auto_mode_done = True
                run_meta_data['subjects_detection_mode_auto_used_method'] = 'file'

        if subjects_detection_mode == 'auto' and not auto_mode_done:            # last chance in auto mode: try to detect subjects from the contents of the subjects dir.
            run_meta_data['subjects_detection_mode_auto_used_method'] = 'search_dir'
            subjects_list = nit.detect_subjects_in_directory(subjects_dir, ignore_dir_names=[average_subject])

    if custom_morphology_file_templates is not None:
        run_meta_data['custom_morphology_file_templates_used'] = True
        run_meta_data['lh.custom_morphology_file_template'] = custom_morphology_file_templates['lh']
        run_meta_data['rh.custom_morphology_file_template'] = custom_morphology_file_templates['rh']
    else:
        run_meta_data['custom_morphology_file_templates_used'] = False

    group_morphology_data = []
    for subject_id in subjects_list:

        custom_morphology_files = None
        subject_meta_data = {}
        if custom_morphology_file_templates is not None:
            surf_file_part = _get_morphology_data_suffix_for_surface(surf)
            substitution_dict_lh = {'MEASURE': measure, 'SURF_RAW': surf, 'SURF': surf_file_part, 'HEMI': 'lh', 'FWHM': fwhm, 'SUBJECT_ID': subject_id, 'AVERAGE_SUBJECT': average_subject}
            substitution_dict_rh = {'MEASURE': measure, 'SURF_RAW': surf, 'SURF': surf_file_part, 'HEMI': 'rh', 'FWHM': fwhm, 'SUBJECT_ID': subject_id, 'AVERAGE_SUBJECT': average_subject}
            custom_morphology_file_lh = nit.fill_template_filename(custom_morphology_file_templates['lh'], substitution_dict_lh)
            custom_morphology_file_rh = nit.fill_template_filename(custom_morphology_file_templates['rh'], substitution_dict_rh)
            custom_morphology_files = {'lh': custom_morphology_file_lh, 'rh': custom_morphology_file_rh}

        # In the next function call, we discard the first two return values (vert_coords and faces), as these are None anyways because we did not load surface files.
        subject_morphology_data, subject_meta_data = parse_subject_standard_space_data(subject_id, measure=measure, surf=surf, hemi=hemi, fwhm=fwhm, subjects_dir=subjects_dir, average_subject=average_subject, meta_data=subject_meta_data, load_surface_files=False, custom_morphology_files=custom_morphology_files)[2:4]
        group_meta_data[subject_id] = subject_meta_data
        group_morphology_data.append(subject_morphology_data)
    group_morphology_data = np.array(group_morphology_data)
    return group_morphology_data, subjects_list, group_meta_data, run_meta_data
