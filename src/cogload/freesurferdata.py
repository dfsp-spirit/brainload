import os
import numpy as np
import nibabel.freesurfer.io as fsio
import nibabel.freesurfer.mghformat as fsmgh

def merge_mesh_files(mesh_file_list, meta_data={}):
    """
    Read brain mesh files into a single mesh.

    Read all brain mesh files in the numpy array mesh_file_list and merge the meshes into a single one. This is intended for reading the left and right hemispheres of a brain.
    """

    print "########## DEPRECATED ############# EXITING."
    exit()

    merged_vert_coords = np.empty((0, 3), dtype=float)
    merged_faces = np.empty((0, 3), dtype=int)
    vertex_count_per_mesh = {}
    face_count_per_mesh = {}
    for mesh_file in mesh_file_list:
        ## merge vertex coordinates: all we have to do is append them and record the number we appended
        vertex_index_shift = merged_vert_coords.shape[0]
        vert_coords, faces = fsio.read_geometry(mesh_file)

        vertex_count_per_mesh[mesh_file] = vert_coords.shape[0]             # collect meta data
        face_count_per_mesh[mesh_file] = faces.shape[0]
        print "Metadata: Found %d vertices, %d faces in file %s." % (vert_coords.shape[0], faces.shape[0], mesh_file)

        merged_vert_coords = np.vstack((merged_vert_coords, vert_coords))
        ## Now merge the new faces. We need to modify the vertex indices: shift them by the number of vertices we already have
        faces_shifted = faces + vertex_index_shift
        merged_faces = np.vstack((merged_faces, faces_shifted))
    meta_data['vertex_count_per_mesh'] = vertex_count_per_mesh
    meta_data['face_count_per_mesh'] = face_count_per_mesh
    return merged_vert_coords, merged_faces


def read_mgh_file(mgh_file_name, collect_meta_data=True):
    """
    Read data from a FreeSurfer output file in mgh format.

    Read all data from the MGH file and return it as a numpy array. Optionally, collect meta data from the mgh file header.

    See also
    --------
        - https://surfer.nmr.mgh.harvard.edu/fswiki/CoordinateSystems
        - https://github.com/nipy/nibabel/blob/master/nibabel/freesurfer/mghformat.py
    """
    mgh_meta_data = {}
    with open(mgh_file_name, 'r') as mgh_file_handle:
        header = fsmgh.MGHHeader.from_fileobj(mgh_file_handle)

        if collect_meta_data:
            mgh_meta_data['data_shape'] = header.get_data_shape()
            mgh_meta_data['affine'] = header.get_affine()
            mgh_meta_data['data_bytes_per_voxel'] = header.get_data_bytespervox()
            mgh_meta_data['data_dtype'] = header.get_data_dtype()
            mgh_meta_data['data_size'] = header.get_data_size()
            mgh_meta_data['ras2vox'] =  header.get_ras2vox()
            mgh_meta_data['vox2ras'] =  header.get_vox2ras()
            mgh_meta_data['vox2ras_tkr'] =  header.get_vox2ras_tkr()
            mgh_meta_data['voxel_spacing'] =  header.get_zooms()
            mgh_meta_data['data_offset'] =  header.get_data_offset()
            mgh_meta_data['footer_offset'] =  header.get_footer_offset()

        mgh_data = header.data_from_fileobj(mgh_file_handle)
        mgh_file_handle.close()
    return mgh_data, mgh_meta_data


def merge_per_vertex_data_files(curv_file_list, curv_file_format='curv', dtype=float):
    '''Reads an array of curv file names and merges the data contained in them into one large array.
       The main purpose is to read the curv files for the left and right hemispheres of a brain.
    '''

    print "########## DEPRECATED ############# EXITING."
    exit()

    if curv_file_format not in ('curv', 'mgh'):
        raise ValueError("ERROR: curv_file_format must be one of {'curv', 'mgh'}")
    merged_data = np.empty((0), dtype=dtype)
    for curv_file in curv_file_list:
        if curv_file_format == 'mgh':
            full_mgh_data, mgh_meta_data = read_mgh_file(curv_file)
            relevant_data_inner_array = full_mgh_data[:,0]        # If this fails, you may need to check mgh_meta_data['data_shape'].
            per_vertex_data = relevant_data_inner_array[:,0]
        else:
            per_vertex_data = fsio.read_morph_data(curv_file)
        merged_data = np.hstack((merged_data, per_vertex_data))
    return merged_data


def merge_morphology_data(morphology_data_arrays, dtype=float):
    merged_data = np.empty((0), dtype=dtype)
    print "merge_morphology_data: shape before merge"
    print merged_data.shape
    for morphology_data in morphology_data_arrays:
        merged_data = np.hstack((merged_data, morphology_data))
        print "merge_morphology_data: merging new with shape:"
        print morphology_data.shape
    print "merge_morphology_data: shape after merge"
    print merged_data.shape
    return merged_data


def parse_brain_files(mesh_lh, mesh_rh, curv_lh=None, curv_rh=None, meta_data={}, curv_file_format='curv'):
    '''Low-level interface to parse FreeSurfer brain data. Parses both hemispheres of a brain and the respective surface data files.'''

    print "########## DEPRECATED ############# EXITING."
    exit()

    verts, faces = merge_mesh_files(np.array([mesh_lh, mesh_rh]), meta_data)
    meta_data['file_mesh_lh'] = mesh_lh
    meta_data['file_mesh_rh'] = mesh_rh
    morphology_data = None
    if not (curv_lh is None or curv_rh is None):
        morphology_data = merge_per_vertex_data_files(np.array([curv_lh, curv_rh]), curv_file_format)
        meta_data['file_curv_lh'] = curv_lh
        meta_data['file_curv_rh'] = curv_rh
        #print morphology_data
    return verts, faces, morphology_data, meta_data


def get_morphology_data_suffix_for_surface(surf):
    '''Determines the substring representing the given surface in a FreeSurfer output curv file.'''
    if surf == 'white':
        return ''
    return '.' + surf


def read_fs_surface_file_and_record_meta_data(surf_file, hemisphere_label, meta_data={}):
    vert_coords, faces = fsio.read_geometry(surf_file)
    label_num_vertices = hemisphere_label + '.num_vertices'
    label_num_faces = hemisphere_label + '.num_faces'
    meta_data[label_num_vertices] = vert_coords.shape[0]
    meta_data[label_num_faces] = faces.shape[0]
    print "read_fs_surface_file_and_record_meta_data: read %d verts and %d faces from file %s." % (vert_coords.shape[0], faces.shape[0], surf_file)
    return vert_coords, faces, meta_data


def read_fs_morphology_data_file_and_record_meta_data(curv_file, hemisphere_label, meta_data={}, format='curv'):
    if format not in ('curv', 'mgh'):
        raise ValueError("ERROR: format must be one of {'curv', 'mgh'}")

    if format == 'mgh':
        full_mgh_data, mgh_meta_data = read_mgh_file(curv_file, collect_meta_data=False)
        relevant_data_inner_array = full_mgh_data[:,0]        # If this fails, you may need to check mgh_meta_data['data_shape'].
        per_vertex_data = relevant_data_inner_array[:,0]
    else:
        per_vertex_data = fsio.read_morph_data(curv_file)

    label_num_values = hemisphere_label + '.num_data_points'
    meta_data[label_num_values] = per_vertex_data.shape[0]
    return per_vertex_data, meta_data


def load_subject_mesh_files(lh_surf_file, rh_surf_file, hemi='both', meta_data={}):
    if hemi not in ('lh', 'rh', 'both'):
        raise ValueError("ERROR: hemi must be one of {'lh', 'rh', 'both'}")

    if hemi == 'lh':
        vert_coords, faces, meta_data = read_fs_surface_file_and_record_meta_data(lh_surf_file, 'lh', meta_data=meta_data)
    elif hemi == 'rh':
        vert_coords, faces, meta_data = read_fs_surface_file_and_record_meta_data(rh_surf_file, 'rh', meta_data=meta_data)
    else:
        lh_vert_coords, lh_faces, meta_data = read_fs_surface_file_and_record_meta_data(lh_surf_file, 'lh', meta_data=meta_data)
        rh_vert_coords, rh_faces, meta_data = read_fs_surface_file_and_record_meta_data(rh_surf_file, 'rh', meta_data=meta_data)
        vert_coords, faces = merge_meshes(np.array([[lh_vert_coords, lh_faces], [rh_vert_coords, rh_faces]]))
    return vert_coords, faces, meta_data


def load_subject_morphology_data_files(lh_morphology_data_file, rh_morphology_data_file, hemi='both', format='curv', meta_data={}):
    if hemi not in ('lh', 'rh', 'both'):
        raise ValueError("ERROR: hemi must be one of {'lh', 'rh', 'both'}")

    if hemi == 'lh':
        morphology_data, meta_data = read_fs_morphology_data_file_and_record_meta_data(lh_morphology_data_file, 'lh', meta_data=meta_data, format=format)
    elif hemi == 'rh':
        morphology_data, meta_data = read_fs_morphology_data_file_and_record_meta_data(rh_morphology_data_file, 'rh', meta_data=meta_data, format=format)
    else:
        lh_morphology_data, meta_data = read_fs_morphology_data_file_and_record_meta_data(lh_morphology_data_file, 'lh', meta_data=meta_data, format=format)
        rh_morphology_data, meta_data = read_fs_morphology_data_file_and_record_meta_data(rh_morphology_data_file, 'rh', meta_data=meta_data, format=format)
        morphology_data = merge_morphology_data(np.array([lh_morphology_data, rh_morphology_data]))
    return morphology_data, meta_data


def parse_subject(subject_id, surf='white', measure='area', hemi='both', subjects_dir=None, meta_data={}):
    '''High-level interface to parse FreeSurfer brain data in subject space.
       Uses knowledge on standard file names to find the data.
       Use the low-level interface 'parse_brain_files' if you have non-standard file names.
    '''
    if hemi not in ('lh', 'rh', 'both'):
        raise ValueError("ERROR: hemi must be one of {'lh', 'rh', 'both'}")

    if subjects_dir is None:
        subjects_dir = os.getenv('SUBJECTS_DIR', '.')
    subject_surf_dir = os.path.join(subjects_dir, subject_id, 'surf')

    lh_surf_file = os.path.join(subject_surf_dir, ('lh.' + surf))
    rh_surf_file = os.path.join(subject_surf_dir, ('rh.' + surf))
    vert_coords, faces, meta_data = load_subject_mesh_files(lh_surf_file, rh_surf_file, hemi=hemi, meta_data=meta_data)
    print "parse_subject: vertcoords shape"
    print vert_coords.shape
    print "parse_subject: faces shape"
    print faces.shape
    print "max face index: %d" % np.max(faces[1][0])

    morphology_data = None
    if measure is not None:
        lh_morphology_file = os.path.join(subject_surf_dir, ('lh.' + measure + get_morphology_data_suffix_for_surface(surf)))
        rh_morphology_file = os.path.join(subject_surf_dir, ('rh.' + measure + get_morphology_data_suffix_for_surface(surf)))
        morphology_data, meta_data = load_subject_morphology_data_files(lh_morphology_file, rh_morphology_file, hemi=hemi, format='curv', meta_data=meta_data)
        meta_data['measure'] = measure
    print "parse_subject: morphology_data shape"
    print morphology_data.shape

    meta_data['subject_id'] = subject_id
    meta_data['surf'] = surf
    meta_data['space'] = 'subject'
    meta_data['hemi'] = hemi

    return vert_coords, faces, morphology_data, meta_data


def merge_meshes(meshes):
    all_vert_coords = np.empty((0, 3), dtype=float)
    all_faces = np.empty((0, 3), dtype=int)

    for mesh in meshes:
        new_vert_coords = mesh[0]
        new_faces = mesh[1]
        print "merging new mesh with %d vertices. we have %d currently." % (new_vert_coords.shape[0], all_vert_coords.shape[0])
        # Keep track of the total number of vertices we had *before* adding the new ones. This is the shift we need for the faces.
        vertex_index_shift = all_vert_coords.shape[0]
        print "vertex_index_shift=%d" % vertex_index_shift
        all_vert_coords = np.vstack((all_vert_coords, new_vert_coords))

        new_faces_shifted = new_faces + vertex_index_shift
        all_faces = np.vstack((all_faces, new_faces_shifted))
    return all_vert_coords, all_faces


def parse_subject_standard_space_data(subject_id, measure='area', surf='white', display_surf='white', hemi='both', fwhm='10', subjects_dir=None, average_subject='fsaverage', meta_data={}):
    if hemi not in ('lh', 'rh', 'both'):
        raise ValueError("ERROR: hemi must be one of {'lh', 'rh', 'both'}")
    if subjects_dir is None:
        subjects_dir = os.getenv('SUBJECTS_DIR', '.')

    # Parse the subject's data, mapped to standard space by FreeSurfer's recon-all.
    morphology_data = None
    if measure is not None:
        subject_surf_dir = os.path.join(subjects_dir, subject_id, 'surf')
        lh_morphology_data_mapped_to_fsaverage = os.path.join(subject_surf_dir, ('lh.' + measure + get_morphology_data_suffix_for_surface(surf) + '.fwhm' + fwhm + '.' + average_subject + '.mgh'))
        rh_morphology_data_mapped_to_fsaverage = os.path.join(subject_surf_dir, ('rh.' + measure + get_morphology_data_suffix_for_surface(surf) + '.fwhm' + fwhm + '.' + average_subject + '.mgh'))
        morphology_data, meta_data = load_subject_morphology_data_files(lh_morphology_data_mapped_to_fsaverage, rh_morphology_data_mapped_to_fsaverage, hemi=hemi, format='mgh', meta_data=meta_data)
        meta_data['measure'] = measure

    # Now retrieve the surface mesh on which to display the data. This is the average subject's surface.
    vert_coords = None
    faces = None
    if display_surf is not None:
        fsaverage_surf_dir = os.path.join(subjects_dir, average_subject, 'surf')
        lh_surf = os.path.join(fsaverage_surf_dir, ('lh.' + display_surf))
        rh_surf = os.path.join(fsaverage_surf_dir, ('rh.' + display_surf))
        vert_coords, faces, meta_data = load_subject_mesh_files(lh_surf_file, rh_surf_file, hemi=hemi, meta_data=meta_data)
        meta_data['display_surf'] = display_surf

    meta_data['subject_id'] = subject_id
    meta_data['surf'] = surf
    meta_data['space'] = 'standard'
    meta_data['average_subject'] = average_subject
    meta_data['fwhm'] = fwhm

    return vert_coords, faces, morphology_data, meta_data
