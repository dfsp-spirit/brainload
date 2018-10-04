import nibabel.freesurfer.io as fsio
import nibabel.freesurfer.mghformat as fsmgh

def merge_meshes(mesh_file_list):
    merged_vert_coords = np.empty((0, 3), dtype=float)
    merged_faces = np.empty((0, 3), dtype=int)
    for mesh_file in mesh_file_list:
        ## merge vertex coordinates: all we have to do is append them and record the number we appended
        vertex_index_shift = merged_vert_coords.shape[0]
        vert_coords, faces = fsio.read_geometry(mesh_file)
        merged_vert_coords = np.vstack((merged_vert_coords, vert_coords))
        ## Now merge the new faces. We need to modify the vertex indices: shift them by the number of vertices we already have
        faces_shifted = faces + vertex_index_shift
        merged_faces = np.vstack((merged_faces, faces_shifted))
    return merged_vert_coords, merged_faces


def read_mgh_file(mgh_file_name, collect_meta_data=True):
    '''Reads all data from the MGH file and returns it as a numpy array.

       You can use the meta data that is collected to interpret the data.

       You may want to read:
         - https://surfer.nmr.mgh.harvard.edu/fswiki/CoordinateSystems
         - https://github.com/nipy/nibabel/blob/master/nibabel/freesurfer/mghformat.py
    '''
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


def merge_per_vertex_data(curv_file_list, curv_file_format='curv', dtype=float):
    '''Reads an array of curv file names and merges the data contained in them into one large array.
       The main purpose is to read the curv files for the left and right hemispheres of a brain.
    '''
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


def parse_brain_files(mesh_lh, mesh_rh, curv_lh=None, curv_rh=None, meta_data={}, curv_file_format='curv'):
    '''Low-level interface to parse FreeSurfer brain data. Parses both hemispheres of a brain and the respective surface data files.'''
    verts, faces = merge_meshes(np.array([mesh_lh, mesh_rh]))
    meta_data['file_mesh_lh'] = mesh_lh
    meta_data['file_mesh_rh'] = mesh_rh
    morphology_data = None
    if not (curv_lh is None or curv_rh is None):
        morphology_data = merge_per_vertex_data(np.array([curv_lh, curv_rh]), curv_file_format)
        meta_data['file_curv_lh'] = curv_lh
        meta_data['file_curv_rh'] = curv_rh
        print morphology_data
    return verts, faces, morphology_data, meta_data


def get_morphology_data_suffix_for_surface(surf):
    '''Determines the substring representing the given surface in a FreeSurfer output curv file.'''
    if surf == 'white':
        return ''
    return '.' + surf


def parse_subject(subject_id, surf='white', measure='area', subjects_dir=None, meta_data={}):
    '''High-level interface to parse FreeSurfer brain data in subject space.
       Uses knowledge on standard file names to find the data.
       Use the low-level interface 'parse_brain_files' if you have non-standard file names.
    '''
    if subjects_dir is None:
        subjects_dir = os.getenv('SUBJECTS_DIR', '.')
    subject_surf_dir = os.path.join(subjects_dir, subject_id, 'surf')
    lh_surf = os.path.join(subject_surf_dir, ('lh.' + surf))
    rh_surf = os.path.join(subject_surf_dir, ('rh.' + surf))
    lh_morphology_data = os.path.join(subject_surf_dir, ('lh.' + measure + get_morphology_data_suffix_for_surface(surf)))
    rh_morphology_data = os.path.join(subject_surf_dir, ('rh.' + measure + get_morphology_data_suffix_for_surface(surf)))
    meta_data['subject_id'] = subject_id
    meta_data['measure'] = measure
    meta_data['surf'] = surf
    meta_data['space'] = 'subject'
    return parse_brain_files(lh_surf, rh_surf, curv_lh=lh_morphology_data, curv_rh=rh_morphology_data, meta_data=meta_data)


def parse_subject_standard_space_data(subject_id, measure='area', surf='white', display_surf='white', fwhm='10', subjects_dir=None, average_subject='fsaverage', meta_data={}):
    if subjects_dir is None:
        subjects_dir = os.getenv('SUBJECTS_DIR', '.')
    # parse the data, mapped to standard space, for the subject
    subject_surf_dir = os.path.join(subjects_dir, subject_id, 'surf')
    lh_morphology_data_mapped_to_fsaverage = os.path.join(subject_surf_dir, ('lh.' + measure + get_morphology_data_suffix_for_surface(surf) + '.fwhm' + fwhm + '.' + average_subject + '.mgh'))
    rh_morphology_data_mapped_to_fsaverage = os.path.join(subject_surf_dir, ('rh.' + measure + get_morphology_data_suffix_for_surface(surf) + '.fwhm' + fwhm + '.' + average_subject + '.mgh'))
    # Now retrieve the surface on which to display the data. This is the average subject's surface.
    fsaverage_surf_dir = os.path.join(subjects_dir, average_subject, 'surf')
    lh_surf = os.path.join(fsaverage_surf_dir, ('lh.' + display_surf))
    rh_surf = os.path.join(fsaverage_surf_dir, ('rh.' + display_surf))
    meta_data['subject_id'] = subject_id
    meta_data['measure'] = measure
    meta_data['surf'] = surf
    meta_data['space'] = 'standard'
    meta_data['display_surf'] = display_surf
    meta_data['average_subject'] = average_subject
    meta_data['fwhm'] = fwhm
    return parse_brain_files(lh_surf, rh_surf, curv_lh=lh_morphology_data_mapped_to_fsaverage, curv_rh=rh_morphology_data_mapped_to_fsaverage, meta_data=meta_data, curv_file_format='mgh')
