import os
import csv
import numpy as np
import nibabel.freesurfer.io as fsio
import nibabel.freesurfer.mghformat as fsmgh

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


def merge_morphology_data(morphology_data_arrays, dtype=float):
    merged_data = np.empty((0), dtype=dtype)
    for morphology_data in morphology_data_arrays:
        merged_data = np.hstack((merged_data, morphology_data))
    return merged_data


def get_morphology_data_suffix_for_surface(surf):
    '''Determines the substring representing the given surface in a FreeSurfer output curv file.'''
    if surf == 'white':
        return ''
    return '.' + surf


def read_fs_surface_file_and_record_meta_data(surf_file, hemisphere_label, meta_data=None):
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
    if format not in ('curv', 'mgh'):
        raise ValueError("ERROR: format must be one of {'curv', 'mgh'} but is '%s'." % format)

    if hemisphere_label not in ('lh', 'rh'):
        raise ValueError("ERROR: hemisphere_label must be one of {'lh', 'rh'} but is '%s'." % hemisphere_label)

    if meta_data is None:
        meta_data = {}

    if format == 'mgh':
        full_mgh_data, mgh_meta_data = read_mgh_file(curv_file, collect_meta_data=False)
        relevant_data_inner_array = full_mgh_data[:,0]        # If this fails, you may need to check mgh_meta_data['data_shape'].
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
        vert_coords, faces = merge_meshes(np.array([[lh_vert_coords, lh_faces], [rh_vert_coords, rh_faces]]))
    return vert_coords, faces, meta_data


def load_subject_morphology_data_files(lh_morphology_data_file, rh_morphology_data_file, hemi='both', format='curv', meta_data=None):
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
    '''High-level interface to parse FreeSurfer brain data in subject space.
       Uses knowledge on standard file names to find the data.
       Use the low-level interface 'parse_brain_files' if you have non-standard file names.
    '''
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
        lh_morphology_file = os.path.join(subject_surf_dir, ('lh.' + measure + get_morphology_data_suffix_for_surface(surf)))
        rh_morphology_file = os.path.join(subject_surf_dir, ('rh.' + measure + get_morphology_data_suffix_for_surface(surf)))
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


def merge_meshes(meshes):
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


def parse_subject_standard_space_data(subject_id, measure='area', surf='white', display_surf='white', hemi='both', fwhm='10', subjects_dir=None, average_subject='fsaverage', subjects_dir_for_average_subject=None, meta_data=None, load_surface_files=True, load_morhology_data=True):
    if hemi not in ('lh', 'rh', 'both'):
        raise ValueError("ERROR: hemi must be one of {'lh', 'rh', 'both'}")
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

        lh_morphology_data_mapped_to_fsaverage = os.path.join(subject_surf_dir, ('lh.' + measure + get_morphology_data_suffix_for_surface(surf) + fhwm_tag + '.' + average_subject + '.mgh'))
        rh_morphology_data_mapped_to_fsaverage = os.path.join(subject_surf_dir, ('rh.' + measure + get_morphology_data_suffix_for_surface(surf) + fhwm_tag + '.' + average_subject + '.mgh'))
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



def read_subjects_file(subjects_file, **kwargs):
    """
    Read a subjects file in CSV format that has the subject id as the first entry on each line. Arbitrary data may follow in the cosecutive fields on each line, and will be ignored.
    """
    subject_ids = []
    with open(subjects_file, 'r') as sfh:
        reader = csv.reader(sfh, **kwargs)
        for row in reader:
            subject_ids.append(row[0])
    return subject_ids


def load_group_data(measure, surf='white', hemi='both', fwhm='10', subjects_dir=None, average_subject='fsaverage', meta_data=None, subjects_list=None, subjects_file='subjects.txt', subjects_file_dir=None):
    if subjects_dir is None:
        subjects_dir = os.getenv('SUBJECTS_DIR', os.getcwd())
        
    if subjects_file_dir is None:
        subjects_file_dir = subjects_dir

    if meta_data is None:
        meta_data = {}

    if subjects_list is None:
        subjects_file_with_path = os.path.join(subjects_file_dir, subjects_file)
        subjects_list = read_subjects_file(subjects_file_with_path)

    group_morphology_data = []
    group_meta_data = {}
    for subject_id in subjects_list:
        # In the next function call, we discard the first two return values (vert_coords and faces), as these are None anyways because we did not load surface files.
        morphology_data, meta_data = parse_subject_standard_space_data(subject_id, measure=measure, surf=surf, hemi=hemi, fwhm=fwhm, subjects_dir=subjects_dir, average_subject=average_subject, meta_data=meta_data, load_surface_files=False)[2:4]
        group_meta_data[subject_id] = meta_data
        group_morphology_data.append(morphology_data)
    group_morphology_data = np.array(group_morphology_data)
    return group_morphology_data, group_meta_data
