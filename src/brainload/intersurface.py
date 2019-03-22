#!/usr/bin/env python
from __future__ import print_function
import os
import sys
import errno
import numpy as np
import argparse
import brainload as bl
from numpy.linalg import norm

# To run this in dev mode (in virtual env, pip -e install of brainview active) from REPO_ROOT:
# PYTHONPATH=./src/brainload python src/brainload/intersurface.py tim -d ~/data/tim_only/ --hemi lh


def intersurface():
    parser = argparse.ArgumentParser(description="Compute and write intersurface data.")
    parser.add_argument("subject", help="The subject you want to load. String, a directory under the subjects_dir.")
    parser.add_argument("-d", "--subjects_dir", help="The subjects_dir containing the subject. Defaults to environment variable SUBJECTS_DIR.", default="")
    parser.add_argument("-f", "--first-surface", help="The first surface to load. String, defaults to 'white'.", default="white")
    parser.add_argument("-s", "--second-surface", help="The second surface to load. String, defaults to 'pial'.", default="pial")
    parser.add_argument("-e", "--hemi", help="The hemisphere to load. One of ('lh, 'rh').", choices=['lh', 'rh'])
    parser.add_argument("-c", "--compute-value", help="What to compute for the two surfaces. One of ('expected_vol', 'actual_vol'). Defaults to 'expected_vol'.", default="expected_vol", choices=['expected_vol', 'actual_vol'])
    parser.add_argument("-o", "--outputfile", help="Output file name. Will be in FreeSurfer surf format (like lh.area or lh.thickness).", default="")
    parser.add_argument("-v", "--verbose", help="Increase output verbosity.", action="store_true")
    args = parser.parse_args()

    if args.subjects_dir == "":
        subjects_dir = os.getenv('SUBJECTS_DIR')
    else:
        subjects_dir = args.subjects_dir

    subject_id = args.subject
    measure = "thickness"
    surf1 = args.first_surface
    surf2 = args.second_surface
    hemi = args.hemi

    vert_coords_surf1, faces_surf1, morphometry_data_surf1, meta_data_surf1 = bl.subject(subject_id, subjects_dir=subjects_dir, measure=measure, surf=surf1, hemi=hemi)
    vert_coords_surf2, faces_surf2, meta_data_surf2 = bl.subject_mesh(subject_id, subjects_dir, surf=surf2, hemi=hemi)

    face_areas_surf1 = get_mesh_face_areas(vert_coords_surf1, faces_surf1)
    print("Computed %d areas for all %d faces." % (face_areas_surf1.shape[0], faces_surf1.shape[0]))

    for vert_idx, vert_coords in enumerate(vert_coords_surf1):
        faces_with_vert_at_pos0 = np.nonzero(faces_surf1[:,0]==vert_idx)[0].tolist()
        faces_with_vert_at_pos1 = np.nonzero(faces_surf1[:,1]==vert_idx)[0].tolist()
        faces_with_vert_at_pos2 = np.nonzero(faces_surf1[:,2]==vert_idx)[0].tolist()
        all_face_indices = list(set(faces_with_vert_at_pos0 + faces_with_vert_at_pos1 + faces_with_vert_at_pos2))
        summed_area = face_areas_surf1[all_face_indices].sum()
        if vert_idx % 1000 == 0:
            print('At vertex %d. Vertex is part of the following %d faces with total area %f: %s' % (vert_idx, len(all_face_indices), summed_area, ','.join([str(x) for x in all_face_indices])))


def get_mesh_face_areas(vert_coords, faces):
    """
    Compute the area of all faces.
    """
    num_faces = faces.shape[0]
    all_faces_first_vert_coords_xyz = vert_coords[faces[:,0]]
    all_faces_second_vert_coords_xyz = vert_coords[faces[:,1]]
    all_faces_third_vert_coords_xyz = vert_coords[faces[:,2]]
    areas = np.zeros((num_faces,))
    for i in range(num_faces):
        areas[i] = face_area(all_faces_first_vert_coords_xyz[i,:], all_faces_second_vert_coords_xyz[i,:], all_faces_third_vert_coords_xyz[i,:])
    return areas


def face_area(a, b, c):
    """
    Compute area of 3D triangles (faces).

    Compute the area of the 3D triangles whos 3 points are given in a, b, and c.
    """
    return 0.5 * norm( np.cross( b-a, c-a ) )



if __name__ == "__main__":
    intersurface()
