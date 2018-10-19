#!/usr/bin/env python

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from brainload.spatial_transform import *
from brainview.singleview import *
from brainload.freesurferdata import *

def run_example():

    subjects_file = 'subjects.txt'

    ## Parse command line
    print "--------------------------------- brain mesh fun --------------------------------------"
    if len(sys.argv) > 1:
        subjects_file = sys.argv[1]

    ## Read subjects file
    df_subjects = pd.read_csv(subjects_file, names=["subject_id"])
    subject_ids = df_subjects["subject_id"]

    ## Read data
    subjects_dir = os.getenv('SUBJECTS_DIR', '.')
    if len(sys.argv) > 2:
        subjects_dir = sys.argv[2]


    for subject_id in subject_ids:
        print "Handling subject %s." % subject_id

        #vert_coords, faces, per_vertex_data, meta_data = parse_brain_files(lh_surf, rh_surf, curv_lh=lh_data, curv_rh=rh_data)
        #vert_coords, faces, per_vertex_data, meta_data = parse_subject(subject_id)
        vert_coords, faces, per_vertex_data, meta_data = parse_subject(subject_id, surf='pial', measure='area')
        #vert_coords, faces, per_vertex_data, meta_data = parse_subject_standard_space_data(subject_id, display_surf='inflated', measure='area')

        debug = True
        if debug:
            vertex_data = { 'vidx': np.arange(vert_coords[:,0].size), 'x': vert_coords[:,0], 'y': vert_coords[:,1], 'z': vert_coords[:,2], 'value': per_vertex_data}
            df_brain_vertices = pd.DataFrame(data=vertex_data)
            print "Vertex data:"
            print df_brain_vertices.describe()

            face_data = { 'fidx': np.arange(faces[:,0].size), 'vidx1': faces[:,0], 'vidx2': faces[:,1], 'vidx3': faces[:,2]}
            df_brain_faces = pd.DataFrame(data=face_data, dtype=np.int32)
            print "Faces data:"
            print df_brain_faces.describe()


        try_pymesh = False
        plot_pymesh_hist = False
        if try_pymesh:
            print "Trying PyMesh"
            import pymesh            # Install this from a wheel on the github page, see https://github.com/PyMesh/PyMesh/issues/110. The one in pip is broken. (Also note that the broken one is called pymesh2 in pip: pymesh on pip is a completely different library.)
            mesh = pymesh.meshio.form_mesh(vert_coords, faces)
            print(mesh.num_vertices, mesh.num_faces, mesh.num_voxels)
            print " * Computing Gaussian curvature for all vertices..."
            mesh.add_attribute('vertex_gaussian_curvature') # compute Gaussian curvature for all verts
            sr_gaussian_curvature = pd.Series(mesh.get_attribute('vertex_gaussian_curvature'))

            df_subject_mesh_data = pd.DataFrame({ 'gaussian_curvature' : sr_gaussian_curvature })
            print df_subject_mesh_data.describe()


            #sr_filtered = sr_gaussian_curvature.where(lambda x : x >-1.0).where(lambda x : x <1.0).dropna()
            limits = [-0.01, 0.01]
            sr_filtered_gaussian_curvature = sr_gaussian_curvature.where(lambda x : x > limits[0]).where(lambda x : x < limits[1])

            if plot_pymesh_hist:
                ax = sr_filtered_gaussian_curvature.plot.hist(alpha=0.5, bins=50, xlim=(-1.5, 1.5))
                plt.title('Histogram of Gaussian curvature')
                plt.xlabel('Gaussian curvature')
                plt.ylabel('Frequency')
                plt.show()

        try_mayavi = True
        if try_mayavi:
            scalars = per_vertex_data                    # data loaded from FreeSurfer morphology (a.k.a. 'curv') files
            #scalars = np.arange(x.size)                 # just map increasing numbers (fake data, but fast)
            #scalars = sr_filtered_gaussian_curvature.values            # Gaussian curvature from PyMesh computation
            plot_brain_data_on_mesh(vert_coords, faces, scalars, meta_data, export_image_file='brain.jpg')

if __name__ == '__main__':
    run_example()
