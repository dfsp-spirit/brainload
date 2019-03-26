#!/usr/bin/env python
from __future__ import print_function
import os
import sys
import errno
import numpy as np
import scipy
import argparse
import brainload as bl
from numpy.linalg import norm
import brainload.surfacegraph as sg
import brainload.freesurferdata as fsd
import networkx
from scipy.spatial import ConvexHull
import csv

# To run this in dev mode (in virtual env, pip -e install of brainview active) from REPO_ROOT:
# PYTHONPATH=./src/brainload python src/brainload/intersurface.py tim -d ~/data/tim_only/ --hemi lh -c expected_vol_fs -v


def intersurface():
    parser = argparse.ArgumentParser(description="Compute and write intersurface data.")
    parser.add_argument("subject", help="The subject you want to load. String, a directory under the subjects_dir.")
    parser.add_argument("-d", "--subjects_dir", help="The subjects_dir containing the subject. Defaults to environment variable SUBJECTS_DIR.", default="")
    parser.add_argument("-f", "--first-surface", help="The first surface to load. String, defaults to 'white'.", default="white")
    parser.add_argument("-s", "--second-surface", help="The second surface to load. String, defaults to 'pial'.", default="pial")
    parser.add_argument("-e", "--hemi", help="The hemisphere to load. One of ('lh, 'rh').", choices=['lh', 'rh'])
    parser.add_argument("-c", "--compute-value", help="What to compute for the two surfaces. One of ('expected_vol', 'expected_vol_fs', 'actual_vol', 'all'). Defaults to 'all'.", default="all", choices=['expected_vol', 'expected_vol_fs', 'actual_vol', 'all'])
    parser.add_argument("-o", "--outputfile", help="Output CSV file name. Ignored unless -c is 'all'. If not given, the output is NOT saved to a file.", default=None)
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
    verbose = args.verbose

    print("-----Intersurface volume-----")
    print("Note that the volume computed at each vertex is based on the area of all faces adjacent to the vertex. The area is NOT divided by 3!")
    print("So the sum of the volumes over all vertices is NOT the volume of the cortex! It is much more - roughly 3 times more, but that depends on expected vs actual.")
    if verbose:
        print("Computing '%s' for subject '%s' in subjects dir '%s'." % (args.compute_value, subject_id, subjects_dir))

    if args.compute_value == "expected_vol":
        if verbose:
            print("Computing '%s' based on manual computation of face area of surface '%s' and cortical thickness, hemi '%s'." % (args.compute_value, surf1, hemi))
        vert_coords_surf1, faces_surf1, cortical_thickness, meta_data_surf1 = bl.subject(subject_id, subjects_dir=subjects_dir, measure=measure, surf=surf1, hemi=hemi)
        expected_volume = get_expected_volume_per_vertex(vert_coords_surf1, faces_surf1, cortical_thickness, verbose=verbose)
        print("Received expected volume for %d vertices." % (expected_volume.shape[0]))

    elif args.compute_value == "expected_vol_fs":
        if verbose:
            print("Computing '%s' based on FreeSurfer area and cortical thickness data for surface '%s', hemi '%s'." % (args.compute_value, surf1, hemi))
        area_curv_file = fsd.get_morphometry_file_path(subjects_dir, subject_id, surf1, hemi, "area")
        per_vertex_area, meta_data_area_curv_file = fsd.read_fs_morphometry_data_file_and_record_meta_data(area_curv_file, hemi)

        thickness_curv_file = fsd.get_morphometry_file_path(subjects_dir, subject_id, surf1, hemi, "thickness")
        per_vertex_thickness, meta_data_thickness_curv_file = fsd.read_fs_morphometry_data_file_and_record_meta_data(thickness_curv_file, hemi)

        expected_fs_volume = (per_vertex_area * 3) * per_vertex_thickness
        print("Received expected FS volume based on thickness and area FreeSurfer files for %d vertices." % (expected_fs_volume.shape[0]))

    elif args.compute_value == "actual_vol":
        if verbose:
            print("Computing '%s' between surfaces '%s' and '%s', hemi '%s'." % (args.compute_value, surf1, surf2, hemi))
        vert_coords_surf1, faces_surf1, meta_data_surf1 = bl.subject_mesh(subject_id, subjects_dir, surf=surf1, hemi=hemi)
        vert_coords_surf2, faces_surf2, meta_data_surf2 = bl.subject_mesh(subject_id, subjects_dir, surf=surf2, hemi=hemi)
        actual_volume = get_actual_volume_per_vertex(vert_coords_surf1, faces_surf1, vert_coords_surf2, faces_surf2, verbose=verbose)
        print("Received actual volume for %d vertices." % (actual_volume.shape[0]))

    elif args.compute_value == "all":
        if verbose:
            print("Computing both expected and actual volumes between surfaces '%s' and '%s', hemi '%s'. May take a while." % (surf1, surf2, hemi))

        # collect data
        area_curv_file_surf2 = fsd.get_morphometry_file_path(subjects_dir, subject_id, surf2, hemi, "area")
        per_vertex_area_surf2, meta_data_area_curv_file_surf2 = fsd.read_fs_morphometry_data_file_and_record_meta_data(area_curv_file_surf2, hemi)
        area_curv_file = fsd.get_morphometry_file_path(subjects_dir, subject_id, surf1, hemi, "area")
        per_vertex_area, meta_data_area_curv_file = fsd.read_fs_morphometry_data_file_and_record_meta_data(area_curv_file, hemi)
        thickness_curv_file = fsd.get_morphometry_file_path(subjects_dir, subject_id, surf1, hemi, "thickness")
        per_vertex_thickness, meta_data_thickness_curv_file = fsd.read_fs_morphometry_data_file_and_record_meta_data(thickness_curv_file, hemi)

        # almost actual but quick: ignores the relative positions of the 2 vertices and assume they are right underneath each other (assumes their surface normals were identical)
        #
        # has_invalid_area_data = (per_vertex_area < 0).any()
        actual_volume_fs = 1./3. * per_vertex_thickness * (per_vertex_area + per_vertex_area_surf2 + np.sqrt(per_vertex_area * per_vertex_area_surf2))
        if verbose:
            print("Actual volume based on FreeSurfer data computed.")

        # expected vol according to our computations
        vert_coords_surf1, faces_surf1, cortical_thickness, meta_data_surf1 = bl.subject(subject_id, subjects_dir=subjects_dir, measure=measure, surf=surf1, hemi=hemi)
        expected_volume = get_expected_volume_per_vertex(vert_coords_surf1, faces_surf1, cortical_thickness, verbose=verbose, verbose_print_each=10000)
        if verbose:
            print("Received expected volume for %d vertices." % (expected_volume.shape[0]))

        # expected vol based on FreeSurfer curv files: area * thickness
        expected_volume_fs = (per_vertex_area * 3) * per_vertex_thickness

        # actual volume, computed from volume of polygon between the points of all neighboring faces of the 2 vertices (one per surface)
        vert_coords_surf2, faces_surf2, meta_data_surf2 = bl.subject_mesh(subject_id, subjects_dir, surf=surf2, hemi=hemi)
        actual_volume = get_actual_volume_per_vertex(vert_coords_surf1, faces_surf1, vert_coords_surf2, faces_surf2, verbose=verbose, verbose_print_each=10000)
        if verbose:
            print("Received actual volume for %d vertices." % (actual_volume.shape[0]))
            print("Total expected volume is %d, total actual volume is %d over the %d vertices." % (expected_volume.sum(), actual_volume.sum(), actual_volume.shape[0]))

        ### report results
        for vertex_id in range(vert_coords_surf1.shape[0]):
            if verbose and vertex_id % 1000 == 0:
                print('At vertex %d. Expected volume (comp. face area * thickness) is %f, expected fs volume (area * thickness) is %f, actual polygon volume is %f, almost is %f.' % (vertex_id, expected_volume[vertex_id], expected_volume_fs[vertex_id], actual_volume[vertex_id], actual_volume_fs[vertex_id]))

        ### write results to CSV file
        if args.outputfile is not None:
            output_csv_file = args.outputfile
            header_field_names = ["vertex_id", "expected_vol", "expected_vol_fs", "actual_vol", "actual_vol_fs", "ratio_expected_by_actual", "ratio_expected_fs_by_actual_fs"]
            ratio_expected_by_actual = expected_volume / actual_volume
            ratio_expected_fs_by_actual_fs = expected_volume_fs / actual_volume_fs
            with open(output_csv_file, 'w') as csvfile:
                feature_writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                feature_writer.writerow(header_field_names)
                for vertex_id in range(vert_coords_surf1.shape[0]):
                    feature_writer.writerow([vertex_id, expected_volume[vertex_id], expected_volume_fs[vertex_id], actual_volume[vertex_id], actual_volume_fs[vertex_id], ratio_expected_by_actual[vertex_id], ratio_expected_fs_by_actual_fs[vertex_id]])
            print("Output CSV file written to '%s'." % (output_csv_file))

    else:
        print("ERROR: Invalid computation value. Exiting.")
        sys.exit(1)



def get_actual_volume_per_vertex(vert_coords_surf1, faces_surf1, vert_coords_surf2, faces_surf2, verbose=False, verbose_print_each=1000):
    num_vertices_surf1 = vert_coords_surf1.shape[0]
    num_vertices_surf2 = vert_coords_surf2.shape[0]

    if num_vertices_surf1 != num_vertices_surf2:
        print("ERROR: Surfaces do not have identical vertex count: %d vs %d. Exiting." % (num_vertices_surf1, num_vertices_surf2))
        sys.exit(1)

    # create surface graphs so we can find all neighboring vertices quickly
    surface_graph_surf1 = sg.SurfaceGraph(vert_coords_surf1, faces_surf1)
    surface_graph_surf2 = sg.SurfaceGraph(vert_coords_surf2, faces_surf2)

    actual_volume = np.zeros((num_vertices_surf1,))

    if verbose:
        print("Created graphs. Computing expected volume at each of the %d vertices of the surface." % (num_vertices_surf1))

    for source_vertex in range(num_vertices_surf1):
        neighbors_surf1 = surface_graph_surf1.get_neighbors_up_to_dist(source_vertex, 1)
        neighbors_surf2 = surface_graph_surf2.get_neighbors_up_to_dist(source_vertex, 1)
        coords_of_surf1_vertices = np.array(vert_coords_surf1[neighbors_surf1])
        coords_of_surf2_vertices = np.array(vert_coords_surf2[neighbors_surf2])
        all_coords = np.concatenate((coords_of_surf1_vertices, coords_of_surf2_vertices))
        actual_volume[source_vertex] = get_convex_polygon_volume(all_coords)
        if verbose and source_vertex % verbose_print_each == 0:
            print('At vertex %d. Vertex has %d neighbors in surface1 and %d in surface2. Actual volume between surfaces at vertex is %f.' % (source_vertex, len(neighbors_surf1), len(neighbors_surf2), actual_volume[source_vertex]))
    return actual_volume


def get_expected_volume_per_vertex(vert_coords, faces, cortical_thickness, verbose=False, verbose_print_each=1000):
    face_areas = get_mesh_face_areas(vert_coords, faces)
    if verbose:
        print("Computed %d areas for all %d faces of the surface." % (face_areas.shape[0], faces.shape[0]))

    num_vertices = vert_coords.shape[0]
    area_all_faces_around_vertex = np.zeros((num_vertices,))

    if verbose:
        print("Computing expected volume at each of the %d vertices of the surface." % (num_vertices))

    for vert_idx in range(num_vertices):
        mask = np.any(faces == vert_idx, axis=1)
        all_face_indices = np.nonzero(mask)[0]
        #faces_with_vert_at_pos0 = np.nonzero(faces_surf1[:,0]==vert_idx)[0]
        #faces_with_vert_at_pos1 = np.nonzero(faces_surf1[:,1]==vert_idx)[0]
        #faces_with_vert_at_pos2 = np.nonzero(faces_surf1[:,2]==vert_idx)[0]
        #all_face_indices = np.unique(np.concatenate((faces_with_vert_at_pos0, faces_with_vert_at_pos1, faces_with_vert_at_pos2)))
        summed_area = face_areas[all_face_indices].sum()
        area_all_faces_around_vertex[vert_idx] = summed_area
        if verbose and vert_idx % verbose_print_each == 0:
            print('At vertex %d. Vertex is part of the following %d faces with total area %f: %s' % (vert_idx, len(all_face_indices), summed_area, ','.join([str(x) for x in all_face_indices])))
    expected_volume = area_all_faces_around_vertex * cortical_thickness
    return expected_volume


def get_convex_polygon_volume(points):
    """
    Compute the volume of a convex polygon.

    Compute the volume of a convex polygon, defined by its points in 3D space. Uses the convex hull of the points.
    """
    return ConvexHull(points).volume      # ConvexHull is: from scipy.spatial import ConvexHull


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
    Compute area of a 3D triangles (face).

    Compute the area of the 3D triangle whos 3 points are given in a, b, and c.
    """
    return 0.5 * norm( np.cross( b-a, c-a ) )



if __name__ == "__main__":
    intersurface()
