"""
Check data quality.

Functions to perform consistency checks on neuroimaging data preprocessed with FreeSurfer.
"""

import os, sys
import numpy as np
import nibabel.freesurfer.io as fsio
import brainload.nitools as nit
import brainload.freesurferdata as fsd
import brainload.stats
import logging
import collections

class BrainDataConsistency:
    """
    Check brain data consistency for one or more subjects.

    This class parses standard data files which are written by FreeSurfer when a subject is preprocessed using the recon-all pipeline. It then performs various consistency checks on the data.
    These checks include comparing the number of vertices in surfaces and morphometry files.
    """
    def __init__(self, subjects_dir, subjects_list, hemi='both'):
        self.subjects_dir = subjects_dir
        self.subjects_list = subjects_list

        self.data = dict()

        self.subject_issues = dict()
        for subject_id in self.subjects_list:
            self.subject_issues[subject_id] = []

        if hemi not in ('lh', 'rh', 'both'):
            raise ValueError("ERROR: hemi must be one of {'lh', 'rh', 'both'} but is '%s'." % hemi)
        if hemi == 'both':
            self.hemis = ['lh', 'rh']
            self.data['lh'] = dict()
            self.data['rh'] = dict()
        else:
            self.hemis = [hemi]
            self.data[hemi] = dict()
        logging.info("BrainDataConsistency instance initialized, handling %s subjects in subjects_dir '%s'." % (len(self.subjects_list), self.subjects_dir))


    def check(self):
        self._count_surface_vertices_and_faces()
        self._check_surfaces_have_identical_vertex_count()
        self._check_native_space_data(["area", "volume", "pial_lgi"])
        self._report_by_subject()


    def _count_surface_vertices_and_faces(self):
        for hemi in self.hemis:
            self.data[hemi]['mesh_vertex_count_white'] = np.zeros((len(self.subjects_list), 0))
            self.data[hemi]['mesh_face_count_white'] = np.zeros((len(self.subjects_list), 0))
            self.data[hemi]['mesh_vertex_count_pial'] = np.zeros((len(self.subjects_list), 0))
            self.data[hemi]['mesh_face_count_pial'] = np.zeros((len(self.subjects_list), 0))

            for subject_index, subject_id in enumerate(self.subjects_list):
                verts_white, faces_white, meta_data_white = fsd.subject_mesh(subject_id, self.subjects_dir, surf='white', hemi=hemi)
                self.data[hemi]['mesh_vertex_count_white'][subject_idx] = len(verts_white)
                self.data[hemi]['mesh_face_count_white'][subject_idx] = len(faces_white)
                verts_pial, faces_pial, meta_data_pial = fsd.subject_mesh(subject_id, self.subjects_dir, surf='white', hemi=hemi)
                self.data[hemi]['mesh_vertex_count_pial'][subject_idx] = len(verts_pial)
                self.data[hemi]['mesh_face_count_pial'][subject_idx] = len(faces_pial), 0))


    def _check_surfaces_have_identical_vertex_count(self):
        for hemi in self.hemis:
            issue_tag = "VERT_MISMATCH_FACES_%s" % (hemi)
            for subject_index, subject_id in enumerate(self.subjects_list):
                if self.data[hemi]['mesh_vertex_count_white'][subject_idx] != self.data[hemi]['mesh_vertex_count_pial'][subject_idx]:
                    logging.warn("[%s][%s] Vertex count mismatch between surfaces white and pial: %d != %d." % (subject_id, hemi, self.data[hemi]['mesh_vertex_count_white'][subject_idx], self.data[hemi]['mesh_vertex_count_pial'][subject_idx]))
                    self.subject_issues[subject_id].append(issue_tag)


    def _check_native_space_data(self, measures_list):
        for measure in measures_list:
            for hemi in self.hemis:
                measure_key = "morphometry_vertex_data_count_%s" % (measure)
                issue_tag = "MORPH_MISMATCH_%s_%s" % (measure, hemi)
                self.data[hemi][measure_key] = np.zeros((len(self.subjects_list), 0))
                for subject_index, subject_id in enumerate(self.subjects_list):
                    morphometry_data, meta_data = fsd.subject_data_native(subject_id, self.subjects_dir, measure, hemi, surf='white')
                    self.data[hemi][measure_key][subject_idx] = len(morphometry_data)
                    if len(morphometry_data) != self.data[hemi]['mesh_vertex_count_white'][subject_idx]:
                        logging.warn("[%s][%s] Mismatch between length of vertex data for native space measure '%s' and number of vertices of surface white: %d != %d." % (subject_id, hemi, measure, len(morphometry_data), self.data[hemi]['mesh_vertex_count_white'][subject_idx]))
                        self.subject_issues[subject_id].append(issue_tag)


    def _report_by_subject(self):
        num_ok = 0
        num_incons = 0
        for subject_index, subject_id in enumerate(self.subjects_list):
            if self.subject_issues[subject_id]:
                subject_report = "%d inconsistencies detected: %s" % (len(self.subject_issues[subject_id]), " ".join(self.subject_issues[subject_id]))
                num_incons = num_incons + 1
            else:
                subject_report = "OK."
                num_ok = num_ok + 1
            print("%s: %s" % (subject_id, subject_report))
        print("In total, %d OK and %d with inconsistencies." % (num_ok, num_incons))
