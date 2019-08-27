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
import errno
import datetime


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
        self.files = dict()

        self.surface_vertices_counted = False
        self.check_file_modification_times = True
        self.time_format = '%Y-%m-%d %H:%M:%S'
        self.time_buffer = 2.0    # For files were one should have been created after the other, define a grace period in seconds.

        self.subject_issues = dict()
        for subject_id in self.subjects_list:
            self.subject_issues[subject_id] = []

        if hemi not in ('lh', 'rh', 'both'):
            raise ValueError("ERROR: hemi must be one of {'lh', 'rh', 'both'} but is '%s'." % hemi)
        if hemi == 'both':
            self.hemis = ['lh', 'rh']
            self.data['lh'] = dict()
            self.data['rh'] = dict()
            self.files['lh'] = dict()
            self.files['rh'] = dict()
        else:
            self.hemis = [hemi]
            self.data[hemi] = dict()
            self.files[hemi] = dict()
        logging.info("BrainDataConsistency instance initialized, handling %s subjects in subjects_dir '%s'." % (len(self.subjects_list), self.subjects_dir))


    def check_essentials(self):
        logging.info("Checking essentials.")
        if not self.surface_vertices_counted:
            self._count_surface_vertices_and_faces()
        self._check_surfaces_have_identical_vertex_count()
        self._check_native_space_data(["area", "volume", "thickness"])
        self._report_by_subject()


    def check_custom(self, native_measures):
        logging.info("Performing custom checks.")
        if not self.surface_vertices_counted:
            self._count_surface_vertices_and_faces()
        self._count_surface_vertices_and_faces()
        self._check_surfaces_have_identical_vertex_count()
        self._check_native_space_data(native_measures)
        self._report_by_subject()


    def _count_surface_vertices_and_faces(self):
        for hemi in self.hemis:
            self.data[hemi]['mesh_vertex_count_white'] = np.zeros((len(self.subjects_list), ))
            self.data[hemi]['mesh_face_count_white'] = np.zeros((len(self.subjects_list), ))
            self.data[hemi]['mesh_vertex_count_pial'] = np.zeros((len(self.subjects_list), ))
            self.data[hemi]['mesh_face_count_pial'] = np.zeros((len(self.subjects_list), ))

            for subject_index, subject_id in enumerate(self.subjects_list):
                # TODO: We should try..catch here in case of missing surface files
                verts_white, faces_white, meta_data_white = fsd.subject_mesh(subject_id, self.subjects_dir, surf='white', hemi=hemi)
                md_surf_file_key = "%s.surf_file" % (hemi)
                self.files[hemi]['surf_white'] = meta_data_white[md_surf_file_key]
                self.data[hemi]['mesh_vertex_count_white'][subject_index] = len(verts_white)
                self.data[hemi]['mesh_face_count_white'][subject_index] = len(faces_white)
                verts_pial, faces_pial, meta_data_pial = fsd.subject_mesh(subject_id, self.subjects_dir, surf='white', hemi=hemi)
                self.files[hemi]['surf_pial'] = meta_data_white[md_surf_file_key]
                self.data[hemi]['mesh_vertex_count_pial'][subject_index] = len(verts_pial)
                self.data[hemi]['mesh_face_count_pial'][subject_index] = len(faces_pial)
        self.surface_vertices_counted = True


    def _check_surfaces_have_identical_vertex_count(self):
        logging.info("Verifying that the white and pial surfaces have identical vertex counts.")
        for hemi in self.hemis:
            issue_tag = "VERT_MISMATCH_FACES_%s" % (hemi)
            for subject_index, subject_id in enumerate(self.subjects_list):
                if self.data[hemi]['mesh_vertex_count_white'][subject_index] != self.data[hemi]['mesh_vertex_count_pial'][subject_index]:
                    logging.warn("[%s][%s] Vertex count mismatch between surfaces white and pial: %d != %d." % (subject_id, hemi, self.data[hemi]['mesh_vertex_count_white'][subject_index], self.data[hemi]['mesh_vertex_count_pial'][subject_index]))
                    self.subject_issues[subject_id].append(issue_tag)


    def _pts(self, timestamp):
        """
        Print a time stamp in readable format.
        """
        return datetime.datetime.utcfromtimestamp(timestamp).strftime(self.time_format)

    def _ptd(self, timediff_seconds):
        """
        Print a time difference in seconds in readable format.
        """
        if timediff_seconds < 0:
            rel = " earlier"
        else:
            rel = " later"
        return str(datetime.timedelta(seconds=abs(timediff_seconds))) + rel


    def _check_native_space_data(self, measures_list):
        for measure in measures_list:
            logging.info("Verifying native space data for measure '%s'." % (measure))
            for hemi in self.hemis:
                measure_key = "morphometry_vertex_data_count_%s" % (measure)
                issue_tag = "MORPH_MISMATCH_%s_%s" % (measure, hemi)
                self.data[hemi][measure_key] = np.zeros((len(self.subjects_list), ))
                for subject_index, subject_id in enumerate(self.subjects_list):
                    try:
                        morphometry_data, meta_data = fsd.subject_data_native(subject_id, self.subjects_dir, measure, hemi, surf='white')
                        self.data[hemi][measure_key][subject_index] = len(morphometry_data)
                        md_morph_file_key = '%s.morphometry_file' % (hemi)
                        morph_data_file = meta_data[md_morph_file_key]
                        if self.check_file_modification_times:
                            if self.files[hemi]['surf_pial'] is not None:
                                ts_morph_file = os.path.getmtime(morph_data_file)
                                ts_surf_file = os.path.getmtime(self.files[hemi]['surf_pial'])
                                if ts_morph_file + self.time_buffer < ts_surf_file:
                                    logging.warn("[%s][%s] Morphometry file for measure '%s' was last changed earlier than surface file: %s is before %s (%s)." % (subject_id, hemi, measure, self._pts(ts_morph_file), self._pts(ts_surf_file), self._ptd(ts_morph_file-ts_surf_file)))
                                    issue_tag_file_time = "TIME_MORPH_FILE_%s_%s" % (measure, hemi)
                                    self.subject_issues[subject_id].append(issue_tag_file_time)

                    except (OSError, IOError):
                        morphometry_data = np.array([])
                        self.data[hemi][measure_key][subject_index] = len(morphometry_data) # = 0
                        issue_tag_no_file = "MISSING_MORPH_FILE_%s_%s" % (measure, hemi)
                        self.subject_issues[subject_id].append(issue_tag_no_file)
                        logging.warn("[%s][%s] Missing file for native space vertex data of measure '%s'." % (subject_id, hemi, measure))


                    if len(morphometry_data) != self.data[hemi]['mesh_vertex_count_white'][subject_index]:
                        logging.warn("[%s][%s] Mismatch between length of vertex data for native space measure '%s' and number of vertices of surface white: %d != %d." % (subject_id, hemi, measure, len(morphometry_data), self.data[hemi]['mesh_vertex_count_white'][subject_index]))
                        self.subject_issues[subject_id].append(issue_tag)


    def _report_by_subject(self):
        num_ok = 0
        num_incons = 0
        print("----- Report by subject follows for %d subjects -----" % (len(self.subjects_list)))
        for subject_index, subject_id in enumerate(self.subjects_list):
            if self.subject_issues[subject_id]:
                subject_report = "%d inconsistencies: %s" % (len(self.subject_issues[subject_id]), " ".join(self.subject_issues[subject_id]))
                num_incons = num_incons + 1
            else:
                subject_report = "OK"
                num_ok = num_ok + 1
            print("%s: %s" % (subject_id, subject_report))
        print("----- End of report by subject -----")
        print("Summary: %d subjects OK and %d with inconsistencies out of %d total." % (num_ok, num_incons, len(self.subjects_list)))
