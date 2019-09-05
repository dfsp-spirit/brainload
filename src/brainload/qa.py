"""
Check data quality.

Functions to perform consistency checks on neuroimaging data preprocessed with FreeSurfer.
"""

import os, sys
import numpy as np
import nibabel.freesurfer.io as fsio
from . import nitools as nit
from . import freesurferdata as fsd
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
        self.check_file_modification_times = False
        self.time_format = '%Y-%m-%d %H:%M:%S'
        self.time_buffer = 2.0    # For files were one should have been created after the other, define a grace period in seconds.

        self.issue_explanations = self._issue_tag_explanation_dict()

        self.subject_issues = dict()
        self.subject_issues_assoc_files = dict()
        for subject_id in self.subjects_list:
            self.subject_issues[subject_id] = []
            self.subject_issues_assoc_files[subject_id] = []

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
        self._check_subject_dirs_exist()
        if not self.surface_vertices_counted:
            self._count_surface_vertices_and_faces()
        self._check_surfaces_have_identical_vertex_count()
        self._check_native_space_data(["area", "volume", "thickness"])
        self._report_by_subject()


    def check_custom(self, native_measures):
        logging.info("Performing custom checks for %s native measures: %s." % (len(native_measures), ", ".join(native_measures)))
        self._check_subject_dirs_exist()
        if not self.surface_vertices_counted:
            self._count_surface_vertices_and_faces()
        self._check_surfaces_have_identical_vertex_count()
        self._check_native_space_data(native_measures)
        self._report_by_subject()


    def _check_subject_dirs_exist(self):
        issue_tag_no_data_dir = "ALL_SUBJECT_DATA_MISSING"
        for subject_index, subject_id in enumerate(self.subjects_list):
            current_subject_dir = os.path.join(self.subjects_dir, subject_id)
            if not os.path.isdir(current_subject_dir):
                logging.warning("[%s] Missing subject data directory '%s'." % (subject_id, current_subject_dir))
                self._append_issue(subject_id, issue_tag_no_data_dir, current_subject_dir)
        logging.info("Checked all subject dirs for existance.")


    def _count_surface_vertices_and_faces(self, surfaces=None):
        if surfaces is None:
            surfaces = ['white', 'pial']

        for hemi in self.hemis:
            self.data[hemi]['mesh_vertex_count_white'] = np.zeros((len(self.subjects_list), ))
            self.data[hemi]['mesh_face_count_white'] = np.zeros((len(self.subjects_list), ))
            self.data[hemi]['mesh_vertex_count_pial'] = np.zeros((len(self.subjects_list), ))
            self.data[hemi]['mesh_face_count_pial'] = np.zeros((len(self.subjects_list), ))

            for subject_index, subject_id in enumerate(self.subjects_list):
                for surf in surfaces:
                    try:
                        verts, faces, meta_data = fsd.subject_mesh(subject_id, self.subjects_dir, surf=surf, hemi=hemi)
                        md_surf_file_key = "%s.surf_file" % (hemi)
                        self.files[hemi]['surf_%s' % (surf)] = meta_data[md_surf_file_key]
                        self.data[hemi]['mesh_vertex_count_%s' % (surf)][subject_index] = len(verts)
                        self.data[hemi]['mesh_face_count_%s' % (surf)][subject_index] = len(faces)
                    except (OSError, IOError):
                        issue_tag = "NO_SURFACE_FILE__%s_%s" % (surf, hemi)
                        missing_surface_file = fsd.get_surface_file_path(self.subjects_dir, subject_id, hemi, surf)
                        self._append_issue(subject_id, issue_tag, missing_surface_file)
                        logging.warning("[%s][%s] Missing surface file for surface '%s': '%s'." % (subject_id, hemi, surf, missing_surface_file))
        self.surface_vertices_counted = True
        logging.info("Counted vertices of %d surfaces for all %s subjects (%s)." % (len(surfaces), len(self.subjects_list), ", ".join(surfaces)))


    def _check_surfaces_have_identical_vertex_count(self, surface_pair=None):
        if surface_pair is None:
            surface_pair = ['white', 'pial']
        s0 = surface_pair[0]
        s1 = surface_pair[1]
        logging.info("Verifying that the '%s' and '%s' surfaces have identical vertex counts." % (s0, s1))
        for hemi in self.hemis:
            issue_tag = "VERT_MISMATCH_SURFACES__%s_%s_%s" % (s0, s1, hemi)
            for subject_index, subject_id in enumerate(self.subjects_list):
                s0_count = self.data[hemi]['mesh_vertex_count_%s' % (s0)][subject_index]
                s1_count = self.data[hemi]['mesh_vertex_count_%s' % (s1)][subject_index]
                if s0_count != s1_count:
                    logging.warning("[%s][%s] Vertex count mismatch between surfaces %s and %s: %d != %d." % (subject_id, hemi, s0, s1, s0_count, s1_count))
                    self._append_issue(subject_id, issue_tag, fsd.get_surface_file_path(self.subjects_dir, subject_id, hemi, s1))
        logging.info("Verified that the surfaces '%s' and '%s' have the same number of vertices for each subject." % (s0, s1))


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


    def _append_issue(self, subject_id, tag, filename):
        self.subject_issues[subject_id].append(tag)
        self.subject_issues_assoc_files[subject_id].append(filename)


    def _check_native_space_data(self, measures_list):
        for measure in measures_list:
            logging.info("Verifying native space data for measure '%s'." % (measure))
            for hemi in self.hemis:
                measure_key = "morphometry_vertex_data_count_%s" % (measure)
                issue_tag = "MORPH_MISMATCH__%s_%s" % (measure, hemi)
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
                                    logging.warning("[%s][%s] Morphometry file for measure '%s' was last changed earlier than surface file: %s is before %s (%s)." % (subject_id, hemi, measure, self._pts(ts_morph_file), self._pts(ts_surf_file), self._ptd(ts_morph_file-ts_surf_file)))
                                    issue_tag_file_time = "TIME_MORPH_FILE__%s_%s" % (measure, hemi)
                                    self._append_issue(subject_id, issue_tag_file_time, morph_data_file)


                    except (OSError, IOError):
                        morphometry_data = np.array([])
                        self.data[hemi][measure_key][subject_index] = len(morphometry_data) # = 0
                        issue_tag_no_file = "MISSING_MORPH_FILE__%s_%s" % (measure, hemi)
                        morph_data_file = fsd.get_morphometry_file_path(self.subjects_dir, subject_id, 'white', hemi, measure)
                        self._append_issue(subject_id, issue_tag_no_file, morph_data_file)
                        logging.warning("[%s][%s] Missing file for native space vertex data of measure '%s': '%s'." % (subject_id, hemi, measure, morph_data_file))


                    if len(morphometry_data) != self.data[hemi]['mesh_vertex_count_white'][subject_index]:
                        logging.warning("[%s][%s] Mismatch between length of vertex data for native space measure '%s' and number of vertices of surface white: %d != %d." % (subject_id, hemi, measure, len(morphometry_data), self.data[hemi]['mesh_vertex_count_white'][subject_index]))
                        self._append_issue(subject_id, issue_tag, morph_data_file)
            logging.info("Checked native space data for measure '%s' for consistency." % (measure))


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


    def get_issue_tag_explanation(self, issue_tag):
        base_issue_tag = issue_tag.split("__")[0]    # If the tag contains "__", just use the prefix.
        return self.issue_explanations.get(base_issue_tag, "Sorry, no explanation available for issue tag '%s' (from '%s')." % (base_issue_tag, issue_tag))


    def _issue_tag_explanation_dict(self):
        expl = dict()
        expl['MISSING_MORPH_FILE'] = "The native space morphology file cannot be read."
        expl['TIME_MORPH_FILE'] = "The native space morphology file was created/modified before the respective surface file. Note: This can report false positives if the files were changed afterwards, depending on your filesystem or from copying the files."
        expl['MORPH_MISMATCH'] = "The value count in the native space morphology file does not match the number of vertices in the surface file."
        expl['VERT_MISMATCH_SURFACES'] = "The vertex count does not match for the surface pair."
        expl['NO_SURFACE_FILE'] = 'The surface file cannot be read.'
        expl['ALL_SUBJECT_DATA_MISSING'] = 'The subject directory for the subject cannot be read.'
        return expl


    def save_html_report(self, filename):
        report = self._report_html()
        with open(filename, "w") as text_file:
            text_file.write(report)
        logging.info("HTML report written to file '%s'." % (filename))


    def _get_css_style(self):
        return """<style>
        body {
          width: 95%;
        }

        table {
          border-collapse: collapse;
          border: 1px solid silver;
          width: 100%;
        }

        th {
          text-align: center;
          padding: 8px;
        }

        td {
          text-align: left;
          padding: 8px;
        }

        td.check_issue {
          background-color: #99555555
        }

        td.count_no_issue {
          background-color: #55995555
        }

        td.count_has_issue {
          background-color: #99555555
        }

        tr:nth-child(even){background-color: #D5D5D5}

        tr:hover {
          background: silver;
          cursor: pointer;
        }

        th {
          background-color: #0000FF;
          color: white;
        }
        </style>"""


    def _report_html(self):
        all_issue_types = []
        for subject_index, subject_id in enumerate(self.subjects_list):
            all_issue_types.extend(self.subject_issues[subject_id])
        unique_issues = list(set(all_issue_types))

        header = "<html>\n<head>\n%s</head>\n<body>\n"  % (self._get_css_style())
        prefix = "<h1>Braindata QA Report</h1><h4>Hover mouse over issues to see full file path.</h4>\n"
        table_end = "</table>\n"
        footer = "</body>\n</html>"

        table_header = "<table class='issues_table'>\n<tr><th title='The subject identifier'>subject_id</th><th title='Number of issues detected for this subject'>num_issues</th>"
        for issue in unique_issues:
            table_header = table_header + "<th title='%s'>%s</th>" % (self.get_issue_tag_explanation(issue), issue)
        table_header = table_header + "</tr>\n"

        table_body = ""

        for subject_index, subject_id in enumerate(self.subjects_list):
            class_issue_or_not = 'count_no_issue'
            if self.subject_issues[subject_id]:
                class_issue_or_not = 'count_has_issue'
            table_row = "<tr><td class='subject_id'>%s</td><td class='issue_count_subject %s'>%d</td>" % (subject_id, class_issue_or_not, len(self.subject_issues[subject_id]))
            for issue in unique_issues:
                if issue in self.subject_issues[subject_id]:
                    issue_index = self.subject_issues[subject_id].index(issue)
                    related_file = self.subject_issues_assoc_files[subject_id][issue_index]
                    table_row = table_row + "<td class='check_issue' title='%s'>%s</td>\n" % (related_file, issue)
                else:
                    table_row = table_row + "<td class='check_ok'>ok</td>\n"
            table_row = table_row + "</tr>\n"
            table_body = table_body + table_row

        table = table_header + table_body + table_end
        suffix = "<h4>Checked %d subjects for issues.</h4>\n" % (len(self.subjects_list))
        html = header + prefix + table + suffix + footer
        return html
