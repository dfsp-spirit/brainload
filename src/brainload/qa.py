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

LOG = logging.getLogger("brainload")        # module logger

SEVERITY_TIMESTAMP = 1
SEVERITY_DATA_MISMATCH = 2
SEVERITY_REQUESTED_FILE_MISSING = 3             # A file which was requested by a query is missing (like 'lh.area' if a check of native measure 'area' was requested)
SEVERITY_REQUESTED_FILE_BAD = 3                  # A file which was requested by a query exists but is broken (parsing it failed)
SEVERITY_EXPECTED_FILE_MISSING = 4              # A standard file which we need in any case to perform the checks is missing (like 'lh.white', which we need to check vertex counts of *any*  native space measure data)
SEVERITY_EXPECTED_FILE_BAD = 4                  # A standard file which we need in any case to perform the checks exists but is broken (parsing it failed)
SEVERITY_SUBJECTS_DIR_MISSING = 5



class BrainDataConsistency:
    """
    Check brain data consistency for one or more subjects.

    This class parses standard data files which are written by FreeSurfer when a subject is preprocessed using the recon-all pipeline. It then performs various consistency checks on the data.
    These checks include comparing the number of vertices in surfaces and morphometry files.
    """
    def __init__(self, subjects_dir, subjects_list, hemi='both', log_level=logging.INFO):
        self.log = logging.getLogger("BrainDataConsistency")
        self.log.setLevel(log_level)

        self.subjects_dir = subjects_dir
        self.subjects_list = subjects_list

        self.data = dict()
        self.files = dict()

        self.surface_vertices_counted = False
        self.check_file_modification_times = False
        self.time_format = '%Y-%m-%d %H:%M:%S'
        self.time_buffer = 2.0    # For files were one should have been created after the other, define a grace period in seconds.

        self.fwhm_list = ["0", "5", "10", "15", "20", "25"]     # The smoothing values (fwhm) settings to use when checking standard space data.
        self.average_subject = "fsaverage"                                # The name of the template subject for standard space data.
        self.average_subject_mesh_vertex_count = None
        self.average_subject_subjects_dir = subjects_dir

        self.issue_explanations = self._issue_tag_explanation_dict()

        self.subject_issues = dict()
        self.subject_issues_assoc_files = dict()
        self.subject_issues_severity = dict()
        for subject_id in self.subjects_list:
            self.subject_issues[subject_id] = []
            self.subject_issues_assoc_files[subject_id] = []
            self.subject_issues_severity[subject_id] = []

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

        self.report_native_measures_checked = list()
        self.report_standard_measures_checked = list()

        self.log.info("BrainDataConsistency instance initialized, handling %s subjects in subjects_dir '%s'." % (len(self.subjects_list), self.subjects_dir))


    def _prepare_native_space_checks(self):
        if not self.surface_vertices_counted:
            self._count_surface_vertices_and_faces()

    def _prepare_standard_space_checks(self):
        if self.average_subject_mesh_vertex_count is None:
            verts, _, _ = fsd.subject_mesh(self.average_subject, self.average_subject_subjects_dir, surf='white', hemi='lh')
            self.average_subject_mesh_vertex_count = len(verts)


    def check_essentials(self):
        self.log.info("Checking essentials.")
        self._check_subject_dirs_exist()
        self._check_surfaces_have_identical_vertex_count()
        self._check_native_space_data(["area", "volume", "thickness"])
        self._report_by_subject()


    def check_custom(self, native_measures, std_measures):
        self.log.info("Performing custom checks for %s native measures: %s." % (len(native_measures), ", ".join(native_measures)))
        self._check_subject_dirs_exist()
        self._check_surfaces_have_identical_vertex_count()
        self._check_native_space_data(native_measures)
        self._check_standard_space_data(std_measures)
        self._report_by_subject()


    def _check_subject_dirs_exist(self):
        issue_tag_no_data_dir = "NO_SUBJECT_DIR"
        for subject_index, subject_id in enumerate(self.subjects_list):
            current_subject_dir = os.path.join(self.subjects_dir, subject_id)
            if not os.path.isdir(current_subject_dir):
                self.log.warning("[%s] Missing subject data directory '%s'." % (subject_id, current_subject_dir))
                self._append_issue(subject_id, issue_tag_no_data_dir, current_subject_dir, SEVERITY_SUBJECTS_DIR_MISSING)
        self.log.info("Checked all subject dirs for existence.")


    def _count_surface_vertices_and_faces(self, surfaces=None):
        if surfaces is None:
            surfaces = ['white', 'pial']
        self.log.info("Counting the vertices and faces for the following %d surfaces: '%s'." % (len(surfaces), ' '.join(surfaces)))

        for hemi in self.hemis:
            self.log.debug("Counting the vertices and faces for hemisphere '%s'." % (hemi))
            self.data[hemi]['mesh_vertex_count_white'] = np.zeros((len(self.subjects_list), ))
            self.data[hemi]['mesh_face_count_white'] = np.zeros((len(self.subjects_list), ))
            self.data[hemi]['mesh_vertex_count_pial'] = np.zeros((len(self.subjects_list), ))
            self.data[hemi]['mesh_face_count_pial'] = np.zeros((len(self.subjects_list), ))

            for subject_index, subject_id in enumerate(self.subjects_list):
                for surf in surfaces:
                    self.log.debug("Counting the vertices and faces for subject '%s' surface '%s' hemisphere '%s' (subject %d / %d))." % (subject_id, surf, hemi, (subject_index+1), len(self.subjects_list)))
                    try:
                        verts, faces, meta_data = fsd.subject_mesh(subject_id, self.subjects_dir, surf=surf, hemi=hemi)
                        md_surf_file_key = "%s.surf_file" % (hemi)
                        self.files[hemi]['surf_%s' % (surf)] = meta_data[md_surf_file_key]
                        self.data[hemi]['mesh_vertex_count_%s' % (surf)][subject_index] = len(verts)
                        self.data[hemi]['mesh_face_count_%s' % (surf)][subject_index] = len(faces)
                    except (OSError, IOError):
                        issue_tag = "NO_SURFACE_FILE__%s_%s" % (surf, hemi)
                        surface_file = fsd.get_surface_file_path(self.subjects_dir, subject_id, hemi, surf)
                        self._append_issue(subject_id, issue_tag, surface_file, SEVERITY_EXPECTED_FILE_MISSING)
                        self.log.warning("[%s][%s] Missing surface file for surface '%s': '%s'." % (subject_id, hemi, surf, surface_file))
                    except (ValueError):
                        issue_tag = "BAD_SURFACE_FILE__%s_%s" % (surf, hemi)
                        surface_file = fsd.get_surface_file_path(self.subjects_dir, subject_id, hemi, surf)
                        self._append_issue(subject_id, issue_tag, surface_file, SEVERITY_EXPECTED_FILE_BAD)
                        self.log.warning("[%s][%s] Bad surface file for surface '%s': '%s'." % (subject_id, hemi, surf, surface_file))
        self.surface_vertices_counted = True
        self.log.info("Counted vertices of %d surfaces for all %s subjects (%s)." % (len(surfaces), len(self.subjects_list), ", ".join(surfaces)))


    def _check_surfaces_have_identical_vertex_count(self, surface_pair=None):
        self._prepare_native_space_checks()
        if surface_pair is None:
            surface_pair = ['white', 'pial']
        s0 = surface_pair[0]
        s1 = surface_pair[1]
        self.log.info("Verifying that the '%s' and '%s' surfaces have identical vertex counts." % (s0, s1))
        for hemi in self.hemis:
            issue_tag = "MISMATCH_VERTS_SURFACES__%s_%s_%s" % (s0, s1, hemi)
            for subject_index, subject_id in enumerate(self.subjects_list):
                s0_count = self.data[hemi]['mesh_vertex_count_%s' % (s0)][subject_index]
                s1_count = self.data[hemi]['mesh_vertex_count_%s' % (s1)][subject_index]
                if s0_count != s1_count:
                    self.log.warning("[%s][%s] Vertex count mismatch between surfaces %s and %s: %d != %d." % (subject_id, hemi, s0, s1, s0_count, s1_count))
                    self._append_issue(subject_id, issue_tag, fsd.get_surface_file_path(self.subjects_dir, subject_id, hemi, s1), SEVERITY_REQUESTED_FILE_MISSING)
        self.log.info("Verified that the surfaces '%s' and '%s' have the same number of vertices for each subject." % (s0, s1))


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


    def _append_issue(self, subject_id, tag, filename, severity):
        """
        Append an issue for a subject.

        Parameters
        ----------
        subject_id: string
            A subject identifier.

        tag: string
            A string identifying the issue. Should be listed in the explanation dictionary.

        filename: string
            Path to the file associated with the isse, pass empty string if none. Example: If the issue is that a file is missing or has a wrong vertex count, pass the file.

        severity: int
            How severe the issue is, from 1 to 5. Higher is more severe. If the whole subject directory is missing, this is more severe than a wrong date on a file. This is used in the report.
        """
        self.subject_issues[subject_id].append(tag)
        self.subject_issues_assoc_files[subject_id].append(filename)
        self.subject_issues_severity[subject_id].append(severity)


    def _check_native_space_data(self, measures_list):
        self._prepare_native_space_checks()
        for measure in measures_list:
            self.log.info("Verifying native space data for measure '%s'." % (measure))
            for hemi in self.hemis:
                measure_key = "morphometry_vertex_data_count_%s" % (measure)
                issue_tag = "MISMATCH_MORPH_NAT_SURFACE__%s_%s" % (measure, hemi)
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
                                    self.log.warning("[%s][%s] Morphometry file for measure '%s' was last changed earlier than surface file: %s is before %s (%s)." % (subject_id, hemi, measure, self._pts(ts_morph_file), self._pts(ts_surf_file), self._ptd(ts_morph_file-ts_surf_file)))
                                    issue_tag_file_time = "TIME_NAT_MORPH_FILE__%s_%s" % (measure, hemi)
                                    self._append_issue(subject_id, issue_tag_file_time, morph_data_file, SEVERITY_TIMESTAMP)


                    except (OSError, IOError):
                        morphometry_data = np.array([])
                        self.data[hemi][measure_key][subject_index] = len(morphometry_data) # = 0
                        issue_tag_no_file = "NO_NAT_MORPH_FILE__%s_%s" % (measure, hemi)
                        morph_data_file = fsd.get_morphometry_file_path(self.subjects_dir, subject_id, 'white', hemi, measure)
                        self._append_issue(subject_id, issue_tag_no_file, morph_data_file, SEVERITY_REQUESTED_FILE_MISSING)
                        self.log.warning("[%s][%s] Missing file for native space vertex data of measure '%s': '%s'." % (subject_id, hemi, measure, morph_data_file))


                    if len(morphometry_data) != self.data[hemi]['mesh_vertex_count_white'][subject_index]:
                        self.log.warning("[%s][%s] Mismatch between length of vertex data for native space measure '%s' and number of vertices of surface white: %d != %d." % (subject_id, hemi, measure, len(morphometry_data), self.data[hemi]['mesh_vertex_count_white'][subject_index]))
                        self._append_issue(subject_id, issue_tag, morph_data_file, SEVERITY_DATA_MISMATCH)
            self.log.info("Checked native space data for measure '%s' for consistency." % (measure))
            self.report_native_measures_checked.append(measure)


    def _check_standard_space_data(self, std_measures_list):
        self._prepare_standard_space_checks()
        for fwhm in self.fwhm_list:
            for measure in std_measures_list:
                self.log.info("Verifying standard space data for measure '%s' at fwhm '%s' (average subject='%s')." % (measure, fwhm, self.average_subject))
                for hemi in self.hemis:
                    measure_key = "std_space_morphometry_vertex_data_count_%s_%s" % (measure, fwhm)
                    issue_tag = "MISMATCH_MORPH_STD_SURFACE__%s_%s_%s" % (measure, hemi, fwhm)
                    self.data[hemi][measure_key] = np.zeros((len(self.subjects_list), ))
                    for subject_index, subject_id in enumerate(self.subjects_list):
                        morph_data_file_std = fsd.get_standard_space_morphometry_file_path(self.subjects_dir, subject_id, hemi, measure, fwhm=fwhm, average_subject=self.average_subject)
                        try:
                            morphometry_data, meta_data = fsd.subject_data_standard(subject_id, self.subjects_dir, measure, hemi, fwhm, average_subject=self.average_subject)
                            self.data[hemi][measure_key][subject_index] = len(morphometry_data)
                            if self.check_file_modification_times:
                                if self.files[hemi]['surf_pial'] is not None:
                                    ts_morph_file = os.path.getmtime(morph_data_file_std)
                                    ts_surf_file = os.path.getmtime(self.files[hemi]['surf_pial'])
                                    if ts_morph_file + self.time_buffer < ts_surf_file:
                                        self.log.warning("[%s][%s] Standard space morphometry file for measure '%s' fwhm '%s' was last changed earlier than surface file: %s is before %s (%s)." % (subject_id, hemi, measure, fwhm, self._pts(ts_morph_file), self._pts(ts_surf_file), self._ptd(ts_morph_file - ts_surf_file)))
                                        issue_tag_file_time = "TIME_STD_MORPH_FILE__%s_%s_%s" % (measure, hemi, fwhm)
                                        self._append_issue(subject_id, issue_tag_file_time, morph_data_file, SEVERITY_TIMESTAMP)


                        except (OSError, IOError):
                            morphometry_data = np.array([])
                            self.data[hemi][measure_key][subject_index] = len(morphometry_data) # = 0
                            issue_tag_no_file = "NO_STD_MORPH_FILE__%s_%s_%s" % (measure, hemi, fwhm)
                            self._append_issue(subject_id, issue_tag_no_file, morph_data_file_std, SEVERITY_REQUESTED_FILE_MISSING)
                            self.log.warning("[%s][%s] Missing file for standard space vertex data of measure '%s' at fwhm '%s': '%s'." % (subject_id, hemi, measure, fwhm, morph_data_file_std))


                        if len(morphometry_data) != self.average_subject_mesh_vertex_count:
                            self.log.warning("[%s][%s] Mismatch between length of vertex data for standard space measure '%s' and number of vertices of average subject '%s' surface: %d != %d." % (subject_id, hemi, measure, self.average_subject, len(morphometry_data), self.average_subject_mesh_vertex_count))
                            self._append_issue(subject_id, issue_tag, morph_data_file_std, SEVERITY_DATA_MISMATCH)
                self.log.info("Checked standard space data for measure '%s' for consistency." % (measure))
                self.report_standard_measures_checked.append(measure)



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
        expl['NO_NAT_MORPH_FILE'] = "The native space morphology file cannot be read."
        expl['NO_STD_MORPH_FILE'] = "The standard space morphology file cannot be read."
        expl['TIME_NAT_MORPH_FILE'] = "The native space morphology file was created/modified before the respective surface file. Note: This can report false positives if the files were changed afterwards, depending on your filesystem or from copying the files."
        expl['TIME_STD_MORPH_FILE'] = "The standard space morphology file was created/modified before the respective surface file. Note: This can report false positives if the files were changed afterwards, depending on your filesystem or from copying the files."
        expl['MISMATCH_MORPH_NAT_SURFACE'] = "The value count in the native space morphology file does not match the number of vertices in the native surface file."
        expl['MISMATCH_MORPH_STD_SURFACE'] = "The value count in the standard space morphology file does not match the number of vertices in the surface file of the template subject."
        expl['MISMATCH_VERTS_SURFACES'] = "The vertex count differs for the native surface pair."
        expl['NO_SURFACE_FILE'] = 'The surface file does not exist or cannot be accessed.'
        expl['NO_SUBJECT_DIR'] = 'The subject directory for the subject cannot be read.'
        expl['BAD_SURFACE_FILE'] = 'The surface file exists but is broken, loading it failed.'
        return expl


    def save_html_report(self, filename):
        report = self._report_html()
        with open(filename, "w") as text_file:
            text_file.write(report)
        self.log.info("HTML report written to file '%s'." % (filename))


    def _get_css_style(self):
        return """<style>
        body {
          width: 95%;
          font-family: Garamond, Georgia, serif;
        }

        table {
          border-collapse: collapse;
          border: 1px solid silver;
          width: 100%;
          font-family: monospace;
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
        }

        td.issue_severity_1 {
          background-color: #99000022
        }

        td.issue_severity_2 {
          background-color: #99000044
        }

        td.issue_severity_3 {
          background-color: #99000066
        }

        td.issue_severity_4 {
          background-color: #990000AA
        }

        td.issue_severity_5 {
          background-color: #990000FF;
          font-weight: bold;
        }

        td.subject_max_severity_0 {
          background-color: #55995555
        }

        td.subject_max_severity_1 {
          background-color: #99000022
        }

        td.subject_max_severity_2 {
          background-color: #99000044
        }

        td.subject_max_severity_3 {
          background-color: #99000066
        }

        td.subject_max_severity_4 {
          background-color: #990000AA
        }

        td.subject_max_severity_5 {
          background-color: #990000FF;
          font-weight: bold;
        }

        td.count_no_issue {
          background-color: #55995555
        }

        td.count_has_issue {
          font-weight: bold;
        }

        tr:nth-child(even){
          background-color: #D5D5D5
        }

        tr:hover {
          background: silver;
          cursor: pointer;
        }

        th {
          background-color: #444444;
          color: white;
        }
        </style>"""


    def _report_html(self):
        all_issue_types = []
        for subject_index, subject_id in enumerate(self.subjects_list):
            all_issue_types.extend(self.subject_issues[subject_id])
        unique_issues = list(set(all_issue_types))

        header = "<html>\n<head>\n%s<title>Brainload QC report</title></head>\n<body>\n"  % (self._get_css_style())
        prefix = "<h1>Braindata QC Report</h1><h2>Report generated by <i>brain_qa</i>, which is part of <a href='https://github.com/dfsp-spirit/brainload' target='_blank'>brainload</a>.</h2>"
        prefix += "<h3>Usage hints</h3><ul><li>You can hover the mouse pointer over an issue category (a cell of the table header) to see a description.</li>\n"
        prefix += "<li>The table only lists issue categories which have been found for at least one subject in your data.</li>\n"
        prefix += "<li>Issues in the table are colored by severity (darker red is worse). Check the ones with highest severity first for each subject, as the others may only be aftereffects.</li>"
        prefix += "<li>You can hover the mouse pointer over an issue (a red cell in the table) to see the path of the related file or directory.</li>\n"

        prefix += "</ul>\n"

        checked_data_info = "<h3>Checked data</h3>\n"
        now = datetime.datetime.now()
        report_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        checked_data_info += "<ul>"
        checked_data_info += "<li>Checked subjects directory: '%s'</li>\n" % (os.path.abspath(self.subjects_dir))
        checked_data_info += "<li>Report generated at %s." % (report_time)
        checked_data_info += "<li>%d Native space morphometry measures checked: %s</li>\n" % (len(self.report_native_measures_checked), " ".join(self.report_native_measures_checked))
        checked_data_info += "<li>%d Standard space morphometry measures checked: %s</li>\n" % (len(self.report_standard_measures_checked), " ".join(self.report_standard_measures_checked))
        checked_data_info += "</ul>\n"
        prefix += checked_data_info
        prefix += "<h2>Results</h2>\n"

        table_end = "</table>\n"
        footer = "</body>\n</html>"

        table_header = "<table class='issues_table'>\n<tr><th title='The subject identifier'>subject_id</th><th title='Maximal severity of issues for this subject, from 0 to 5. Higher is worse.'>severity</th><th title='Number of issues detected for this subject'>num_issues</th>"
        for issue in unique_issues:
            table_header += "<th title='%s'>%s</th>" % (self.get_issue_tag_explanation(issue), issue)
        table_header += "</tr>\n"

        table_body = ""

        num_subjects_with_issues = 0
        for subject_index, subject_id in enumerate(self.subjects_list):
            if self.subject_issues[subject_id]:
                num_subjects_with_issues += 1

        summary = "<p>Checked %d subjects for issues, found %d with issues, %d ok.</p>\n" % (len(self.subjects_list), num_subjects_with_issues, len(self.subjects_list)- num_subjects_with_issues)
        prefix += summary

        for subject_index, subject_id in enumerate(self.subjects_list):
            class_issue_or_not = 'count_no_issue'
            max_severity = 0
            tag_max_severity = 'subject_max_severity_0'
            if self.subject_issues[subject_id]:
                max_severity = max(self.subject_issues_severity[subject_id])
                tag_max_severity = 'subject_max_severity_%s' % (str(max_severity))
                class_issue_or_not = 'count_has_issue'
            table_row = "<tr><td class='subject_id'>%s</td><td class='subject_severity %s'>%d</td><td class='issue_count_subject %s %s'>%d</td>" % (subject_id, tag_max_severity, max_severity, class_issue_or_not, tag_max_severity, len(self.subject_issues[subject_id]))
            for issue in unique_issues:
                if issue in self.subject_issues[subject_id]:
                    issue_index = self.subject_issues[subject_id].index(issue)
                    related_file = self.subject_issues_assoc_files[subject_id][issue_index]
                    severity = self.subject_issues_severity[subject_id][issue_index]
                    severity_tag = "issue_severity_%s" % (str(severity))
                    table_row += "<td class='check_issue %s' title='%s'>%s</td>\n" % (severity_tag, related_file, issue)
                else:
                    table_row += "<td class='check_ok'>ok</td>\n"
            table_row += "</tr>\n"
            table_body += table_row

        table = table_header + table_body + table_end
        suffix = summary
        html = header + prefix + table + suffix + footer
        return html
