import os
import numpy as np
import brainload as bl
import brainload.freesurferdata as fsd
import brainload.braindescriptors as bd
import brainload.nitools
import logging


def run_desc():
    subjects_list = ['subject1', 'subject2']
    subjects_dir = os.path.join("tests", "test_data")

    #subjects_dir = os.path.join(os.getenv("HOME"), "data", "abide", "structural")
    #subjects_dir = os.path.join("/Volumes", "shared-1", "projects", "abide", "structural")
    #subjects_file = os.path.join(subjects_dir, 'subjects.txt')
    #subjects_list = brainload.nitools.read_subjects_file(subjects_file)

    #logging.basicConfig(level=logging.DEBUG)

    bdi = bd.BrainDescriptors(subjects_dir, subjects_list)
    bdi.add_parcellation_stats(['aparc', 'aparc.a2009s'])
    #bdi.add_segmentation_stats(['aseg', 'wmparc'])
    #bdi.add_custom_measure_stats(['aparc'], ['area'])
    #bdi.report_descriptors()
    bdi.save("braindescriptors.csv", subjects_file="subjects.txt")

    import collections
    print("duplicates:")
    dups = [item for item, count in collections.Counter(bdi.descriptor_names).items() if count > 1]
    print(" ".join(dups))



if __name__ == "__main__":
  run_desc()
