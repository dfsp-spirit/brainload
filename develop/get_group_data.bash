#!/bin/bash
## get_group_data.bash -- Generates the subjects subject2, subject3, ..., subject5 by copying subject1. Also writes a subjects.txt file for these fake subjects.
##
## The data is part of the FreeSurfer software (see https://surfer.nmr.mgh.harvard.edu/) and falls under FreeSurfer's license.
## See https://github.com/freesurfer/freesurfer/blob/dev/LICENSE or your local copy at $FREESURFER_HOME/LICENSE if you have FreeSurfer installed.
##


if [ -z "${COGLOAD_TEST_DATA_DIR}" ]; then
    # Check whether we are in correct dir
    if [ ! -d "tests/test_data" ]; then
        echo "ERROR: Run this script from the repo root or set the environment variable COGLOAD_TEST_DATA_DIR."
        exit 1
    else
        COGLOAD_TEST_DATA_DIR="tests/test_data"        # This is the level equivalent to SUBJECTS_DIR from FreeSurfer
    fi
else
    echo "INFO: Environment variable COGLOAD_TEST_DATA_DIR is set, using test data dir '${COGLOAD_TEST_DATA_DIR}'."
fi

if [ ! -d "${COGLOAD_TEST_DATA_DIR}" ]; then
    echo "ERROR: The test data directory '${COGLOAD_TEST_DATA_DIR}' does not exist. Please fix the environment variable COGLOAD_TEST_DATA_DIR."
    exit 1
fi


if [ -d "${COGLOAD_TEST_DATA_DIR}/subject1" ]; then

    cd "${COGLOAD_TEST_DATA_DIR}" || { echo "ERROR: Could not change into directory '${COGLOAD_TEST_DATA_DIR}'." ; exit 1; }

    SUBJECTS_FILE='subjects.txt'
    echo "Creating subjects file '${SUBJECTS_FILE}' in directory '${COGLOAD_TEST_DATA_DIR}' and adding subject1 ..."
    echo "subject1" > "${SUBJECTS_FILE}" || { echo "ERROR: Could not write to subjects file '${SUBJECTS_FILE}'. Check write permissions for file and dir '${COGLOAD_TEST_DATA_DIR}'." ; exit 1; }

    echo "Creating new test subjects in directory '${COGLOAD_TEST_DATA_DIR}' ..."
    for SUBJECT_NUMBER in {2..5}; do
        NEW_SUBJECT_ID="subject${SUBJECT_NUMBER}"
        cp -r subject1 "${NEW_SUBJECT_ID}" && echo "${NEW_SUBJECT_ID}" >> "${SUBJECTS_FILE}" && echo " * ${NEW_SUBJECT_ID}"
    done
else
    echo "ERROR: Data for subject1 not found at '${COGLOAD_TEST_DATA_DIR}/subject1'. Exiting."
    exit 1
fi

echo "Done. Please check the data in '${COGLOAD_TEST_DATA_DIR}'."
