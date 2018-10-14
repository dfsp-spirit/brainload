#!/bin/bash
## get_test_data_group.bash -- Generates the subjects subject2, subject3, ..., subject5 by copying subject1. Also writes a subjects.txt file for these fake subjects.
## author: Tim Schäfer
## This file is part of brainload. Copright Tim Schäfer, 2018. See the LICENSE file for the license.

APPTAG="[TD_GRP]"

if [ -z "${BRAINLOAD_TEST_DATA_DIR}" ]; then
    # Check whether we are in correct dir
    if [ ! -d "tests/test_data" ]; then
        echo "${APPTAG} ERROR: Run this script from the repo root or set the environment variable BRAINLOAD_TEST_DATA_DIR."
        exit 1
    else
        BRAINLOAD_TEST_DATA_DIR="tests/test_data"        # This is the level equivalent to SUBJECTS_DIR from FreeSurfer
        echo "${APPTAG} Environment variable BRAINLOAD_TEST_DATA_DIR was not set, assuming '${BRAINLOAD_TEST_DATA_DIR}'."
    fi
else
    echo "${APPTAG} INFO: Environment variable BRAINLOAD_TEST_DATA_DIR is set, using test data dir '${BRAINLOAD_TEST_DATA_DIR}'."
fi

if [ ! -d "${BRAINLOAD_TEST_DATA_DIR}" ]; then
    echo "${APPTAG} ERROR: The test data directory '${BRAINLOAD_TEST_DATA_DIR}' does not exist. Please fix the environment variable BRAINLOAD_TEST_DATA_DIR."
    exit 1
fi


if [ -d "${BRAINLOAD_TEST_DATA_DIR}/subject1" ]; then

    cd "${BRAINLOAD_TEST_DATA_DIR}" || { echo "${APPTAG} ERROR: Could not change into directory '${BRAINLOAD_TEST_DATA_DIR}'." ; exit 1; }

    SUBJECTS_FILE='subjects.txt'
    echo "${APPTAG} Creating subjects file '${SUBJECTS_FILE}' in directory '${BRAINLOAD_TEST_DATA_DIR}' and adding subject1 ..."
    echo "subject1" > "${SUBJECTS_FILE}" || { echo "${APPTAG} ERROR: Could not write to subjects file '${SUBJECTS_FILE}'. Check write permissions for file and dir '${BRAINLOAD_TEST_DATA_DIR}'." ; exit 1; }

    echo "${APPTAG} Creating new test subjects in directory '${BRAINLOAD_TEST_DATA_DIR}' ..."
    for SUBJECT_NUMBER in {2..5}; do
        NEW_SUBJECT_ID="subject${SUBJECT_NUMBER}"
        cp -r subject1 "${NEW_SUBJECT_ID}" && echo "${NEW_SUBJECT_ID}" >> "${SUBJECTS_FILE}" && echo "${APPTAG}  * ${NEW_SUBJECT_ID}"
    done
else
    echo "${APPTAG} ERROR: Data for subject1 not found at '${BRAINLOAD_TEST_DATA_DIR}/subject1'. Exiting."
    exit 1
fi

echo "${APPTAG} Done. Please check the data in '${BRAINLOAD_TEST_DATA_DIR}'."
