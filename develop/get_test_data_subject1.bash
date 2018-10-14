#!/bin/bash
## get_test_data_subject1.bash -- Retrieves the subject1 minimal test data from the internet.
##
## author: Tim Schäfer
## This file is part of brainload. Copright Tim Schäfer, 2018. See the LICENSE file for the license.
##

APPTAG="[TD_SUBJ1]"

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

BRAINLOAD_TEST_DATA_DIR_SUBJECT1="${BRAINLOAD_TEST_DATA_DIR}/subject1"

if [ -d "${BRAINLOAD_TEST_DATA_DIR_SUBJECT1}" ]; then
    echo "${APPTAG} INFO: Test data for subject1 already exists at '${BRAINLOAD_TEST_DATA_DIR_SUBJECT1}'. Refreshing."
fsi

ARCHIVE_NAME="subject1_min.zip"
REMOTE_ZIP_URL="https://github.com/dfsp-spirit/neuroimaging_testdata/raw/master/freesurfer/ts/${ARCHIVE_NAME}"


echo "${APPTAG} Trying to download data from remote location '${REMOTE_ZIP_URL}'..."

if [ -f "${BRAINLOAD_TEST_DATA_DIR}/${ARCHIVE_NAME}" ]; then
    rm "${BRAINLOAD_TEST_DATA_DIR}/${ARCHIVE_NAME}"
fi

cd "${BRAINLOAD_TEST_DATA_DIR}" && wget "${REMOTE_ZIP_URL}" && unzip -f -o ${ARCHIVE_NAME} && rm ${ARCHIVE_NAME} && echo "${APPTAG} OK."
