#!/bin/bash
# get_test_data_fsaverage.bash

COGLOAD_BASE_DIR="."
COGLOAD_TEST_DATA_DIR_FSAVERAGE_SURF="${COGLOAD_BASE_DIR}/tests/test_data/fsaverage/surf"

# Check whether we are in correct dir
if [ ! -d "src/cogload" ]; then
    echo "ERROR: Run this script from the repo root."
    exit 1
fi

if [ -n "${FREESURFER_HOME}" ]; then
    FD="${FREESURFER_HOME}/subjects/fsaverage/surf"
    mkdir -p "${COGLOAD_TEST_DATA_DIR_FSAVERAGE_SURF}"
    echo "Copying fsaverage data from '${FD}' to '${COGLOAD_TEST_DATA_DIR_FSAVERAGE_SURF}'."
    for TFILE in lh.inflated lh.pial lh.sphere lh.white rh.inflated rh.pial rh.sphere rh.white; do
        cp "${FD}/${TFILE}" "${COGLOAD_TEST_DATA_DIR_FSAVERAGE_SURF}" && echo " * ${TFILE}"
    done
else
    echo "ERROR: Environment variable FREESURFER_HOME not set. Please install and configure FreeSurfer before running this script."
fi
