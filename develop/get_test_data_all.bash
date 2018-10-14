#!/bin/bash
## get_test_data_all.bash -- Wrapper that calls the other script and retrieves all test data.
## author: Tim Schäfer
## This file is part of brainload. Copright Tim Schäfer, 2018. See the LICENSE file for the license.

APPTAG="[TD_ALL]"


DEV_SCRIPTS_DIR="./develop"
TEST_DATA_SCRIPTS="get_test_data_subject1.bash get_test_data_fsaverage.bash get_test_data_group.bash"
if [ ! -d "${DEV_SCRIPTS_DIR}" ]; then
    echo "${APPTAG} Could not find directory '${DEV_SCRIPTS_DIR}' relative to current path. Call this script from the repo root."
    exit 1
else
    echo "${APPTAG} Executing scripts to retrieve test data:"
    for SCRIPT in ${TEST_DATA_SCRIPTS}; do
        CMD="${DEV_SCRIPTS_DIR}/${SCRIPT}"
        echo "${APPTAG} Running script '${CMD}'..."
        ${CMD}     # Actually run it.
    done
fi

Echo "${APPTAG} All done."
