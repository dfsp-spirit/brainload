#!/bin/sh
STUDY_DATA_DIR="${HOME}/data/tim_only"
COGLOAD_DIR="${HOME}/develop/cogload"
COGLOAD_SOURCE_DIR="${COGLOAD_DIR}/src"

export SUBJECTS_DIR="${STUDY_DATA_DIR}"
cd "${STUDY_DATA_DIR}" && PYTHONPATH=${PYTHONPATH}:${COGLOAD_SOURCE_DIR} python "${COGLOAD_DIR}/example_client/example.py"
