#!/bin/sh
## This shell script runs a demo client which uses cogload without installation, directly from the source directory.
## It is for internal development only and will not work on other machines as it loads data that only exists on my machine. I guess you could supply your own MRI/FreeSurfer data and use it, but it does nothing to fancy.
STUDY_DATA_DIR="${HOME}/data/tim_only"
COGLOAD_DIR="${HOME}/develop/cogload"
COGLOAD_SOURCE_DIR="${COGLOAD_DIR}/src"

export SUBJECTS_DIR="${STUDY_DATA_DIR}"
cd "${STUDY_DATA_DIR}" && PYTHONPATH=${PYTHONPATH}:${COGLOAD_SOURCE_DIR} python "${COGLOAD_DIR}/develop/example_client/example.py"
