#!/bin/sh
## This shell script runs a demo client which uses brainload without installation, directly from the source directory.
## It is for internal development only and will not work on other machines as it loads data that only exists on my machine. I guess you could supply your own MRI/FreeSurfer data and use it, but it does nothing to fancy.
STUDY_DATA_DIR="${HOME}/data/tim_only"
BRAINLOAD_DIR="${HOME}/develop/brainload"
BRAINLOAD_SOURCE_DIR="${BRAINLOAD_DIR}/src"

export SUBJECTS_DIR="${STUDY_DATA_DIR}"
cd "${STUDY_DATA_DIR}" && PYTHONPATH=${PYTHONPATH}:${BRAINLOAD_SOURCE_DIR} python "${BRAINLOAD_DIR}/develop/example_client/example.py"
