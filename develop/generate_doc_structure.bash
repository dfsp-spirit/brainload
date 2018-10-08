#!/bin/bash
## This script was used to create the initial structure of the sphinx documentation.
## !!!!! This script does NOT rebuild the documentation from the source code. !!!!!
## It erases the settings for generating the documentation from the source code and sets new ones. Read the dev README file if you want to rebuild the docs only.
## There should be no need to run this script if the directory doc/ already exists in the repo. Just edit the files in there and re-run `make` in the doc/ directory.
##
## This script requires sphinx-apidoc, which is provided by sphinx: `pip install sphinx` or see http://www.sphinx-doc.org

PROJECT="cogload"
PROJECT_SOURCE_DIR="src/${PROJECT}"
DOC_DIR="doc"
AUTHOR="Tim Schäfer"

## Check whether we are in the correct path, this must be run from the root of the repo
if [ ! -d "${PROJECT_SOURCE_DIR}" ]; then
    echo "This script must be executed from the root of the repository! Exiting."
    exit 1
fi

if [ ! -d "${DOC_DIR}" ]; then
    echo "Directory '${DOC_DIR}' does not exist in current path. It must exist (and be empty). Exiting."
    exit 1
fi

if [ "$(ls -A ${DOC_DIR})" ]; then
    echo "It looks like the documentation dir '${DOC_DIR}' already has some files in it."
    echo "If all you wanted to do is rebuild the documentation after you changed the code, do NOT run this script! Read the development readme file instead."
    echo "Only if you really know what you are doing: delete all files in doc/ and then re-run this script."
    exit 1
fi

sphinx-apidoc --full --module-first -H "${PROJECT}" -A "${AUTHOR}" -o ${DOC_DIR} src/${PROJECT}
