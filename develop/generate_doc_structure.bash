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
    echo "Directory '${DOC_DIR}' does not exist in current path. It must exist and should have the settings from your run of `sphinx-quickstart` in there already. Exiting."
    exit 1
fi


sphinx-apidoc --full -o ${DOC_DIR} src/${PROJECT}
