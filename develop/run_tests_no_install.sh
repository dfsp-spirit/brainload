#!/bin/sh
## run-tests.sh -- Runs the unit tests for cogload without installation, i.e., from the source in the local dir.
## You should execute this script from the repo root dir, i.e., as `./develop/run_tests.sh`.

# Check whether we are in correct dir
if [ ! -d "src/cogload" ]; then
    echo "ERROR: Run this script from the repo root."
    exit 1
fi

PYTHONPATH=src pytest --pyargs tests/
