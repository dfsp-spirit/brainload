#!/bin/sh
## run-tests.sh -- Runs the unit tests for cogload.
## You should execute this script from with its directory, i.e., as `./run_tests.sh`.
PYTHONPATH=src pytest --pyargs tests/
