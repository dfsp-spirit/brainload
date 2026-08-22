#!/bin/bash

echo "Make sure to run 'pytest' to generate coverage data before running this script."
coverage html && firefox htmlcov/index.html

