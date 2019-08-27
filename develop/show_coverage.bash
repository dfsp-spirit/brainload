#!/bin/bash

echo "Make sure to run 'python setup.py test' to generate coverage data before running this script."
coverage html && firefox htmlcov/index.html

