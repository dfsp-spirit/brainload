#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

setup(
    name='cogload',
    version='0.1.0',
    description='Load FreeSurfer brain imaging data with minimal cognitive load',
    long_description='Provides a high-level interface for loading FreeSurfer brain imaging data by wrapping around nibabel. Optionally uses MayaVi to plot surface data onto the respective brain mesh in 3D.',
    keywords='neuroimaging freesurfer nibabel load mgh curv',
    author='Tim Schäfer',
    url='https://github.com/dfsp-spirit/cogload',
    packages=find_packages(where='src'),
    classifiers = ['Development Status :: 2 - Pre-Alpha',     # See https://pypi.org/pypi?%3Aaction=list_classifiers for full classifier list
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 2'],
    license='MIT',
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'pytest-cov'],
    install_requires=['numpy', 'nibabel'],
    extras_require={'plot_brain': ['mayavi']},               # To use this, you have to `pip install cogload[plot_brain]`
    package_dir = {'': 'src'},                               # The root directory that contains the source for the modules (relative to setup.py) is ./src/
    zip_safe=False
)
