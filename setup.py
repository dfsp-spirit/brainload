#!/usr/bin/env pytho6
# -*- coding: utf-8 -*-

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

setup(
    name='brainload',
    version='0.3.4',
    description='Load FreeSurfer brain imaging data with minimal cognitive load.',
    long_description='Python module to reduce your brain load while accessing FreeSurfer brain surface meshes and morphometry data files for single subjects and groups.',
    keywords='neuroimaging freesurfer nibabel load mgh curv MRI morphometry',
    author='Tim Schäfer',
    author_email='ts+code@rcmd.org',
    url='https://github.com/dfsp-spirit/brainload',
    packages=find_packages(where='src'),
    classifiers = ['Development Status :: 3 - Alpha',     # See https://pypi.org/pypi?%3Aaction=list_classifiers for full classifier list
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7'],
    license='MIT',
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'pytest-cov', 'pytest-console-scripts'],
    install_requires=['numpy', 'nibabel'],
    package_dir = {'': 'src'},                               # The root directory that contains the source for the modules (relative to setup.py) is ./src/,
    include_package_data=True,                               # respect MANIFEST.in at install time
    zip_safe=False,
    entry_points = {
        'console_scripts': [
            'visualize_verts = brainload.clients.visualize_verts:visualize_verts',
            'brain_mesh_info= brainload.clients.brain_mesh_info:brain_mesh_info',
            'brain_surf_info = brainload.clients.brain_surf_info:brain_surf_info',
            'brain_vol_info = brainload.clients.brain_vol_info:brain_vol_info',
            'brain_morph_info = brainload.clients.brain_morph_info:brain_morph_info',
        ]
    }
)
