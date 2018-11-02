# brainload
A python module designed to reduce your brain load while accessing FreeSurfer brain surface meshes and morphometry data files.

## About

`brainload` provides a simple high-level interface to access [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/) neuroimaging data in python. It is intended for developers and scientists who need to access neuroimaging data for their research.

`brainload` makes use of the standard output file name patterns of the FreeSurfer pre-processing pipeline (i.e., `recon-all`) to find the respective files and then uses [nibabel](http://nipy.org/nibabel/) to open them. It provides an easy-to-use functional interface to neuroimaging data on multiple levels, from accessing individual files to loading group data for a directory filled with hundreds of subjects.

[![Build Status](https://travis-ci.org/dfsp-spirit/brainload.svg?branch=master)](https://travis-ci.org/dfsp-spirit/brainload)

## Development stage

This is now (October 24, 2018) in alpha stage and ready for usage. The API is not 100% stable yet though, so be prepared for changes in the future.


## Interface Examples

Here are some example usages that load brain surface data. See the API documentation for more examples.

### Load the brain mesh and morphometry data for a single subject in subject space

```python
import brainload as bl
subject_id = 'bert'
vert_coords, faces, per_vertex_data, meta_data = bl.subject(subject_id, surf='pial', measure='area')
```
Now you have the brain surface mesh (defined by `vert_coords` and `faces`) and the morphometry data from the curv file in `per_vertex_data`. The `meta_data` holds information like the subject id and the full paths of the files that were used to retrieve the data. The `subject` function uses the environment variable `SUBJECTS_DIR` to determine where your data is. You can override that (and change many other things via optional named arguments), of course.

Note that the data we retrieved in the example above is in native space. You may want to retrieve standard space data, i.e., subject data mapped to an average subject like FreeSurfer's `fsaverage` subject, instead. Here is an example for that:

### Load the fsaverage brain mesh and morphometry data for a single subject that has been mapped to fsaverage

```python
import brainload as bl
subject_id = 'bert'
vert_coords, faces, per_vertex_data, meta_data = bl.subject_avg(subject_id, surf='white', display_surf='inflated', measure='area')
```

This time, the mesh you get is the inflated surface of the `fsaverage` subject (since that is the default for the named parameter `average_subject`, which we omitted in the example above). The `per_vertex_data` represents the area data for the white matter surface of your subject, mapped to the vertices of the average subject and ready for group comparison.

### Load fsaverage data for all subjects in your SUBJECTS_DIR

```python
import brainload as bl
group_data, group_data_subjects, group_meta_data, run_meta_data = bl.group('area', fwhm='15', surf='pial', hemi='lh')
print group_data.shape          # will print '(260, 163842)' if your SUBJECTS_DIR contains 260 subjects. Note that 163842 is the number of vertices of the left hemisphere of the 'fsaverage' subject in FreeSurfer.
```

This will load the standard space area data for all subjects in the SUBJECTS_DIR. Of course, you could specify more settings instead of using the defaults, e.g., if you used an average subject that is not fsaverage. And you can always see which files and settings were used under the hood:

```python
# continued from last code sample
print group_meta_data['subject1']['lh.morphometry_file']             # will print SUBJECTS_DIR/subject1/surf/lh.area.pial.fwhm15.fsaverage.mgh
print group_meta_data['subject1']['hemi']                           # will print 'lh'
```

Whatever function you used, you can now use the data for statistical analysis in python, e.g., using Pandas, Statsmodels, or whatever you prefer. You could also load the mesh into PyMesh and mess with it.

## Full API Documentation

The brainload API documentation is now available:
- [Brainload API documentation for the latest release](http://dfsp-spirit.github.io/brainload)
- [Brainload API documentation: Older versions](http://dfsp-spirit.github.io/brainload/versions.html)


## Installation

#### Recommended: via pip

```console
pip install --user brainload
```

You can also install into a virtual environment (python2: virtualenv, python3: venv) of course, omit the `--user` part in that case.

Both source and wheel packages are also available here in the [brainload releases](https://github.com/dfsp-spirit/brainload/releases) section at GitHub, but you should not need them.

#### via Anaconda

I started building conda packages for different operating systems, check https://anaconda.org/dfspspirit/brainload to see whether one is available for yours. In case it is:

```console
conda install -c dfspspirit brainload
```

[![Anaconda-Server Badge](https://anaconda.org/dfspspirit/brainload/badges/version.svg)](https://anaconda.org/dfspspirit/brainload) [![Anaconda-Server Badge](https://anaconda.org/dfspspirit/brainload/badges/platforms.svg)](https://anaconda.org/dfspspirit/brainload) [![Anaconda-Server Badge](https://anaconda.org/dfspspirit/brainload/badges/license.svg)](https://anaconda.org/dfspspirit/brainload)

If it is not, you can use the recipe in this repo to build it yourself, see [README_DEVELOPMENT](README_DEVELOPMENT.md). 

#### Supported Python versions

Brainload 0.2.0 is available for Python 2.7. Starting from the next release, Brainload will also support Python 3.


## What about visualization?

While `brainload` itself does not care about visualization, I am working on the [brainview](https://github.com/dfsp-spirit/brainview) package that is designed to visualize morphometry data in 3D on brain surfaces. The goal is to provide an easy-to-use interface to quickly visualize data loaded with `brainload` or any other tool.

The module will only provide basic visualization functions intended for live inspection of your data. If you are interesting in a full solution that can provide output in publication quality, I suggest you have a look at [PySurfer](https://pysurfer.github.io/) instead.

## Obtaining suitable pre-processed sMRI input data for brainload

The brainload module is designed to work on Magnetic Resonance Imaging (MRI) data that has been pre-processed with the popular [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/) software suite.

If you do not have your MRI data / FreeSurfer output at hand but still want to try `brainload`, you could use the `bert` example subject that comes with FreeSurfer.


## Development and tests

Tests and test data are not shipped in the releases, see the file [README_DEVELOPMENT](README_DEVELOPMENT.md) for instructions on installing the development version and running the tests.


## Alternatives and similar tools (in python)

Alternatives to `brainload`:

- If you want a full brain visualization package for Python that allows you to plot morphometry data in various ways, you may want to have a look at [PySurfer](https://pysurfer.github.io/) instead.
- You could also use the `freesurfer.io` and `freesurfer.mghformat` modules from [nibabel](http://nipy.org/nibabel/) directly and open the FreeSurfer files yourself. (Most likely you would end up with boilerplate that is pretty similar to `brainload`.)

Less related but still useful:

- If you want a full python interface that wraps the command line utilities of various existing neuroimaging software (FSL, FreeSurfer, ...) and allows you to create a full neuroimaging pipeline, you should definitely have a look at [nipype](http://nipy.org/packages/nipype/index.html).
- I highly recommend that you have a look at some of the great neuroimaging tools for Python at [nipy.org](http://nipy.org/).


## License

Brainload is free software, released under the [MIT license](https://opensource.org/licenses/MIT).
