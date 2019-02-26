# brainload
Python module to reduce your brain load while accessing FreeSurfer brain surface meshes and morphometry data files for single subjects and groups.

This is my personal collection of Python/numpy functions for (structural) neuroimaging. It's free software under the MIT license.

## About

`brainload` provides a simple high-level interface to access [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/) neuroimaging data in python. It is intended for developers and scientists who need to access neuroimaging data for their research.

`brainload` makes use of the standard output file name patterns of the FreeSurfer pre-processing pipeline (i.e., `recon-all`) to find the respective files and then uses [nibabel](http://nipy.org/nibabel/) to open them. It provides an easy-to-use functional interface to neuroimaging data on multiple levels, from accessing individual files to loading group data for a directory filled with hundreds of subjects.



## Development stage

This has been in alpha stage and ready for usage since December 10, 2018. The API is not 100% stable yet, be prepared for minor changes in the future.


## Features

* Load FreeSurfer meshes (like `lh.white`, `lh.pial`) and/or morphometry data (like `lh.area` or `lh.thickness`) for a single subject or groups of subjects
* Determine which subjects to load for a study directory full of subjects (FreeSurfer SUBJECTS_DIR) based on one of the following:
  - A **subjects file** or demographics file in text format as commonly used in neuroimaging (can be just a list of subjects directories or a CSV file that includes other data for each subject in additional rows)
  - A **custom list of subject names** (you can get this by whatever means necessary, e.g., from a database query, the internet or whatever)
  - **Auto-detect** all valid FreeSurfer subject directories within the SUBJECTS_DIR
* Uses knowledge on the standard FreeSurfer directory structure, file names and file extensions so you do not have to specify the full path to files (e.g., it knows that the pial surface for a subject is stored in `subject/surf/lh.pial` and `subject/surf/rh.pial` for the two hemispheres). If you prefer to specify everything manually, that is also possible of course.
* Support for loading annotations (brain atlases, e.g., `lh.aparc.annot`) and labels (surface patches, e.g., `lh.cortex.label`).
* Support for loading statistical information from FreeSurfer stats files (e.g., total brain volume, average thickness, total surface area for subjects from `aseg.stats` or atlas-specific information from files like `lh.aparc.stats`).
* Can export brain meshes to standard 3D modeling software formats (e.g. OBJ and PLY formats, for advanced visualization). With matplotlib installed, you can even export vertex-colored meshes using any matplotlib colormap.
* Good documentation
* A suite of unit tests with high test coverage
* Very permissive license


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


## Full Documentation (API, examples)

Brainload is fully documented. The full API documentation and some workflow examples can be found here:
- [Brainload documentation for the latest release](http://dfsp-spirit.github.io/brainload)
- [Brainload documentation: Older versions](http://dfsp-spirit.github.io/brainload/versions.html)


## Installation

#### Recommended: via pip

```console
pip install --user brainload
```

You can also install into a virtual environment (python2: virtualenv, python3: venv) of course, omit the `--user` part in that case.

[![PyPI version](https://badge.fury.io/py/brainload.svg)](https://badge.fury.io/py/brainload)

Both source and wheel packages are also available here in the [brainload releases](https://github.com/dfsp-spirit/brainload/releases) section at GitHub, but you should not need them.

#### via Anaconda

I started building conda packages for different operating systems, check https://anaconda.org/dfspspirit/brainload to see whether one is available for yours. In case it is:

```console
conda install -c dfspspirit brainload
```

[![Anaconda-Server Badge](https://anaconda.org/dfspspirit/brainload/badges/version.svg)](https://anaconda.org/dfspspirit/brainload)


If it is not, you can use the recipe in this repo to build it yourself, see [README_DEVELOPMENT](README_DEVELOPMENT.md).

#### Supported Python versions

Brainload works on both Python 2 and Python 3. You can see all supported versions in the `.travis.yml` file in this directory.


## What about visualization?

Brainload loads (and in very few cases writes) files in standard neuroimaging formats (nifti, mgh/mgz) and does not introduce any new file formats, so you can use any standard viewer for the data. All the standard neuroimaging software packages come with a viewer, and most people will have several viewers installed anyways I guess. Examples include `fsleyes` from FSL, `freeview` that is included with FreeSurfer, or the stand-alone viewer `3DSlicer`.

To visualize directly in Python, e.g. for debugging while developing a script, or in an interactive jupyter notebook, there are different options, usually based on matplotlib (2D) and mayavi (3D):
- If you are interesting in a full solution that can provide output in publication quality, I suggest you have a look at [PySurfer](https://pysurfer.github.io/). The project is mature and works great.
- While `brainload` itself does not care about visualization, I am working on the [brainview](https://github.com/dfsp-spirit/brainview) package that is designed to visualize morphometry data in 3D on brain surfaces. The goal is to provide an easy-to-use interface to quickly visualize data loaded with `brainload` or any other tool. The module will only provide basic visualization functions intended for live inspection of your data though.


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

Brainload is free software, released under the [MIT license](https://opensource.org/licenses/MIT). See the LICENSE file for the full license text.
