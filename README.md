# cogload
A python module designed to reduce your cognitive load while accessing FreeSurfer brain data files ;). A thin wrapper around nibabel and mayavi.

## About

`CogLoad` provides a simple high-level interface to access [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/) neuroimaging data in python. It is intended for developers and scientists who need to access neuroimaging data for their research.

`CogLoad` makes use of the standard output file name patterns of the FreeSurfer pre-processing pipeline (i.e., `recon-all`) to find the files and then uses [nibabel](http://nipy.org/nibabel/) to open them in the background. Optionally, it can use [MayaVi](http://code.enthought.com/pages/mayavi-project.html) to create simple 3D plots of morphometry data on meshes representing brain surfaces.


## Development stage

This is pre-alpha and not ready for usage yet. Come back another day.


## Interface (WIP)

Here is an example usage that loads surface data for a subject:

```python
from cogload.freesurferdata import parse_subject
subject_id = 'bert'
vert_coords, faces, per_vertex_data, meta_data = parse_subject(subject_id, surf='pial', measure='area')
```
Now you have the brain surface mesh (defined by `vert_coords` and `faces`) and the morphology data from the curv file in `per_vertex_data`. The `meta_data` holds information like the subject id and the full paths of the files that were used to retrieve the data. The parse_subject function uses the environment variable `SUBJECTS_DIR` to determine where your data is. You can override that (and change many other things via optional named arguments), of course.

Note that the data we retrieved in the example above is in native space. You may want to retrieve standard space data, i.e., subject data mapped to an average subject like FreeSurfer's `fsaverage` subject, instead. Here is an example for that:

```python
from cogload.freesurferdata import parse_subject_standard_space_data
subject_id = 'bert'
vert_coords, faces, per_vertex_data, meta_data = parse_subject_standard_space_data(subject_id, surf='white', display_surf='inflated', measure='area')
```

This time, the mesh you get is the inflated surface of the `fsaverage` subject (since that is the default for the named parameter `average_subject`, which we omitted in the example above). The `per_vertex_data` represents the area data for the white matter surface of your subject, mapped to the vertices of the average subject and ready for group comparison.

You can now use the data for statistical analysis in python, e.g., using Pandas, Statsmodels, or whatever you prefer. You could also load the mesh into PyMesh and mess with it.

## API Documentation

It's a bit too early for that.



## Development

It is recommended to use a virtual environment for hacking on `cogload`.

```console
pip install --user virtualenv      # unless you already have it
cd develop/cogload/                # or wherever you cloned the repo
python -m virtualenv env/          # creates a virtual python environment in the new directory env/
```


Once you have created the virtual environment, all you have to do is use it:

```console
source env/bin/activate            # to change into it

some_command...

deactivate                         # to leave it
```


To install `cogload` in development mode (you should be in the virtual environment, of course):

```console
cd develop/cogload/
pip install --editable .           # also installs the dependencies
```

You can now use `cogload` by typing `import cogload` in your application or an interactive python session.


See the file `DEVELOPMENT.md` for details.


## Obtaining suitable pre-processed sMRI input data for cogload

The cogload module is designed to work on structural Magnetic Resonance Imaging (sMRI) data that has been pre-processed with the popular [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/) software suite.

If you do not have your MRI data / FreeSurfer output at hand but still want to try cogload, you could use the `bert` example subject that comes with FreeSurfer.

## Tests

To run the unit tests, you need `pytest` and `pytest-cov`, both of which can be installed via `pip`. In the virtual environment and the top-level cogload directory, just run:

```console
pytest
```

To run the tests with code coverage:

```console
pytest --cov=src/
```


## Alternatives and similar tools (in python)

Alternatives to `CogLoad`:

- If you want a full brain visualization package for python, you may want to have a look at [PySurfer](https://pysurfer.github.io/) instead.
- You could also use the `freesurfer.io` and `freesurfer.mghformat` modules from [nibabel](http://nipy.org/nibabel/) directly and open the FreeSurfer files yourself. (Most likely you would end up with boilerplate that is pretty similar to `CogLoad`.)

Less related but still useful:

- If you want a full python interface that wraps the command line utilities of various existing neuroimaging software (FSL, FreeSurfer, ...) and allows you to create a full neuroimaging pipeline, you should definitely have a look at [nipype](http://nipy.org/packages/nipype/index.html).
- In case you do not yet know it, I highly recommend that you have a look at some of the great neuroimaging tools for python at [nipy.org](http://nipy.org/).



## License

[MIT](https://opensource.org/licenses/MIT)
