# brainlog Development information

This is intended for developers who modify `brainlog`.

## Detailed development instructions

Here is a full interactive example session on how to perform the development build and use the module in dev mode, from Ubuntu 18.04 LTS:

Installation:

```console
[ts@box:~/develop/brainlog] $ source env/bin/activate
(env) [ts@box:~/develop/brainlog] $ pip install --editable .
Obtaining file:///home/ts/develop/brainlog
Collecting numpy (from brainlog==0.1.0)
  Using cached https://files.pythonhosted.org/packages/40/c5/f1ed15dd931d6667b40f1ab1c2fe1f26805fc2b6c3e25e45664f838de9d0/numpy-1.15.2-cp27-cp27mu-manylinux1_x86_64.whl
Collecting nibabel (from brainlog==0.1.0)
Installing collected packages: numpy, nibabel, brainlog
  Running setup.py develop for brainlog
Successfully installed brainlog nibabel-2.3.0 numpy-1.15.2
```

Using the module:

```console
(env) [ts@box:~/develop/brainlog] $ python
Python 2.7.15rc1 (default, Apr 15 2018, 21:51:34)
[GCC 7.3.0] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>> import brainlog
>>> from brainlog.spatial_transform import *
>>> deg2rad(90)
1.5707963267948966
>>> exit()
(env) [ts@box:~/develop/brainlog] $ deactivate
[ts@box:~/develop/brainlog] $
```

## Building the documentation

We use sphinx to generate the documentation. In the virtual environment:

```console
pip install sphinx
cd doc/
make html
```

We will put the documentation online later (maybe on a GitHub page), but that does not make any sense yet.

Note that if you added new modules in separate directories, for the documentation to show up,
you will have to tell autodoc about the paths to the new directories by adding them to `sys.path`
at the top of the `doc/conf.py` file.

## Packaging

We are following the [official Python packaging user guide](https://packaging.python.org/tutorials/packaging-projects/) here.

You can use the `setup.py` file to generate a wheel package. This should be done in the virtual environment.

IMPORTANT: Be sure to adapt the meta data in the `setup.py` file before packaging, especially the version information.

```console
pip install --upgrade setuptools wheel              # just make sure we have the latest versions
python setup.py sdist bdist_wheel                   # will create the packages in the sub directory dist/
```

Then deactivate the virtual environment and upload the packages to PiPy:

```console
deactivate                                    # only if you were in the virtual environment
pip install --user --upgrade twine            # just to be sure
twine upload dist/*                           # will ask for your PyPI credentials
```
