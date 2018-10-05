# cogload Development information

This is intended for developers who modify `cogload`.

## Detailed development instructions

Here is a full interactive example session on how to perform the development build and use the module in dev mode, from Ubuntu 18.04 LTS:

Installation:

```console
[ts@box:~/develop/cogload] $ source env/bin/activate
(env) [ts@box:~/develop/cogload] $ pip install --editable .
Obtaining file:///home/ts/develop/cogload
Collecting numpy (from cogload==0.1.0)
  Using cached https://files.pythonhosted.org/packages/40/c5/f1ed15dd931d6667b40f1ab1c2fe1f26805fc2b6c3e25e45664f838de9d0/numpy-1.15.2-cp27-cp27mu-manylinux1_x86_64.whl
Collecting nibabel (from cogload==0.1.0)
Installing collected packages: numpy, nibabel, cogload
  Running setup.py develop for cogload
Successfully installed cogload nibabel-2.3.0 numpy-1.15.2
```

Using the module:

```console
(env) [ts@box:~/develop/cogload] $ python
Python 2.7.15rc1 (default, Apr 15 2018, 21:51:34)
[GCC 7.3.0] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>> import cogload
>>> from cogload.spatial_transform import *
>>> deg2rad(90)
1.5707963267948966
>>> exit()
(env) [ts@box:~/develop/cogload] $ deactivate
[ts@box:~/develop/cogload] $
```

## Packaging

We are following the [official Python packaging user guide](https://packaging.python.org/tutorials/packaging-projects/) here.

You can use the `setup.py` file to generate a wheel package. This should be done in the virtual environment.

```console
pip install --upgrade setuptools wheel              # just make sure we have the latest versions
python setup.py sdist bdist_wheel                   # will create the packages in the sub directory dist/
```

Then deactivate the virtual environment and upload the packages to PiPy:

```console
deactivate                                    # only if you were in the virtual environment
pip install --user --upgrade twine            # just to be sure
twine upload dist/*
```
