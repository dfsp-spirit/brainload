from setuptools import setup, findpackages

setup(
    name='CognitiveLoad',
    version='0.1.0',
    description='Load FreeSurfer brain imaging data with minimal cognitive load',
    long_description='Provides a high-level interface for loading FreeSurfer brain imaging data by wrapping around nibabel. Optionally uses MayaVi to plot surface data onto the respective brain mesh in 3D.',
    keywords='neuroimaging freesurfer nibabel load mgh curv',
    author='Tim Schäfer',
    url='https://github.com/dfsp-spirit/cogload',
    #packages = ['cogload'],
    packages=find_packages(),
    license='MIT',
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    install_requires=['numpy', 'nibabel'],
    package_dir = {'': 'src'}             # The root directory that contains the source for the modules (relative to setup.py) is ./src/
)
