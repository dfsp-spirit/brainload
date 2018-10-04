from setuptools import setup, findpackages

setup(
    name='CognitiveLoad',
    version='0.1.0',
    description='Load FreeSurfer brain imaging data with minimal cognitive load',
    author='Tim Schäfer',
    url='https://github.com/dfsp-spirit/cogload',
    #packages = ['cogload'],
    packages=find_packages(),
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    package_dir = {'': 'src'}             # The root directory that contains the source for the modules (relative to setup.py) is ./src/
)

[aliases]
test=pytest
