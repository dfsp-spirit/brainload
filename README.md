# cogload
A python module designed to reduce your cognitive load while accessing FreeSurfer brain data files. A thin wrapper around nibabel and mayavi.

## Development stage

This is pre-alpha and not ready for usage yet. Come back another day.

## Development

It is recommended to use a virtual environment for hacking on cogload.

    pip install --user virtualenv      # unless you already have it
    cd develop/cogload/                # or wherever you cloned the repo
    python -m virtualenv env/          # creates a virtual python environment under env/


Once you have created the virtual environment, all you have to do is use it:

    source env/bin/activate            # to change into it

    some_command...

    deactivate                          # to leave it


To install cogload in development mode (you should be in the virtual environment, of course):

    cd develop/cogload/
    pip install --editable .
