.. _installing:

Installation and dependencies
=============================

Installing
----------

Stable versions can be installed from PyPI:

    pip install lyman

It is also possible to install development code from Github:

    pip install git+https://github.com/mwaskom/lyman.git

Dependencies
------------

Neuroimaging packages
~~~~~~~~~~~~~~~~~~~~~

Lyman's preprocessing workflows take advantage of image registration algorithms
from established neuroimaging analysis libraries, so these packages must be
installed to run the lyman workflows.

- Freesurfer 6+

- FSL 5.0.9+

Python libraries 
~~~~~~~~~~~~~~~~

Lyman also depends on a number of packages from the scientific Python
ecosystem. They will be installed automatically if they are not present when you install lyman.

- numpy

- scipy

- pandas

- matplotlib

- nipype

- nibabel

- traits

- pyyaml
