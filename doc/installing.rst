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
from established neuroimaging analysis libraries. These must be installed to run
the lyman workflows, but the lyman signal processing, modeling, and
visualization library code are independent of them.

- Freesurfer 6+

- FSL 5.0.7+

Python libraries 
~~~~~~~~~~~~~~~~

Lyman also depends on a number of packages from the scientific Python
ecosystem. They will be included automatically if they are not present when you first install lyman.

- numpy

- scipy

- pandas

- matplotlib

- nipype

- nibabel

- traits

- pyyaml
