lyman: neuroimaging analysis in Python
======================================

[![PyPI Version](https://img.shields.io/pypi/v/lyman.svg)](https://pypi.org/project/lyman/)
[![License](https://img.shields.io/pypi/l/lyman.svg)](https://github.com/mwaskom/lyman/blob/master/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.824307.svg)](https://doi.org/10.5281/zenodo.824307)
[![Build Status](https://travis-ci.org/mwaskom/lyman.svg?branch=master)](https://travis-ci.org/mwaskom/lyman)
[![Code Coverage](https://codecov.io/gh/mwaskom/lyman/branch/master/graph/badge.svg)](https://codecov.io/gh/mwaskom/lyman)

Lyman is a platform for analyzing neuroimaging (primarily MRI) data using Python. It comprises an interface to [Nipype](http://nipype.readthedocs.io/) data processing workflows and a library of classes and functions for signal processing, model fitting, visualization, and other tasks.

Documentation
-------------

Online documentation can be found
[here](http://www.cns.nyu.edu/~mwaskom/software/lyman).

Dependencies
------------

Lyman supports Python 3.7+ and does not support Python 2.

Lyman preprocessing requires [FSL](http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/) and [Freesurfer](https://surfer.nmr.mgh.harvard.edu/). Information about Python library dependency information is available in the [online documentation](http://www.cns.nyu.edu/~mwaskom/software/lyman/installing.html#dependencies).

Installation
------------

Stable versions can be installed from PyPI:

    pip install lyman

It is also possible to install development code from Github:

    pip install git+https://github.com/mwaskom/lyman.git

License
-------

Lyman is freely available under the BSD (3-clause) license.

Support
-------

Lyman is publicly released with some hope that it might be useful but no promise of support. With that said, reproducible bugs may be reported to the [Github issue tracker](https://github.com/mwaskom/lyman/issues).
