Lyman: a Python fMRI Analysis Ecosystem
=======================================

[![Build Status](https://travis-ci.org/mwaskom/lyman.svg?branch=master)](https://travis-ci.org/mwaskom/lyman) [![codecov](https://codecov.io/gh/mwaskom/lyman/branch/master/graph/badge.svg)](https://codecov.io/gh/mwaskom/lyman) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.824307.svg)](https://doi.org/10.5281/zenodo.824307)

Lyman is a high-level ecosystem for analyzing neuroimaging data using
open-source software. It aims to support an analysis workflow that is powerful,
flexible, and reproducible, while automating as much of the processing as
possible.


Documentation
-------------

Online documentation can be found
[here](http://www.cns.nyu.edu/~mwaskom/software/lyman)

Dependencies
------------

Python 2.7 or 3.6

### External

- [FSL](http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/)

- [Freesurfer](https://surfer.nmr.mgh.harvard.edu/)

- [ANTS](http://stnava.github.io/ANTs/) (optional)

### Python


- Core scientific Python environment (ipython, numpy, scipy, matplotlib)

- [pandas](https://github.com/pydata/pandas)

- [scikit-learn](https://github.com/scikit-learn/scikit-learn)

- [scikit-image](https://github.com/scikit-image/scikit-image)

- [nipype](https://github.com/nipy/nipype)

- [nibabel](https://github.com/nipy/nibabel)

- [seaborn](https://github.com/mwaskom/seaborn)

- [moss](https://github.com/mwaskom/moss)

Installation
------------

To install the released version, just do

    pip install lyman

You may instead want to use the development version from Github, by running

    pip install git+https://github.com/mwaskom/lyman.git

Basic Workflow
--------------

All stages of processing assume that your anatomical data have been
processed in Freesurfer (recon-all)

- `run_warp.py`: estimate anatomical normalization

- `anatomy_snapshots.py`: generate static images summarizing the Freesurfer reconstruction.

- `run_fmri.py`: perform subject-level functional preprocessing and analyses

- `make_masks.py`: generate ROI masks in native EPI space from a variety of sources

- `run_group.py`: perform basic whole-brain mixed-effects analyses

- `surface_snapshots.py`: plot the results of the subject- and group-level models on a surface mesh

Ziegler
-------

The processing scripts generate a variety of static images that can be used for quality control and understanding the analysis. The best way to browse these is with the [ziegler](https://github.com/mwaskom/ziegler) app, which runs in the browser and makes it easy to visualize the data.

Development
-----------

https://github.com/mwaskom/lyman

Please [submit](https://github.com/mwaskom/lyman/issues/new) any bugs you
encounter to the Github issue tracker.

Testing
-------

[![Build Status](https://travis-ci.org/mwaskom/lyman.svg?branch=master)](https://travis-ci.org/mwaskom/lyman)

You can exercise the unit-test suite by running `nosetests` in the source directory.

License
-------

Released under a BSD (3-clause) license

