Lyman: a Python fMRI Analysis Ecosystem
=======================================

Lyman is a high-level ecosystem for analyzing neuroimaging data using
open-source software. It aims to support an analysis workflow that is powerful,
flexible, and reproducible, while automating as much of the processing as
possible.


Documentation
-------------

Online documentation can be found
[here](http://www.stanford.edu/~mwaskom/software/lyman)

Dependencies
------------

Python 2.7

### External

- FSL

- Freesurfer

### Python


- Core scientific Python environment (ipython, numpy, scipy, matplotlib)

- [pandas](https://github.com/pydata/pandas)

- [scikit-learn](https://github.com/scikit-learn/scikit-learn)

- [nipype](https://github.com/nipy/nipype)

- [nibabel](https://github.com/nipy/nibabel)

- [nipy](https://github.com/nipy/nipy)

- [nibabel](https://github.com/nipy/nibabel)

- [statsmodels](https://github.com/statsmodels/statsmodels)

- [seaborn](https://github.com/mwaskom/seaborn)

- [moss](https://github.com/mwaskom/moss)

Installation
------------

To install the released version, just do

    pip install lyman

You may instead want to use the development version from Github, by running

    pip install git+git://github.com/mwaskom/lyman.git#egg=lyman

Basic Workflow
--------------

- All stages of processing assume that your anatomical data have been
  processed in Freesurfer (recon-all)

- `run_warp.py`: perform anatomical normalization

- `run_fmri.py`: perform subject-level functional analyses

- `run_group.py`: perform basic group level mixed effects analyses

Development
-----------

https://github.com/mwaskom/lyman

Please [submit](https://github.com/mwaskom/lyman/issues/new) any bugs you
encounter to the Github issue tracker.

License
-------

Released under a BSD (3-clause) license

