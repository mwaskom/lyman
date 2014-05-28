Python fMRI Analysis Ecosystem
==============================

This repository contains a set of Python-based code for analyzing
functional MRI data using FSL and Freesurfer tools and Nipype.

Documentation
-------------

Online documentation can be found [here](http://www.stanford.edu/~mwaskom/software/lyman)

Dependencies
------------

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

- My utility packages: [seaborn](https://github.com/mwaskom/seaborn) and [moss](https://github.com/mwaskom/moss)

Basic Workflow
--------------

- All stages of processing assume that your anatomical data have been
  processed in Freesurfer (recon-all)

- `run_warp.py`: perform anatomical normalization

- `run_fmri.py`: perform subject-level functional analyses

- `run_group.py`: perform basic group level mixed effects analyses

License
-------

BSD (3 Clause)

Notes
-----

Although all are welcome to use this code in their own work, it is not officially
"released" in any capacity and many elements of it are under active development and
subject to change at any time.

Please submit bug reports/feature ideas through the Github issues framework.

