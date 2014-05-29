.. _dependencies:

Lyman Dependencies
==================

As a high-level package, lyman depends on a relatively large amount of external
software. Getting set up to run an initial analysis can thus be somewhat of a
hassle, which we apologize for. Below, we attempt to list the required packages
and versions that are known to work with lyman. It is likely that, in many
cases, older versions will also work, but we do not have the bandwidth to
rigorously test the package across multiple versions or maintain significant
backwards compatibility. In very rare cases, lyman updates will be associated
with upstream contributions to the dependent packages, and lyman will
occasionally rely on development versions. We'll attempt to avoid this as much
as possible, but sometimes it is the best approach.

Depending on what you are trying to do, it may not be necessary to install all
of these packages. In the event that you don't want to exhaust this list, the
unit tests and the ``-dontrun`` switch for the command line interfaces can be
useful for finding missing dependencies.

Lyman requires Python 2.7, and does not run on Python 3. We strongly recommend
using the `Anaconda <https://store.continuum.io/cshop/anaconda/>`_
distribution, which ships with the majority of the Python packages needed to
run lyman.

Non-Python Software
-------------------

- FSL 5.0

- Freesurfer 5.3

Released Python Packages
------------------------

- Python 2.7

- IPython 2.0

- numpy 1.7

- scipy 0.12

- matplotlib 1.3

- scikit-learn 0.14

- scikit-image 0.9

- statsmodels 0.5

- pandas 0.12

- nibabel 1.3

- nipype 0.9

- nipy 0.3

- PySurfer 0.4

Lyman Ecosystem
---------------

- `ziegler <https://github.com/mwaskom/ziegler>`_

- `moss <https://github.com/mwaskom/moss>`_

- `seaborn <https://github.com/mwaskom/seaborn>`_

