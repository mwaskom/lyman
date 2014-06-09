.. _installing:

Installing lyman
================


To install lyman, you can run

    pip install lyman

This will install the stable version. To get the development version, you
should run

   pip install git+git://github.com/mwaskom/lyman.git#egg=lyman

However, there are a number of dependencies that should be installed first.

Dependencies
------------

As a high-level package, lyman depends on a relatively large amount of external
software. Getting set up to run an initial analysis can thus be somewhat of a
hassle, which we apologize for. Below, we attempt to list the required packages
and versions that are known to work with lyman. (For the most part 

Depending on what you are trying to do, it may not be necessary to install all
of these packages. In the event that you don't want to exhaust this list, the
unit tests and the ``-dontrun`` switch for the command line interfaces can be
useful for finding missing dependencies.

Lyman requires Python 2.7, and does not run on Python 3. We strongly recommend
using the `Anaconda <https://store.continuum.io/cshop/anaconda/>`_
distribution, which ships with the majority of the Python packages needed to
run lyman. The rest can be easily intalled with pip.


Non-Python Software
~~~~~~~~~~~~~~~~~~~

- FSL 5.0

- Freesurfer 5.3

- ANTs 1.9


Python Packages
~~~~~~~~~~~~~~~

- Python 2.7

- IPython 2.0

- numpy 1.7

- scipy 0.12

- matplotlib 1.3

- nipype 0.9

- nibabel 1.3

- pandas 0.12

- scikit-learn 0.14

- scikit-image 0.9

- statsmodels 0.5

- PySurfer 0.4

- `moss <https://github.com/mwaskom/moss>`_

- `seaborn <https://github.com/mwaskom/seaborn>`_
