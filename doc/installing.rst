.. _installing:

Installing lyman
================


To install lyman, you can run

    pip install lyman

This will install the stable version. To get the development version, you
should run

   pip install git+git://github.com/mwaskom/lyman.git

However, there are a number of dependencies that should be installed first.

Dependencies
------------

As a high-level package, lyman depends on a relatively large amount of external
software. Getting set up to run an initial analysis can thus be somewhat of a
hassle, which I apologize for. Below, I attempt to list the required packages
and versions that are known to work with lyman. This isn't updated that often,
so later versions of these libraries will probably work. Older ones might too.
It's a hard problem.

Depending on what you are trying to do, it may not be necessary to install all
of these packages. In the event that you don't want to exhaust this list, the
unit tests and the ``-dontrun`` switch for the command line interfaces can be
useful for finding missing dependencies.

Lyman requires Python 2.7, and does not run on Python 3. We strongly recommend
using the `Anaconda <https://store.continuum.io/cshop/anaconda/>`_
distribution, which ships with the majority of the Python packages needed to
run lyman. The rest can be easily installed with `conda`, or failing that,
`pip`.


Non-Python Software
~~~~~~~~~~~~~~~~~~~

- Freesurfer 5.3

- FSL 5.0.7+

.. note::

   Due to changes in FSL, the lyman 0.0.7 and earlier only compatibile
   with FSL 5.0.6 and earlier, and lyman 0.0.8 and later are only compatibile
   with FSL 5.0.7 and later.

- ANTs 1.9

.. note::

    Lyman is not compatible with later versions of ANTs.

Python Packages
~~~~~~~~~~~~~~~

- Python 2.7

- IPython 2.0

- numpy 1.7

- scipy 0.12

- matplotlib 1.3

- seaborn 0.6

- nipype 0.10

- nibabel 2.0

- pandas 0.12

- scikit-learn 0.14

- scikit-image 0.9

- statsmodels 0.5

- moss 0.3.4

- pysurfer 0.6
