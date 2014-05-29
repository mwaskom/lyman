Lyman: an fMRI Analysis Ecosystem
=================================

Lyman is a high-level ecosystem for analyzing neuroimaging data using
open-source software. It aims to support an analysis workflow that is
powerful, flexible, and reproducible, while automating as much of the
processing as possible.

Lyman is broadly divided into two sets of tools. There is a command-line based
interface to a set of pipelines, where FSL, Freesurfer, and Python-based tools
are integrated using `Nipype <http://nipy.sourceforge.net/nipype/>`_. These
pipelines will take raw Nifti files and process them all the way through a
basic group analysis with minimal manual intervention. There is also a library
for multivariate analyses that adapts `Scikit-Learn
<http://scikit-learn.org/stable/>`_ for common fMRI applications. Both sets of
tools cache their intermediate processing steps and can be easily parallelized,
which helps to make data analysis more efficient and encourages the development
of reproducible analysis notebooks.

Because the processing is heavily automated, lyman generates a number of static
plots and images that are useful for understanding the results of the analyses
and diagnosing any problems that might arise. These files are stored alongside
the data they correspond with in the output directories. Although it is
possible to manually browse them, a much better approach is to use the
companion `zielger <https://github.com/mwaskom/ziegler>`_ webapp, which is
tightly integrated with the lyman results and makes it very easy to understand
what has happened with your data.

Installing
----------

To install lyman, you can run

    pip install lyman

This will install the stable version. To get the development version, you
should run

   pip install git+git://github.com/mwaskom/lyman.git#egg=lyman

However, there are a number of :ref:`dependencies <dependencies>` that should
be installed first, in either case.

To check out the code, please see the `github repository
<https://github.com/mwaskom/lyman>`_.

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2

   glossary
   dependencies
   workflow
   experiments
   commandline
   procstream

