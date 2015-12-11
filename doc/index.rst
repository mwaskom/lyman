Lyman: an fMRI Analysis Ecosystem
=================================

Lyman is a high-level ecosystem for analyzing neuroimaging data using
open-source software. It aims to support an analysis workflow that is
powerful, flexible, and reproducible, while automating as much of the
processing as possible.

Lyman offers a command-line based interface to a set of pipelines, where FSL,
Freesurfer, and Python-based tools are integrated using `Nipype
<http://nipy.sourceforge.net/nipype/>`_. These pipelines will take raw Nifti
files and process them all the way through a basic group analysis with minimal
manual intervention. Important intermediate files that might be useful for
later analysis are saved in predictable locations at the completion of the
pipelines.

Because the processing is heavily automated, lyman also generates a
number of static plots and images that are useful for understanding the results
of the analyses and diagnosing any problems that might arise. These files are
stored alongside the data they correspond with in the output directories.
Although it is possible to manually browse them, a much better approach is to
use the companion `zielger <https://github.com/mwaskom/ziegler>`_ webapp, which
is tightly integrated with the lyman results and makes it very easy to
understand what has happened with your data.

Lyman is provided freely and with open source in the hope that it might be
useful. However, there is no guarantee of support or stability. There has been
some effort put into documentation, but not every aspect of using the tools
will be obvious. Lyman supports a specific approach to analyzing data and may
not work for every experiment. Finally, the code may change between releases in
a way that is not backwards compatible.

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2

   installing
   releases
   workflow
   experiments
   glossary
   issues
   commandline
   procstream

