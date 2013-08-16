Lyman: an fMRI Analysis Ecosystem
=================================

Lyman offers a high-level ecosystem for analyzing neuroimaging data.
It aims to support an analysis workflow that is powerful, flexible, and
reproducible while eliminating as much drudgery as possible. The main
consequence of these objectives is that lyman is fairly opinionated about how
many aspects of the analyses should be performed. It is thus not a general
framework for implementing fMRI analysis workflows. However, the hope is that
these constraints apply only to the implementation level and that the tools can
be applied to a wide range of scientific questions within cognitive
neuroscience.

Lyman is broadly divided into two sets of tools. There is a command-line based
interface to a set of processing pipelines, implemented with `Nipype
<http://nipy.sourceforge.net/nipype/>`_ and mostly using FSL programs,
that will process an imaging dataset from Nifti files fresh off the scanner all
the way through a basic group analysis with minimal manual intervention. There is
also a library for multivariate pattern analysis that adapts `Scikit-Learn
<http://scikit-learn.org/stable/>`_ to common fMRI applications. An attractive
feature of both sets of tools is that they can be parallelized very easily.

Because the basic steps in an fMRI analysis are heavily automated, the
user ends up quite far removed from his or her data as it is processed. To
mitigate this, lyman generates a number of plots and images that are useful for
understanding the results of the analyses and diagnosing any problems that
might arise. These files are stored alongside the data they correspond with in
the output directories. Although it is possible to manually browse them,
a much better approach is to use the companion `zielger
<https://github.com/mwaskom/ziegler>`_ webapp, which is tightly integrated with
the lyman results and makes it very easy to understand what has happened with
your data.

It is important to note that these tools are not officially released in any
capacity. You'll have to get the code from the `Github repository
<https://github.com/mwaskom/lyman>`_ and be aware that things may change or
break (possibly without warning). So, it's good practice to keep on top of the
Github commit log if you're updating while actively analyzing data. With that
said, the main body of code has about three years of development under it, and
much of the package has gotten quite mature and stable.

Contents:
---------

.. toctree::
   :maxdepth: 2

   glossary
   workflow
   experiments
   procstream

