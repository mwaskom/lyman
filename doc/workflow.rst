High-Level Lyman Workflow
=========================

Although not every analysis performed with lyman will be identical, there is a
relatively stereotyped progression of steps that will go into those analyses.
That workflow is documented here at a high level. There is more detailed
information about many of these steps elsewhere in the documentation.

Install Lyman
-------------

If you haven't already done this, you'll need to :ref:`install <installing>`
lyman and its dependencies. The current development approach aims for
relatively small, incremental releases, so if you're starting a new project it
might be good to check if there is an update.

Lyman's unit-test suite can be run by executing the command ``nosetests`` in
the source directory. Because it is difficult to unit-test high-level scripting
code, the coverage is lighter than would be ideal (benchmarking the workflows
on real data is an active area of development). However, the test suite will
catch basic things like missing dependencies.

Setup the Lyman Project
-----------------------

Create a directory that will serve as your *lyman directory* and execute the
command ``setup_project.py``  within it. This will take you through defining the
location of your *data_dir* and *analysis_dir*, along with a few other pieces
of basic information. The output of this script is a file called
``project.py``. Once it's been created, you probably won't need to touch it
again unless you move things around.

Next, set up one or more *experiment* files describing the processing parameters
for each of your experiments. There is information on how to do that :ref:`here <experiments>`.

If you want to have a list of default subjects to perform processing over, put
their subject ids in a plain text file called ``subjects.txt`` within this
directory. You can define different groups of subjects too in files called
``<groupname>.txt``. These files should have one subject id per row, and each
subject id should correspond to a subdirectory in your *data_dir*.

Prepare the Data
----------------

The complexities of conversion from DICOM format are outside of lyman's scope.
We assume that raw functional data is stored in a single 4D nifti file at a
predictable location within the *data_dir*. You should be able to write a
template string that plugs in the subject id and a wildcard for multiple runs
to find all files relative to your data_dir. For example, I usually go with
something like ``<data_dir>/<subject_id>/bold/functional_<run>.nii.gz``.  If
you have more than 9 runs of data, you should zero-pad the run number.

Lyman also relies on the outputs from `Freesurfer
<http://surfer.nmr.mgh.harvard.edu/>`_. Before using the lyman tools, you
should processes your anatomical data using ``recon-all``. The lyman *data_dir*
should be synonymous with your Freesurfer ``$SUBJECTS_DIR``, and you should use
the same subject ids with both tools. The ``anatomy_snapshots.py`` script that
ships with lyman can be useful for performing quality control on these
outputs.

To fit any models (either univariate timeseries models or multivariate decoding
models), you'll need a description of your experiment design. The basic format
for this is a file living at ``<data_dir>/<subject>/design/<design name>.csv``
that, at a minimum, contains the run number, onset time, and condition name for
each event in your design. This information can be augmented with details about
the length of presentation and one or more parametric values associated with
each event. See the :ref:`design docs <design>` for more information.

Process the Data
----------------

At this point, you're ready to process your data. This primarily happens
through a command-line interface, which is made up of three scripts:
``run_fmri.py``, ``run_warp.py``, and  ``run_group.py``. The first performs all
of the single-subject functional processing and analysis, the second estimates
the volume-based anatomical normalization parameters, and the third performs
basic group analyses. The fMRI script can be run over smaller chunks of
processing using the ``-workflows`` argument. For just about anything you'll
want to do with lyman, you have to preprocess the data. (``run_fmri.py -w
preproc``). At that point, you may continue on with subject-level timeseries
modeling (``run_fmri.py -w model reg ffx``), possibly on several different
models (``run_fmri.py -w model -altmodel <altmodel>``).

For detailed information about these scripts, you can run them with the ``-h``
flag to see what they do and how to use them. That information is also
reproduced :ref:`online <commandline>`.
