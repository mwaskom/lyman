.. _experiments:

Defining Experimental Details
=============================

There are two places where you specify the experimental details for an
analysis performed using lyman. General experimental parameters that are
constant across a group of subjects go in an *experiment file* in your *lyman
directory*. You also must define the paradigm information, which can be
different for each subject. Both modes of specification are documented
here.

The Experiment File
-------------------

Experiments are defined by creating a Python file in your lyman directory
called ``<experiment>.py``.  This is just a regular Python file that gets
imported as a module at runtime.

There is a default set of parameters that will be overridden by the fields in
this file; to see these defaults, run the following code::

    In [1]: import lyman

    In [2]: print lyman.default_experiment_parameters()

The namespace of the imported experiment file is used to update these values.

Here is a description of each of these fields.

Preprocessing Parameters
~~~~~~~~~~~~~~~~~~~~~~~~

.. glossary::

   source_template
    A string that, relative to your *data directory*, provides a path to the
    raw functional files for your experiment. It should include the string
    formatting key ``{subject_id}`` in place of the subject id in the path.
    When you have multiple runs of data, replace the run number with a glob
    wildcard. For instance, your template might look like
    ``"{subject_id}/bold/func_*.nii.gz"``.

   whole_brain_template
    If your acquisition covered only part of the brain (e.g. you are doing a
    high-res study), you can also supply a single EPI image in the same slice
    prescription but with enough slices to cover the full brain. This image
    will be used to initialize the anatomical coregistration. This field
    provides a template path to this image, as with the ``source_template``
    (although you specify only one whole-brain file).  If you don't have such
    an image, just leave this out of the experiment file.

   n_runs
    An integer with the number of functional runs in the experiment.

   TR
    A float with the repetition time of the acquisition in seconds.

   frames_to_toss
    An integer with the number of frames to trim from the beginning of the
    timeseries.

   temporal_interp
    A boolean that specifies whether slice-time correction should be performed.

   interleaved
    A boolean that specifies whether the data were acquired with an interleaved
    protocol. Only relevant if ``temporal_interp`` is True.

   coreg_init
    A string that is either ``"fsl"`` or ``"header"``. This controls how the
    boundary-based registration algorithm is initialized. Using ``"header"``
    may give better performance for partial brain acquisitions, but only
    if the anatomicals were acquired in the same session as the functionals.

   slice_order
    A string that is either ``"up"`` or ``"down"``. This corresponds to the
    slice acquisition order, and is only relevant if ``temporal_interp`` is
    True.

   intensity_threshold
    A float specifying the threshold for intensity artifacts. Frames where the
    median signal intensity across the brain exceeds the grand run median by
    this many median absolute deviations will be excluded from the model.

   spike_threshold
    A float specifying the threshold for white noise spike artifacts, or
    ``None`` to avoid detecting spikes (a plot is produced in either case).
    This is in the same units as the *intensity_threshold*, but spikes are
    usually more extreme. Unlike the intensity_threshold, white noise spikes
    will be unidirectional, but whether they are going to be positive or
    negative deflections will depend on your scanner. Ask your physicist. 

   motion_threshold
    A float specifying the threshold for motion artifacts. Frames where the
    total displacement (in mm) relative to the previous frame exceeds this
    number will be excluded from the model.

   smooth_fwhm
    A float with the smoothing kernel size for the volume-based SUSAN smoothing.
    Note that an unsmoothed version of the timeseries is always produced.

   hpf_cutoff
    A float with the cutoff time (in seconds) for the highpass filter. This
    value is used in both preprocessing and in the model.

Model Parameters
~~~~~~~~~~~~~~~~

.. glossary::


   design_name
    A string used to build the name of the file with paradigm information (see
    below).

   condition_names
    A list of strings with condition names. If this is absent or set to None,
    the sorted unique values in the ``condition`` field of the design file are
    used. Otherwise, the design matrix will include only the conditions named
    in this list (in the order provided here).

   regressor_file
    The name of a file containing information about other regressors to add to
    the timeseries model (see below).

   regressor_names
    A list of strings that can be used to select specific columns from the
    regressor file specified above. If None, all columns in the csv file
    are used.

   hrf_model
    A string corresponding to the name of the HRF model class. Currently
    only ``GammaDifferenceHRF`` is supported.

   temporal_deriv
    Boolean specifying whether a derivative regressor should be used in the
    model for each explanatory variable (these are considered regressors of
    no interest).

   confound_pca
    A boolean specifying whether the dimensionality of the confound matrix
    (currently just the 6 motion parameters) should be reduced using PCA
    to include dimensions explaining 99% of the variance.

   hrf_params
    A dictionary with keyword arguments for the HRF model class.

   contrasts
    A list of tuples, with one entry per contrast. Each contrast is defined by
    the 3-tuple ``(<name>, [<conditions>], [<weights>])``. For instance, if you
    want to test the contrast of hard events vs. easy events, you would use
    ``("hard-easy", ["hard", "easy"], [1, -1])``. The conditions must be
    present in the design, but you do not have to include the names of any
    conditions not involved in the contrast. If you provided a list of
    condition names, baseline contrasts are automatically generated for each of
    these conditions and prepended to this list. Importantly, the contrast names
    end up in file paths for the analysis results, so you should avoid spaces.

   memory_request
    An integer with the number of gigabytes of memory to request for model
    workflow nodes that involve large memory computations. This only applies to
    submission through a ``qsub``-based distribution plugin.

Group Analysis Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~

.. glossary::

   flame_mode
    A string indicating the type of inference that should be performed in the
    group model. Options are ``ols``, ``flame1``, and ``flame12``, for ordinary
    least squares, MAP mixed effects, and full MCMC mixed effects,
    respectively.

   cluster_zthresh
    A float indicating the threshold used to initially define clusters in the
    Z-stat maps during multiple comparisons correction.

   grf_pthresh
    A float indicating the *p* value threshold for thresholding the corrected
    Z-stat images.

   peak_distance
    A float specifying the minimum distance (in mm) between local
    minima when finding activation peaks.

   surf_name
    The name of a Freesurfer surface to plot group results on.

   surf_smooth
    Extent of spatial smoothing (in mm) to apply after sampling to the surface.

   sampling_range
    A 3-tuple of floats where where to start, stop and the size of the step
    (all in ``sampling_units``) when projecting data onto the white surface. This
    only applies to group analysis in fsaverage space.

   sampling_units
    A string that is either "frac" or "mm" that makes up part of the
    specification for projecting results onto the surface manifold (it
    determines the units of the ``sampling_range`` paramters). This only applies
    to group analysis in fsaverage space.

   sampling_method
    A string that is either "average", "max", or "point" that makes up part of
    the specification for projecting results onto the surface manifold (it
    determines how to summarize the samples obtained using ``sampling_range`` and
    ``sampling_method`` into a single value at each verex). This only applies to
    group analysis in fsaverage space.

   surf_corr_sign
    A string that is either "pos", "neg", or "abs" for the sign of the test to
    run. This only applies to group analysis in fsaverage space.

The parameters that were present in this file at runtime will be saved with the
other processing outputs in the preproc and model analysis directories (in a
file called ``experiment_info.json``). Any comments in the docstring to this
module will be included in this archive. Note that if you preprocess your data,
change the experiment definition, and then run the model without rerunning
preproc, the preprocessing parameters in this archived file will be inaccurate.

.. _design:

Detailed Design Information
---------------------------

The design file
~~~~~~~~~~~~~~~

You also have to generate a file in ``csv`` format for each subject specifying
what actually happened during the scan. This file should live at
``<data_dir>/<subject_id>/design/<design_name>.csv``, where ``design_name`` is
specified in the experiment file. Each row in this file corresponds to an
event, where the term "event" is used broadly and can mean a "block" in a block
design experiment. At a minimum, the following fields need to be present in
this file:

.. glossary::

   run
    1-based index for the run number.

   condition
    A string with the condition name for the event.

   onset
    Onset time (in seconds) of the event. 0s is considered to be the onset of
    the first frame that is not trimmed (by ``frames_to_toss`` in the
    experiment file).

For example, an extremely basic design might look like this::

    run,condition,onset
    1,easy,0
    1,hard,12
    2,easy,0
    2,hard,12

Of course, you'll almost certainly want to write this file using
`Pandas <http://pandas.pydata.org/>`_ and not by hand.

This information can be augmented with the following fields:

.. glossary::

   duration
    Duration (in seconds) of the event. If duration is 0 (which is the default),
    it is assumed to be an "impulse".

   value
    A parametric value corresponding to the height of the response. The defualt
    value is 1.

Additionally, other columns can be included with some parametric value for that
event (e.g. reaction time). This information is not used in the timeseries
model, but it can be used in decoding analyses to regress confounds out of the
data.

A more complete file that will result in the same design as the simple example
above would read

::

    run,condition,onset,duration,value,rt
    1,easy,0,0,1,0.894
    1,hard,12,0,1,1.217
    2,easy,0,0,1,0.993
    2,hard,12,0,1,1.328

The regressors file
~~~~~~~~~~~~~~~~~~~

A secondary and optional way to add design information uses a ``regressor``
file.  This file, like the ``design`` file, should be a ``csv`` and should live
at ``<data_dir>/<subject_id>/design/<regressor_file>.csv``, where
``regressor_file`` is specified in the experiment file. The format is a csv
where column names are regressor names and rows are observations of the
regressors at each timepoint in the experiment. Additionally, the file must
have a ``run`` column, specifying the 1-based run number for each observation.
This information is not transformed when building the design matrix beyond
de-meaning by run.  This is intended to allow the use of, e.g., BOLD timeseries
information extracted from seed ROIs for functional connectivity analyses. The
regressors are considered elements "of interest" in the design matrix, can be
included in contrasts, and contribute to the "main model" R^2 calculation.

An example file for an experiment where each run has 3 TRs and the experimenter
is interested in functional connectivity early visual areas might look like

::

    V1,V2,run
    1.46,1.55,1
    0.80,-0.37,1
    -1.91,-1.01,1
    -0.65,0.38,2
    1.00,1.01,2
    -0.88,-2.00,2


Each experiment can take information from at most one regressor file, but you
can create multiple regressor files for different experiments. It is also
possible to include all possible regressors in a single file and select the
specific columns for each experiment using the ``regressor_names`` field in
the experiment definition.

Specifying Alternate Models
---------------------------

You can fit several models to the same preprocessed data, which in lyman is
called an *altmodel* or *alternate model*. To fit an alternate model, create an
experiment file called ``<experiment>-<altmodel>.py`` and execute
``run_fmri.py`` with the arguments ``-experiment <experiment> -altmodel
<altmodel>``. This module "inherits" from the base experiment file, so you only
need to include information if it differs from what was previously defined.
Note that the experiment parser isn't smart enough to detect when an altmodel
overrides parameters that affect preprocessing.

