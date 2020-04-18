.. _releases:

Release notes
=============

v2.0.1 (unreleased)
-------------------

- Fixed issues with the volume segmentation: added the "Accumbens area" to the subcoritcal gray matter seg and added the 3rd and 4th ventricles to the CSF seg. This change will require rerunning workflows.


v2.0.0 (April 9, 2020)
----------------------

Lyman version 2 comprises a set major change to the library (and an essentially complete rewrite of the codebase). Major aspects of the changes are summarized here; more details are available throughout the collection of related `pull requests <https://github.com/mwaskom/lyman/projects/1>`_.

- The preprocessing workflow now transforms all images into a cross-experiment functional template space that is defined in register with the Freesurfer anatomy. All spatial operations (motion-correction, unwarping, and transformation into template space) are applied in one step to minimize interpolation error.

- Preprocessing has been streamlined to involve mainly spatial transformations of the images, The various signal-processing operations (smoothing, temporal filtering, artifact detection) are now considered part of the modeling workflow, meaning that parameters that control these operations are model-specific. There are no longer separate "smoothed" and "unsmoothed" paths through the workflows.

- Model-fitting and contrast estimation are now implemented in Python, instead of FSL binaries (although univariate model fitting still uses the FILM GLS prewhitening algorithm).

- An experiment-independent ``template`` workflow has been added, and the ``reg`` and ``ffx`` workflows have been removed.

- There is no longer the concept of an "altmodel"; instead, different models are a first-class level of the lyman organization hierarchy.

- Lyman currently supports only HCP-style datasets that include a single pair of spin-echo EPI images with opposite phase encoding directions, which are used for unwarping susceptibility-induced distortions and registration to the anatomy.

- There is currently no support for anatomical normalization or group analyses.

- There is better support for experiments where subjects are scanned in multiple sessions.

- The contrast estimation code now properly handles contrasts between parameter estimates where one of the parameters is undefined in some runs.

- Spatial smoothing is performed using a novel algorithm that smooths in volume space using Gaussian weights determined by distance on the surface manifold.

- A number of static images that were generated for quality control have been removed, and others have been added. 

- The command-line interface has changed so that all interaction happens through a single ``lyman`` command line script that has sub-modes corresponding to different workflows (and, in the future, other functionality such as results visualization).

- Automated test coverage of the codebase has been dramatically improved.

- Various aspects of the supporting library code has been moved from `moss <https://github.com/mwaskom/moss>`_ into lyman itself, and moss is no longer a dependency of lyman.


v1.0.0 (July 7, 2017)
---------------------

This release is being marked as version 1.0 to signify that it is the final iteration of lyman as it existed in the versions leading up to 1.0. From here on, there will be major, breaking changes to the workflows, both from an API perspective and in terms of what kind of analyses are supported. It is possible, but not promised, that minor bugs will be fixed on the 1.0 branch. But going forward, all major development will take place in a 2.0 version that might look substantially different.

- Internal code changes to bring provisional support for Python 3 and later versions of numpy.

- Fixed slice time correction for data collected on a Siemens scanner. The Siemens interleaved pulse sequence changes the order of slices depending on whether there is an odd or even number of slices (for some insane reason). It is now possible to set ``interleaved="siemens"`` in your experiment file to handle this issue properly. See `this blog post <https://practicalfmri.blogspot.com/2012/07/siemens-slice-ordering.html>`_ for more information.

- Made it possible to turn off the FSL highpass filter by setting the ``hpf_cutoff` variable in the experiment file to ``None``.

- The ``lyman.mvpa`` and ``lyman.evoked`` modules have been removed.

- Added the ``view_ffx_results.py`` script, which is a wrapper around ``Freeview`` to boot up a useful visualization of fixed effects statistics on the high-resolution anatomical image and surface mesh.

- Added the ``view_reg.py`` script, which is a wrapper around ``Freeview`` to boot up a useful visualization of the functional-to-anatomical registration quality.

- The top-level cache directory is no longer removed at the end of workflow execution. This means that it is possible to be running multiple workflows (provided that they correspond to different experiments/models) simultaneously without interference from the first one to finish.

v0.0.10 (May 19, 2016)
----------------------

Command line interface
~~~~~~~~~~~~~~~~~~~~~~

- The full command-line argument namespace is now saved out with the other
  experiment parameters for better reproducibility.

Registration workflow
~~~~~~~~~~~~~~~~~~~~~

- The correct registration matrix to go from the epi space to the anatomy is
  now written out during the registration so that downstream results from
  experiments that use the ``-regexp`` flag are correct.

v0.0.9 (December 11, 2015)
--------------------------

Preproc workflow
~~~~~~~~~~~~~~~~

- Added the ability to supply fieldmap images that can be used to unwarp
  distortions caused by susceptibility regions. This uses FSL's ``topup`` and
  ``applytopup`` utilities. The images you should provide aren't actually
  traditional "fieldmaps", but rather images with normal and reversed phase
  encoding directions, from which a map of the distortions can be computed.
  See the new experiment option ``fieldmap_template``.


Model workflow
~~~~~~~~~~~~~~

- Added the ability to include additional nuisance variables in the model.
  This can now include eigenvariates of deep white matter timeseries and the
  mean signal from across the whole brain. This involves the new experiment
  options ``wm_components`` and ``confound_sources``. This change requires that
  you rerun the preproc workflow before running the model workflow after
  updating lyman.

- Made the inclusion of artifact indicator vectors in the design matrix
  optional.  See the new experiment option ``remove_artifacts``.

- Made some changes to the model summary node to use memory more efficiently.
  The model summary code should now use a similar amount of memory as the
  ``film_gls`` model fitting process.


Fixed effects workflow
~~~~~~~~~~~~~~~~~~~~~~

- Fixed effects workflow now saves out a mean functional image that is the
  grand mean across runs.

v0.0.8 (July 10, 2015)
----------------------

Model workflow
~~~~~~~~~~~~~~

- Upgraded the model workflow to be compatible with FSL 5.0.7 and later. If you
  upgrade lyman, you will have to upgrade your FSL installation (i.e. it does
  not maintain backwards compatibility with older FSL). You should also upgrade
  to nipype 0.10. The main advantage of upgrading should be increased memory
  performance in the model estimation.

Registration workflow
~~~~~~~~~~~~~~~~~~~~~

- Added the ability to do cross-experiment registration, e.g. in the case where
  you have a functional localizer. This is accomplished through the ``-regexp``
  flag in ``run_fmri.py``. For example, the cmdline ``run_fmri.py -exp A
  -regexp B -regspace epi -timeseries`` will combine the func-to-anat matrices
  from experiment A and the anat-to-func matrix from the first run of
  experiment B, placing the experiment A files in a common space with
  experiment B files.

Fixed-effects workflow
~~~~~~~~~~~~~~~~~~~~~~

- The fixed effects analysis no longer crashes when a subject did not have any
  observations for an event.

Mixed-effects workflow
~~~~~~~~~~~~~~~~~~~~~~

- The mixed effects workflow now excludes empty images, allowing you to run it
  on a group where some subjects did not have any observations for a specific
  event/contrast.
- Updated the mixed effects boxplot code for compatibility with seaborn 0.6.

Anatomy snapshots script
~~~~~~~~~~~~~~~~~~~~~~~~

- Added plots of the native white and pial surfaces
- Surface plots are now saved in one image file with all views, and the subplot
  size is automatically inferred to maximize the usage of space
- Added ventral views to the surface images
- Changed how the surface normalization is summarized. The new visualization
  highlights vertices where the binarized curvature value is different between
  the normalized subject and template
- Remove the "-noclose" option, as better ways to avoid the problem that
  motivated it have been identified.

Note that there are corresponding changes in ziegler that are needed to
properly view the new images, and there isn't backwards compatibility
with the old outputs. This script can be rerun on older lyman analyses
without affecting any results.

Surface snapshots script
~~~~~~~~~~~~~~~~~~~~~~~~

- Changed how the individual frames of the surface snapshots are stitched
  together to maximize the use of space. This (and the changes in the anatomy
  snapshots script) rely on some new functions in ``lyman.tools.plotting``
  that may be generally useful.
- Remove the "-keep-open" option, as better ways to avoid the problem that
  motivated it have been identified, and removed the "-no-window" option,
  as it is not clear whether this ever worked.

v0.0.7 (February 26, 2015)
--------------------------

Execution
~~~~~~~~~

- Added the option to submit jobs using slurm.
- Added the ``crash_dir`` parameter at the project level (i.e. it will be
  defined when you run ``setup_project.py`` and will be stored in
  ``$LYMAN_DIR/project.py``). This allows you to specify where debugging
  information will be written if something goes wrong during workflow
  execution. The previous approach to selecting where crash files would be
  written was not robust in all execution contexts. **Important:** if you
  upgrade to this version and try to rerun something in an existing project,
  you will get an error.  This can be avoided by defining ``crash_dir`` in your
  project file. These files are usually only transiently useful, so the default
  location for new projects is ``/tmp/nipype-$USER-crashes``.

Registration workflow
~~~~~~~~~~~~~~~~~~~~~~

- Added ability to register the residual timeseries after model-fitting, using
  the ``-residual`` flag in ``run_fmri.py`` when ``reg`` is in the workflow
  spec. This file will be called ``res4d_xfm.nii.gz`` in the registration
  output.

v0.0.6 (November 10, 2014)
--------------------------

This is a bugfix release that anyone using v0.0.5 should upgrade to.

Preprocessing workflow
~~~~~~~~~~~~~~~~~~~~~~

- Fix a bug that was introduced in v0.0.5 where the preprocessed timeseries
  was not being written out by the DataSink.

v0.0.5 (November 7, 2014)
-------------------------

Preprocessing workflow
~~~~~~~~~~~~~~~~~~~~~~

- Added a workaround some changes in later versions of FSL
  that now return a de-meaned timeseries from the highpass filter.
  In FEAT, the mean is replaced, and the rest of the processing carries
  on as usual. Because I don't want to break compatability with older
  versions of FSL, this adds back in the mean but only if it looks
  like the filtered timeseries has been de-meaned. **Note**: This uses
  a simple heuristic, which may not be robust in all cases, so it is
  important to check that the signal-to-noise maps make sense if you are
  doing something that expects a nonzero timeseries mean.

v0.0.4 (October 28, 2014)
-------------------------

Infrastructure
~~~~~~~~~~~~~~

- Added continuous integration with TravisCI.

Mixed effects workflow
~~~~~~~~~~~~~~~~~~~~~~

- Fixed a bug where the analysis mask was getting smoothed on the surface.

FNIRT-based normalization workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Fixed a bug where the outputs of FNIRT were not properly renamed and thus
  were not correctly picked up by the registration stage of the functional
  pipeline.

Surface snapshots script
~~~~~~~~~~~~~~~~~~~~~~~~

- Fixed a bug where surface visualization would crash when the analysis mask
  includes all vertices.

- Added a brief pause between updating the view and saving a snapshot to allow
  redrawing to finish.

v0.0.3 (September 16, 2014)
---------------------------

Preprocessing workflow
~~~~~~~~~~~~~~~~~~~~~~

- Added the ``coreg_init`` field to the experiment file. This is ``"fsl"`` by
  default, which uses ``FLIRT`` to get a rough coregistration before using the
  boundary-based algorithm (this was the old behavior). It can also be set to
  ``"header"``, which assumes that the functional and anatomy are roughly in
  register in real space and that a mapping can be found with the header
  geometry.

Model workflow
~~~~~~~~~~~~~~

- Added the ``memory_request`` field to the experiment file so that you can
  request more memory on memory-intensive nodes (those involving model
  estimation) when using a managed distribution engine (such as SGE). This can
  be helpful for whole-brain high-resolution studies.

Anatomical normalization
~~~~~~~~~~~~~~~~~~~~~~~~

- Fixed a bug in ANTS-based anatomical normalization that affected non-OSX
  systems. This bug caused a workflow crash, so if you haven't seen it, don't
  worry about it.

v0.0.2 (June 18, 2014)
----------------------

Anatomical normalization
~~~~~~~~~~~~~~~~~~~~~~~~

- Added ANTS-based volume normalization. This provides substantial improvements
  over the FSL-based normalization that was previously used. However, ANTS can
  be difficult to install, so this is optional and off by default. It controled
  through a variable in the ``project.py`` file, ``ants_normalization``, which
  should be either ``True`` or ``False``. After enabling it, you can use the
  command-line tools as before, and ANTS will be used in ``run_warp.py`` and
  ``run_fmri.py -workflow reg``.

Preprocessing workflow
~~~~~~~~~~~~~~~~~~~~~~

- The artifact detection code now uses robust metrics (median and median
  absolute deviation). Previously, it used mean and standard deviation.
  **Importantly**, this means that the your intensity threshold should be
  adjusted by a scaling factor to provide a similarly stringent threshold.
  As a general rule of thumb, 1 SD is about 1.48 MADs.

- Added white noise spike detection. This is controlled through the
  ``spike_threshod`` in the experiment file. It is also in units of median
  absolute deviation. It is ``None`` by default, indicating that no volumes
  will be excluded for white noise spikes. Additionally, a plot that can be
  used to diagnose spikes has been added to the artifact detection report.

- Changed the derivation of the brain mask. Previously, this mask was
  intensity based (although the intensity threshold was determined within a
  mask output by BET). Now, the Freesurfer segmentation is used to define
  an anatomical brain mask, which is then transformed into native run
  space. This should avoid losing voxels in magnetic susceptibility areas
  like ventral temporal cortex.

- Otherwise updated the preproc report with better summary figures.

Subject-level modelling
~~~~~~~~~~~~~~~~~~~~~~~

- It should now be possible to run the model workflow on task-free data
  (i.e. for functional connectivity analysis) by setting "``design_name``"
  to ``None`` in the experiment file.

- Added computation and reporting of residual tSNR.

- Improved the colormaps used for reporting summary statistics about the
  mode (residual variance, R squared, etc.)

- Improved the plot showing correlations between confound and task
  variables

- Otherwise improved the logic and testing of the model workflow.

- Added to and improved the model report at the fixed effects stage.

Mixed effects workflow
~~~~~~~~~~~~~~~~~~~~~~

- Updated the mixed effects model reporting and simplified the workflow graph.

- The boxplot of COPE effect sizes in the mixed effects report is now taken
  from a sphere (with the same size as in the activation peak image)
  centered at each peak voxel rather than just from the single voxel
  itself.
