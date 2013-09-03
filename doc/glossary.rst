A Glossary of Concepts
======================

The following concepts pop up throughout the package.

.. glossary::

    altmodel
        You can define multiple different timeseries models that will
        be fit to the same set of preprocessed data. An *altmodel* is defined
        in an *experiment* file named ``<experiment>-<altmodel>.py`` and will
        automatically inherit any information not defined within that file from
        its parent. Altmodel results are also stored within the *analysis_dir*
        under ``<experiment name>-<altmodel>``.

    analysis_dir
        One of the two main root directories. With very few exceptions,
        files that result from using lyman tools are stored here with a
        specific (and, hopefully, intuitive) organization. The basic hierarchy
        is organized by *experiment* at the highest level, followed by subject
        (or group) and then *workflows*. This directory is
        defined in the ``project.py`` file in your *lyman directory*, and used
        by almost every tool in the package.

    data_dir
        The other main root directory. Broadly speaking, the
        *data_dir* is full of the raw data that will be fed into your analyses
        (both imaging data and the details of your experimental paradigms). The
        Lyman tools won't mess with this directory much, so you are encouraged
        to store other raw data here too so that everything is in the same
        place, even if it's not relevant to lyman analyses. However, the
        outputs of anatomical normalization (stored in
        ``<data_dir>/normalization``) and ROI masks (in ``<data_dir>/masks``)
        end up here, so be aware of that. Lyman has some expectations about how
        this directory should be laid out. For starters, each subject must have
        her own directory within the *data_dir*. The *data_dir* should also be
        your Freesurfer ``SUBJECTS_DIR``, so lyman expects Freesurfer outputs
        to be in their normal place within this structure. The other important
        subdirectory is ``<data_dir>/<subj>/design``, which is where lyman
        looks for the ``csv`` files with information about the experimental
        design. This directory is defined in the ``project.py`` file in your
        *lyman directory*, and used by almost every tool in the package.

    experiment
        An *experiment* is the second level of a project hierarchy. An
        experiment consists of one or more functional runs of data that are
        preprocessed with a particular set of parameters. You can also define a
        timeseries model corresponding to a particular experiment, although the
        *altmodel* concept allows for a many-to-one mapping of models to
        experiments. Experiments are defined within files named
        ``<experiment>.py`` within your *lyman directory*. The experiment level
        is the highest point in the *analyis_dir* tree. A default experiment is
        defined within the ``project.py`` file in your *lyman directory*, so
        that you can focus on a particular experiment without frequently
        repeating yourself.
        
    lyman directory
        The *lyman directory* lives somewhere near your *analyis_dir* and
        *data_dir* and is the home for all of the files that specify how you
        and lyman jointly interact with your data for a given *project*. Most
        lyman tools expect an environment variable named ``LYMAN_DIR`` to point
        at this directory. The most important file here is the ``project.py``
        module, which tells lyman where it can find the raw data and where it
        should put processed files.  This directory also contains a set of
        *experiment* files with parameters for different analyses. You may
        additionally keep a plain text file with a list of *subjects* named
        ``subjects.txt`` here. Many tools will default to these subjects if not
        otherwise instructed.

    project
        The *project* is the highest level of the hierarchy with which lyman
        thinks about your data. Broadly speaking, a project is just a common
        set of subjects that have been scanned on one or more experimental
        paradigms. This concept probably maps fairly directly to how you think
        about "projects" in your day-to-day life as a researcher.

    subject
        Subjects are people who contribute their time and the characteristics
        of their hydrogen resonance to our scientific endeavors. Once they are
        done with that, a *subject* is a set of data belonging to a single
        individual (possibly scanned in multiple sessions) who is provided with
        a unique identifier. Within the *data_dir*, a subject's data can be
        found at the highest level of the hierarchy stored in a directory that
        shares her ID. Within the *analysis_dir*, a subject is the level of
        hierarchy directly under *experiment*.

    workflow
        Although we use the term "workflow" in a rather general sense to
        describe a reproducible set of processing steps applied to some data,
        it also has a specific sense in the context of lyman tools. The
        ``run_fmri.py`` Nipype script that controls the processing of
        subject-level fMRI data is broken into four *workflows* (*preproc*,
        *model*, *reg*, and *ffx*), and this is the lowest level of granularity
        at which you can control the fMRI processing though this script. In the
        *analysis_dir* directory tree, *workflow* is the level of the hierarchy
        directly below *subject*.

    working_dir
        When the Nipype workflows execute, intermediate results are cached
        within the *working_dir*, which is defined in the ``project.py`` file
        in your *lyman directory*. When this directory exists, rerunning the
        ``run_fmri.py`` script will only re-execute stages that have
        had some change to their inputs since the last execution. This can be
        convenient if you are quickly iterating over some parameters. However,
        the intermediate storage can become very large, and lyman saves every
        needed output to a persistent location in the *analysis_dir*. For this
        reason, it is recommended that you set the ``remove_working_dir``
        option in the ``project.py`` file to ``True`` for normal use. This will
        delete the working cache after each successful execution of the
        ``run_fmri.py`` script.
