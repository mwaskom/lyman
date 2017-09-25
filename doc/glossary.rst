A glossary of concepts
======================

Seeing how the parts of lyman fit together requires understanding several
hierarchies of concepts that structure the codebase, interactions with the
library, and the organization of raw and processed data on disk.

Dataset organization
--------------------

The first hierarchy of concepts pertains to the granularity of data:

.. glossary::

    subject
     An individual who participates in the project.

    session
     A continuous block of time that a subject was in the scanner.

    run
     A continuous acquisition of data from the scanner corresponding to a time
     series image.

Analysis parameterization
-------------------------

The second hierarchy of concepts pertains to the granularity of analysis
parameterization:

.. glossary::

    project
     Corresponds to the whole data set that any invocation of lyman will
     interact with.

    experiment
     Corresponds to a subset of the data that was collected with consistent
     acquisition parameters.

    model 
     Corresponds to the set of signal-processing parameters and elements of a
     linear model that are used to analyze an experiment's data in a particular
     way.

Directory structure
-------------------

.. glossary::

    lyman directory
     The location where files that define analysis parameters are stored.
     Should be identified with the ``LYMAN_DIR`` environment variable when
     executing the workflows.

    data directory
     The location where data that will be input into the lyman workflows are
     stored. Should also correspond to the Freesurfer ``SUBJECTS_DIR``.

    analysis directory
     The location where lyman will write output files for persistent storage
     after workflow execution.

    cache directory
     The location where lyman will write intermediate files for temporary
     storage during workflow execution.
