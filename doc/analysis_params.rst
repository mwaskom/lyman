.. _analysis_params:

Specifying analysis parameters
==============================

To control the execution of the lyman processing workflows, it is necessary to
provide information about different aspects of the dataset, image acquisition,
and experimental design. This information is generally communicated through text files stored in different locations. They are documented here.

Scan information
----------------

There must be a file called ``scans.yaml`` saved in the :term:`lyman_dir`. It
should contain specifiers for every :term:`subject`, :term:`session`, and
:term:`run` in the :term:`project`.  The expected structure of the file is complicated
to explain, although it is fairly straightforward when you see an example. The
file is essentially a set of nested dictionaries: a dictionary mapping
:term:`subject` names to a dictionary mapping :term:`session` ids to a
dictionary mapping :term:`experiment` names to a list of :term:`run` ids. That
is, something like this::

    subj01:
      sess01:
        exp_a: [run_1, run_2]
      sess02:
        exp_a: [run_1]
        exp_b: [run_1, run_2]
    subj02:
      sess01:
        exp_b: [run_1, run_3]

Note that the session and run identifiers can be any string: instead of
``sess01`` you could use a date and instead of ``run01`` you could use a time.

Project-level parameters
------------------------

Information that is consistent for the entire :term:`project` must be defined
in a file named ``project.py`` that is present in the :term:`lyman_dir`. Note
that this is a Python module that can define the following variables. Some
parameters have default values that will be used if the variable is not present
in the project file.

.. include:: traits/project.txt

Experiment-level parameters
---------------------------

Information that is consistent for an entire experiment must be defined in a
file named ``<experiment>.py`` that is present in the :term:`lyman_dir`.
Like the :term:`project` file, this should be a Python module that defines the
variables explained below. The :term:`experiment` file can also define
:term:`model` parameters that are documented further below. In this case, every
model associated with the experiment will use that value, unless it is
overridden in the specific model file.

.. include:: traits/experiment.txt

Model-level parameters
----------------------

Information that is specific to a particular model must be defined in a file
named ``<experiment>-<model>.py`` that is present in the :term:`lyman_dir`.
This is also a Python module that defines the variables listed below. Note that
as explained above, model-level parameters that are defined in the experiment
file will be used in all models associated with that experiment, although an
experiment-level model parameter can be overridden in a specific model file.

.. include:: traits/model.txt

Design information
------------------

Information that determines the structure of the design matrix must be defined
in a file named ``<model>.csv`` that is present for each subject in the
directory ``<data_dir>/<subject>/design/<model>.py``. This should should be a
CSV file that can be loaded into a pandas DataFrame. Each row in the file
should correspond to an event that will be modeled.

The design file must have the following columns: ``session``, ``run``,
``onset``, and ``condition``. The :term:`session` and :term:`run` identifiers
should correspond to the keys used in the ``scans.yaml`` file (see above). The
``onset`` information should define the time (in seconds, relative to the start
of each run) that the event occurred. The ``condition`` column should be a
string that defines the type of event; the unique condition values will become
columns in the design matrix. 

The design file may also have the following columns: ``duration`` and
``value``. If ``duration`` information is present, it determines the duration 
of the modeled event (in seconds) before convolution with the hemodynamic
response model. It defaults to ``0``, which specifies an impulse. If ``value``
information is present, it determines the amplitude of the modeled event before
convolution with the hemodynamic response model. It defaults to ``1``.
