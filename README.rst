Nipype fMRI Analysis Ecosystem
==============================

This repository contains a set of Python-based code for analyzing
functional MRI data using FSL and Freesurfer tools and Nipype.

Dependencies
------------

External
^^^^^^^^

- Freesurfer

- FSL

Python
^^^^^^

- Nipype and related dependencies

- Core scientific Python environment (best to use the Enthought Python Distribution)

- Nibabel

- rst2pdf

- Docutils 0.7 or 0.9 (rst2pdf does not work with 0.8)

Basic Workflow
--------------

- All stages of processing assume that your anatomical data have been
  processed in Freesurfer (recon-all)

- run_warp.py: perform anatomical normalization

- run_fmri.py: perform subject-level functional analyses

- run_group.py: perform basic group level mixed effects analyses

Note
----

Surface-based workflows are not currently operational, although
some surface information is used (e.g. for functional coregistration)

