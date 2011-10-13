




preproc_report_template = """\
********************
Preprocessing Report
********************

**Report generated:** %(now)s

**Subject ID:** %(subject_id)s

Input Timeseries
================

**Source file:**

%(timeseries_file)s

**Original path:** 

%(orig_timeseries_path)s

**Image Dimensions:** %(image_dimensions)s

**Timepoints:** %(image_timepoints)d

Motion Correction Target
========================

.. image:: %(example_func_slices)s
    :width: 6.5in

Mean Functional Image and Brain Mask
====================================

.. image:: %(mean_func_slices)s
    :width: 6.5in

Whole-brain Intensity and Outliers
==================================

.. image:: %(intensity_plot)s
    :width: 6.5in

**Total outliers frames:** %(n_outliers)d

Maximum RMS Motion
------------------

**Absolute:** %(max_abs_motion)s mm

**Relative:** %(max_rel_motion)s mm

Motion Plots
============

.. image:: %(displacement_plot)s
    :width: 6.5in

.. image:: %(translation_plot)s
    :width: 6.5in

.. image:: %(rotation_plot)s
    :width: 6.5in

Functional to Structural Coregistration
=======================================

.. image:: %(func_to_anat_slices)s
    :width: 5.5in

**Final optimization cost:** %(func_to_anat_cost)s

"""
