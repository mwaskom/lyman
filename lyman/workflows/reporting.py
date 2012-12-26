preproc_report_template = """\
********************
Preprocessing Report
********************

**Report generated:** %(now)s

**Subject ID:** %(subject_id)s

Input Timeseries
----------------

**Source file:**

%(timeseries_file)s

**Original path:**

%(orig_timeseries_path)s

**Image Dimensions:** %(image_dimensions)s

**Timepoints:** %(image_timepoints)d

Motion Correction Target
------------------------

.. image:: %(example_func_slices)s
    :width: 6.5in

Mean Functional Image and Brain Mask
------------------------------------

.. image:: %(mean_func_slices)s
    :width: 6.5in

Whole-brain Intensity and Outliers
----------------------------------

.. image:: %(intensity_plot)s
    :width: 6.5in

**Total outliers frames:** %(n_outliers)d

RMS Motion
^^^^^^^^^^

**Max Absolute:** %(max_abs_motion)s mm

**Max Relative:** %(max_rel_motion)s mm

**Total Motion:** %(total_motion)s mm

Motion Plots
------------

.. image:: %(realignment_plot)s

Functional to Structural Coregistration
---------------------------------------

.. image:: %(func_to_anat_slices)s
    :width: 5.5in

**Final optimization cost:** %(func_to_anat_cost)s

"""

model_report_template = """\

***********************
Timeseries Model Report
***********************

**Report generated:** %(now)s

**Subject ID:** %(subject_id)s

Model Design
------------

Design Matrix
^^^^^^^^^^^^^

.. image:: %(design_image)s
    :width: 6.5in

Design Correlation
^^^^^^^^^^^^^^^^^^

.. image:: %(design_corr)s
    :width: 6.5in

Residual Map
^^^^^^^^^^^^

.. image:: %(residual)s
    :width: 6.5in

Zstat Maps
----------
"""

ffx_report_template = """\

**************************
Fixed Effects Model Report
**************************

**Report generated:** %(now)s

**Subject ID:** %(subject_id)s

**Contrast:** %(contrast_name)s

Mask Overlap
------------

.. image:: %(mask_png)s
    :width: 6.5in

Zstat Maps
----------

.. image:: %(zstat_png)s
    :width: 6.5in

"""

mfx_report_template = """

**************************
Mixed Effects Model Report
**************************

**Report generated:** %(now)s

**First-level contrast:** %(l1_contrast)s

**Number of subjects:** %(n_subs)d

**Subjects included:** %(subject_list)s

Brain Mask
----------

.. image:: %(mask_png)s
    :width: 6.5in

Zstat Maps
----------
"""
