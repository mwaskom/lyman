.. _issues:

Known Issues
============

Here is a list of some known issues in lyman that can lead to incorrect results and require particular care.

**Missing events:** Lyman is currently compatible with designs where some
events do not happen in all runs (e.g., if you are modeling error trials for a
task with high accuracy). The higher-level "contrasts" for the test of those
effects will be handled properly. However, there is a tricky issue when those
events are included in subtraction contrasts. Because the parameter map is full
of zeros, the contrast of (A - B) just looks like the map of A, which slips
past the downstream test for missing events. This will eventually be fixed, but
it will require some substantial changes to the inner mechanics of the modeling
workflow. A workaround for the meantime would be to manually zero out the
varcope image for the bad contrast (using fslmaths, or similar), which will
cause that contrast to be identified as "missing" in the fixed effects
workflow.  
