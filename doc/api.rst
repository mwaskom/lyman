.. _api_ref:

API reference
=============

.. currentmodule:: lyman

:mod:`lyman.frontend`: Front-end interface
------------------------------------------

.. automodule:: lyman.frontend
    :no-members:
    :no-inherited-members:

.. currentmodule:: lyman

Interface functions
^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: api

    frontend.info
    frontend.subjects
    frontend.execute

Information classes
^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: api
    :template: class.rst

    frontend.LymanInfo
    frontend.ProjectInfo
    frontend.ExperimentInfo
    frontend.ModelInfo

:mod:`lyman.glm`: General linear modeling
-----------------------------------------

.. automodule:: lyman.glm
    :no-members:
    :no-inherited-members:

.. currentmodule:: lyman

HRF models
^^^^^^^^^^

.. autosummary::
    :toctree: api
    :template: class.rst

    glm.HRFModel
    glm.GammaHRF
    glm.GammaBasis
    glm.FIRBasis
    glm.IdentityHRF

Design construction
^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: api

    glm.build_design_matrix
    glm.condition_to_regressors
    glm.contrast_matrix

Model estimation
^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: api

    glm.prewhiten_image_data
    glm.estimate_residual_autocorrelation
    glm.iterative_ols_fit
    glm.iterative_contrast_estimation
    glm.contrast_fixed_effects

Temporal filtering
^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: api

    glm.highpass_filter_matrix
    glm.highpass_filter


:mod:`lyman.signals`: Signal processing
---------------------------------------

.. automodule:: lyman.signals
    :no-members:
    :no-inherited-members:

.. currentmodule:: lyman

Time series transformation and diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: api

    signals.percent_change
    signals.detrend
    signals.pca_transform
    signals.identify_noisy_voxels
    signals.cv

Spatial filtering
^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: api

    signals.smooth_volume
    signals.smooth_segmentation
    signals.smooth_surface
    signals.smoothing_matrix
    signals.voxel_sigmas

:mod:`lyman.surface`: Surface mesh operations
---------------------------------------------

.. automodule:: lyman.surface
    :no-members:
    :no-inherited-members:

.. currentmodule:: lyman

Geodesic distance measurement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: api
    :template: class.rst

    surface.SurfaceMeasure

Data representation conversion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: api

    surface.vol_to_surf

:mod:`lyman.utils`: Utilities
-----------------------------

.. automodule:: lyman.utils
    :no-members:
    :no-inherited-members:

.. currentmodule:: lyman

Custom Nipype interfaces
^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: api
    :template: class.rst

    utils.LymanInterface
    utils.SaveInfo

Data representation conversion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: api

    utils.image_to_matrix
    utils.matrix_to_image
    utils.check_mask

:mod:`lyman.visualization`: Data visualization
----------------------------------------------

.. automodule:: lyman.visualizations
    :no-members:
    :no-inherited-members:

.. currentmodule:: lyman

Classes for image representation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: api
    :template: class.rst

    visualizations.Mosaic
    visualizations.CarpetPlot

Model visualization
^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: api

    visualizations.plot_design_matrix
    visualizations.plot_nuisance_variables
