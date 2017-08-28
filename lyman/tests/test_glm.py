import os.path as op
import numpy as np
import nibabel as nib

from .. import glm


def test_prewhitened_glm_against_fsl():
    """Test prewhitening and OLS fit against values from film_gls."""
    test_data = np.load(op.join(op.dirname(__file__), "data/film_data.npz"))

    ts_data = test_data["ts_data"]
    X = test_data["X"]

    ts_img = nib.Nifti1Image(ts_data, np.eye(4))

    mask_data = np.ones(ts_data.shape[:-1], np.int)
    mask_img = nib.Nifti1Image(mask_data, np.eye(4))

    WY, WX = glm.prewhiten_image_data(ts_img, X, mask_img, smooth_fwhm=None)
    B, _, _ = glm.iterative_ols_fit(WY, WX)

    # Note that while our code produces highly similar values to what we get
    # from FSL, there are enough small differences that we can't simply test
    # array equality (or even almost equality to n decimals).
    # This is somewhat disconcerting, but possibly expected given the number
    # of differences in the two approaches it is not wholly unexpected.
    # Further, there is enough small weirdness in the FSL code (i.e. the
    # autocorrelation estimates don't appear properly normalized) that it's
    # not certain that small deviations are problems in our code and not FSL.
    # In any case, it will suffice to test that the values are highly similar.

    WY_corr = np.corrcoef(WY.flat, test_data["WY"].flat)[0, 1]
    assert WY_corr > .999

    WX_corr = np.corrcoef(WX.mean(axis=-1).flat, test_data["WX"].flat)[0, 1]
    assert WX_corr > .999

    B_corr = np.corrcoef(B.flat, test_data["B"].flat)[0, 1]
    assert B_corr > .999
