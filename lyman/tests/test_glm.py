import os.path as op
import numpy as np
import nibabel as nib

import pytest
from pytest import approx

from .. import glm


class TestLinearModel(object):

    @pytest.fixture
    def test_data(self):

        data_path = op.join(op.dirname(__file__), "data/film_data.npz")
        test_data_obj = np.load(data_path)
        test_data = dict(test_data_obj)

        ts_data = test_data["ts_data"]
        test_data["ts_img"] = nib.Nifti1Image(ts_data, np.eye(4))

        nx, ny, nz, n_tp = ts_data.shape
        n_vox = nx * ny * nz
        test_data["data_matrix_shape"] = n_tp, n_vox

        mask = np.ones(ts_data.shape[:-1], np.int)
        test_data["mask_img"] = nib.Nifti1Image(mask, np.eye(4))

        yield test_data
        test_data_obj.close()

    def test_image_prewhitening_outputs(self, test_data):

        ts_img = test_data["ts_img"]
        mask_img = test_data["mask_img"]
        X = test_data["X"]
        smooth_fwhm = None

        n_tp, n_vox = test_data["data_matrix_shape"]
        _, n_ev = X.shape

        # Test output shapes with the full data
        WY, WX = glm.prewhiten_image_data(ts_img, mask_img, X, smooth_fwhm)
        assert WY.shape == (n_tp, n_vox)
        assert WX.shape == (n_tp, n_ev, n_vox)

        # Test output shapes using a more restrictive mask
        n_mask = 10
        mask = np.zeros(mask_img.shape, np.int)
        mask.flat[:n_mask] = 1
        mask_img = nib.Nifti1Image(mask, np.eye(4))
        WY, WX = glm.prewhiten_image_data(ts_img, mask_img, X, smooth_fwhm)
        assert WY.shape == (n_tp, n_mask)
        assert WX.shape == (n_tp, n_ev, n_mask)

        # Smoke test smoothing
        smooth_fwhm = 2
        WY, WX = glm.prewhiten_image_data(ts_img, mask_img, X, smooth_fwhm)
        assert WY.shape == (n_tp, n_mask)
        assert WX.shape == (n_tp, n_ev, n_mask)

    def test_residual_autocorrelation_outputs(self, test_data):

        ts_data = test_data["ts_data"]
        n_tp, n_vox = test_data["data_matrix_shape"]
        Y = ts_data.reshape(n_vox, n_tp).T
        X = test_data["X"]

        # Test outputs with default Tukey window taper
        auto_tukey_m = glm.default_tukey_window(n_tp)
        acf = glm.estimate_residual_autocorrelation(Y, X)
        assert acf.shape == (auto_tukey_m, n_vox)

        # Test normalization of autocorrelation estimates
        assert np.array_equal(acf[0], np.ones(n_vox))
        assert acf[1:].max() < 1
        assert acf.min() > -1

        # Test outputs with specifed tukey taper size
        tukey_m = 10
        acf = glm.estimate_residual_autocorrelation(Y, X, tukey_m)
        assert acf.shape == (tukey_m, n_vox)

    def test_prewhitened_glm_against_fsl(self, test_data):

        ts_img = test_data["ts_img"]
        mask_img = test_data["mask_img"]
        n_tp, n_vox = test_data["data_matrix_shape"]
        Y = test_data["ts_data"].reshape(n_vox, n_tp).T
        X = test_data["X"]
        smooth_fwhm = None

        acf = glm.estimate_residual_autocorrelation(Y, X)
        WY, WX = glm.prewhiten_image_data(ts_img, mask_img, X, smooth_fwhm)
        B, _, _, _ = glm.iterative_ols_fit(WY, WX)

        # Note that while our code produces highly similar values to what we
        # get from FSL, there are enough small differences that we can't simply
        # test array equality (or even almost equality to n decimals).  This is
        # somewhat disconcerting, but possibly expected given the number of
        # differences in the two implementations it is not wholly unexpected.
        # Further, there is enough small weirdness in the FSL code (i.e. the
        # autocorrelation estimates don't appear properly normalized) that it's
        # not certain that small deviations are problems in our code and not
        # FSL. In any case, it will suffice to test that the values are highly
        # similar.

        acf_corr = np.corrcoef(acf.flat, test_data["acf"].flat)[0, 1]
        assert acf_corr > .999

        WY_corr = np.corrcoef(WY.flat, test_data["WY"].flat)[0, 1]
        assert WY_corr > .999

        WX_mean = WX.mean(axis=-1)
        WX_corr = np.corrcoef(WX_mean.flat, test_data["WX"].flat)[0, 1]
        assert WX_corr > .999

        B_corr = np.corrcoef(B.flat, test_data["B"].flat)[0, 1]
        assert B_corr > .999


class TestHighpassFilter(object):

    @pytest.fixture
    def test_data(self):

        data_path = op.join(op.dirname(__file__), "data/hpf_data.npz")
        test_data = np.load(data_path)
        yield test_data
        test_data.close()

    def test_highpass_filter_against_fsl(self, test_data):
        """Test highpass filter performance against fslmaths."""
        filt = glm.highpass_filter(test_data["orig"], test_data["cutoff"])

        # Note that similar to the prewhitening, our code doesn't achieve exact
        # parity with FSL. In the case of the hpf, this seems to be a recent
        # development and FSL has maybe added some code to handle edge effects,
        # as the differences occur at the beginning and end of the time series.
        # In any case, we will test that the results are highly similar, and
        # test basic attributes of hpf functionality elsewhere.

        corr = np.corrcoef(filt.flat, test_data["filt"].flat)[0, 1]
        assert corr > .999


class TestFixedEffectsContrasts(object):

    @pytest.fixture
    def test_data(self):

        data_path = op.join(op.dirname(__file__), "data/ffx_data.npz")
        test_data = np.load(data_path)
        yield test_data
        test_data.close()

    def test_fixed_effects_contrasts_against_fsl(self, test_data):
        """Test higher-level fixed effects estimation against flameo."""
        from numpy import sqrt
        con, var = test_data["con"], test_data["var"]
        con_ffx, var_ffx, t_ffx = glm.contrast_fixed_effects(con, var)

        assert con_ffx == approx(test_data["con_ffx"], rel=1e-3)
        assert sqrt(var_ffx) == approx(sqrt(test_data["var_ffx"]), rel=1e-3)
        assert t_ffx == approx(test_data["t_ffx"], rel=1e-3)
