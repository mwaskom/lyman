from __future__ import division
import os.path as op
from itertools import product
import numpy as np
import pandas as pd
import nibabel as nib

from scipy.signal import periodogram

import pytest
from pytest import approx

from .. import glm


def assert_highly_correlated(a, b, thresh=.999):
    corr = np.corrcoef(a.flat, b.flat)[0, 1]
    assert corr > thresh


class TestHRFs(object):

    @pytest.fixture
    def random(self):

        seed = sum(map(ord, "hrfs"))
        return np.random.RandomState(seed)

    @pytest.fixture
    def randn_input(self, random):

        return random.randn(100)

    @pytest.fixture
    def delta_input(self, random):

        return random.randn(100) > .05

    def test_base(self):

        with pytest.raises(NotImplementedError):
            glm.HRFModel().transform(None)

    def test_identity(self, randn_input):

        output = glm.IdentityHRF().transform(randn_input)
        assert np.array_equal(output, randn_input)

    @pytest.mark.parametrize(
        "res,duration",
        product((10, 20), (24, 42)))
    def test_gamma_hrf_kernel_size(self, res, duration):

        hrf = glm.GammaHRF(res=res, duration=duration)
        assert len(hrf.kernel) == res * duration

    def test_kernel_normalization(self):

        hrf = glm.GammaHRF()
        assert hrf.kernel.sum() == pytest.approx(1)

    def test_undershoot(self, random):

        double = glm.GammaHRF()
        assert double.kernel.min() < 0

        single = glm.GammaHRF(ratio=0)
        assert single.kernel.min() >= 0

    def test_gamma_hrf_output_type(self, random, randn_input):

        a = np.asarray(randn_input)
        s = pd.Series(randn_input, name="event")

        hrf = glm.GammaHRF()
        a_out = hrf.transform(a)
        s_out = hrf.transform(s)

        assert isinstance(a_out, np.ndarray)
        assert isinstance(s_out, pd.Series)

    def test_gamma_hrf_convolution(self, random, randn_input):

        hrf = glm.GammaHRF()
        convolution = np.convolve(randn_input, hrf.kernel)[:len(randn_input)]
        assert hrf.transform(randn_input) == pytest.approx(convolution)

    @pytest.mark.parametrize("name", ["event", 180])
    def test_gamma_hrf_output_name(self, random, randn_input, name):

        s = pd.Series(randn_input, name=name)
        hrf = glm.GammaHRF()
        y = hrf.transform(s)
        assert y.name == str(name)

    def test_gamma_hrf_output_index(self, random, randn_input):

        n = len(randn_input)
        name = "event"
        idx = pd.Index(random.permutation(np.arange(n)))
        s = pd.Series(randn_input, idx, name=name)

        hrf = glm.GammaHRF()
        y = hrf.transform(s)
        assert y.index.equals(idx)

    @pytest.mark.parametrize("name", ["event", 180])
    def test_gamma_basis_output_names(self, random, randn_input, name):

        s = pd.Series(randn_input, name=name)
        hrf = glm.GammaBasis()
        y = hrf.transform(s)
        assert list(y.columns) == [f"{name}", f"{name}-dydt", f"{name}-dydw"]

        hrf = glm.GammaBasis(time_derivative=False)
        y = hrf.transform(s)
        assert list(y.columns) == [f"{name}", f"{name}-dydw"]

        hrf = glm.GammaBasis(disp_derivative=False)
        y = hrf.transform(s)
        assert list(y.columns) == [f"{name}", f"{name}-dydt"]

    def test_gamma_basis_output_index(self, random, randn_input):

        n = len(randn_input)
        name = "event"
        idx = pd.Index(random.permutation(np.arange(n)))
        s = pd.Series(randn_input, idx, name=name)

        hrf = glm.GammaBasis()
        y = hrf.transform(s)
        assert y.index.equals(idx)

    def test_gamma_basis_sum_squares(self):

        hrf = glm.GammaBasis()
        ss = np.sum(hrf.kernel ** 2, axis=0)
        assert np.allclose(np.diff(ss), 0)

    def test_gamma_basis_convolution_orthogonality(self, randn_input):

        hrf = glm.GammaBasis()
        y = hrf.transform(randn_input)
        yty = y.T @ y

        np.allclose(np.triu(yty, 1), 0)
        np.allclose(np.tril(yty, 1), 0)

    @pytest.mark.parametrize("n", [10, 14])
    def test_fir_matrix_size(self, delta_input, n):

        n_tp = len(delta_input)
        hrf = glm.FIRBasis(n)
        assert hrf.transform(delta_input).shape == (n_tp, n)

    @pytest.mark.parametrize("offset", [0, 2])
    def test_fir_matrix_values(self, delta_input, random, offset):

        n = 12
        hrf = glm.FIRBasis(n, offset)
        onsets, = np.nonzero(delta_input)

        delta_basis = hrf.transform(delta_input)

        for i in onsets:
            for j, row in zip(np.arange(n), delta_basis[(i - offset):]):
                assert row[j] == 1

        parametric_input = delta_input * random.randn(delta_input.size)
        parametric_basis = hrf.transform(parametric_input)

        for i in onsets:
            for j, row in zip(np.arange(n), parametric_basis[(i - offset):]):
                assert row[j] == parametric_input[i]

    def test_fir_matrix_output_type(self, delta_input):

        hrf = glm.FIRBasis(12)
        assert isinstance(hrf.transform(delta_input), np.ndarray)
        assert isinstance(hrf.transform(pd.Series(delta_input)), pd.DataFrame)

    def test_fir_basis_output_indices(self, random, delta_input):

        n = 24
        name = "event"
        idx = pd.Index(random.permutation(np.arange(delta_input.size)))
        cols = pd.Index([f"{name}_{i:02d}" for i in range(n)])
        s = pd.Series(delta_input, idx, name=name)

        hrf = glm.FIRBasis(n)
        basis = hrf.transform(s)

        assert basis.index.equals(idx)
        assert basis.columns.equals(cols)

    def test_fir_basis_suffix(self, delta_input):

        n = 12
        name = "event"
        idx = np.arange(delta_input.size)

        suffix = ":"
        hrf = glm.FIRBasis(n, suffix=suffix)
        s = pd.Series(delta_input, idx, name=name)
        basis = hrf.transform(s)

        cols = pd.Index([f"{name}{suffix}{i:02d}" for i in range(n)])
        assert basis.columns.equals(cols)


class TestDesignMatrix(object):

    @pytest.fixture
    def random(self):

        seed = sum(map(ord, "design_matrix"))
        return np.random.RandomState(seed)

    @pytest.fixture
    def conditions(self):

        conditions = pd.DataFrame(dict(
            condition=["a", "b", "a", "b"],
            onset=[0, 12, 24, 36],
            duration=[2, 2, 2, 2],
            value=[1, 1, 1, 1],
        ))
        return conditions

    @pytest.fixture
    def regressors(self, random):

        data = random.normal(2, 1, (48, 3))
        columns = ["x", "y", "z"]
        regressors = pd.DataFrame(data, columns=columns)
        return regressors

    @pytest.fixture
    def artifacts(self, random):

        return pd.Series(random.rand(48) < .1)

    @pytest.mark.parametrize(
        "n_tp,tr,time_deriv,disp_deriv",
        product((24, 28), (1, 2), (False, True), (False, True)))
    def test_design_shape_and_index(self, conditions, tr, n_tp,
                                    time_deriv, disp_deriv):

        X = glm.build_design_matrix(conditions,
                                    glm.GammaBasis(time_deriv, disp_deriv),
                                    n_tp=n_tp, tr=tr)

        n_cond = len(np.unique(conditions["condition"]))
        assert isinstance(X, pd.DataFrame)
        assert X.shape == (n_tp, n_cond * sum([1, time_deriv, disp_deriv]))

        tps = np.arange(0, n_tp * tr, tr)
        assert np.array_equal(X.index.values, tps)

    def test_design_contents(self, conditions):

        n_tp, tr = 48, 1
        X = glm.build_design_matrix(conditions, glm.IdentityHRF(),
                                    n_tp=n_tp, tr=tr, demean=False)

        expected_a = np.zeros(n_tp)
        expected_a[[0, 1, 24, 25]] = 1
        assert np.array_equal(X["a"].values, expected_a)

        n_tp, tr = 24, 2
        X = glm.build_design_matrix(conditions, glm.IdentityHRF(),
                                    n_tp=n_tp, tr=tr, demean=False)

        expected_a = np.zeros(n_tp)
        expected_a[[0 // tr, 24 // tr]] = 1
        assert isinstance(X, pd.DataFrame)
        assert np.array_equal(X["a"].values, expected_a)

    def test_design_regressors(self, conditions, regressors):

        cols = regressors.columns.tolist()

        X = glm.build_design_matrix(regressors=regressors)
        assert X.columns.tolist() == cols
        assert np.array_equal(X[cols], regressors - regressors.mean())

        X = glm.build_design_matrix(conditions, regressors=regressors)
        assert X.columns.tolist() == ["a", "b"] + cols
        assert np.array_equal(X[cols], regressors - regressors.mean())

        X = glm.build_design_matrix(regressors=regressors, demean=False)
        assert np.array_equal(X[cols], regressors)

    def test_design_artifacts(self, conditions, artifacts):

        cols = ["art{:02d}".format(i) for i in range(artifacts.sum())]

        X = glm.build_design_matrix(artifacts=artifacts)
        assert X.columns.tolist() == cols
        assert X.shape == (48, artifacts.sum())

        X = glm.build_design_matrix(conditions, artifacts=artifacts)
        assert X.columns.tolist() == ["a", "b"] + cols

    def test_design_fir(self, conditions):

        n = 12
        n_tp = 48
        X = glm.build_design_matrix(conditions, glm.FIRBasis(n), n_tp=n_tp,
                                    demean=False, res=1, shift=0)

        n_cond = conditions["condition"].unique().size
        assert X.shape == (n_tp, n * n_cond)

    def test_n_tp_errors(self, conditions, regressors, artifacts):

        with pytest.raises(ValueError):
            glm.build_design_matrix(regressors=regressors, n_tp=20)

        with pytest.raises(ValueError):
            glm.build_design_matrix(artifacts=artifacts, n_tp=20)

        with pytest.raises(ValueError):
            glm.build_design_matrix(regressors=regressors,
                                    artifacts=artifacts.iloc[:20])

    def test_hpf(self, conditions):

        n_tp = 48
        F = glm.highpass_filter_matrix(n_tp, 20, 1)
        X = glm.build_design_matrix(conditions, hpf_matrix=F, n_tp=n_tp)
        assert len(X) == len(F)

    def test_condition_defaults(self, conditions):

        conditions.loc[:, "duration"] = 0
        conditions.loc[:, "value"] = 1
        min_cols = ["condition", "onset"]

        X1 = glm.build_design_matrix(conditions, n_tp=48)
        X2 = glm.build_design_matrix(conditions[min_cols].copy(), n_tp=48)
        assert np.array_equal(X1.values, X2.values)

    @pytest.mark.parametrize(
        "tr,res",
        product((.1, .5, 1, 1.5, 2), (60, 30, 1)))
    def test_tr_and_sampling(self, tr, res):

        condition = pd.DataFrame(dict(
            onset=[2],
            value=[1],
            duration=[0],
        ))

        hrf = glm.GammaHRF(res=res)

        out, *_ = glm.condition_to_regressors(
            "test", condition, hrf, 30 / tr, tr, res, 0
        )

        assert 7 <= out.idxmax() <= 8


class TestContrastMatrix(object):

    @pytest.fixture
    def random(self):

        seed = sum(map(ord, "contrast_matrix"))
        return np.random.RandomState(seed)

    @pytest.fixture
    def design(self, random):

        cols = list("abcd")
        return pd.DataFrame(random.normal(0, 1, (48, 4)), columns=cols)

    def test_contrast_matrix(self, design):

        contrast = ("a", ["a"], [1])
        C = glm.contrast_matrix(contrast, design)
        assert np.array_equal(C, [1, 0, 0, 0])

        contrast = ("c", ["c"], [1])
        C = glm.contrast_matrix(contrast, design)
        assert np.array_equal(C, [0, 0, 1, 0])

        contrast = ("a-c", ["a", "c"], [1, -1])
        C = glm.contrast_matrix(contrast, design)
        assert np.array_equal(C, [1, 0, -1, 0])

        contrast = ("a-c", ["c", "a"], [-1, 1])
        C = glm.contrast_matrix(contrast, design)
        assert np.array_equal(C, [1, 0, -1, 0])

        contrast = ("a-bd", ["a", "b", "d"], [1, -.5, -.5])
        C = glm.contrast_matrix(contrast, design)
        assert np.array_equal(C, [1, -.5, 0, -.5])


class TestLinearModel(object):

    @pytest.fixture()
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

    def test_iterative_ols_fit(self, test_data):

        ts_img = test_data["ts_img"]
        mask_img = test_data["mask_img"]
        X = test_data["X"]
        smooth_fwhm = None

        n_tp, n_vox = test_data["data_matrix_shape"]
        _, n_ev = X.shape

        WY, WX = glm.prewhiten_image_data(ts_img, mask_img, X, smooth_fwhm)
        B, SS, XtXinv, E = glm.iterative_ols_fit(WY, WX)

        # Test output shapes
        assert B.shape == (n_vox, n_ev)
        assert SS.shape == (n_vox,)
        assert XtXinv.shape == (n_vox, n_ev, n_ev)
        assert E.shape == (n_tp, n_vox)

        # Test against numpy's basic least squares estimation
        for i in range(n_vox):
            B_i, _, _, _ = np.linalg.lstsq(WX[:, :, i], WY[:, i], rcond=None)
            assert B_i == approx(B[i])

        # Test XtXinv symmetry
        for XtXinv_i in XtXinv:
            assert np.array_equal(XtXinv_i, XtXinv_i.T)

    def test_iterative_contrast_estimation(self, test_data):

        ts_img = test_data["ts_img"]
        mask_img = test_data["mask_img"]
        n_tp, n_vox = test_data["data_matrix_shape"]
        X = test_data["X"]
        C = test_data["C"]
        smooth_fwhm = None

        n_tp, n_vox = test_data["data_matrix_shape"]
        n_con, n_ev = C.shape

        WY, WX = glm.prewhiten_image_data(ts_img, mask_img, X, smooth_fwhm)
        B, SS, XtXinv, _ = glm.iterative_ols_fit(WY, WX)
        G, V, T = glm.iterative_contrast_estimation(B, SS, XtXinv, C)

        # Test output shapes
        assert G.shape == (n_vox, n_con)
        assert V.shape == (n_vox, n_con)
        assert T.shape == (n_vox, n_con)

        # Test computation of contrast parameter estimates
        assert np.array_equal(G, np.dot(B, C.T))

        # Test that variances are all positive
        assert np.all(V > 0)

        # Test that t stats have the same sign as the effect sizes
        assert np.all(np.sign(T) == np.sign(G))

    def test_prewhitened_glm_against_fsl(self, test_data):

        ts_img = test_data["ts_img"]
        mask_img = test_data["mask_img"]
        n_tp, n_vox = test_data["data_matrix_shape"]
        Y = test_data["ts_data"].reshape(n_vox, n_tp).T
        X = test_data["X"]
        C = test_data["C"]
        smooth_fwhm = None

        acf = glm.estimate_residual_autocorrelation(Y - Y.mean(axis=0), X)
        WY, WX = glm.prewhiten_image_data(ts_img, mask_img, X, smooth_fwhm)
        WX_mean = WX.mean(axis=-1)
        B, SS, XtXinv, _ = glm.iterative_ols_fit(WY, WX)
        G, V, T = glm.iterative_contrast_estimation(B, SS, XtXinv, C)

        # Note that while our code produces highly similar values to what we
        # get from FSL, there are enough small differences that we can't simply
        # test array equality (or even almost equality to n decimals).  This is
        # somewhat disconcerting, but given the number of differences in the
        # two implementations it is not wholly unexpected.  Further, there is
        # enough small weirdness in the FSL code (i.e. the autocorrelation
        # estimates don't appear properly normalized) that it's not certain
        # that small deviations are problems in our code and not FSL. In any
        # case, it will suffice to test that the values are highly similar.

        # Test residual autocorrelation estimate
        assert_highly_correlated(acf, test_data["acf"])

        # Test prewhitened fMRI data
        assert_highly_correlated(WY, test_data["WY"])

        # Test (average) prewhitened design
        assert_highly_correlated(WX_mean, test_data["WX"])

        # Test model parameter estimates
        assert_highly_correlated(B, test_data["B"])

        # Test model error summary
        assert_highly_correlated(SS, test_data["SS"])

        # Test contrast of parameter estimates
        assert_highly_correlated(G, test_data["G"])

        # Test variance of contrast of parameter estimates
        assert_highly_correlated(V, test_data["V"])

        # Test contrast t statistics
        assert_highly_correlated(T, test_data["T"])


class TestHighpassFilter(object):

    @pytest.fixture
    def test_data(self):

        data_path = op.join(op.dirname(__file__), "data/hpf_data.npz")
        test_data_obj = np.load(data_path)
        test_data = dict(test_data_obj)

        n_tp = test_data["orig"].shape[-1]
        test_data["orig"] = test_data["orig"].reshape(-1, n_tp).T
        test_data["filt"] = test_data["filt"].reshape(-1, n_tp).T

        yield test_data
        test_data_obj.close()

    def test_highpass_filter_matrix(self, test_data):

        n_tp = 200
        cutoff = test_data["cutoff"]
        tr = test_data["tr"]

        F = glm.highpass_filter_matrix(n_tp, cutoff, tr)

        # Test filter shape
        assert F.shape == (n_tp, n_tp)

        # Test row normalization
        assert F.sum(axis=1) == approx(np.zeros(n_tp))

        # Test no filter
        cutoff = None
        F = glm.highpass_filter_matrix(n_tp, cutoff, tr)
        assert np.array_equal(F, np.eye(n_tp))

    def test_highpass_filter_spectrum(self, test_data):

        orig = test_data["orig"][:, 0]
        tr = test_data["tr"]

        cutoff_a = 80
        cutoff_b = 40

        fs, spec_orig = periodogram(orig, 1 / tr)

        filt_a = glm.highpass_filter(orig, cutoff_a, tr)
        filt_b = glm.highpass_filter(orig, cutoff_b, tr)

        _, spec_a = periodogram(filt_a, 1 / tr)
        _, spec_b = periodogram(filt_b, 1 / tr)

        stop_a = fs < (1 / cutoff_a)
        stop_b = fs < (1 / cutoff_b)

        # Test spectral density in the approximate stop-band
        assert spec_a[stop_a].sum() < spec_orig[stop_a].sum()
        assert spec_b[stop_b].sum() < spec_orig[stop_b].sum()

        # Test total spectral density with different cutoffs
        assert spec_b.sum() < spec_a.sum()

        assert True

    def test_highpass_filter(self, test_data):

        orig = test_data["orig"]
        cutoff = test_data["cutoff"]
        tr = test_data["tr"]

        n_tp, n_vox = orig.shape

        filt_2d = glm.highpass_filter(orig, cutoff, tr)
        assert filt_2d.shape == (n_tp, n_vox)

        filt_1d = glm.highpass_filter(orig[:, 0], cutoff, tr)
        assert filt_1d.shape == (n_tp,)

    def test_highpass_filter_copy(self, test_data):

        orig = test_data["orig"].copy()
        cutoff = test_data["cutoff"]

        filt = glm.highpass_filter(orig, cutoff, copy=True)
        assert not orig == approx(filt)

        filt = glm.highpass_filter(orig, cutoff, copy=False)
        assert orig == approx(filt)

    def test_highpass_filter_against_fsl(self, test_data):

        orig = test_data["orig"]
        cutoff = test_data["cutoff"]
        tr = test_data["tr"]

        filt = glm.highpass_filter(orig, cutoff, tr)

        # Note that similar to the prewhitening, our code doesn't achieve exact
        # parity with FSL. In the case of the hpf, this seems to be a recent
        # development and FSL has maybe added some code to handle edge effects,
        # as the differences occur at the beginning and end of the time series.
        # In any case, we will test that the results are highly similar, and
        # test basic attributes of hpf functionality elsewhere.

        assert_highly_correlated(filt, test_data["filt"])


class TestFixedEffectsContrasts(object):

    @pytest.fixture
    def test_data(self):

        data_path = op.join(op.dirname(__file__), "data/ffx_data.npz")
        test_data_obj = np.load(data_path)
        test_data = dict(test_data_obj)

        yield test_data
        test_data_obj.close()

    def test_fixed_effects_contrasts_outputs(self, test_data):

        con, var = test_data["con"], test_data["var"]
        n_vox, n_run = con.shape

        con_ffx, var_ffx, t_ffx = glm.contrast_fixed_effects(con, var)

        # Test output shapes
        assert con_ffx.shape == (n_vox,)
        assert var_ffx.shape == (n_vox,)
        assert t_ffx.shape == (n_vox,)

        # Test that variances are positive
        assert np.all(var_ffx > 0)

        # Test that t stats have the same sign as the effect sizes
        assert np.all(np.sign(t_ffx) == np.sign(con_ffx))

    def test_fixed_effects_contrasts_against_fsl(self, test_data):

        from numpy import sqrt
        con, var = test_data["con"], test_data["var"]
        con_ffx, var_ffx, t_ffx = glm.contrast_fixed_effects(con, var)

        assert con_ffx == approx(test_data["con_ffx"], rel=1e-3)
        assert sqrt(var_ffx) == approx(sqrt(test_data["var_ffx"]), rel=1e-3)
        assert t_ffx == approx(test_data["t_ffx"], rel=1e-3)
