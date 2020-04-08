from itertools import product
import numpy as np
import scipy.signal as scipy_signal
from scipy import sparse, stats
import nibabel as nib

import pytest

from .. import signals, surface


class TestSignals(object):

    @pytest.fixture(scope="class")
    def random(self):
        seed = sum(map(ord, "signals"))
        return np.random.RandomState(seed)

    @pytest.mark.parametrize("axis", [0, 1])
    def test_detrend(self, random, axis):

        x = random.normal(2, 1, (40, 20))
        x_out = signals.detrend(x, axis=axis)
        x_out_scipy = scipy_signal.detrend(x, axis=axis)
        assert np.array_equal(x_out, x_out_scipy)

    @pytest.mark.parametrize("axis", [0, 1])
    def test_detrend_replace_mean(self, random, axis):

        x = random.normal(2, 1, (40, 20))
        x_mean = x.mean(axis=axis)
        x_out = signals.detrend(x, axis=axis, replace_mean=True)
        x_out_mean = x_out.mean(axis=axis)
        assert x_mean == pytest.approx(x_out_mean)

    @pytest.mark.parametrize(
        "axis, detrend, keepdims, ddof",
        product((0, 1), (False, True), (False, True), (0, 1)))
    def test_cv(self, random, axis, detrend, keepdims, ddof):

        kws = dict(axis=axis, keepdims=keepdims)
        x = random.normal(2, 1, (40, 20))

        cv = signals.cv(x, detrend=detrend, ddof=ddof, **kws)

        m = x.mean(**kws)
        if detrend:
            x = scipy_signal.detrend(x, axis=axis)
        s = x.std(ddof=ddof, **kws)

        assert cv == pytest.approx(s / m)

    @pytest.mark.parametrize(
        "detrend, keepdims, ddof",
        product((False, True), (False, True), (0, 1)))
    def test_cv_mask(self, random, detrend, keepdims, ddof):

        kws = dict(axis=-1, keepdims=keepdims)
        x = random.normal(2, 1, (40, 20))
        mask = random.uniform(0, 1, x.shape[0]) > .3

        cv = signals.cv(x, mask=mask, detrend=detrend, ddof=ddof, **kws)

        m = x.mean(**kws)
        if detrend:
            x = scipy_signal.detrend(x, axis=-1)
        s = x.std(ddof=ddof, **kws)

        cv_by_hand = np.zeros(mask.shape, np.float)
        if keepdims:
            cv_by_hand = np.expand_dims(cv_by_hand, axis=-1)
        cv_by_hand[mask] = (s / m)[mask]

        assert cv == pytest.approx(cv_by_hand)

        with pytest.raises(ValueError):
            cv = signals.cv(x, mask=mask, axis=0)

    def test_percent_change(self, random):

        x = [95, 95, 110, 100]
        x_pch = signals.percent_change(x)
        expected = [-5, -5, 10, 0]
        assert x_pch == pytest.approx(expected)

        x = np.c_[x, x]
        x_pch = signals.percent_change(x, axis=0)
        expected = np.c_[expected, expected]

        assert x_pch == pytest.approx(expected)

    def test_identify_noisy_voxels(self, random):

        shape = 5, 5, 5, 100
        affine = np.eye(4)
        affine[:3, :3] *= 2

        data = random.normal(100, 5, shape)
        data[2, 2, (2, 3)] = random.normal(100, 10, (2, shape[-1]))
        data_img = nib.Nifti1Image(data, affine)

        mask = np.zeros(shape[:-1], np.int)
        mask[:, :, :3] = 1
        mask_img = nib.Nifti1Image(mask, affine)

        noise_img = signals.identify_noisy_voxels(data_img, mask_img)
        noise = noise_img.get_fdata()

        assert noise.shape == shape[:-1]
        assert np.array_equal(noise_img.affine, affine)

        assert noise.sum() == 1
        assert noise[2, 2, 2] == 1

    def test_smooth_volume_kernel_size(self, random):

        shape = 12, 8, 4
        data_img = nib.Nifti1Image(random.normal(0, 1, shape), np.eye(4))

        std_0 = data_img.get_fdata().std()
        std_2 = signals.smooth_volume(data_img, 2).get_fdata().std()
        std_4 = signals.smooth_volume(data_img, 4).get_fdata().std()

        assert std_4 < std_2 < std_0

    def test_smooth_volume_no_smoothing(self, random):

        shape = 12, 8, 4
        orig_data = random.normal(0, 1, shape)
        orig_img = nib.Nifti1Image(orig_data, np.eye(4))
        new_data = signals.smooth_volume(orig_img, fwhm=None,).get_fdata()

        assert np.array_equal(orig_data, new_data)

    def test_smooth_volume_mask(self, random):

        shape = 12, 8, 4
        data_img = nib.Nifti1Image(random.normal(0, 1, shape), np.eye(4))

        mask = random.choice([False, True], shape)
        mask_img = nib.Nifti1Image(mask.astype(np.int), np.eye(4))

        out = signals.smooth_volume(data_img, 2, mask_img).get_fdata()

        assert out[mask].all()
        assert not out[~mask].any()

    def test_smooth_volume_noise(self, random):

        shape = 12, 8, 4, 100
        data = random.normal(0, 1, shape)

        noise = np.zeros(shape[:-1], np.bool)
        noise[6, 4, 2] = True
        data[noise] = random.normal(0, 10, shape[-1])

        data_img = nib.Nifti1Image(data, np.eye(4))
        noise_img = nib.Nifti1Image(noise.astype(np.int), np.eye(4))

        out_noisy = signals.smooth_volume(data_img, 2)
        out_clean = signals.smooth_volume(data_img, 2, noise_img=noise_img)

        noisy_std = out_noisy.get_fdata()[noise].std()
        clean_std = out_clean.get_fdata()[noise].std()

        assert clean_std < noisy_std

    def test_smooth_volume_inplace(self, random):

        shape = 12, 8, 4
        data = random.normal(0, 1, shape).astype(np.float32)
        data_img = nib.Nifti1Image(data, np.eye(4))
        out_img = signals.smooth_volume(data_img, 4, inplace=True)
        assert np.array_equal(data_img.get_fdata(), out_img.get_fdata())

    def test_smooth_segmentation(self, random):

        shape = 12, 8, 4
        seg = random.randint(0, 4, shape)
        data = random.uniform(0, 1, shape)
        data[seg == 2] += 3

        data_img = nib.Nifti1Image(data, np.eye(4))
        seg_img = nib.Nifti1Image(seg, np.eye(4))

        out_img = signals.smooth_segmentation(data_img, seg_img, 4)
        assert out_img.get_fdata()[seg != 2].max() < 1

    def test_smooth_segmentation_inplace(self, random):

        shape = 12, 8, 4

        data = random.normal(0, 1, shape)
        seg = random.randint(0, 4, shape)

        data_img = nib.Nifti1Image(data, np.eye(4))
        seg_img = nib.Nifti1Image(seg, np.eye(4))

        out_img = signals.smooth_segmentation(data_img, seg_img, 4,
                                              inplace=True)
        assert np.array_equal(out_img.get_fdata(), data.astype(np.float))

    def test_smoothing_matrix(self, meshdata):

        sm = surface.SurfaceMeasure(meshdata["verts"], meshdata["faces"])
        vertids = np.arange(sm.n_v)
        n_vox = len(vertids)
        fwhm = 4

        # Test basics of the smoothing weight matrix output
        S = signals.smoothing_matrix(sm, vertids, fwhm)
        assert isinstance(S, sparse.csr_matrix)
        assert S.shape == (n_vox, n_vox)
        assert S.sum(axis=1) == pytest.approx(np.ones((n_vox, 1)))

        for i, row_ws in enumerate(S.toarray()):
            row_ds = sm(i)
            row_ds = np.array([row_ds[j] for j in vertids])
            assert np.array_equal((1 / row_ws).argsort(), row_ds.argsort())

        # Test the computed weights in one matrix
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        norm = stats.norm(0, sigma)
        distmap = sm(0)
        w = np.array([norm.pdf(d) for v, d in distmap.items()])
        w /= w.sum()
        s = S[0].toarray().squeeze()
        assert s == pytest.approx(w)

        # Lightly test the rest of the weights in the matrices
        S1 = signals.smoothing_matrix(sm, vertids, 1).toarray()
        S2 = signals.smoothing_matrix(sm, vertids, 2).toarray()
        assert np.all(np.diag(S1, 0) > np.diag(S2, 0))
        assert np.all(np.diag(S1, 1) < np.diag(S2, 1))

        # Test exclusion of noisy voxels
        noise = np.zeros(sm.n_v, np.int)
        noise[0] = 1
        S = signals.smoothing_matrix(sm, vertids, 1, noise).toarray()
        assert S.shape == (n_vox, n_vox)
        assert S[:, 0] == pytest.approx(np.zeros(n_vox))

        # Test bad input
        with pytest.raises(ValueError):
            signals.smoothing_matrix(sm, vertids, 0)

        # Test avoidance of infinite loop
        verts = meshdata["verts"].copy()
        verts[0] += 100
        sm = surface.SurfaceMeasure(verts, meshdata["faces"])
        with pytest.raises(RuntimeError):
            signals.smoothing_matrix(sm, vertids, 1)

        # Test null smoothing
        S = signals.smoothing_matrix(sm, vertids, None)
        assert np.array_equal(S.toarray(), np.eye(n_vox))

    def test_smooth_surface(self, random, meshdata):

        shape = 4, 3, 2
        affine = np.eye(4)
        fwhm = 4

        n_v = len(meshdata["verts"])
        subject = meshdata["subj"]
        surf = meshdata["surf"]

        data = random.normal(10, 2, shape).astype(np.float32)
        data_img = nib.Nifti1Image(data, affine)

        surf_vox = random.choice(np.arange(np.product(shape)), n_v, False)
        vertvol = np.full(shape + (2,), -1, np.int)
        vertvol.flat[surf_vox] = np.arange(n_v)
        vert_img = nib.Nifti1Image(vertvol, affine)

        ribbon = (vertvol > -1).any(axis=-1)

        out_img = signals.smooth_surface(
            data_img, vert_img, fwhm, subject, surf,
        )
        out_data = out_img.get_fdata()

        assert out_data[ribbon].std() < data[ribbon].std()
        assert np.array_equal(out_data[~ribbon], data[~ribbon])

        shape = 4, 3, 2, 10

        data = random.normal(10, 2, shape)
        data_img = nib.Nifti1Image(data, affine)

        out_img = signals.smooth_surface(
            data_img, vert_img, fwhm, subject, surf,
        )
        out_data = out_img.get_fdata()

        assert out_data[ribbon].std() < data[ribbon].std()
        assert np.array_equal(out_data[~ribbon], data[~ribbon])

        n_tp = shape[-1]
        noise_mask = (vertvol == vertvol.max()).any(axis=-1)
        data[noise_mask] = random.normal(10, 10, n_tp)
        noise_img = nib.Nifti1Image(noise_mask.astype(int), affine)

        noise_out_img = signals.smooth_surface(
            data_img, vert_img, fwhm, subject, surf,
        )
        clean_out_img = signals.smooth_surface(
            data_img, vert_img, fwhm, subject, surf, noise_img
        )

        noise_sd = noise_out_img.get_fdata()[noise_mask].std()
        clean_sd = clean_out_img.get_fdata()[noise_mask].std()

        assert clean_sd < noise_sd

        out_img = signals.smooth_surface(
            data_img, vert_img, fwhm, subject, surf, inplace=True
        )

        assert np.array_equal(out_img.get_fdata(), data)

        with pytest.raises(ValueError):
            vert_img = nib.Nifti1Image(vertvol[..., 0], affine)
            out_img = signals.smooth_surface(
                data_img, vert_img, fwhm, subject, surf,
            )

    def test_load_float_maybe_inplace(self, random):

        dtype = np.float32
        input_data = random.randn(10).astype(dtype)
        img = nib.Nifti1Image(input_data, np.eye(4))

        data = signals._load_float_data_maybe_copy(img, True)
        assert data.dtype == dtype
        assert data is input_data

        data = signals._load_float_data_maybe_copy(img, False)
        assert data.dtype == dtype
        assert data is not input_data

        data = random.randint(0, 1, 10)
        img = nib.Nifti1Image(data, np.eye(4))
        data = signals._load_float_data_maybe_copy(img, False)
        assert data.dtype == np.float

        with pytest.raises(ValueError):
            data = signals._load_float_data_maybe_copy(img, True)

    def test_pca_transform(self):

        data = np.array([[5.1, 3.5, 1.4, 0.2],
                         [4.9, 3., 1.4, 0.2],
                         [4.7, 3.2, 1.3, 0.2],
                         [4.6, 3.1, 1.5, 0.2],
                         [5., 3.6, 1.4, 0.2],
                         [5.4, 3.9, 1.7, 0.4],
                         [4.6, 3.4, 1.4, 0.3],
                         [5., 3.4, 1.5, 0.2],
                         [4.4, 2.9, 1.4, 0.2],
                         [4.9, 3.1, 1.5, 0.1],
                         [5.4, 3.7, 1.5, 0.2],
                         [4.8, 3.4, 1.6, 0.2],
                         [4.8, 3., 1.4, 0.1],
                         [4.3, 3., 1.1, 0.1],
                         [5.8, 4., 1.2, 0.2],
                         [5.7, 4.4, 1.5, 0.4],
                         [5.4, 3.9, 1.3, 0.4],
                         [5.1, 3.5, 1.4, 0.3],
                         [5.7, 3.8, 1.7, 0.3],
                         [5.1, 3.8, 1.5, 0.3]])

        out = signals.pca_transform(data)
        assert out.shape == data.shape

        keep = 2
        out = signals.pca_transform(data, keep)
        assert out.shape == (data.shape[0], keep)

        # Expected array computed using scikit-learn
        out_expected = np.array([[9.28962685e-02, 1.38021850e-01],
                                 [-7.47325708e-01, 1.47550080e+00],
                                 [-7.73909135e-01, -4.20352896e-01],
                                 [-9.88476987e-01, 8.92756140e-02],
                                 [8.66479137e-02, -6.89381702e-01],
                                 [1.02016565e+00, 7.72298294e-02],
                                 [-6.28554855e-01, -1.57110749e+00],
                                 [-1.35758302e-01, 4.06603766e-01],
                                 [-1.48804628e+00, -9.67003378e-02],
                                 [-6.34261973e-01, 1.42600616e+00],
                                 [7.16960384e-01, 7.23931917e-01],
                                 [-3.70661227e-01, -1.52217871e-01],
                                 [-8.91089979e-01, 1.21249489e+00],
                                 [-1.55582424e+00, -1.51031547e+00],
                                 [1.52741892e+00, 3.18120286e-01],
                                 [1.95670901e+00, -1.34240820e+00],
                                 [9.63818782e-01, -8.86956548e-01],
                                 [1.12165719e-01, 1.09364858e-03],
                                 [1.25613420e+00, 1.84142981e+00],
                                 [4.80991836e-01, -1.04026807e+00]])

        out = signals.pca_transform(data, keep)
        assert out == pytest.approx(out_expected)

        out = signals.pca_transform(data, whiten=False)
        col_var = np.var(out, axis=0)
        assert list(np.argsort(col_var)) == [3, 2, 1, 0]
