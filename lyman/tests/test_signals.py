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
        noise = noise_img.get_data()

        assert noise.shape == shape[:-1]
        assert np.array_equal(noise_img.affine, affine)

        assert noise.sum() == 1
        assert noise[2, 2, 2] == 1

    def test_smooth_volume_kernel_size(self, random):

        shape = 12, 8, 4
        data_img = nib.Nifti1Image(random.normal(0, 1, shape), np.eye(4))

        std_0 = data_img.get_data().std()
        std_2 = signals.smooth_volume(data_img, 2).get_data().std()
        std_4 = signals.smooth_volume(data_img, 4).get_data().std()

        assert std_4 < std_2 < std_0

    def test_smooth_volume_mask(self, random):

        shape = 12, 8, 4
        data_img = nib.Nifti1Image(random.normal(0, 1, shape), np.eye(4))

        mask = random.choice([False, True], shape)
        mask_img = nib.Nifti1Image(mask.astype(np.int), np.eye(4))

        out = signals.smooth_volume(data_img, 2, mask_img).get_data()

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

        noisy_std = out_noisy.get_data()[noise].std()
        clean_std = out_clean.get_data()[noise].std()

        assert clean_std < noisy_std

    def test_smooth_volume_inplace(self, random):

        shape = 12, 8, 4
        data_img = nib.Nifti1Image(random.normal(0, 1, shape), np.eye(4))
        out_img = signals.smooth_volume(data_img, 4, inplace=True)
        assert np.array_equal(data_img.get_data(), out_img.get_data())

    def test_smooth_segmentation(self, random):

        shape = 12, 8, 4
        data = seg = random.randint(0, 4, shape)

        data_img = nib.Nifti1Image(data, np.eye(4))
        seg_img = nib.Nifti1Image(seg, np.eye(4))

        out_img = signals.smooth_segmentation(data_img, 4, seg_img)
        assert out_img.get_data() == pytest.approx(data.astype(np.float))

    def test_smooth_segmentation_inplace(self, random):

        shape = 12, 8, 4

        data = random.normal(0, 1, shape)
        seg = random.randint(0, 4, shape)

        data_img = nib.Nifti1Image(data, np.eye(4))
        seg_img = nib.Nifti1Image(seg, np.eye(4))

        out_img = signals.smooth_segmentation(data_img, 4, seg_img,
                                              inplace=True)
        assert np.array_equal(out_img.get_data(), data.astype(np.float))

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

    def test_smooth_surface(self, random, meshdata):

        shape = 4, 3, 2
        affine = np.eye(4)
        fwhm = 4

        sm = surface.SurfaceMeasure(meshdata["verts"], meshdata["faces"])

        data = random.normal(10, 2, shape)
        data_img = nib.Nifti1Image(data, affine)

        surf_vox = random.choice(np.arange(np.product(shape)), sm.n_v, False)
        vertvol = np.full(shape, -1, np.int)
        vertvol.flat[surf_vox] = np.arange(sm.n_v)
        vert_img = nib.Nifti1Image(vertvol, affine)

        ribbon = vertvol > -1

        out_img = signals.smooth_surface(data_img, vert_img, sm, fwhm)
        out_data = out_img.get_data()

        assert out_data[ribbon].std() < data[ribbon].std()
        assert np.array_equal(out_data[~ribbon], data[~ribbon])

        shape = 4, 3, 2, 10

        data = random.normal(10, 2, shape)
        data_img = nib.Nifti1Image(data, affine)

        out_img = signals.smooth_surface(data_img, vert_img, sm, fwhm)
        out_data = out_img.get_data()

        assert out_data[ribbon].std() < data[ribbon].std()
        assert np.array_equal(out_data[~ribbon], data[~ribbon])

        n_tp = shape[-1]
        noise_vox = surf_vox[0]
        noise_mask = vertvol == noise_vox
        data[noise_mask] = random.normal(10, 10, n_tp)
        noise_img = nib.Nifti1Image(noise_mask.astype(int), affine)

        noise_out_img = signals.smooth_surface(data_img, vert_img, sm, fwhm)
        clean_out_img = signals.smooth_surface(data_img, vert_img, sm, fwhm,
                                               noise_img)

        noise_sd = noise_out_img.get_data()[noise_mask].std()
        clean_sd = clean_out_img.get_data()[noise_mask].std()

        assert clean_sd < noise_sd

        out_img = signals.smooth_surface(data_img, vert_img, sm, fwhm,
                                         inplace=True)

        assert np.array_equal(out_img.get_data(), data)
