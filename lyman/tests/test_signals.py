from itertools import product
import numpy as np
import scipy.signal as scipy_signal
import nibabel as nib

import pytest

from .. import signals


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
