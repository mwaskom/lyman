from itertools import product
import numpy as np
import scipy.signal as scipy_signal

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

        mask = random.uniform(0, 1, x.shape[1]) > .3
        with pytest.raises(ValueError):
            cv = signals.cv(x, mask=mask, axis=0)

        mask = random.uniform(0, 1, x.shape[1]) > .3
        with pytest.raises(ValueError):
            cv = signals.cv(x, mask=mask, axis=-1)

    def test_percent_change(self, random):

        x = [90, 110, 100]
        x_ptc = signals.percent_change(x)
        assert x_ptc == [-10, 10, 0]
