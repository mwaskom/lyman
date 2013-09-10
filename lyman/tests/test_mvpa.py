import inspect
import numpy as np
import scipy as sp
import pandas as pd
from scipy import stats
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import (cross_val_score,
                                      LeaveOneOut,
                                      LeaveOneLabelOut,
                                      KFold)

from numpy.testing import assert_array_equal, assert_array_almost_equal
import numpy.testing as npt
import nose.tools
from nose.tools import assert_equal, raises

from .. import mvpa

evs = [np.array([[6, 0, 1],
                 [18, 0, 1]]),
       np.array([[12, 0, 1],
                 [24, 0, 1]])]

dataset = dict(X=stats.norm(0, 1).rvs((24, 12)),
               y=stats.bernoulli(.5).rvs(24),
               runs=np.repeat([0, 1], 12))

dataset_3d = dict(X=stats.norm(0, 1).rvs((4, 24, 12)),
                  y=stats.bernoulli(.5).rvs(24),
                  runs=np.repeat([0, 1], 12))


def test_design_generator():
    """Test that event_designs is a generator."""
    assert(inspect.isgeneratorfunction(mvpa.event_designs))


def test_design_tps():
    """Test that design matrices have the right number of rows."""
    ntp = 15
    designs = mvpa.event_designs(evs, ntp)
    design_rows = [d.shape[0] for d in designs]
    assert(all([r == ntp for r in design_rows]))


def test_design_columns():
    """Test split_confounds."""
    split_ds = mvpa.event_designs(evs, 15, split_confounds=True)
    n_col_list = [d.shape[1] for d in split_ds]
    n_cols = n_col_list[0]
    assert_equal(n_cols, 3)
    assert(all([i == n_cols for i in n_col_list]))

    single_ds = mvpa.event_designs(evs, 15, split_confounds=False)
    n_col_list = [d.shape[1] for d in single_ds]
    n_cols = n_col_list[0]
    assert_equal(n_cols, 2)
    assert(all([i == n_cols for i in n_col_list]))


def test_fir_design():
    """Check assumptions when using FIR deconvolution."""
    gen = mvpa.event_designs(evs, 20, 2, True, "fir", 5)
    mat = gen.next()
    ncol = mat.shape[1]
    assert_equal(ncol, 15)
    nose.tools.assert_less_equal(mat.max(), 1)
    assert_array_equal(mat.min(axis=0), np.zeros(ncol))
    r1 = np.argwhere(mat[:, 0])
    r2 = np.argwhere(mat[:, 1])
    rdiff = r1 - r2
    assert(np.all(rdiff <= 0))


def test_event_confounds():
    """Test that event of interest is removed from confound columns."""
    gen = mvpa.event_designs(evs, 15, split_confounds=False)
    mat = gen.next()
    peak = np.argmax(mat[:, 0])
    nose.tools.assert_not_equal(mat[peak, 0], mat[peak, 1])


def test_deconvolved_shape():
    """Test shape of deconvolution output."""
    data = np.random.randn(16, 10)
    deonv = mvpa.iterated_deconvolution(data, evs)
    assert_equal(deonv.shape, (4, 10))


def test_deconvolve_estimate():
    """Roughly test deconvolution performance."""
    data = np.random.randn(16, 10)
    data[4:7] += 50
    data[10:13] += 50
    deconv = mvpa.iterated_deconvolution(data, evs)
    high = deconv[(0, 2)]
    low = deconv[(1, 3)]
    nose.tools.assert_greater(high, low)


def test_extract_dataset():
    """Test simple case."""
    evs = pd.DataFrame(dict(onset=[1, 2, 3],
                            condition=["foo", "foo", "bar"]),
                            dtype=float)
    ts = np.random.randn(5, 5, 5, 4)
    mask = ts[..., 0] > .5
    X, y, m = mvpa.extract_dataset(evs, ts, mask, 1)

    assert_array_equal(y, np.array([1, 1, 0]))

    should_be = sp.stats.zscore(ts[mask].T[np.array([1, 2, 3])])
    assert_array_equal(X, should_be)

    X_, y_, m_ = mvpa.extract_dataset(evs, ts, mask, 1,
                                  event_names=["bar", "foo"])
    assert_array_equal(X_, X)
    assert_array_equal(y_, y)
    assert_array_equal(m_, m)


def test_extract_sizes():
    """Test different frame sizes."""
    evs = pd.DataFrame(dict(onset=[1, 2, 3],
                            condition=["foo", "foo", "bar"]),
                            dtype=float)
    ts = np.random.randn(5, 5, 5, 4)
    mask = ts[..., 0] > .5

    X_1, y_1, m_1 = mvpa.extract_dataset(evs, ts, mask, 1)
    assert_equal(X_1.shape, (3, mask.sum()))
    assert_equal(len(m_1), mask.sum())

    X_2, y_2, m_2 = mvpa.extract_dataset(evs, ts, mask, 1, [-1, 0])
    assert_equal(X_2.shape, (2, 3, mask.sum()))


def test_extract_upsample():
    """Test upsampling during extraction."""
    evs = pd.DataFrame(dict(onset=[1, 2, 3],
                            condition=["foo", "foo", "bar"]),
                            dtype=float)
    ts = np.random.randn(5, 5, 5, 10)
    mask = ts[..., 0] > .5

    X, y, m = mvpa.extract_dataset(evs, ts, mask, tr=1,
                                frames=[-1, 0], upsample=2)
    assert_equal(X.shape, (4, 3, mask.sum()))
    assert_equal(len(m), mask.sum())


def test_extract_zero_var():
    """Test that zero-variance features are masked out."""
    evs = pd.DataFrame(dict(onset=[1, 2, 3],
                            condition=["foo", "foo", "bar"]),
                            dtype=float)
    ts = np.random.randn(5, 5, 5, 10)
    ts[0, 0, 0, :] = 0
    mask = np.ones((5, 5, 5), np.bool)

    X, y, m = mvpa.extract_dataset(evs, ts, mask, 1)
    assert_array_equal(m, np.array([0] + [1] * (5 ** 3 - 1), np.bool))


@raises(ValueError)
def test_extract_mask_error():
    """Make sure mask is enforced as boolean."""
    evs = pd.DataFrame(dict(onset=[1], condition="foo"), dtype=float)
    ts = np.random.randn(10, 10, 10, 5)
    mask = np.random.rand(10, 10, 10)
    mvpa.extract_dataset(evs, ts, mask)


def test_decode_shapes():
    """Test that we get expected shapes from decode function."""
    model = GaussianNB()
    accs = mvpa._decode_subject(dataset, model)
    assert_equal(accs.shape, (1,))
    accs = mvpa._decode_subject(dataset_3d, model)
    assert_equal(accs.shape, (4,))

    splits = stats.bernoulli(.5).rvs(24)
    accs = mvpa._decode_subject(dataset, model, split_pred=splits)
    assert_equal(accs.shape, (2,))
    accs = mvpa._decode_subject(dataset_3d, model, split_pred=splits)
    assert_equal(accs.shape, (4, 2))

    accs = mvpa._decode_subject(dataset, model, trialwise=True)
    assert_equal(accs.shape, (len(dataset["y"]),))
    accs = mvpa._decode_subject(dataset_3d, model, trialwise=True)
    assert_equal(accs.shape, (4, len(dataset["y"])))


def test_decode_options():
    """Test some other options for the decode function."""
    model = GaussianNB()
    mvpa._decode_subject(dataset, model)
    mvpa._decode_subject(dataset_3d, model)
    splits = stats.bernoulli(.5).rvs(24)
    mvpa._decode_subject(dataset, model, cv="sample")
    mvpa._decode_subject(dataset, model, cv=5)
    mvpa._decode_subject(dataset, model, split_pred=splits)
    mvpa._decode_subject(dataset, model, trialwise=True)
    mvpa._decode_subject(dataset, model, logits=True)


@raises(ValueError)
def test_trialwise_split_exclusivity():
    """Test that we can't split predictions and get trialwise scores."""
    model = GaussianNB()
    splits = stats.bernoulli(.5).rvs(24)
    mvpa._decode_subject(dataset, model, split_pred=splits, trialwise=True)


def test_decode_cross_val():
    """Test that cv strings are correct."""
    model = GaussianNB()
    X = dataset["X"]
    y = dataset["y"]

    acc1 = mvpa._decode_subject(dataset, model, cv="run")
    cv = LeaveOneLabelOut(dataset["runs"])
    acc2 = cross_val_score(model, X, y, cv=cv).mean()
    assert_array_almost_equal(acc1, acc2)

    acc1 = mvpa._decode_subject(dataset, model, cv="sample")
    cv = LeaveOneOut(24)
    acc2 = cross_val_score(model, X, y, cv=cv).mean()
    assert_array_almost_equal(acc1, acc2)

    acc1 = mvpa._decode_subject(dataset, model, cv=4)
    cv = KFold(len(y), 4)
    acc2 = cross_val_score(model, X, y, cv=cv).mean()
    assert_array_almost_equal(acc1, acc2)


def test_accs_vs_logits():
    """Test that accs and logits give consisitent information."""
    model = GaussianNB()
    accs = mvpa._decode_subject(dataset, model, trialwise=True)
    logits = mvpa._decode_subject(dataset, model,
                                  logits=True, trialwise=True)
    logit_accs = np.where(logits >= 0, 1., 0.)
    assert_array_equal(accs, logit_accs)
