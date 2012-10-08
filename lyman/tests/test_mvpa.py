import inspect
import numpy as np
import scipy as sp
from scipy import stats
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import (LeaveOneOut, LeaveOneLabelOut,
                                      StratifiedKFold)

from numpy.testing import assert_array_equal, assert_array_almost_equal
import nose.tools
from nose.tools import assert_equal, raises

from .. import mvpa

evs = [np.array([[6, 0, 1],
                 [18, 0, 1]]),
       np.array([[12, 0, 1],
                 [24, 0, 1]])]

datasets = [dict(X=stats.norm(0, 1).rvs((24, 12)),
                 y=stats.bernoulli(.5).rvs(24),
                 runs=np.repeat([0, 1], 12)) for i in range(3)]

datasets_3d = [dict(X=stats.norm(0, 1).rvs((4, 24, 12)),
                    y=stats.bernoulli(.5).rvs(24),
                    runs=np.repeat([0, 1], 12)) for i in range(3)]


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
    evs = [np.array([[1, 0, 1],
                     [2, 0, 1]]),
           np.array([[3, 0, 1]])]
    ts = np.random.randn(5, 5, 5, 4)
    mask = ts[..., 0] > .5
    X, y = mvpa.extract_dataset(evs, ts, mask, 1)

    assert_array_equal(y, np.array([0, 0, 1]))

    should_be = sp.stats.zscore(ts[mask].T[np.array([1, 2, 3])])
    assert_array_equal(X, should_be)


def test_extract_sizes():
    """Test different frame sizes."""
    evs = [np.array([[1, 0, 1],
                     [2, 0, 1]]),
           np.array([[3, 0, 1]])]
    ts = np.random.randn(5, 5, 5, 4)
    mask = ts[..., 0] > .5

    X_1, y_1 = mvpa.extract_dataset(evs, ts, mask, 1)
    assert_equal(X_1.shape, (3, mask.sum()))

    X_1, y_1 = mvpa.extract_dataset(evs, ts, mask, 1, [-1, 0])
    assert_equal(X_1.shape, (2, 3, mask.sum()))


@raises(ValueError)
def test_extract_mask_error():
    """Make sure mask is enforced as boolean."""
    evs = [[[1], [0], [1]]]
    ts = np.random.randn(10, 10, 10, 5)
    mask = np.random.rand(10, 10, 10)
    mvpa.extract_dataset(evs, ts, mask)


def test_decode_shapes():
    """Test that we get expected shapes from decode function."""
    model = GaussianNB()
    accs = mvpa.decode(datasets, model)
    assert_equal(accs.shape, (3,))
    accs = mvpa.decode(datasets_3d, model)
    assert_equal(accs.shape, (3, 4))

    splits = stats.bernoulli(.5).rvs(24)
    accs = mvpa.decode(datasets, model, splits)
    assert_equal(accs.shape, (3, 2))
    accs = mvpa.decode(datasets_3d, model, splits)
    assert_equal(accs.shape, (3, 4, 2))


def test_decode_options():
    """Test some other options for the decode function."""
    model = GaussianNB()
    mvpa.decode(datasets, model)
    mvpa.decode(datasets_3d, model)
    splits = stats.bernoulli(.5).rvs(24)
    mvpa.decode(datasets, model, splits)
    mvpa.decode(datasets, model, cv_method="sample")
    mvpa.decode(datasets, model, cv_method=5)
    mvpa.decode(datasets, model, cv_method=LeaveOneOut(24))
    mvpa.decode(datasets, model, n_jobs=2)


def test_decode_cross_val():
    """Test that cv_method strings are correct."""
    model = GaussianNB()

    acc1 = mvpa.decode(datasets, model, cv_method="run")
    cv = [LeaveOneLabelOut(d["runs"]) for d in datasets]
    acc2 = mvpa.decode(datasets, model, cv_method=cv)
    assert_array_equal(acc1, acc2)

    acc1 = mvpa.decode(datasets, model, cv_method="sample")
    cv = [LeaveOneOut(24) for d in datasets]
    acc2 = mvpa.decode(datasets, model, cv_method=cv)
    assert_array_equal(acc1, acc2)

    acc1 = mvpa.decode(datasets, model, cv_method=LeaveOneOut(24))
    cv = [LeaveOneOut(24) for d in datasets]
    acc2 = mvpa.decode(datasets, model, cv_method=cv)
    assert_array_equal(acc1, acc2)

    acc1 = mvpa.decode(datasets, model, cv_method=4)
    cv = [StratifiedKFold(d["y"], 4) for d in datasets]
    acc2 = mvpa.decode(datasets, model, cv_method=cv)
    assert_array_equal(acc1, acc2)
