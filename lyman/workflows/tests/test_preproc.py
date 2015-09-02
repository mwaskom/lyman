import numpy as np
import nose.tools as nt
import numpy.testing as npt
import nibabel as nib
from .. import preproc


def test_prep_timeseries():

    prepper = preproc.PrepTimeseries(frames_to_toss=5)
    data = np.random.randn(60, 60, 33, 100)
    out_data = prepper.trim_timeseries(data)
    nt.assert_equal(out_data.shape, (60, 60, 33, 95))

    prepper = preproc.PrepTimeseries(frames_to_toss=0)
    data = np.random.randn(60, 60, 33, 100)
    out_data = prepper.trim_timeseries(data)
    nt.assert_equal(out_data.shape, (60, 60, 33, 100))


def test_extract_realignment_target():

    extractor = preproc.ExtractRealignmentTarget()

    for ntp in [20, 21]:
        index = np.arange(ntp)[None, None, None, :]
        data = np.ones((45, 45, 30, ntp)) * index
        img = nib.Nifti1Image(data, np.eye(4))

        targ = extractor.extract_target(img)
        nt.assert_equal(np.asscalar(np.unique(targ)), ntp // 2)


def test_robust_normalization():

    rs = np.random.RandomState(99)
    vol_shape = [40, 40, 30]
    ntp = 100
    ts = rs.normal(0, 1, size=vol_shape + [ntp])
    mask = rs.uniform(size=vol_shape) > .4

    art = preproc.ArtifactDetection()
    out = art.normalize_timeseries(ts, mask)
    nt.assert_equal(out.shape, (ntp,))


def test_scale_timeseries():

    rs = np.random.RandomState(99)
    data = rs.randn(10, 10, 10, 20)
    mask = rs.rand(10, 10, 10) > .25

    scaler = preproc.ScaleTimeseries()

    for target in [20, 40]:
        for func in [np.mean, np.median]:
            out = scaler.scale_timeseries(func, data, mask, target)
            npt.assert_almost_equal(func(out[mask]), target)
