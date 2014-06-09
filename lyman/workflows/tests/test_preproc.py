import numpy as np
import nose.tools as nt
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

        targ = extractor.extract_target(data)
        nt.assert_equal(np.asscalar(np.unique(targ)), ntp // 2)
