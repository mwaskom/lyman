from itertools import product
import numpy as np
import pandas as pd
from nipype.interfaces.base import Bunch

import nose.tools as nt
import numpy.testing as npt

from .. import model


class TestModelSummary(object):

    rs = np.random.RandomState(101)

    def test_dot_by_slice(self):

        n_x, n_y, n_z, n_t, n_pe = 10, 11, 12, 20, 5
        X = Bunch(design_matrix=pd.DataFrame(self.rs.randn(n_t, n_pe)))
        pes = self.rs.randn(n_x, n_y, n_z, n_pe)

        ms = model.ModelSummary()
        out = ms.dot_by_slice(X, pes)

        nt.assert_equal(out.shape, (n_x, n_y, n_z, n_t))

        for i, j, k in product(range(n_x), range(n_y), range(n_z)):
            pe_ijk = pes[i, j, k]
            yhat_ijk = X.design_matrix.dot(pe_ijk)
            npt.assert_array_almost_equal(yhat_ijk, out[i, j, k])

    def test_dot_by_slice_submatrix(self):

        n_x, n_y, n_z, n_t, n_pe = 11, 10, 9, 25, 4
        X = Bunch(design_matrix=pd.DataFrame(self.rs.randn(n_t, n_pe)),
                  confound_vector=np.array([[1, 0, 0, 1]]).T)
        pes = self.rs.randn(n_x, n_y, n_z, n_pe)

        ms = model.ModelSummary()
        out = ms.dot_by_slice(X, pes, "confound")

        nt.assert_equal(out.shape, (n_x, n_y, n_z, n_t))

        for i, j, k in product(range(n_x), range(n_y), range(n_z)):
            pe_ijk = pes[i, j, k] * X.confound_vector.T
            yhat_ijk = X.design_matrix.dot(pe_ijk.T).squeeze()
            npt.assert_array_almost_equal(yhat_ijk, out[i, j, k])

    def test_compute_r2(self):

        shape = 10, 10, 15, 20
        n_x, n_y, n_z, n_t = shape
        y = self.rs.randn(*shape)
        yhat = y + 2 * self.rs.randn(*shape)

        y -= y.mean(axis=-1)[..., np.newaxis]
        yhat -= yhat.mean(axis=-1)[..., np.newaxis]

        ms = model.ModelSummary()
        ms.y = y
        ms.sstot = np.sum(np.square(y), axis=-1)

        ssres, r2 = ms.compute_r2(yhat)
        for i, j, k in product(range(n_x), range(n_y), range(n_z)):
            y_ijk = y[i, j, k]
            yhat_ijk = yhat[i, j, k]
            ssres_ijk = np.sum(np.square(y_ijk - yhat_ijk))
            npt.assert_almost_equal(ssres[i, j, k], ssres_ijk)
            r2_ijk = 1 - ssres_ijk / np.square(y_ijk).sum()
            npt.assert_almost_equal(r2[i, j, k], r2_ijk)
