from __future__ import division
import numpy as np
import pandas as pd
import scipy as sp
from scipy import sparse, stats

from .signals import smooth_volume
from .utils import image_to_matrix, matrix_to_image


class HRFModel(object):

    def transform(self, input):
        """Generate a predicted response ot the input."""
        raise NotImplementedError


class GammaHRF(HRFModel):

    def __init__(self, derivative=False, res=60, duration=32,
                 pos_shape=6, pos_scale=1, neg_shape=16, neg_scale=1,
                 ratio=1/6):
        """Gamma PDF model of the HRF, possibly with temporal derivative.

        Parameters
        ----------
        derivative : bool
            If True, include a temporal derivative basis function.
        res : float
            Sampling frequency at which to generate the convolution kernel.
        duration : float
            Duration of the convolution kernel.
        pos_{shape, scale} : floats
            Parameters for scipy.stats.gamma defining the initial positive
            component of the response.
        neg_{shape, scale} : floats
            Parameters for scipy.stats.gamma defining the later negative
            component of the response.
        ratio : float
            Ratio between the amplitude of the positive component to the
            amplitude of the negative component.

        """
        pos = stats.gamma(pos_shape, scale=pos_scale).pdf
        neg = stats.gamma(neg_shape, scale=neg_scale).pdf
        tps = np.arange(0, duration, 1 / res, np.float)

        y = pos(tps) - ratio * neg(tps)
        y /= y.sum()

        if derivative:
            dydt = self._temporal_derivative(y)
            self.kernel = y, dydt
        else:
            self.kernel = y, None

    def _temporal_derivative(self, y):
        """Compute the scaled temporal derivative of the input."""
        from numpy import diff, sum, square, sqrt
        dydt = np.zeros_like(y)
        dydt[1:] = diff(y)
        dydt *= sqrt(sum(square(y)) / sum(square(dydt)))
        return dydt

    def transform(self, input):
        """Generate a prediction basis set for the input through convolution.

        Parameters
        ----------
        input : array or series
            Input data; should be one-dimensional

        Returns
        -------
        output : pair of arrays or series or None
            Output data; has same type of input, and the derivative output
            is always included but may be None if the model has no derivative.

        """
        n_tp = len(input)
        y, dydt = self.kernel

        y = np.convolve(input, y)[:n_tp]
        if dydt is None:
            dy = None
        else:
            dy = np.convolve(input, dydt)[:n_tp]

        if isinstance(input, pd.Series):
            y = pd.Series(y, input.index, name=input.name)
            if dy is not None:
                dy = pd.Series(dy, input.index, name=input.name + "-dydt")

        # TODO is always returning a pair helpful or a nuisance?
        # upside: when building the design matrix, a flat column list can be
        # extended, and then pd.concat will drop the Nones
        # downside: when just interested in the cannonical prediction, it's
        # a minor pain to have to index into the return value
        # in any case, transform() should probably have the same behavior
        # as kernel in terms of what it returns

        return y, dy


def build_design_matrix(events=None, hrf_model=None,
                        regressors=None, artifacts=None,
                        n_tp=None, tr=1, res=60, shift=.5,
                        hpf_matrix=None, demean=True):

    pass


def prewhiten_image_data(ts_img, mask_img, X, smooth_fwhm=5):
    """Estimate autocorrelation and transform data and design for OLS.

    Parameters
    ----------
    ts_img : nibabel image
        4D image with fMRI data. If using autocorrelation smoothing, the
        affine must have correct information about voxel size.
    mask_img : nibabel image
        3D image with mask defining voxels to include in model. Also used
        to constrain the autocorrelation estimate smoothing.
    X : n_tp x n_ev array
        Design matrix array. Should have zero mean and no constant.
    smooth_fwhm : float
        Size (in mm) of the smoothing kernel for smoothing the autocorrelation
        estimates. Requires that the time series image affine has correct
        information about voxel size.

    Returns
    -------
    WY : n_tp x n_vox array
        Prewhitened time series data for voxels in mask.
    WX : n_tp x n_ev x n_vox array
        Prewhitened design matrix for voxels in mask.

    """
    from numpy.fft import fft, ifft

    # TODO this should be enhanced to take a segmentation image, not
    # just a mask image. That will need to update information accodingly.

    Y = image_to_matrix(ts_img, mask_img)
    Y = Y - Y.mean(axis=0)

    n_tp, n_vox = Y.shape
    n_ev = X.shape[1]
    assert X.shape[0] == n_tp

    # Estimate the autocorrelation function of the model residuals
    acf = estimate_residual_autocorrelation(Y, X)
    tukey_m, _ = acf.shape

    # Smooth the autocorrelation estimates
    if smooth_fwhm is None:
        acf_smooth = acf
    else:
        acf_img = matrix_to_image(acf, mask_img)
        acf_img_smooth = smooth_volume(acf_img, smooth_fwhm, mask_img)
        acf_smooth = image_to_matrix(acf_img_smooth, mask_img)

    # Compute the autocorrelation kernel
    w_pad = n_tp + tukey_m
    acf_kernel = np.zeros((w_pad, n_vox))
    acf_kernel[:tukey_m] = acf_smooth
    acf_kernel[-tukey_m + 1:] = acf_smooth[:0:-1]
    assert (acf_kernel != 0).sum() == (n_vox * (tukey_m * 2 - 1))

    # Compute the prewhitening kernel in the spectral domain
    acf_fft = fft(acf_kernel, axis=0).real
    W_fft = np.zeros((w_pad, n_vox))
    W_fft[1:] = 1 / np.sqrt(np.abs(acf_fft[1:]))
    W_fft /= np.sqrt(np.sum(W_fft[1:] ** 2, axis=0, keepdims=True) / w_pad)

    # Prewhiten the data
    Y_fft = fft(Y, axis=0, n=w_pad)
    WY = ifft(W_fft * Y_fft, axis=0).real[:n_tp].astype(np.float32)
    assert WY.shape == (n_tp, n_vox)

    # Prewhiten the design
    WX = np.empty((n_tp, n_ev, n_vox), np.float32)
    for i in range(n_ev):
        X_i = X[:, [i]]
        X_fft_i = fft(X_i, axis=0, n=w_pad)
        WX_i = ifft(W_fft * X_fft_i, axis=0).real[:n_tp]
        WX[:, i, :] = WX_i.astype(np.float32)

    return WY, WX


def estimate_residual_autocorrelation(Y, X, tukey_m=None):
    """Fit OLS model and estimate residual autocorrelation with regularization.

    Parameters
    ----------
    Y : n_tp x n_vox array
        Array of time series data for multiple voxels.
    X : n_tp x n_ev array
        Design matrix for the model.
    tukey_m: int or None
        Size of tukey taper window or None to use default rule.

    Returns
    -------
    acf : tukey_m x n_vox array
        Regularized autocorrelation function estimate for each voxel.

    """
    from numpy.fft import fft, ifft

    # Fit initial iteration OLS model in one step
    B_ols, _, _, _ = np.linalg.lstsq(X, Y)
    Yhat_ols = X.dot(B_ols)
    resid_ols = Y - Yhat_ols

    # Compute empircal residual autocorrelation function
    n_tp = Y.shape[0]
    if tukey_m is None:
        tukey_m = default_tukey_window(n_tp)
    acf_pad = n_tp * 2 - 1
    resid_fft = fft(resid_ols, n=acf_pad, axis=0)
    acf_fft = resid_fft * resid_fft.conjugate()
    acf = ifft(acf_fft, axis=0).real[:tukey_m]
    acf /= acf[[0]]

    # Regularize the autocorrelation estimate with a tukey taper
    lag = np.expand_dims(np.arange(tukey_m), 1)
    window = .5 * (1 + np.cos(np.pi * lag / tukey_m))
    acf *= window

    return acf


def default_tukey_window(n):
    """The default rule for choosing the Tukey taper window used by FSL."""
    return int(np.floor(np.sqrt(n)))


def iterative_ols_fit(Y, X):
    """Fit a linear model using ordinary least squares in each voxel.

    The design matrix is expected to be 3D because this function is intended
    to be used in the context of a prewhitened model, where each voxel has a
    slightly different (whitened) design.

    Parameters
    ----------
    Y : n_tp x n_vox array
        Time series for each voxel.
    X : n_tp x n_ev x n_vox array
        Design matrix for each voxel.

    Returns
    -------
    B : n_vox x n_ev array
        Parameter estimates at each voxel.
    SS : n_vox array
        Model error summary at each voxel.
    XtXinv : n_vox x n_ev x n_ev array
        The pinv(X' * X) matrices at each voxel.
    E : n_tp x n_vox array
        Residual time series at each voxel.

    """
    from numpy import dot
    from numpy.linalg import pinv

    Y = Y.astype(np.float64)
    X = X.astype(np.float64)

    assert Y.shape[0] == X.shape[0]
    assert Y.shape[1] == X.shape[2]

    n_tp, n_ev, n_vox = X.shape

    B = np.empty((n_vox, n_ev), np.float32)
    SS = np.empty(n_vox, np.float32)
    XtXinv = np.empty((n_vox, n_ev, n_ev), np.float32)
    E = np.empty((n_tp, n_vox), np.float32)

    I = np.eye(n_tp)

    for i in range(n_vox):

        y_i, X_i = Y[..., i], X[..., i]
        XtXinv_i = pinv(dot(X_i.T, X_i))
        b_i = dot(XtXinv_i, dot(X_i.T, y_i))
        R_i = I - dot(X_i, dot(XtXinv_i, X_i.T))
        e_i = dot(R_i, y_i)
        ss_i = dot(e_i, e_i.T) / R_i.trace()

        B[i] = b_i
        SS[i] = ss_i
        XtXinv[i] = XtXinv_i
        E[:, i] = e_i

    return B, SS, XtXinv, E


def iterative_contrast_estimation(B, SS, XtXinv, C):
    """Compute contrast parameter and variance estimates in each voxel.

    Parameters
    ----------
    B : n_vox x n_ev array
        Parameter estimates for each voxel.
    SS : n_vox array
        The model error summary at each voxel.
    XtXinv : n_vox x n_ev x n_ev array
        The pinv(X' * X) matrices for each voxel.
    C : n_con x n_ev array
        List of contrast vectors.

    Returns
    -------
    G : n_vox x n_con array
        Contrast parameter estimates.
    V : n_vox x n_con array
        Contrast parameter variance estimates.
    T : n_vox x n_con array
        Contrast t statistics.

    """
    from numpy import dot

    assert B.shape[0] == XtXinv.shape[0] == SS.shape[0]
    assert B.shape[1] == XtXinv.shape[1] == XtXinv.shape[2]

    n_vox, n_ev = B.shape
    n_con = len(C)

    G = np.empty((n_vox, n_con))
    V = np.empty((n_vox, n_con))

    for i in range(n_vox):

        b_i = B[i]
        ss_i = SS[i]
        XtXinv_i = XtXinv[i]

        for j, c_j in enumerate(C):

            keff_ij = dot(c_j, dot(XtXinv_i, c_j))
            g_ij = dot(c_j, b_i)
            v_ij = keff_ij * ss_i

            G[i, j] = g_ij
            V[i, j] = v_ij

    T = G / np.sqrt(V)

    return G, V, T


def contrast_fixed_effects(G, V):
    """Compute higher-order fixed effects parameters.

    Parameters
    ----------
    G : n_vox x n_run array
        First-level contrast parameter estimates.
    V : n_Vox x n_run array
        First-level contrast parameter variance estimates.

    Returns
    -------
    con : n_vox array
        Fixed effects contrast parameter estimates.
    var : n_vox  array
        Fixed effects contrast parameter variance estimates.
    t : n_vox array
        Fixed effects t statistics.

    """
    var = 1 / (1 / V).sum(axis=-1)
    con = var * (G / V).sum(axis=-1)
    t = con / np.sqrt(var)
    return con, var, t


def highpass_filter_matrix(n_tp, cutoff, tr=1):
    """Return an array to implement a gaussian running line filter.

    To implement the filter, premultiply your data with this array.

    Parameters
    ----------
    n_tp : int
        Number of timepoints in data.
    cutoff : float
        Filter cutoff, in seconds.
    tr : float
        Temporal resolution of data, in seconds.

    Return
    ------
    F : n_ntp x n_tp array
        Filter matrix.

    """
    cutoff = cutoff / tr
    sig2n = np.square(cutoff / np.sqrt(2))

    kernel = np.exp(-np.square(np.arange(n_tp)) / (2 * sig2n))
    kernel = 1 / np.sqrt(2 * np.pi * sig2n) * kernel

    K = sp.linalg.toeplitz(kernel)
    K = np.dot(np.diag(1 / K.sum(axis=1)), K)

    H = np.empty((n_tp, n_tp))
    X = np.column_stack((np.ones(n_tp), np.arange(n_tp)))
    for k in range(n_tp):
        W = sparse.diags(K[k])
        hat = np.dot(X, np.linalg.pinv(W * X) * W)
        H[k] = hat[k]
    F = np.eye(n_tp) - H

    return F


def highpass_filter(data, cutoff, tr=1, copy=True):
    """Highpass filter data with gaussian running line filter.

    Parameters
    ----------
    data : 1d or 2d array
        Data array where first dimension is time.
    cutoff : float
        Filter cutoff in seconds.
    tr : float
        TR of data in seconds.
    copy : boolean
        If False, data is filtered in place.

    Returns
    -------
    data : 1d or 2d array
        Filtered version of the data.

    """
    if copy:
        data = data.copy()

    # Ensure data is a matrix
    if data.ndim == 1:
        need_squeeze = True
        data = data[:, np.newaxis]
    else:
        need_squeeze = False

    # Filter each column of the data
    n_tp = data.shape[0]
    F = highpass_filter_matrix(n_tp, cutoff, tr)
    data[:] = np.dot(F, data).astype(data.dtype)

    # Remove the residueal mean of each timeseries to match FSL
    data -= data.mean(axis=0, keepdims=True)

    # Remove added dimensions
    if need_squeeze:
        data = data.squeeze()

    return data
