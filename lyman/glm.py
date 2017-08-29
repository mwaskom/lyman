from __future__ import division
import numpy as np
import scipy as sp
from scipy import sparse
import nibabel as nib

from .signals import smooth_volume


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

    """
    from numpy.fft import fft, ifft
    from numpy.linalg import lstsq

    # TODO this should be enhanced to take a segmentation image, not
    # just a mask image. That will need to update information accodingly.
    mask = mask_img.get_data().astype(np.bool)
    affine = mask_img.affine

    Y = ts_img.get_data()[mask].T
    Y = Y - Y.mean(axis=0)

    nvox = mask.sum()
    ntp = ts_img.shape[-1]
    nev = X.shape[1]
    assert X.shape[0] == ntp

    # Fit initial iteration OLS model in one step
    B_ols, _, _, _ = lstsq(X, Y)
    Yhat_ols = X.dot(B_ols)
    resid_ols = Y - Yhat_ols
    assert resid_ols.shape == (ntp, nvox)

    # Estimate the residual autocorrelation function
    tukey_m = int(np.floor(np.sqrt(ntp)))
    acf_pad = ntp * 2 - 1
    resid_fft = fft(resid_ols, n=acf_pad, axis=0)
    acf_fft = resid_fft * resid_fft.conjugate()
    acf = ifft(acf_fft, axis=0).real[:tukey_m]
    acf /= acf[[0]]
    assert acf.shape == (tukey_m, nvox)

    # Regularize the autocorrelation estimates with a tukey taper
    lag = np.arange(tukey_m)
    window = .5 * (1 + np.cos(np.pi * lag / tukey_m))
    acf_tukey = acf * window[:, np.newaxis]
    assert acf_tukey.shape == (tukey_m, nvox)

    # Smooth the autocorrelation estimates
    if smooth_fwhm is None:
        acf_smooth = acf_tukey
    else:
        nx, ny, nz = mask_img.shape
        acf_img_data = np.zeros((nx, ny, nz, tukey_m))
        acf_img_data[mask] = acf_tukey.T
        acf_img = nib.Nifti1Image(acf_img_data, affine)
        acf_img_smooth = smooth_volume(acf_img, smooth_fwhm, mask_img)
        acf_smooth = acf_img_smooth.get_data()[mask].T

    # Compute the autocorrelation kernel
    w_pad = ntp + tukey_m
    acf_kernel = np.zeros((w_pad, nvox))
    acf_kernel[:tukey_m] = acf_smooth
    acf_kernel[-tukey_m + 1:] = acf_smooth[:0:-1]
    assert (acf_kernel != 0).sum() == (nvox * (tukey_m * 2 - 1))

    # Compute the prewhitening kernel in the spectral domain
    acf_fft = fft(acf_kernel, axis=0).real
    W_fft = np.zeros((w_pad, nvox))
    W_fft[1:] = 1 / np.sqrt(np.abs(acf_fft[1:]))
    W_fft /= np.sqrt(np.sum(W_fft[1:] ** 2, axis=0, keepdims=True) / w_pad)

    # Prewhiten the data
    Y_fft = fft(Y, axis=0, n=w_pad)
    WY = ifft(W_fft * Y_fft, axis=0).real[:ntp].astype(np.float32)
    assert WY.shape == (ntp, nvox)

    # Prewhiten the design
    WX = np.empty((ntp, nev, nvox), np.float32)
    for i in range(nev):
        X_i = X[:, [i]]
        X_fft_i = fft(X_i, axis=0, n=w_pad)
        WX_i = ifft(W_fft * X_fft_i, axis=0).real[:ntp]
        WX[:, i, :] = WX_i.astype(np.float32)

    return WY, WX


def iterative_ols_fit(Y, X):
    """Fit an OLS model in each voxel.

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
    XtXinv : n_vox x n_ev x n_ev array
        The pinv(X' * X) matrices at each voxel.
    SS : n_vox array
        Model error summary at each voxel.
    X : n_tp x n_vox array
        Residual time series at each voxel.

    """
    from numpy import dot
    from numpy.linalg import pinv

    assert Y.shape[0] == X.shape[0]
    assert Y.shape[1] == X.shape[2]

    ntp, nev, nvox = X.shape

    B = np.empty((nvox, nev))
    XtXinv = np.empty((nvox, nev, nev))
    SS = np.empty(nvox)
    E = np.empty((ntp, nvox))

    I = np.eye(ntp)

    for i in range(nvox):

        y_i, X_i = Y[..., i], X[..., i]
        XtXinv_i = pinv(dot(X_i.T, X_i))
        b_i = dot(XtXinv_i, dot(X_i.T, y_i))
        R_i = I - dot(X_i, dot(XtXinv_i, X_i.T))
        e_i = dot(R_i, y_i)
        ss_i = dot(e_i, e_i.T) / R_i.trace()

        B[i] = b_i
        XtXinv[i] = XtXinv_i
        SS[i] = ss_i
        E[:, i] = e_i

    return B, XtXinv, SS, E


def iterative_contrast_estimation(B, XtXinv, SS, C):
    """Compute contrast parameter and variance estimates in each voxel.

    Parameters
    ----------
    B : n_vox x n_ev array
        Parameter estimates for each voxel.
    XtXinv : n_vox x n_ev x n_ev array
        The pinv(X' * X) matrices for each voxel.
    SS : n_vox array
        The model error summary at each voxel.
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

    nvox, nev = B.shape
    ncon = len(C)

    G = np.empty((nvox, ncon))
    V = np.empty((nvox, ncon))

    for i in range(nvox):
        b_i = B[i]
        XtXinv_i = XtXinv[i]
        ss_i = SS[i]

        for j, c_j in enumerate(C):

            keff_i = dot(c_j, dot(XtXinv_i, c_j))
            g_ij = dot(c_j, b_i)
            v_ij = keff_i * ss_i

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


def highpass_filter_matrix(ntp, cutoff, tr=1):
    """Return an array to implement a gaussian running line filter.

    To implement the filter, premultiply your data with this array.

    Parameters
    ----------
    ntp : int
        Number of timepoints in data.
    cutoff : float
        Filter cutoff, in seconds.
    tr : float
        Temporal resolution of data, in seconds.

    Return
    ------
    F : ntp x ntp array
        Filter matrix.

    """
    cutoff = cutoff / tr
    sig2n = np.square(cutoff / np.sqrt(2))

    kernel = np.exp(-np.square(np.arange(ntp)) / (2 * sig2n))
    kernel = 1 / np.sqrt(2 * np.pi * sig2n) * kernel

    K = sp.linalg.toeplitz(kernel)
    K = np.dot(np.diag(1 / K.sum(axis=1)), K)

    H = np.empty((ntp, ntp))
    X = np.column_stack((np.ones(ntp), np.arange(ntp)))
    for k in range(ntp):
        W = sparse.diags(K[k])
        hat = np.dot(X, np.linalg.pinv(W * X) * W)
        H[k] = hat[k]
    F = np.eye(ntp) - H

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
    ntp = data.shape[0]
    F = highpass_filter_matrix(ntp, cutoff, tr)
    data[:] = np.dot(F, data)

    # Remove the residueal mean of each timeseries to match FSL
    data -= data.mean(axis=0, keepdims=True)

    # Remove added dimensions
    if need_squeeze:
        data = data.squeeze()

    return data
