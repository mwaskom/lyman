import numpy as np
import nibabel as nib

from .signals import smooth_volume


def prewhiten_image_data(ts_img, X, mask_img, smooth_fwhm=5):

    from numpy.fft import fft, ifft
    from numpy.linalg import lstsq

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
    tukey_m = int(np.round(np.sqrt(ntp)))
    acf_pad = ntp * 2 - 1
    resid_fft = fft(resid_ols, n=acf_pad, axis=0)
    acf_fft = resid_fft * resid_fft.conjugate()
    acf = ifft(acf_fft, axis=0).real[:tukey_m]
    acf /= acf[0]
    assert acf.shape == (tukey_m, nvox)

    # Regularize the autocorrelation estimates with a tukey taper
    lag = np.arange(tukey_m)
    window = .5 * (1 + np.cos(np.pi * lag / tukey_m))
    acf_tukey = acf * window[:, np.newaxis]
    assert acf_tukey.shape == (tukey_m, nvox)

    # Smooth the autocorrelation estimates
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
    W_fft /= np.sqrt(np.sum(W_fft[1:] ** 2, axis=0, keepdims=True)) / w_pad

    # Prewhiten the data
    Y_fft = fft(Y, axis=0, n=w_pad)
    WY = ifft(W_fft * Y_fft.real
              + W_fft * Y_fft.imag * 1j,
              axis=0).real[:ntp].astype(np.float32)
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

    from numpy import dot
    from numpy.linalg import pinv

    assert Y.shape[0] == X.shape[0]
    assert Y.shape[1] == X.shape[2]

    ntp, nev, nvox = X.shape

    B = np.empty((nvox, nev))
    XtXinv = np.empty((nvox, nev, nev))
    SS = np.empty(nvox)

    I = np.eye(ntp)

    for i in range(nvox):

        y_i, X_i = Y[..., i], X[..., i]
        XtXinv_i = pinv(dot(X_i.T, X_i))
        b_i = dot(XtXinv_i, dot(X_i.T, y_i))
        R_i = I - dot(X_i, dot(XtXinv_i, X_i.T))
        r_i = dot(R_i, y_i)
        ss_i = dot(r_i, r_i.T) / R_i.trace()

        B[i] = b_i
        XtXinv[i] = XtXinv_i
        SS[i] = ss_i

    return B, XtXinv, SS


def iterative_contrast_estimation(B, XtXinv, SS, C):

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
