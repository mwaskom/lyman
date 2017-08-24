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
              axis=0).real[:ntp]

    # Prewhiten the design
    X_fft = fft(X, axis=0, n=w_pad)
    X_fft_exp = X_fft[:, :, np.newaxis]
    W_fft_exp = W_fft[:, np.newaxis, :]
    WX = ifft(W_fft_exp * X_fft_exp.real
              + W_fft_exp * X_fft_exp.imag * 1j,
              axis=0).real[:ntp]

    return WY, WX
