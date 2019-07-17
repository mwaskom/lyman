from __future__ import division
import numpy as np
import pandas as pd
import scipy as sp
from scipy import sparse, stats, signal, linalg
from scipy.interpolate import interp1d

from .signals import smooth_volume
from .utils import image_to_matrix, matrix_to_image


# =========================================================================== #
# Design matrix construction
# =========================================================================== #


class HRFModel(object):
    """Abstract base class for HRF models used in design construction."""
    def transform(self, input):
        """Generate a basis set for the predicted response to the input."""
        raise NotImplementedError


class IdentityHRF(HRFModel):
    """Model that does not alter input during transform; useful for testing."""
    def transform(self, x):
        """Return input witout altering."""
        y = x
        return y


class GammaHRF(HRFModel):
    """Double gamma variate model of cannonical HRF."""
    def __init__(self, res=60, duration=32, pos_shape=6, pos_scale=1,
                 neg_shape=16, neg_scale=1, ratio=1/6):
        """Initialize the object with parameters that define response shape.

        Parameters
        ----------
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

        k = pos(tps) - ratio * neg(tps)
        k /= k.sum()
        self.kernel = k

    def transform(self, x):
        """Generate a prediction basis set for the input through convolution.

        Parameters
        ----------
        x : array or series
            Input data; should be one-dimensional

        Returns
        -------
        output : pair of arrays or series or None
            Output data; has same type of input, and the derivative output
            is always included but may be None if the model has no derivative.

        """
        n_tp = len(x)

        y = np.convolve(x, self.kernel)[:n_tp]

        if isinstance(x, pd.Series):
            y = pd.Series(y, x.index, name=str(x.name))

        return y


class GammaBasis(HRFModel):
    """Basis set for HRF based on Gamma variate model."""
    def __init__(self, time_derivative=True, disp_derivative=True,
                 res=60, duration=32, pos_shape=6, pos_scale=1,
                 neg_shape=16, neg_scale=1, ratio=1/6):
        """Initialize the object with parameters that define response shape.

        Parameters
        ----------
        time_derivative : bool
            Add temporal derivative of the response.
        disp_derivative : bool
            Add dispersion derivative of the response.
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
        # Build the main Gamma variate
        main = GammaHRF(res=res, duration=duration,
                        pos_shape=pos_shape, pos_scale=pos_scale,
                        neg_shape=neg_shape, neg_scale=neg_scale,
                        ratio=ratio).kernel
        parts = [main]

        # Build the time derivative
        time_dx = np.zeros_like(main)
        time_dx[1:] = np.diff(main)
        time_dx *= np.sqrt(np.sum(main ** 2) / np.sum(time_dx ** 2))
        self._time_derivative = time_derivative
        if time_derivative:
            parts.append(time_dx)

        # Build the dispersion derivative
        alt = GammaHRF(res=res, duration=duration,
                       pos_shape=pos_shape / 1.01, pos_scale=pos_scale * 1.01,
                       neg_shape=neg_shape, neg_scale=neg_scale,
                       ratio=ratio).kernel
        disp_dx = (main - alt) / .01
        disp_dx *= np.sqrt(np.sum(main ** 2) / np.sum(disp_dx ** 2))
        self._disp_derivative = disp_derivative
        if disp_derivative:
            parts.append(disp_dx)

        self.kernel = np.column_stack(parts)

    def transform(self, x):
        """Generate a prediction basis set for the input through convolution.

        Parameters
        ----------
        x : array or series
            Input data; should be one-dimensional

        Returns
        -------
        output : pair of arrays or series or None
            Output data; has same type of input, and the derivative output
            is always included but may be None if the model has no derivative.

        """
        n_tp = len(x)

        # Convolve the input with each element of the basis set
        ys = []
        for col in self.kernel.T:
            ys.append(np.convolve(x, col)[:n_tp])
        y_orig = np.column_stack(ys)

        # Orthogonalize derivative regressors
        y, r = np.linalg.qr(y_orig)
        y *= np.linalg.norm(y_orig, axis=0) * np.sign(np.diag(r))

        if isinstance(x, pd.Series):
            cols = [str(x.name)]
            if self._time_derivative:
                cols.append(f"{x.name}-dydt")
            if self._disp_derivative:
                cols.append(f"{x.name}-dydw")
            y = pd.DataFrame(y, x.index, cols)

        return y


class FIRBasis(HRFModel):
    """Finite Impulse Response basis model."""
    def __init__(self, n, offset=0, suffix="_"):
        """Initialize the model with its parameters.

        Parameters
        ----------
        n : int
            Number of delta functions to use in the basis set.
        offset : int
            Number of time points to model before the event.

        """
        self.n = n
        self.offset = offset
        self.suffix = suffix

    def transform(self, x):
        """Generate a prediction basis set for the input as Toeplitz matrix.

        Parameters
        ----------
        input : array or series
            Input data; should be one-dimensional

        Returns
        -------
        output : 2D array or DataFrame
            Output data; has same type of input but will have multiple columns.

        """
        row = signal.unit_impulse(self.n, 0)
        x_full = np.concatenate([x, np.zeros(self.offset)])
        basis = linalg.toeplitz(x_full, row)[self.offset:]

        if isinstance(x, pd.Series):
            pad = int(np.floor(np.log10(self.n))) + 1
            cols = [
                f"{x.name}{self.suffix}{i:0{pad}d}" for i in range(self.n)
            ]
            basis = pd.DataFrame(basis, index=x.index, columns=cols)

        return basis


def condition_to_regressors(name, condition, hrf_model,
                            n_tp, tr, res, shift):
    """Generate design matrix columns from information about event occurrence.

    Parameters
    ----------
    name : string
        Condition name.
    condition : dataframe
        Event information corresponding to a single condition. Must have onset
        (in seconds), duration (in seconds), and value (in arbitrary units)
        columns; and should correspond to event occurrences.
    hrf_model : HRFModel object
        Object that implements `.transform()` to return a basis set for the
        predicted response.
    n_tp : int
        Number of time points in the final output.
    tr : float
        Time resolution of the output regressor, in seconds.
    res : float
        Sampling resolution at which to construct the regressor and perform
        convolution with the HRF model.
    shift : float
        Proportion of the TR to shift the predicted response when downsampling
        to the output resolution.

    Returns
    -------
    regressors : column(s)
        One or more output regressors that will form columns in the design
        matrix corresponding to this event type.

    """
    onset = condition["onset"]
    duration = condition["duration"]
    value = condition["value"]

    # Define hires and output resolution timepoints
    # TODO should output timepoints reflect shifting or not?
    hires_tps = np.arange(0, n_tp * tr + tr, 1 / res)
    tps = np.arange(0, n_tp * tr, tr)

    # Initialize the array that will be transformed
    hires_input = np.zeros_like(hires_tps, np.float)

    # Determine the time points at which each event starts and stops
    onset_at = np.round(onset * res).astype(int)
    offset_at = np.round((onset + duration) * res).astype(int)

    # Insert specified amplitudes for each event duration
    for start, end, value in zip(onset_at, offset_at, value):
        hires_input[start:(end + 1)] = value

    # Transform into a regressor basis set
    hires_input = pd.Series(hires_input, index=hires_tps, name=name)
    hires_output = hrf_model.transform(hires_input)

    # TODO It's annoying that we have to do this!
    if isinstance(hires_output, pd.Series):
        hires_output = (hires_output,)
    elif isinstance(hires_output, pd.DataFrame):
        hires_output = (col for _, col in hires_output.iteritems())

    # Downsample the predicted regressors to native sampling
    # TODO This crashes when hires_output is an ndarray
    output = []
    for hires_col in hires_output:
        col = interp1d(hires_tps, hires_col)(tps + shift)
        output.append(pd.Series(col, index=tps, name=hires_col.name))

    return tuple(output)


# TODO should this function get `res` from the `hrf_model`? They need to
# match and so they would always need to be double-specified...
def build_design_matrix(conditions=None, hrf_model=None,
                        regressors=None, artifacts=None,
                        n_tp=None, tr=1, res=60, shift=.5,
                        hpf_matrix=None, demean=True):
    """Use design information to build a matrix for a BOLD time series GLM.

    Parameters
    ----------
    conditions : dataframe
        Must have an `onset` (in seconds) columns.  Can also have `duration`
        (in seconds, defaulting to 0), and `value` (in arbitrary units,
        defaulting to 1), and `condition` (strings, defaulting to "event")
        columns; rows should correspond to event occurrences.
    hrf_model : HRFModel object
        Object that implements `.transform()` to return a basis set for the
        predicted response. Defaults to GammaHRF with default parameters.
    regressors : dataframe
        Additional columns to include in the design matrix without any
        transformation (aside from optional de-meaning). It must have an
        index with valid time points.
    artifacts : boolean series
        A Series indicating which row should have indicator regressors
        included in the design matrix to account for signal artifacts.
    n_tp : int
        The number of timepoints in the
    tr : float
        Time resolution of the output regressors, in seconds.
    res : float
        Sampling resolution at which to construct the condition regressors and
        perform convolution with the HRF model.
    shift : float
        Proportion of the TR to shift the predicted response when downsampling
        to the output resolution.
    hpf_matrix : n_tp x n_tp array
        Matrix for high-pass filtering the condition regressors.
    demean : bool
        If True, each column in the output matrix will be mean-centered.

    Returns
    -------
    X : dataframe
        Design matrix with timepoints in rows and regressors in columns.

    """
    if hrf_model is None:
        hrf_model = GammaHRF(res=res)

    # -- Default design size and quality control on input shapes

    if regressors is not None:
        n_reg_tp = len(regressors)
        n_tp = n_reg_tp if n_tp is None else n_tp
        if n_tp != n_reg_tp:
            err = "Size of regressors does not correspond with `n_tp`"
            raise ValueError(err)

    if artifacts is not None:  # TODO rename to censors?
        n_art_tp = len(artifacts)
        n_tp = n_art_tp if n_tp is None else n_tp
        if n_tp != n_art_tp:
            err = "Size of artifacts does not correspond with `n_tp`"
            raise ValueError(err)

    index = pd.Index(np.arange(0, n_tp * tr, tr))

    # -- Condition information (i.e. experimental design)
    design_components = []
    if conditions is not None:

        # Default values for some fields
        if "duration" not in conditions:
            conditions.loc[:, "duration"] = 0
        if "value" not in conditions:
            conditions.loc[:, "value"] = 1
        if "condition" not in conditions:
            conditions.loc[:, "condition"] = "event"

        # Build regressors for each condition
        condition_columns = []
        for name, info in conditions.groupby("condition"):
            cols = condition_to_regressors(name, info, hrf_model,
                                           n_tp, tr, res, shift)
            condition_columns.extend(cols)

        # Assemble the initial component of the design matrix
        condition_columns = pd.concat(condition_columns, axis=1)

        # High-pass filter the condition information
        # TODO revisit to decide whether only conditions should be filtered
        if hpf_matrix is not None:
            prefilter_means = condition_columns.mean()
            condition_columns.values[:] = hpf_matrix.dot(condition_columns)
            condition_columns += prefilter_means

        design_components.append(condition_columns)

    # -- Other regressors
    if regressors is not None:
        design_components.append(regressors)

    # -- Indicator regressors for signal artifacts
    if artifacts is not None:
        indicators = np.eye(n_tp)[:, artifacts.astype(np.bool)]
        columns = ["art{:02d}".format(i) for i in range(artifacts.sum())]
        indicators = pd.DataFrame(indicators, index, columns, np.float)
        design_components.append(indicators)

    # TODO Add polynomial regressors as alternative to HPF?

    # -- Final assembly
    X = pd.concat(design_components, axis=1)
    if demean:
        X -= X.mean()

    return X


def contrast_matrix(contrast, design_matrix):
    """Return a contrast matrix that is valid for a given design matrix.

    Parameters
    ----------
    contrast : tuple
        A tuple with (1) the name of the contrast, (2) the involved regressors,
        and (3) the weight to use for each of those regressors.
    design_matrix : dataframe
        Design matrix with regressor names corresponding to contrast elements.

    Returns
    -------
    contrast_matrix : series
        Contrast weights with regressor names as the index.

    """
    columns = design_matrix.columns.tolist()
    C = np.zeros(len(columns))
    _, names, weights = contrast
    for name, weight in zip(names, weights):
        C[columns.index(name)] = weight
    return C


# =========================================================================== #
# Generalized least squares estimation
# =========================================================================== #


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
    B_ols, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
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

    Imat = np.eye(n_tp)

    for i in range(n_vox):

        y_i, X_i = Y[..., i], X[..., i]
        XtXinv_i = pinv(dot(X_i.T, X_i))
        b_i = dot(XtXinv_i, dot(X_i.T, y_i))
        R_i = Imat - dot(X_i, dot(XtXinv_i, X_i.T))
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


# =========================================================================== #
# Temporal filtering
# =========================================================================== #
# TODO move to lyman.signals?


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

    Returns
    -------
    F : n_ntp x n_tp array
        Filter matrix.

    """
    if cutoff is None:
        return np.eye(n_tp)

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
