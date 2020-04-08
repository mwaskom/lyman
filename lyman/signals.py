import numpy as np
from scipy import signal, sparse, stats, linalg
from scipy.ndimage import gaussian_filter
import nibabel as nib

from .utils import check_mask
from .surface import SurfaceMeasure


def detrend(data, axis=-1, replace_mean=False):
    """Linearly detrend on an axis, optionally replacing the original mean.

    Parameters
    ----------
    data : array_like
        The input data.
    axis : int
        The axis along which to detrend the data.
    replace_mean : bool
        If True, compute the mean before detrending then add it back after.

    Returns
    -------
    data : array_like
        Detrended data.

    """
    # TODO enhance to preserve pandas index information
    # TODO enhance to remove higher-order polynomials?
    if replace_mean:
        orig_mean = np.mean(data, axis=axis, keepdims=True)

    data = signal.detrend(data, axis=axis)

    if replace_mean:
        data += orig_mean

    return data


def cv(data, axis=0, detrend=True, mask=None, keepdims=False, ddof=0):
    """Compute the temporal coefficient of variation.

    Parameters
    ----------
    data : numpy array
        Time series data.
    axis : int
        Axis to compute the mean and standard deviation over.
    detrend : bool
        If True, linearly detrend the data before computing standard deviation.
    mask : boolean array with same shape as data on all but final axis
        If present, cv is computed only for entries that are True in the mask.
        Only valid when the computation axis is the final axis in the data.
        Values in the returned array outside of the mask will be 0.
    keepdims : bool
        If True, output will have the same dimensions as input.
    ddof : int
        Means delta degrees of freedom for the standard deviation.

    Returns
    -------
    cv : numpy array
        Array with the coeficient of variation values.

    """
    if mask is not None:
        check_mask(mask, data)
        data = data[mask]

    mean = data.mean(axis=axis)
    if detrend:
        data = signal.detrend(data, axis=axis)
    std = data.std(axis=axis, ddof=ddof)

    with np.errstate(all="ignore"):
        cv = std / mean
        cv[mean == 0] = 0

    if mask is not None:
        out = np.zeros(mask.shape, np.float)
        out[mask] = cv
        cv = out

    if keepdims:
        cv = np.expand_dims(cv, axis=axis)

    return cv


def percent_change(data, axis=-1):
    """Convert data to percent signal change over specified axis.

    Parameters
    ----------
    data : numpy array
        Input data to convert to percent signal change.
    axis : int
        Axis of the ``data`` array to compute mean over.

    Returns
    -------
    pch_data : numpy array
        Input data divided by its mean and multiplied by 100.

    """
    data = np.asarray(data).astype(np.float)
    return (data / data.mean(axis=axis, keepdims=True) - 1) * 100


def identify_noisy_voxels(ts_img, mask_img, neighborhood=5, threshold=1.5,
                          detrend=True):
    """Create a mask of voxels that are unusually noisy given neighbors.

    Parameters
    ----------
    ts_img : nibabel image
        4D timeseries data.
    mask_img : nibabel image
        3D binary mask of relevant voxels
    neighborhood : float
        FWHM (in mm) to define local neighborhood for voxels.
    threshold : float
        Voxels with a relative coefficient of variation above this level
        will be defined as noisy
    detrend : bool
        If True, linearly detrend the data before computing coefficient of
        variation.

    Returns
    -------
    noise_img : nibabel image
        3D binary image with 1s in the position of locally noisy voxels.

    """
    affine, header = ts_img.affine, ts_img.header
    data = ts_img.get_fdata()
    mask = mask_img.get_fdata().astype(bool)
    check_mask(mask, data)

    # Compute temporal coefficient of variation
    data_cv = cv(data, axis=-1, detrend=detrend, mask=mask)

    # Smooth the cov within the cortex mask
    cv_img = nib.Nifti1Image(data_cv, affine, header)
    smooth_cv = smooth_volume(cv_img, neighborhood, mask_img).get_fdata()

    # Compute the cov relative to the neighborhood
    with np.errstate(all="ignore"):
        relative_cv = data_cv / smooth_cv
        relative_cv[~mask] = 0

    # Define and return a mask of noisy voxels
    noise_voxels = relative_cv > threshold
    noise_img = nib.Nifti1Image(noise_voxels, affine, header)
    return noise_img


def voxel_sigmas(fwhm, img):
    """Convert isotropic fwhm in mm to an array of voxelwise sigmas."""
    zooms = img.header.get_zooms()[:3]
    sigma_mm = fwhm / (2 * np.sqrt(2 * np.log(2)))
    sigma_vox = np.divide(sigma_mm, zooms)
    return sigma_vox


def smooth_volume(data_img, fwhm, mask_img=None, noise_img=None,
                  inplace=False):
    """Filter volume data with an isotropic gaussian kernel.

    Smoothing can be constrained to occur within the voxels defined within
    ``mask_img`` and optionally can ignore/interpolate out the voxels
    identified within ``noise_img``.

    Parameters
    ----------
    data_img : nibabel image
        3D or 4D image data.
    fwhm : positive float or None
        Size of isotropic smoothing kernel, in mm, or None to return input
        (as float and possibly copied).
    mask_img : nibabel image
        3D binary image defining smoothing range.
    noise_img : nibabel image
        3D binary image defining voxels to be interpolated out.
    inplace : bool
        If True, overwrite data in data_img. Otherwise perform a copy.

    Returns
    -------
    smooth_data : nibabel image
        Image like ``data_img`` but after smoothing.

    """
    data = _load_float_data_maybe_copy(data_img, inplace)

    if fwhm is None or fwhm == 0:
        return nib.Nifti1Image(data, data_img.affine, data_img.header)

    if np.ndim(data) == 3:
        need_squeeze = True
        data = np.expand_dims(data, -1)
    else:
        need_squeeze = False

    if mask_img is None:
        mask = np.ones(data.shape[:3], np.bool)
    else:
        mask = mask_img.get_fdata().astype(np.bool)
    smooth_from = mask.copy()

    if noise_img is not None:
        smooth_from &= ~noise_img.get_fdata().astype(np.bool)

    sigma = voxel_sigmas(fwhm, data_img)
    norm = gaussian_filter(smooth_from.astype(np.float), sigma)
    valid_norm = norm > 0

    for f in range(data.shape[-1]):
        with np.errstate(all="ignore"):
            data_f = gaussian_filter(data[..., f] * smooth_from, sigma) / norm
        data_f[~valid_norm] = 0
        data[..., f] = data_f

    data[~mask] = 0

    if need_squeeze:
        data = data.squeeze()

    return nib.Nifti1Image(data, data_img.affine, data_img.header)


def smooth_segmentation(data_img, seg_img, fwhm, noise_img=None,
                        inplace=False):
    """Filter each compartment of a segmentation with an isotropic gaussian.

    Parameters
    ----------
    data_img : nibabel image
        3D or 4D image data.
    seg_img : nibabel image
        3D label image defining smoothing ranges.
    fwhm : positive float or None
        Size of isotropic smoothing kernel, in mm, or None to return input
        (as float and possibly copied).
    noise_img : nibabel image
        3D binary image defining voxels to be interpolated out.
    inplace : bool
        If True, overwrite data in data_img. Otherwise perform a copy.

    Returns
    -------
    smooth_data : nibabel image
        Image like ``data_img`` but after smoothing.

    """
    affine, header = data_img.affine, data_img.header
    data = _load_float_data_maybe_copy(data_img, inplace)

    seg = seg_img.get_fdata().astype(np.int)
    seg_ids = np.sort(np.unique(seg))

    for id in seg_ids:

        if not id:
            continue

        id_mask = seg == id
        id_data = np.zeros_like(data)
        id_data[id_mask] = data[id_mask]

        id_mask_img = nib.Nifti1Image(id_mask.astype(np.int), affine)
        id_data_img = nib.Nifti1Image(id_data, affine)
        smooth_volume(id_data_img, fwhm, id_mask_img, noise_img, inplace=True)
        data[id_mask] = id_data_img.get_fdata()[id_mask].astype(np.int)

    return nib.Nifti1Image(data, affine, header)


def smoothing_matrix(measure, vertids, fwhm, exclude=None, minpool=6):
    """Define a matrix to smooth voxels using surface geometry.

    If T is an n_voxel x n_tp timeseries matrix, the resulting object S can
    be used to smooth the timeseries with the matrix operation S * T.

    Parameters
    ----------
    measure : surface.SurfaceMeasure object
        Object for measuring distance along a cortical mesh.
    vertids : 1d numpy array
        Array of vertex IDs corresponding to each cortical voxel.
    fwhm : positive float or None
        Size of the smoothing kernel, in mm, or None to return identity matrix.
    exclude : 1d numpy array
        Binary array defining voxels that should be excluded and interpolated
        during smoothing.
    minpool : int
        Minimum number of neighborhood vertices to include in smoothing pool.

    Returns
    -------
    S : csr sparse matrix
        Matrix with smoothing weights.

    """
    # Handle null smoothing
    if fwhm is None:
        return sparse.diags(np.ones_like(vertids)).tocsr()

    # Define the weighting function
    if fwhm <= 0:
        raise ValueError("Smoothing kernel fwhm must be positive")
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    norm = stats.norm(0, sigma)

    # Define the vertex ids that will be included in the smoothing
    if exclude is None:
        exclude = np.zeros_like(vertids)
    clean = ~(exclude.astype(bool))
    clean_verts = set(vertids[clean])

    # Define a mapping from vertex index to voxel index
    voxids = np.full(measure.n_v, -1, np.int)
    for i, v in enumerate(vertids):
        voxids[v] = i

    # Initialize the sparse smoothing matrix
    n_voxels = len(vertids)
    mat_size = n_voxels, n_voxels
    S = sparse.lil_matrix(mat_size)

    # Ensure that the minpool isn't larger than the surface
    minpool = min(minpool, clean.sum())

    # Build the matrix by rows
    for voxid, vertid in enumerate(vertids):

        # Find the distance to a minmimum number of neighboring voxels
        factor = 4
        pool = 0
        while pool < minpool:
            all_dist = measure(vertid, sigma * factor)
            distmap = {v: d for v, d in all_dist.items() if v in clean_verts}
            pool = len(distmap)
            factor += 1
            if factor > 10:
                # Error out to avoid an infinite loop
                raise RuntimeError("Could not find enough neighbors in mesh")

        # Find weights for nearby voxels
        verts, distances = zip(*distmap.items())
        voxels = voxids[list(verts)]
        w = norm.pdf(distances)
        w /= w.sum()

        # Update the matrix
        S[voxid, voxels] = w

    return S.tocsr()


def smooth_surface(data_img, vert_img, fwhm, subject, surf="graymid",
                   noise_img=None, inplace=False, subjects_dir=None):
    """Smooth cortical voxels with Gaussian weighted surface distances.

    Parameters
    ----------
    data_img : nibabel image
        3D or 4D input data.
    vert_img : 4D nibabel image
        Image where voxels have their corresponding surfave vertex ID or are -1
        if they do not correspond to cortex. The first frame should have left
        hemisphere vertices and the second right hemisphere vertices.
    fwhm : positive float or None
        Size of isotropic smoothing kernel, in mm, or None to return input
        (as float and possibly copied).
    subject : string
        Subject ID to locate data in data directory.
    surf : string
        Name of the surface defining the mesh geometry.
    noise_img : nibabel image
        Binary image defining voxels that should be interpolated out.
    inplace :bool
        If True, overwrite data in data_img. Otherwise perform a copy.
    subjects_dir : string
        Path to the Freesurfer data directory root; if absent, get from the
        SUBJECTS_DIR environment variable.

    Returns
    -------
    smooth_img : nibabel image
        Image like ``data_img`` but after smoothing.

    """
    # ---
    # Load the data
    data = _load_float_data_maybe_copy(data_img, inplace)
    vertvol = vert_img.get_fdata().astype(np.int)
    noise = None if noise_img is None else noise_img.get_fdata()

    if not (len(vertvol.shape) == 4) and (vertvol.shape[-1] == 2):
        raise ValueError("`vert_img` must have two frames (lh and rh verts)")

    for i, hemi in enumerate(["lh", "rh"]):

        # Extract the cortical data
        ribbon = vertvol[..., i] > -1
        vertids = vertvol[ribbon, i]
        surf_data = data[ribbon]
        if noise is None:
            surf_noise = None
        else:
            surf_noise = noise[ribbon]

        # Generate a smoothing matrix
        measure = SurfaceMeasure.from_names(subject, hemi, surf, subjects_dir)
        S = smoothing_matrix(measure, vertids, fwhm, surf_noise)

        # Smooth the data
        data[ribbon] = S * surf_data

    return nib.Nifti1Image(data, data_img.affine, data_img.header)


def _load_float_data_maybe_copy(img, inplace):
    """Load data from an image and convert to a float dtype, optionally copying.

    This function preserves input float dtypes but converts others to np.float.

    """
    dtype = img.get_data_dtype()
    if np.issubdtype(dtype, np.floating):
        to_dtype = dtype
    else:
        to_dtype = np.float
        if inplace:
            raise ValueError("Cannot operate on non-float data in place")

    return img.get_fdata(dtype=to_dtype).astype(to_dtype, copy=not inplace)


def pca_transform(data, keep=None, whiten=True):
    """Transform data matrix using PCA.

    Parameters
    ----------
    data : n_tp x n_vox array
        Data to be transformed.
    keep : int, optional
        Number of components to return.
    whiten : bool, optional
        If True, components will have unit variance.

    Returns
    -------
    out : n_tp x keep array
        Data projected onto principle components, and maybe whitened.

    """
    if keep is None:
        keep = data.shape[1]

    # Perform singular value decomposition
    data = data - data.mean(axis=0)
    u, s, v = linalg.svd(data, full_matrices=False)

    # Flip eigenvector signs to enforce deterministic output
    # Code taken from sckit-learn PCA implementaiton
    max_abs_cols = np.argmax(np.abs(u), axis=0)
    signs = np.sign(u[max_abs_cols, range(u.shape[1])])
    u *= signs
    v *= signs[:, np.newaxis]

    # Tranform the data
    if whiten:
        out = u * np.sqrt(data.shape[0] - 1)
    else:
        out = u * s

    return out[:, :keep]
