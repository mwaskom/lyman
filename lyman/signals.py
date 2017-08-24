import numpy as np
from scipy import signal, sparse, stats
from scipy.ndimage import gaussian_filter
import nibabel as nib


def compute_cov(data, detrend=True, mask=None):
    """Compute the temporal coefficient of variation.

    Parameters
    ----------
    data : numpy array
        Timeseries data with time in the final axis.
    detrend : bool
        If True, linearly detrend the data before computing coefficient of
        variation.

    Returns
    -------
    cov : numpy array
        Array with the coeficient of variation

    """
    if mask is not None:
        data = data[mask]

    mean = data.mean(axis=-1)
    if detrend:
        std = signal.detrend(data).std(axis=-1)
    else:
        std = data.std(axis=-1)

    # Compute temporal coefficient of variation
    cov = std / mean
    cov[mean == 0] = 0

    if mask is not None:
        out_cov = np.zeros_like(mask, np.float)
        out_cov[mask] = cov
        cov = out_cov

    return cov


def percent_change(data):
    """Convert data to percent signal change over final axis."""
    return (data / data.mean(axis=-1, keepdims=True) - 1) * 100


def identify_noisy_voxels(ts, mask, neighborhood=5, threshold=1.5,
                          detrend=True):
    """Create a mask of voxels that are unusually noisy given neighbors.

    Parameters
    ----------
    ts : nibabel image
        4D timeseries data.
    mask : nibabel image
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
    out : nibabel image
        3D binary image with 1s in the position of unusually noisy voxels.

    """
    data = ts.get_data()
    mask = mask.get_data().astype(bool)

    # Convert sigma units from mm to voxels
    sigma = neighborhood / np.array(ts.header.get_zooms()[:3])

    # Squelch divide-by-zero warnings
    with np.errstate(all="ignore"):

        # Compute temporal mean and standard deviation
        cov = compute_cov(data, detrend, mask)

        # Smooth the cov within the cortex mask
        smooth_norm = gaussian_filter(mask.astype(np.float), sigma)
        ribbon_cov = cov * mask
        neighborhood_cov = gaussian_filter(ribbon_cov, sigma) / smooth_norm

        # Compute the cov relative to the neighborhood
        relative_cov = ribbon_cov / neighborhood_cov
        relative_cov[~mask] = 0

    # Define and return a mask of noisy voxels
    noisy_voxels = relative_cov > threshold
    out = nib.Nifti1Image(noisy_voxels, ts.affine, ts.header)
    return out


def smoothing_matrix(surface, vertids, noisy_voxels=None, fwhm=2):
    """Define a matrix to smooth voxels using surface geometry.

    If T is an n_voxel x n_tp timeseries matrix, the resulting object S can
    be used to smooth the timeseries with the matrix operation S * T.

    Parameters
    ----------
    surface : Surface object
        Object representing the surface geometry that defines `.nvertices` and
        `.dijkstra_distance()`.
    vertids : 1d numpy array
        Array of vertex IDs corresponding to each cortical voxel.
    noisy_voxels : 1d numpy array
        Binary array defining voxels that should be excluded and interpolated
        during smoothing.
    fwhm : float
        Size of the smoothing kernel, in mm.

    Returns
    -------
    S : csr sparse matrix
        Matrix with smoothing weights.

    """
    # Define the weighting function
    assert fwhm > 0
    sigma = fwhm / 2.355
    norm = stats.norm(0, sigma)

    # Define the vertex ids that will be included in the smoothing
    if noisy_voxels is None:
        noisy_voxels = np.zeros_like(vertids)
    else:
        assert len(noisy_voxels) == len(vertids)
    clean = ~(noisy_voxels.astype(bool))
    clean_vertices = set(vertids[clean])

    # Define a mapping from vertex index to voxel index
    voxids = np.full(surface.nvertices, -1, np.int)
    for i, v in enumerate(vertids):
        voxids[v] = i

    # Initialize the sparse smoothing matrix
    n_voxels = len(vertids)
    mat_size = n_voxels, n_voxels
    S = sparse.lil_matrix(mat_size)

    # Build the matrix by rows
    for voxid, vertid in enumerate(vertids):

        # Find the distance to a minmimum number of neighboring voxels
        factor = 4
        neighbors = 0
        while neighbors < 6:
            d = surface.dijkstra_distance(vertid, sigma * factor)
            d = {vert: dist for vert, dist in d.items()
                 if vert in clean_vertices}
            neighbors = len(d)
            factor += 1

        # Find weights for nearby voxels
        verts, distances = zip(*d.items())
        voxels = voxids[list(verts)]
        w = norm.pdf(distances)
        w /= w.sum()

        # Update the matrix
        S[voxid, voxels] = w

    return S.tocsr()


def smooth_cortex(surface, ts, vertvol, noisy_voxels=None, fwhm=2):
    """Smooth voxels corresponding to cortex using surface-based distances.

    Parameters
    ----------
    surface : Surface object
        Object representing the surface geometry that defines `.nvertices` and
        `.dijkstra_distance()`.
    ts : nibabel image
        4D timeseries data.
    vertvol : nibabel image
        Image where voxels have their corresponding vertex ID or are -1 if not
        part of cortex.
    noisy_voxels : nibabel image
        Binary image defining voxels that should be excluded and interpolated
        during smoothing.
    fwhm : float
        Size of smoothing kernel in mm.

    Returns
    -------
    smooth_ts : nibabel image
        Image like `ts` but after smoothing.

    """
    # Load the data
    data = ts.get_data().copy()  # TODO allow inplace as an option
    vertvol = vertvol.get_data()
    ribbon = vertvol > -1
    if noisy_voxels is not None:
        noisy_voxels = noisy_voxels.get_data()

    # Extract the cortical data
    vertids = vertvol[ribbon]
    cortex_data = data[ribbon]
    if noisy_voxels is not None:
        noisy_voxels = noisy_voxels[ribbon]

    # Generate a smoothing matrix
    S = smoothing_matrix(surface, vertids, noisy_voxels, fwhm)

    # Smooth the data
    data[ribbon] = S * cortex_data

    return nib.Nifti1Image(data, ts.affine, ts.header)


def smooth_volume(ts, fwhm, mask=None, noise=None, mask_output=True):
    """Filter in volume space with an isotropic gaussian kernel.

    Parameters
    ----------
    ts : nibabel image
        4D timeseries data.
    fwhm : float
        Size of smoothing kernel in mm.
    mask : nibabel image
        3D binary image defining smoothing range.
    mask : nibabel image
        3D binary image defining noisy voxels to be interpolated out.
    mask_output : bool
        If True, apply the smoothing mask to the output.

    Returns
    -------
    smooth_ts : nibabel image
        Image like `ts` but after smoothing.

    """
    data = ts.get_data().copy()  # TODO allow inplace as an option
    if np.ndim(data) == 3:
        data = np.expanddims(data, 3)

    # TODO use nibabel function?
    sigma = np.divide(fwhm / 2.355, ts.header.get_zooms()[:3])

    if mask is None:
        smooth_mask = mask = norm = 1
    else:
        mask = mask.get_data().astype(np.bool)
        if noise is None:
            smooth_mask = mask.astype(np.float)
        else:
            noise = noise.get_data().astype(np.bool)
            smooth_mask = (mask & ~noise).astype(np.float)
        norm = gaussian_filter(smooth_mask, sigma)

    # TODO squelch divide by zero warnings
    with np.errstate(all="ignore"):
        for f in range(data.shape[-1]):
            data_f = gaussian_filter(data[..., f] * smooth_mask, sigma) / norm
            if mask_output:
                data_f[~mask] = 0
            data[..., f] = data_f

    return nib.Nifti1Image(data.squeeze(), ts.affine, ts.header)
