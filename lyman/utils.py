import numpy as np
import nibabel as nib


def image_to_matrix(img, mask_img):
    """Extract image data from voxels where mask is nonzero.

    Parameters
    ----------
    data : n_vox or n_tp, n_vox numpy array
        Data matrix; if a time series, time should be on the first axis.
    mask_img : 3D nifti image
        Image defining the voxels where data will be extracted. All nonzero
        voxels will be used; the mask does not have to be binary. Must be 3D.

    Returns
    -------
    data : n_vox or n_tp, n_vox numpy array
        Array with voxel data from ``img`` where the mask is nonzero. Note that
        when the input image is a time series, the time dimension is the first
        axis of the matrix (corresponding to the structure of a GLM).

    """
    vol_data = img.get_data()
    mask = mask_img.get_data() > 0
    data = vol_data[mask].T
    return data


def matrix_to_image(data, mask_img, template_img=None):
    """Convert a vector or matrix of data into a nibabel image.

    Parameters
    ----------
    data : n_vox or n_tp, n_vox numpy array
        Data matrix; if a time series, time should be on the first axis.
    mask_img : 3D nifti image
        Image defining which voxels in the output should be filled with
        ``data``. All nonzero voxels will be used; the mask does not have to be
        binary. Must be 3D.
    template_img : nifti image
        If present, the affine matrix and Nifti header will be assigned to the
        output image; otherwise, these are taken from ``mask_img`.

    Returns
    -------
    img : nifti image
        Image with voxel data from ``data`` where the mask is nonzero, with
        affine and header data from either the ``mask_img`` or, if present, the
        ``template_img``.

    """
    # Determine the volumetric image shape
    n_x, n_y, n_z = mask_img.shape
    try:
        n_tp, n_vox = data.shape
        vol_shape = n_x, n_y, n_z, n_tp
    except ValueError:
        n_vox, = data.shape
        vol_shape = n_x, n_y, n_z

    # Determine the basis for the output affine/header
    if template_img is None:
        template_img = mask_img

    # Put the data matrix into the volume where the mask is nonzero
    mask = mask_img.get_data() > 0
    vol_data = np.zeros(vol_shape, data.dtype)
    vol_data[mask] = data.T
    img = nib.Nifti1Image(vol_data, template_img.affine, template_img.header)

    return img
