import os.path as op
import subprocess as sp
import json

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from nipype.interfaces.base import BaseInterface, TraitedSpec, traits


class LymanInterface(BaseInterface):
    """Enhanced Interface object that custom interface should inherit from."""
    def __init__(self, **inputs):

        super(LymanInterface, self).__init__(**inputs)
        self._results = {}

    def _list_outputs(self):
        """Override BaseInterface._list_outputs using out _results dict."""
        return self._results

    def define_output(self, field, fname):
        """Set an interface output field using an absolute path to `fname`."""
        fname = op.abspath(fname)
        self._results[field] = fname
        return fname

    def write_image(self, field, fname, data, affine=None, header=None):
        """Write a nibabel image to disk and assign path to output field."""
        fname = self.define_output(field, fname)
        if isinstance(data, nib.Nifti1Image):
            img = data
        else:
            img = nib.Nifti1Image(data, affine, header)
        img.to_filename(fname)
        return img

    def write_visualization(self, field, fname, viz):
        """Write a visualization to disk and assign path to output field."""
        if viz is None:
            return

        fname = self.define_output(field, fname)

        if isinstance(viz, plt.Figure):
            viz.savefig(fname, dpi=100)
            plt.close(viz)
        elif hasattr(viz, "savefig"):
            viz.savefig(fname, close=True)
        else:
            raise RuntimeError(f"It is unknown how to plot {type(viz)} object")

    def submit_cmdline(self, runtime, cmdline):
        """Submit a command-line job and capture the output."""
        for attr in ["stdout", "stderr", "cmdline"]:
            if not hasattr(runtime, attr):
                setattr(runtime, attr, "")

        if runtime.get("returncode", None) is None:
            runtime.returncode = 0

        if isinstance(cmdline, list):
            cmdline = " ".join(cmdline)

        proc = sp.Popen(cmdline,
                        stdout=sp.PIPE,
                        stderr=sp.PIPE,
                        shell=True,
                        cwd=runtime.cwd,
                        env=runtime.environ,
                        universal_newlines=True)

        stdout, stderr = proc.communicate()

        runtime.stdout += "\n" + stdout + "\n"
        runtime.stderr += "\n" + stderr + "\n"
        runtime.cmdline += "\n" + cmdline + "\n"
        runtime.returncode += proc.returncode

        if proc.returncode is None or proc.returncode != 0:
            message = "\n\nCommand:\n" + runtime.cmdline + "\n"
            message += "Standard output:\n" + runtime.stdout + "\n"
            message += "Standard error:\n" + runtime.stderr + "\n"
            message += "Return code: " + str(runtime.returncode)
            raise RuntimeError(message)

        return runtime


class SaveInfo(LymanInterface):

    class input_spec(TraitedSpec):
        info_dict = traits.Dict()
        parameterization = traits.Either(traits.Str, traits.Tuple)

    class output_spec(TraitedSpec):
        info_file = traits.File(exists=True)

    def _run_interface(self, runtime):

        fname = self.define_output("info_file", "info.json")
        with open(fname, "w") as fid:
            json.dump(self.inputs.info_dict, fid, indent=2)
        return runtime


def image_to_matrix(img, mask_img, use=None):
    """Extract image data from voxels using segmentation mask.

    Parameters
    ----------
    img : nifti image
        3D or 4D nifti image with data to extract.
    mask_img : 3D nifti image
        Image defining the voxels where data will be extracted. Must be 3D.
    use : int, list of ints, or None, optional
        Value(s) within ``mask_img`` to use. If ``None``, use all nonzero.

    Returns
    -------
    data : n_vox or n_tp, n_vox numpy array
        Array with voxel data from ``img`` matching the mask. Note that when
        the input image is a time series, the time dimension is the first axis
        of the matrix (corresponding to the structure of a GLM).

    """
    vol_data = img.get_fdata()
    if use is None:
        mask = mask_img.get_fdata() > 0
    else:
        mask = np.isin(mask_img.get_fdata(), use)
    check_mask(mask, vol_data)
    data = vol_data[mask].T
    return data


def matrix_to_image(data, mask_img, template_img=None, use=None):
    """Convert a vector or matrix of data into a nibabel image.

    Parameters
    ----------
    data : n_vox or n_tp, n_vox numpy array
        Data matrix; if a time series, time should be on the first axis.
    mask_img : 3D nifti image
        Image defining which voxels in the output should be filled with
        ``data``. Must be 3D.
    template_img : nifti image
        If present, the affine matrix and Nifti header will be assigned to the
        output image; otherwise, these are taken from ``mask_img``.
    use : int, list of ints, or None, optional
        Value(s) within ``mask_img`` to use. If ``None``, use all nonzero.

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
    if use is None:
        mask = mask_img.get_fdata() > 0
    else:
        mask = np.isin(mask_img.get_fdata(), use)
    vol_data = np.zeros(vol_shape, data.dtype)
    vol_data[mask] = data.T
    img = nib.Nifti1Image(vol_data, template_img.affine, template_img.header)

    return img


def check_mask(mask, data):
    """Check the dtype and shape of a mask array.

    This will raise a TypeError if ``mask`` is not boolean or a ValueError if
    the shape of the mask does not match the first (n - 1) dimensions of the
    data.

    Parameters
    ----------
    mask : boolean numpy array
        Mask array to validate.
    data : numpy array
        Data that the mask will index into.

    """
    if mask.dtype != np.bool:
        raise TypeError("mask must have boolean datatype")
    if (mask.shape != data.shape) and (mask.shape != data.shape[:-1]):
        msg = ("mask shape {} is not aligned with data shape {}"
               .format(mask.shape, data.shape))
        raise ValueError(msg)
