from __future__ import division
import os
import numpy as np
from scipy import ndimage
import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib
from six import string_types


class Mosaic(object):

    def __init__(self, anat=None, stat=None, mask=None, n_col=9, step=2,
                 tight=True, show_mask=True, slice_dir="axial",
                 anat_lims=None):
        """Plot a mosaic of axial slices through an MRI volume.

        Parameters
        ----------
        anat : filename, nibabel image, or array
            The anatomical image that will form the background of the mosaic.
            If only an array is passed, an identity matrix will be used as
            the affine and orientation could be incorrect. If absent, try
            to find the FSL data and uses the MNI152 brain.
        stat : filename, nibabel image, or array
            A statistical map to plot as an overlay (which happens by calling
            one of the methods). If only an array is passed, it is assumed
            to have the same orientation as the anatomy.
        mask : filename, nibabel image, or array
            A binary image where voxels included in the statistical analysis
            are True. This will be used to gray-out voxels in the anatomical
            image that are outside the field of view. If you want to overlay
            the mask itself, pass it to ``stat``.
        n_col : int
            Number of columns in the mosaic.
        step : int
            Take every ``step`` slices along the slice_dir for the mosaic.
        tight : bool
            If True, try to crop panes to focus on the brain volume.
        show_mask : bool
            If True, gray-out voxels in the anat image that are outside
            of the mask image.
        slice_dir : axial | coronal | sagital
            Direction to slice the mosaic on.
        anat_lims : pair of floats
            Limits for the anatomical (background) image colormap

        """

        # Load and reorient the anatomical image
        if anat is None:
            if "FSLDIR" in os.environ:
                anat = os.path.join(os.environ["FSLDIR"],
                                    "data/standard/avg152T1_brain.nii.gz")
        if isinstance(anat, string_types):
            anat_img = nib.load(anat)
            have_orientation = True
        elif isinstance(anat, np.ndarray):
            anat_img = nib.Nifti1Image(anat, np.eye(4))
            have_orientation = False
        else:
            anat_img = anat
            have_orientation = True
        self.anat_img = nib.as_closest_canonical(anat_img)
        self.anat_data = self.anat_img.get_data()

        # Load and reorient the statistical image
        if stat is not None:
            if isinstance(stat, string_types):
                stat_img = nib.load(stat)
            elif isinstance(stat, np.ndarray):
                if stat.dtype is np.dtype("bool"):
                    stat = stat.astype(np.int)
                stat_img = nib.Nifti1Image(stat,
                                           anat_img.affine,
                                           anat_img.header)
            else:
                stat_img = stat
            self.stat_img = nib.as_closest_canonical(stat_img)
        # Load and reorient the mask image
        if mask is not None:
            if isinstance(mask, string_types):
                mask_img = nib.load(mask)
            elif isinstance(mask, np.ndarray):
                if mask.dtype is np.dtype("bool"):
                    mask = mask.astype(np.int)
                mask_img = nib.Nifti1Image(mask,
                                           anat_img.affine,
                                           anat_img.header)
            else:
                mask_img = mask
            self.mask_img = nib.as_closest_canonical(mask_img)
            mask_data = self.mask_img.get_data().astype(bool)
        else:
            mask_data = None

        if slice_dir[0] not in "sca":
            err = "Slice direction {} not understood".format(slice_dir)
            raise ValueError(err)

        # Find a field of view that tries to eliminate empty voxels
        anat_fov = self.anat_img.get_data() > 1e-5
        if tight:
            self.fov = anat_fov
            if mask is not None:
                self.fov &= mask_data
        else:
            self.fov = np.ones_like(anat_fov)

        # Save the mosaic parameters
        self.n_col = n_col
        self.step = step
        self.slice_dir = slice_dir

        # Define slice objects to crop to the volume
        slices, = ndimage.find_objects(self.fov)
        self.x_slice, self.y_slice, self.z_slice = slices

        # Update the slice on the mosiac axis with steps
        slice_ax = dict(s="x", c="y", a="z")[slice_dir[0]]
        ms = getattr(self, slice_ax + "_slice")
        mosaic_slice = slice(ms.start, ms.stop, step)
        setattr(self, slice_ax + "_slice", mosaic_slice)
        self.n_slices = (ms.stop - ms.start) // step

        # Initialize the figure and plot the constant info
        self._setup_figure()
        self._plot_anat(anat_lims)
        if mask is not None and show_mask:
            self._plot_inverse_mask()

        # Label the anatomy
        if have_orientation:
            l_label, r_label = dict(s="PA", c="LR", a="LR")[self.slice_dir[0]]
            self.fig.text(.01, .03, l_label, size=14, color="w",
                          ha="left", va="center")
            self.fig.text(.99, .03, r_label, size=14, color="w",
                          ha="right", va="center")

    def _setup_figure(self):
        """Initialize the figure and axes."""
        n_row = np.ceil(self.n_slices / self.n_col)
        if self.slice_dir.startswith("s"):
            slc_i, slc_j = self.y_slice, self.z_slice
        elif self.slice_dir.startswith("c"):
            slc_i, slc_j = self.x_slice, self.z_slice
        elif self.slice_dir.startswith("a"):
            slc_i, slc_j = self.x_slice, self.y_slice
        nx, ny, _ = self.anat_data[slc_i, slc_j].shape
        figsize = self.n_col, (ny / nx) * n_row
        plot_kws = dict(nrows=int(n_row), ncols=int(self.n_col),
                        figsize=figsize, facecolor="k")

        self.fig, self.axes = plt.subplots(**plot_kws)
        [ax.set_axis_off() for ax in self.axes.flat]
        self.fig.subplots_adjust(0, 0, 1, 1, 0, 0)

    def _plot_anat(self, lims=None):
        """Plot the anatomy in grayscale."""
        anat_data = self.anat_img.get_data()
        if lims is None:
            vmin, vmax = 0, np.percentile(anat_data[self.fov], 99)
        else:
            vmin, vmax = lims
        anat_fov = anat_data[self.x_slice, self.y_slice, self.z_slice]
        self._map("imshow", anat_fov, cmap="gray", vmin=vmin, vmax=vmax)

        empty_slices = len(self.axes.flat) - anat_fov.shape[2]
        if empty_slices > 0:
            i, j, _ = anat_fov.shape
            for ax in self.axes.flat[-empty_slices:]:
                ax.imshow(np.zeros((i, j)), cmap="gray", vmin=0, vmax=10)

    def _plot_inverse_mask(self):
        """Dim the voxels outside of the statistical analysis FOV."""
        mask_data = self.mask_img.get_data().astype(np.bool)
        anat_data = self.anat_img.get_data()
        mask_data = np.where(mask_data | (anat_data < 1e-5), np.nan, 1)
        mask_fov = mask_data[self.x_slice, self.y_slice, self.z_slice]
        self._map("imshow", mask_fov, cmap="bone", vmin=0, vmax=3,
                  interpolation="nearest", alpha=.5)

    def _map(self, func_name, data, ignore_value_error=False, **kwargs):
        """Apply a named function to a 3D volume of data on each axes."""
        trans_order = dict(s=(0, 1, 2),
                           c=(1, 0, 2),
                           a=(2, 0, 1))[self.slice_dir[0]]
        slices = data.transpose(*trans_order)
        for slice, ax in zip(slices, self.axes.flat):
            func = getattr(ax, func_name)
            try:
                func(np.rot90(slice), **kwargs)
            except ValueError:
                if ignore_value_error:
                    pass
                else:
                    raise

    def plot_activation(self, thresh=2, vmin=None, vmax=None, vmax_perc=99,
                        vfloor=None, pos_cmap="Reds_r", neg_cmap=None,
                        alpha=1, fmt=".2g"):
        """Plot the stat image as uni- or bi-polar activation with a threshold.

        Parameters
        ----------
        thresh : float
            Threshold value for the statistic; overlay will not be visible
            between -thresh and thresh.
        vmin, vmax : floats
            The anchor values for the colormap. The same values will be used
            for the positive and negative overlay.
        vmax_perc : int
            The percentile of the data (within the fov and above the threshold)
            at which to saturate the colormap by default. Overriden if a there
            is a specific value passed for vmax.
        vfloor : float or None
            If not None, this sets the vmax value, if the value at the provided
            vmax_perc does not exceed it.
        pos_cmap, neg_cmap : names of colormaps or colormap objects
            The colormapping for the positive and negative overlays.
        alpha : float
            The transparancy of the overlay.
        fmt : {}-style format key
            Format of the colormap annotation.

        """
        stat_data = self.stat_img.get_data()[self.x_slice,
                                             self.y_slice,
                                             self.z_slice]
        pos_data = stat_data.copy()
        pos_data[pos_data < thresh] = np.nan
        if vmin is None:
            vmin = thresh
        if vmax is None:
            calc_data = stat_data[np.abs(stat_data) > thresh]
            if calc_data.any():
                vmax = np.percentile(np.abs(calc_data), vmax_perc)
            else:
                vmax = vmin * 2

        pos_cmap = self._get_cmap(pos_cmap)

        self._map("imshow", pos_data, cmap=pos_cmap,
                  vmin=vmin, vmax=vmax, alpha=alpha)

        if neg_cmap is not None:
            thresh, nvmin, nvmax = -thresh, -vmax, -vmin
            neg_data = stat_data.copy()
            neg_data[neg_data > thresh] = np.nan

            neg_cmap = self._get_cmap(neg_cmap)

            self._map("imshow", neg_data, cmap=neg_cmap,
                      vmin=nvmin, vmax=nvmax, alpha=alpha)

            self._add_double_colorbar(vmin, vmax, pos_cmap, neg_cmap, fmt)
        else:
            self._add_single_colorbar(vmin, vmax, pos_cmap, fmt)

    def plot_overlay(self, cmap, vmin=None, vmax=None, center=False,
                     vmin_perc=1, vmax_perc=99, thresh=None,
                     alpha=1, fmt=".2g", colorbar=True):
        """Plot the stat image as a single overlay with a threshold.

        Parameters
        ----------
        cmap : name of colormap or colormap object
            The colormapping for the overlay.
        vmin, vmax : floats
            The anchor values for the colormap. The same values will be used
            for the positive and negative overlay.
        center : bool
            If true, center the colormap. This respects the larger absolute
            value from the other (vmin, vmax) arguments, but overrides the
            smaller one.
        vmin_perc, vmax_perc : ints
            The percentiles of the data (within fov and above threshold)
            that will be anchor points for the colormap by default. Overriden
            if specific values are passed for vmin or vmax.
        thresh : float
            Threshold value for the statistic; overlay will not be visible
            between -thresh and thresh.
        alpha : float
            The transparancy of the overlay.
        fmt : {}-style format string
            Format of the colormap annotation.
        colorbar : bool
            If true, add a colorbar.

        """
        stat_data = self.stat_img.get_data()[self.x_slice,
                                             self.y_slice,
                                             self.z_slice]
        if hasattr(self, "mask_img"):
            fov = self.mask_img.get_data()[self.x_slice,
                                           self.y_slice,
                                           self.z_slice].astype(bool)
        else:
            fov = np.ones_like(stat_data).astype(bool)

        if vmin is None:
            vmin = np.percentile(stat_data[fov], vmin_perc)
        if vmax is None:
            if stat_data.any():
                vmax = np.percentile(stat_data[fov], vmax_perc)
            else:
                vmax = vmin * 2
        if center:
            vabs = max(np.abs(vmin), vmax)
            vmin, vmax = -vabs, vabs
        if thresh is not None:
            stat_data[stat_data < thresh] = np.nan

        stat_data[~fov] = np.nan

        cmap = self._get_cmap(cmap)

        self._map("imshow", stat_data, cmap=cmap,
                  vmin=vmin, vmax=vmax, alpha=alpha)

        if colorbar:
            self._add_single_colorbar(vmin, vmax, cmap, fmt)

    def plot_mask(self, color="#dd2222", alpha=.66):
        """Plot the statistical volume as a binary mask."""
        mask_data = self.stat_img.get_data()[self.x_slice,
                                             self.y_slice,
                                             self.z_slice]
        bool_mask = mask_data.astype(bool)
        mask_data = bool_mask.astype(np.float)
        mask_data[~bool_mask] = np.nan

        cmap = mpl.colors.ListedColormap([color])
        self._map("imshow", mask_data, cmap=cmap, vmin=.5, vmax=1.5,
                  interpolation="nearest", alpha=alpha)

    def plot_mask_edges(self, color="#dd2222", linewidth=1):
        """Plot the edges of possibly multiple masks to show overlap."""
        cmap = mpl.colors.ListedColormap([color])

        slices = self.stat_img.get_data()[self.x_slice,
                                          self.y_slice,
                                          self.z_slice]

        self._map("contour", slices, ignore_value_error=True,
                  levels=[0, 1], cmap=cmap, vmin=0, vmax=1,
                  linewidths=linewidth)

    def map(self, func_name, data, thresh=None, **kwargs):
        """Map a dataset across the mosaic of axes.

        Parameters
        ----------
        func_name : str
            Name of a pyplot function.
        data : filename, nibabel image, or array
            Dataset to plot.
        thresh : float
            Don't map voxels in ``data`` below this threshold.
        kwargs : key, value mappings
            Other keyword arguments are passed to the plotting function.

        """
        if isinstance(data, string_types):
            data_img = nib.load(data)
        elif isinstance(data, np.ndarray):
            data_img = nib.Nifti1Image(data, np.eye(4))
        else:
            data_img = data
        data_img = nib.as_closest_canonical(data_img)
        data = data_img.get_data()
        data = data.astype(np.float)
        if thresh is not None:
            data[data < thresh] = np.nan
        data = data[self.x_slice, self.y_slice, self.z_slice]
        self._map(func_name, data, **kwargs)

    def _pad_for_cbar(self):
        """Add extra space to the bottom of the figure for the colorbars."""
        w, h = self.fig.get_size_inches()
        cbar_inches = .3
        self.fig.set_size_inches(w, h + cbar_inches)
        cbar_height = cbar_inches / (h + cbar_inches)
        self.fig.subplots_adjust(0, cbar_height, 1, 1)

        #  Needed so things look nice in the notebook
        bg_ax = self.fig.add_axes([0, 0, 1, cbar_height])
        bg_ax.set_axis_off()
        bg_ax.pcolormesh(np.array([[1]]), cmap="Greys", vmin=0, vmax=1)

        return cbar_height

    def _add_single_colorbar(self, vmin, vmax, cmap, fmt):
        """Add colorbars for a single overlay."""
        cbar_height = self._pad_for_cbar()
        cbar_ax = self.fig.add_axes([.3, .01, .4, cbar_height - .01])
        cbar_ax.set(xticks=[], yticks=[])
        for side, spine in cbar_ax.spines.items():
            spine.set_visible(False)

        bar_data = np.linspace(0, 1, 256).reshape(1, 256)
        cbar_ax.pcolormesh(bar_data, cmap=cmap)

        if fmt is not None:

            fmt = "{:" + fmt + "}"

            self.fig.text(.29, .005 + cbar_height * .5, fmt.format(vmin),
                          color="white", size=14, ha="right", va="center")
            self.fig.text(.71, .005 + cbar_height * .5, fmt.format(vmax),
                          color="white", size=14, ha="left", va="center")

    def _add_double_colorbar(self, vmin, vmax, pos_cmap, neg_cmap, fmt):
        """Add colorbars for a positive and a negative overlay."""
        cbar_height = self._pad_for_cbar()

        pos_ax = self.fig.add_axes([.55, .01, .3, cbar_height - .01])
        pos_ax.set(xticks=[], yticks=[])
        for side, spine in pos_ax.spines.items():
            spine.set_visible(False)

        neg_ax = self.fig.add_axes([.15, .01, .3, cbar_height - .01])
        neg_ax.set(xticks=[], yticks=[])
        for side, spine in neg_ax.spines.items():
            spine.set_visible(False)

        bar_data = np.linspace(0, 1, 256).reshape(1, 256)
        pos_ax.pcolormesh(bar_data, cmap=pos_cmap)
        neg_ax.pcolormesh(bar_data, cmap=neg_cmap)

        if fmt is not None:

            fmt = "{:" + fmt + "}"

            self.fig.text(.54, .005 + cbar_height * .5, fmt.format(vmin),
                          color="white", size=14, ha="right", va="center")
            self.fig.text(.86, .005 + cbar_height * .5, fmt.format(vmax),
                          color="white", size=14, ha="left", va="center")

            self.fig.text(.14, .005 + cbar_height * .5, fmt.format(-vmax),
                          color="white", size=14, ha="right", va="center")
            self.fig.text(.46, .005 + cbar_height * .5, fmt.format(-vmin),
                          color="white", size=14, ha="left", va="center")

    def _get_cmap(self, cmap):
        """Parse a string spec of a cubehelix palette."""
        if isinstance(cmap, string_types):
            if cmap.startswith("cube"):
                if cmap.endswith("_r"):
                    reverse = False
                    cmap = cmap[:-2]
                else:
                    reverse = True
                _, start, rot = cmap.split(":")
                cube_rgb = mpl._cm.cubehelix(s=float(start),
                                             r=float(rot))
                cube_cmap = mpl.colors.LinearSegmentedColormap(cmap, cube_rgb)
                lut = cube_cmap(np.linspace(.95, 0, 256))
                if reverse:
                    lut = lut[::-1]
                cmap = mpl.colors.ListedColormap(lut)
        return cmap

    def savefig(self, fname, **kwargs):
        """Save the figure."""
        self.fig.savefig(fname, facecolor="k", edgecolor="k", **kwargs)

    def close(self):
        """Close the figure."""
        plt.close(self.fig)
