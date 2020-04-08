from __future__ import division
import warnings
from six import string_types

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import ndimage, signal
import nibabel as nib


class Mosaic(object):

    def __init__(self, anat, stat=None, mask=None, n_col=9, step=2,
                 tight=True, show_mask=True, slice_dir="axial",
                 anat_lims=None, title=None):
        """Plot a mosaic of axial slices through an MRI volume.

        Parameters
        ----------
        anat : filename, nibabel image, or array
            The anatomical image that will form the background of the mosaic.
            If only an array is passed, an identity matrix will be used as
            the affine and orientation could be incorrect.
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
            Number of columns in the mosaic. This will also determine the size
            of the figure (1 inch per column).
        step : int
            Show every ``step`` slice along the slice_dir in the mosaic.
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
        # -- Load and reorient the anatomical image

        if isinstance(anat, string_types):
            anat_img = nib.load(anat)
            have_orientation = True
        elif isinstance(anat, nib.spatialimages.SpatialImage):
            anat_img = anat
            have_orientation = True
        elif isinstance(anat, np.ndarray):
            anat_img = nib.Nifti1Image(anat, np.eye(4))
            have_orientation = False
        else:
            raise TypeError("anat type {} not understood".format(type(anat)))
        self.anat_img = nib.as_closest_canonical(anat_img)
        self.anat_data = self.anat_img.get_fdata()

        # -- Load and reorient the statistical image

        if isinstance(stat, string_types):
            stat_img = nib.load(stat)
        elif isinstance(stat, nib.spatialimages.SpatialImage):
            stat_img = stat
        elif isinstance(stat, np.ndarray):
            if stat.dtype is np.dtype("bool"):
                stat = stat.astype(np.int)
            stat_img = nib.Nifti1Image(stat, anat_img.affine, anat_img.header)
        elif stat is not None:
            raise TypeError("stat type {} not understood".format(type(stat)))
        else:
            stat_img = None

        if stat_img is not None:
            self.stat_img = nib.as_closest_canonical(stat_img)

        # -- Load and reorient the mask image

        if isinstance(mask, string_types):
            mask_img = nib.load(mask)
        elif isinstance(mask, nib.spatialimages.SpatialImage):
            mask_img = mask
        elif isinstance(mask, np.ndarray):
            if mask.dtype is np.dtype("bool"):
                mask = mask.astype(np.int)
            mask_img = nib.Nifti1Image(mask, anat_img.affine, anat_img.header)
        elif mask is not None:
            raise TypeError("mask type {} not understood".format(type(mask)))
        else:
            mask_img = None
            mask_data = None

        if mask is not None:
            self.mask_img = nib.as_closest_canonical(mask_img)
            mask_data = self.mask_img.get_fdata().astype(bool)

        if slice_dir[0] not in "sca":
            err = "Slice direction {} not understood".format(slice_dir)
            raise ValueError(err)

        # Find a field of view that tries to eliminate empty voxels
        anat_fov = self.anat_img.get_fdata() > 1e-5
        if tight:
            self.fov = anat_fov
            if mask is not None:
                self.fov &= mask_data
        else:
            self.fov = np.ones_like(anat_fov, np.bool)

        # Save the mosaic parameters
        self.n_col = n_col
        self.step = step
        self.slice_dir = slice_dir
        self.title = title

        # Define slice objects to crop to the volume
        slices, = ndimage.find_objects(self.fov.astype(np.int))
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
        width, height = self.n_col, (ny / nx) * n_row
        if self.title is None:
            top = 1
        else:
            pad = .25
            top = height / (height + pad)
            height += pad
        plot_kws = dict(nrows=int(n_row), ncols=int(self.n_col),
                        figsize=(width, height), facecolor="0",
                        subplot_kw=dict(xticks=[], yticks=[]))

        self.fig, self.axes = plt.subplots(**plot_kws)
        [ax.set_axis_off() for ax in self.axes.flat]
        self.fig.subplots_adjust(0, 0, 1, top, 0, 0)

        if self.title is not None:
            self.fig.text(.5, top + (1 - top) / 2, self.title,
                          ha="center", va="center", color="w", size=10)

    def _plot_anat(self, lims=None):
        """Plot the anatomy in grayscale."""
        anat_data = self.anat_img.get_fdata()
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
        mask_data = self.mask_img.get_fdata().astype(np.bool)
        anat_data = self.anat_img.get_fdata()
        mask_data = np.where(mask_data | (anat_data < 1e-5), np.nan, 1)
        mask_fov = mask_data[self.x_slice, self.y_slice, self.z_slice]
        self._map("imshow", mask_fov, cmap="bone", vmin=0, vmax=3,
                  interpolation="nearest", alpha=.5)

    def _map(self, func_name, data, **kwargs):
        """Apply a named function to a 3D volume of data on each axes."""
        transpose_orders = dict(s=(0, 1, 2), c=(1, 0, 2), a=(2, 0, 1))
        slice_key = self.slice_dir[0]
        slices = data.transpose(*transpose_orders[slice_key])
        for slice, ax in zip(slices, self.axes.flat):
            if np.isnan(slice).all():
                continue  # avoid bug in matplotlib 2.1.0
            func = getattr(ax, func_name)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                func(np.rot90(slice), **kwargs)

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
        stat_data = self.stat_img.get_fdata()[self.x_slice,
                                              self.y_slice,
                                              self.z_slice].copy()
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
        stat_data = self.stat_img.get_fdata()[self.x_slice,
                                              self.y_slice,
                                              self.z_slice].copy()
        if hasattr(self, "mask_img"):
            fov = self.mask_img.get_fdata()[self.x_slice,
                                            self.y_slice,
                                            self.z_slice].astype(bool)
        else:
            fov = np.ones_like(stat_data).astype(bool)

        if vmin is None:
            vmin = np.percentile(stat_data[fov], vmin_perc)
        if vmax is None:
            vmax = np.percentile(stat_data[fov], vmax_perc)

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
        mask_data = self.stat_img.get_fdata()[self.x_slice,
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

        slices = self.stat_img.get_fdata()[self.x_slice,
                                           self.y_slice,
                                           self.z_slice]

        self._map("contour", slices,
                  levels=[0, 1], cmap=cmap, vmin=0, vmax=1,
                  linewidths=linewidth)

    def _pad_for_cbar(self):
        """Add extra space to the bottom of the figure for the colorbars."""
        w, h = self.fig.get_size_inches()
        cbar_inches = .3
        self.fig.set_size_inches(w, h + cbar_inches)
        cbar_height = cbar_inches / (h + cbar_inches)
        self.fig.subplots_adjust(0, cbar_height, 1, 1)

        # Needed so things look nice in the notebook
        bg_ax = self.fig.add_axes([0, 0, 1, cbar_height])
        bg_ax.set_axis_off()
        bg_ax.pcolormesh(np.array([[1]]), cmap="Greys", vmin=0, vmax=1)

        return cbar_height

    def _add_single_colorbar(self, vmin, vmax, cmap, fmt):
        """Add colorbars for a single overlay."""
        cbar_height = self._pad_for_cbar()
        cbar_ax = self.fig.add_axes([.3, .01, .4, cbar_height - .01])
        cbar_ax.set_axis_off()

        bar_data = np.linspace(0, 1, 256).reshape(1, 256)
        cbar_ax.pcolormesh(bar_data, cmap=cmap)

        if fmt is not None:

            fmt = "{:" + fmt + "}"
            kws = dict(y=.005 + cbar_height * .5,
                       color="white", size=14, va="center")

            self.fig.text(.29, s=fmt.format(vmin), ha="right", **kws)
            self.fig.text(.71, s=fmt.format(vmax), ha="left", **kws)

    def _add_double_colorbar(self, vmin, vmax, pos_cmap, neg_cmap, fmt):
        """Add colorbars for a positive and a negative overlay."""
        cbar_height = self._pad_for_cbar()

        bar_data = np.linspace(0, 1, 256).reshape(1, 256)

        pos_ax = self.fig.add_axes([.55, .01, .3, cbar_height - .01])
        pos_ax.set_axis_off()
        pos_ax.pcolormesh(bar_data, cmap=pos_cmap)

        neg_ax = self.fig.add_axes([.15, .01, .3, cbar_height - .01])
        neg_ax.set_axis_off()
        neg_ax.pcolormesh(bar_data, cmap=neg_cmap)

        if fmt is not None:

            fmt = "{:" + fmt + "}"
            kws = dict(y=.005 + cbar_height * .5,
                       color="white", size=14, va="center")

            self.fig.text(.54, s=fmt.format(vmin), ha="right", **kws)
            self.fig.text(.86, s=fmt.format(vmax), ha="left", **kws)
            self.fig.text(.14, s=fmt.format(-vmax), ha="right", **kws)
            self.fig.text(.46, s=fmt.format(-vmin), ha="left", **kws)

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
                cube_rgb = mpl._cm.cubehelix(s=float(start), r=float(rot))
                cube_cmap = mpl.colors.LinearSegmentedColormap(cmap, cube_rgb)
                lut = cube_cmap(np.linspace(.95, 0, 256))
                if reverse:
                    lut = lut[::-1]
                cmap = mpl.colors.ListedColormap(lut)
        return cmap

    def savefig(self, fname, close=False, **kwargs):
        """Save the figure; optionally close it."""
        self.fig.savefig(fname, facecolor="0", edgecolor="0", **kwargs)
        if close:
            self.close()

    def close(self):
        """Close the figure."""
        plt.close(self.fig)


class CarpetPlot(object):

    components = [
        "cortex", "subgm", "brainstem", "cerebellum",
        "cortwm", "deepwm", "cerebwm", "csf"
    ]

    def __init__(self, data, seg, mc_params=None, smooth_fwhm=5,
                 percent_change=True, vlim=None, title=None):
        """Heatmap rendering of an fMRI timeseries for quality control.

        The Freesurfer segmentation is used to organize data by different
        components of the brain.

        Instantiating the class will load, preprocess, and plot the data.

        Parameters
        ----------
        data : filename or nibabel image
            4D time series data to plot.
        wmparc : filename or nibabel image
            Freesurfer wmparc image in functional space.
        mc_params : filename or DataFrame, optional
            Text file or array of realignment parameters. If present, the time
            series of framewise displacements will be shown at the top of the
            figure.
        smooth_fwhm : float or None, optional
            Size of the smoothing kernel, in mm, to apply. Smoothing is
            restricted within the mask for each component (cortex, cerebellum,
            etc.). Smoothing reduces white noise and makes global image
            artifacts much more apparent. Set to None to skip smoothing.
        percent_change : bool, optional
            If True, convert data to percent signal change over time.
        vlim : None or int, optional
            Colormap limits (will be symmetric around 0).
        title : string
            Title to show at the top of the plot.

        Attributes
        ----------
        fig : matplotlib Figure
        axes : dict of matplotlib Axes
        segdata : dict of arrays with data in the main plot
        fd : 1d array of framewise displacements

        """
        # TODO Accept mean_img and use that to convert to pct change if present
        # TODO accept a lut? (Also make the anat segmentation generate one)
        # Load the timeseries data
        if isinstance(data, str):
            img = nib.load(data)
        else:
            img = data
        data = img.get_fdata()

        # Load the Freesurfer parcellation
        if isinstance(seg, str):
            seg = nib.load(seg).get_fdata()
        else:
            seg = seg.get_fdata()
        masks, brain = self.define_masks(seg)

        # Use header geometry to convert smoothing sigma from mm to voxels
        sx, sy, sz, _ = img.header.get_zooms()
        voxel_sizes = sx, sy, sz
        if smooth_fwhm is not None and smooth_fwhm > 0:
            smooth_sigma = np.divide(smooth_fwhm / 2.355, voxel_sizes)
        else:
            smooth_sigma = None

        # Preprocess and segment the data
        if percent_change:
            data[brain] = self.percent_change(data[brain])
        data[brain] = signal.detrend(data[brain])
        data = self.smooth_data(data, masks, smooth_sigma)
        segdata = self.segment_data(data, masks)
        fd = self.framewise_displacement(mc_params)

        # Get a default limit for the colormap
        if vlim is None:
            sd = np.percentile(segdata["cortex"].std(axis=1), 95)
            vlim = int(np.round(sd))

        # Make the plot
        fig, axes = self.setup_figure()
        self.fig, self.axes = fig, axes
        self.plot_fd(axes["motion"], fd)
        self.plot_data(axes, segdata, vlim)
        if title is not None:
            fig.suptitle(title, size=10)

        # Store useful attributes
        self.segdata = segdata
        self.fd = fd

    def savefig(self, fname, close=True, **kwargs):

        self.fig.savefig(fname, **kwargs)
        if close:
            self.close()

    def close(self):

        plt.close(self.fig)

    def percent_change(self, data):
        """Convert to percent signal change over the mean for each voxel."""
        null = data.mean(axis=-1) == 0
        with np.errstate(all="ignore"):
            data /= data.mean(axis=-1, keepdims=True)
        data -= 1
        data *= 100
        data[null] = 0
        return data

    def define_masks(self, seg):
        """Create masks for anatomical components using Freesurfer labeling."""
        masks = {c: seg == i for i, c in enumerate(self.components, 1)}
        brain = seg > 0

        return masks, brain

    def smooth_data(self, data, masks, sigma):
        """Smooth the 4D image separately within each component."""
        if sigma is None:
            return data

        for comp, mask in masks.items():
            data[mask] = self._smooth_within_mask(data, mask, sigma)

        return data

    def _smooth_within_mask(self, data, mask, sigmas):
        """Smooth each with a Gaussian kernel, restricted to a mask."""
        # TODO move this to a central lyman function?
        comp_data = data * np.expand_dims(mask, -1)
        for f in range(comp_data.shape[-1]):
            comp_data[..., f] = ndimage.gaussian_filter(comp_data[..., f],
                                                        sigmas)

        smooth_mask = ndimage.gaussian_filter(mask.astype(float), sigmas)
        with np.errstate(all="ignore"):
            comp_data = comp_data / np.expand_dims(smooth_mask, -1)

        return comp_data[mask]

    def segment_data(self, data, masks):
        """Convert the 4D data image into a set of 2D matrices."""
        segdata = {comp: data[mask] for comp, mask in masks.items()}
        return segdata

    def framewise_displacement(self, realign_params):
        """Compute the time series of framewise displacements."""
        if isinstance(realign_params, str):
            rp = pd.read_csv(realign_params)
        elif isinstance(realign_params, pd.DataFrame):
            rp = realign_params
        else:
            return None

        r = rp.filter(regex="rot").values
        t = rp.filter(regex="trans").values
        s = r * 50
        ad = np.hstack([s, t])
        rd = np.abs(np.diff(ad, axis=0))
        fd = np.sum(rd, axis=1)
        return fd

    def setup_figure(self):
        """Initialize and organize the matplotlib objects."""
        width, height = 8, 10
        f = plt.figure(figsize=(width, height))

        gs = plt.GridSpec(nrows=2, ncols=2, figure=f,
                          left=.07, right=.98,
                          bottom=.05, top=.96,
                          wspace=0, hspace=.02,
                          height_ratios=[.1, .9],
                          width_ratios=[.01, .99])

        ax_i = f.add_subplot(gs[1, 1])
        ax_m = f.add_subplot(gs[0, 1], sharex=ax_i)
        ax_c = f.add_subplot(gs[1, 0], sharey=ax_i)
        ax_b = f.add_axes([.035, .35, .0125, .2])

        ax_i.set(xlabel="Volume", yticks=[])
        ax_m.set(ylabel="FD (mm)")
        ax_c.set(xticks=[])

        axes = dict(image=ax_i, motion=ax_m, comp=ax_c, cbar=ax_b)

        return f, axes

    def plot_fd(self, ax, fd):
        """Show a line plot of the framewise displacement data."""
        if fd is None:
            fd = []

        ax.plot(np.arange(1, len(fd) + 1), fd, lw=1.5, color=".15")
        _, ymax = ax.get_ylim()
        ax.set(ylabel="FD (mm)", ylim=(0, max(.5, ymax)))
        for label in ax.get_xticklabels():
            label.set_visible(False)

    def plot_data(self, axes, segdata, vlim):
        """Draw the elements corresponding to the image data."""

        # Concatenate and plot the time series data
        plot_data = np.vstack([segdata[comp] for comp in self.components])
        axes["image"].imshow(plot_data, cmap="gray", vmin=-vlim, vmax=vlim,
                             aspect="auto", rasterized=True)

        # Separate the anatomical components
        sizes = [len(segdata[comp]) for comp in self.components]
        for y in np.cumsum(sizes)[:-1]:
            axes["image"].axhline(y, c="w", lw=1)

        # Add colors to identify the anatomical components
        comp_ids = np.vstack([
            np.full((len(segdata[comp]), 1), i, dtype=np.int)
            for i, comp in enumerate(self.components, 1)
        ])
        comp_colors = ['#3b5f8a', '#5b81b1', '#7ea3d1', '#a8c5e9',
                       '#ce8186', '#b8676d', '#9b4e53', '#fbdd7a']
        comp_cmap = mpl.colors.ListedColormap(comp_colors)
        axes["comp"].imshow(comp_ids,
                            vmin=1, vmax=len(self.components),
                            aspect="auto", rasterized=True,
                            cmap=comp_cmap)

        # Add the colorbar
        xx = np.expand_dims(np.linspace(1, 0, 100), -1)
        ax = axes["cbar"]
        ax.imshow(xx, aspect="auto", cmap="gray")
        ax.set(xticks=[], yticks=[], ylabel="Percent signal change")
        ax.text(0, -2, "$+${}".format(vlim),
                ha="center", va="bottom", clip_on=False)
        ax.text(0, 103, "$-${}".format(vlim),
                ha="center", va="top", clip_on=False)


def plot_design_matrix(X, title=None):
    """Show the design matrix as a transposed heatmap.

    Parameters
    ----------
    X : dataframe
        The design matrix with regressors in the columns.

    Returns
    -------
    f : figure
        The matplotlib figure with the heatmap drawn on the only axes.
    """
    # TODO add option to filter out nuisance components?
    X = X - X.min()
    X = X / X.max()

    n_col = X.shape[1]
    figsize = 8, .5 * n_col
    f, ax = plt.subplots(figsize=figsize)

    ax.pcolormesh(X.T, cmap="gray")
    ax.invert_yaxis()

    for i in range(1, n_col):
        ax.axhline(i, color="w", lw=1)

    yticks = np.arange(n_col) + .5
    ax.set_yticks(yticks)
    ax.set_yticklabels(X.columns)
    ax.set_xticks([])

    if title is not None:
        ax.set_title(title)

    f.tight_layout()

    return f


def plot_nuisance_variables(X, title=None):
    """Show the timeseries of each nuisance variable, by source.

    Parameters
    ----------
    X : dataframe
        The design matrix (can include design components that will be ignored).
    title : str, optional
        If present, will be used to title the figure.

    Returns
    -------
    f : figure
        The matplotlib figure with multiple axes for each source.

    """
    possible_sources = ["wm", "csf", "edge", "noise"]
    sources = []
    for source in possible_sources:
        if X.columns.str.startswith(source).any():
            sources.append(source)

    if not sources:
        return None

    # Make the figure
    n_row = len(sources)
    f, axes = plt.subplots(n_row, figsize=(8, 2 * n_row),
                           sharex=True, sharey=True)

    # Drop the index so x units are TRs
    X = X.reset_index(drop=True)

    hues = dict(wm=.9, csf=1.25, edge=2.4, noise=2.9)

    # Plot each source on an axes
    for ax, source in zip(axes, sources):

        X_sub = X.loc[:, X.columns.str.startswith(source)]

        n = X_sub.shape[1]
        colors = cubehelix_palette(n, start=hues.get(source, 0),
                                   rot=0, low=.5, high=.2, hue=1)

        X_sub.plot(ax=ax, color=colors, legend=False, linewidth=1)
        ax.set(ylabel=source)
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

    axes[-1].set(xlabel="Time (TR)", xlim=(-1, len(X)))

    if title is not None:
        axes[0].set_title(title)

    f.tight_layout()

    return f


def cubehelix_palette(n_colors=6, start=0, rot=.4, gamma=1.0, hue=0.8,
                      low=.15, high=.85):
    """Generate colors from the cubehelix system.

    Lightly adapted from seaborn. See parameter documentation there.

    """
    def get_color_function(p0, p1):
        # Copied from matplotlib because it lives in private module
        def color(x):
            # Apply gamma factor to emphasise low or high intensity values
            xg = x ** gamma

            # Calculate amplitude and angle of deviation from the black
            # to white diagonal in the plane of constant
            # perceived intensity.
            a = hue * xg * (1 - xg) / 2

            phi = 2 * np.pi * (start / 3 + rot * x)

            return xg + a * (p0 * np.cos(phi) + p1 * np.sin(phi))
        return color

    cdict = {
            "red": get_color_function(-0.14861, 1.78277),
            "green": get_color_function(-0.29227, -0.90649),
            "blue": get_color_function(1.97294, 0.0),
    }

    cmap = mpl.colors.LinearSegmentedColormap("cubehelix", cdict)

    x = np.linspace(low, high, n_colors)
    pal = cmap(x)[:, :3].tolist()

    return pal
