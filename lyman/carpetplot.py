from __future__ import division
import numpy as np
import pandas as pd
from scipy.signal import detrend
from scipy.ndimage import gaussian_filter
import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib


class CarpetPlot(object):

    components = [
        "cortex", "subgm", "brainstem", "cerebellum",
        "cortwm", "deepwm", "cerebwm", "csf"
    ]

    def __init__(self, data, seg, mc_params=None, smooth_fwhm=5,
                 vlim=None, title=None):
        """Heatmap rendering of an fMRI timeseries for quality control.

        The Freesurfer segmentation is used to organize data by different
        components of the brain. The components are organized from top to
        bottom and color-coded as follows:

        Instantiating the class will load, preprocess, and plot the data.

        Parameters
        ----------
        data : filename or nibabel image
            4D time series data to plot.
        wmparc : filename or nibabel image
            Freesurfer wmparc image in functional space.
        realign_params : filename or DataFrame, optional
            Text file or array of realignment parameters. If present, the time
            series of framewise displacements will be shown at the top of the
            figure.
        smooth_fwhm : float or None, optional
            Size of the smoothing kernel, in mm, to apply. Smoothing is
            restricted within the mask for each component (cortex, cerebellum,
            etc.). Smoothing reduces white noise and makes global image
            artifacts much more apparent. Set to None to skip smoothing.
        vlim : None or int, optional
            Colormap limits (will be symmetric) in percent signal change units.
        title : string
            Title to show at the top of the plot.

        Attributes
        ----------
        fig : matplotlib Figure
        axes : dict of matplotlib Axes
        segdata : dict of arrays with data in the main plot
        fd : 1d array of framewise displacements

        """
        # Load the timeseries data
        if isinstance(data, str):
            img = nib.load(data)
        else:
            img = data
        data = img.get_data().astype(np.float)

        # Load the Freesurfer parcellation
        if isinstance(seg, str):
            seg = nib.load(seg).get_data()
        else:
            seg = seg.get_data()

        # Use header geometry to convert smoothing sigma from mm to voxels
        sx, sy, sz, _ = img.header.get_zooms()
        voxel_sizes = sx, sy, sz
        if smooth_fwhm is not None:
            if smooth_fwhm > 0:
                smooth_sigma = np.divide(smooth_fwhm / 2.355, voxel_sizes)
            else:
                smooth_sigma = None

        # Preprocess and segment the data
        masks, brain = self.define_masks(seg)
        data[brain] = self.percent_change(data[brain])
        data[brain] = detrend(data[brain])
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
            fig.suptitle(title)

        # Store useful attributes
        self.segdata = segdata
        self.fd = fd

    def savefig(self, fname, **kwargs):

        self.fig.savefig(fname, **kwargs)

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
        comp_data = data * np.expand_dims(mask, -1)
        for f in range(comp_data.shape[-1]):
            comp_data[..., f] = gaussian_filter(comp_data[..., f], sigmas)

        smooth_mask = gaussian_filter(mask.astype(float), sigmas)
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
        elif isinstance(realign_params, np.ndarray):
            rp = realign_params
        else:
            return None

        r = rp.filter(regexp="rot").values
        t = rp.filter(regexp="trans").values
        s = r * 50
        ad = np.hstack([s, t])
        rd = np.abs(np.diff(ad, axis=0))
        fd = np.sum(rd, axis=1)
        return fd

    def setup_figure(self):
        """Initialize and organize the matplotlib objects."""
        width, height = 8, 10
        f = plt.figure(figsize=(width, height))

        gs = plt.GridSpec(nrows=2, ncols=2,
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

        ax.set(ylim=(0, .5))
        ax.plot(np.arange(1, len(fd) + 1), fd, lw=1.5, color=".15")
        ax.set(ylabel="FD (mm)", ylim=(0, None))
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
