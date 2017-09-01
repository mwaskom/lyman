from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


def crop(img):
    """Closely crop a brain screenshot.

    Assumes a white background and no colorbar.
    """
    x, y = np.argwhere((img != 255).any(axis=-1)).T
    return img[x.min():x.max(), y.min():y.max(), :]


def multi_panel_brain_figure(panels):
    """Make a matplotlib figure with the brain screenshots.

    Parameters
    ----------
    panels : list of arrays
        Assumes the list has screenshots from the left hemisphere and then
        screenshots of the same views from the right hemisphere. The
        screenshots should be "cropped" for best results.

    Returns
    -------
    f: matplotlib figure
        Figure with the brains plotted onto it.

    """
    # Reorient the brains to be "wide"
    plot_panels = []
    for img in panels:
        if (img.shape[1] < img.shape[0]):
            img = np.rot90(img)
        plot_panels.append(img)

    # Infer the size of the figure and the axes
    shots_per_hemi = int(len(panels) / 2)
    sizes = np.array([p.shape for p in plot_panels[:shots_per_hemi]])
    full_size = sizes.sum(axis=0)
    height_ratios = sizes[:, 0] / full_size[0]
    ratio = full_size[0] / (sizes.max(axis=0)[1] * 2)
    figsize = (9, 9 * ratio)

    # Plot the brains onto the figure
    f, axes = plt.subplots(shots_per_hemi, 2, figsize=figsize,
                           gridspec_kw={"height_ratios": height_ratios})
    for ax, img in zip(axes.T.flat, plot_panels):
        ax.imshow(img)
        ax.set_axis_off()
    f.subplots_adjust(0.02, 0.02, .98, .98, .05, .05)

    return f


def _add_cbar_to_ax(ax, min, max, cmap):
    """Make a colorbar and draw it to fill an Axes."""
    # Create dummy data and plot a heatmap
    x = np.c_[np.linspace(0, 1, 256)].T
    ax.pcolormesh(x, cmap=cmap)
    ax.set(xlim=(0, 256))
    ax.set(xticks=[], yticks=[])

    # Add labels to show the min and max points of the colorbar
    ax.annotate("{:.2g}".format(min),
                ha="right", va="center", size=14, family="Arial",
                xy=(-.05, .5), xycoords="axes fraction")
    ax.annotate("{:.2g}".format(max),
                ha="left", va="center", size=14, family="Arial",
                xy=(1.05, .5), xycoords="axes fraction")


def add_colorbars(f, min, max, pos_cmap="YlOrRd_r", neg_cmap=None):
    """Add colorbars to the bottom of a brain image.

    Parameters
    ----------
    f : matplotlib figure
        Figure with brains that need colorbars.
    min, max : floats
        Min and max values for the colorbars. If positive and negative
        colorbars are to be shown, they should have the same limits.
    {pos, neg}_cmap : colormap names
        Names for the colormaps to use for the positive and negative bars.

    Returns
    -------
    f : matplotlib figure
        Returns the figure with the colorbars added.
    """
    # Add space at the bottom of the figure for the colorbars
    height = f.get_figheight()
    f.set_figheight(height + .5)
    edge_height = .5 / height
    f.subplots_adjust(bottom=edge_height)

    # Determine the size parameters of the bars
    bottom, height = edge_height * .2, edge_height * .4
    width = .35

    # Plot the positive and negative colorbars
    if pos_cmap is not None:
        pos_ax = f.add_axes([.55, bottom, width, height])
        _add_cbar_to_ax(pos_ax, min, max, pos_cmap)
    if neg_cmap is not None:
        pos_ax = f.add_axes([.10, bottom, width, height])
        _add_cbar_to_ax(pos_ax, -max, -min, neg_cmap)

    return f
