#! /usr/bin/env python
"""Generate static images summarizing the Freesurfer reconstruction.

This script is part of the lyman package. LYMAN_DIR must be defined.

The subject arg can be one or more subject IDs, name of subject file, or
path to subject file. Running with no arguments will use the default
subjects.txt file in the lyman directory.

Dependencies:
- Nibabel
- PySurfer

The resulting files can be most easily viewed using the Ziegler app.

There is a bug in Mayavi on some 64 bit versions of Python that causes
a segfault when repeatedly closing figures. If you encounter this bug,
you can try the `-noclose` switch which will keep all of the PySurfer
figures open until the script exits.

"""
import os
import os.path as op
import sys
import subprocess as sub
import argparse

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib
from mayavi import mlab
from surfer import Brain

import lyman


def main(arglist):

    # Find the subjects for this execution
    args = parse_args(arglist)
    subjects = lyman.determine_subjects(args.subjects)

    # Find the project details
    proj = lyman.gather_project_info()
    data_dir = proj["data_dir"]

    # Make images for each subject
    for subj in subjects:

        # Make sure the output directory exists
        out_dir = op.join(data_dir, subj, "snapshots")
        if not op.exists(out_dir):
            os.mkdir(out_dir)

        # Do each chunk of reporting
        inflated_surfaces(out_dir, subj, args.close)
        curvature_normalization(data_dir, subj, args.close)
        volume_images(data_dir, subj)


def inflated_surfaces(out_dir, subj, close=True):
    """Native inflated surfaces with cortical label."""
    for hemi in ["lh", "rh"]:
        b = Brain(subj, hemi, "inflated", curv=False,
                  config_opts=dict(background="white",
                                   width=800, height=500))
        b.add_label("cortex", color="#6B6B6B")

        for view in ["lat", "med"]:
            b.show_view(view)
            mlab.view(distance="auto")
            png = op.join(out_dir, "%s.surface_%s.png" % (hemi, view))
            b.save_image(png)
        if close:
            b.close()


def curvature_normalization(data_dir, subj, close=True):
    """Normalize the curvature map and plot contour over fsaverage."""
    surf_dir = op.join(data_dir, subj, "surf")
    snap_dir = op.join(data_dir, subj, "snapshots")
    for hemi in ["lh", "rh"]:

        cmd = ["mri_surf2surf",
               "--srcsubject", subj,
               "--trgsubject", "fsaverage",
               "--hemi", hemi,
               "--sval", op.join(surf_dir, "%s.curv" % hemi),
               "--tval", op.join(surf_dir, "%s.curv.fsaverage.mgz" % hemi)]

        sub.check_output(cmd)

        b = Brain("fsaverage", hemi, "inflated",
                  config_opts=dict(background="white",
                                   width=800, height=500))
        curv = nib.load(op.join(surf_dir, "%s.curv.fsaverage.mgz" % hemi))
        curv = (curv.get_data() > 0).squeeze()
        b.add_contour_overlay(curv, 0, 1.15, 2, 3)
        b.contour["colorbar"].visible = False
        for view in ["lat", "med"]:
            b.show_view(view)
            mlab.view(distance="auto")
            png = op.join(snap_dir, "%s.surf_warp_%s.png" % (hemi, view))
            b.save_image(png)

        if close:
            b.close()


def volume_images(data_dir, subj):
    """Plot a mosiac of the brainmask and aseg volumes."""
    # Plot the volume slices
    brain_file = op.join(data_dir, subj, "mri/brainmask.mgz")
    brain_data = nib.load(brain_file).get_data()

    ribbon_file = op.join(data_dir, subj, "mri/ribbon.mgz")
    ribbon_data = nib.load(ribbon_file).get_data().astype(float)
    ribbon_data[ribbon_data == 3] = 1
    ribbon_data[ribbon_data == 42] = 1
    ribbon_data[ribbon_data != 1] = np.nan

    aseg_file = op.join(data_dir, subj, "mri/aseg.mgz")
    aseg_data = nib.load(aseg_file).get_data()
    aseg_data[aseg_data == 41] = 2
    aseg_lut = np.genfromtxt(op.join(os.environ["FREESURFER_HOME"],
                                     "FreeSurferColorLUT.txt"))

    # Find the limits of the data
    # note that FS conformed space is not (x, y, z)
    xdata = np.flatnonzero(brain_data.any(axis=1).any(axis=1))
    xmin, xmax = xdata.min(), xdata.max()
    ydata = np.flatnonzero(brain_data.any(axis=0).any(axis=0))
    ymin, ymax = ydata.min(), ydata.max()
    zdata = np.flatnonzero(brain_data.any(axis=0).any(axis=1))
    zmin, zmax = zdata.min() + 5, zdata.max() - 15

    # Figure out the plot parameters
    n_slices = (zmax - zmin) // 3
    n_row, n_col = n_slices // 8, 8
    start = n_slices % n_col // 2 + zmin
    figsize = (10, 1.375 * n_row)
    slices = (start + np.arange(zmax - zmin))[::3][:n_slices]

    # Draw the brainmask and cortical ribbon
    vmin, vmax = 0, 100
    f, axes = plt.subplots(n_row, n_col, figsize=figsize, facecolor="k")
    cmap = mpl.colors.ListedColormap(["#C41E3A"])
    for i, ax in enumerate(reversed(axes.ravel())):
        i = slices[i]
        ax.imshow(np.flipud(brain_data[xmin:xmax, i, ymin:ymax].T),
                  cmap="gray", vmin=vmin, vmax=vmax)
        ax.imshow(np.flipud(ribbon_data[xmin:xmax, i, ymin:ymax].T),
                  cmap=cmap, vmin=.05, vmax=1.5, alpha=.8)
        ax.set_xticks([])
        ax.set_yticks([])

    out_file = op.join(data_dir, subj, "snapshots/volume.png")
    plt.savefig(out_file, dpi=100, bbox_inches="tight",
                facecolor="k", edgecolor="k")
    plt.close(f)

    # Draw the brainmask and cortical ribbon
    f, axes = plt.subplots(n_row, n_col, figsize=figsize, facecolor="k")
    aseg_cmap = mpl.colors.ListedColormap(aseg_lut[:64, 2:5] / 255)
    for i, ax in enumerate(reversed(axes.ravel())):
        i = slices[i]
        ax.imshow(np.flipud(brain_data[xmin:xmax, i, ymin:ymax].T),
                  cmap="gray", vmin=vmin, vmax=vmax)
        ax.imshow(np.flipud(aseg_data[xmin:xmax, i, ymin:ymax].T),
                  cmap=aseg_cmap, vmin=0, vmax=63, alpha=.75)
        ax.set_xticks([])
        ax.set_yticks([])

    out_file = op.join(data_dir, subj, "snapshots/aseg.png")
    plt.savefig(out_file, dpi=100, bbox_inches="tight",
                facecolor="k", edgecolor="k")
    plt.close(f)


def parse_args(arglist):

    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-subjects", nargs="*", help="lyman subjects argument")
    parser.add_argument("-noclose", dest="close", action="store_false",
                        help="don't close mayavi figures during runtime")
    args = parser.parse_args(arglist)

    return args


if __name__ == "__main__":

    main(sys.argv[1:])
