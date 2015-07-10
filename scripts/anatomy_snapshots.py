#! /usr/bin/env python
"""Generate static images summarizing the Freesurfer reconstruction.

This script is part of the lyman package. LYMAN_DIR must be defined.

The subject arg can be one or more subject IDs, name of subject file, or
path to subject file. Running with no arguments will use the default
subjects.txt file in the lyman directory.

Dependencies:
- Nibabel
- PySurfer

The resulting files can be most easily viewed using the ziegler app.

"""
from __future__ import division
import os
import os.path as op
import sys
import argparse

import numpy as np
import nibabel as nib
from scipy.interpolate import NearestNDInterpolator

import matplotlib as mpl
import matplotlib.pyplot as plt
from surfer import Brain

from moss.mosaic import Mosaic
import lyman
from lyman.tools.plotting import crop, multi_panel_brain_figure


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
        surface_images(out_dir, subj)
        curvature_normalization(data_dir, subj)
        volume_images(data_dir, subj)


def surface_images(out_dir, subj):
    """Plot the white, pial, and inflated surfaces to look for defects."""
    for surf in ["white", "pial", "inflated"]:
        panels = []
        for hemi in ["lh", "rh"]:

            try:
                b = Brain(subj, hemi, surf, curv=False, background="white")
            except TypeError:
                # PySurfer <= 0.5
                b = Brain(subj, hemi, surf, curv=False,
                          config_opts=dict(background="white"))

            for view in ["lat", "med", "ven"]:
                b.show_view(view, distance="auto")
                panels.append(crop(b.screenshot()))
            b.close()

        # Make and save a figure
        f = multi_panel_brain_figure(panels)
        fname = op.join(out_dir, "{}_surface.png".format(surf))
        f.savefig(fname, bbox_inches="tight")
        plt.close(f)


def apply_surface_warp(data_dir, subj, hemi, vals):
    """Apply the spherical normalization from subject to fsaverage space."""
    # Load the files containing registration information
    sphere_reg_fname = op.join(data_dir, subj, "surf",
                               "{}.sphere.reg".format(hemi))
    avg_sphere_fname = op.join(data_dir, "fsaverage/surf",
                               "{}.sphere".format(hemi))
    sphere_reg, _ = nib.freesurfer.read_geometry(sphere_reg_fname)
    avg_sphere, _ = nib.freesurfer.read_geometry(avg_sphere_fname)

    # Apply the registration
    interpolator = NearestNDInterpolator(sphere_reg, vals, 0)
    normalized_vals = interpolator(avg_sphere)
    return normalized_vals


def curvature_normalization(data_dir, subj):
    """Normalize the curvature map and plot contour over fsaverage."""
    surf_dir = op.join(data_dir, subj, "surf")
    snap_dir = op.join(data_dir, subj, "snapshots")
    panels = []
    for hemi in ["lh", "rh"]:

        # Load the curv values and apply registration to fsaverage
        curv_fname = op.join(surf_dir, "{}.curv".format(hemi))
        curv_vals = nib.freesurfer.read_morph_data(curv_fname)
        subj_curv_vals = apply_surface_warp(data_dir, subj,
                                            hemi, curv_vals)
        subj_curv_binary = (subj_curv_vals > 0)

        # Load the template curvature
        norm_fname = op.join(data_dir, "fsaverage", "surf",
                             "{}.curv".format(hemi))
        norm_curv_vals = nib.freesurfer.read_morph_data(norm_fname)
        norm_curv_binary = (norm_curv_vals > 0)

        # Compute the curvature overlap image
        curv_overlap = np.zeros_like(norm_curv_binary, np.int)
        curv_overlap[norm_curv_binary & subj_curv_binary] = 1
        curv_overlap[norm_curv_binary ^ subj_curv_binary] = 2

        # Mask out the medial wall
        cortex_fname = op.join(data_dir, "fsaverage", "label",
                               "{}.cortex.label".format(hemi))
        cortex = nib.freesurfer.read_label(cortex_fname)
        medial_wall = ~np.in1d(np.arange(curv_overlap.size), cortex)
        curv_overlap[medial_wall] = 1

        # Plot the curvature overlap image
        try:
            b = Brain("fsaverage", hemi, "inflated", background="white")
        except TypeError:
            # PySurfer <= 0.5
            b = Brain("fsaverage", hemi, "inflated",
                      config_opts=dict(background="white"))

        b.add_data(curv_overlap, min=0, max=2,
                   colormap=[".9", ".45", "indianred"], colorbar=False)

        for view in ["lat", "med", "ven"]:
            b.show_view(view, distance="auto")
            panels.append(crop(b.screenshot()))
        b.close()

    # Make and save a figure
    f = multi_panel_brain_figure(panels)
    fname = op.join(snap_dir, "surface_registration.png")
    f.savefig(fname, bbox_inches="tight")
    plt.close(f)


def volume_images(data_dir, subj):
    """Plot a mosiac of the brainmask and aseg volumes."""
    # Plot the volume slices
    brain_file = op.join(data_dir, subj, "mri/brainmask.mgz")

    # Load the cortical ribbon file and use same index for both hemis
    ribbon_file = op.join(data_dir, subj, "mri/ribbon.mgz")
    ribbon_data = nib.load(ribbon_file).get_data().astype(float)
    ribbon_data[ribbon_data == 3] = 1
    ribbon_data[ribbon_data == 42] = 1
    ribbon_data[ribbon_data != 1] = np.nan

    # Load the aseg file and use it to derive a FOV mask
    aseg_file = op.join(data_dir, subj, "mri/aseg.mgz")
    aseg_data = nib.load(aseg_file).get_data()
    aseg_data[aseg_data == 41] = 2
    mask_data = (aseg_data > 0).astype(int)

    # Load the lookup table for the aseg volume
    aseg_lut = np.genfromtxt(op.join(os.environ["FREESURFER_HOME"],
                                     "FreeSurferColorLUT.txt"))

    # Draw the brainmask and cortical ribbon
    m = Mosaic(brain_file, ribbon_data, mask_data, step=3)
    m.plot_mask("#C41E3A")
    m.savefig(op.join(data_dir, subj, "snapshots/volume.png"))
    m.close()

    # Draw the brainmask and cortical ribbon
    aseg_cmap = mpl.colors.ListedColormap(aseg_lut[:64, 2:5] / 255)
    m = Mosaic(brain_file, aseg_data, mask_data, step=3)
    m.plot_overlay(aseg_cmap, vmin=0, vmax=63, alpha=.75, colorbar=False)
    m.savefig(op.join(data_dir, subj, "snapshots/aseg.png"))
    m.close()


def parse_args(arglist):

    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-subjects", nargs="*", help="lyman subjects argument")
    args = parser.parse_args(arglist)

    return args


if __name__ == "__main__":

    main(sys.argv[1:])
