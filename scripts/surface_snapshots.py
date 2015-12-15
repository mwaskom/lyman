#! /usr/bin/env python
"""Make static images of lyman results using PySurfer."""
import os.path as op
import sys
import argparse
from textwrap import dedent
from time import sleep

import numpy as np
from scipy import stats
import nibabel as nib
import matplotlib.pyplot as plt
from surfer import Brain

import lyman
from lyman.tools.plotting import multi_panel_brain_figure, crop, add_colorbars


def main(arglist):

    # Parse the command line
    args = parse_args(arglist)

    # Load the lyman data
    subjects = lyman.determine_subjects(args.subjects)
    project = lyman.gather_project_info()
    exp = lyman.gather_experiment_info(args.experiment, args.altmodel)
    contrasts = exp["contrast_names"]
    z_thresh = exp["cluster_zthresh"]

    # Get the full correct name for the experiment
    if args.experiment is None:
        exp_name = project["default_exp"]
    else:
        exp_name = args.experiment
    exp_base = exp_name
    if args.altmodel is not None:
        exp_name = "-".join([exp_base, args.altmodel])

    # Group-level
    # ===========

    if args.level == "group":
        temp_base = op.join(project["analysis_dir"], exp_name, args.output,
                            args.regspace, "{contrast}")
        if args.regspace == "fsaverage":
            sig_thresh = -np.log10(stats.norm.sf(z_thresh))
            sig_thresh = np.round(sig_thresh) * 10
            corr_sign = exp["surf_corr_sign"]
            sig_name = "cache.th%d.%s.sig.masked.mgh" % (sig_thresh, corr_sign)
            stat_temp = op.join(temp_base, "{hemi}/osgm", sig_name)
            mask_temp = op.join(temp_base, "{hemi}/mask.mgh")
            png_temp = op.join(temp_base, "{hemi}/osgm/zstat_threshold.png")
        else:
            stat_temp = op.join(temp_base, "{hemi}.zstat1_threshold.mgz")
            mask_temp = op.join(temp_base, "{hemi}.group_mask.mgz")
            png_temp = op.join(temp_base, "zstat1_threshold_surf.png")
            corr_sign = "pos"

        contrast_loop("fsaverage", contrasts, stat_temp, mask_temp, png_temp,
                      args, z_thresh, corr_sign)

    # Subject-level
    # =============

    elif args.level == "subject":
        temp_base = op.join(project["analysis_dir"], exp_name, "{subj}",
                            "ffx", args.regspace, "smoothed/{contrast}")
        mask_temp = op.join(temp_base, "{hemi}.mask.mgz")
        stat_temp = op.join(temp_base, "{hemi}.zstat1.mgz")
        png_temp = op.join(temp_base, "zstat1_surf.png")

        for subj in subjects:
            contrast_loop(subj, contrasts, stat_temp, mask_temp, png_temp,
                          args, 1.96, "abs")


def contrast_loop(subj, contrasts, stat_temp, mask_temp, png_temp,
                  args, z_thresh, sign):
    """Iterate over contrasts and make surface images."""
    for contrast in contrasts:

        # Calculate where the overlay should saturate
        z_max = calculate_sat_point(stat_temp, contrast, sign, subj)
        panels = []
        for hemi in ["lh", "rh"]:

            # Initialize the brain object
            b_subj = subj if args.regspace == "epi" else "fsaverage"

            try:
                b = Brain(b_subj, hemi, args.geometry, background="white")
            except TypeError:
                # PySurfer <= v0.5
                b = Brain(b_subj, hemi, args.geometry,
                          config_opts={"background": "white"})

            # Plot the mask
            mask_file = mask_temp.format(contrast=contrast,
                                         hemi=hemi, subj=subj)
            add_mask_overlay(b, mask_file)

            # Plot the overlay
            stat_file = stat_temp.format(contrast=contrast,
                                         hemi=hemi, subj=subj)
            add_stat_overlay(b, stat_file, z_thresh, z_max, sign,
                             sig_to_z=args.regspace == "fsaverage")

            # Take screenshots
            for view in ["lat", "med", "ven"]:
                b.show_view(view, distance="auto")
                sleep(.1)
                panels.append(crop(b.screenshot()))
            b.close()

        # Make a single figure with all the panels
        f = multi_panel_brain_figure(panels)
        kwargs = {}
        if sign in ["pos", "abs"]:
            kwargs["pos_cmap"] = "Reds_r"
        if sign in ["neg", "abs"]:
            kwargs["neg_cmap"] = "Blues"
        add_colorbars(f, z_thresh, z_max, **kwargs)

        # Save the figure in both hemisphere outputs
        for hemi in ["lh", "rh"]:
            png_file = png_temp.format(hemi=hemi, contrast=contrast, subj=subj)
            f.savefig(png_file, bbox_inches="tight")
        plt.close(f)


def calculate_sat_point(template, contrast, sign, subj=None):
    """Calculate the point at which the colormap should saturate."""
    data = []
    for hemi in ["lh", "rh"]:
        hemi_file = template.format(contrast=contrast, subj=subj, hemi=hemi)
        hemi_data = nib.load(hemi_file).get_data()
        data.append(hemi_data)
    data = np.concatenate(data)
    if sign == "pos":
        z_max = max(3.71, np.percentile(data, 98))
    elif sign == "neg":
        z_max = max(3.71, np.percentile(-data, 98))
    elif sign == "abs":
        z_max = max(3.71, np.percentile(np.abs(data), 98))
    return z_max


def add_mask_overlay(b, mask_file):
    """Gray-out vertices outside of the common-space mask."""
    mask_data = nib.load(mask_file).get_data()

    # Plot the mask
    mask_data = np.logical_not(mask_data.astype(bool)).squeeze()
    if mask_data.any():
        b.add_data(mask_data, min=0, max=10, thresh=.5,
                   colormap="bone", alpha=.6, colorbar=False)


def add_stat_overlay(b, stat_file, thresh, max, sign, sig_to_z=False):
    """Plot a surface-encoded statistical overlay."""
    stat_data = nib.load(stat_file).get_data()

    # Possibly convert -log10(p) images to z stats
    if sig_to_z:
        stat_sign = np.sign(stat_data)
        p_data = 10 ** -np.abs(stat_data)
        z_data = stats.norm.ppf(p_data)
        z_data[np.sign(z_data) != stat_sign] *= -1
        stat_data = z_data

    # Plot the statistical data
    stat_data = stat_data.squeeze()
    if sign in ["pos", "abs"] and (stat_data > thresh).any():
        b.add_data(stat_data, thresh, max, thresh,
                   colormap="Reds_r", colorbar=False)

    if sign in ["neg", "abs"] and (stat_data < -thresh).any():
        b.add_data(-stat_data, thresh, max, thresh,
                   colormap="Blues_r", colorbar=False)


def parse_args(arglist):

    help = dedent("""
    Plot the outputs of lyman analyses on a 3D surface mesh.

    This script uses PySurfer to generate surface images, which can provide
    considerably more information about the distribution of activation than
    volume-based images. Because the 3D rendering can be difficult to work
    with, the script is outside of the Nipype workflows that actually generate
    the results. Unfortunately, that means the script cannot be parallelized
    and does not cache its intermediate results.

    Images can be generated either at the group level or at the subject level,
    in which case the fixed-effects outputs are plotted. Currently, the
    statistics are plotted as Z statistics (even for Freesurfer results, which
    are stored as -log10[p]), and regions that were not included in the
    analysis mask are grayed out to represent their non-inclusion. For the
    group-level plots, some aspects of how the results are rendered onto the
    cortex can be controlled through parameters in the experiment file. Other
    parameters are available as command-line options.

    It is important to emphasize that because this script must be executed
    separately from the processing workflows, it is possible for the static
    images to get out of sync with the actual results. It is up to the user
    to ensure that this does not transpire by always updating the snapshots
    when rerunning the workflows.

    Examples
    --------

    Note that the parameter switches match any unique short version
    of the full parameter name.

    surface_snapshots.py

        With no arguments, this will make snapshots for the default experiment
        at the group level in MNI space.

    surface_snapshots.py -r fsaverage -o pilot

        Make snapshots from the outputs of the surface workflow that are stored
        in <analysis_dir>/<experiment>/pilot/fsaverage. The -log10(p) maps that
        are written to Freesurfer will be converted to Z stats before plotting.

    surface_snapshots.py -l subject -e nback -a parametric -r epi

        Make snapshots of the fixed-effects model outputs on the native surface
        for an alternate model of the `nback` experiment for all subjects
        defined in the $LYMAN_DIR/subjects.txt file.

    surface_snapshots.py -s subj1 subj2 -r mni -l subject -g smoothwm

        Plot the default experiment fixed effects model outputs for subjects
        `subj1` and `subj2` in MNI space on the `smoothwm` surface of the
        fsaverage brain.

    Usage Details
    -------------

    """)

    parser = argparse.ArgumentParser(description=help)
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.add_argument("-subjects", nargs="*",
                        help=("list of subject ids, name of file in lyman  "
                              "directory, or full path to text file with  "
                              "subject ids"))
    parser.add_argument("-experiment", help="experimental paradigm")
    parser.add_argument("-altmodel", help="alternate model to fit")
    parser.add_argument("-level", choices=["subject", "group"],
                        default="group",
                        help="analysis level to make images from")
    parser.add_argument("-regspace", default="mni",
                        choices=["mni", "fsaverage", "epi"],
                        help="common space where data are registered")
    parser.add_argument("-output", default="group",
                        help="group analysis output name")
    parser.add_argument("-geometry", default="inflated",
                        help="surface geometry for the rendering.")
    return parser.parse_args(arglist)


if __name__ == "__main__":
    main(sys.argv[1:])
