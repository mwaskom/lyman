#! /usr/bin/env python
"""Make static images of lyman results using PySurfer."""
import os
import os.path as op
import sys
import tempfile
import argparse
from textwrap import dedent

import numpy as np
from scipy import stats
import nibabel as nib
import matplotlib.image as mplimg
from mayavi import mlab
from surfer import Brain, io

import moss
import lyman


def main(arglist):

    # Parse the command line
    args = parse_args(arglist)

    # Configure mayavi
    mlab.options.offscreen = args.nowindow

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
    config_opts = dict(background="white", width=600, height=370)

    for contrast in contrasts:

        # Calculate where the overlay should saturate
        z_max = calculate_sat_point(stat_temp, contrast, subj)
        png_panes = []
        for hemi in ["lh", "rh"]:

            # Initialize the brain object
            b_subj = subj if args.regspace == "epi" else "fsaverage"
            b = Brain(b_subj, hemi, args.geometry,
                      config_opts=config_opts)

            # Plot the mask
            mask_file = mask_temp.format(contrast=contrast,
                                         hemi=hemi, subj=subj)
            add_mask_overlay(b, mask_file)

            # Plot the overlay
            stat_file = stat_temp.format(contrast=contrast,
                                         hemi=hemi, subj=subj)
            add_stat_overlay(b, stat_file, z_thresh, z_max, sign,
                             sig_to_z=args.regspace == "fsaverage")
            png_panes.append(save_view_panes(b, sign, hemi))

            # Maybe close the current figure
            if args.close:
                b.close()

        # Stitch the hemisphere pngs together and save
        full_png = np.concatenate(png_panes, axis=1)
        png_file = png_temp.format(contrast=contrast, hemi=hemi, subj=subj)
        mplimg.imsave(png_file, full_png)


def calculate_sat_point(template, contrast, subj=None):
    """Calculate the point at which the colormap should saturate."""
    data = []
    for hemi in ["lh", "rh"]:
        hemi_file = template.format(contrast=contrast, subj=subj, hemi=hemi)
        hemi_data = nib.load(hemi_file).get_data()
        data.append(hemi_data)
    data = np.concatenate(data)
    return max(3.71, moss.percentiles(data, 98))


def add_mask_overlay(b, mask_file):
    """Gray-out vertices outside of the common-space mask."""
    mask_data = nib.load(mask_file).get_data()

    # Plot the mask
    mask_data = np.logical_not(mask_data.astype(bool)).squeeze()
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
    plot_stat = ((sign == "pos" and (stat_data > thresh).any())
                 or (sign == "neg" and (stat_data < -thresh).any())
                 or (sign == "abs" and (np.abs(stat_data) > thresh).any()))
    if plot_stat:
        stat_data = stat_data.squeeze()
        b.add_overlay(stat_data, min=thresh, max=max, sign=sign, name="stat")


def remove_stat_overlay(b):
    """Remove any overlays from the brain."""
    for name, overlay in b.overlays.items():
        overlay.remove()


def save_view_panes(b, sign, hemi):
    """Save lat, med and ven views, return a stacked image array."""
    views = ["lat", "med", "ven"]
    image_panes = []
    for view in views:

        # Handle the colorbar
        if "stat" in b.overlays:
            show_pos = hemi == "rh" and view == "ven"
            show_neg = hemi == "lh" and view == "ven"
            if sign in ("pos", "abs"):
                b.overlays["stat"].pos_bar.visible = show_pos
            if sign in ("neg", "abs"):
                b.overlays["stat"].neg_bar.visible = show_neg

        # Set the view and screenshot
        b.show_view(view, distance=330)
        image_panes.append(b.screenshot())

    # Stitch the images together and return the array
    full_image = np.concatenate(image_panes, axis=0)

    return full_image


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

    In an effort to allow flexibility for some particular issues that PySurfer
    can have, a few command-line options control what happens with the PySurfer
    window during the execution of the script. By default, a window is actually
    opened as the plot is drawn, but this can happen offscreen on systems for
    which this is possible. Additionally, on some platforms closing multiple
    PySurfer figures will cause a segfault, so the plots can be drawn into a
    single window for the whole script (unfortunately this may cause issues
    with the orientation of the scene lighting).

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

    surface_snapshots.py -k

        Same as the first example, but don't close the PySurfer window until the
        very end of the script. This can avoid a nasty segmentation fault issue
        on some systems.

    surface_snapshots.py -n

        Same as the first example, but don't show a PySurfer window while
        drawing. This is not guaranteed to work on all systems.

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
    parser.add_argument("-nowindow", action="store_true",
                        help="plot offscreen (does not work on all platforms")
    parser.add_argument("-keepopen", dest="close", action="store_false",
                        help="do not close each figure after plotting")
    return parser.parse_args(arglist)


if __name__ == "__main__":
    main(sys.argv[1:])
