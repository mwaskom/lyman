#! /usr/bin/env python
"""Wrapper script for booting up freeview with lyman results."""
import os
import sys
from pathlib import Path
from textwrap import dedent
import argparse
import numpy as np
import nibabel as nib
import lyman


def main(arglist):

    args = parse_args(arglist)

    project = lyman.gather_project_info()
    data_dir = Path(project["data_dir"])
    anal_dir = Path(project["analysis_dir"])

    # Work out elements of the path to the data
    if args.experiment is None:
        exp_name = project["default_exp"]
    else:
        exp_name = args.experiment
    smoothing = "unsmoothed" if args.unsmoothed else "smoothed"

    exp_base = exp_name
    if args.altmodel is not None:
        exp_name = "-".join([exp_base, args.altmodel])

    # Start building the command line
    cmdline = ["freeview"]

    # Background hires anatomy
    anat_vol = data_dir / args.subject / "mri/brain.mgz"
    cmdline.extend(["-v", str(anat_vol)])

    # Get a reference to the functional-to-anatomical registration
    reg_file = (anal_dir / exp_name / args.subject / "reg" / "epi" /
                smoothing / "run_1" / "func2anat_tkreg.dat")

    # Load the mean functional volume in the background
    mean_vol = (anal_dir / exp_name / args.subject / "ffx" / "epi" /
                smoothing / "mean_func.nii.gz")

    mean_arg = (str(mean_vol)
                + ":reg=" + str(reg_file)
                + ":colormap=gecolor"
                + ":colorscale=0,30000"
                + ":visible=0"
                + ":sample=trilinear")
    cmdline.extend(["-v", mean_arg])

    # Find the statistic volume to compute the colormap parameters
    stat_vol = (anal_dir / exp_name / args.subject / "ffx" / "epi" /
                smoothing / args.contrast / "zstat1.nii.gz")

    # Determine limits for the statistical colormap
    stat = np.abs(nib.load(str(stat_vol)).get_data())
    if args.vlims is None:
        cmap_max = max(4.2, np.percentile(np.abs(stat[stat > 2.3]), 98))
        cmap_arg = "1.64,2.3,{:.1f}".format(cmap_max)
    else:
        cmap_arg= "{},{},{}".format(*args.vlims)

    # Load the statistical overlay
    stat_arg = (str(stat_vol)
                + ":reg=" + str(reg_file)
                + ":colormap=heat"
                + ":heatscale=" + cmap_arg
                + ":sample=trilinear")
    cmdline.extend(["-v", stat_arg])

    # Mesh overlay
    for mesh in ["smoothwm", "inflated"]:
        for hemi in ["lh", "rh"]:
            surf = data_dir / args.subject / "surf" / (hemi + "." + mesh)
            stat = (anal_dir / exp_name / args.subject / "ffx" / "epi" /
                    smoothing / args.contrast / (hemi + ".zstat1.mgz"))
            surf_arg = (str(surf)
                        + ":edgecolor=limegreen"
                        + ":overlay=" + str(stat)
                        + ":overlay_color=heat"
                        + ":overlay_method=piecewise"
                        + ":overlay_threshold=" + cmap_arg
                        )
            if mesh == "smoothwm":
                surf_arg += ":hide_in_3d=true"
            cmdline.extend(["-f", surf_arg])

        cmdline.append("--hide-3d-slices")

    # Freeview spews a lot of garbage to the terminal; typcially silence that
    if not args.debug:
        cmdline.append("> /dev/null 2>&1")

    # Call out to freeview
    os.system(" ".join(cmdline))


def parse_args(arglist):

    help = dedent("""
    Display single-subject fixed effects results in Freeview.

    This script is a simple wrapper for the `freeview` binary that plugs in
    relevant paths to files in the lyman results hierarchy.

    """)

    parser = argparse.ArgumentParser()
    parser.description = help
    parser.add_argument("-subject", help="subject id")
    parser.add_argument("-experiment", help="experimental paradigm")
    parser.add_argument("-altmodel", help="show results from model name")
    parser.add_argument("-contrast", help="contrast name")
    parser.add_argument("-unsmoothed", action="store_true",
                        help="show unsmoothed results")
    parser.add_argument("-vlims", nargs=3, type=float,
                        help="custom colormap limits")
    parser.add_argument("-debug", action="store_true",
                        help="print freeview output in terminal")

    return parser.parse_args(arglist)


if __name__ == "__main__":
    main(sys.argv[1:])
