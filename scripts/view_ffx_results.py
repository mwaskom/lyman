#! /usr/bin/env python
"""Wrapper script for booting up freeview with lyman results."""
import os
import sys
from pathlib import Path
from textwrap import dedent
import argparse
import lyman


def main(arglist):

    args = parse_args(arglist)

    project = lyman.gather_project_info()
    data_dir = Path(project["data_dir"])
    anal_dir = Path(project["analysis_dir"])

    # Get the full correct name for the experiment
    if args.experiment is None:
        exp_name = project["default_exp"]
    else:
        exp_name = args.experiment

    exp_base = exp_name
    if args.altmodel is not None:
        exp_name = "-".join([exp_base, args.altmodel])

    # Start building the command line
    cmdline = ["freeview"]

    # Background hires anatomy
    anat_vol = data_dir / args.subject / "mri/brain.mgz"
    cmdline.extend(["-v", str(anat_vol)])

    # Statistical overlay
    smoothing = "unsmoothed" if args.unsmoothed else "smoothed"
    stat_vol = (anal_dir / exp_name / args.subject / "ffx" / "epi" /
                smoothing / args.contrast / "zstat1.nii.gz")
    reg_file = (anal_dir / exp_name / args.subject / "reg" / "epi" /
                smoothing / "run_1" / "func2anat_tkreg.dat")
    stat_arg = (str(stat_vol) +
                ":reg=" + str(reg_file) +
                ":colormap=heat" +
                ":heatscale=1.64,2.3,4.2" +
                ":sample=trilinear")
    cmdline.extend(["-v", stat_arg])

    # Mesh overlay
    if args.mesh is not None:
        for hemi in ["lh", "rh"]:
            surf = data_dir / args.subject/ "surf" / (hemi + "." + args.mesh)
            surf_arg = str(surf) + ":edgecolor=limegreen"
            cmdline.extend(["-f", surf_arg])

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
    parser.add_argument("-mesh", help="surface mesh to plot")

    return parser.parse_args(arglist)


if __name__ == "__main__":
    main(sys.argv[1:])
