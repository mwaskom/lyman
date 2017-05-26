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

    # Get the full correct name for the experiment
    if args.experiment is None:
        exp_name = project["default_exp"]
    else:
        exp_name = args.experiment

    # Start building the command line
    cmdline = ["freeview"]

    # Background hires anatomy
    anat_vol = data_dir / args.subject / "mri/brain.mgz"
    cmdline.extend(["-v", str(anat_vol)])

    # Load the mean functional volume with its registration matrix
    preproc_dir = (anal_dir / exp_name / args.subject
                   / "preproc" / ("run_" + args.run))
    epi_vol = preproc_dir / "mean_func.nii.gz"
    reg_mat = preproc_dir / "func2anat_tkreg.dat"
    epi_arg = (str(epi_vol)
               + ":reg=" + str(reg_mat)
               + ":sample=cubic")
    cmdline.append(epi_arg)

    # Load the white and pial surfaces
    cmdline.append("-f")
    meshes = ["white" ,"pial"]
    colors = ['#fac205', u'#c44240']
    for mesh, color in zip(meshes, colors):
        for hemi in ["lh", "rh"]:
            surf = data_dir / args.subject / "surf" / (hemi + "." + mesh)
            surf_arg = (str(surf)
                        + ":edgecolor=" + color
                        + ":hide_in_3d=true")
            cmdline.append(surf_arg)

    # Show the coronal view by default
    cmdline.extend(["-viewport", "coronal"])

    # Freeview spews a lot of garbage to the terminal; typcially silence that
    if not args.debug:
        cmdline.append("> /dev/null 2>&1")

    # Call out to freeview
    os.system(" ".join(cmdline))


def parse_args(arglist):

    help = dedent("""
    Display the results of the functional-to-anatomical registration.

    This script is a simple wrapper for the `freeview` binary that plugs in
    relevant paths to files in the lyman results hierarchy.

    """)

    parser = argparse.ArgumentParser()
    parser.description = help
    parser.add_argument("-subject", help="subject id")
    parser.add_argument("-experiment", help="experimental paradigm")
    parser.add_argument("-run", default="1", help="experimental run")
    parser.add_argument("-debug", action="store_true",
                        help="print freeview output in terminal")

    return parser.parse_args(arglist)


if __name__ == "__main__":
    main(sys.argv[1:])
