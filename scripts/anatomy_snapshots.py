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
import nibabel as nib
from mayavi import mlab
from surfer import Brain

from moss.mosaic import Mosaic
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
            mlab.view(distance=400)
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
                                   width=700, height=500))
        curv = nib.load(op.join(surf_dir, "%s.curv.fsaverage.mgz" % hemi))
        curv = (curv.get_data() > 0).squeeze()
        b.add_contour_overlay(curv, min=0, max=1.5, n_contours=2, line_width=4)
        b.contour["colorbar"].visible = False
        for view in ["lat", "med"]:
            b.show_view(view)
            mlab.view(distance=330)
            png = op.join(snap_dir, "%s.surf_warp_%s.png" % (hemi, view))
            b.save_image(png)

        if close:
            b.close()


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
    parser.add_argument("-noclose", dest="close", action="store_false",
                        help="don't close mayavi figures during runtime")
    args = parser.parse_args(arglist)

    return args


if __name__ == "__main__":

    main(sys.argv[1:])
