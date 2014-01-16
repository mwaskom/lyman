#! /usr/bin/env python
"""Create masks in native functional space from a variety of sources.
"""
import sys
import json
import time
import os.path as op
import argparse
from textwrap import dedent

from lyman.maskfactory import MaskFactory


def main(arglist):

    # Parse command line arguments
    args = parse_args(arglist)

    # Determine the type of processing we will do
    # First look for shortcut keys
    if args.label is not None:
        orig_type = "native_label" if args.native else "fsaverage_label"
    elif args.contrast is not None:
        orig_type = "stat_volume"
    elif args.aseg is not None:
        orig_type = "index_volume"
    # Otherwise we have to figure it out from the other arguments
    elif args.orig is not None and args.orig.endswith(".label"):
        orig_type = "native_label" if args.label else "fsaverage_label"
    elif args.id is not None:
        orig_type = "index_volume"
    else:
        raise ValueError("Could not determine orig type from arguments.")
    if args.debug:
        print "Processing type: %s" % orig_type

    # Initialise a factory object
    factory = MaskFactory(args.subjects, args.exp, args.roi, orig_type,
                          args.serial, args.debug)

    # Ensure that the orig file is an absolute path
    if args.orig is not None:
        args.orig = op.abspath(args.orig)

    # Execute the stream

    # Label files
    if "label" in orig_type:
        if args.label is not None:
            file_temp = "%(hemi)s." + args.label + ".label"
        hemis = ["lh", "rh"] if args.hemi is None else [args.hemi]
        if args.sample is not None:
            proj_args = dict(white=["frac", "0", "0", "0"],
                             graymid=["frac", ".5", ".5", "0"],
                             pial=["frac", "1", "1", "0"],
                             cortex=["frac", "0", "1", ".1"])[args.sample]
        else:
            proj_args = args.proj
        if orig_type == "native_label":
            label_temp = op.join(factory.data_dir, "%(subj)s",
                                 "label", file_temp)
            factory.from_native_label(label_temp, hemis, proj_args)
        elif orig_type == "fsaverage_label":
            label_temp = op.join(factory.data_dir, "fsaverage",
                                 "label", file_temp)
            factory.from_common_label(label_temp, hemis,
                                      proj_args, args.save_native)

    # Index volumes (atlas type)
    elif orig_type == "index_volume":
        if args.aseg:
            atlas_temp = op.join(factory.data_dir,
                                 "%(subj)s",
                                 "mri", "aseg.mgz")
        else:
            atlas_temp = args.orig
        
        factory.from_hires_atlas(atlas_temp, args.id, args.erode)

    # Combine mask image with a thresholded stat volume
    if args.contrast is not None:
        smooth_dir = "unsmoothed" if args.unsmoothed else "smoothed"
        if args.altmodel is None:
            exp = factory.experiment
        else:
            exp = factory.experiment + "-" + args.altmodel
        stat_file_temp = op.join(factory.anal_dir,
                                 exp,
                                 "%(subj)s",
                                 "ffx", "epi",
                                 smooth_dir,
                                 args.contrast,
                                 "zstat1.nii.gz")
        factory.apply_statistical_mask(stat_file_temp, args.thresh)

    # Write an image of the mask
    factory.write_png()

    # Get the current date
    args.created = time.asctime()
    # Write provenence information
    for subj in factory.subject_list:
        json_file = op.join(factory.data_dir, subj, "masks/%s.json" % args.roi)
        with open(json_file, "w") as fid:
            json.dump(args.__dict__, fid, sort_keys=True)


def parse_args(arglist):
    """Handle the command line."""
    help = dedent("""
    Currently this can start with ROIs defined as a surface label on
    fsaverage, labels defined on each subject's native surface, ROIs
    defined on the high-res volume in Freesurfer space, or a statistical
    volume from a subject-level analysis.

    You can always pass a filepath (possibly with ``subj`` and ``hemi``
    string format keys to -orig and the program will work out what to
    do from the file type and other arugments. Alternatively, if files
    are in expected places (Freesurfer data hierarchy, lyman analysis
    hierarchy) there are shortcuts for the corresponding image type.

    The processing is almost entirely dependent on external binaries
    from FSL and Freesurfer, so both must be availible.

    The resulting masks are defined in the space of the first functional
    run. This is also the target of the ``-regspace epi`` registration
    in the main lyman fmri workflows.

    The processing here is closely tied to these fmri workflows and requires
    subject-level preprocessing to have been performed. This program
    should be executed from a directory containing a project.py file
    that defines the relevant data and analysis paths.

    The script will also write a mosiac png with the mask overlaid on
    the mean functional image defining the epi space. Additionally, it
    will write a json file with the command line argument dictionary
    for provenence tracking.

    If an IPython cluster is running, the processing will be executed
    in parallel by default on all availible engines. This can be avoided
    by using the -serial option.

    Usage Details
    -------------

    """)

    parser = argparse.ArgumentParser(description=help,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Necessary arguments
    parser.add_argument("-s", "-subjects", nargs="*", dest="subjects",
                        help="lyman subjects argument")
    parser.add_argument("-roi", required=True,
                        help="will form name of output mask file")

    # Optional argument (that always applies)
    parser.add_argument("-exp",
                        help="experiment (can use default from project.py)")

    # Generic input
    parser.add_argument("-orig",
        help="path to original file with subj and hemi format keys")

    # Label relavent arguments
    parser.add_argument("-label", help="label name if in Freesurfer hierachy")
    parser.add_argument("-native", action="store_true",
                        help="orig label is defined on native surface")
    parser.add_argument("-hemi", choices=["lh", "rh"],
        help="hemisphere if unilateral label - otherwise combine both")
    parser.add_argument("-sample",
                        choices=["white", "graymid", "pial", "cortex"],
                        help="shortcut for projection arguments")
    parser.add_argument("-proj", nargs=4,
                        help="projection args passed directly mri_label2vol")
    parser.add_argument("-save_native", action="store_true",
                        help="save label file after warping from common space")

    # Atlas-type image relevant images
    parser.add_argument("-aseg", action="store_true",
                        help="atlas image is aseg.mgz")
    parser.add_argument("-erode",
                        help="amount of mask erosion performed \
                        by mri_binarize (default: None)")
    parser.add_argument("-id", type=int, nargs="*",
                        help="roi id(s) if orig is index volume")
    

    # Thresholded statistic relevant arguments
    parser.add_argument("-contrast",
                        help="first-level contrast to binarize z-stat map")
    parser.add_argument("-thresh", help="z-stat threshold")
    parser.add_argument("-unsmoothed", action="store_true",
                        help="use unsmoothed fixed effects zstats")
    parser.add_argument("-altmodel",
                        help="stat file is from alternative model")

    # Generic execution relevant arguments
    parser.add_argument("-serial", action="store_true",
                        help="force serial execution")
    parser.add_argument("-debug", action="store_true",
                        help="enable debug mode")

    return parser.parse_args(arglist)


if __name__ == "__main__":
    main(sys.argv[1:])
