#! /usr/bin/env python
import os
import sys
import time
import shutil
import argparse
import nipype
import lyman
from lyman import tools
from lyman.workflows import anatwarp

help = """
Estimate a volume-based normalization to the MNI template.

This script can use either FSL tools (FLIRT and FNIRT) or ANTS to estimate a
nonlinear warp from the native anatomy to the MNI152 (nonlinear) template.
The normalization method is controlled through a variable in the project file.

Using ANTS can provide substantially improved accuracy, although ANTS can be
difficult to install, so this is not the default. The two methods are mutually
exclusive, and the outputs will overwrite each other.

Unlike other lyman scripts, the out is written to the `data_dir`, rather than
the `analysis_dir`.

This script will also produce a static image of the target overlaid on the
moving image for quality control. This is best viewed using ziegler.

Examples
--------

run_warp.py

    With no arugments, this will estimate the warp for all subjects using
    multiprocessing.

Usage Details
-------------

"""


def main(arglist):

    # Process cmdline args
    parser = tools.parser
    parser.description = help
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    args = tools.parser.parse_args(arglist)
    plugin, plugin_args = lyman.determine_engine(args)

    # Load up the lyman info
    subject_list = lyman.determine_subjects(args.subjects)
    project = lyman.gather_project_info()

    # Create the workflow object
    method = project["normalization"]
    wf_func = getattr(anatwarp, "create_{}_workflow".format(method))
    normalize = wf_func(project["data_dir"], subject_list)
    normalize.base_dir = project["working_dir"]
    
    # Put crashdumps somewhere out of the way
    nipype.config.set("execution", "crashdump_dir", project["crash_dir"])

    # Execute the workflow
    lyman.run_workflow(normalize, args=args)

    # Clean up
    if project["rm_working_dir"]:
        shutil.rmtree(normalize.base_dir)

if __name__ == "__main__":
    main(sys.argv[1:])
