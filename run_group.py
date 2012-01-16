#! /usr/bin/env python
"""
Group fMRI analysis frontend for Lyman ecosystem

"""
import os
import sys
import time
import shutil
import os.path as op

import matplotlib as mpl
mpl.use("Agg")
from nipype.pipeline.engine import Node, MapNode
from nipype.interfaces.io import DataGrabber, DataSink
from nipype.interfaces.utility import IdentityInterface

import workflows as wf
import tools
from tools.commandline import parser


def main(arglist):
    """Main function for workflow setup and execution."""
    args = parse_args(arglist)

    # Get and process specific information
    project = tools.gather_project_info()
    exp = tools.gather_experiment_info(args.experiment, args.altmodel)

    if args.altmodel:
        exp_name = "-".join([args.experiment, args.altmodel])
    else:
        exp_name = args.experiment

    # Make sure some paths are set properly
    os.environ["SUBJECTS_DIR"] = project["data_dir"]
    sys.path.insert(0, os.path.abspath("."))

    # Set roots of output storage
    anal_dir_base = op.join(project["analysis_dir"], exp_name)
    work_dir_base = op.join(project["working_dir"], exp_name)
    crashdump_dir = "/tmp/%d" % time.time()

    # Set up the regressors and contrasts
    regressors = dict(group_mean=[1] * len(args.subjects))
    contrasts = ["group_mean", "T", ["group_mean"], [1]]

    # Subject source (no iterables here)
    subj_source = Node(IdentityInterface(fields=["subject_id"]),
                       name="subj_source")
    subj_source.inputs.subject_id = args.subjects

    # Subject level contrast source
    contrast_source = Node(IdentityInterface(fields=["l1_contrast"]),
                           iterables=("l1_contrast", exp["contrast_names"]),
                           name="contrast_source")

    # Mixed effects group workflow
    mfx, mfx_input, mfx_output = wf.create_volume_mixedfx_workflow(
        regressors=regressors, contrasts=contrasts)

    # Mixed effects inputs
    mfx_template = "%s/ffx/" + args.regspace + "/smoothed/%s/%s.nii.gz"
    mfx_source = MapNode(DataGrabber(infields=["subject_id",
                                               "l1_contrast"],
                                     outfields=["copes", "varcopes", "dofs"],
                                     base_directory=anal_dir_base,
                                     template=mfx_template,
                                     sort_filelist=True),
                      iterfield=["subject_id"],
                      name="mfx_source")
    mfx_source.inputs.template_args = dict(
        copes=[["subject_id", "l1_contrast", "cope"]],
        varcopes=[["subject_id", "l1_contrast", "varcope"]],
        dofs=[["subject_id", "l1_contrast", "tdof_t"]])

    mfx.connect([
        (contrast_source, mfx_source, 
            [("l1_contrast", "l1_contrast")]),
        (subj_source, mfx_source,
            [("subject_id", "subject_it")]),
        (mfx_source, mfx_input,
            [("copes", "copes"),
             ("varcopes", "varcopes"),
             ("dofs", "dofs")]),
             ])

    # Mixed effects outputs
    mfx_sink = Node(DataSink(base_directory=anal_dir_base,
                             parameterization=False),
                    name="mfx_sink")

    mfx_outwrap = tools.OutputWrapper(mfx, subj_source,
                                      mfx_sink, mfx_output)
    mfx_outwrap.sink_outputs("group.%s" % args.regspace)
    mfx.connect(contrast_source, "l1_contrast",
                mfx_sink, "container")

    # Set a few last things
    mfx.base_dir = work_dir_base
    mfx.config = dict(crashdump_dir=crashdump_dir)

    # Execute
    tools.run_workflow(mfx)

    # Clean up
    if project["rm_working_dir"]:
        shutil.rmtree(project["working_dir"])


def parse_args(arglist):
    """Take an arglist and return an argparse Namespace."""
    parser.add_argument("-experiment", help="experimental paradigm")
    parser.add_argument("-altmodel", help="alternate model to fit")
    parser.add_argument("-regspace", default="mni",
                        choices=wf.spaces,
                        help="common space for registration and fixed effects")
    return parser.parse_args(arglist)

if __name__ == "__main__":
    main(sys.argv[1:])
