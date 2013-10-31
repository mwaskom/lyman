#! /usr/bin/env python
"""
Group fMRI analysis frontend for Lyman ecosystem.

"""
import os
import sys
import time
import shutil
import os.path as op

import matplotlib as mpl
mpl.use("Agg")

import nipype
from nipype import Node, MapNode, SelectFiles, DataSink, IdentityInterface

import lyman
import lyman.workflows as wf
from lyman import tools


def main(arglist):
    """Main function for workflow setup and execution."""
    args = parse_args(arglist)

    # Get and process specific information
    project = lyman.gather_project_info()
    exp = lyman.gather_experiment_info(args.experiment, args.altmodel)

    if args.experiment is None:
        args.experiment = project["default_exp"]

    if args.altmodel:
        exp_name = "-".join([args.experiment, args.altmodel])
    else:
        exp_name = args.experiment

    # Make sure some paths are set properly
    os.environ["SUBJECTS_DIR"] = project["data_dir"]

    # Set roots of output storage
    anal_dir_base = op.join(project["analysis_dir"], exp_name)
    work_dir_base = op.join(project["working_dir"], exp_name)
    nipype.config.set("execution", "crashdump_dir",
                      "/tmp/%s-nipype_crashes-%d" % (os.getlogin(),
                                                     time.time()))

    # Subject source (no iterables here)
    subject_list = lyman.determine_subjects(args.subjects)
    subj_source = Node(IdentityInterface(fields=["subject_id"]),
                       name="subj_source")
    subj_source.inputs.subject_id = subject_list

    # Set up the regressors and contrasts
    regressors = dict(group_mean=[1] * len(subject_list))
    contrasts = [["group_mean", "T", ["group_mean"], [1]]]

    # Subject level contrast source
    contrast_source = Node(IdentityInterface(fields=["l1_contrast"]),
                           iterables=("l1_contrast", exp["contrast_names"]),
                           name="contrast_source")

    # Group workflow
    space = args.regspace
    if space == "mni":
        mfx, mfx_input, mfx_output = wf.create_volume_mixedfx_workflow(
            args.output, subject_list, regressors, contrasts, exp)
    else:
        mfx, mfx_input, mfx_output = wf.create_surface_ols_workflow(
            args.output, subject_list, exp)

    # Mixed effects inputs
    ffxspace = "mni" if space == "mni" else "epi"
    mfx_base = op.join("{subject_id}/ffx/%s/smoothed/{l1_contrast}" % ffxspace)
    templates = dict(copes=op.join(mfx_base, "cope1.nii.gz"))
    if space == "mni":
        templates.update(dict(
            varcopes=op.join(mfx_base, "varcope1.nii.gz"),
            dofs=op.join(mfx_base, "tdof_t1.nii.gz")))
    else:
        templates.update(dict(
            reg_file=op.join(anal_dir_base, "{subject_id}/preproc/run_1",
                             "func2anat_tkreg.dat")))

    # Workflow source node
    mfx_source = MapNode(SelectFiles(templates,
                                     base_directory=anal_dir_base,
                                     sort_filelist=True),
                         "subject_id",
                         "mfx_source")

    # Workflow input connections
    mfx.connect([
        (contrast_source, mfx_source,
            [("l1_contrast", "l1_contrast")]),
        (contrast_source, mfx_input,
            [("l1_contrast", "l1_contrast")]),
        (subj_source, mfx_source,
            [("subject_id", "subject_id")]),
        (mfx_source, mfx_input,
            [("copes", "copes")])
                 ]),
    if space == "mni":
        mfx.connect([
            (mfx_source, mfx_input,
                [("varcopes", "varcopes"),
                 ("dofs", "dofs")]),
                     ])
    else:
        mfx.connect([
            (mfx_source, mfx_input,
                [("reg_file", "reg_file")]),
            (subj_source, mfx_input,
                [("subject_id", "subject_id")])
                     ])

    # Mixed effects outputs
    mfx_sink = Node(DataSink(base_directory="/".join([anal_dir_base,
                                                      args.output,
                                                      space]),
                             substitutions=[("/stats", "/"),
                                            ("_hemi_", ""),
                                            ("_glm_results", "")],
                             parameterization=True),
                    name="mfx_sink")

    mfx_outwrap = tools.OutputWrapper(mfx, subj_source,
                                      mfx_sink, mfx_output)
    mfx_outwrap.sink_outputs()
    mfx_outwrap.set_mapnode_substitutions(1)
    mfx_outwrap.add_regexp_substitutions([("_l1_contrast_\w*/", "/")])
    mfx.connect(contrast_source, "l1_contrast",
                mfx_sink, "container")

    # Set a few last things
    mfx.base_dir = work_dir_base

    # Execute
    lyman.run_workflow(mfx, args=args)

    # Clean up
    if project["rm_working_dir"]:
        shutil.rmtree(project["working_dir"])


def parse_args(arglist):
    """Take an arglist and return an argparse Namespace."""
    parser = tools.parser
    parser.add_argument("-experiment", help="experimental paradigm")
    parser.add_argument("-altmodel", help="alternate model to fit")
    parser.add_argument("-regspace", default="mni",
                        choices=["mni", "fsaverage"],
                        help="common space for group analysis")
    parser.add_argument("-output", default="group",
                        help="output directory name")
    return parser.parse_args(arglist)

if __name__ == "__main__":
    main(sys.argv[1:])
