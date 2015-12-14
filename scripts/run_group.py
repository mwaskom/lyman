#! /usr/bin/env python
"""
Group fMRI analysis frontend for Lyman ecosystem.

"""
import os
import sys
import time
import shutil
import os.path as op
from textwrap import dedent
import argparse

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
    exp = lyman.gather_experiment_info(args.experiment, args.altmodel, args)

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
    nipype.config.set("execution", "crashdump_dir", project["crash_dir"])

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
    wf_name = "_".join([space, args.output])
    if space == "mni":
        mfx, mfx_input, mfx_output = wf.create_volume_mixedfx_workflow(
            wf_name, subject_list, regressors, contrasts, exp)
    else:
        mfx, mfx_input, mfx_output = wf.create_surface_ols_workflow(
            wf_name, subject_list, exp)

    # Mixed effects inputs
    ffxspace = "mni" if space == "mni" else "epi"
    ffxsmooth = "unsmoothed" if args.unsmoothed else "smoothed"
    mfx_base = op.join("{subject_id}/ffx/%s/%s/{l1_contrast}" % (ffxspace,
                                                                 ffxsmooth))
    templates = dict(copes=op.join(mfx_base, "cope1.nii.gz"))
    if space == "mni":
        templates.update(dict(
            varcopes=op.join(mfx_base, "varcope1.nii.gz"),
            dofs=op.join(mfx_base, "tdof_t1.nii.gz")))
    else:
        templates.update(dict(
            reg_file=op.join(anal_dir_base,
                             "{subject_id}/reg/epi/", ffxsmooth,
                             "run_1/func2anat_tkreg.dat")))

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
                                            ("/_hemi_", "/"),
                                            ("_glm_results", "")],
                             parameterization=True),
                    name="mfx_sink")

    mfx_outwrap = tools.OutputWrapper(mfx, subj_source,
                                      mfx_sink, mfx_output)
    mfx_outwrap.sink_outputs()
    mfx_outwrap.set_mapnode_substitutions(1)
    mfx_outwrap.add_regexp_substitutions([
        (r"_l1_contrast_[-\w]*/", "/"),
        (r"_mni_hemi_[lr]h", "")
        ])
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
    help = dedent("""
    Perform a basic group analysis in lyman.

    This script currently only handles one-sample group mean tests on each of
    the fixed-effects contrasts. It is possible to run the group model in the
    volume or on the surface, although the actual model changes depending on
    this choice.

    The volume model runs FSL's FLAME mixed effects for hierarchical inference,
    which uses the lower-level variance estimates, and it applies standard
    GRF-based correction for multiple comparisons.  The details of the
    model-fitting procedure are set in the experiment file, along with the
    thresholds used for correction.

    The surface model uses a standard ordinary least squares fit and does
    correction with an approach based on a Monte Carlo simulation of the null
    distribution of cluster sizes for smoothed Gaussian data. Fortunately, the
    simulations are cached so this runs very quickly. Unfortunately, the cached
    simulations used a whole-brain search space, so this will be overly
    conservative for partial-brain acquisitions.

    Because of how GRF-based correction works, the thresholded volume images
    only have positive voxels. It is up to you to define "negative" versions of
    any contrasts where you are interested in relative deactivation.  The
    surface correction does not have this constraint, and the test sign is
    configurable in the experiment file (and will thus apply to all contrasts).

    By default the results are written under `group` next to the subject level
    data in the lyman analysis directory, although the output directory name
    can be changed.

    Examples
    --------

    Note that the parameter switches match any unique short version
    of the full parameter name.

    run_group.py

        With no arguments, this will process the default experiment with the
        subjects defined in $LYMAN_DIR/subjects.txt in the MNI space using the
        MultiProc plugin with 4 processes.

    run_group.py -s pilot_subjects -r fsaverage -o pilot -unsmoothed

        This will processes the subjects defined in a file at
        $LYMAN_DIR/pilot_subjects.txt as above but with the surface workflow.
        Unsmoothed fixed effects parameter estimates will be sampled to the
        surface and smoothed there. The resulting files will be stored under
        <analysis_dir>/<experiment>/pilot/fsaverage/<contrast>/<hemi>

    run_group.py -e nback -a parametric -p sge -q batch.q

        This will process an alternate model for the `nback` experiment using
        the SGE plugin by submitting jobs to the batch.q queue.

    Usage Details
    -------------

    """)

    parser = tools.parser
    parser.description = help
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.add_argument("-experiment", help="experimental paradigm")
    parser.add_argument("-altmodel", help="alternate model to fit")
    parser.add_argument("-regspace", default="mni",
                        choices=["mni", "fsaverage"],
                        help="common space for group analysis")
    parser.add_argument("-unsmoothed", action="store_true",
                        help="used unsmoothed fixed effects outputs")
    parser.add_argument("-output", default="group",
                        help="output directory name")
    return parser.parse_args(arglist)

if __name__ == "__main__":
    main(sys.argv[1:])
