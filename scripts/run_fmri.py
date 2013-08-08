#! /usr/bin/env python
"""
Main execution script for fMRI analysis in the Lyman ecosystem.

"""
import os
import sys
import time
import shutil
import os.path as op

import matplotlib as mpl
mpl.use("Agg")

import nipype
from nipype import Node, SelectFiles, DataSink, IdentityInterface

import lyman
import lyman.workflows as wf
from lyman import tools


def main(arglist):
    """Main function for workflow setup and execution."""
    args = parse_args(arglist)

    # Get and process specific information
    project = lyman.gather_project_info()
    exp = lyman.gather_experiment_info(args.experiment, args.altmodel)

    # Set up the SUBJECTS_DIR for Freesurfer
    os.environ["SUBJECTS_DIR"] = project["data_dir"]

    # Subject is always highest level of parameterization
    subject_list = lyman.determine_subjects(args.subjects)
    subj_source = tools.make_subject_source(subject_list)

    # Get the full correct name for the experiment
    if args.experiment is None:
        exp_name = project["default_exp"]
    else:
        exp_name = args.experiment

    exp_base = exp_name
    if args.altmodel is not None:
        exp_name = "-".join([exp_base, args.altmodel])

    # Set roots of output storage
    data_dir = project["data_dir"]
    analysis_dir = op.join(project["analysis_dir"], exp_name)
    working_dir = op.join(project["working_dir"], exp_name)
    preproc_dir = op.join(project["analysis_dir"], exp_base)

    nipype.config.set("execution", "crashdump_dir",
                      "/tmp/%s-%d" % (os.getlogin(), time.time()))

    # This might not exist if we're running an altmodel
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)

    # Preprocessing Workflow
    # ======================

    # Create workflow in function defined elsewhere in this package
    preproc, preproc_input, preproc_output = wf.create_preprocessing_workflow(
                              temporal_interp=exp["temporal_interp"],
                              frames_to_toss=exp["frames_to_toss"],
                              interleaved=exp["interleaved"],
                              slice_order=exp["slice_order"],
                              TR=exp["TR"],
                              intensity_threshold=3,
                              motion_threshold=1,
                              smooth_fwhm=exp["smooth_fwhm"],
                              highpass_sigma=exp["hpf_sigma"],
                              partial_brain=exp["partial_brain"])

    # Collect raw nifti data
    preproc_templates = dict(timeseries=exp["source_template"])
    if exp["partial_brain"]:
        preproc_templates["whole_brain_epi"] = exp["whole_brain_template"]

    preproc_source = Node(SelectFiles(infields=["subject_id"],
                                      templates=preproc_templates,
                                      base_directory=project["data_dir"],
                                      sort_filelist=True),
                          name="preproc_source")

    # Convenience class to handle some sterotyped connections
    # between run-specific nodes (defined here) and the inputs
    # to the prepackaged workflow returned above
    preproc_inwrap = tools.InputWrapper(preproc, subj_source,
                                        preproc_source, preproc_input)
    preproc_inwrap.connect_inputs()

    # Store workflow outputs to persistant location
    preproc_sink = Node(DataSink(base_directory=analysis_dir),
                        name="preproc_sink")

    # Similar to above, class to handle sterotyped output connections
    preproc_outwrap = tools.OutputWrapper(preproc, subj_source,
                                          preproc_sink, preproc_output)
    preproc_outwrap.set_subject_container()
    preproc_outwrap.set_mapnode_substitutions(exp["n_runs"])
    preproc_outwrap.sink_outputs("preproc")

    # Set the base for the possibly temporary working directory
    preproc.base_dir = working_dir

    # Possibly execute the workflow, depending on the command line
    lyman.run_workflow(preproc, "preproc", args)

    # Timeseries Model
    # ================

    # Variable to control whether volume processing inputs are smoothed
    # Surface inputs are always unsmoothed, but that's not relevant till below
    model_smooth = "unsmoothed" if args.unsmoothed else "smoothed"

    # Create a modelfitting workflow and specific nodes as above
    model, model_input, model_output = wf.create_timeseries_model_workflow(
        name=model_smooth + "_model", exp_info=exp)

    preproc_template = op.join(analysis_dir, "{subject_id}/preproc/run_*/")
    design_file = exp["design_name"] + ".csv"
    timeseries_file = model_smooth + "_timeseries.nii.gz"
    model_templates = dict(
        design_file=op.join(data_dir, "{subject_id}/design", design_file),
        realign_file=op.join(preproc_template, "realignment_params.csv"),
        artifact_file=op.join(preproc_template, "artifacts.csv"),
        timeseries=op.join(preproc_template, timeseries_file),
                           )

    model_source = Node(SelectFiles(["subject_id"],
                                    templates=model_templates,
                                    sort_filelist=True),
                        name="model_source")

    model_inwrap = tools.InputWrapper(model, subj_source,
                                      model_source, model_input)
    model_inwrap.connect_inputs()

    model_sink = Node(DataSink(base_directory=analysis_dir),
                               name="model_sink")

    model_outwrap = tools.OutputWrapper(model, subj_source,
                                        model_sink, model_output)
    model_outwrap.set_subject_container()
    model_outwrap.set_mapnode_substitutions(exp["n_runs"])
    model_outwrap.sink_outputs("model." + model_smooth)

    # Set temporary output locations
    model.base_dir = working_dir

    # Possibly execute the workflow
    lyman.run_workflow(model, "model", args)

    # Across-Run Registration
    # =======================

    # Short ref to the common space we're using on this execution
    space = args.regspace

    # Is this a model or timeseries registration?
    regtype = "timeseries" if args.timeseries else "model"

    # Retrieve the right workflow function for registration
    # Get the workflow function dynamically based on the space
    flow_name = "%s_%s_reg" % (space, regtype)
    reg, reg_input, reg_output = wf.create_reg_workflow(flow_name,
                                                        space,
                                                        regtype)

    # Define a smooth variable here. Use an iterable so that running
    # with/without smoothing doesn't clobber working directory files
    # for the other kind of execution
    reg_smooth = "unsmoothed" if args.unsmoothed else "smoothed"
    smooth_source = Node(IdentityInterface(fields=["smoothing"]),
                         iterables=("smoothing", [reg_smooth]),
                         name="smooth_source")

    # Set up the registration inputs and templates
    reg_infields = ["subject_id", "smoothing"]

    if regtype == "model":
        reg_templates = dict(
            copes="{subject_id}/model/{smoothing}/run_*/cope*.nii.gz",
            varcopes="{subject_id}/model/{smoothing}/run_*/varcope*.nii.gz"
                             )
    else:
        reg_templates = dict(
            timeseries=op.join("{subject_id}/preproc/run_*/",
                               "{smoothing}_timeseries.nii.gz"),
                             )
    reg_templates.update(dict(
        masks="{subject_id}/preproc/run_*/functional_mask.nii.gz",
        affines="{subject_id}/preproc/run_*/func2anat_flirt.mat"
                              ))

    if space == "mni":
        reg_templates.update(dict(
            warpfield=op.join(data_dir,
                              "{subject_id}/normalization/warpfield.nii.gz")
                                  ))

    # Define the registration data source node
    reg_source = Node(SelectFiles(reg_infields,
                                  templates=reg_templates,
                                  base_directory=analysis_dir,
                                  sort_filelist=True),
                      "reg_source")

    # Registration inutnode
    reg_inwrap = tools.InputWrapper(reg, subj_source,
                                    reg_source, reg_input)
    reg_inwrap.connect_inputs()

    # This source node also needs to know about the smoothing on this run
    reg.connect(smooth_source, "smoothing", reg_source, "smoothing")

    # Reg output and datasink
    reg_sink = Node(DataSink(base_directory=analysis_dir),
                             name="reg_sink")

    reg_outwrap = tools.OutputWrapper(reg, subj_source,
                                    reg_sink, reg_output)
    reg_outwrap.set_subject_container()
    reg_outwrap.sink_outputs("reg.%s" % space)

    # Reg has some additional substitutions to strip out interables
    reg_outwrap.add_regexp_substitutions([
        (r"_smoothing_", ""),
        (r"(un)*(smoothed_time)", "time"),
                                          ])

    reg.base_dir = working_dir

    # Possibly run registration workflow and clean up
    lyman.run_workflow(reg, "reg", args)

    # Cross-Run Fixed Effects Model
    # =============================

    space_source = Node(IdentityInterface(fields=["space"]),
                        iterables=("space", [space]),
                        name="space_source")
    contrast_source.iterables = ("contrast", exp["contrast_names"])

    # Dynamically get the workflow
    manifold = "volume"
    workflow_function = getattr(wf, "create_%s_ffx_workflow" % manifold)
    ffx, ffx_input, ffx_output = workflow_function()

    ffx_infields = ["subject_id", "contrast_number", "space"]
    ffx_outfields = ["copes", "varcopes", "dof_files",
                     "masks", "background_file"]

    ffx_template = "%s/reg/%s/%s/run_*/%s%d_*.nii.gz"

    ffx_template_args = dict(
        copes=[["subject_id", "space", "smooth", "cope", "contrast_number"]],
        varcopes=[["subject_id", "space", "smooth",
                   "varcope", "contrast_number"]],
        masks=[["subject_id", "space", "smooth"]],
        dof_files=[["subject_id", "smooth"]])

    ffx_field_template = {"dof_files": "%s/model/%s/run_*/results/dof",
        "masks": "%s/reg/%s/%s/run_*/functional_mask_*.nii.gz"}
    if space == "epi":
        ffx_field_template["background_file"] = op.join(
            preproc_dir, "%s/preproc//run_1/mean_func.nii.gz")
    elif space == "mni":
        ffx_field_template["background_file"] = op.join(
            project["data_dir"], "%s/normalization/brain_warp.nii.gz")
    ffx_template_args["background_file"] = [["subject_id"]]

    # Define the ffxistration data source node
    ffx_source = Node(DataGrabber(infields=ffx_infields,
                                  outfields=ffx_outfields,
                                  base_directory=analysis_dir,
                                  template=ffx_template,
                                  sort_filelist=True),
                      name="ffx_source")
    ffx_source.inputs.field_template = ffx_field_template
    ffx_source.inputs.template_args = ffx_template_args

    # Fixed effects inutnode
    ffx_inwrap = tools.InputWrapper(ffx, subj_source,
                                    ffx_source, ffx_input)
    ffx_inwrap.connect_inputs()

    # There are some additional connections to this input
    ffx.connect(space_source, "space", ffx_source, "space")
    ffx.connect(contrast_source, "contrast", ffx_input, "contrast")
    if not args.timeseries:
        names = exp["contrast_names"]
        ffx.connect(
             contrast_source, ("contrast", tools.find_contrast_number, names),
             ffx_source, "contrast_number")
    ffx.connect(smooth_source, "smooth", ffx_source, "smooth")

    # Fixed effects output and datasink
    ffx_sink = Node(DataSink(base_directory=analysis_dir),
                             name="ffx_sink")

    ffx_outwrap = tools.OutputWrapper(ffx, subj_source,
                                      ffx_sink, ffx_output)
    ffx_outwrap.set_mapnode_substitutions(exp["n_runs"])
    ffx_outwrap.set_subject_container()
    ffx_outwrap.sink_outputs("ffx.%s.%s" % (space, model_smooth))

    # Fixed effects has some additional substitutions to strip out interables
    ffx_outwrap.add_regexp_substitutions([
        (r"/_source_image_[^/]*/", ""),
        (r"/_space_[^/]*/", "/"),
        (r"/_smooth_[^/]*/", "/"),
        (r"/_contrast_", "/")])

    ffx.base_dir = working_dir

    # Possibly run fixed effects workflow
    lyman.run_workflow(ffx, "ffx", args)

    # Clean-up
    # --------

    if project["rm_working_dir"]:
        shutil.rmtree(project["working_dir"])


def parse_args(arglist):
    """Take an arglist and return an argparse Namespace."""
    parser = tools.parser
    parser.add_argument("-experiment", help="experimental paradigm")
    parser.add_argument("-altmodel", help="alternate model to fit")
    parser.add_argument("-workflows", nargs="*",
                        choices=["all", "preproc", "model", "reg", "ffx"],
                        help="which workflows to run")
    parser.add_argument("-regspace", default="mni",
                        choices=wf.spaces,
                        help="common space for registration and fixed effects")
    parser.add_argument("-timeseries", action="store_true",
                        help="perform registration on preprocessed timeseries")
    parser.add_argument("-unsmoothed", action="store_true",
                        help="model and reg use unsmoothed image in volume")
    return parser.parse_args(arglist)

if __name__ == "__main__":
    main(sys.argv[1:])
