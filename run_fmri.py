#! /usr/bin/env python
"""
Main execution script for fMRI analysis in the Lyman ecosystem.

"""
import os
import re
import sys
import time
import shutil
import os.path as op

import matplotlib as mpl
mpl.use("Agg")
from nipype.pipeline.engine import Node
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
    exp = gather_experiment_info(args.experiment)

    # Make sure some paths are set properly
    os.environ["SUBJECTS_DIR"] = project["data_dir"]
    sys.path.insert(0, os.path.abspath("."))

    # Subject is always highest level of parameterization
    subject_list = tools.determine_subjects(args.subjects)
    subj_source = tools.make_subject_source(subject_list)

    # Can run model+ processing several times on preprocessed data
    if args.altmodel:
        exp_name = "-".join([args.experiment, args.altmodel])
    else:
        exp_name = args.experiment

    # Set roots of output storage
    anal_dir_base = op.join(project["analysis_dir"], exp_name)
    work_dir_base = op.join(project["working_dir"], exp_name)
    preproc_dir = op.join(project["analysis_dir"], args.experiment)
    crashdump_dir = "/tmp/%d" % time.time()

    # Preprocessing Workflow
    # ======================

    # Create workflow in function defined elsewhere in this package
    preproc, preproc_input, preproc_output = wf.create_preprocessing_workflow(
                              do_slice_time_cor=exp["slice_time_correction"],
                              frames_to_toss=exp["frames_to_toss"],
                              interleaved=exp["interleaved"],
                              slice_order=exp["slice_order"],
                              TR=exp["TR"],
                              smooth_fwhm=exp["smooth_fwhm"],
                              highpass_sigma=exp["hpf_sigma"])

    # Collect raw nifti data
    preproc_source = Node(DataGrabber(infields=["subject_id"],
                                      outfields=["timeseries"],
                                      base_directory=project["data_dir"],
                                      template=exp["source_template"],
                                      sort_filelist=True),
                          name="preproc_source")
    preproc_source.inputs.template_args = dict(timeseries=[["subject_id"]])

    # Convenience class to handle some sterotyped connections
    # between run-specific nodes (defined here) and the inputs
    # to the prepackaged workflow returned above
    preproc_inwrap = tools.InputWrapper(preproc, subj_source,
                                        preproc_source, preproc_input)
    preproc_inwrap.connect_inputs()

    # Store workflow outputs to persistant location
    preproc_sink = Node(DataSink(base_directory=anal_dir_base),
                        name="preproc_sink")

    # Similar to above, class to handle sterotyped output connections
    preproc_outwrap = tools.OutputWrapper(preproc, subj_source,
                                          preproc_sink, preproc_output)
    preproc_outwrap.set_subject_container()
    preproc_outwrap.set_mapnode_substitutions(exp["n_runs"])
    preproc_outwrap.sink_outputs("preproc")

    # Set the base for the possibly temporary working directory
    preproc.base_dir = work_dir_base

    # Just write crashes to the system /tmp location
    # This may break with next nipype release
    preproc.config = dict(crashdump_dir=crashdump_dir)

    # Possibly execute the workflow, depending on the command line
    run_workflow(preproc, "preproc", args)

    # Timeseries Model
    # ================

    # Variable to control whether volume processing inputs are smoothed
    # Surface inputs are always unsmoothed, but that's not relevant till below
    model_smooth = "unsmoothed" if args.unsmoothed else "smoothed"

    # Create a modelfitting workflow and specific nodes as above
    model, model_input, model_output = wf.create_timeseries_model_workflow(
        name=model_smooth + "_model", exp_info=exp)

    model_source = Node(DataGrabber(infields=["subject_id"],
                                    outfields=["outlier_files",
                                               "mean_func",
                                               "realign_params",
                                               "timeseries"],
                                    base_directory=anal_dir_base,
                                    template="%s/preproc/run_*/%s",
                                    sort_filelist=True),
                        name="model_source")
    model_source.inputs.template_args = dict(
        outlier_files=[["subject_id", "outlier_volumes.txt"]],
        mean_func=[["subject_id", "mean_func.nii.gz"]],
        realign_params=[["subject_id", "realignment_parameters.par"]],
        timeseries=[["subject_id", model_smooth + "_timeseries.nii.gz"]])

    model_inwrap = tools.InputWrapper(model, subj_source,
                                      model_source, model_input)
    model_inwrap.connect_inputs()

    model_sink = Node(DataSink(base_directory=anal_dir_base),
                               name="model_sink")

    model_outwrap = tools.OutputWrapper(model, subj_source,
                                       model_sink, model_output)
    model_outwrap.set_subject_container()
    model_outwrap.set_mapnode_substitutions(exp["n_runs"])
    model_outwrap.sink_outputs("model." + model_smooth)

    # Set temporary output locations
    model.base_dir = work_dir_base
    model.config = dict(crashdump_dir=crashdump_dir)

    # Possibly execute the workflow
    run_workflow(model, "model", args)

    # Across-Run Registration
    # =======================

    # Set up a variable to control whether stuff is happening on the surface
    surface = args.regspace in ["cortex", "fsaverage"]

    # Short ref to the common space we're using on this execution
    space = args.regspace

    # Retrieve the right workflow function for registration
    # Get the workflow function dynamically based on the space
    workflow_function = getattr(wf, "create_%s_reg_workflow" % space)
    reg, reg_input, reg_output = workflow_function()

    # Define a smooth variable here
    # Can unsmoothed or smoothed in volume, always unsmoothed for surface
    reg_smooth = "unsmoothed" if (
        args.unsmoothed or surface) else "smoothed"
    smooth_source = Node(IdentityInterface(fields=["smooth"]),
                         iterables=("smooth", [reg_smooth]),
                         name="smooth_source")

    # Determine which type of registration is happening
    # (model output or timeseries) and set things accordingly
    timeseries_str = reg_smooth + "_timeseries"
    source_iter = [timeseries_str] if args.timeseries else ["cope", "varcope"]
    source_source = Node(IdentityInterface(fields=["source_image"]),
                         iterables=("source_image", source_iter),
                         name="source_soure")

    # Use the mask as a "contrast" to transform it appropriately
    reg_contrast_iterables = ["_mask"] + exp["contrast_names"]
    # Here we add contrast as an aditional layer of parameterization
    contrast_source = Node(IdentityInterface(fields=["contrast"]),
                           iterables=("contrast", reg_contrast_iterables),
                           name="contrast_source")

    # Build the registration inputs conditional on type of registration
    reg_infields = ["subject_id"]
    aff_template_base = op.join(preproc_dir, "%s/preproc/run_*/func2anat_")

    if args.timeseries:
        base_directory = preproc_dir
        reg_template = "%s/preproc/run_*/%s.%s"
        reg_template_args = {"source_image":
            [["subject_id", "source_image", "nii.gz"]]}
    else:
        base_directory = anal_dir_base
        reg_infields.append("contrast_number")
        reg_template = "%s/model/%s/run_*/%s%d.%s"
        reg_template_args = {"source_image":
            [["subject_id", reg_smooth,
              "source_image", "contrast_number", "nii.gz"]]}

    # There's a bit of a hack to pick up and register the mask correctly
    mask_template = "%s/preproc/run_*/functional_mask.nii.gz"
    mask_template_args = dict(source_image=[["subject_id"]])

    # Add options conditional on space
    aff_key = "%s_affine" % ("tk" if surface else "fsl")
    reg_template_args[aff_key] = [["subject_id"]]
    mask_template_args[aff_key] = [["subject_id"]]
    if surface:
        field_template = {"tk_affine": aff_template_base + "tkreg.dat"}
        reg_infields.append("smooth")
    else:
        field_template = {"fsl_affine": aff_template_base + "flirt.mat"}
        if space == "mni":
            field_template["warpfield"] = op.join(
                project["data_dir"], "%s/normalization/warpfield.nii.gz")
            reg_template_args["warpfield"] = [["subject_id"]]
            mask_template_args["warpfield"] = [["subject_id"]]

    # Same thing for the outputs, but this is only dependant on space
    reg_outfields = dict(
        mni=["source_image", "warpfield", "fsl_affine"],
        epi=["source_image", "fsl_affine"],
        cortex=["source_image", "tk_affine"],
        fsaverage=["source_image", "tk_affine"])[space]

    # Define the registration data source node
    reg_source = Node(DataGrabber(infields=reg_infields,
                                  outfields=reg_outfields,
                                  base_directory=base_directory,
                                  sort_filelist=True),
                      name="reg_source")
    reg_source.inputs.field_template = field_template

    # Registration inutnode
    reg_inwrap = tools.InputWrapper(reg, subj_source,
                                    reg_source, reg_input)
    reg_inwrap.connect_inputs()

    # There are some additional connections to this input
    reg.connect(source_source, "source_image", reg_source, "source_image")
    if not args.timeseries:
        names = exp["contrast_names"]
        reg.connect(
             contrast_source, ("contrast", tools.find_contrast_number, names),
             reg_source, "contrast_number")
        reg.connect(contrast_source, ("contrast", tools.reg_template,
                                      mask_template, reg_template),
                    reg_source, "template")

        reg.connect(contrast_source, ("contrast", tools.reg_template_args,
                                      mask_template_args, reg_template_args),
                    reg_source, "template_args")
    else:
        reg_source.inputs.template = reg_template
        reg_source.inputs.template_args = reg_template_args
    if not surface:
        reg.connect(smooth_source, "smooth", reg_source, "smooth")

    # Reg output and datasink
    reg_sink = Node(DataSink(base_directory=anal_dir_base),
                             name="reg_sink")

    reg_outwrap = tools.OutputWrapper(reg, subj_source,
                                    reg_sink, reg_output)
    reg_outwrap.set_subject_container()
    reg_outwrap.set_mapnode_substitutions(exp["n_runs"])
    reg_outwrap.sink_outputs("reg.%s" % space)

    # Reg has some additional substitutions to strip out interables
    reg_outwrap.add_regexp_substitutions([
        (r"_contrast_[^/]*/", ""),
        (r"_source_image_[^/]*/", ""),
        (r"_space_", ""),
        (r"_smooth_", ""),
        (r"(un)*(smoothed_time)", "time"),
        (r"_run_", "run_")])  # This one's wired to interal function

    reg.base_dir = work_dir_base
    reg.config = dict(crashdump_dir=crashdump_dir)

    # Possibly run registration workflow and clean up
    run_workflow(reg, "reg", args)

    # Cross-Run Fixed Effects Model
    # -----------------------------

    space_source = Node(IdentityInterface(fields=["space"]),
                        iterables=("space", [space]),
                        name="space_source")
    contrast_source.iterables = ("contrast", exp["contrast_names"])

    # Dynamically get the workflow
    manifold = "surface" if surface else "volume"
    workflow_function = getattr(wf, "create_%s_ffx_workflow" % manifold)
    ffx, ffx_input, ffx_output = workflow_function()

    ffx_infields = ["subject_id", "contrast_number", "space"]
    ffx_outfields = ["copes", "varcopes", "dof_files", "masks"]
    if not surface:
        ffx_outfields.append("background_file")

    ffx_template = "%s/reg/%s/%s/run_*/%s%d_*.nii.gz"

    ffx_template_args = dict(
        copes=[["subject_id", "space", "smooth", "cope", "contrast_number"]],
        varcopes=[["subject_id", "space", "smooth",
                   "varcope", "contrast_number"]],
        masks=[["subject_id", "space", "smooth"]],
        dof_files=[["subject_id", "smooth"]])

    ffx_field_template = {"dof_files": "%s/model/%s/run_*/dof",
        "masks": "%s/reg/%s/%s/run_*/functional_mask_*.nii.gz"}
    if space == "epi":
        ffx_field_template["background_file"] = op.join(
            preproc_dir, "%s/preproc//run_1/mean_func.nii.gz")
    elif space == "mni":
        ffx_field_template["background_file"] = op.join(
            project["data_dir"], "%s/normalization/brain_warp.nii.gz")

    if not surface:
        ffx_template_args["background_file"] = [["subject_id"]]

    # Define the ffxistration data source node
    ffx_source = Node(DataGrabber(infields=ffx_infields,
                                  outfields=ffx_outfields,
                                  base_directory=anal_dir_base,
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
    if not surface:
        ffx.connect(smooth_source, "smooth", ffx_source, "smooth")

    # Fixed effects output and datasink
    ffx_sink = Node(DataSink(base_directory=anal_dir_base),
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

    ffx.base_dir = work_dir_base
    ffx.config = dict(crashdump_dir=crashdump_dir)

    # Possibly run fixed effects workflow
    run_workflow(ffx, "ffx", args)

    # Clean-up
    # --------

    if project["rm_working_dir"]:
        shutil.rmtree(op.join(work_dir_base))


def gather_experiment_info(experiment_name, altmodel=None):
    """Import an experiment module and add some formatted information."""
    try:
        if altmodel is not None:
            experiment_name = "%s-%s" % (experiment_name, altmodel)
        exp = __import__("experiments." + experiment_name,
                         fromlist=["experiments"])
    except ImportError:
        print "ERROR: Could not import experiments/%s.py" % experiment_name
        sys.exit()

    # Create an experiment dict stripping the OOP hooks
    exp_dict = dict(
        [(k, v) for k, v in exp.__dict__.items() if not re.match("__.*__", k)])

    # Verify some experiment dict attributes
    verify_experiment_info(exp_dict)

    # Convert HPF cutoff to sigma for fslmaths
    exp_dict["TR"] = float(exp_dict["TR"])
    exp_dict["hpf_cutoff"] = float(exp_dict["hpf_cutoff"])
    exp_dict["hpf_sigma"] = (exp_dict["hpf_cutoff"] / 2.35) / exp_dict["TR"]

    # Setup the hrf_bases dictionary
    exp_dict["hrf_bases"] = {exp_dict["hrf_model"]:
                                {"derivs": exp_dict["hrf_derivs"]}}

    # Build contrasts list
    conkeys = sorted([k for k in exp_dict if re.match("cont\d+", k)])
    exp_dict["contrasts"] = [exp_dict[key] for key in conkeys]
    exp_dict["contrast_names"] = [c[0] for c in exp_dict["contrasts"]]

    return exp_dict


def verify_experiment_info(exp_dict):
    """Catch setup errors that might lead to confusing workflow crashes."""
    if exp_dict["units"] not in ["secs", "scans"]:
        raise ValueError("units must be 'secs' or 'scans'")

    if (exp_dict["slice_time_correction"]
        and exp_dict["slice_order"] not in ["up", "down"]):
        raise ValueError("slice_order must be 'up' or 'down'")


def run_workflow(wf, name, args):
    """Run a workflow, if we asked to do so on the command line."""
    plugin, plugin_args = tools.determine_engine(args)
    if name in args.workflows:
        wf.run(plugin, plugin_args)


def parse_args(arglist):
    """Take an arglist and return an argparse Namespace."""
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
