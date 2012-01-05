#! /usr/bin/env python
"""
Main execution script for fMRI analysis in the Lyman ecosystem.

"""
import os
import re
import sys
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

    project = tools.gather_project_info()
    exp = gather_experiment_info(args.experiment)

    os.environ["SUBJECTS_DIR"] = project["data_dir"]

    sys.path.insert(0, os.path.abspath("."))

    subject_list = tools.determine_subjects(args.subjects)

    subj_source = tools.make_subject_source(subject_list)

    if args.altmodel:
        exp_name = "-".join([args.experiment, args.altmodel])
    else:
        exp_name = args.experiment
    preproc_dir = op.join(project["analysis_dir"], args.experiment)
    anal_dir_base = op.join(project["analysis_dir"], exp_name)
    work_dir_base = op.join(project["working_dir"], exp_name)

    # Preprocessing Workflow
    # ======================

    preproc, preproc_input, preproc_output = wf.create_preprocessing_workflow(
                              do_slice_time_cor=exp["slice_time_correction"],
                              frames_to_toss=exp["frames_to_toss"],
                              interleaved=exp["interleaved"],
                              slice_order=exp["slice_order"],
                              TR=exp["TR"],
                              smooth_fwhm=exp["smooth_fwhm"],
                              highpass_sigma=exp["hpf_sigma"])

    preproc_source = Node(DataGrabber(infields=["subject_id"],
                                      outfields=["timeseries"],
                                      base_directory=project["data_dir"],
                                      template=exp["source_template"],
                                      sort_filelist=True),
                          name="preproc_source")

    preproc_source.inputs.template_args = dict(timeseries=[["subject_id"]])

    preproc_inwrap = tools.InputWrapper(preproc, subj_source,
                                        preproc_source, preproc_input)

    preproc_inwrap.connect_inputs()

    preproc_sink = Node(DataSink(base_directory=anal_dir_base),
                        name="preproc_sink")

    preproc_outwrap = tools.OutputWrapper(preproc, subj_source,
                                          preproc_sink, preproc_output)

    preproc_outwrap.set_subject_container()
    preproc_outwrap.set_mapnode_substitutions(exp["n_runs"])
    preproc_outwrap.sink_outputs("preproc")

    preproc.base_dir = work_dir_base

    preproc.config = dict(crashdump_dir="/tmp")

    run_workflow(preproc, "preproc", args)

    if project["rm_working_dir"]:
        shutil.rmtree(op.join(work_dir_base, "preproc"))

    # Timeseries Model
    # ================

    surface = args.regspace in ["cortex", "fsaverage"]
    model_smooth = "unsmoothed" if surface else "smoothed"

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

    model.base_dir = work_dir_base

    model.config = dict(crashdump_dir="/tmp")

    run_workflow(model, "model", args)

    if project["rm_working_dir"]:
        shutil.rmtree(op.join(work_dir_base, model_smooth + "_model"))

    # Across-Run Registration
    # =======================

    space = args.regspace

    # Retrieve the right workflow function for registration
    workflow_function = getattr(wf, "create_%s_reg_workflow" % space)
    reg, reg_input, reg_output = workflow_function()

    reg_smooth = "unsmoothed" if args.unsmoothed else "smoothed"
    timeseries = reg_smooth + "_timeseries"
    source_iter = [timeseries] if args.timeseries else ["cope", "varcope"]
    source_source = Node(IdentityInterface(fields=["source_image"]),
                         iterables=("source_image", source_iter),
                         name="source_soure")

    contrast_source = Node(IdentityInterface(fields=["contrast"]),
                           iterables=("contrast", exp["contrast_names"]),
                           name="contrast_source")

    reg_infields = ["subject_id", "source_image"]
    if not args.timeseries:
        reg_infields.extend(["source_image", "contrast_number"])

    reg_outfields = dict(
        mni=["source_image", "warpfield", "fsl_affine"],
        epi=["source_image", "fsl_affine"],
        cortex=["source_image", "tk_affine"],
        fsaverage=["source_image", "tk_affine"])[space]

    reg_template = ("%s/preproc/run_*/%s.%s" if args.timeseries
                    else  "%s/model/%s/run_*/%s%d.%s")

    reg_source = Node(DataGrabber(infields=reg_infields,
                                  outfields=reg_outfields,
                                  base_directory=anal_dir_base,
                                  template=reg_template,
                                  sort_filelist=True),
                      name="reg_source")

    reg_source.inputs.field_template = dict(
        fsl_affine=op.join(preproc_dir,
                          "%s/preproc/run_*/func2anat_flirt.mat"))

    template_args = dict()

    if args.timeseries:
        template_args["source_image"] = [
            ["subject_id", "source_image", "nii.gz"]]
    else:
        template_args["source_image"] = [
            ["subject_id", model_smooth,
             "source_image", "contrast_number", "nii.gz"]]

    template_args["fsl_affine"] = [["subject_id"]]

    if space == "mni":
        reg_source.inputs.field_template["warpfield"] = op.join(
            project["data_dir"], "%s/normalization/warpfield.nii.gz")
        template_args["warpfield"] = [["subject_id"]]

    reg_source.inputs.template_args = template_args

    reg_inwrap = tools.InputWrapper(reg, subj_source,
                                    reg_source, reg_input)

    reg_inwrap.connect_inputs()
    reg.connect([(source_source, reg_source, [("source_image", "source_image")])
            ])

    if not args.timeseries:
        reg.connect([
            (contrast_source, reg_source,
                [(("contrast", tools.find_contrast_number, exp["contrast_names"]),
                  "contrast_number")]),
            ])

    reg_sink = Node(DataSink(base_directory=anal_dir_base),
                             name="reg_sink")

    reg_outwrap = tools.OutputWrapper(reg, subj_source,
                                    reg_sink, reg_output)
    reg_outwrap.set_subject_container()
    reg_outwrap.set_mapnode_substitutions(exp["n_runs"])
    reg_outwrap.sink_outputs("reg.%s" % space)
    reg_outwrap.add_regexp_substitutions([
        (r"_contrast_[^/]*/", ""),
        (r"_source_image_[^/]*/", ""),
        (r"_space_", ""),
        (r"_run_", "run_")])

    reg.base_dir = work_dir_base

    run_workflow(reg, "reg", args)


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
                        help="which workflos to run")
    parser.add_argument("-regspace", default="mni",
                        choices=["mni", "epi", "cortex", "fsaverage"],
                        help="common space for registration and fixed effects")
    parser.add_argument("-timeseries", action="store_true",
                        help="perform registration on preprocessed timeseries")
    parser.add_argument("-unsmoothed", action="store_true",
                        help="register unsmoothed timeseries")
    return parser.parse_args(arglist)

if __name__ == "__main__":
    main(sys.argv[1:])
