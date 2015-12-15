#! /usr/bin/env python
"""
Main execution script for fMRI analysis in the Lyman ecosystem.

"""
import os

# Needed currently to avoid crash in model code
# Also nipype parallelism doesn't play well with this
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import shutil
import os.path as op
from textwrap import dedent
import argparse

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
    exp = lyman.gather_experiment_info(args.experiment, args.altmodel, args)

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
    nipype.config.set("execution", "crashdump_dir", project["crash_dir"])

    # Create symlinks to the preproc directory for altmodels
    if not op.exists(analysis_dir):
        os.makedirs(analysis_dir)
    if exp_base != exp_name:
        for subj in subject_list:
            subj_dir = op.join(analysis_dir, subj)
            if not op.exists(subj_dir):
                os.makedirs(subj_dir)
            link_dir = op.join(analysis_dir, subj, "preproc")
            if not op.exists(link_dir):
                preproc_dir = op.join("../..", exp_base, subj, "preproc")
                os.symlink(preproc_dir, link_dir)

    # For later processing steps, are we using smoothed inputs?
    smoothing = "unsmoothed" if args.unsmoothed else "smoothed"

    # Also define the regspace variable here
    space = args.regspace

    # ----------------------------------------------------------------------- #
    # Preprocessing Workflow
    # ----------------------------------------------------------------------- #

    # Create workflow in function defined elsewhere in this package
    preproc, preproc_input, preproc_output = wf.create_preprocessing_workflow(
                                                exp_info=exp)

    # Collect raw nifti data
    preproc_templates = dict(timeseries=exp["source_template"])
    if exp["partial_brain"]:
        preproc_templates["whole_brain"] = exp["whole_brain_template"]
    if exp["fieldmap_template"]:
        preproc_templates["fieldmap"] = exp["fieldmap_template"]

    preproc_source = Node(SelectFiles(preproc_templates,
                                      base_directory=project["data_dir"]),
                          "preproc_source")

    # Convenience class to handle some sterotyped connections
    # between run-specific nodes (defined here) and the inputs
    # to the prepackaged workflow returned above
    preproc_inwrap = tools.InputWrapper(preproc, subj_source,
                                        preproc_source, preproc_input)
    preproc_inwrap.connect_inputs()

    # Store workflow outputs to persistant location
    preproc_sink = Node(DataSink(base_directory=analysis_dir), "preproc_sink")

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

    # ----------------------------------------------------------------------- #
    # Timeseries Model
    # ----------------------------------------------------------------------- #

    # Create a modelfitting workflow and specific nodes as above
    model, model_input, model_output = wf.create_timeseries_model_workflow(
        name=smoothing + "_model", exp_info=exp)

    model_base = op.join(analysis_dir, "{subject_id}/preproc/run_*/")
    model_templates = dict(
        timeseries=op.join(model_base, smoothing + "_timeseries.nii.gz"),
        realign_file=op.join(model_base, "realignment_params.csv"),
        nuisance_file=op.join(model_base, "nuisance_variables.csv"),
        artifact_file=op.join(model_base, "artifacts.csv"),
        )

    if exp["design_name"] is not None:
        design_file = exp["design_name"] + ".csv"
        regressor_file = exp["design_name"] + ".csv"
        model_templates["design_file"] = op.join(data_dir, "{subject_id}",
                                                    "design", design_file)
    if exp["regressor_file"] is not None:
        regressor_file = exp["regressor_file"] + ".csv"
        model_templates["regressor_file"] = op.join(data_dir, "{subject_id}",
                                                    "design", regressor_file)

    model_source = Node(SelectFiles(model_templates), "model_source")

    model_inwrap = tools.InputWrapper(model, subj_source,
                                      model_source, model_input)
    model_inwrap.connect_inputs()

    model_sink = Node(DataSink(base_directory=analysis_dir), "model_sink")

    model_outwrap = tools.OutputWrapper(model, subj_source,
                                        model_sink, model_output)
    model_outwrap.set_subject_container()
    model_outwrap.set_mapnode_substitutions(exp["n_runs"])
    model_outwrap.sink_outputs("model." + smoothing)

    # Set temporary output locations
    model.base_dir = working_dir

    # Possibly execute the workflow
    lyman.run_workflow(model, "model", args)

    # ----------------------------------------------------------------------- #
    # Across-Run Registration
    # ----------------------------------------------------------------------- #

    # Is this a model or timeseries registration?
    regtype = "timeseries" if (args.timeseries or args.residual) else "model"

    # Are we registering across experiments?
    cross_exp = args.regexp is not None

    # Retrieve the right workflow function for registration
    # Get the workflow function dynamically based on the space
    warp_method = project["normalization"]
    flow_name = "%s_%s_reg" % (space, regtype)
    reg, reg_input, reg_output = wf.create_reg_workflow(flow_name,
                                                        space,
                                                        regtype,
                                                        warp_method,
                                                        args.residual,
                                                        cross_exp)

    # Define a smoothing info node here. Use an iterable so that running
    # with/without smoothing doesn't clobber working directory files
    # for the other kind of execution
    smooth_source = Node(IdentityInterface(fields=["smoothing"]),
                         iterables=("smoothing", [smoothing]),
                         name="smooth_source")

    # Set up the registration inputs and templates
    reg_templates = dict(
        masks="{subject_id}/preproc/run_*/functional_mask.nii.gz",
        means="{subject_id}/preproc/run_*/mean_func.nii.gz",
                         )

    if regtype == "model":
        # First-level model summary statistic images
        reg_base = "{subject_id}/model/{smoothing}/run_*/"
        reg_templates.update(dict(
            copes=op.join(reg_base, "cope*.nii.gz"),
            varcopes=op.join(reg_base, "varcope*.nii.gz"),
            sumsquares=op.join(reg_base, "ss*.nii.gz"),
                                  ))
    else:
        # Timeseries images
        if args.residual:
            ts_file = op.join("{subject_id}/model/{smoothing}/run_*/",
                              "results/res4d.nii.gz")
        else:
            ts_file = op.join("{subject_id}/preproc/run_*/",
                              "{smoothing}_timeseries.nii.gz")
        reg_templates.update(dict(timeseries=ts_file))
    reg_lists = reg_templates.keys()

    # Native anatomy to group anatomy affine matrix and warpfield
    if space == "mni":
        aff_ext = "mat" if warp_method == "fsl" else "txt"
        reg_templates["warpfield"] = op.join(data_dir, "{subject_id}",
                                             "normalization/warpfield.nii.gz")
        reg_templates["affine"] = op.join(data_dir, "{subject_id}",
                                          "normalization/affine." + aff_ext)
    else:
        if args.regexp is None:
            tkreg_base = analysis_dir
        else:
            tkreg_base = op.join(project["analysis_dir"], args.regexp)
        reg_templates["tkreg_rigid"] = op.join(tkreg_base,
                                               "{subject_id}", "preproc",
                                               "run_1", "func2anat_tkreg.dat")

    # Rigid (6dof) functional-to-anatomical matrices
    rigid_stem = "{subject_id}/preproc/run_*/func2anat_"
    if warp_method == "ants" and space == "mni":
        reg_templates["rigids"] = rigid_stem + "tkreg.dat"
    else:
        reg_templates["rigids"] = rigid_stem + "flirt.mat"

    # Rigid matrix from anatomy to target experiment space
    if args.regexp is not None:
        targ_analysis_dir = op.join(project["analysis_dir"], args.regexp)
        reg_templates["first_rigid"] = op.join(targ_analysis_dir,
                                               "{subject_id}", "preproc",
                                               "run_1", "func2anat_flirt.mat")

    # Define the registration data source node
    reg_source = Node(SelectFiles(reg_templates,
                                  force_lists=reg_lists,
                                  base_directory=analysis_dir),
                      "reg_source")

    # Registration inputnode
    reg_inwrap = tools.InputWrapper(reg, subj_source,
                                    reg_source, reg_input)
    reg_inwrap.connect_inputs()

    # The source node also needs to know about the smoothing on this run
    reg.connect(smooth_source, "smoothing", reg_source, "smoothing")

    # Set up the registration output and datasink
    reg_sink = Node(DataSink(base_directory=analysis_dir), "reg_sink")

    reg_outwrap = tools.OutputWrapper(reg, subj_source,
                                    reg_sink, reg_output)
    reg_outwrap.set_subject_container()
    reg_outwrap.sink_outputs("reg.%s" % space)

    # Reg has some additional substitutions to strip out iterables
    # and rename the timeseries file
    reg_subs = [("_smoothing_", "")]
    reg_outwrap.add_regexp_substitutions(reg_subs)

    # Add dummy substitutions for the contasts to make sure the DataSink
    # reruns when the deisgn has changed. This accounts for the problem where
    # directory inputs are treated as strings and the contents/timestamps are
    # not hashed, which should be fixed upstream soon.
    contrast_subs = [(c, c) for c in exp["contrast_names"]]
    reg_outwrap.add_regexp_substitutions(contrast_subs)

    reg.base_dir = working_dir

    # Possibly run registration workflow and clean up
    lyman.run_workflow(reg, "reg", args)

    # ----------------------------------------------------------------------- #
    # Across-Run Fixed Effects Model
    # ----------------------------------------------------------------------- #

    # Dynamically get the workflow
    wf_name = space + "_ffx"
    ffx, ffx_input, ffx_output = wf.create_ffx_workflow(wf_name,
                                                        space,
                                                        exp["contrast_names"],
                                                        exp_info=exp)

    ext = "_warp.nii.gz" if space == "mni" else "_xfm.nii.gz"
    ffx_base = op.join("{subject_id}/reg", space, "{smoothing}/run_*")
    ffx_templates = dict(
        copes=op.join(ffx_base, "cope*" + ext),
        varcopes=op.join(ffx_base, "varcope*" + ext),
        masks=op.join(ffx_base, "functional_mask" + ext),
        means=op.join(ffx_base, "mean_func" + ext),
        dofs="{subject_id}/model/{smoothing}/run_*/results/dof",
        ss_files=op.join(ffx_base, "ss*" + ext),
        timeseries="{subject_id}/preproc/run_*/{smoothing}_timeseries.nii.gz",
                         )
    ffx_lists = ffx_templates.keys()

    # Space-conditional inputs
    if space == "mni":
        bg = op.join(data_dir, "{subject_id}/normalization/brain_warp.nii.gz")
        reg = op.join(os.environ["FREESURFER_HOME"],
                      "average/mni152.register.dat")
    else:
        reg_dir = "{subject_id}/reg/epi/{smoothing}/run_1"
        bg = op.join(reg_dir, "mean_func_xfm.nii.gz")
        reg = op.join(reg_dir, "func2anat_tkreg.dat")
    ffx_templates["anatomy"] = bg
    ffx_templates["reg_file"] = reg

    # Define the ffxistration data source node
    ffx_source = Node(SelectFiles(ffx_templates,
                                  force_lists=ffx_lists,
                                  base_directory=analysis_dir),
                      "ffx_source")

    # Fixed effects inutnode
    ffx_inwrap = tools.InputWrapper(ffx, subj_source,
                                    ffx_source, ffx_input)
    ffx_inwrap.connect_inputs()

    # Connect the smoothing information
    ffx.connect(smooth_source, "smoothing", ffx_source, "smoothing")

    # Fixed effects output and datasink
    ffx_sink = Node(DataSink(base_directory=analysis_dir), "ffx_sink")

    ffx_outwrap = tools.OutputWrapper(ffx, subj_source,
                                      ffx_sink, ffx_output)
    ffx_outwrap.set_subject_container()
    ffx_outwrap.sink_outputs("ffx.%s" % space)

    # Fixed effects has some additional substitutions to strip out interables
    ffx_outwrap.add_regexp_substitutions([
        ("_smoothing_", ""), ("flamestats", "")
                                          ])

    ffx.base_dir = working_dir

    # Possibly run fixed effects workflow
    lyman.run_workflow(ffx, "ffx", args)

    # -------- #
    # Clean-up
    # -------- #

    if project["rm_working_dir"]:
        shutil.rmtree(project["working_dir"])


def parse_args(arglist):
    """Take an arglist and return an argparse Namespace."""
    help = dedent("""
    Process subject-level data in lyman.

    This script controls the workflows that process data from raw Nifti files
    through a subject-level fixed effects model. It is based on the FSL FEAT
    processing stream and is enhanced with Freesurfer tools for coregistration.
    The other main difference is that the design generation is performed with
    custom code from the `moss.glm` package, although the design matrix
    creation uses the same rules as in FEAT and is expected to give highly
    similar results.

    By using Nipype's parallel machinery, the execution of this script can be
    distributed across a local or managed cluster. The script can thus be run
    for several subjects at once, and (with a large enough cluster) all of the
    subjects can be processed in the time it takes to process a single run of
    data linearly.

    At each stage of the pipeline, a number of static image files are created
    to summarize the results of the processing and facilitate quality
    assurance. These files are stored in the output directories alongside the
    data they correspond with and can be easily browsed using the ziegler
    web-app.

    The processing is organized into four large workflows that save their
    outputs in the analyis_dir hierarchy and can be executed independently.
    The structure of these workflows is represented in detail with the graphs
    that are on the website and in the source distribution. Briefly:

        preproc:

            Preprocess the raw timeseries files by realigning, skull-stripping,
            and filtering. Additionally, artifact detection is performed and
            coregistation to the anatomy is estimated, although the results of
            these stages are not applied to the data until later in the
            pipeline. A smoothed and an unsmoothed version of the final
            timeseries are always written to the analysis_dir.

        model:

            Estimate the timeseries model and generate inferential maps for the
            contrasts of interest. This model is estimated in the native run
            space, and separate models can be estimated for the smoothed and
            unsmoothed versions of the data.

        reg:

            Align the data from each run in a common space. There are two
            options for the target space: `mni`, which uses nonlinear
            normalization to the MNI template (this requires that the
            run_warp.py script has been executed), and `epi`, which transforms
            runs 2-n into the space of the first run. By default this registers
            the summary statistic images from the model, but it is also
            possible to transform the preprocessed timeseries without having
            run the model workflow. Additionally, there is an option to
            transform the unsmoothed version of these data. (The results are
            saved separately for each of these choices, so it is possible to
            use several or all of them). The ROI mask generation script
            (make_masks.py) produces masks in the epi space, so this workflow
            must be run before doing ROI/decoding analyses.

        ffx:

            Estimate the across-run fixed effects model. This model combines
            the summary statistics from each of the runs and produces a single
            set of model results, organized by contrast, for each subject. It
            is possible to fit the ffx model in either the mni or epi space and
            on either smoothed or unsmoothed data. Fixed effects results in the
            mni space can be used for volume-based group analysis, and the
            results in the epi space can be used with the surface-based group
            pipeline. 

    Many details of these workflows can be configured by setting values in the
    experiment file. Additionally, it is possible to preprocess the data for an
    experiment once and then estimate several different models using altmodel
    files.

    If you do not delete your cache directory after running (which is
    configured in the project file), repeated use of this script will only
    rerun the nodes that have changes to their inputs. Otherwise, you will
    have to rerun at the level of the workflows.


    Examples
    --------

    
    Note that the parameter switches match any unique short version
    of the full parameter name.

    run_fmri.py -w preproc model reg ffx

        Run every stage of processing for the default experiment for each
        subject defined in $LYMAN_DIR/subjects.txt. Coregistration will be
        performed for smoothed model outputs in the mni space. The processing
        will be distributed locally with the MultiProc plugin using 4
        processes.

    run_fmri.py -s subj1 subj2 subj3 -w preproc -p sge -q batch.q

        Run preprocessing of the default experiment for subjects `subj1`,
        `subj2`, and `subj3` with distributed execution in the `batch.q` queue
        of the Sun Grid Engine.

    run_fmri.py -s pilot_subjects -w preproc -e nback -n 8

        Preprocess the subjects enumerated in $LYMAN_DIR/pilot_subjects.txt
        with the experiment details in $LYMAN_DIR/nback.py. Distribute the
        execution locally with 8 parallel processes.

    run_fmri.py -s subj1 -w model reg ffx -e nback -a parametric

        Fit the model, register, and combine across runs for subject `subj1`
        with the experiment details defined in $LYMAN_DIR/nback-parametric.py.
        This assumes preprocessing has been performed for the nback experiment.

    run_fmri.py -w preproc reg -t -u -reg epi

        Preprocess the default experiment for all subjects, and then align
        the unsmoothed timeseries into the epi space. This is the standard set
        of processing that must be performed before multivariate analyses.

    run_fmri.py -w reg ffx -reg epi

        Align the summary statistics for all subjects into the epi space and
        then combine across runs. This is the standard processing that must
        be added to use surface-based group analyses.

    run_fmri.py -w preproc model reg ffx -dontrun

        Set up all of the workflows for the default experiment, but do not
        actually submit them for execution. This can be useful for testing
        before starting a large job.

    Usage Details
    -------------

    """)
    parser = tools.parser
    parser.description = help
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.add_argument("-experiment", help="experimental paradigm")
    parser.add_argument("-altmodel", help="alternate model to fit")
    parser.add_argument("-workflows", nargs="*",
                        choices=["preproc", "model", "reg", "ffx"],
                        help="which workflows to run")
    parser.add_argument("-regspace", default="mni", choices=wf.spaces,
                        help="common space for registration and fixed effects")
    parser.add_argument("-regexp",
                        help="perform cross-experiment epi registration")
    parser.add_argument("-timeseries", action="store_true",
                        help="perform registration on preprocessed timeseries")
    parser.add_argument("-residual", action="store_true",
                        help="perform registration on residual timeseries")
    parser.add_argument("-unsmoothed", action="store_true",
                        help="used unsmoothed data for model, reg, and ffx")
    return parser.parse_args(arglist)

if __name__ == "__main__":
    main(sys.argv[1:])
