"""Forward facing lyman tools with information about ecosystem."""
import os
import re
import sys
import imp
import os.path as op
import yaml

import numpy as np

# TODO put this locally, not in moss
from moss import Bunch

# TODO mayne defer imports?
from .workflows.template import define_template_workflow
from .workflows.preproc import define_preproc_workflow
from .workflows.model import (define_model_fit_workflow,
                              define_model_results_workflow)


__all__ = []


def gather_project_info(lyman_dir=None):
    """Import project information based on environment settings."""
    if lyman_dir is None:
        lyman_dir = os.environ["LYMAN_DIR"]
    proj_file = op.join(lyman_dir, "project.py")
    try:
        project = sys.modules["project"]
    except KeyError:
        project = imp.load_source("project", proj_file)

    project_dict = dict()
    for dir in ["data", "analysis", "cache"]:
        path = op.abspath(op.join(lyman_dir, getattr(project, dir + "_dir")))
        project_dict[dir + "_dir"] = path

    for scan in ["fm", "ts", "sb"]:
        project_dict[scan + "_template"] = getattr(project, scan + "_template")

    scan_fname = op.join(lyman_dir, "scan_info.yaml")
    with open(scan_fname) as fid:
        scan_info = yaml.load(fid)
    project_dict["scan_info"] = scan_info

    project_dict["phase_encoding"] = getattr(project, "phase_encoding")

    return Bunch(project_dict)


def gather_experiment_info(exp_name=None, altmodel=None, args=None):
    """Import an experiment module and add some formatted information."""
    lyman_dir = os.environ["LYMAN_DIR"]

    # Allow easy use of default experiment
    if exp_name is None:
        project = gather_project_info()
        exp_name = project["default_exp"]

    # Import the base experiment
    try:
        exp = sys.modules[exp_name]
    except KeyError:
        exp_file = op.join(lyman_dir, exp_name + ".py")
        exp = imp.load_source(exp_name, exp_file)

    exp_dict = default_experiment_parameters()

    def keep(k):
        return not re.match("__.*__", k)

    exp_dict.update({k: v for k, v in exp.__dict__.items() if keep(k)})

    # Possibly import the alternate model details
    if altmodel is not None:
        try:
            alt = sys.modules[altmodel]
        except KeyError:
            alt_file = op.join(lyman_dir, "%s-%s.py" % (exp_name, altmodel))
            alt = imp.load_source(altmodel, alt_file)

        alt_dict = {k: v for k, v in alt.__dict__.items() if keep(k)}

        # Update the base information with the altmodel info
        exp_dict.update(alt_dict)

    # Save the __doc__ attribute to the dict
    exp_dict["comments"] = "" if exp.__doc__ is None else exp.__doc__
    if altmodel is not None:
        exp_dict["comments"] += "\n"
        exp_dict["comments"] += "" if alt.__doc__ is None else alt.__doc__

    # Check if it looks like this is a partial FOV acquisition
    exp_dict["partial_brain"] = bool(exp_dict.get("whole_brain_template"))

    # Temporal resolution. Mandatory.
    exp_dict["TR"] = float(exp_dict["TR"])

    # Set up the default contrasts
    if exp_dict["condition_names"] is not None:
        cs = [(name, [name], [1]) for name in exp_dict["condition_names"]]
        exp_dict["contrasts"] = cs + exp_dict["contrasts"]

    # Build contrasts list if neccesary
    exp_dict["contrast_names"] = [c[0] for c in exp_dict["contrasts"]]

    # Add command line arguments for reproducibility
    if args is not None:
        exp_dict["command_line"] = vars(args)

    exp_dict["name"] = exp_name

    return Bunch(exp_dict)


def gather_model_info(experiment, model):

    lyman_dir = os.environ["LYMAN_DIR"]

    model_file = op.join(lyman_dir, "{}-{}.py".format(experiment, model))
    if not op.exists(model_file):
        model_file = op.join(lyman_dir, "{}.py".format(model))

    module_name = "lyman_model_{}".format(model)
    info = imp.load_source(module_name, model_file)

    # TODO hacked to get going
    fields = ["smooth_fwhm", "hpf_cutoff", "interpolate_noise", "contrasts"]
    info_dict = {k: getattr(info, k) for k in fields}
    info_dict["name"] = model
    return Bunch(info_dict)


def default_experiment_parameters():
    """Return default values for experiments."""
    exp = dict(

        source_template="",
        whole_brain_template="",
        fieldmap_template="",
        n_runs=0,

        TR=2,
        frames_to_toss=0,
        fieldmap_pe=("y", "y-"),
        temporal_interp=False,
        interleaved=True,
        coreg_init="fsl",
        slice_order="up",
        intensity_threshold=4.5,
        motion_threshold=1,
        spike_threshold=None,
        wm_components=6,
        smooth_fwhm=6,
        hpf_cutoff=128,

        design_name=None,
        condition_names=None,
        regressor_file=None,
        regressor_names=None,
        confound_sources=["motion"],
        remove_artifacts=True,
        hrf_model="GammaDifferenceHRF",
        temporal_deriv=False,
        confound_pca=False,
        hrf_params={},
        contrasts=[],
        memory_request=5,

        flame_mode="flame1",
        cluster_zthresh=2.3,
        grf_pthresh=0.05,
        peak_distance=30,
        surf_name="inflated",
        surf_smooth=5,
        sampling_units="frac",
        sampling_method="average",
        sampling_range=(0, 1, .1),
        surf_corr_sign="pos",

        )

    return exp


def determine_subjects(subject_arg=None):
    """Intelligently find a list of subjects in a variety of ways."""
    if subject_arg is None:
        subject_file = op.join(os.environ["LYMAN_DIR"], "subjects.txt")
        subjects = np.loadtxt(subject_file, str).tolist()
    elif op.isfile(subject_arg[0]):
        subjects = np.loadtxt(subject_arg[0], str).tolist()
    else:
        try:
            subject_file = op.join(os.environ["LYMAN_DIR"],
                                   subject_arg[0] + ".txt")
            subjects = np.loadtxt(subject_file, str).tolist()
        except IOError:
            subjects = subject_arg
    return subjects


def determine_engine(args):
    """Read command line args and return Workflow.run() args."""
    plugin_dict = dict(linear="Linear", multiproc="MultiProc",
                       ipython="IPython", torque="PBS", sge="SGE",
                       slurm="SLURM")

    plugin = plugin_dict[args.plugin]

    plugin_args = dict()

    if plugin == "MultiProc":
        plugin_args['n_procs'] = args.nprocs

    elif plugin in ["SGE", "PBS"]:

        qsub_args = "-V -e /dev/null -o /dev/null "

        if args.queue is not None:
            qsub_args += "-q %s " % args.queue

        plugin_args["qsub_args"] = qsub_args

    return plugin, plugin_args


def run_workflow(wf, args=None):
    """Run a workflow, if we asked to do so on the command line."""
    if args.graph:
        wf.write_graph(args.stage, "orig", "svg")
    else:
        plugin, plugin_args = determine_engine(args)
        wf.run(plugin, plugin_args)


def execute_workflow(args):

    # TODO maybe this code should just be in the lyman script

    stage = args.stage

    proj_info = gather_project_info()

    # TODO either both or neither of subject(s)/session(s)
    # should be plural at this point

    subjects = determine_subjects(args.subject)
    qc = args.qc

    # TODO Oof this logic needs to be reworked
    if stage == "template":
        wf = define_template_workflow(proj_info, subjects, qc)
    if stage == "preproc":
        exp_info = gather_experiment_info(args.experiment)
        session = args.session
        if len(subjects) > 1 and session is not None:
            raise RuntimeError("Can only specify session for single subject")
        wf = define_preproc_workflow(proj_info, subjects, session,
                                     exp_info, qc)
    if stage in ["model", "model-fit"]:
        exp_info = gather_experiment_info(args.experiment)
        model_info = gather_model_info(args.experiment, args.model)
        session = args.session
        if len(subjects) > 1 and session is not None:
            raise RuntimeError("Can only specify session for single subject")
        wf = define_model_fit_workflow(proj_info, subjects, session,
                                       exp_info, model_info, qc)
    if stage in ["model", "model-res"]:
        exp_info = gather_experiment_info(args.experiment)
        model_info = gather_model_info(args.experiment, args.model)
        session = args.session
        if len(subjects) > 1 and session is not None:
            raise RuntimeError("Can only specify session for single subject")
        wf = define_model_results_workflow(proj_info, subjects, session,
                                           exp_info, model_info, qc)

    crash_dir = op.join(proj_info.cache_dir, "crashdumps")
    wf.config["execution"]["crashdump_dir"] = crash_dir

    run_workflow(wf, args)
