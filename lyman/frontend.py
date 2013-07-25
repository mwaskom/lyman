"""Forward facing lyman tools with information about ecosystem."""
import os
import re
import sys
import imp
import os.path as op

import numpy as np


def gather_project_info():
    """Import project information based on environment settings."""
    proj_file = op.join(os.environ["LYMAN_DIR"], "project.py")
    try:
        project = sys.modules["project"]
    except KeyError:
        project = imp.load_source("project", proj_file)
    return dict(
        [(k, v) for k, v in project.__dict__.items()
            if not re.match("__.*__", k)])


def gather_experiment_info(exp_name=None, altmodel=None):
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

    keep = lambda k: not re.match("__.*__", k)
    exp_dict = {k: v for k, v in exp.__dict__.items() if keep(k)}

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

    # Verify some experiment dict attributes
    verify_experiment_info(exp_dict)

    # Save the __doc__ attribute to the dict
    exp_dict["comments"] = exp.__doc__
    if altmodel is not None:
        exp_dict["comments"] += alt.__doc__

    # Check if it looks like this is a partial FOV acquisition
    exp_dict["partial_brain"] = bool(exp_dict.get("whole_brain_epi"))

    # Temporal resolution. Mandatory.
    exp_dict["TR"] = float(exp_dict["TR"])

    # Convert HPF cutoff to sigma for fslmaths
    exp_dict["hpf_cutoff"] = float(exp_dict.get("hpf_cutoff", np.inf))
    exp_dict["hpf_sigma"] = (exp_dict["hpf_cutoff"] / 2.35) / exp_dict["TR"]

    # Setup the hrf_bases dictionary
    exp_dict["hrf_bases"] = {exp_dict.get("hrf_model"):
                             {"derivs": exp_dict.get("hrf_derivs")}}

    # Build contrasts list if neccesary
    if "contrasts" not in exp_dict:
        conkeys = sorted([k for k in exp_dict if re.match("cont\d+", k)])
        exp_dict["contrasts"] = [exp_dict[key] for key in conkeys]
    exp_dict["contrast_names"] = [c[0] for c in exp_dict["contrasts"]]

    if "regressors" not in exp_dict:
        exp_dict["regressors"] = []

    return exp_dict


def verify_experiment_info(exp_dict):
    """Catch setup errors that might lead to confusing workflow crashes."""
    if exp_dict["units"] not in ["secs", "scans"]:
        raise ValueError("units must be 'secs' or 'scans'")

    if (exp_dict["temporal_interp"]
            and exp_dict["slice_order"] not in ["up", "down"]):
        raise ValueError("slice_order must be 'up' or 'down'")


def determine_subjects(subject_arg=None):
    """Intelligently find a list of subjects in a variety of ways."""
    if subject_arg is None:
        subject_file = op.join(os.environ["LYMAN_DIR"], "subjects.txt")
        subjects = np.loadtxt(subject_file, str).tolist()
    elif op.isfile(subject_arg[0]):
        subjects = np.loadtxt(subject_arg[0], str).tolist()
    else:
        subjects = subject_arg
    return subjects


def determine_engine(args):
    """Read command line args and return Workflow.run() args."""
    plugin_dict = dict(linear="Linear", multiproc="MultiProc",
                       ipython="IPython", torque="PBS", sge="SGE")

    plugin = plugin_dict[args.plugin]

    plugin_args = dict()
    qsub_args = ""

    if plugin == "MultiProc":
        plugin_args['n_procs'] = args.nprocs
    elif plugin in ["SGE", "PBS"]:
        qsub_args += "-V -e /dev/null -o /dev/null "

    if args.queue is not None:
        qsub_args += "-q %s " % args.queue

    plugin_args["qsub_args"] = qsub_args

    return plugin, plugin_args


def run_workflow(wf, name=None, args=None):
    """Run a workflow, if we asked to do so on the command line."""
    plugin, plugin_args = determine_engine(args)
    if name is None or name in args.workflows:
        wf.run(plugin, plugin_args)
