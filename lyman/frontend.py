"""Forward facing lyman tools with information about ecosystem."""
import os
import os.path as op
import tempfile
import re
import sys
import imp
import shutil
from textwrap import dedent
import yaml

import numpy as np

import nipype
from traits.api import (HasTraits, Str, Bool, Float, Int,
                        Tuple, List, Dict, Enum, Either)

__all__ = ["info", "subjects", "execute"]


class ProjectInfo(HasTraits):
    """General information common to a project."""
    data_dir = Str(
        "../data",
        desc=dedent("""
        The location where raw data is stored. Should be defined relative
        to the ``lyman_dir``.
        """),
    )
    proc_dir = Str(
        "../proc",
        desc=dedent("""
        The location where lyman workflows will output persistent data. Should
        be defined relative to the ``lyman_dir``.
        """),
    )
    cache_dir = Str(
        "../cache",
        desc=dedent("""
        The location where lyman workflows will write intermediate files during
        execution. Should be defined relative to the ``lyman_dir``.
        """),
    )
    remove_cache = Bool(
        True,
        desc=dedent("""
        If True, delete the cache directory containing intermediate files after
        successful execution of the workflow. This behavior can be overridden
        at runtime by command-line arguments.
        """),
    )
    fm_template = Str(
        "{session}_fieldmap_{encoding}.nii.gz",
        desc=dedent("""
        A template string to identify session-specific fieldmap files.
        """),
    )
    ts_template = Str(
        "{session}_{experiment}_{run}.nii.gz",
        desc=dedent("""
        A template string to identify time series data files.
        """),
    )
    sb_template = Str(
        "{session}_{experiment}_{run}_ref.nii.gz",
        desc=dedent("""
        A template string to identify reference volumes corresponding to each
        run of time series data.
        """),
    )
    voxel_size = Tuple(
        Float(2), Float(2), Float(2),
        desc=dedent("""
        The voxel size to use for the functional template.
        """),
    )
    phase_encoding = Enum(
        "pa", "ap",
        desc=dedent("""
        The phase encoding direction used in the functional acquisition.
        """),
    )
    scan_info = Dict(
        Str, Dict(Str, Dict(Str, List(Str))),
        desc=dedent("""
        Information about scanning sessions. (Automatically populated by
        reading the ``scans.yaml`` file).
        """),
    )


class ModelInfo(HasTraits):
    """Model-specific level of information about a specific model."""
    model_name = Str(
        desc=dedent("""
        The name of the model. (Automatically populated from module name).
        """)
    )
    task_model = Bool(
        True,
        desc=dedent("""
        If True, model the task using a design file matching the model name.
        """)
    )
    smooth_fwhm = Either(
        Float(2), None,
        desc=dedent("""
        The size of the Gaussian smoothing kernel for spatial filtering.
        """),
    )
    surface_smoothing = Bool(
        True,
        desc=dedent("""
        If True, filter cortical voxels using Gaussian weights computed along
        the surface mesh.
        """),
    )
    interpolate_noise = Bool(
        False,
        desc=dedent("""
        If True, identify locally noisy voxels and replace replace their values
        using interpolation during spatial filtering. Warning: this option is
        still being refined.
        """),
    )
    hpf_cutoff = Either(
        Float(128), None,
        usedefault=True,
        desc=dedent("""
        The cutoff value (in seconds) for the temporal high-pass filter.
        """),
    )
    percent_change = Bool(
        False,
        desc=dedent("""
        If True, convert data to percent signal change units before model fit.
        """),
    )
    nuisance_components = Dict(
        Enum("wm", "csf", "edge", "noise"), Int,
        usedefault=True,
        desc=dedent("""
        Anatomical sources and number of components per source to include.
        """)
    )
    save_residuals = Bool(
        False,
        desc=dedent("""
        If True, write out an image with the residual time series in each voxel
        after model fitting.
        """),
    )
    hrf_derivative = Bool(
        True,
        desc=dedent("""
        If True, include the temporal derivative of the HRF model.
        """),
    )
    # TODO parameter names to filter the design and generate default contrasts?
    contrasts = List(
        Tuple(Str, List(Str), List(Float)),
        desc=dedent("""
        Definitions for model parameter contrasts. Each item in the list should
        be a tuple with the fields: (1) the name of the contrast, (2) the names
        of the parameters included in the contrast, and (3) the weights to
        apply to the parameters.
        """),
    )


class ExperimentInfo(ModelInfo):
    """More specific experiment-level information."""
    experiment_name = Str(
        desc=dedent("""
        The name of the experiment. (Automatically populated from module name).
        """),
    )
    tr = Float(
        desc=dedent("""
        The temporal resolution of the functional acquisition in seconds.
        """),
    )
    crop_frames = Int(
        0,
        desc=dedent("""
        The number of frames to remove from the beginning of each time series
        during preprocessing.
        """),
    )


class LymanInfo(ProjectInfo, ExperimentInfo):
    """Combination of all information classes."""
    pass


def load_info_from_module(module_name, lyman_dir):
    """Load lyman information from a Python module as specified by name."""
    module_file_name = op.join(lyman_dir, module_name + ".py")
    module_sys_name = "lyman_" + module_name

    # Load the module from memory or disk
    try:
        module = sys.modules[module_sys_name]
    except KeyError:
        module = imp.load_source(module_sys_name, module_file_name)

    # Parse the "normal" variables from the info module
    module_vars = {k: v
                   for k, v in vars(module).items()
                   if not re.match("__.+__", k)}

    return module_vars


def load_scan_info(lyman_dir=None):
    """Load information about subjects, sessions, and runs from scans.yaml."""
    if lyman_dir is None:
        lyman_dir = os.environ.get("LYMAN_DIR", None)

    if lyman_dir is None:
        return {}

    scan_fname = op.join(lyman_dir, "scans.yaml")
    with open(scan_fname) as fid:
        info = yaml.load(fid, Loader=yaml.BaseLoader)

    return info


def check_extra_vars(module_vars, spec):
    """Raise when unexpected information is defined to avoid errors."""
    kind = spec.__name__.lower().strip("info")
    extra_vars = set(module_vars) - set(spec().trait_names())
    if extra_vars:
        msg = ("The following variables were unexpectedly present in the "
               "{} information module: {}".format(kind, ", ".join(extra_vars)))
        raise RuntimeError(msg)


def info(experiment=None, model=None, lyman_dir=None):
    """Load information from various files to control analyses.

    The default behavior (when called with no arguments) is to load project
    level information. Additional information can be loaded by specifying an
    experiment or an experiment and a model.

    Parameters
    ----------
    experiment : string
        Name of the experiment to load information for.  Will include
        information from the file <lyman_dir>/<experiment>.py.
    model : string
        Name of the model to load information for. Requires having also
        specified an experiment. Will include information from the file
        <lyman_dir>/<experiment>-<model>.py.
    lyman_dir : string
        Path to the directory where files with information are stored;
        otherwise read from the $LYMAN_DIR environment variable.

    Returns
    -------
    info : LymanInfo
        This object has traits with various analysis-related parameters.

    """
    if lyman_dir is None:
        lyman_dir = os.environ.get("LYMAN_DIR", None)

    # --- Load project-level information
    if lyman_dir is None:
        project_info = {}
    else:
        project_info = load_info_from_module("project", lyman_dir)
        check_extra_vars(project_info, ProjectInfo)
        project_info["scan_info"] = load_scan_info(lyman_dir)

    # --- Load the experiment-level information
    if experiment is None:
        experiment_info = {}
    else:
        experiment_info = load_info_from_module(experiment, lyman_dir)
        experiment_info["experiment_name"] = experiment
        check_extra_vars(experiment_info, ExperimentInfo)

    # --- Load the model-level information
    if model is None:
        model_info = {}
    else:
        if experiment is None:
            raise RuntimeError("Loading a model requires an experiment")
        model_info = load_info_from_module(experiment + "-" + model, lyman_dir)
        model_info["model_name"] = model
        check_extra_vars(model_info, ModelInfo)

    # TODO set default single parameter contrasts?

    # --- Set the output traits, respecting inheritance
    info = (LymanInfo()
            .trait_set(**project_info)
            .trait_set(**experiment_info)
            .trait_set(**model_info))

    # Ensure that directories are specified as real absolute paths
    if lyman_dir is None:
        base = op.join(tempfile.mkdtemp(), "lyman")
    else:
        base = lyman_dir
    directories = ["data_dir", "proc_dir", "cache_dir"]
    orig = info.trait_get(directories)
    full = {k: op.abspath(op.join(base, v)) for k, v in orig.items()}
    for d in full.values():
        if not op.exists(d):
            os.mkdir(d)

    info.trait_set(**full)

    return info


def subjects(subject_arg=None, sessions=None, lyman_dir=None):
    """Find a list of subjects in a variety of ways.

    Parameters
    ----------
    subject_arg : list or string
        This argument can take several forms:
           - None, in which case all subject ids in scans.yaml are used.
           - A list of subject ids or single subject id which will be used.
           - The name (without extension) of a text file in the <lyman_dir>
             containing list of subject ids, or a list with a single entry
             corresponding to the name of a file.
           - A single subject id, which will be used.
    sessions : list
        A list of session ids. Only valid when there is a single subject
        in the returned list. This parameter can be used to validate that
        the requested sessions exist for the requested subject.
    lyman_dir : string
        Path to the directory where files with information are stored;
        otherwise read from the $LYMAN_DIR environment variable.

    Returns
    -------
    subjects : list of strings
        A list of subject ids.

    """
    scan_info = load_scan_info(lyman_dir)

    if lyman_dir is None:
        lyman_dir = os.environ.get("LYMAN_DIR", None)

    if lyman_dir is None:
        return []

    # -- Parse the input

    if isinstance(subject_arg, list) and len(subject_arg) == 1:
        subject_arg = subject_arg[0]

    string_arg = isinstance(subject_arg, str)

    if subject_arg is None:
        subjects = list(scan_info)
    elif string_arg:
        subject_path = op.join(lyman_dir, subject_arg + ".txt")
        if op.isfile(subject_path):
            subjects = np.loadtxt(subject_path, str, ndmin=1).tolist()
        else:
            subjects = [subject_arg]
    else:
        subjects = subject_arg

    # -- Check the input

    unexepected_subjects = set(subjects) - set(scan_info)
    if unexepected_subjects:
        msg = "Specified subjects were not in scans.yaml: {}"
        raise RuntimeError(unexepected_subjects)

    if sessions is not None:
        try:
            subject, = subjects
        except ValueError:
            raise RuntimeError("Can only specify session for single subject")

        unexpected_sessions = set(sessions) - set(scan_info[subject])
        if unexpected_sessions:
            msg = "Specified sessions were not in scans.yaml for {}: {}"
            raise RuntimeError(msg.format(subject, unexpected_sessions))

    return subjects


def execute(wf, args, info):
    """Main interface for (probably) executing a nipype workflow.

    Depending on the command-line and module-based parameters, different things
    might happen with the workflow object. See the code for details.

    Parameters
    ----------
    wf : Workflow
        Nipype workflow graph with processing steps.
    args : argparse Namespace
        Parsed arguments from lyman command-line interface.
    info : LymanInfo
        Analysis execution parameters from lyman info files.

    Returns
    -------
    res : variable
        The result of the execution, which can take several forms. See the
        code to understand the relationship between input parameters and
        output type.

    """
    # Set a location for the workflow to save debugging files on a crash
    crash_dir = op.join(info.cache_dir, "crashdumps")
    wf.config["execution"]["crashdump_dir"] = crash_dir

    # Set various nipype config options if debugging
    if args.debug:
        nipype.config.enable_debug_mode()

    # Locate the directory where intermediate processing outputs will be stored
    # and optionally remove it to force a full clean re-run of the workflow.
    cache_dir = op.join(wf.base_dir, wf.name)
    if args.clear_cache:
        if op.exists(cache_dir):
            shutil.rmtree(cache_dir)

    # One option is to do nothing (allowing a check from the command-line that
    # everything is specified properly),
    if not args.execute:
        res = None

    # Another option is to generate an svg image of the workflow graph
    elif args.graph:
        if args.graph is True:
            fname = args.stage
        else:
            fname = args.graph
        res = wf.write_graph(fname, "orig", "svg")

    # Alternatively, submit the workflow to the nipype execution engine
    else:

        # TODO expose other nipype plugins as a command-line parameter
        if args.n_procs == 1:
            plugin, plugin_args = "Linear", {}
        else:
            plugin, plugin_args = "MultiProc", {"n_procs": args.n_procs}
        res = wf.run(plugin, plugin_args)

    # After successful completion of the workflow, optionally delete the
    # intermediate files, which are not usually needed aside from debugging
    # (persistent outputs go into the `info.proc_dir`).
    if info.remove_cache and not args.debug and op.exists(cache_dir):
        shutil.rmtree(cache_dir)

    return res
