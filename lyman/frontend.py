"""Forward facing lyman tools with information about ecosystem."""
import os
import os.path as op
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

from .workflows.template import define_template_workflow
from .workflows.preproc import define_preproc_workflow
from .workflows.model import (define_model_fit_workflow,
                              define_model_results_workflow)


__all__ = []


class ProjectInfo(HasTraits):

    data_dir = Str(
        "../data",
        desc=dedent("""
        A relative path to the directory where raw data is stored.
        """),
    )
    proc_dir = Str(
        "../proc",
        desc=dedent("""
        A relative path to the directory where lyman workflows will output
        persistent data.
        """),
    )
    cache_dir = Str(
        "../cache",
        desc=dedent("""
        A relative path to the directory where lyman workflows will write
        intermediate files during execution.
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
    phase_encoding = Enum(
        "pa", "ap",
        desc=dedent("""
        The phase encoding direction used in the functional acquisition.
        """),
    )
    scan_info = Dict(
        Str, Dict(Str, Dict(Str, List(Str))),
        desc=dedent("""
        Information about scanning sessions, populted by reading the
        ``scan_info.yaml`` file.
        """),
    )


class ExperimentInfo(HasTraits):

    experiment_name = Str(
        desc="The name of the experiment."
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


class ModelInfo(ExperimentInfo):

    model_name = Str(
        desc="The name of the model."
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
        True,
        desc=dedent("""
        If True, identify locally noisy voxels and replace replace their values
        using interpolation during spatial filtering.
        """),
    )
    hpf_cutoff = Either(
        Float(128), None,
        usedefault=True,
        desc=dedent("""
        The cutoff value (in seconds) for the temporal high-pass filter.
        """),
    )
    save_residuals = Bool(
        False,
        desc=dedent("""
        If True, write out an image with the residual time series in each voxel
        after model fitting.
        """),
    )
    # TODO HRF model and params
    # TODO model confounds and artifact-related params
    contrasts = List(
        Tuple(Str, List(Str), List(Float)),
        desc=dedent("""
        Definitions for model parameter contrasts. Each item in the list should
        be a tuple with the fields: (1) the name of the contrast, (2) the names
        of the parameters included in the contrast, and (3) the weights to
        apply to the parameters.
        """),
    )


class LymanInfo(ProjectInfo, ModelInfo):

    pass


def load_info_from_module(module_name, lyman_dir):
    """Load lyman information from a Python module."""
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


def check_extra_vars(module_vars, spec):
    """Raise when unexpected information is defined to avoid errors."""
    kind = spec.__name__.lower().strip("info")
    extra_vars = set(module_vars) - set(spec.trait_names())
    if extra_vars:
        msg = ("The following variables were unexpectedly present in the "
               "{} information module: {}".format(kind, extra_vars))
        raise RuntimeError(msg)


def lyman_info(experiment=None, model=None, lyman_dir=None):
    """Load information from various modules."""
    # TODO best name for this?
    if lyman_dir is None:
        lyman_dir = os.environ["LYMAN_DIR"]

    # Load project-level information
    project_info = load_info_from_module("project", lyman_dir)
    check_extra_vars(project_info, ProjectInfo)

    # Load scan information
    # TODO load from yaml file

    # Load the experiment-level information
    if experiment is None:
        experiment_info = {}
    else:
        experiment_info = load_info_from_module(experiment, lyman_dir)
        experiment_info["experiment_name"] = experiment
        check_extra_vars(experiment_info, ExperimentInfo)

    # Load the model-level information
    if model is None:
        model_info = {}
    else:
        model_info = load_info_from_module(experiment + "-" + model, lyman_dir)
        model_info["model_name"] = model
        check_extra_vars(model_info, ModelInfo)

    # Set the output traits in descending order of granularity
    info = (LymanInfo()
            .trait_set(project_info)
            .trait_set(experiment_info)
            .trait_set(model_info))

    return info


def determine_subjects(subject_arg=None):
    """Intelligently find a list of subjects in a variety of ways."""
    # TODO best name for this?
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


def execute(wf, args, info):
    """Execute a workflow from command line arguments."""

    crash_dir = op.join(info.cache_dir, "crashdumps")
    wf.config["execution"]["crashdump_dir"] = crash_dir

    if args.debug:
        nipype.config.enable_debug_mode()

    cache_dir = op.join(wf.base_dir, wf.name)
    if args.clear_cache:
        if op.exists(cache_dir):
            shutil.rmtree(cache_dir)

    if args.graph:
        if args.graph is True:
            fname = args.stage
        else:
            fname = args.graph
        wf.write_graph(fname, "orig", "svg")

    else:
        if args.execute:
            plugin = "MultiProc"
            plugin_args = dict(n_procs=args.n_procs)
            wf.run(plugin, plugin_args)

    if not args.debug:
        shutil.rmtree(cache_dir)
