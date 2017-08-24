from __future__ import division
import os.path as op

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib

from nipype import (Workflow, Node, MapNode, JoinNode,
                    IdentityInterface, DataSink)
from nipype.interfaces.base import traits, TraitedSpec
from nipype.interfaces import fsl, freesurfer as fs

from .. import signals  # TODO confusingly close to scipy.signal
from ..mosaic import Mosaic
from ..graphutils import SimpleInterface


def define_model_fit_workflow(proj_info, subjects, session,
                              exp_info, model_info, qc=True):

    # --- Workflow parameterization and data input

    # We just need two levels of iterables here: one subject-level and
    # one "flat" run-level iterable (i.e. all runs collapsing over
    # sessions). But we want to be able to specify sessions to process.

    scan_info = proj_info.scan_info
    experiment = exp_info.name
    model = model_info.name

    iterables = generate_iterables(scan_info, experiment, subjects, session)
    subject_iterables, run_iterables = iterables

    subject_source = Node(IdentityInterface(["subject"]),
                          name="subject_source",
                          iterables=("subject", subject_iterables))

    run_source = Node(IdentityInterface(["subject", "run"]),
                      name="run_source",
                      itersource=("subject_source", "subject"),
                      iterables=("run", run_iterables))

    data_input = Node(DataInput(experiment=experiment,
                                model=model,
                                analysis_dir=proj_info.analysis_dir),
                      "data_input")

    # --- Data filtering and model fitting

    fit_model = Node(FitModel(data_dir=proj_info.data_dir,
                              exp_info=exp_info,
                              model_info=model_info),
                     "fit_model")

    # --- Data output

    data_output = Node(DataSink(base_directory=proj_info.analysis_dir,
                                parameterization=False),
                       "data_output")

    # === Assemble pipeline

    cache_base = op.join(proj_info.cache_dir, exp_info.name)
    workflow = Workflow(name="model_fit", base_dir=cache_base)

    # Connect processing nodes

    processing_edges = [

        (subject_source, run_source,
            [("subject", "subject")]),
        (subject_source, data_input,
            [("subject", "subject")]),
        (run_source, data_input,
            [("run", "run_tuple")]),

        (data_input, fit_model,
            [("subject", "subject"),
             ("session", "session"),
             ("run", "run"),
             ("seg_file", "seg_file"),
             ("surf_file", "surf_file"),
             ("ts_file", "ts_file"),
             ("noise_file", "noise_file"),
             ("mc_file", "mc_file")]),

        (data_input, data_output,
            [("output_path", "container")]),
        (fit_model, data_output,
            [("beta_file", "@beta"),
             ("ols_file", "@ols"),
             ("sigmasquares_file", "@sigmasquares"),
             ("resid_file", "@resid"),
             ("design_file", "@design")]),

    ]
    workflow.connect(processing_edges)

    qc_edges = [

    ]
    if qc:
        workflow.connect(qc_edges)

    return workflow


# =========================================================================== #
# Custom processing code
# =========================================================================== #


def generate_iterables(scan_info, experiment, subjects, sessions=None):

    subject_iterables = subjects
    run_iterables = {}
    for subject in subjects:
        run_iterables[subject] = []
        for session in scan_info[subject]:
            if sessions is not None and session not in sessions:
                continue
            sess_runs = scan_info[subject][session].get(experiment, [])
            run_tuples = [(subject, session, run) for run in sess_runs]
            run_iterables[subject].extend(run_tuples)

    return subject_iterables, run_iterables


class DataInput(SimpleInterface):

    class input_spec(TraitedSpec):
        experiment = traits.Str()
        model = traits.Str()
        analysis_dir = traits.Directory(exists=True)
        subject = traits.Str()
        run_tuple = traits.Tuple(traits.Str(), traits.Str())

    class output_spec(TraitedSpec):
        subject = traits.Str()
        session = traits.Str()
        run = traits.Str()
        seg_file = traits.File(exists=True)
        surf_file = traits.File(exists=True)
        ts_file = traits.File(exists=True)
        noise_file = traits.File(Exists=True)
        mc_file = traits.File(exists=True)
        output_path = traits.Directory()

    def _run_interface(self, runtime):

        subject = self.inputs.subject
        session, run = self.inputs.run
        experiment = self.inputs.experiment
        model = self.inputs.model_info.name
        anal_dir = self.inputs.analysis_dir

        template_path = op.join(anal_dir, subject, "template")
        timeseries_path = op.join(anal_dir, subject, experiment, "timeseries")

        results = dict(

            seg_file=op.join(template_path, "seg.nii.gz"),
            surf_file=op.join(template_path, "surf.nii.gz"),

            mc_file=op.join(timeseries_path, "mc.csv"),
            ts_file=op.join(timeseries_path, "func.nii.gz"),
            noise_file=op.join(timeseries_path, "noise.nii.gz"),

            output_path=op.join(anal_dir, subject, experiment, model)
        )
        self._results.update(results)

        return runtime


class FitModel(SimpleInterface):

    class input_spec(TraitedSpec):
        subject = traits.Str()
        session = traits.Str()
        run = traits.Str()
        data_dir = traits.Directory(exists=True)
        exp_info = traits.Dict()
        model_info = traits.Dict()
        seg_file = traits.File(exists=True)
        surf_file = traits.File(exists=True)
        ts_file = traits.File(exists=True)
        noise_file = traits.File(exists=True)
        mc_file = traits.File(exists=True)

    class output_spec(TraitedSpec):
        beta_file = traits.File(exists=True)
        ols_file = traits.File(exists=True)
        sigmasquares_file = traits.File(exists=True)
        resid_file = traits.File()  # TODO do we want?
        design_file = traits.File(exists=True)
        design_plot = traits.File(exists=True)

    def _run_interface(self, runtime):

        subject = self.inputs.subject
        exp_info = self.inputs.exp_info
        model_info = self.inputs.model_info
        data_dir = self.inputs.data_dir

        design_file = op.join(data_dir, subject, "design", model_info.name)

        return runtime
