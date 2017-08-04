import os
import os.path as op

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from nipype import (Workflow, Node, MapNode, JoinNode,
                    Function, IdentityInterface, SelectFiles, DataSink)
from nipype.interfaces.base import (traits, File, TraitedSpec,
                                    InputMultiPath, OutputMultiPath)
from nipype.interfaces import fsl, freesurfer as fs, utility as pipeutil

from moss.mosaic import Mosaic  # TODO move into lyman
from ..graphutils import SimpleInterface


def define_preproc_workflow(proj_info, sess_info, exp_info):

    # proj_info will be a bunch or other object with data_dir, etc. fields

    # sess info is a nested dictionary:
    # outer keys are subjects
    # inner keys are sessions
    # inner values are lists of runs

    # exp_info is a bunch or dict or other obj with experiment parameters

    # --- Workflow parameterization

    subject_iterables = list(sess_info.keys())
    subject_source = Node(IdentityInterface(["subject"]),
                          name="subject_source",
                          iterables=("subject", subject_iterables))

    session_iterables = {
        subj: [(subj, sess) for sess in sess_info[subj]]
        for subj in sess_info
    }
    session_source = Node(IdentityInterface(["subject", "session"]),
                          name="session_source",
                          itersource=("subject_source", "subject"),
                          iterables=("session", session_iterables))

    run_iterables = {
        (subj, sess): [(subj, sess, run) for run in sess_info[subj][sess]]
        for subj, subj_info in sess_info.items()
        for sess in subj_info
    }
    run_source = Node(IdentityInterface(["subject", "session", "run"]),
                      name="run_source",
                      itersource=("session_source", "session"),
                      iterables=("run", run_iterables))

    # --- Semantic information

    def info_func(info_tuple):
        try:
            subject, session = info_tuple
            return subject, session
        except ValueError:
            subject, session, run = info_tuple
            return subject, session, run

    sesswise_info = Node(Function("info_tuple",
                                  ["subject", "session"],
                                  info_func),
                         "sesswise_info")

    runwise_info = Node(Function("info_tuple",
                                 ["subject", "session", "run"],
                                 info_func),
                        "runwise_info")

    # --- Input file selection

    session_templates = dict(se=exp_info.se_template)
    sesswise_input = Node(SelectFiles(session_templates,
                                      base_directory=proj_info.data_dir),
                          "sesswise_input")

    run_templates = dict(ts=exp_info.ts_template, sb=exp_info.sb_template)
    runwise_input = Node(SelectFiles(run_templates,
                                     base_directory=proj_info.data_dir),
                         "runwise_input")

    # --- Reorientation of functional data

    reorient_ts = Node(fsl.Reorient2Std(), "reorient_ts")
    reorient_se = reorient_ts.clone("reorient_se")
    reorient_sb = reorient_ts.clone("reorient_sb")

    # --- Warpfield estimation using topup

    # Distortion warpfield estimation
    phase_encoding = ["y", "y", "y", "y-", "y-", "y-"]
    readout_times = [1, 1, 1, 1, 1, 1]
    estimate_distortions = Node(fsl.TOPUP(encoding_direction=phase_encoding,
                                          readout_times=readout_times,
                                          config="b02b0.cnf"),
                                "estimate_distortions")

    # Average distortion-corrected spin-echo images
    average_se = Node(fsl.MeanImage(out_file="se_restored.nii.gz"),
                      "average_se")

    # Select first warpfield image from output list
    select_warp = Node(pipeutil.Select(index=[0]), "select_warp")

    # Define a mask of areas with large distortions
    thresh_ops = "-abs -thr 4 -Tmax -binv"
    mask_distortions = Node(fsl.ImageMaths(op_string=thresh_ops),
                            "mask_distortions")

    # --- Registration of SBRef to SE-EPI (with distortions)

    sb2se = Node(fsl.FLIRT(dof=6), "sb2se")

    # --- Registration of SE-EPI (without distortions) to Freesurfer anatomy

    se2anat = Node(fs.BBRegister(init="fsl",
                                 contrast_type="t2",
                                 out_fsl_file="se2anat_flirt.mat",
                                 out_reg_file="se2anat_tkreg.dat"),
                   "se2anat")

    # --- Definition of common cross-session space (template space)

    se2template = JoinNode(TemplateTransform(),
                           name="se2template",
                           joinsource="session_source",
                           joinfield=["session_info",
                                      "in_matrices", "in_volumes"])

    # --- Associate template-space transforms with data from correct session

    def select_transform_func(session_info, subject, session, in_matrices):

        for info, matrix in zip(session_info, in_matrices):
            if info == (subject, session):
                out_matrix = matrix
        return out_matrix

    select_sesswise = Node(Function(["session_info",
                                     "subject", "session",
                                     "in_matrices"],
                                    "out_matrix",
                                    select_transform_func),
                           "select_sesswise")

    select_runwise = select_sesswise.clone("select_runwise")

    # --- Restore each sessions SE image in template space then average

    split_se = Node(fsl.Split(dimension="t"), "split_se")

    restore_se = MapNode(fsl.ApplyWarp(interp="spline", relwarp=True),
                         ["in_file", "premat", "field_file"],
                         "restore_se")

    def flatten_file_list(in_files):
        out_files = [item for sublist in in_files for item in sublist]
        return out_files

    combine_template = JoinNode(Function("in_files", "out_files",
                                         flatten_file_list),
                                name="combine_template",
                                joinsource="session_source",
                                joinfield=["in_files"])

    merge_template = Node(fsl.Merge(dimension="t"), name="merge_template")

    average_template = Node(fsl.MeanImage(out_file="func.nii.gz"),
                            "average_template")

    # --- Motion correction of timeseries to SBRef (with distortions)

    ts2sb = Node(fsl.MCFLIRT(save_mats=True, save_plots=True),
                 "ts2sb")

    realign_qc = Node(RealignmentReport(), "realign_qc")

    # --- Combined motion correction and unwarping of timeseries

    # Split the timeseries into each frame
    split_ts = Node(fsl.Split(dimension="t"), "split_ts")

    # Concatenation ts2sb and sb2se rigid transform
    combine_rigids = MapNode(fsl.ConvertXFM(concat_xfm=True),
                             "in_file", "combine_rigids")

    # Simultaneously apply rigid transform and nonlinear warpfield
    restore_ts_frames = MapNode(fsl.ApplyWarp(interp="spline", relwarp=True),
                                ["in_file", "premat"],
                                "restore_ts")

    # Recombine the timeseries frames into a 4D image
    merge_ts = Node(fsl.Merge(merged_file="timeseries.nii.gz",
                              dimension="t"), "merge_ts")

    # Take a temporal average of the timeseries
    average_ts = Node(fsl.MeanImage(out_file="mean.nii.gz"),
                      "average_ts")

    # --- Workflow ouptut

    output_dir = op.join(proj_info.analysis_dir, exp_info.name)

    runwise_substitutions = []
    subjwise_substitutions = []
    for subject, subj_info in sess_info.items():
        subjwise_substitutions.append(
          ("_subject_{}".format(subject), "{}/preproc".format(subject))
        )
        for session, session_runs in subj_info.items():
            for run in session_runs:
                kws = dict(subject=subject, session=session, run=run)
                runwise_substitutions.append(
                    (("_subject_{subject}/"
                      "_session_{subject}.{session}/"
                      "_run_{subject}.{session}.{run}").format(**kws),
                     "{subject}/preproc/{session}_{run}".format(**kws))
                )

    runwise_output = Node(DataSink(base_directory=output_dir,
                                   substitutions=runwise_substitutions),
                          "runwise_output")

    subjwise_output = Node(DataSink(base_directory=output_dir,
                                    substitutions=subjwise_substitutions),
                           "subjwise_output")

    # === Assemble pipeline

    cache_base = op.join(proj_info.cache_dir, exp_info.name)
    workflow = Workflow(name="preproc", base_dir=cache_base)

    workflow.connect([
        (subject_source, session_source,
            [("subject", "subject")]),
        (subject_source, run_source,
            [("subject", "subject")]),
        (session_source, run_source,
            [("session", "session")]),
        (session_source, sesswise_info,
            [("session", "info_tuple")]),
        (run_source, runwise_info,
            [("run", "info_tuple")]),
        (sesswise_info, sesswise_input,
            [("subject", "subject"),
             ("session", "session")]),
        (runwise_info, runwise_input,
            [("subject", "subject"),
             ("session", "session"),
             ("run", "run")]),
        (sesswise_input, reorient_se,
            [("se", "in_file")]),
        (reorient_se, estimate_distortions,
            [("out_file", "in_file")]),
        (estimate_distortions, select_warp,
            [("out_warps", "inlist")]),
        (select_warp, mask_distortions,
            [("out", "in_file")]),
        (estimate_distortions, average_se,
            [("out_corrected", "in_file")]),
        (sesswise_info, se2anat,
            [("subject", "subject_id")]),
        (average_se, se2anat,
            [("out_file", "source_file")]),
        (session_source, se2template,
            [("session", "session_info")]),
        (reorient_se, se2template,
            [("out_file", "in_volumes")]),
        (se2anat, se2template,
            [("out_fsl_file", "in_matrices")]),
        (se2template, select_sesswise,
            [("out_matrices", "in_matrices"),
             ("out_template", "in_templates"),
             ("session_info", "session_info")]),
        (sesswise_info, select_sesswise,
            [("subject", "subject"),
             ("session", "session")]),
        (reorient_se, split_se,
            [("out_file", "in_file")]),
        (split_se, restore_se,
            [("out_files", "in_file")]),
        (estimate_distortions, restore_se,
            [("out_mats", "premat"),
             ("out_warps", "field_file")]),
        (se2template, restore_se,
            [("out_template", "ref_file")]),
        (select_sesswise, restore_se,
            [("out_matrix", "postmat")]),
        (restore_se, combine_template,
            [("out_file", "in_files")]),
        (combine_template, merge_template,
            [("out_files", "in_files")]),
        (merge_template, average_template,
            [("merged_file", "in_file")]),
        (runwise_input, reorient_ts,
            [("ts", "in_file")]),
        (runwise_input, reorient_sb,
            [("sb", "in_file")]),
        (reorient_ts, ts2sb,
            [("out_file", "in_file")]),
        (reorient_se, ts2sb,
            [("out_file", "ref_file")]),
        (reorient_ts, split_ts,
            [("out_file", "in_file")]),
        (reorient_sb, sb2se,
            [("out_file", "in_file")]),
        (reorient_se, sb2se,
            [("out_file", "reference")]),
        (mask_distortions, sb2se,
            [("out_file", "ref_weight")]),
        (ts2sb, combine_rigids,
            [("mat_file", "in_file")]),
        (reorient_sb, realign_qc,
            [("out_file", "target_file")]),
        (ts2sb, realign_qc,
            [("par_file", "realign_params")]),
        (sb2se, combine_rigids,
            [("out_matrix_file", "in_file2")]),
        (split_ts, restore_ts_frames,
            [("out_files", "in_file")]),
        (combine_rigids, restore_ts_frames,
            [("out_file", "premat")]),
        (select_warp, restore_ts_frames,
            [("out", "field_file")]),
        (se2template, select_runwise,
            [("out_matrices", "in_matrices"),
             ("session_info", "session_info")]),
        (runwise_info, select_runwise,
            [("subject", "subject"),
             ("session", "session")]),
        (se2template, restore_ts_frames,
            [("out_template", "ref_file")]),
        (select_runwise, restore_ts_frames,
            [("out_matrix", "postmat")]),
        (restore_ts_frames, merge_ts,
            [("out_file", "in_files")]),
        (merge_ts, average_ts,
            [("merged_file", "in_file")]),
        (merge_ts, runwise_output,
            [("merged_file", "@restored_timeseries")]),
        (average_ts, runwise_output,
            [("out_file", "@mean_func")]),
        (realign_qc, runwise_output,
            [("params_file", "@realign_params"),
             ("params_plot", "qc.@params_plot"),
             ("target_plot", "qc.@target_plot")]),
        (average_template, subjwise_output,
            [("out_file", "@template")]),
        (se2template, subjwise_output,
            [("out_tkreg_file", "@tkreg_file")]),
    ])

    return workflow


class RealignmentReport(SimpleInterface):

    class input_spec(TraitedSpec):
        target_file = File(exists=True)
        realign_params = File(exists=True)

    class output_spec(TraitedSpec):
        params_file = File(exists=True)
        params_plot = File(exists=True)
        target_plot = File(exists=True)

    def _run_interface(self, runtime):

        # Load the realignment parameters
        params = np.loadtxt(self.inputs.realign_params)
        cols = ["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"]
        df = pd.DataFrame(params, columns=cols)

        # Write the motion file to csv
        params_file = op.abspath("realign_params.csv")
        self._results["params_file"] = params_file
        df.to_csv(params_file, index=False)

        # Plot the motion timeseries
        params_plot = op.abspath("realign_params.png")
        self._results["params_plot"] = params_plot
        f = self.plot_motion(df)
        f.savefig(params_plot, dpi=100)
        plt.close(f)

        # Plot the target image
        target_plot = op.abspath("realign_target.png")
        self._results["target_plot"] = target_plot
        m = self.plot_target()
        m.savefig(target_plot)
        m.close()

        return runtime

    def plot_motion(self, df):
        """Plot the timecourses of realignment parameters."""
        with sns.axes_style("whitegrid"):
            fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True)

        # Trim off all but the axis name
        def axis(s):
            return s[-1]

        # Plot rotations
        pal = sns.color_palette("Reds_d", 3)
        rot_df = df.filter(like="rot").apply(np.rad2deg).rename(columns=axis)
        rot_df.plot(ax=axes[0], color=pal, lw=1.5)

        # Plot translations
        pal = sns.color_palette("Blues_d", 3)
        trans_df = df.filter(like="trans").rename(columns=axis)
        trans_df.plot(ax=axes[1], color=pal, lw=1.5)

        # Label the graphs
        axes[0].set_xlim(0, len(df) - 1)
        axes[0].axhline(0, c=".4", ls="--", zorder=1)
        axes[1].axhline(0, c=".4", ls="--", zorder=1)

        for ax in axes:
            ax.legend(frameon=True, ncol=3, loc="best")
            ax.legend_.get_frame().set_color("white")

        axes[0].set_ylabel("Rotations (degrees)")
        axes[1].set_ylabel("Translations (mm)")
        fig.tight_layout()
        return fig

    def plot_target(self):
        """Plot a mosaic of the motion correction target image."""
        return Mosaic(self.inputs.target_file, step=1)


class TemplateTransform(SimpleInterface):

    class input_spec(TraitedSpec):
        session_info = traits.List(traits.Tuple())
        in_matrices = InputMultiPath(File(exists=True))
        in_volumes = InputMultiPath(File(exists=True))

    class output_spec(TraitedSpec):
        session_info = traits.List(traits.Tuple())
        out_template = File(exists=True)
        out_flirt_file = File(exists=True)
        out_tkreg_file = File(exists=True)
        out_matrices = OutputMultiPath(File(exists=True))

    def _run_interface(self, runtime):

        self._results["session_info"] = self.inputs.session_info

        subjects_dir = os.environ["SUBJECTS_DIR"]
        subject_ids = set([s for s, _ in self.inputs.session_info])
        assert len(subject_ids) == 1
        subj = subject_ids.pop()

        # -- Convert the anatomical image to nifti
        anat_file = "orig.nii.gz"
        cmdline = ["mri_convert",
                   op.join(subjects_dir, subj, "mri/orig.mgz"),
                   anat_file]

        self.submit_cmdline(runtime, cmdline)

        # -- Compute the intermediate transform

        cmdline = ["midtrans",
                   "--template=" + anat_file,
                   "--separate=se2template_",
                   "--out=anat2func_flirt.mat"]
        cmdline.extend(self.inputs.in_matrices)
        out_matrices = [
            op.abspath("se2template_{:04d}.mat".format(i))
            for i, _ in enumerate(self.inputs.in_matrices, 1)
        ]

        self.submit_cmdline(runtime, cmdline, out_matrices=out_matrices)

        # -- Invert the anat2temp transformation
        out_tkreg_file = op.abspath("func2anat_flirt.mat")
        cmdline = ["convert_xfm",
                   "-omat", out_tkreg_file,
                   "-inverse",
                   "anat2func_flirt.mat"]

        self.submit_cmdline(runtime, cmdline, out_tkreg_file=out_tkreg_file)

        # -- Transform first volume into template space to get the geometry
        out_template = op.abspath("template_space.nii.gz")
        cmdline = ["flirt",
                   "-in", self.inputs.in_volumes[0],
                   "-ref", self.inputs.in_volumes[0],
                   "-init", out_matrices[0],
                   "-out", out_template,
                   "-applyxfm"]

        self.submit_cmdline(runtime, cmdline, out_template=out_template)

        # -- Convert the FSL matrices to tkreg matrix format
        out_flirt_file = op.abspath("func2anat_flirt.mat")
        cmdline = ["tkregister2",
                   "--s", subj,
                   "--mov", "template_space.nii.gz",
                   "--fsl", out_flirt_file,
                   "--reg", out_tkreg_file,
                   "--noedit"]

        self.submit_cmdline(runtime, cmdline, out_flirt_file=out_flirt_file)

        return runtime
