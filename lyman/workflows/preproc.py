import os
import os.path as op

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nibabel as nib

from nipype import (Workflow, Node, MapNode, JoinNode,
                    Function, IdentityInterface, SelectFiles, DataSink)
from nipype.interfaces.base import (traits, File, TraitedSpec,
                                    InputMultiPath, OutputMultiPath,
                                    isdefined)
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

    # TODO change se to fm, it will be less confusing

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

    # TODO we all need to impelement removal of first n frames
    # (not relevant for Prisma data but necessary elsewhere)
    # Also the FSL workflow converts input data to float32.
    # All of these steps can be done in pure Python, so perhaps a
    # custom interface should be defined and replace these FSL calls

    reorient_ts = Node(fsl.Reorient2Std(), "reorient_ts")
    reorient_fm = reorient_ts.clone("reorient_fm")
    reorient_sb = reorient_ts.clone("reorient_sb")

    # --- Warpfield estimation using topup

    # Distortion warpfield estimation
    # TODO this needs to be in the experiment file!
    phase_encoding = ["y", "y", "y", "y-", "y-", "y-"]
    readout_times = [1, 1, 1, 1, 1, 1]
    estimate_distortions = Node(fsl.TOPUP(encoding_direction=phase_encoding,
                                          readout_times=readout_times,
                                          config="b02b0.cnf"),
                                "estimate_distortions")

    fieldmap_qc = Node(DistortionGIF(phase_encoding=phase_encoding,
                                     out_file="fieldmap.gif"),
                       "fieldmap_qc")

    unwarp_qc = Node(DistortionGIF(phase_encoding=phase_encoding,
                                   out_file="unwarp.gif"),
                     "unwarp_qc")

    # Average distortion-corrected spin-echo images
    average_fm = Node(fsl.MeanImage(out_file="fm_restored.nii.gz"),
                      "average_fm")

    # Select first warpfield image from output list
    select_warp = Node(pipeutil.Select(index=[0]), "select_warp")

    # Define a mask of areas with large distortions
    thresh_ops = "-abs -thr 4 -Tmax -binv"
    mask_distortions = Node(fsl.ImageMaths(op_string=thresh_ops),
                            "mask_distortions")

    # --- Registration of SBRef to SE-EPI (with distortions)

    sb2fm = Node(fsl.FLIRT(dof=6, interp="spline"), "sb2fm")

    sb2fm_qc = Node(CoregGIF(out_file="coreg.gif"), "sb2fm_qc")

    # --- Registration of SE-EPI (without distortions) to Freesurfer anatomy

    fm2anat = Node(fs.BBRegister(init="fsl",
                                 contrast_type="t2",
                                 out_fsl_file="fm2anat_flirt.mat",
                                 out_reg_file="fm2anat_tkreg.dat"),
                   "fm2anat")

    # TODO anatomy registration QC
    fm2anat_qc = Node(AnatRegReport(out_file="func2anat.png"), "fm2anat_qc")

    # --- Definition of common cross-session space (template space)

    fm2template = JoinNode(TemplateTransform(),
                           name="fm2template",
                           joinsource="session_source",
                           joinfield=["session_info",
                                      "in_matrices", "in_volumes"])

    # TODO template creation QC
    func2anat_qc = Node(AnatRegReport(out_file="func2anat.png"),
                        "func2anat_qc")

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

    split_fm = Node(fsl.Split(dimension="t"), "split_fm")

    restore_fm = MapNode(fsl.ApplyWarp(interp="spline", relwarp=True),
                         ["in_file", "premat", "field_file"],
                         "restore_fm")

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

    template_qc = Node(FrameGIF(out_file="func_frames.gif", delay=20),
                       "template_qc")

    # --- Motion correction of time series to SBRef (with distortions)

    ts2sb = Node(fsl.MCFLIRT(save_mats=True, save_plots=True),
                 "ts2sb")

    realign_qc = Node(RealignmentReport(), "realign_qc")

    # --- Combined motion correction and unwarping of time series

    # Split the timeseries into each frame
    split_ts = Node(fsl.Split(dimension="t"), "split_ts")

    # Concatenation ts2sb and sb2sfmrigid transform
    combine_rigids = MapNode(fsl.ConvertXFM(concat_xfm=True),
                             "in_file", "combine_rigids")

    # Simultaneously apply rigid transform and nonlinear warpfield
    restore_ts_frames = MapNode(fsl.ApplyWarp(interp="spline", relwarp=True),
                                ["in_file", "premat"],
                                "restore_ts")

    # Recombine the time series frames into a 4D image
    merge_ts = Node(fsl.Merge(merged_file="timeseries.nii.gz",
                              dimension="t"), "merge_ts")

    # Take a temporal average of the time series
    average_ts = Node(fsl.MeanImage(out_file="mean.nii.gz"),
                      "average_ts")

    # --- Workflow ouptut

    output_dir = op.join(proj_info.analysis_dir, exp_info.name)

    def define_timeseries_container(subject, session, run):
        return "{}/preproc/{}_{}".format(subject, session, run)

    timeseries_container = Node(Function(["subject", "session", "run"],
                                         "path", define_timeseries_container),
                                "runwise_container")

    timeseries_output = Node(DataSink(base_directory=output_dir,
                                      parameterization=False),
                             "timeseries_output")

    def define_template_container(subject):
        return "{}/template".format(subject)

    template_container = Node(Function("subject",
                                       "path", define_template_container),
                              "template_container")

    template_output = Node(DataSink(base_directory=output_dir,
                                    parameterization=False),
                           "template_output")

    # === Assemble pipeline

    cache_base = op.join(proj_info.cache_dir, exp_info.name)
    workflow = Workflow(name="preproc", base_dir=cache_base)

    workflow.connect([

        # --- Workflow setup and data ingest

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

        # --- Distortion correction and template definition

        (sesswise_input, reorient_fm,
            [("se", "in_file")]),
        (reorient_fm, estimate_distortions,
            [("out_file", "in_file")]),
        (reorient_fm, fieldmap_qc,
            [("out_file", "in_file")]),
        (sesswise_info, fieldmap_qc,
            [("session", "session")]),
        (estimate_distortions, select_warp,
            [("out_warps", "inlist")]),
        (sesswise_info, unwarp_qc,
            [("session", "session")]),
        (estimate_distortions, unwarp_qc,
            [("out_corrected", "in_file")]),
        (select_warp, mask_distortions,
            [("out", "in_file")]),
        (estimate_distortions, average_fm,
            [("out_corrected", "in_file")]),
        (sesswise_info, fm2anat,
            [("subject", "subject_id")]),
        (average_fm, fm2anat,
            [("out_file", "source_file")]),
        (sesswise_info, fm2anat_qc,
            [("subject", "subject_id"),
             ("session", "session")]),
        (average_fm, fm2anat_qc,
            [("out_file", "in_file")]),
        (fm2anat, fm2anat_qc,
            [("out_reg_file", "reg_file"),
             ("min_cost_file", "cost_file")]),
        (session_source, fm2template,
            [("session", "session_info")]),
        (reorient_fm, fm2template,
            [("out_file", "in_volumes")]),
        (fm2anat, fm2template,
            [("out_fsl_file", "in_matrices")]),
        (fm2template, select_sesswise,
            [("out_matrices", "in_matrices"),
             ("out_template", "in_templates"),
             ("session_info", "session_info")]),
        (sesswise_info, func2anat_qc,
            [("subject", "subject_id")]),
        (fm2template, func2anat_qc,
            [("out_tkreg_file", "reg_file")]),
        (sesswise_info, select_sesswise,
            [("subject", "subject"),
             ("session", "session")]),
        (reorient_fm, split_fm,
            [("out_file", "in_file")]),
        (split_fm, restore_fm,
            [("out_files", "in_file")]),
        (estimate_distortions, restore_fm,
            [("out_mats", "premat"),
             ("out_warps", "field_file")]),
        (fm2template, restore_fm,
            [("out_template", "ref_file")]),
        (select_sesswise, restore_fm,
            [("out_matrix", "postmat")]),
        (restore_fm, combine_template,
            [("out_file", "in_files")]),
        (combine_template, merge_template,
            [("out_files", "in_files")]),
        (merge_template, average_template,
            [("merged_file", "in_file")]),
        (average_template, func2anat_qc,
            [("out_file", "in_file")]),
        (merge_template, template_qc,
            [("merged_file", "in_file")]),

        # --- Time series realignment

        (runwise_input, reorient_ts,
            [("ts", "in_file")]),
        (runwise_input, reorient_sb,
            [("sb", "in_file")]),
        (reorient_ts, ts2sb,
            [("out_file", "in_file")]),
        (reorient_sb, ts2sb,
            [("out_file", "ref_file")]),
        (reorient_ts, split_ts,
            [("out_file", "in_file")]),
        (reorient_sb, sb2fm,
            [("out_file", "in_file")]),
        (reorient_fm, sb2fm,
            [("out_file", "reference")]),
        (mask_distortions, sb2fm,
            [("out_file", "ref_weight")]),
        (sb2fm, sb2fm_qc,
            [("out_file", "in_file")]),
        (reorient_fm, sb2fm_qc,
            [("out_file", "ref_file")]),
        (ts2sb, combine_rigids,
            [("mat_file", "in_file")]),
        (reorient_sb, realign_qc,
            [("out_file", "target_file")]),
        (ts2sb, realign_qc,
            [("par_file", "realign_params")]),
        (sb2fm, combine_rigids,
            [("out_matrix_file", "in_file2")]),
        (split_ts, restore_ts_frames,
            [("out_files", "in_file")]),
        (combine_rigids, restore_ts_frames,
            [("out_file", "premat")]),
        (select_warp, restore_ts_frames,
            [("out", "field_file")]),
        (fm2template, select_runwise,
            [("out_matrices", "in_matrices"),
             ("session_info", "session_info")]),
        (runwise_info, select_runwise,
            [("subject", "subject"),
             ("session", "session")]),
        (fm2template, restore_ts_frames,
            [("out_template", "ref_file")]),
        (select_runwise, restore_ts_frames,
            [("out_matrix", "postmat")]),
        (restore_ts_frames, merge_ts,
            [("out_file", "in_files")]),
        (merge_ts, average_ts,
            [("merged_file", "in_file")]),

        # --- Persistent data storage

        (sesswise_info, template_container,
            [("subject", "subject")]),
        (template_container, template_output,
            [("path", "container")]),
        (average_template, template_output,
            [("out_file", "@template")]),
        (fm2template, template_output,
            [("out_tkreg_file", "@tkreg_file")]),
        (fieldmap_qc, template_output,
            [("out_file", "qc.@fieldmap_gif")]),
        (unwarp_qc, template_output,
            [("out_file", "qc.@unwarp_gif")]),
        (fm2anat_qc, template_output,
            [("out_file", "qc.@fm2anat_plot")]),
        (func2anat_qc, template_output,
            [("out_file", "qc.@func2anat_plot")]),
        (template_qc, template_output,
            [("out_file", "qc.@template_gif")]),

        (runwise_info, timeseries_container,
            [("subject", "subject"),
             ("session", "session"),
             ("run", "run")]),
        (timeseries_container, timeseries_output,
            [("path", "container")]),
        (merge_ts, timeseries_output,
            [("merged_file", "@restored_timeseries")]),
        (average_ts, timeseries_output,
            [("out_file", "@mean_func")]),
        (sb2fm_qc, timeseries_output,
            [("out_file", "qc.@sb2fm_gif")]),
        (realign_qc, timeseries_output,
            [("params_file", "@realign_params"),
             ("params_plot", "qc.@params_plot"),
             ("target_plot", "qc.@target_plot")]),
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
        params_file = op.abspath("mc_params.csv")
        self._results["params_file"] = params_file
        df.to_csv(params_file, index=False)

        # Plot the motion timeseries
        params_plot = op.abspath("mc_params.png")
        self._results["params_plot"] = params_plot
        f = self.plot_motion(df)
        f.savefig(params_plot, dpi=100)
        plt.close(f)

        # Plot the target image
        target_plot = op.abspath("mc_target.png")
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
        return Mosaic(self.inputs.target_file, step=2)


class AnatRegReport(SimpleInterface):

    class input_spec(TraitedSpec):
        subject_id = traits.Str()
        session = traits.Str()
        in_file = traits.File(exists=True)
        reg_file = traits.File(exists=True)
        cost_file = traits.File(exists=True)
        out_file = traits.File()

    class output_spec(TraitedSpec):
        out_file = traits.File(exists=True)

    def _run_interface(self, runtime):

        # Use the registration to transform the input file
        registered_file = "func_in_anat.nii.gz"
        cmdline = ["mri_vol2vol",
                   "--mov", self.inputs.in_file,
                   "--reg", self.inputs.reg_file,
                   "--o", registered_file,
                   "--fstarg",
                   "--cubic"]

        self.submit_cmdline(runtime, cmdline)

        # Load the WM segmentation and a brain mask
        mri_dir = op.join(os.environ["SUBJECTS_DIR"],
                          self.inputs.subject_id, "mri")

        wm_file = op.join(mri_dir, "wm.mgz")
        wm_data = (nib.load(wm_file).get_data() > 0).astype(int)

        aseg_file = op.join(mri_dir, "aseg.mgz")
        mask = (nib.load(aseg_file).get_data() > 0).astype(int)

        # Read the final registration cost
        if isdefined(self.inputs.cost_file):
            cost = np.loadtxt(self.inputs.cost_file)[0]
        else:
            cost = None

        # Make a mosaic of the registration from func to wm seg
        # TODO this should be an OrthoMosaic when that is implemented
        if isdefined(self.inputs.session):
            fname = "{}_{}".format(self.inputs.session, self.inputs.out_file)
        else:
            fname = self.inputs.out_file
        self._results["out_file"] = op.abspath(fname)

        m = Mosaic(registered_file, wm_data, mask, step=3, show_mask=False)
        m.plot_mask_edges("#DD2222")
        if cost is not None:
            m.fig.suptitle("Final cost: {:.2f}".format(cost),
                           size=10, color="white")
        m.savefig(fname)
        m.close()

        return runtime


class FrameGIF(SimpleInterface):

    class input_spec(TraitedSpec):
        in_file = traits.File(exists=True)
        out_file = traits.File()
        delay = traits.Int(100, usedefault=True)

    class output_spec(TraitedSpec):
        out_file = traits.File(exists=True)

    def _run_interface(self, runtime):

        img = nib.load(self.inputs.in_file)

        assert len(img.shape) == 4
        n_frames = img.shape[-1]

        frame_pngs = []

        for i in range(n_frames):

            png_fname = "frame{:02d}.png".format(i)
            frame_pngs.append(png_fname)

            vol_data = img.get_data()[..., i]
            vol = nib.Nifti1Image(vol_data, img.affine, img.header)
            m = Mosaic(vol, tight=False, step=2)
            m.savefig(png_fname)
            m.close()

        cmdline = ["convert",
                   "-loop", "0",
                   "-delay", str(self.inputs.delay)]
        cmdline += frame_pngs
        cmdline.append(self.inputs.out_file)

        self.submit_cmdline(runtime, cmdline,
                            out_file=op.abspath(self.inputs.out_file))

        return runtime


class CoregGIF(SimpleInterface):

    class input_spec(TraitedSpec):
        in_file = traits.File(exists=True)
        ref_file = traits.File(exists=True)
        out_file = traits.File()

    class output_spec(TraitedSpec):
        out_file = traits.File(exists=True)

    def _run_interface(self, runtime):

        in_img = nib.load(self.inputs.in_file)
        ref_img = nib.load(self.inputs.ref_file)
        out_fname = self.inputs.out_file

        if len(ref_img.shape) > 3:
            ref_data = ref_img.get_data()[..., 0]
            ref_img = nib.Nifti1Image(ref_data, ref_img.affine, ref_img.header)

        self.write_mosaic_gif(runtime, in_img, ref_img, out_fname)

        self._results["out_file"] = op.abspath(out_fname)

        return runtime

    def write_mosaic_gif(self, runtime, img1, img2, fname, **kws):

        m1 = Mosaic(img1, **kws)
        m1.savefig("img1.png")
        m1.close()

        m2 = Mosaic(img2, **kws)
        m2.savefig("img2.png")
        m2.close()

        cmdline = ["convert", "-loop", "0", "-delay", "100",
                   "img1.png", "img2.png", fname]

        self.submit_cmdline(runtime, cmdline)


class DistortionGIF(CoregGIF):

    class input_spec(TraitedSpec):
        in_file = traits.File(exists=True)
        out_file = traits.File()
        phase_encoding = traits.List(traits.Str)
        session = traits.Str()

    def _run_interface(self, runtime):

        img = nib.load(self.inputs.in_file)
        data = img.get_data()

        imgs = []

        pe = np.array(self.inputs.phase_encoding)
        for enc in np.unique(pe):

            enc_trs = pe == enc
            enc_data = data[..., enc_trs].mean(axis=-1)

            imgs.append(nib.Nifti1Image(enc_data, img.affine, img.header))

        if isdefined(self.inputs.session):
            fname = "{}_{}".format(self.inputs.session, self.inputs.out_file)
        else:
            fname = self.inputs.out_file
        img1, img2 = imgs

        self.write_mosaic_gif(runtime, img1, img2, fname,
                              slice_dir="sag", tight=False)

        self._results["out_file"] = op.abspath(fname)

        return runtime


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
