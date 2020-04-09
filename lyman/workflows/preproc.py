from __future__ import division
import os
import os.path as op

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib

from nipype import (Workflow, Node, MapNode, JoinNode,
                    IdentityInterface, DataSink)
from nipype.interfaces.base import traits, TraitedSpec
from nipype.interfaces import fsl, freesurfer as fs

from .. import signals
from ..utils import LymanInterface, SaveInfo
from ..visualizations import Mosaic, CarpetPlot


def define_preproc_workflow(info, subjects, sessions, qc=True):

    # --- Workflow parameterization and data input

    scan_info = info.scan_info
    experiment = info.experiment_name

    iterables = generate_iterables(scan_info, experiment, subjects, sessions)
    subject_iterables, session_iterables, run_iterables = iterables

    subject_iterables = subjects

    subject_source = Node(IdentityInterface(["subject"]),
                          name="subject_source",
                          iterables=("subject", subject_iterables))

    session_source = Node(IdentityInterface(["subject", "session"]),
                          name="session_source",
                          itersource=("subject_source", "subject"),
                          iterables=("session", session_iterables))

    run_source = Node(IdentityInterface(["subject", "session", "run"]),
                      name="run_source",
                      itersource=("session_source", "session"),
                      iterables=("run", run_iterables))

    session_input = Node(SessionInput(data_dir=info.data_dir,
                                      proc_dir=info.proc_dir,
                                      fm_template=info.fm_template,
                                      phase_encoding=info.phase_encoding),
                         "session_input")

    run_input = Node(RunInput(experiment=experiment,
                              data_dir=info.data_dir,
                              proc_dir=info.proc_dir,
                              sb_template=info.sb_template,
                              ts_template=info.ts_template,
                              crop_frames=info.crop_frames),
                     name="run_input")

    # --- Warpfield estimation using topup

    # Distortion warpfield estimation
    #  TODO figure out how to parameterize for testing
    # topup_config = op.realpath(op.join(__file__, "../../../topup_fast.cnf"))
    topup_config = "b02b0.cnf"
    estimate_distortions = Node(fsl.TOPUP(config=topup_config),
                                "estimate_distortions")

    # Post-process the TOPUP outputs
    finalize_unwarping = Node(FinalizeUnwarping(), "finalize_unwarping")

    # --- Registration of SE-EPI (without distortions) to Freesurfer anatomy

    fm2anat = Node(fs.BBRegister(init="fsl",
                                 contrast_type="t2",
                                 registered_file=True,
                                 out_fsl_file="sess2anat.mat",
                                 out_reg_file="sess2anat.dat"),
                   "fm2anat")

    fm2anat_qc = Node(AnatRegReport(data_dir=info.data_dir), "fm2anat_qc")

    # --- Registration of SBRef to SE-EPI (with distortions)

    sb2fm = Node(fsl.FLIRT(dof=6, interp="spline"), "sb2fm")

    sb2fm_qc = Node(CoregGIF(out_file="coreg.gif"), "sb2fm_qc")

    # --- Motion correction of time series to SBRef (with distortions)

    ts2sb = Node(fsl.MCFLIRT(save_mats=True, save_plots=True),
                 "ts2sb")

    ts2sb_qc = Node(RealignmentReport(), "ts2sb_qc")

    # --- Combined motion correction, unwarping, and template registration

    # Combine pre-and post-warp linear transforms
    combine_premats = MapNode(fsl.ConvertXFM(concat_xfm=True),
                              "in_file", "combine_premats")

    combine_postmats = Node(fsl.ConvertXFM(concat_xfm=True),
                            "combine_postmats")

    # Transform Jacobian images into the template space
    transform_jacobian = Node(fsl.ApplyWarp(relwarp=True),
                              "transform_jacobian")

    # Apply rigid transforms and nonlinear warpfield to time series frames
    restore_timeseries = MapNode(fsl.ApplyWarp(interp="spline", relwarp=True),
                                 ["in_file", "premat"],
                                 "restore_timeseries")

    # Apply rigid transforms and nonlinear warpfield to template frames
    restore_template = MapNode(fsl.ApplyWarp(interp="spline", relwarp=True),
                               ["in_file", "premat", "field_file"],
                               "restore_template")

    # Perform final preprocessing operations on timeseries
    finalize_timeseries = Node(FinalizeTimeseries(experiment=experiment),
                               "finalize_timeseries")

    # Perform final preprocessing operations on template
    finalize_template = JoinNode(FinalizeTemplate(experiment=experiment),
                                 name="finalize_template",
                                 joinsource="run_source",
                                 joinfield=["mean_files", "tsnr_files",
                                            "mask_files", "noise_files"])

    # --- Workflow ouptut

    save_info = Node(SaveInfo(info_dict=info.trait_get()), "save_info")

    template_output = Node(DataSink(base_directory=info.proc_dir,
                                    parameterization=False),
                           "template_output")

    timeseries_output = Node(DataSink(base_directory=info.proc_dir,
                                      parameterization=False),
                             "timeseries_output")

    # === Assemble pipeline

    cache_base = op.join(info.cache_dir, info.experiment_name)
    workflow = Workflow(name="preproc", base_dir=cache_base)

    # Connect processing nodes

    processing_edges = [

        (subject_source, session_source,
            [("subject", "subject")]),
        (subject_source, run_source,
            [("subject", "subject")]),
        (session_source, run_source,
            [("session", "session")]),
        (session_source, session_input,
            [("session", "session")]),
        (run_source, run_input,
            [("run", "run")]),

        # Phase-encode distortion estimation

        (session_input, estimate_distortions,
            [("fm_file", "in_file"),
             ("phase_encoding", "encoding_direction"),
             ("readout_times", "readout_times")]),
        (session_input, finalize_unwarping,
            [("session_tuple", "session_tuple"),
             ("fm_file", "raw_file"),
             ("phase_encoding", "phase_encoding")]),
        (estimate_distortions, finalize_unwarping,
            [("out_corrected", "corrected_file"),
             ("out_warps", "warp_files"),
             ("out_jacs", "jacobian_files")]),

        # Registration of corrected SE-EPI to anatomy

        (session_input, fm2anat,
            [("subject", "subject_id")]),
        (finalize_unwarping, fm2anat,
            [("corrected_file", "source_file")]),

        # Registration of each frame to SBRef image

        (run_input, ts2sb,
            [("ts_file", "in_file"),
             ("sb_file", "ref_file")]),
        (ts2sb, finalize_timeseries,
            [("par_file", "mc_file")]),

        # Registration of SBRef volume to SE-EPI fieldmap

        (run_input, sb2fm,
            [("sb_file", "in_file")]),
        (finalize_unwarping, sb2fm,
            [("raw_file", "reference"),
             ("mask_file", "ref_weight")]),

        # Single-interpolation spatial realignment and unwarping

        (ts2sb, combine_premats,
            [("mat_file", "in_file")]),
        (sb2fm, combine_premats,
            [("out_matrix_file", "in_file2")]),
        (fm2anat, combine_postmats,
            [("out_fsl_file", "in_file")]),
        (session_input, combine_postmats,
            [("reg_file", "in_file2")]),

        (run_input, transform_jacobian,
            [("anat_file", "ref_file")]),
        (finalize_unwarping, transform_jacobian,
            [("jacobian_file", "in_file")]),
        (combine_postmats, transform_jacobian,
            [("out_file", "premat")]),

        (run_input, restore_timeseries,
            [("ts_frames", "in_file")]),
        (run_input, restore_timeseries,
            [("anat_file", "ref_file")]),
        (combine_premats, restore_timeseries,
            [("out_file", "premat")]),
        (finalize_unwarping, restore_timeseries,
            [("warp_file", "field_file")]),
        (combine_postmats, restore_timeseries,
            [("out_file", "postmat")]),
        (run_input, finalize_timeseries,
            [("run_tuple", "run_tuple"),
             ("anat_file", "anat_file"),
             ("seg_file", "seg_file"),
             ("mask_file", "mask_file")]),
        (transform_jacobian, finalize_timeseries,
            [("out_file", "jacobian_file")]),
        (restore_timeseries, finalize_timeseries,
            [("out_file", "in_files")]),

        (session_input, restore_template,
            [("fm_frames", "in_file"),
             ("anat_file", "ref_file")]),
        (estimate_distortions, restore_template,
            [("out_mats", "premat"),
             ("out_warps", "field_file")]),
        (combine_postmats, restore_template,
            [("out_file", "postmat")]),
        (session_input, finalize_template,
            [("session_tuple", "session_tuple"),
             ("seg_file", "seg_file"),
             ("anat_file", "anat_file")]),
        (transform_jacobian, finalize_template,
            [("out_file", "jacobian_file")]),
        (restore_template, finalize_template,
            [("out_file", "in_files")]),

        (finalize_timeseries, finalize_template,
            [("mean_file", "mean_files"),
             ("tsnr_file", "tsnr_files"),
             ("mask_file", "mask_files"),
             ("noise_file", "noise_files")]),

        # --- Persistent data storage

        # Ouputs associated with each scanner run

        (finalize_timeseries, timeseries_output,
            [("output_path", "container"),
             ("out_file", "@func"),
             ("mean_file", "@mean"),
             ("mask_file", "@mask"),
             ("tsnr_file", "@tsnr"),
             ("noise_file", "@noise"),
             ("mc_file", "@mc")]),

        # Ouputs associated with the session template

        (finalize_template, template_output,
            [("output_path", "container"),
             ("out_file", "@func"),
             ("mean_file", "@mean"),
             ("tsnr_file", "@tsnr"),
             ("mask_file", "@mask"),
             ("noise_file", "@noise")]),

    ]
    workflow.connect(processing_edges)

    # Optionally connect QC nodes

    qc_edges = [

        # Registration of each frame to SBRef image

        (run_input, ts2sb_qc,
            [("run_tuple", "run_tuple")]),
        (run_input, ts2sb_qc,
            [("sb_file", "target_file")]),
        (ts2sb, ts2sb_qc,
            [("par_file", "realign_params")]),

        # Registration of corrected SE-EPI to anatomy

        (session_input, fm2anat_qc,
            [("subject", "subject_id"),
             ("session_tuple", "session_tuple")]),
        (fm2anat, fm2anat_qc,
            [("registered_file", "in_file"),
             ("min_cost_file", "cost_file")]),

        # Registration of SBRef volume to SE-EPI fieldmap

        (run_input, sb2fm_qc,
            [("run_tuple", "run_tuple")]),
        (sb2fm, sb2fm_qc,
            [("out_file", "in_file")]),
        (finalize_unwarping, sb2fm_qc,
            [("raw_file", "ref_file")]),

        # Ouputs associated with each scanner run

        (run_source, save_info,
            [("run", "parameterization")]),
        (save_info, timeseries_output,
            [("info_file", "qc.@info_json")]),

        (run_input, timeseries_output,
            [("ts_plot", "qc.@raw_gif")]),
        (sb2fm_qc, timeseries_output,
            [("out_file", "qc.@sb2fm_gif")]),
        (ts2sb_qc, timeseries_output,
            [("params_plot", "qc.@params_plot"),
             ("target_plot", "qc.@target_plot")]),
        (finalize_timeseries, timeseries_output,
            [("out_gif", "qc.@ts_gif"),
             ("out_png", "qc.@ts_png"),
             ("mask_plot", "qc.@mask_plot"),
             ("mean_plot", "qc.@ts_mean_plot"),
             ("tsnr_plot", "qc.@ts_tsnr_plot"),
             ("noise_plot", "qc.@noise_plot")]),

        # Outputs associated with the session template

        (finalize_unwarping, template_output,
            [("warp_plot", "qc.@warp_png"),
             ("unwarp_gif", "qc.@unwarp_gif")]),
        (fm2anat_qc, template_output,
            [("out_file", "qc.@reg_png")]),
        (finalize_template, template_output,
            [("out_plot", "qc.@func_png"),
             ("mean_plot", "qc.@mean"),
             ("tsnr_plot", "qc.@tsnr"),
             ("mask_plot", "qc.@mask"),
             ("noise_plot", "qc.@noise")]),

    ]

    if qc:
        workflow.connect(qc_edges)

    return workflow


# =========================================================================== #
# Custom processing code
# =========================================================================== #


def generate_iterables(scan_info, experiment, subjects, sessions=None):
    """Return lists of variables for preproc workflow iterables.

    Parameters
    ----------
    scan_info : nested dictionaries
        A nested dictionary structure with the following key levels:
            - subject ids
            - session ids
            - experiment names
        Where the inner values are lists of run ids.
    experiment : string
        Name of the experiment to generate iterables for.
    subjects : list of strings
        List of subject ids to generate iterables for.
    sessions : list of strings, optional
        List of sessions to generate iterables for.

    Returns
    -------
    subject_iterables: list of strings
        A list of the subjects with runs for this experiment.
    session_iterables : dict
        A dictionary where keys are subject ids and values are lists of
        (subject id, session id) pairs
    run_iterables : dict
        A dictionary where keys are (subject id, session id) pairs and values
        lists of (subject id, session id, run id) pairs.

    """
    subject_iterables = []
    session_iterables = {}
    run_iterables = {}

    for subj in subjects:

        subject_session_iterables = []

        for sess in scan_info[subj]:

            session_run_iterables = []

            if sessions is not None and sess not in sessions:
                continue

            if experiment in scan_info[subj][sess]:

                for run in scan_info[subj][sess][experiment]:
                    session_run_iterables.append((subj, sess, run))

            if session_run_iterables:
                sess_key = subj, sess
                subject_session_iterables.append(sess_key)
                run_iterables[sess_key] = session_run_iterables

        if subject_session_iterables:
            subject_iterables.append(subj)
            session_iterables[subj] = subject_session_iterables

    return subject_iterables, session_iterables, run_iterables


# ---- Quality control mixins

class TimeSeriesGIF(object):

    def write_time_series_gif(self, runtime, img, fname, title=None):

        os.mkdir("png")

        nx, ny, nz, nt = img.shape
        delay = 10

        width = 5
        height = width * max([nx, ny, nz]) / sum([nz, ny, nz])

        top = 1
        if title is not None:
            pad = .25
            top = height / (height + pad)
            height += pad

        f, axes = plt.subplots(ncols=3, figsize=(width, height))
        for ax in axes:
            ax.set_axis_off()

        if title is not None:
            f.text(.5, top + (1 - top) / 2, title,
                   ha="center", va="center", color="w", size=10)

        data = img.get_fdata()
        vmin, vmax = np.percentile(data, [2, 98])

        kws = dict(vmin=vmin, vmax=vmax, cmap="gray")
        im_x = axes[0].imshow(np.zeros((nz, ny)), **kws)
        im_y = axes[1].imshow(np.zeros((nz, nx)), **kws)
        im_z = axes[2].imshow(np.zeros((ny, nx)), **kws)

        f.subplots_adjust(0, 0, 1, top, 0, 0)

        x, y, z = nx // 2, ny // 2, nz // 2

        text = f.text(0.02, 0.02, "",
                      size=10, ha="left", va="bottom",
                      color="w", backgroundcolor="0")

        pngs = []

        for t in range(nt):

            vol = data[..., t]

            im_x.set_data(np.rot90(vol[x, :, :]))
            im_y.set_data(np.rot90(vol[:, y, :]))
            im_z.set_data(np.rot90(vol[:, :, z]))

            if not t % 10:
                text.set_text("T: {:d}".format(t))

            frame_png = "png/{:04d}.png".format(t)
            f.savefig(frame_png, facecolor="0", edgecolor="0")
            pngs.append(frame_png)

        cmdline = ["convert",
                   "-loop", "0",
                   "-delay", str(delay),
                   "-limit", "thread", "1"]
        cmdline.extend(pngs)
        cmdline.append(fname)

        self.submit_cmdline(runtime, cmdline)


# ---- Data input and pre-preprocessing


class SessionInput(LymanInterface):

    class input_spec(TraitedSpec):
        session = traits.Tuple(traits.Str(), traits.Str())
        data_dir = traits.Directory(exists=True)
        proc_dir = traits.Directory(exists=True)
        fm_template = traits.Str()
        phase_encoding = traits.Either("ap", "pa")

    class output_spec(TraitedSpec):
        session_tuple = traits.Tuple(traits.Str(), traits.Str())
        subject = traits.Str()
        session = traits.Str()
        fm_file = traits.File(exists=True)
        fm_frames = traits.List(traits.File(exists=True))
        reg_file = traits.File(exists=True)
        seg_file = traits.File(exists=True)
        anat_file = traits.File(exists=True)
        mask_file = traits.File(exists=True)
        phase_encoding = traits.List(traits.Str())
        readout_times = traits.List(traits.Float())

    def _run_interface(self, runtime):

        # Determine the execution parameters
        subject, session = self.inputs.session
        self._results["session_tuple"] = self.inputs.session
        self._results["subject"] = str(subject)
        self._results["session"] = str(session)

        # Determine the phase encoding directions
        pe = self.inputs.phase_encoding
        if pe == "ap":
            pos_pe, neg_pe = "ap", "pa"
        elif pe == "pa":
            pos_pe, neg_pe = "pa", "ap"

        # Spec out full paths to the pair of fieldmap files
        keys = dict(subject=subject, session=session)
        template = self.inputs.fm_template
        func_dir = op.join(self.inputs.data_dir, subject, "func")
        pos_fname = op.join(func_dir,
                            template.format(encoding=pos_pe, **keys))
        neg_fname = op.join(func_dir,
                            template.format(encoding=neg_pe, **keys))

        # Load the two images in canonical orientation
        pos_img = nib.as_closest_canonical(nib.load(pos_fname))
        neg_img = nib.as_closest_canonical(nib.load(neg_fname))
        affine, header = pos_img.affine, pos_img.header

        # Concatenate the images into a single volume
        pos_data = pos_img.get_fdata()
        neg_data = neg_img.get_fdata()
        data = np.concatenate([pos_data, neg_data], axis=-1)
        assert len(data.shape) == 4

        # Convert image datatype to float
        header.set_data_dtype(np.float32)

        # Write out a 4D file
        fname = self.define_output("fm_file", "fieldmap.nii.gz")
        img = nib.Nifti1Image(data, affine, header)
        img.to_filename(fname)

        # Write out a set of 3D files for each frame
        fm_frames = []
        frames = nib.four_to_three(img)
        for i, frame in enumerate(frames):
            fname = op.abspath("fieldmap_{:02d}.nii.gz".format(i))
            fm_frames.append(fname)
            frame.to_filename(fname)
        self._results["fm_frames"] = fm_frames

        # Define phase encoding and readout times for TOPUP
        pe_dir = ["y"] * pos_img.shape[-1] + ["y-"] * neg_img.shape[-1]
        readout_times = [1 for _ in pe_dir]
        self._results["phase_encoding"] = pe_dir
        self._results["readout_times"] = readout_times

        # Load files from the template directory
        template_path = op.join(self.inputs.proc_dir, subject, "template")
        results = dict(
            reg_file=op.join(template_path, "anat2func.mat"),
            seg_file=op.join(template_path, "seg.nii.gz"),
            anat_file=op.join(template_path, "anat.nii.gz"),
            mask_file=op.join(template_path, "mask.nii.gz"),
        )
        self._results.update(results)

        return runtime


class RunInput(LymanInterface, TimeSeriesGIF):

    class input_spec(TraitedSpec):
        run = traits.Tuple(traits.Str(), traits.Str(), traits.Str())
        data_dir = traits.Directory(exists=True)
        proc_dir = traits.Directory(exists=True)
        experiment = traits.Str()
        sb_template = traits.Str()
        ts_template = traits.Str()
        crop_frames = traits.Int(0, usedefault=True)

    class output_spec(TraitedSpec):
        run_tuple = traits.Tuple(traits.Str(), traits.Str(), traits.Str())
        subject = traits.Str()
        session = traits.Str()
        run = traits.Str()
        sb_file = traits.File(exists=True)
        ts_file = traits.File(exists=True)
        ts_frames = traits.List(traits.File(exists=True))
        ts_plot = traits.File(exists=True)
        reg_file = traits.File(exists=True)
        seg_file = traits.File(exists=True)
        anat_file = traits.File(exists=True)
        mask_file = traits.File(exists=True)
        output_path = traits.Directory()

    def _run_interface(self, runtime):

        # Determine the parameters
        experiment = self.inputs.experiment
        subject, session, run = self.inputs.run
        self._results["run_tuple"] = self.inputs.run
        self._results["subject"] = subject
        self._results["session"] = session
        self._results["run"] = run

        # Spec out paths to the input files
        keys = dict(subject=subject, experiment=experiment,
                    session=session, run=run)
        sb_fname = op.join(self.inputs.data_dir, subject, "func",
                           self.inputs.sb_template.format(**keys))
        ts_fname = op.join(self.inputs.data_dir, subject, "func",
                           self.inputs.ts_template.format(**keys))

        # Load the input images in canonical orientation
        sb_img = nib.as_closest_canonical(nib.load(sb_fname))
        ts_img = nib.as_closest_canonical(nib.load(ts_fname))

        # Convert image datatypes to float
        sb_img.set_data_dtype(np.float32)
        ts_img.set_data_dtype(np.float32)

        # Optionally crop the first n frames of the timeseries
        if self.inputs.crop_frames > 0:
            ts_data = ts_img.get_fdata()
            ts_data = ts_data[..., self.inputs.crop_frames:]
            ts_img = nib.Nifti1Image(ts_data, ts_img.affine, ts_img.header)

        # Write out the new images
        self.write_image("sb_file", "sb.nii.gz", sb_img)
        self.write_image("ts_file", "ts.nii.gz", ts_img)

        # Write out each frame of the timeseries
        os.mkdir("frames")
        ts_frames = []
        ts_frame_imgs = nib.four_to_three(ts_img)
        for i, frame_img in enumerate(ts_frame_imgs):
            frame_fname = op.abspath("frames/frame{:04d}.nii.gz".format(i))
            ts_frames.append(frame_fname)
            frame_img.to_filename(frame_fname)
        self._results["ts_frames"] = ts_frames

        # Make a GIF movie of the raw timeseries
        qc_title = "{} {} {}".format(subject, session, run)
        out_plot = self.define_output("ts_plot", "raw.gif")
        self.write_time_series_gif(runtime, ts_img, out_plot, title=qc_title)

        # Load files from the template directory
        template_path = op.join(self.inputs.proc_dir, subject, "template")
        results = dict(
            reg_file=op.join(template_path, "anat2func.mat"),
            seg_file=op.join(template_path, "seg.nii.gz"),
            anat_file=op.join(template_path, "anat.nii.gz"),
            mask_file=op.join(template_path, "mask.nii.gz"),
        )
        self._results.update(results)

        return runtime


# --- Preprocessing operations


class CombineLinearTransforms(LymanInterface):

    class input_spec(TraitedSpec):
        ts2sb_file = traits.File(exists=True)
        sb2fm_file = traits.File(exists=True)
        fm2anat_file = traits.File(exits=True)
        anat2temp_file = traits.File(exists=True)

    class output_spec(TraitedSpec):
        ts2fm_file = traits.File(exists=True)
        fm2temp_file = traits.File(exists=True)

    def _run_interface(self, runtime):

        # Combine the pre-warp transform
        ts2sb_mat = np.loadtxt(self.inputs.ts2sb_file)
        sb2fm_mat = np.loadtxt(self.inputs.sb2fm_file)
        ts2fm_mat = np.dot(sb2fm_mat, ts2sb_mat)
        ts2fm_file = self.define_output("ts2fm_file", "ts2fm.mat")
        np.savetxt(ts2fm_file, ts2fm_mat, delimiter="  ")

        # Combine the post-warp transform
        fm2anat_mat = np.loadtxt(self.inputs.fm2anat_file)
        anat2temp_mat = np.loadtxt(self.inputs.anat2temp_file)
        fm2temp_mat = np.dot(anat2temp_mat, fm2anat_mat)
        fm2temp_file = self.define_output("fm2temp_file", "fm2temp.mat")
        np.savetxt(fm2temp_file, fm2temp_mat, delimiter="  ")

        return runtime


class FinalizeUnwarping(LymanInterface):

    class input_spec(TraitedSpec):
        raw_file = traits.File(exists=True)
        corrected_file = traits.File(exists=True)
        warp_files = traits.List(traits.File(exists=True))
        jacobian_files = traits.List(traits.File(Exists=True))
        phase_encoding = traits.List(traits.Str)
        session_tuple = traits.Tuple(traits.Str(), traits.Str())

    class output_spec(TraitedSpec):
        raw_file = traits.File(exists=True)
        corrected_file = traits.File(exists=True)
        warp_file = traits.File(exists=True)
        mask_file = traits.File(exists=True)
        jacobian_file = traits.File(Exists=True)
        warp_plot = traits.File(exists=True)
        unwarp_gif = traits.File(exists=True)

    def _run_interface(self, runtime):

        # Load the 4D raw fieldmap image and select first frame
        raw_img_frames = nib.load(self.inputs.raw_file)
        raw_img = nib.four_to_three(raw_img_frames)[0]
        affine, header = raw_img.affine, raw_img.header

        # Write out the raw image to serve as a registration target
        self.write_image("raw_file", "raw.nii.gz", raw_img)

        # Load the 4D jacobian image
        jac_img_frames = nib.concat_images(self.inputs.jacobian_files)
        jac_data = jac_img_frames.get_fdata()

        # Load the 4D corrected fieldmap image
        corr_img_frames = nib.load(self.inputs.corrected_file)
        corr_data = corr_img_frames.get_fdata()

        # Average the corrected image over the final dimension and write
        corr_data = corr_data.mean(axis=-1)
        self.write_image("corrected_file", "func.nii.gz",
                         corr_data, affine, header)

        # Save the jacobian images using the raw geometry
        self.write_image("jacobian_file", "jacobian.nii.gz",
                         jac_data, affine, header)

        # Select the first warpfield image
        # We combine the two fieldmap images so that the first one has
        # a phase encoding that matches the time series data.
        # Also note that when the fieldmap images have multiple frames,
        # the warps corresponding to those frames are identical.
        warp_file = self.inputs.warp_files[0]

        # Load in the the warp file and save out with the correct affine
        # (topup doesn't save the header geometry correctly for some reason)
        warp_data = nib.load(warp_file).get_fdata()
        self.write_image("warp_file", "warp.nii.gz", warp_data, affine, header)

        # Select the warp along the phase encode direction
        # Note: we elsewhere currently require phase encoding to be AP or PA
        # so because the input node transforms the fieldmap to canonical
        # orientation this will work. But in the future we might want to ne
        # more flexible with what we accept and will need to change this.
        warp_data_y = warp_data[..., 1]

        # Write out a mask to exclude voxels with large distortions/dropout
        mask_data = (np.abs(warp_data_y) < 4).astype(np.int)
        self.write_image("mask_file", "warp_mask.nii.gz",
                         mask_data, affine, header)

        # Get the metadata
        qc_title = " ".join(self.inputs.session_tuple)

        # Generate a QC image of the warpfield
        m = Mosaic(raw_img, warp_data_y, title=qc_title)
        m.plot_overlay("coolwarm", vmin=-6, vmax=6, alpha=.75)
        self.write_visualization("warp_plot", "warp.png", m)

        # Generate a QC gif of the unwarping performance
        self.generate_unwarp_gif(runtime, raw_img_frames, corr_img_frames)

        return runtime

    def generate_unwarp_gif(self, runtime, raw_img, corrected_img):

        # Load the input and output files
        vol_data = dict(
            orig=raw_img.get_fdata(),
            corr=corrected_img.get_fdata(),
        )

        # Average over the frames that correspond to unique encoding directions
        pe_data = dict(orig=[], corr=[])
        pe = np.array(self.inputs.phase_encoding)
        for enc in np.unique(pe):
            enc_trs = pe == enc
            for scan in ["orig", "corr"]:
                enc_data = vol_data[scan][..., enc_trs].mean(axis=-1)
                pe_data[scan].append(enc_data)

        # Compute the spatial correlation within image pairs
        r_vals = dict()
        for scan, (scan_pos, scan_neg) in pe_data.items():
            r_vals[scan] = np.corrcoef(scan_pos.flat, scan_neg.flat)[0, 1]

        # Set up the figure parameters
        nx, ny, nz, _ = vol_data["orig"].shape
        x_slc = (np.linspace(.2, .8, 8) * nx).astype(np.int)

        vmin, vmax = np.percentile(vol_data["orig"].flat, [2, 98])
        kws = dict(vmin=vmin, vmax=vmax, cmap="gray")
        text_kws = dict(size=7, color="w", backgroundcolor="0",
                        ha="left", va="bottom")

        qc_title = " ".join(self.inputs.session_tuple)

        width = len(x_slc)
        height = (nz / ny) * 2.75

        png_fnames = []
        for i, enc in enumerate(["pos", "neg"]):

            # Initialize the figure and axes
            f = plt.figure(figsize=(width, height))
            gs = dict(
                orig=plt.GridSpec(
                    nrows=1, ncols=len(x_slc), figure=f,
                    left=0, bottom=.5, right=1, top=.94,
                    wspace=0, hspace=0,
                ),
                corr=plt.GridSpec(
                    nrows=1, ncols=len(x_slc), figure=f,
                    left=0, bottom=0, right=1, top=.44,
                    wspace=0, hspace=0,
                )
            )

            # Add text with the image pair correlation
            f.text(.05, .93,
                   "Original similarity: {:.2f}".format(r_vals["orig"]),
                   **text_kws)
            f.text(.05, .43,
                   "Corrected similarity: {:.2f}".format(r_vals["corr"]),
                   **text_kws)

            # Add a title with the metadata
            f.suptitle(qc_title, y=.99, size=10, color="w")

            # Plot the image data and save the static figure
            for scan in ["orig", "corr"]:
                axes = [f.add_subplot(pos) for pos in gs[scan]]
                vol = pe_data[scan][i]

                for ax, x in zip(axes, x_slc):
                    slice = np.rot90(vol[x])
                    ax.imshow(slice, **kws)
                    ax.set_axis_off()

                png_fname = "frame{}.png".format(i)
                png_fnames.append(png_fname)
                f.savefig(png_fname, facecolor="0", edgecolor="0")
                plt.close(f)

        # Combine frames into an animated gif
        out_file = self.define_output("unwarp_gif", "unwarp.gif")
        cmdline = ["convert",
                   "-loop", "0",
                   "-delay", "100",
                   "-limit", "thread", "1"]
        cmdline.extend(png_fnames)
        cmdline.append(out_file)

        self.submit_cmdline(runtime, cmdline)


class FinalizeTimeseries(LymanInterface, TimeSeriesGIF):

    class input_spec(TraitedSpec):
        experiment = traits.Str()
        run_tuple = traits.Tuple(traits.Str(), traits.Str(), traits.Str())
        anat_file = traits.File(exists=True)
        in_files = traits.List(traits.File(exists=True))
        seg_file = traits.File(exists=True)
        mask_file = traits.File(exists=True)
        jacobian_file = traits.File(exists=True)
        mc_file = traits.File(exists=True)

    class output_spec(TraitedSpec):
        out_file = traits.File(exists=True)
        out_gif = traits.File(exists=True)
        out_png = traits.File(exists=True)
        mean_file = traits.File(exists=True)
        mean_plot = traits.File(exists=True)
        tsnr_file = traits.File(exists=True)
        tsnr_plot = traits.File(exists=True)
        mask_file = traits.File(exists=True)
        mask_plot = traits.File(exists=True)
        noise_file = traits.File(exists=True)
        noise_plot = traits.File(exists=True)
        mc_file = traits.File(exists=True)
        output_path = traits.Directory()

    def _run_interface(self, runtime):

        # Concatenate timeseries frames into 4D image
        # TODO Note that the TR information is not propogated into the header
        img = nib.concat_images(self.inputs.in_files)
        affine, header = img.affine, img.header
        data = img.get_fdata()

        # Load the template brain mask image
        mask_img = nib.load(self.inputs.mask_file)
        mask = mask_img.get_fdata().astype(np.bool)

        # Compute a run-specfic mask that excludes voxels outside the FOV
        mask &= data.var(axis=-1) > 0
        self.write_image("mask_file", "mask.nii.gz",
                         mask.astype(np.int), affine, header)

        # Zero-out data outside the mask
        data[~mask] = 0

        # Jacobian modulate each frame of the timeseries image
        jacobian_img = nib.load(self.inputs.jacobian_file)
        jacobian = jacobian_img.get_fdata()[..., [0]]
        data *= jacobian

        # Scale the timeseries for cross-run intensity normalization
        target = 100
        scale_value = target / data[mask].mean()
        data = data * scale_value

        # Remove linear but not constant trend
        data[mask] = signals.detrend(data[mask], axis=-1, replace_mean=True)

        # Save out the final time series
        out_img = self.write_image("out_file", "func.nii.gz",
                                   data, affine, header)

        # Generate the temporal mean and SNR images
        mean = data.mean(axis=-1)
        sd = data.std(axis=-1)
        mask &= sd > 0
        with np.errstate(all="ignore"):
            tsnr = mean / sd
            tsnr[~mask] = 0

        self.write_image("mean_file", "mean.nii.gz", mean, affine, header)
        self.write_image("tsnr_file", "tsnr.nii.gz", tsnr, affine, header)

        # Load the template anatomical image
        anat_img = nib.load(self.inputs.anat_file)

        # Load the template segmentation image
        seg_img = nib.load(self.inputs.seg_file)
        seg = seg_img.get_fdata()

        # Identify unusually noisy voxels
        gray_mask = (0 < seg) & (seg < 5)
        gray_img = nib.Nifti1Image(gray_mask, img.affine, img.header)
        noise_img = signals.identify_noisy_voxels(
            out_img, gray_img, neighborhood=5, threshold=1.5, detrend=False
        )
        self.write_image("noise_file", "noise.nii.gz", noise_img)

        # Load the motion correction params and convert to CSV with header
        mc_file = self.define_output("mc_file", "mc.csv")
        mc_data = np.loadtxt(self.inputs.mc_file)
        cols = ["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"]
        mc_data = pd.DataFrame(mc_data, columns=cols)
        mc_data.to_csv(mc_file, index=False)

        # Define a title to use for QC plots
        qc_title = " ".join(self.inputs.run_tuple)

        # Make a carpet plot of the final timeseries
        p = CarpetPlot(out_img, seg_img, mc_data, title=qc_title)
        self.write_visualization("out_png", "func.png", p)

        # Make a GIF movie of the final timeseries
        out_gif = self.define_output("out_gif", "func.gif")
        self.write_time_series_gif(runtime, out_img, out_gif, title=qc_title)

        # Make a mosaic of the temporal mean normalized to mean cortical signal
        norm_mean = mean / mean[seg == 1].mean()
        mean_m = Mosaic(anat_img, norm_mean, title=qc_title)
        mean_m.plot_overlay("cube:-.15:.5", vmin=0, vmax=2, fmt="d")
        self.write_visualization("mean_plot", "mean.png", mean_m)

        # Make a mosaic of the tSNR
        tsnr_m = Mosaic(anat_img, tsnr, title=qc_title)
        tsnr_m.plot_overlay("cube:.25:-.5", vmin=0, vmax=100, fmt="d")
        self.write_visualization("tsnr_plot", "tsnr.png", tsnr_m)

        # Make a mosaic of the run mask
        # TODO is the better QC showing the run mask over the unmasked mean
        # image so that we can see if the brain is getting cut off?
        mask_m = Mosaic(anat_img, mask_img, title=qc_title)
        mask_m.plot_mask()
        self.write_visualization("mask_plot", "mask.png", mask_m)

        # Make a mosaic of the noisy voxels
        noise_m = Mosaic(anat_img, noise_img, mask_img,
                         show_mask=False, title=qc_title)
        noise_m.plot_mask(alpha=1)
        self.write_visualization("noise_plot", "noise.png", noise_m)

        # Spec out the root path for the timeseries outputs
        subject, session, run = self.inputs.run_tuple
        experiment = self.inputs.experiment
        output_path = op.join(subject, experiment, "timeseries",
                              "{}_{}".format(session, run))
        self._results["output_path"] = output_path

        return runtime


class FinalizeTemplate(LymanInterface):

    class input_spec(TraitedSpec):
        experiment = traits.Str()
        session_tuple = traits.Tuple(traits.Str(), traits.Str())
        in_files = traits.List(traits.File(exists=True))
        seg_file = traits.File(exists=True)
        anat_file = traits.File(exists=True)
        jacobian_file = traits.File(exists=True)
        mask_files = traits.List(traits.File(exists=True))
        mean_files = traits.List(traits.File(exists=True))
        tsnr_files = traits.List(traits.File(exsits=True))
        noise_files = traits.List(traits.File(exists=True))

    class output_spec(TraitedSpec):
        out_file = traits.File(exists=True)
        out_plot = traits.File(exists=True)
        mask_file = traits.File(exists=True)
        mask_plot = traits.File(exists=True)
        noise_file = traits.File(exists=True)
        noise_plot = traits.File(exists=True)
        mean_file = traits.File(exists=True)
        mean_plot = traits.File(exists=True)
        tsnr_file = traits.File(exists=True)
        tsnr_plot = traits.File(exists=True)
        output_path = traits.Directory()

    def _run_interface(self, runtime):

        # Concatenate timeseries frames into 4D image
        img = nib.concat_images(self.inputs.in_files)
        affine, header = img.affine, img.header
        data = img.get_fdata()

        # Load the anatomical template
        anat_img = nib.load(self.inputs.anat_file)

        # Load each run's brain mask and find the intersection
        mask_img_frames = nib.concat_images(self.inputs.mask_files)
        mask_data = mask_img_frames.get_fdata()
        mask = mask_data.all(axis=-1)

        mask_img = self.write_image("mask_file", "mask.nii.gz",
                                    mask.astype(np.int), affine, header)

        # Zero-out data outside the mask
        data[~mask] = 0

        # Jacobian modulate each frame of the template image
        jacobian_img = nib.load(self.inputs.jacobian_file)
        jacobian = jacobian_img.get_fdata()
        data *= jacobian

        # Scale each frame to a common mean value
        target = 100
        scale_value = target / data[mask].mean(axis=0, keepdims=True)
        data = data * scale_value

        # Average over the frames of the template
        data = data.mean(axis=-1)

        # Save out the final template image
        out_img = self.write_image("out_file", "func.nii.gz",
                                   data, affine, header)

        # Load each run's noise mask and find the union
        noise_img_frames = nib.concat_images(self.inputs.noise_files)
        noise_mask = noise_img_frames.get_fdata().any(axis=-1)
        noise_img = self.write_image("noise_file", "noise.nii.gz",
                                     noise_mask, affine, header)

        # Load each run's mean image and take the grand mean
        mean_img = nib.concat_images(self.inputs.mean_files)
        mean = mean_img.get_fdata().mean(axis=-1)
        mean[~mask] = 0
        mean_img = self.write_image("mean_file", "mean.nii.gz",
                                    mean, affine, header)

        # Load each run's tsnr image and take its mean
        tsnr_img = nib.concat_images(self.inputs.tsnr_files)
        tsnr_data = tsnr_img.get_fdata().mean(axis=-1)
        tsnr_data[~mask] = 0
        tsnr_img = self.write_image("tsnr_file", "tsnr.nii.gz",
                                    tsnr_data, affine, header)

        # Prepare QC metadata
        qc_title = " ".join(self.inputs.session_tuple)

        # Write static mosaic image
        m = Mosaic(out_img, title=qc_title)
        self.write_visualization("out_plot", "func.png", m)

        # Make a mosaic of the temporal mean normalized to mean cortical signal
        # TODO copied from timeseries interface!
        seg_img = nib.load(self.inputs.seg_file)
        seg = seg_img.get_fdata().astype(np.int)
        norm_mean = mean / mean[seg == 1].mean()
        mean_m = Mosaic(anat_img, norm_mean, mask_img,
                        show_mask=False, title=qc_title)
        mean_m.plot_overlay("cube:-.15:.5", vmin=0, vmax=2, fmt="d")
        self.write_visualization("mean_plot", "mean.png", mean_m)

        # Make a mosaic of the tSNR
        tsnr_m = Mosaic(anat_img, tsnr_img, mask_img,
                        show_mask=False, title=qc_title)
        tsnr_m.plot_overlay("cube:.25:-.5", vmin=0, vmax=100, fmt="d")
        self.write_visualization("tsnr_plot", "tsnr.png", tsnr_m)

        # Make a QC plot of the run mask
        # TODO should this emphasize areas where runs don't overlap?
        mask_m = Mosaic(anat_img, mask_img, title=qc_title)
        mask_m.plot_mask()
        self.write_visualization("mask_plot", "mask.png", mask_m)

        # Make a QC plot of the session noise mask
        noise_m = Mosaic(anat_img, noise_img, title=qc_title)
        noise_m.plot_mask(alpha=1)
        self.write_visualization("noise_plot", "noise.png", noise_m)

        # Spec out the root path for the template outputs
        experiment = self.inputs.experiment
        subject, session = self.inputs.session_tuple
        output_path = op.join(subject, experiment, "template", session)
        self._results["output_path"] = output_path

        return runtime


# --- Preprocessing quality control


class RealignmentReport(LymanInterface):

    class input_spec(TraitedSpec):
        target_file = traits.File(exists=True)
        realign_params = traits.File(exists=True)
        run_tuple = traits.Tuple(traits.Str(), traits.Str(), traits.Str())

    class output_spec(TraitedSpec):
        params_plot = traits.File(exists=True)
        target_plot = traits.File(exists=True)

    def _run_interface(self, runtime):

        # Load the realignment parameters
        params = np.loadtxt(self.inputs.realign_params)
        cols = ["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"]
        df = pd.DataFrame(params, columns=cols)

        # Plot the motion timeseries
        f = self.plot_motion(df)
        self.write_visualization("params_plot", "mc_params.png", f)

        # Plot the target image
        m = self.plot_target()
        self.write_visualization("target_plot", "mc_target.png", m)

        return runtime

    def plot_motion(self, df):
        """Plot the timecourses of realignment parameters."""
        fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True)

        # Trim off all but the axis name
        def axis(s):
            return s[-1]

        # Plot rotations
        rot_pal = ["#8b312d", "#e33029", "#f06855"]
        rot_df = df.filter(like="rot").apply(np.rad2deg).rename(columns=axis)
        rot_df.plot(ax=axes[0], color=rot_pal, lw=1.5)

        # Plot translations
        trans_pal = ["#355d9a", "#3787c0", "#72acd3"]
        trans_df = df.filter(like="trans").rename(columns=axis)
        trans_df.plot(ax=axes[1], color=trans_pal, lw=1.5)

        # Label the graphs
        axes[0].set_xlim(0, len(df) - 1)
        axes[0].axhline(0, c=".4", ls="--", zorder=1)
        axes[1].axhline(0, c=".4", ls="--", zorder=1)

        for ax in axes:
            ax.legend(ncol=3, loc="best")

        title = " ".join(self.inputs.run_tuple)
        axes[0].set_title(title, size=10)

        axes[0].set_ylabel("Rotations (degrees)")
        axes[1].set_ylabel("Translations (mm)")
        fig.tight_layout()
        return fig

    def plot_target(self):
        """Plot a mosaic of the motion correction target image."""
        title = " ".join(self.inputs.run_tuple)
        return Mosaic(self.inputs.target_file, step=2, title=title)


class AnatRegReport(LymanInterface):

    class input_spec(TraitedSpec):
        subject_id = traits.Str()
        session_tuple = traits.Tuple(traits.Str(), traits.Str())
        data_dir = traits.Directory(exists=True)
        in_file = traits.File(exists=True)
        cost_file = traits.File(exists=True)

    class output_spec(TraitedSpec):
        out_file = traits.File(exists=True)

    def _run_interface(self, runtime):

        # Load the WM segmentation and a brain mask
        mri_dir = op.join(self.inputs.data_dir,
                          self.inputs.subject_id, "mri")

        wm_file = op.join(mri_dir, "wm.mgz")
        wm_data = (nib.load(wm_file).get_fdata() > 0).astype(int)

        aseg_file = op.join(mri_dir, "aseg.mgz")
        mask = (nib.load(aseg_file).get_fdata() > 0).astype(int)

        # Read the final registration cost
        cost = np.loadtxt(self.inputs.cost_file)[0]

        # Make a mosaic of the registration from func to wm seg
        # TODO this should be an OrthoMosaic when that is implemented
        qc_title = " ".join(self.inputs.session_tuple)
        m = Mosaic(self.inputs.in_file, wm_data, mask,
                   step=3, show_mask=False, title=qc_title)
        m.plot_mask_edges()
        if cost is not None:
            cost_text = "Final cost: {:.2f}".format(cost)
            nrow = len(m.axes)
            m.fig.text(.95, .5 * 1 / nrow, cost_text,
                       ha="right", va="center", size=10, color="white")
        self.write_visualization("out_file", "reg.png", m)

        return runtime


class CoregGIF(LymanInterface):

    class input_spec(TraitedSpec):
        run_tuple = traits.Tuple(traits.Str(), traits.Str(), traits.Str())
        in_file = traits.File(exists=True)
        ref_file = traits.File(exists=True)
        out_file = traits.File()

    class output_spec(TraitedSpec):
        out_file = traits.File(exists=True)

    def _run_interface(self, runtime):

        in_img = nib.load(self.inputs.in_file)
        ref_img = nib.load(self.inputs.ref_file)
        out_fname = self.define_output("out_file", self.inputs.out_file)

        if len(ref_img.shape) > 3:
            ref_data = ref_img.get_fdata()[..., 0]
            ref_img = nib.Nifti1Image(ref_data, ref_img.affine, ref_img.header)

        qc_title = " ".join(self.inputs.run_tuple)

        self.write_mosaic_gif(runtime, in_img, ref_img, out_fname,
                              title=qc_title, tight=False)

        return runtime

    def write_mosaic_gif(self, runtime, img1, img2, fname, **kws):

        Mosaic(img1, **kws).savefig("img1.png", close=True)
        Mosaic(img2, **kws).savefig("img2.png", close=True)

        cmdline = ["convert",
                   "-loop", "0",
                   "-delay", "100",
                   "-limit", "thread", "1",
                   "img1.png", "img2.png", fname]

        self.submit_cmdline(runtime, cmdline)
