import os
import os.path as op

import numpy as np
from scipy import ndimage, signal
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib

from nipype import (Workflow, Node, MapNode, JoinNode,
                    IdentityInterface, Function, Rename, DataSink)
from nipype.interfaces.base import (traits, File, TraitedSpec,
                                    InputMultiPath, OutputMultiPath,
                                    isdefined)
from nipype.interfaces import fsl, freesurfer as fs, utility as pipeutil

from .. import signals  # TODO confusingly close to scipy.signal
from ..mosaic import Mosaic, MosaicInterface
from ..carpetplot import CarpetPlot
from ..graphutils import SimpleInterface


def define_preproc_workflow(proj_info, sess_info, exp_info, qc=True):

    # proj_info will be a bunch or other object with data_dir, etc. fields

    # sess info is a nested dictionary:
    # outer keys are subjects
    # inner keys are sessions
    # inner values are lists of runs

    # exp_info is a bunch or dict or other obj with experiment parameters

    # --- Workflow parameterization

    # TODO The iterable creation should be moved to separate functions
    # and tested well, as the logic is fairly complicated. Also there
    # should be a validation of the session info with informative errors

    subject_iterables = [subj for subj in sess_info]

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

    # TODO The comprehension below drives home that `sess_info` is a
    # confusion name when the outer layer of parameterization is subject
    # Need a new name!

    run_iterables = {
        (subj, sess): [(subj, sess, run) for run in sess_info[subj][sess]]
        for subj in sess_info
        for sess in sess_info[subj]
    }
    run_source = Node(IdentityInterface(["subject", "session", "run"]),
                      name="run_source",
                      itersource=("session_source", "session"),
                      iterables=("run", run_iterables))

    session_input = Node(SessionInput(base_directory=proj_info.data_dir,
                                      fm_template=exp_info.fm_template,
                                      phase_encoding=exp_info.phase_encoding),
                         "session_input")

    run_input = Node(RunInput(base_directory=proj_info.data_dir,
                              sb_template=exp_info.sb_template,
                              ts_template=exp_info.ts_template),
                     name="run_input")

    # --- Warpfield estimation using topup

    # Distortion warpfield estimation
    estimate_distortions = Node(fsl.TOPUP(config="b02b0.cnf"),
                                "estimate_distortions")

    # TODO do this in one image with both pairs, and compute correlations
    fieldmap_qc = Node(DistortionGIF(out_file="fieldmap.gif"),
                       "fieldmap_qc")

    unwarp_qc = Node(DistortionGIF(out_file="unwarp.gif"),
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

    fm2anat_qc = Node(AnatRegReport(out_file="reg.png"), "fm2anat_qc")

    # --- Definition of common cross-session space (template space)

    define_template = JoinNode(DefineTemplateSpace(),
                               name="define_template",
                               joinsource="session_source",
                               joinfield=["session_info",
                                          "in_matrices", "in_volumes"])

    func2anat_qc = Node(AnatRegReport(out_file="reg.png"),
                        "func2anat_qc")

    # --- Segementation of anatomical tissue in functional space

    anat_segment = Node(AnatomicalSegmentation(), "anat_segment")

    # --- Associate template-space transforms with data from correct session

    # The logic here is a little complex. The template creation node collapses
    # the session-wise iterables and returns list of files that then need
    # to hook back into the iterable parameterization so that they can be
    # associated with data from the correct session when they are applied.

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

    # TODO everything from here on could probably be done using Python
    # in a single interface to simplify the logic of the workflow

    def flatten_file_list(in_files):
        out_files = [item for sublist in in_files for item in sublist]
        return out_files

    combine_template = JoinNode(Function("in_files", "out_files",
                                         flatten_file_list),
                                name="combine_template",
                                joinsource="session_source",
                                joinfield=["in_files"])

    merge_template = Node(fsl.Merge(dimension="t"), name="merge_template")

    # TODO scale the templates somehow before averaging?

    average_template = Node(fsl.MeanImage(), "average_template")

    mask_template = Node(fsl.ApplyMask(out_file="func.nii.gz"),
                         "mask_template")

    static_template_qc = Node(MosaicInterface(out_file="func.png"),
                              "static_template_qc")

    dynamic_template_qc = Node(FrameGIF(out_file="func.gif", delay=20),
                               "dynamic_template_qc")

    # --- Motion correction of time series to SBRef (with distortions)

    ts2sb = Node(fsl.MCFLIRT(save_mats=True, save_plots=True),
                 "ts2sb")

    rename_params = Node(Rename("mc.txt"), "rename_params")

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
    merge_ts = Node(fsl.Merge(dimension="t"), "merge_ts")

    # Mask, scale, and detrend the timeseries; make a gif
    finalize_ts = Node(FinalizeTimeseries(), "finalize_ts")

    # Descriptive timeseries statistics
    ts_stats = Node(TimeSeriesStats(), "ts_stats")

    # Generate a mask of noise voxels
    # TODO parameterize the neighborhood and/or threshold
    # also maybe this should be move downstream into modeling
    # so those can be changed without rerunning preproc...
    identify_noise = Node(IdentifyNoiseVoxels(), "identify_noise")

    # QC plots for the preprocessed timeseries
    static_ts_qc = Node(CarpetPlot(out_file="func.png"), "static_ts_qc")

    # --- Workflow ouptut

    output_dir = op.join(proj_info.analysis_dir, exp_info.name)

    def define_timeseries_container(subject, session, run):
        return "{}/timeseries/{}_{}".format(subject, session, run)

    timeseries_container = Node(Function(["subject", "session", "run"],
                                         "path", define_timeseries_container),
                                "timeseries_container")

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

        # --- SE-EPI fieldmap processing and template creation

        # Phase-encode distortion estimation

        (session_input, estimate_distortions,
            [("fm", "in_file"),
             ("phase_encoding", "encoding_direction"),
             ("readout_times", "readout_times")]),
        (estimate_distortions, select_warp,
            [("out_warps", "inlist")]),
        (select_warp, mask_distortions,
            [("out", "in_file")]),
        (estimate_distortions, average_fm,
            [("out_corrected", "in_file")]),

        # Registration of corrected SE-EPI to anatomy

        (session_input, fm2anat,
            [("subject", "subject_id")]),
        (average_fm, fm2anat,
            [("out_file", "source_file")]),

        # Creation of cross-session subject-specific template

        (session_source, define_template,
            [("session", "session_info")]),
        (session_input, define_template,
            [("fm", "in_volumes")]),
        (fm2anat, define_template,
            [("out_fsl_file", "in_matrices")]),
        (define_template, select_sesswise,
            [("out_matrices", "in_matrices"),
             ("out_template", "in_templates"),
             ("session_info", "session_info")]),
        (session_input, select_sesswise,
            [("subject", "subject"),
             ("session", "session")]),
        (session_input, split_fm,
            [("fm", "in_file")]),
        (split_fm, restore_fm,
            [("out_files", "in_file")]),
        (estimate_distortions, restore_fm,
            [("out_mats", "premat"),
             ("out_warps", "field_file")]),
        (define_template, restore_fm,
            [("out_template", "ref_file")]),
        (select_sesswise, restore_fm,
            [("out_matrix", "postmat")]),
        (restore_fm, combine_template,
            [("out_file", "in_files")]),
        (combine_template, merge_template,
            [("out_files", "in_files")]),
        (merge_template, average_template,
            [("merged_file", "in_file")]),
        (average_template, mask_template,
            [("out_file", "in_file")]),
        (anat_segment, mask_template,
            [("mask_file", "mask_file")]),

        # Segementation of anatomical tissue in functional space

        (define_template, anat_segment,
            [("subject_id", "subject_id"),
             ("reg_file", "reg_file")]),
        (average_template, anat_segment,
            [("out_file", "template_file")]),

        # --- Time series spatial processing

        # Registration of each frame to SBRef image

        (run_input, ts2sb,
            [("ts", "in_file"),
             ("sb", "ref_file")]),
        (ts2sb, rename_params,
            [("par_file", "in_file")]),

        # Registration of SBRef volume to SE-EPI fieldmap

        (run_input, sb2fm,
            [("sb", "in_file")]),
        (session_input, sb2fm,
            [("fm", "reference")]),
        (mask_distortions, sb2fm,
            [("out_file", "ref_weight")]),

        # Single-interpolation spatial realignment and unwarping

        (ts2sb, combine_rigids,
            [("mat_file", "in_file")]),
        (sb2fm, combine_rigids,
            [("out_matrix_file", "in_file2")]),
        (run_input, split_ts,
            [("ts", "in_file")]),
        (define_template, select_runwise,
            [("out_matrices", "in_matrices"),
             ("session_info", "session_info")]),
        (run_input, select_runwise,
            [("subject", "subject"),
             ("session", "session")]),
        (split_ts, restore_ts_frames,
            [("out_files", "in_file")]),
        (combine_rigids, restore_ts_frames,
            [("out_file", "premat")]),
        (select_warp, restore_ts_frames,
            [("out", "field_file")]),
        (define_template, restore_ts_frames,
            [("out_template", "ref_file")]),
        (select_runwise, restore_ts_frames,
            [("out_matrix", "postmat")]),
        (restore_ts_frames, merge_ts,
            [("out_file", "in_files")]),
        (merge_ts, finalize_ts,
            [("merged_file", "in_file")]),
        (anat_segment, finalize_ts,
            [("mask_file", "mask_file")]),
        (finalize_ts, ts_stats,
            [("out_file", "in_file")]),
        (anat_segment, ts_stats,
            [("mask_file", "mask_file")]),
        (finalize_ts, identify_noise,
            [("out_file", "in_file")]),
        (anat_segment, identify_noise,
            [("seg_file", "seg_file")]),

        # --- Persistent data storage

        # Ouputs associated with the subject-specific template

        (session_input, template_container,
            [("subject", "subject")]),
        (template_container, template_output,
            [("path", "container")]),
        (mask_template, template_output,
            [("out_file", "@template")]),
        (define_template, template_output,
            [("reg_file", "@reg")]),
        (anat_segment, template_output,
            [("seg_file", "@seg"),
             ("mask_file", "@mask"),
             ("anat_file", "@anat"),
             ("surf_file", "@surf")]),

        # Ouputs associated with each scanner run

        (run_input, timeseries_container,
            [("subject", "subject"),
             ("session", "session"),
             ("run", "run")]),
        (timeseries_container, timeseries_output,
            [("path", "container")]),
        (finalize_ts, timeseries_output,
            [("out_file", "@timeseries")]),
        (rename_params, timeseries_output,
            [("out_file", "@mc_params")]),
        (ts_stats, timeseries_output,
            [("mean_file", "@ts_mean"),
             ("tsnr_file", "@ts_tsnr")]),
        (identify_noise, timeseries_output,
            [("out_file", "@noise")]),

    ]
    workflow.connect(processing_edges)

    # Optionally connect QC nodes

    qc_edges = [

        # Phase-encode distortion estimation

        (session_input, fieldmap_qc,
            [("fm", "in_file"),
             ("session", "session"),
             ("phase_encoding", "phase_encoding")]),
        (session_input, unwarp_qc,
            [("session", "session"),
             ("phase_encoding", "phase_encoding")]),
        (estimate_distortions, unwarp_qc,
            [("out_corrected", "in_file")]),

        # Registration of corrected SE-EPI to anatomy

        (session_input, fm2anat_qc,
            [("subject", "subject_id"),
             ("session", "session")]),
        (average_fm, fm2anat_qc,
            [("out_file", "in_file")]),
        (fm2anat, fm2anat_qc,
            [("out_reg_file", "reg_file"),
             ("min_cost_file", "cost_file")]),

        # Creation of cross-session subject-specific template

        (session_input, func2anat_qc,
            [("subject", "subject_id")]),
        (define_template, func2anat_qc,
            [("reg_file", "reg_file")]),
        (average_template, func2anat_qc,
            [("out_file", "in_file")]),
        (merge_template, dynamic_template_qc,
            [("merged_file", "in_file")]),
        (mask_template, static_template_qc,
            [("out_file", "anat_file")]),

        # Registration of each frame to SBRef image

        (run_input, realign_qc,
            [("sb", "target_file")]),
        (ts2sb, realign_qc,
            [("par_file", "realign_params")]),

        # Registration of SBRef volume to SE-EPI fieldmap

        (sb2fm, sb2fm_qc,
            [("out_file", "in_file")]),
        (session_input, sb2fm_qc,
            [("fm", "ref_file")]),

        # Single-interpolation spatial realignment and unwarping

        (finalize_ts, static_ts_qc,
            [("out_file", "in_file")]),
        (anat_segment, static_ts_qc,
            [("seg_file", "seg_file")]),
        (ts2sb, static_ts_qc,
            [("par_file", "mc_file")]),

        # Ouputs associated with the subject-specific template

        (anat_segment, template_output,
            [("seg_plot", "qc.@seg_plot"),
             ("mask_plot", "qc.@mask_plot"),
             ("anat_plot", "qc.@anat_plot"),
             ("surf_plot", "qc.@surf_plot")]),
        (func2anat_qc, template_output,
            [("out_file", "qc.@func2anat_plot")]),
        (static_template_qc, template_output,
            [("out_file", "qc.@template_png")]),
        (dynamic_template_qc, template_output,
            [("out_file", "qc.@template_gif")]),
        (fieldmap_qc, template_output,
            [("out_file", "qc.sessions.@fieldmap_gif")]),
        (unwarp_qc, template_output,
            [("out_file", "qc.sessions.@unwarp_gif")]),
        (fm2anat_qc, template_output,
            [("out_file", "qc.sessions.@fm2anat_plot")]),

        # Ouputs associated with each scanner run

        (run_input, timeseries_output,
            [("ts_plot", "qc.@raw_gif")]),
        (sb2fm_qc, timeseries_output,
            [("out_file", "qc.@sb2fm_gif")]),
        (realign_qc, timeseries_output,
            [("params_plot", "qc.@params_plot"),
             ("target_plot", "qc.@target_plot")]),
        (static_ts_qc, timeseries_output,
            [("out_file", "qc.@ts_png")]),
        (finalize_ts, timeseries_output,
            [("out_plot", "qc.@ts_gif")]),
        (ts_stats, timeseries_output,
            [("mean_plot", "qc.@ts_mean_plot"),
             ("tsnr_plot", "qc.@ts_tsnr_plot")]),
        (identify_noise, timeseries_output,
            [("out_plot", "qc.@noise_plot")]),

    ]
    if qc:
        workflow.connect(qc_edges)

    return workflow


# =========================================================================== #
# Custom processing nodes
# =========================================================================== #

# ---- Quality control mixins

class TimeSeriesGIF(object):

    def write_time_series_gif(self, runtime, img, fname):

        os.mkdir("png")

        nx, ny, nz, nt = img.shape
        tr = img.header.get_zooms()[3]
        delay = tr * 1000 / 75

        width = 5
        height = width * max([nx, ny, nz]) / sum([nz, ny, nz])

        f, axes = plt.subplots(ncols=3, figsize=(width, height))
        for ax in axes:
            ax.set_axis_off()

        data = img.get_data()
        vmin, vmax = np.percentile(data, [2, 98])

        kws = dict(vmin=vmin, vmax=vmax, cmap="gray")
        im_x = axes[0].imshow(np.zeros((nz, ny)), **kws)
        im_y = axes[1].imshow(np.zeros((nz, nx)), **kws)
        im_z = axes[2].imshow(np.zeros((ny, nx)), **kws)

        f.subplots_adjust(0, 0, 1, 1, 0, 0)

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
            pngs.append(frame_png)
            f.savefig(frame_png,
                      facecolor="0", edgecolor="0")

        cmdline = ["convert", "-loop", "0", "-delay", str(delay)]
        cmdline.extend(pngs)
        cmdline.append(fname)

        self.submit_cmdline(runtime, cmdline)


# ---- Data input and pre-preprocessing


class SessionInput(SimpleInterface):

    class input_spec(TraitedSpec):
        session = traits.Tuple()
        base_directory = traits.Str()
        fm_template = traits.Str()
        phase_encoding = traits.Str()

    class output_spec(TraitedSpec):
        fm = traits.File(exists=True)
        phase_encoding = traits.List(traits.Str())
        readout_times = traits.List(traits.Float())
        session_key = traits.Tuple()
        subject = traits.Str()
        session = traits.Str()

    def _run_interface(self, runtime):

        # Determine the phase encoding directions
        pe = self.inputs.phase_encoding
        if pe == "ap":
            pos_pe, neg_pe = "ap", "pa"
        elif pe == "pa":
            pos_pe, neg_pe = "pa", "ap"
        else:
            raise ValueError("Phase encoding must be 'ap' or 'pa'")

        # Determine the parameters
        subject, session = self.inputs.session
        self._results["session_key"] = self.inputs.session
        self._results["subject"] = str(subject)
        self._results["session"] = str(session)

        # Spec out full paths to the pair of fieldmap files
        keys = dict(subject=subject, session=session)
        template = self.inputs.fm_template
        pos_fname = op.join(self.inputs.base_directory,
                            template.format(encoding=pos_pe, **keys))
        neg_fname = op.join(self.inputs.base_directory,
                            template.format(encoding=neg_pe, **keys))

        # Load the two images in canonical orientation
        pos_img = nib.as_closest_canonical(nib.load(pos_fname))
        neg_img = nib.as_closest_canonical(nib.load(neg_fname))
        affine, header = pos_img.affine, pos_img.header

        # Concatenate the images into a single volume
        pos_data = pos_img.get_data()
        neg_data = neg_img.get_data()
        data = np.concatenate([pos_data, neg_data], axis=-1)
        assert len(data.shape) == 4

        # Convert image datatype to float
        header.set_data_dtype(np.float32)

        # Write out a 4D file
        fname = self.define_output("fm", "fieldmap.nii.gz")
        img = nib.Nifti1Image(data, affine, header)
        img.to_filename(fname)

        # Define phase encoding and readout times for TOPUP
        pe_dir = ["y"] * pos_img.shape[-1] + ["y-"] * neg_img.shape[-1]
        readout_times = [1 for _ in pe_dir]
        self._results["phase_encoding"] = pe_dir
        self._results["readout_times"] = readout_times

        return runtime


class RunInput(SimpleInterface, TimeSeriesGIF):

    class input_spec(TraitedSpec):
        run = traits.Tuple()
        base_directory = traits.Str()
        sb_template = traits.Str()
        ts_template = traits.Str()
        crop_frames = traits.Int(0, usedefault=True)

    class output_spec(TraitedSpec):
        sb = traits.File(exists=True)
        ts = traits.File(exists=True)
        ts_plot = traits.File(exists=True)
        subject = traits.Str()
        session = traits.Str()
        run = traits.Str()

    def _run_interface(self, runtime):

        # Determine the parameters
        subject, session, run = self.inputs.run
        self._results["subject"] = subject
        self._results["session"] = session
        self._results["run"] = run

        # Spec out paths to the input files
        keys = dict(subject=subject, session=session, run=run)
        sb_fname = op.join(self.inputs.base_directory,
                           self.inputs.sb_template.format(**keys))
        ts_fname = op.join(self.inputs.base_directory,
                           self.inputs.ts_template.format(**keys))

        # Load the input images in canonical orientation
        sb_img = nib.as_closest_canonical(nib.load(sb_fname))
        ts_img = nib.as_closest_canonical(nib.load(ts_fname))

        # Convert image datatypes to float
        sb_img.set_data_dtype(np.float32)
        ts_img.set_data_dtype(np.float32)

        # Optionally crop the first n frames of the timeseries
        if self.inputs.crop_frames > 0:
            ts_data = ts_img.get_data()
            ts_data = ts_data[..., self.inputs.crop_frames:]
            ts_img = nib.Nifti1Image(ts_data, ts_img.affine, ts_img.header)

        # Write out the new images
        self.save_image(sb_img, "sb", "sb.nii.gz")
        self.save_image(ts_img, "ts", "ts.nii.gz")

        # Make a GIF movie of the raw timeseries
        out_plot = self.define_output("out_plot", "func.gif")
        self.write_time_series_gif(runtime, ts_img, out_plot)

        return runtime


# --- Preprocessing operations


class FinalizeTimeseries(SimpleInterface, TimeSeriesGIF):

    class input_spec(TraitedSpec):
        in_file = traits.File(exists=True)
        mask_file = traits.File(exists=True)

    class output_spec(TraitedSpec):
        out_file = traits.File(exists=True)
        out_plot = traits.File(exists=True)

    def _run_interface(self, runtime):

        img = nib.load(self.inputs.in_file)
        data = img.get_data()

        mask = nib.load(self.inputs.mask_file).get_data().astype(np.bool)

        # Zero-out data outside the mask
        data[~mask] = 0

        # Scale the timeseries for cross-run intensity normalization
        target = 10000
        data = self.scale_timeseries(data, mask, target)
        mean = data.mean(axis=-1, keepdims=True)

        # Detrend the timeseries
        data[mask] = signal.detrend(data[mask])

        # Add the original temporal mean back in
        data += mean

        # Save out the final time series
        out_file = op.abspath("func.nii.gz")
        self._results["out_file"] = out_file

        out_img = nib.Nifti1Image(data, img.affine, img.header)
        out_img.to_filename(out_file)

        # Make a GIF movie of the final timeseries
        out_plot = self.define_output("out_plot", "func.gif")
        self.write_time_series_gif(runtime, out_img, out_plot)

        # TODO move much more into here!
        # TODO also make the carpet plot here?
        # Just to save an additional loading of the time series
        # Although also we would need the moco params to do so,
        # and thus could rename / add a header to obviate that extra node

        return runtime

    def scale_timeseries(self, data, mask, target):
        """Make scale timeseries across four dimensions to a target."""
        stat_value = data[mask].mean()
        scale_value = target / stat_value
        scaled_data = data * scale_value
        return scaled_data


class IdentifyNoiseVoxels(SimpleInterface):

    class input_spec(TraitedSpec):
        in_file = traits.File(exists=True)
        seg_file = traits.File(exists=True)
        neighborhood = traits.Float(5, usedefault=True)
        threshold = traits.Float(1.5, usedefault=True)
        subcortical = traits.Bool(True, usedefault=True)

    class output_spec(TraitedSpec):
        out_file = traits.File(exists=True)
        out_plot = traits.File(exists=True)

    def _run_interface(self, runtime):

        ts_img = nib.load(self.inputs.in_file)

        seg_img = nib.load(self.inputs.seg_file)
        seg = seg_img.get_data()
        if self.inputs.subcortical:
            mask = (seg > 0) & (seg < 5)
        else:
            mask = seg == 1
        mask_img = nib.Nifti1Image(mask, seg_img.affine, seg_img.header)

        noise_img = signals.identify_noisy_voxels(
            ts_img, mask_img,
            neighborhood=self.inputs.neighborhood,
            threshold=self.inputs.threshold,
            detrend=False
        )

        out_file = op.abspath("noise.nii.gz")
        self._results["out_file"] = out_file
        noise_img.to_filename(out_file)

        data = ts_img.get_data()
        mean = data.mean(axis=-1)
        mean_img = nib.Nifti1Image(mean, ts_img.affine, ts_img.header)

        out_plot = op.abspath("noise.png")
        self._results["out_plot"] = out_plot

        m = Mosaic(mean_img, noise_img, seg > 0, show_mask=False)
        m.plot_mask(alpha=1)
        m.savefig(out_plot)
        m.close()

        return runtime


class TimeSeriesStats(SimpleInterface):

    class input_spec(TraitedSpec):
        in_file = traits.File(exists=True)
        mask_file = traits.File(exists=True)

    class output_spec(TraitedSpec):
        mean_file = traits.File(exists=True)
        tsnr_file = traits.File(exists=True)
        mean_plot = traits.File(exists=True)
        tsnr_plot = traits.File(exists=True)

    def _run_interface(self, runtime):

        ts_img = nib.load(self.inputs.in_file)
        affine, header = ts_img.affine, ts_img.header
        data = ts_img.get_data()

        mask_img = nib.load(self.inputs.mask_file)
        mask = mask_img.get_data().astype(np.bool)

        mean = data.mean(axis=-1)
        data[mask] = signal.detrend(data[mask])
        sd = data.std(axis=-1)

        with np.errstate(all="ignore"):
            tsnr = mean / sd
            tsnr[~mask] = 0

        mean_file = op.abspath("mean.nii.gz")
        self._results["mean_file"] = mean_file
        mean_img = nib.Nifti1Image(mean, affine, header)
        mean_img.to_filename(mean_file)

        tsnr_file = op.abspath("tsnr.nii.gz")
        self._results["tsnr_file"] = tsnr_file
        tsnr_img = nib.Nifti1Image(tsnr, affine, header)
        tsnr_img.to_filename(tsnr_file)

        # TODO normalize by within cortex mean?
        # TODO use cubehelix colormaps

        mean_plot = op.abspath("mean.png")
        self._results["mean_plot"] = mean_plot
        norm_mean = mean / mean.max()
        mean_m = Mosaic(mean_img, norm_mean, mask_img,
                        tight=True, show_mask=False)
        mean_m.plot_overlay("magma", vmin=0, vmax=1, fmt="d")
        mean_m.savefig(mean_plot)
        mean_m.close()

        tsnr_plot = op.abspath("tsnr.png")
        self._results["tsnr_plot"] = tsnr_plot
        tsnr_m = Mosaic(mean_img, tsnr, mask_img,
                        tight=True, show_mask=False)
        tsnr_m.plot_overlay("viridis", vmin=20, vmax=100, fmt="d")
        tsnr_m.savefig(tsnr_plot)
        tsnr_m.close()

        return runtime


class AnatomicalSegmentation(SimpleInterface):

    class input_spec(TraitedSpec):
        subject_id = traits.Str()
        template_file = traits.File(exists=True)
        reg_file = traits.File(exists=True)

    class output_spec(TraitedSpec):
        seg_file = traits.File(exists=True)
        seg_plot = traits.File(exists=True)
        anat_file = traits.File(exists=True)
        anat_plot = traits.File(exists=True)
        mask_file = traits.File(exists=True)
        mask_plot = traits.File(exists=True)
        surf_file = traits.File(exists=True)
        surf_plot = traits.File(exists=True)

    def _run_interface(self, runtime):

        # Define the template space geometry

        template_img = nib.load(self.inputs.template_file)
        affine, header = template_img.affine, template_img.header

        # --- Coarse segmentation into anatomical components

        # Transform the wmparc image into functional space

        fs_fname = "wmparc.nii.gz"
        cmdline = ["mri_vol2vol",
                   "--nearest",
                   "--inv",
                   "--mov", self.inputs.template_file,
                   "--fstarg", "wmparc.mgz",
                   "--reg", self.inputs.reg_file,
                   "--o", fs_fname]

        self.submit_cmdline(runtime, cmdline)

        # Load the template-space wmparc and reclassify voxels

        fs_img = nib.load(fs_fname)
        fs_data = fs_img.get_data()

        seg_data = np.zeros_like(fs_data)

        seg_ids = [
            np.arange(1000, 3000),  # Cortical gray matter
            [10, 11, 12, 13, 17, 18, 49, 50, 51, 52, 53, 54],  # Subcortical
            [16, 28, 60],  # Brain stem and ventral diencephalon
            [8, 47],  # Cerebellar gray matter
            np.arange(3000, 5000),  # Superficial ("cortical") white matter
            [5001, 5002],  # Deep white matter
            [7, 46],  # Cerebellar white matter
            [4, 43, 31, 63],  # Lateral ventricle CSF
        ]

        for seg_val, id_vals in enumerate(seg_ids, 1):
            mask = np.in1d(fs_data.flat, id_vals).reshape(seg_data.shape)
            seg_data[mask] = seg_val

        seg_file = op.abspath("seg.nii.gz")
        seg_img = nib.Nifti1Image(seg_data, affine, header)
        seg_img.to_filename(seg_file)
        self._results["seg_file"] = seg_file

        # --- Whole brain mask

        # Binarize the segmentation and dilate to generate a brain mask

        brainmask = seg_data > 0
        vox_mm = np.mean(header.get_zooms()[:3])
        iterations = int(np.ceil(4 / vox_mm))
        brainmask = ndimage.binary_dilation(brainmask, iterations=iterations)
        brainmask = ndimage.binary_closing(brainmask)
        brainmask = ndimage.binary_fill_holes(brainmask)

        mask_file = op.abspath("mask.nii.gz")
        mask_img = nib.Nifti1Image(brainmask, affine, header)
        mask_img.to_filename(mask_file)
        self._results["mask_file"] = mask_file

        # --- T1w anatomical image in functional space

        anat_file = op.abspath("anat.nii.gz")
        cmdline = ["mri_vol2vol",
                   "--cubic",
                   "--inv",
                   "--mov", self.inputs.template_file,
                   "--fstarg", "brain.finalsurfs.mgz",
                   "--reg", self.inputs.reg_file,
                   "--o", anat_file]

        self.submit_cmdline(runtime, cmdline, anat_file=anat_file)

        # --- Surface vertex mapping

        hemi_files = []
        for hemi in ["lh", "rh"]:
            hemi_mask = "{}.ribbon.nii.gz".format(hemi)
            hemi_file = "{}.surf.nii.gz".format(hemi)
            cmdline = ["mri_surf2vol",
                       "--template", self.inputs.template_file,
                       "--reg", self.inputs.reg_file,
                       "--surf", "graymid",
                       "--hemi", hemi,
                       "--mkmask",
                       "--o", hemi_mask,
                       "--vtxvol", hemi_file]

            self.submit_cmdline(runtime, cmdline)
            hemi_files.append(hemi_file)

        hemi_data = [nib.load(f).get_data() for f in hemi_files]
        surf = np.stack(hemi_data, axis=-1)

        surf_file = op.abspath("surf.nii.gz")
        self._results["surf_file"] = surf_file
        surf_img = nib.Nifti1Image(surf, affine, header)
        surf_img.to_filename(surf_file)

        # --- Generate QC mosaics

        # Anatomical segmentation

        seg_plot = op.abspath("seg.png")
        self._results["seg_plot"] = seg_plot
        seg_cmap = mpl.colors.ListedColormap(
            ['#3b5f8a', '#5b81b1', '#7ea3d1', '#a8c5e9',
             '#ce8186', '#b8676d', '#9b4e53', '#fbdd7a']
         )
        m_seg = Mosaic(template_img, seg_img, mask_img,
                       step=2, tight=True, show_mask=False)
        m_seg.plot_overlay(seg_cmap, 1, 8, thresh=.5, fmt=None)
        m_seg.savefig(seg_plot)
        m_seg.close()

        # Brain mask

        mask_plot = op.abspath("mask.png")
        self._results["mask_plot"] = mask_plot
        m_mask = Mosaic(template_img, mask_img, mask_img,
                        step=2, tight=True, show_mask=False)
        m_mask.plot_mask()
        m_mask.savefig(mask_plot)
        m_mask.close()

        # Anatomical image

        anat_plot = op.abspath("anat.png")
        self._results["anat_plot"] = anat_plot
        m_mask = Mosaic(anat_file, mask=mask_img,
                        step=2, tight=True, show_mask=False)
        m_mask.savefig(anat_plot)
        m_mask.close()

        # Surface ribbon

        surf_plot = op.abspath("surf.png")
        self._results["surf_plot"] = surf_plot
        ribbon = np.zeros(template_img.shape)
        ribbon[surf[..., 0] > 0] = 1
        ribbon[surf[..., 1] > 0] = 2
        ribbon_cmap = mpl.colors.ListedColormap(["#5ebe82", "#ec966f"])
        m_surf = Mosaic(template_img, ribbon, mask_img,
                        step=2, tight=True, show_mask=False)
        m_surf.plot_overlay(ribbon_cmap, 1, 2, thresh=.5, fmt=None)
        m_surf.savefig(surf_plot)
        m_surf.close()

        return runtime


class DefineTemplateSpace(SimpleInterface):

    class input_spec(TraitedSpec):
        session_info = traits.List(traits.Tuple())
        in_matrices = InputMultiPath(File(exists=True))
        in_volumes = InputMultiPath(File(exists=True))

    class output_spec(TraitedSpec):
        subject_id = traits.Str()
        session_info = traits.List(traits.Tuple())
        out_template = File(exists=True)
        reg_file = File(exists=True)
        out_matrices = OutputMultiPath(File(exists=True))

    def _run_interface(self, runtime):

        subjects_dir = os.environ["SUBJECTS_DIR"]

        subject_ids = set([s for s, _ in self.inputs.session_info])
        assert len(subject_ids) == 1
        subj = subject_ids.pop()

        self._results["subject_id"] = subj
        self._results["session_info"] = self.inputs.session_info

        # -- Convert the anatomical image to nifti
        anat_file = "orig.nii.gz"
        cmdline = ["mri_convert",
                   op.join(subjects_dir, subj, "mri/orig.mgz"),
                   anat_file]

        self.submit_cmdline(runtime, cmdline)

        # -- Compute the intermediate transform
        midtrans_file = "anat2func.mat"
        cmdline = ["midtrans",
                   "--template=" + anat_file,
                   "--separate=se2template_",
                   "--out=" + midtrans_file]
        cmdline.extend(self.inputs.in_matrices)
        out_matrices = [
            op.abspath("se2template_{:04d}.mat".format(i))
            for i, _ in enumerate(self.inputs.in_matrices, 1)
        ]

        self.submit_cmdline(runtime, cmdline, out_matrices=out_matrices)

        # -- Invert the anat2temp transformation
        flirt_file = "func2anat.mat"
        cmdline = ["convert_xfm",
                   "-omat", flirt_file,
                   "-inverse",
                   midtrans_file]

        self.submit_cmdline(runtime, cmdline)

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
        reg_file = op.abspath("reg.dat")
        cmdline = ["tkregister2",
                   "--s", subj,
                   "--mov", "template_space.nii.gz",
                   "--fsl", flirt_file,
                   "--reg", reg_file,
                   "--noedit"]

        self.submit_cmdline(runtime, cmdline, reg_file=reg_file)

        return runtime


# --- Preprocessing quality control


class RealignmentReport(SimpleInterface):

    class input_spec(TraitedSpec):
        target_file = File(exists=True)
        realign_params = File(exists=True)

    class output_spec(TraitedSpec):
        params_plot = File(exists=True)
        target_plot = File(exists=True)

    def _run_interface(self, runtime):

        # Load the realignment parameters
        params = np.loadtxt(self.inputs.realign_params)
        cols = ["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"]
        df = pd.DataFrame(params, columns=cols)

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
        m.plot_mask_edges()
        if cost is not None:
            m.fig.suptitle("Final cost: {:.2f}".format(cost),
                           size=10, color="white")
        m.savefig(fname)
        m.close()

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
        lims = 0, np.percentile(data, 98)

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
                              slice_dir="sag", tight=False, anat_lims=lims)

        self._results["out_file"] = op.abspath(fname)

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
        data = img.get_data()

        lims = 0, np.percentile(data, 98)

        assert len(img.shape) == 4
        n_frames = img.shape[-1]

        frame_pngs = []

        for i in range(n_frames):

            png_fname = "frame{:02d}.png".format(i)
            frame_pngs.append(png_fname)

            vol_data = data[..., i]
            vol = nib.Nifti1Image(vol_data, img.affine, img.header)
            m = Mosaic(vol, tight=False, step=2, anat_lims=lims)
            m.savefig(png_fname)
            m.close()

        out_file = op.abspath(self.inputs.out_file)
        cmdline = ["convert", "-loop", "0", "-delay", str(self.inputs.delay)]
        cmdline.extend(frame_pngs)
        cmdline.append(out_file)

        self.submit_cmdline(runtime, cmdline, out_file=out_file)

        return runtime
