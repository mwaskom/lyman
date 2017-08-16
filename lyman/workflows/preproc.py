import os
import os.path as op

import numpy as np
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib

from nipype import (Workflow, Node, MapNode,
                    IdentityInterface, Function, DataSink)
from nipype.interfaces.base import traits, TraitedSpec
from nipype.interfaces import fsl

from .. import signals  # TODO confusingly close to scipy.signal
from ..mosaic import Mosaic
from ..carpetplot import CarpetPlot
from ..graphutils import SimpleInterface


def define_preproc_workflow(proj_info, subjects, exp_info, qc=True):

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
    scan_info = proj_info.scan_info

    subject_iterables = subjects

    subject_source = Node(IdentityInterface(["subject"]),
                          name="subject_source",
                          iterables=("subject", subject_iterables))

    session_iterables = {
        subj: [(subj, sess) for sess in scan_info[subj]
               if exp_info.name in scan_info[subj][sess]]
        for subj in subjects
    }
    session_source = Node(IdentityInterface(["subject", "session"]),
                          name="session_source",
                          itersource=("subject_source", "subject"),
                          iterables=("session", session_iterables))

    # TODO The comprehension below drives home that `sess_info` is a
    # confusion name when the outer layer of parameterization is subject
    # Need a new name!

    run_iterables = {
        (subj, sess): [(subj, sess, run)
                       for run in scan_info[subj][sess][exp_info.name]]
        for subj in subjects
        for sess in scan_info[subj]
        if exp_info.name in scan_info[subj][sess]
    }
    run_source = Node(IdentityInterface(["subject", "session", "run"]),
                      name="run_source",
                      itersource=("session_source", "session"),
                      iterables=("run", run_iterables))

    session_input = Node(SessionInput(base_directory=proj_info.data_dir,
                                      fm_template=proj_info.fm_template,
                                      phase_encoding=proj_info.phase_encoding),
                         "session_input")

    run_input = Node(RunInput(base_directory=proj_info.data_dir,
                              sb_template=proj_info.sb_template,
                              ts_template=proj_info.ts_template),
                     name="run_input")

    # Define a mask of areas with large distortions
    # TODO this is unconnected, do we still want to do it?
    thresh_ops = "-abs -thr 4 -Tmax -binv"
    mask_distortions = Node(fsl.ImageMaths(op_string=thresh_ops),
                            "mask_distortions")

    # --- Registration of SBRef to SE-EPI (with distortions)

    sb2fm = Node(fsl.FLIRT(dof=6, interp="spline"), "sb2fm")

    sb2fm_qc = Node(CoregGIF(out_file="coreg.gif"), "sb2fm_qc")

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

    # Perform final preprocessing operations on timeseries
    finalize_ts = Node(FinalizeTimeseries(), "finalize_ts")

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

        # --- Time series spatial processing

        # Registration of each frame to SBRef image

        (run_input, ts2sb,
            [("ts", "in_file"),
             ("sb", "ref_file")]),
        (ts2sb, finalize_ts,
            [("par_file", "mc_file")]),

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
        (run_input, select_runwise,
            [("subject", "subject"),
             ("session", "session")]),
        (split_ts, restore_ts_frames,
            [("out_files", "in_file")]),
        (combine_rigids, restore_ts_frames,
            [("out_file", "premat")]),
        (select_runwise, restore_ts_frames,
            [("out_matrix", "postmat")]),
        (restore_ts_frames, finalize_ts,
            [("out_file", "in_files")]),

        # --- Persistent data storage

        # Ouputs associated with each scanner run

        (run_input, timeseries_container,
            [("subject", "subject"),
             ("session", "session"),
             ("run", "run")]),
        (timeseries_container, timeseries_output,
            [("path", "container")]),
        (finalize_ts, timeseries_output,
            [("out_file", "@func"),
             ("mean_file", "@mean"),
             ("tsnr_file", "@tsnr"),
             ("noise_file", "@noise"),
             ("mc_file", "@mc")]),

    ]
    workflow.connect(processing_edges)

    # Optionally connect QC nodes

    qc_edges = [

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

        # Ouputs associated with each scanner run

        (run_input, timeseries_output,
            [("ts_plot", "qc.@raw_gif")]),
        (sb2fm_qc, timeseries_output,
            [("out_file", "qc.@sb2fm_gif")]),
        (realign_qc, timeseries_output,
            [("params_plot", "qc.@params_plot"),
             ("target_plot", "qc.@target_plot")]),
        (finalize_ts, timeseries_output,
            [("out_gif", "qc.@ts_gif"),
             ("out_png", "qc.@ts_png"),
             ("mean_plot", "qc.@ts_mean_plot"),
             ("tsnr_plot", "qc.@ts_tsnr_plot"),
             ("noise_plot", "qc.@noise_plot")]),

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
        out_plot = self.define_output("ts_plot", "raw.gif")
        self.write_time_series_gif(runtime, ts_img, out_plot)

        return runtime


# --- Preprocessing operations


class FinalizeTimeseries(SimpleInterface, TimeSeriesGIF):

    class input_spec(TraitedSpec):
        in_files = traits.List(traits.File(exists=True))
        seg_file = traits.File(exists=True)
        mask_file = traits.File(exists=True)
        mc_file = traits.File(exists=True)

    class output_spec(TraitedSpec):
        out_file = traits.File(exists=True)
        out_gif = traits.File(exists=True)
        out_png = traits.File(exists=True)
        mean_file = traits.File(exists=True)
        mean_plot = traits.File(exists=True)
        tsnr_file = traits.File(exists=True)
        tsnr_plot = traits.File(exists=True)
        noise_file = traits.File(exists=True)
        noise_plot = traits.File(exists=True)
        mc_file = traits.File(exists=True)

    def _run_interface(self, runtime):

        # Concatenate timeseries frames into 4D image
        img = nib.concat_images(self.inputs.in_files)
        affine, header = img.affine, img.header
        data = img.get_data()

        # Load the brain mask and seg images
        seg_img = nib.load(self.inputs.seg_file)
        seg = seg_img.get_data()

        mask_img = nib.load(self.inputs.mask_file)
        mask = mask_img.get_data().astype(np.bool)

        # Zero-out data outside the mask
        data[~mask] = 0

        # Scale the timeseries for cross-run intensity normalization
        target = 10000
        scale_value = target / data[mask].mean()
        data = data * scale_value

        # Remove linear but not constant trend
        mean = data.mean(axis=-1, keepdims=True)
        data[mask] = signal.detrend(data[mask])
        data += mean

        # Save out the final time series
        out_file = self.define_output("out_file", "func.nii.gz")
        out_img = nib.Nifti1Image(data, affine, header)
        out_img.to_filename(out_file)

        # Generate the temporal mean and SNR images
        mean = data.mean(axis=-1)
        sd = data.std(axis=-1)
        with np.errstate(all="ignore"):
            tsnr = mean / sd
            tsnr[~mask] = 0

        mean_file = self.define_output("mean_file", "mean.nii.gz")
        mean_img = nib.Nifti1Image(mean, affine, header)
        mean_img.to_filename(mean_file)

        tsnr_file = self.define_output("tsnr_file", "tsnr.nii.gz")
        tsnr_img = nib.Nifti1Image(tsnr, affine, header)
        tsnr_img.to_filename(tsnr_file)

        # Identify unusually noisy voxels
        noise_file = self.define_output("noise_file", "noise.nii.gz")
        gray_mask = (0 < seg) & (seg < 5)
        gray_img = nib.Nifti1Image(gray_mask, img.affine, img.header)
        noise_img = signals.identify_noisy_voxels(
            out_img, gray_img, neighborhood=5, threshold=1.5, detrend=False
        )
        noise_img.to_filename(noise_file)

        # Load the motion correction params and convert to CSV with header
        mc_file = self.define_output("mc_file", "mc.csv")
        mc_data = np.loadtxt(self.inputs.mc_file)
        cols = ["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"]
        mc_data = pd.DataFrame(mc_data, columns=cols)
        mc_data.to_csv(mc_file, index=False)

        # Make a carpet plot of the final timeseries
        out_png = self.define_output("out_png", "func.png")
        p = CarpetPlot(out_img, seg_img, mc_data)
        p.savefig(out_png)
        p.close()

        # Make a GIF movie of the final timeseries
        out_gif = self.define_output("out_gif", "func.gif")
        self.write_time_series_gif(runtime, out_img, out_gif)

        # Make a mosaic of the temporal mean normalized to mean cortical signal
        mean_plot = self.define_output("mean_plot", "mean.png")
        norm_mean = mean / mean[seg == 1].mean()
        mean_m = Mosaic(mean_img, norm_mean, mask_img, show_mask=False)
        mean_m.plot_overlay("cube:-.15:.5", vmin=0, vmax=1, fmt="d")
        mean_m.savefig(mean_plot)
        mean_m.close()

        # Make a mosaic of the tSNR
        tsnr_plot = self.define_output("tsnr_plot", "tsnr.png")
        tsnr_m = Mosaic(tsnr_img, tsnr_img, mask_img, show_mask=False)
        tsnr_m.plot_overlay("cube:.25:-.5", vmin=0, vmax=100, fmt="d")
        tsnr_m.savefig(tsnr_plot)
        tsnr_m.close()

        # Make a mosaic of the noisy voxels
        noise_plot = self.define_output("noise_plot", "noise.png")
        m = Mosaic(mean_img, noise_img, mask_img, show_mask=False)
        m.plot_mask(alpha=1)
        m.savefig(noise_plot)
        m.close()

        return runtime


# --- Preprocessing quality control


class RealignmentReport(SimpleInterface):

    class input_spec(TraitedSpec):
        target_file = traits.File(exists=True)
        realign_params = traits.File(exists=True)

    class output_spec(TraitedSpec):
        params_plot = traits.File(exists=True)
        target_plot = traits.File(exists=True)

    def _run_interface(self, runtime):

        # Load the realignment parameters
        params = np.loadtxt(self.inputs.realign_params)
        cols = ["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"]
        df = pd.DataFrame(params, columns=cols)

        # Plot the motion timeseries
        params_plot = self.define_output("params_plot", "mc_params.png")
        f = self.plot_motion(df)
        f.savefig(params_plot, dpi=100)
        plt.close(f)

        # Plot the target image
        target_plot = self.define_output("target_plot", "mc_target.png")
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
        out_fname = self.define_output("out_file", self.inputs.out_file)

        if len(ref_img.shape) > 3:
            ref_data = ref_img.get_data()[..., 0]
            ref_img = nib.Nifti1Image(ref_data, ref_img.affine, ref_img.header)

        self.write_mosaic_gif(runtime, in_img, ref_img, out_fname)

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
