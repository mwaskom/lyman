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
from nipype.interfaces import fsl, freesurfer as fs

from .. import signals  # TODO confusingly close to scipy.signal
from ..mosaic import Mosaic
from ..carpetplot import CarpetPlot
from ..graphutils import SimpleInterface


def define_preproc_workflow(proj_info, subjects, session, exp_info, qc=True):

    # proj_info will be a bunch or other object with data_dir, etc. fields

    # sess info is a nested dictionary:
    # outer keys are subjects
    # inner keys are sessions
    # inner values are lists of runs

    # exp_info is a bunch or dict or other obj with experiment parameters

    # --- Workflow parameterization and data input

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
               if exp_info.name in scan_info[subj][sess]
               and session is None or sess in session]
        for subj in subjects
    }
    session_source = Node(IdentityInterface(["subject", "session"]),
                          name="session_source",
                          itersource=("subject_source", "subject"),
                          iterables=("session", session_iterables))

    run_iterables = {
        (subj, sess): [(subj, sess, run)
                       for run in scan_info[subj][sess][exp_info.name]]
        for subj in subjects
        for sess in scan_info[subj]
        if exp_info.name in scan_info[subj][sess]
        and session is None or sess in session
    }
    run_source = Node(IdentityInterface(["subject", "session", "run"]),
                      name="run_source",
                      itersource=("session_source", "session"),
                      iterables=("run", run_iterables))

    session_input = Node(SessionInput(data_dir=proj_info.data_dir,
                                      analysis_dir=proj_info.analysis_dir,
                                      fm_template=proj_info.fm_template,
                                      phase_encoding=proj_info.phase_encoding),
                         "session_input")

    run_input = Node(RunInput(experiment=exp_info.name,
                              data_dir=proj_info.data_dir,
                              sb_template=proj_info.sb_template,
                              ts_template=proj_info.ts_template),
                     name="run_input")

    # --- Warpfield estimation using topup

    # Distortion warpfield estimation
    estimate_distortions = Node(fsl.TOPUP(config="b02b0.cnf"),
                                "estimate_distortions")

    # Post-process the TOPUP outputs
    finalize_unwarping = Node(FinalizeUnwarping(), "finalize_unwarping")

    # --- Registration of SE-EPI (without distortions) to Freesurfer anatomy

    fm2anat = Node(fs.BBRegister(init="fsl",
                                 contrast_type="t2",
                                 out_fsl_file="sess2anat.mat",
                                 out_reg_file="sess2anat.dat"),
                   "fm2anat")

    fm2anat_qc = Node(AnatRegReport(), "fm2anat_qc")

    # --- Registration of SBRef to SE-EPI (with distortions)

    sb2fm = Node(fsl.FLIRT(dof=6, interp="spline"), "sb2fm")

    sb2fm_qc = Node(CoregGIF(out_file="coreg.gif"), "sb2fm_qc")

    # --- Motion correction of time series to SBRef (with distortions)

    ts2sb = Node(fsl.MCFLIRT(save_mats=True, save_plots=True),
                 "ts2sb")

    ts2sb_qc = Node(RealignmentReport(), "ts2sb_qc")

    # --- Combined motion correction and unwarping of time series

    # Combine pre-and post-warp linaer transforms
    combine_rigids = MapNode(CombineLinearTransforms(),
                             "ts2sb_file", "combine_rigids")

    # Split the timeseries into each frame
    split_ts = Node(fsl.Split(dimension="t"), "split_ts")

    # Simultaneously apply rigid transforms and nonlinear warpfield
    restore_ts_frames = MapNode(fsl.ApplyWarp(interp="spline", relwarp=True),
                                ["in_file", "premat", "postmat"],
                                "restore_ts")

    # Perform final preprocessing operations on timeseries
    # TODO do jacobian modulation here?
    finalize_ts = Node(FinalizeTimeseries(), "finalize_ts")

    # --- TODO Combined motion correction and unwarping of template

    # --- Workflow ouptut

    def define_timeseries_container(subject, experiment, session, run):
        return "{}/{}/timeseries/{}_{}".format(subject, experiment,
                                               session, run)

    timeseries_container = Node(Function(["subject", "experiment",
                                          "session", "run"],
                                         "path", define_timeseries_container),
                                "timeseries_container")
    timeseries_container.inputs.experiment = exp_info.name

    timeseries_output = Node(DataSink(base_directory=proj_info.analysis_dir,
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

        # --- SE-EPI fieldmap processing and anatomical registration

        # Phase-encode distortion estimation

        (session_input, estimate_distortions,
            [("fm", "in_file"),
             ("phase_encoding", "encoding_direction"),
             ("readout_times", "readout_times")]),
        (session_input, finalize_unwarping,
            [("fm", "raw_file"),
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
        (finalize_unwarping, sb2fm,
            [("raw_file", "reference"),
             ("mask_file", "ref_weight")]),

        # Single-interpolation spatial realignment and unwarping

        (ts2sb, combine_rigids,
            [("mat_file", "ts2sb_file")]),
        (sb2fm, combine_rigids,
            [("out_matrix_file", "sb2fm_file")]),
        (fm2anat, combine_rigids,
            [("out_fsl_file", "fm2anat_file")]),
        (session_input, combine_rigids,
            [("reg_file", "anat2temp_file")]),
        (run_input, split_ts,
            [("ts", "in_file")]),
        (split_ts, restore_ts_frames,
            [("out_files", "in_file")]),
        (combine_rigids, restore_ts_frames,
            [("ts2fm_file", "premat"),
             ("fm2temp_file", "postmat")]),
        (session_input, restore_ts_frames,
            [("seg_file", "ref_file")]),
        (session_input, finalize_ts,
            [("seg_file", "seg_file"),
             ("mask_file", "mask_file")]),
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

        (run_input, ts2sb_qc,
            [("sb", "target_file")]),
        (ts2sb, ts2sb_qc,
            [("par_file", "realign_params")]),

        # Registration of corrected SE-EPI to anatomy

        (session_input, fm2anat_qc,
            [("subject", "subject_id")]),
        (finalize_unwarping, fm2anat_qc,
            [("corrected_file", "in_file")]),
        (fm2anat, fm2anat_qc,
            [("out_reg_file", "reg_file"),
             ("min_cost_file", "cost_file")]),

        # Registration of SBRef volume to SE-EPI fieldmap

        (sb2fm, sb2fm_qc,
            [("out_file", "in_file")]),
        (finalize_unwarping, sb2fm_qc,
            [("raw_file", "ref_file")]),

        # Ouputs associated with each scanner run

        (run_input, timeseries_output,
            [("ts_plot", "qc.@raw_gif")]),
        (sb2fm_qc, timeseries_output,
            [("out_file", "qc.@sb2fm_gif")]),
        (ts2sb_qc, timeseries_output,
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
        data_dir = traits.Directory(exists=True)
        analysis_dir = traits.Directory(exists=True)
        fm_template = traits.Str()
        phase_encoding = traits.Str()

    class output_spec(TraitedSpec):
        session_key = traits.Tuple()
        subject = traits.Str()
        session = traits.Str()
        fm = traits.File(exists=True)
        fm_frames = traits.List(traits.File(exists=True))
        reg_file = traits.File(exists=True)
        seg_file = traits.File(exists=True)
        mask_file = traits.File(exists=True)
        phase_encoding = traits.List(traits.Str())
        readout_times = traits.List(traits.Float())

    def _run_interface(self, runtime):

        # Determine the execution parameters
        subject, session = self.inputs.session
        self._results["session_key"] = self.inputs.session
        self._results["subject"] = str(subject)
        self._results["session"] = str(session)

        # Determine the phase encoding directions
        pe = self.inputs.phase_encoding
        if pe == "ap":
            pos_pe, neg_pe = "ap", "pa"
        elif pe == "pa":
            pos_pe, neg_pe = "pa", "ap"
        else:
            raise ValueError("Phase encoding must be 'ap' or 'pa'")

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
        template_path = op.join(self.inputs.analysis_dir, subject, "template")
        results = dict(
            reg_file=op.join(template_path, "anat2func.mat"),
            seg_file=op.join(template_path, "seg.nii.gz"),
            mask_file=op.join(template_path, "mask.nii.gz"),
        )
        self._results.update(results)

        return runtime


class RunInput(SimpleInterface, TimeSeriesGIF):

    class input_spec(TraitedSpec):
        run = traits.Tuple()
        data_dir = traits.Directory(exists=True)
        experiment = traits.Str()
        sb_template = traits.Str()
        ts_template = traits.Str()

        # TODO this default should be defined at the project/experiment level
        crop_frames = traits.Int(0, usedefault=True)

    class output_spec(TraitedSpec):
        subject = traits.Str()
        session = traits.Str()
        run = traits.Str()
        sb = traits.File(exists=True)
        ts = traits.File(exists=True)
        ts_plot = traits.File(exists=True)

    def _run_interface(self, runtime):

        # Determine the parameters
        experiment = self.inputs.experiment
        subject, session, run = self.inputs.run
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


class CombineLinearTransforms(SimpleInterface):

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


class FinalizeUnwarping(SimpleInterface):

    class input_spec(TraitedSpec):
        raw_file = traits.File(exists=True)
        corrected_file = traits.File(exists=True)
        warp_files = traits.List(traits.File(exists=True))
        jacobian_files = traits.List(traits.File(Exists=True))
        phase_encoding = traits.List(traits.Str)

    class output_spec(TraitedSpec):
        raw_file = traits.File(exists=True)
        corrected_file = traits.File(exists=True)
        warp_file = traits.File(exists=True)
        mask_file = traits.File(exists=True)
        jacobian_file = traits.File(Exists=True)
        corrected_plot = traits.File(exists=True)
        warp_plot = traits.File(exists=True)
        unwarp_gif = traits.File(exists=True)

    def _run_interface(self, runtime):

        # Load the 4D raw fieldmap image and select first frame
        raw_img_frames = nib.load(self.inputs.raw_file)
        raw_img = nib.four_to_three(raw_img_frames)[0]

        # Write out the raw image to serve as a registration target
        raw_file = self.define_output("raw_file", "raw.nii.gz")
        raw_img.to_filename(raw_file)

        # Load the 4D jacobian image
        jac_img_frames = nib.concat_images(self.inputs.jacobian_files)
        jac_data = jac_img_frames.get_data()

        # Load the 4D corrected fieldmap image
        corr_img_frames = nib.load(self.inputs.corrected_file)
        corr_data = corr_img_frames.get_data()

        # Jacobian modulate the corrected image
        corr_img_frames = nib.Nifti1Image(corr_data * jac_data,
                                          corr_img_frames.affine,
                                          corr_img_frames.header)

        # Average the corrected image over the final dimension and write
        corrected_file = self.define_output("corrected_file", "func.nii.gz")
        corr_data_avg = corr_data.mean(axis=-1)
        corr_img = nib.Nifti1Image(corr_data_avg,
                                   corr_img_frames.affine,
                                   corr_img_frames.header)
        corr_img.to_filename(corrected_file)

        # Save the first frame of the jacobian image using the raw geometry
        jacobian_file = self.define_output("jacobian_file", "jacobian.nii.gz")
        jac_img = nib.Nifti1Image(jac_data[..., 0],
                                  raw_img.affine,
                                  raw_img.header)
        jac_img.to_filename(jacobian_file)

        # Select the first warpfield image
        # We combine the two fieldmap images so that the first one has
        # a phase encoding that matches the time series data.
        # Also note that when the fieldmap images have multiple frames,
        # the warps corresponding to those frames are identical.
        warp_file = self.inputs.warp_files[0]

        # Load in the the warp file and save out with the correct affine
        # (topup doesn't save the header geometry correctly for some reason)
        out_warp_file = self.define_output("warp_file", "warp.nii.gz")
        affine, header = raw_img.affine, raw_img.header
        warp_data = nib.load(warp_file).get_data()
        warp_img = nib.Nifti1Image(warp_data, affine, header)
        warp_img.to_filename(out_warp_file)

        # Select the warp along the phase encode direction
        # Note: we elsewhere currently require phase encoding to be AP or PA
        # so because the input node transforms the fieldmap to canonical
        # orientation this will work. But in the future we might want to ne
        # more flexible with what we accept and will need to change this.
        warp_data_y = warp_data[..., 1]

        # Write out a mask to exclude voxels with large distortions/dropout
        mask_file = self.define_output("mask_file", "warp_mask.nii.gz")
        mask_data = (np.abs(warp_data_y) < 4).astype(np.int)
        mask_img = nib.Nifti1Image(mask_data, raw_img.affine, raw_img.header)
        mask_img.to_filename(mask_file)

        # Generate a QC image of the corrected volume
        corrected_plot = self.define_output("corrected_plot", "func.png")
        m = Mosaic(corr_img)
        m.savefig(corrected_plot)
        m.close()

        # Generate a QC image of the warpfield
        warp_plot = self.define_output("warp_plot", "warp.png")
        m = Mosaic(raw_img, warp_data_y)
        m.plot_overlay("coolwarm", vmin=-6, vmax=6, alpha=.75)
        m.savefig(warp_plot)
        m.close()

        # Generate a QC gif of the unwarping performance
        self.generate_unwarp_gif(runtime, raw_img_frames, corr_img_frames)

        return runtime

    def generate_unwarp_gif(self, runtime, raw_img, corrected_img):

        vol_data = dict(orig=raw_img.get_data(), corr=corrected_img.get_data())

        pe_data = dict(orig=[], corr=[])
        pe = np.array(self.inputs.phase_encoding)
        for enc in np.unique(pe):
            enc_trs = pe == enc
            for scan in ["orig", "corr"]:
                enc_data = vol_data[scan][..., enc_trs].mean(axis=-1)
                pe_data[scan].append(enc_data)

        r_vals = dict()
        for scan, (scan_pos, scan_neg) in pe_data.items():
            r_vals[scan] = np.corrcoef(scan_pos.flat, scan_neg.flat)[0, 1]

        nx, ny, nz, _ = vol_data["orig"].shape
        x_slc = (np.linspace(.2, .8, 8) * nx).astype(np.int)

        vmin, vmax = np.percentile(vol_data["orig"].flat, [2, 98])
        kws = dict(vmin=vmin, vmax=vmax, cmap="gray")

        width = len(x_slc)
        height = (nz / ny) * 2.75

        png_fnames = []
        for i, enc in enumerate(["pos", "neg"]):

            f = plt.figure(figsize=(width, height))
            gs = dict(
                orig=plt.GridSpec(1, len(x_slc), 0, .5, 1, .95, 0, 0),
                corr=plt.GridSpec(1, len(x_slc), 0, 0, 1, .45, 0, 0)
            )

            text_kws = dict(size=7, color="w", backgroundcolor="0",
                            ha="center", va="bottom")
            f.text(.5, .93,
                   "Original similarity: {:.2f}".format(r_vals["orig"]),
                   **text_kws)
            f.text(.5, .43,
                   "Corrected similarity: {:.2f}".format(r_vals["corr"]),
                   **text_kws)

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

        out_file = self.define_output("unwarp_gif", "unwarp.gif")
        cmdline = ["convert", "-loop", "0", "-delay", "100"]
        cmdline.extend(png_fnames)
        cmdline.append(out_file)

        self.submit_cmdline(runtime, cmdline)


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
        mask &= sd > 0
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


class AnatRegReport(SimpleInterface):

    class input_spec(TraitedSpec):
        subject_id = traits.Str()
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
        cost = np.loadtxt(self.inputs.cost_file)[0]

        # Make a mosaic of the registration from func to wm seg
        # TODO this should be an OrthoMosaic when that is implemented
        out_file = self.define_output("out_file", "reg.png")

        m = Mosaic(registered_file, wm_data, mask, step=3, show_mask=False)
        m.plot_mask_edges()
        if cost is not None:
            m.fig.suptitle("Final cost: {:.2f}".format(cost),
                           size=10, color="white")
        m.savefig(out_file)
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
