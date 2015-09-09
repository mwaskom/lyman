"""Fixed effects model to combine across runs for a single subject."""
import os
import os.path as op
import numpy as np
import nibabel as nib
from moss.mosaic import Mosaic

from nipype import Workflow, Node, IdentityInterface
from nipype.interfaces import fsl
from nipype.interfaces.base import (BaseInterface,
                                    BaseInterfaceInputSpec,
                                    TraitedSpec, File, Directory,
                                    InputMultiPath, OutputMultiPath,
                                    traits)

import lyman
from lyman.tools import SaveParameters, nii_to_png, submit_cmdline


def create_ffx_workflow(name="mni_ffx", space="mni",
                        contrasts=None, exp_info=None):
    """Return a workflow object to execute a fixed-effects mode."""
    if contrasts is None:
        contrasts = []
    if exp_info is None:
        exp_info = lyman.default_experiment_parameters()

    inputnode = Node(IdentityInterface(["copes",
                                        "varcopes",
                                        "masks",
                                        "means",
                                        "dofs",
                                        "ss_files",
                                        "anatomy",
                                        "reg_file",
                                        "timeseries"]),
                     name="inputnode")

    # Fit the fixedfx model for each contrast
    ffxmodel = Node(FFXModel(contrasts=contrasts), "ffxmodel")

    # Calculate the fixed effects Rsquared maps
    ffxsummary = Node(FFXSummary(), "ffxsummary")

    # Plot the fixedfx results
    report = Node(FFXReport(space=space), "report")

    # Save the experiment info
    saveparams = Node(SaveParameters(exp_info=exp_info), "saveparams")

    outputnode = Node(IdentityInterface(["flame_results",
                                         "r2_files",
                                         "tsnr_file",
                                         "mean_file",
                                         "summary_report",
                                         "json_file",
                                         "zstat_report"]),
                      "outputs")

    ffx = Workflow(name=name)
    ffx.connect([
        (inputnode, ffxmodel,
            [("copes", "copes"),
             ("varcopes", "varcopes"),
             ("dofs", "dofs"),
             ("masks", "masks"),
             ("reg_file", "reg_file")]),
        (inputnode, ffxsummary,
            [("ss_files", "ss_files"),
             ("means", "means"),
             ("timeseries", "timeseries")]),
        (inputnode, report,
            [("anatomy", "anatomy"),
             ("masks", "masks")]),
        (inputnode, saveparams,
            [("timeseries", "in_file")]),
        (ffxmodel, report,
            [("zstat_files", "zstat_files")]),
        (ffxsummary, report,
            [("r2_files", "r2_files"),
             ("tsnr_file", "tsnr_file")]),
        (ffxmodel, outputnode,
            [("flame_results", "flame_results")]),
        (ffxsummary, outputnode,
            [("r2_files", "r2_files"),
             ("tsnr_file", "tsnr_file"),
             ("mean_file", "mean_file")]),
        (report, outputnode,
            [("summary_files", "summary_report"),
             ("zstat_files", "zstat_report")]),
        (saveparams, outputnode,
            [("json_file", "json_file")]),
    ])

    return ffx, inputnode, outputnode


# =========================================================================== #


class FFXModelInput(BaseInterfaceInputSpec):

    contrasts = traits.List()
    copes = InputMultiPath(File(exists=True))
    varcopes = InputMultiPath(File(exists=True))
    dofs = InputMultiPath(File(exists=True))
    masks = InputMultiPath(File(exists=True))
    reg_file = File(exists=True)


class FFXModelOutput(TraitedSpec):

    flame_results = OutputMultiPath(Directory(exists=True))
    zstat_files = OutputMultiPath(File(exists=True))


class FFXModel(BaseInterface):

    input_spec = FFXModelInput
    output_spec = FFXModelOutput

    def _run_interface(self, runtime):

        n_con = len(self.inputs.contrasts)

        # Find the basic geometry of the image
        img = nib.load(self.inputs.copes[0])
        x, y, z = img.shape
        aff, hdr = img.get_affine(), img.get_header()

        # Get lists of files for each contrast
        copes = self._unpack_files(self.inputs.copes, n_con)
        varcopes = self._unpack_files(self.inputs.varcopes, n_con)

        # Make an image with the DOF for each run
        dofs = np.array([np.loadtxt(f) for f in self.inputs.dofs])
        dof_data = np.ones((x, y, z, len(dofs))) * dofs

        # Find the intersection of the masks
        mask_data = [nib.load(f).get_data() for f in self.inputs.masks]
        common_mask = np.all(mask_data, axis=0)
        nib.Nifti1Image(common_mask, aff, hdr).to_filename("mask.nii.gz")

        # Run the flame models
        flame_results = []
        zstat_files = []
        for i, contrast in enumerate(self.inputs.contrasts):

            # Load each run of cope and varcope files into a list
            cs = [nib.load(f).get_data()[..., np.newaxis] for f in copes[i]]
            vs = [nib.load(f).get_data()[..., np.newaxis] for f in varcopes[i]]

            # Find all of the nonzero copes
            # This handles cases where there were no events for some of
            # the runs for the contrast we're currently dealing with
            good_cs = [not np.allclose(d, 0) for d in cs]
            good_vs = [not np.allclose(d, 0) for d in vs]
            good = np.all([good_cs, good_vs], axis=0)

            # Handle the case where no events occured for this contrast
            if not good.any():
                good = np.ones(len(cs), bool)

            # Concatenate the cope and varcope data, save only the good frames
            c_data = np.concatenate(cs, axis=-1)[:, :, :, good]
            v_data = np.concatenate(vs, axis=-1)[:, :, :, good]

            # Write out the concatenated copes and varcopes
            nib.Nifti1Image(c_data, aff, hdr).to_filename("cope_4d.nii.gz")
            nib.Nifti1Image(v_data, aff, hdr).to_filename("varcope_4d.nii.gz")

            # Write out a correctly sized design for this contrast
            fsl.L2Model(num_copes=int(good.sum())).run()

            # Mask the DOF data and write it out for this run
            contrast_dof = dof_data[:, :, :, good]
            nib.Nifti1Image(contrast_dof, aff, hdr).to_filename("dof.nii.gz")

            # Build the flamo commandline and run
            flamecmd = ["flameo",
                        "--cope=cope_4d.nii.gz",
                        "--varcope=varcope_4d.nii.gz",
                        "--mask=mask.nii.gz",
                        "--dvc=dof.nii.gz",
                        "--runmode=fe",
                        "--dm=design.mat",
                        "--tc=design.con",
                        "--cs=design.grp",
                        "--ld=" + contrast,
                        "--npo"]
            runtime = submit_cmdline(runtime, flamecmd)

            # Rename the written file and append to the outputs
            for kind in ["cope", "varcope"]:
                os.rename(kind + "_4d.nii.gz",
                          "%s/%s_4d.nii.gz" % (contrast, kind))

            # Put the zstats and mask on the surface
            for hemi in ["lh", "rh"]:
                projcmd = ["mri_vol2surf",
                           "--mov", "%s/zstat1.nii.gz" % contrast,
                           "--reg", self.inputs.reg_file,
                           "--hemi", hemi,
                           "--projfrac-avg", "0", "1", ".1",
                           "--o", "%s/%s.zstat1.mgz" % (contrast, hemi)]
                submit_cmdline(runtime, projcmd)

                # Mask image
                projcmd = ["mri_vol2surf",
                           "--mov", "%s/mask.nii.gz" % contrast,
                           "--reg", self.inputs.reg_file,
                           "--hemi", hemi,
                           "--projfrac-max", "0", "1", ".1",
                           "--o", "%s/%s.mask.mgz" % (contrast, hemi)]
                submit_cmdline(runtime, projcmd)

            flame_results.append(op.abspath(contrast))
            zstat_files.append(op.abspath("%s/zstat1.nii.gz" % contrast))

        self.flame_results = flame_results
        self.zstat_files = zstat_files

        return runtime

    def _unpack_files(self, files, n_con):
        """Unpack a list of (var)copes into a list of lists by contrast."""
        out = []
        for con in range(n_con):
            out.append([f for f in files if "cope{}_".format(con + 1) in f])
        return out

    def _list_outputs(self):

        outputs = self._outputs().get()
        outputs["flame_results"] = self.flame_results
        outputs["zstat_files"] = self.zstat_files
        return outputs


class FFXSummaryInput(BaseInterfaceInputSpec):

    ss_files = InputMultiPath(File(exists=True))
    means = InputMultiPath(File(exists=True))
    timeseries = InputMultiPath(File(exists=True))


class FFXSummaryOutput(TraitedSpec):

    r2_files = OutputMultiPath(File(exists=True))
    tsnr_file = File(exists=True)
    mean_file = File(exists=True)


class FFXSummary(BaseInterface):

    input_spec = FFXSummaryInput
    output_spec = FFXSummaryOutput

    def _run_interface(self, runtime):

        # Get basic info about the image
        img = nib.load(self.inputs.means[0])
        self.affine = img.get_affine()
        self.header = img.get_header()

        # Save the fixed effects r squared maps
        self.compute_rsquared()

        # Save the tsnr map
        self.compute_tsnr()

        return runtime

    def compute_rsquared(self):
        """Compute and save the full and main rsquared for fixed effects."""
        # First read the total sum of squares for each run
        total_sumsquares = self.sum_squares("sstot")

        for part in ["full", "main"]:

            # Read in the residual sum of squares and take grand sum
            res_sumsquares = self.sum_squares("ssres", part)

            # Calculate the full model R2
            r2 = 1 - res_sumsquares / total_sumsquares

            # Save an image with these data
            self.save_image(r2, "r2_" + part)

    def sum_squares(self, kind, part=None):
        """Sum the sum of squares images across runs."""
        ss_files = self.inputs.ss_files
        stem = kind
        if kind == "ssres":
            stem += "_" + part
        files = [f for f in ss_files if stem in f]
        data = [nib.load(f).get_data() for f in files]
        sums = np.sum(data, axis=0)
        return sums

    def compute_tsnr(self):
        """Compute the toal signal to noise."""
        # Load the mean images from each run and take the grand mean."""
        mean_imgs = [nib.load(f) for f in self.inputs.means]
        mean_data = np.mean([img.get_data() for img in mean_imgs], axis=0)

        # Use the timeseries to get the total number of frames
        ts_lengths = [nib.load(f).shape[-1] for f in self.inputs.timeseries]
        n = sum(ts_lengths)

        # Compute and save tsnr
        res_sumsquares = self.sum_squares("ssres", "full")
        res_std = np.sqrt(res_sumsquares / n)
        tsnr = mean_data / res_std
        tsnr = np.nan_to_num(tsnr)
        self.save_image(tsnr, "tsnr")

        # Save a mean file
        self.save_image(mean_data, "mean_func")

    def save_image(self, data, fname):
        """Save data to the output structure."""
        img = nib.Nifti1Image(data, self.affine, self.header)
        img.to_filename(fname + ".nii.gz")

    def _list_outputs(self):

        outputs = self._outputs().get()
        outputs["r2_files"] = [op.abspath("r2_full.nii.gz"),
                               op.abspath("r2_main.nii.gz")]
        outputs["tsnr_file"] = op.abspath("tsnr.nii.gz")
        outputs["mean_file"] = op.abspath("mean_func.nii.gz")
        return outputs


class FFXReportInput(BaseInterfaceInputSpec):

    space = traits.Either("mni", "epi")
    anatomy = File(exists=True)
    tsnr_file = File(exists=True)
    zstat_files = InputMultiPath(File(exists=True))
    r2_files = InputMultiPath(File(exists=True))
    masks = InputMultiPath(File(exists=True))


class FFXReportOutput(TraitedSpec):

    summary_files = OutputMultiPath(File(exists=True))
    zstat_files = OutputMultiPath(Directory(exists=True))


class FFXReport(BaseInterface):

    input_spec = FFXReportInput
    output_spec = FFXReportOutput

    def _run_interface(self, runtime):

        self.step = 1 if self.inputs.space == "epi" else 2
        self.summary_files = []
        self.zstat_files = []
        self.write_mask_image()
        self.write_tsnr_image()
        self.write_r2_images()
        self.write_zstat_images()

        return runtime

    def write_mask_image(self):
        """Show the overlap of each run's mask."""
        mask_imgs = [nib.load(f) for f in self.inputs.masks]
        mask_data = [img.get_data()[..., np.newaxis] for img in mask_imgs]
        each_mask = np.concatenate(mask_data, axis=-1)
        any_mask = each_mask.max(axis=-1)
        self.every_mask = each_mask.min(axis=-1)

        m = Mosaic(self.inputs.anatomy, each_mask, any_mask,
                   stat_interp="nearest", step=self.step, show_mask=False)
        m.plot_mask_edges()
        out_fname = op.abspath("mask_overlap.png")
        self.summary_files.append(out_fname)
        m.savefig(out_fname)
        m.close()

    def write_tsnr_image(self):
        """Show the total subject SNR."""
        m = Mosaic(self.inputs.anatomy,
                   self.inputs.tsnr_file,
                   self.every_mask,
                   step=self.step)
        m.plot_overlay("cube:1.9:.5", 0, alpha=1, fmt="%d")
        out_fname = op.abspath("tsnr.png")
        self.summary_files.append(out_fname)
        m.savefig(out_fname)
        m.close()

    def write_r2_images(self):
        """Show the fixed effects model fit."""
        cmaps = ["cube:2:0", "cube:2.6:0"]
        for fname, cmap in zip(self.inputs.r2_files, cmaps):
            m = Mosaic(self.inputs.anatomy, fname,
                       self.every_mask, step=self.step)
            m.plot_overlay(cmap, 0, alpha=.85)
            out_fname = op.abspath(nii_to_png(fname))
            self.summary_files.append(out_fname)
            m.savefig(out_fname)
            m.close()

    def write_zstat_images(self):
        """Show the fixed effects inference maps."""
        for fname in self.inputs.zstat_files:
            m = Mosaic(self.inputs.anatomy, fname,
                       self.every_mask, step=self.step)
            m.plot_activation(pos_cmap="Reds_r", neg_cmap="Blues",
                              thresh=2.3, alpha=.85)
            contrast = fname.split("/")[-2]
            os.mkdir(contrast)
            out_fname = op.join(contrast, "zstat1.png")
            self.zstat_files.append(op.abspath(contrast))
            m.savefig(out_fname)
            m.close()

    def _list_outputs(self):

        outputs = self._outputs().get()
        outputs["summary_files"] = self.summary_files
        outputs["zstat_files"] = self.zstat_files
        return outputs
