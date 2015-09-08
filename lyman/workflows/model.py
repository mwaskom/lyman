"""Timeseries model using FSL's gaussian least squares."""
import re
import os.path as op
import numpy as np
from scipy import stats, signal
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from moss import glm
from moss.mosaic import Mosaic
import seaborn as sns

from nipype import Node, MapNode, Workflow, IdentityInterface
from nipype.interfaces import fsl
from nipype.interfaces.base import (BaseInterface,
                                    BaseInterfaceInputSpec,
                                    InputMultiPath, OutputMultiPath,
                                    TraitedSpec, File, traits,
                                    isdefined)
import lyman
from lyman.tools import ManyOutFiles, SaveParameters, nii_to_png


def create_timeseries_model_workflow(name="model", exp_info=None):

    # Default experiment parameters for generating graph image, testing, etc.
    if exp_info is None:
        exp_info = lyman.default_experiment_parameters()

    # Define constant inputs
    inputs = ["realign_file", "nuisance_file", "artifact_file", "timeseries"]

    # Possibly add the design and regressor files to the inputs
    if exp_info["design_name"] is not None:
        inputs.append("design_file")
    if exp_info["regressor_file"] is not None:
        inputs.append("regressor_file")

    # Define the workflow inputs
    inputnode = Node(IdentityInterface(inputs), "inputs")

    # Set up the experimental design
    modelsetup = MapNode(ModelSetup(exp_info=exp_info),
                         ["timeseries", "realign_file",
                          "nuisance_file", "artifact_file"],
                         "modelsetup")

    # For some nodes, make it possible to request extra memory
    mem_request = {"qsub_args": "-l h_vmem=%dG" % exp_info["memory_request"]}

    # Use film_gls to estimate the timeseries model
    modelestimate = MapNode(fsl.FILMGLS(smooth_autocorr=True,
                                        mask_size=5,
                                        threshold=100),
                            ["design_file", "in_file", "tcon_file"],
                            "modelestimate")
    modelestimate.plugin_args = mem_request

    # Compute summary statistics about the model fit
    modelsummary = MapNode(ModelSummary(),
                           ["design_matrix_pkl",
                            "timeseries",
                            "pe_files"],
                           "modelsummary")
    modelsummary.plugin_args = mem_request

    # Save the experiment info for this run
    # Save the experiment info for this run
    saveparams = MapNode(SaveParameters(exp_info=exp_info),
                         "in_file", "saveparams")

    # Report on the results of the model
    # Note: see below for a conditional iterfield
    modelreport = MapNode(ModelReport(),
                          ["timeseries", "sigmasquareds_file",
                           "tsnr_file", "r2_files"],
                          "modelreport")

    # Define the workflow outputs
    outputnode = Node(IdentityInterface(["results",
                                         "copes",
                                         "varcopes",
                                         "zstats",
                                         "r2_files",
                                         "ss_files",
                                         "tsnr_file",
                                         "report",
                                         "design_mat",
                                         "contrast_mat",
                                         "design_pkl",
                                         "design_report",
                                         "json_file"]),
                      "outputs")

    # Define the workflow and connect the nodes
    model = Workflow(name=name)
    model.connect([
        (inputnode, modelsetup,
            [("realign_file", "realign_file"),
             ("nuisance_file", "nuisance_file"),
             ("artifact_file", "artifact_file"),
             ("timeseries", "timeseries")]),
        (inputnode, modelestimate,
            [("timeseries", "in_file")]),
        (inputnode, saveparams,
            [("timeseries", "in_file")]),
        (modelsetup, modelestimate,
            [("design_matrix_file", "design_file"),
             ("contrast_file", "tcon_file")]),
        (modelsetup, modelsummary,
            [("design_matrix_pkl", "design_matrix_pkl")]),
        (inputnode, modelsummary,
            [("timeseries", "timeseries")]),
        (modelestimate, modelsummary,
            [("param_estimates", "pe_files")]),
        (inputnode, modelreport,
            [("timeseries", "timeseries")]),
        (modelestimate, modelreport,
            [("sigmasquareds", "sigmasquareds_file")]),
        (modelsummary, modelreport,
            [("r2_files", "r2_files"),
             ("tsnr_file", "tsnr_file")]),
        (modelsetup, outputnode,
            [("design_matrix_file", "design_mat"),
             ("contrast_file", "contrast_mat"),
             ("design_matrix_pkl", "design_pkl"),
             ("report", "design_report")]),
        (saveparams, outputnode,
            [("json_file", "json_file")]),
        (modelestimate, outputnode,
            [("results_dir", "results"),
             ("copes", "copes"),
             ("varcopes", "varcopes"),
             ("zstats", "zstats")]),
        (modelsummary, outputnode,
            [("r2_files", "r2_files"),
             ("ss_files", "ss_files"),
             ("tsnr_file", "tsnr_file")]),
        (modelreport, outputnode,
            [("out_files", "report")]),
        ])

    if exp_info["design_name"] is not None:
        model.connect(inputnode, "design_file",
                      modelsetup, "design_file")
    if exp_info["regressor_file"] is not None:
        model.connect(inputnode, "regressor_file",
                      modelsetup, "regressor_file")
    if exp_info["contrasts"]:
        model.connect(modelestimate, "zstats",
                      modelreport, "zstat_files")
        modelreport.iterfield.append("zstat_files")

    return model, inputnode, outputnode


# =========================================================================== #


class ModelSetupInput(BaseInterfaceInputSpec):

    exp_info = traits.Dict()
    timeseries = File(exists=True)
    design_file = File(exists=True)
    realign_file = File(exists=True)
    nuisance_file = File(exists=True)
    artifact_file = File(exists=True)
    regressor_file = File(exists=True)


class ModelSetupOutput(TraitedSpec):

    design_matrix_file = File(exists=True)
    contrast_file = File(exists=True)
    design_matrix_pkl = File(exists=True)
    report = OutputMultiPath(File(exists=True))


class ModelSetup(BaseInterface):

    input_spec = ModelSetupInput
    output_spec = ModelSetupOutput

    def _run_interface(self, runtime):

        # Get all the information for the design
        design_kwargs = self.build_design_information()

        # Initialize the design matrix object
        X = glm.DesignMatrix(**design_kwargs)

        # Report on the design
        self.design_report(self.inputs.exp_info, X, design_kwargs)

        # Write out the design object as a pkl to pass to the report function
        X.to_pickle("design.pkl")

        # Finally, write out the design files in FSL format
        X.to_fsl_files("design", self.inputs.exp_info["contrasts"])

        return runtime

    def build_design_information(self):

        # Load in the design information
        exp_info = self.inputs.exp_info
        tr = self.inputs.exp_info["TR"]

        # Derive the length of the scan and run number from the timeseries
        ntp = nib.load(self.inputs.timeseries).shape[-1]
        run = int(re.search("run_(\d+)", self.inputs.timeseries).group(1))

        # Get the experimental design
        if isdefined(self.inputs.design_file):
            design = pd.read_csv(self.inputs.design_file)
            design = design[design["run"] == run]
        else:
            design = None

        # Get confound information to add to the model
        confounds = []
        sources = exp_info["confound_sources"]
        bad_sources = set(sources) - set(["motion", "wm", "brain"])
        if bad_sources:
            msg = ("Invalid confound source specification: {}"
                   .format(list(bad_sources)))
            raise ValueError(msg)

        # Get the motion correction parameters
        if "motion" in sources:
            realign = pd.read_csv(self.inputs.realign_file)
            realign = realign.filter(regex="rot|trans").apply(stats.zscore)
            confounds.append(realign)

        # Get the anatomical nuisance sources
        nuisance = pd.read_csv(self.inputs.nuisance_file).apply(stats.zscore)
        if "wm" in sources:
            wm = nuisance.filter(regex="wm")
            confounds.append(wm)
        if "brain" in sources:
            brain = nuisance["brain"]
            confounds.append(brain)

        # Combine the different confound sources
        if confounds:
            confounds = pd.concat(confounds, axis=1)
        else:
            confounds = None

        # Get the image artifacts
        if exp_info["remove_artifacts"]:
            artifacts = pd.read_csv(self.inputs.artifact_file).max(axis=1)
        else:
            artifacts = None

        # Get the additional model regressors
        if isdefined(self.inputs.regressor_file):
            regressors = pd.read_csv(self.inputs.regressor_file)
            regressors = regressors[regressors["run"] == run]
            regressors = regressors.drop("run", axis=1)
            if exp_info["regressor_names"] is not None:
                regressors = regressors[exp_info["regressor_names"]]
            regressors.index = np.arange(ntp) * tr
        else:
            regressors = None

        # Set up the HRF model
        hrf = getattr(glm, exp_info["hrf_model"])
        hrf = hrf(exp_info["temporal_deriv"], tr, **exp_info["hrf_params"])

        # Build a dict of keyword arguments for the design matrix
        design_kwargs = dict(design=design,
                             hrf_model=hrf,
                             ntp=ntp,
                             tr=tr,
                             confounds=confounds,
                             artifacts=artifacts,
                             regressors=regressors,
                             condition_names=exp_info["condition_names"],
                             confound_pca=exp_info["confound_pca"],
                             hpf_cutoff=exp_info["hpf_cutoff"])

        return design_kwargs

    def design_report(self, exp_info, X, design_kwargs):
        """Generate static images summarizing the design."""
        # Plot the design itself
        design_png = op.abspath("design.png")
        X.plot(fname=design_png, close=True)

        with sns.axes_style("whitegrid"):
            # Plot the eigenvalue spectrum
            svd_png = op.abspath("design_singular_values.png")
            X.plot_singular_values(fname=svd_png, close=True)

            # Plot the correlations between design elements and confounds
            corr_png = op.abspath("design_correlation.png")
            if design_kwargs["design"] is None:
                with open(corr_png, "wb"):
                    pass
            else:
                X.plot_confound_correlation(fname=corr_png, close=True)

        # Build a list of images sumarrizing the model
        report = [design_png, corr_png, svd_png]

        # Now plot the information loss from the high-pass filter
        design_kwargs["hpf_cutoff"] = None
        X_unfiltered = glm.DesignMatrix(**design_kwargs)
        tr = design_kwargs["tr"]
        ntp = design_kwargs["ntp"]

        # Plot for each contrast
        for i, (name, cols, weights) in enumerate(exp_info["contrasts"], 1):

            # Compute the contrast predictors
            C = X.contrast_vector(cols, weights)
            y_filt = X.design_matrix.dot(C)
            y_unfilt = X_unfiltered.design_matrix.dot(C)

            # Compute the spectral density for filtered and unfiltered
            fs, pxx_filt = signal.welch(y_filt, 1. / tr, nperseg=ntp)
            fs, pxx_unfilt = signal.welch(y_unfilt, 1. / tr, nperseg=ntp)

            # Draw the spectral density
            with sns.axes_style("whitegrid"):
                f, ax = plt.subplots(figsize=(9, 3))
            ax.fill_between(fs, pxx_unfilt, color="#C41E3A")
            ax.axvline(1.0 / exp_info["hpf_cutoff"], c=".3", ls=":", lw=1.5)
            ax.fill_between(fs, pxx_filt, color=".5")

            # Label the plot
            ax.set(xlabel="Frequency",
                   ylabel="Spectral Density",
                   xlim=(0, .15))
            plt.tight_layout()

            # Save the plot
            fname = op.abspath("cope%d_filter.png" % i)
            f.savefig(fname, dpi=100)
            plt.close(f)
            report.append(fname)

        # Store the report files for later
        self.report_files = report

    def _list_outputs(self):

        outputs = self._outputs().get()
        outputs["report"] = self.report_files
        outputs["contrast_file"] = op.abspath("design.con")
        outputs["design_matrix_pkl"] = op.abspath("design.pkl")
        outputs["design_matrix_file"] = op.abspath("design.mat")
        return outputs


class ModelSummaryInput(BaseInterfaceInputSpec):

    design_matrix_pkl = File(exists=True)
    timeseries = File(exists=True)
    pe_files = InputMultiPath(File(exists=True))


class ModelSummaryOutput(TraitedSpec):

    r2_files = OutputMultiPath(File(exists=True))
    ss_files = OutputMultiPath(File(exists=True))
    tsnr_file = File(exists=True)


class ModelSummary(BaseInterface):

    input_spec = ModelSummaryInput
    output_spec = ModelSummaryOutput

    def _run_interface(self, runtime):

        # Load the design matrix object
        X = glm.DesignMatrix.from_pickle(self.inputs.design_matrix_pkl)

        # Load and de-mean the timeseries
        ts_img = nib.load(self.inputs.timeseries)
        ts_aff, ts_header = ts_img.get_affine(), ts_img.get_header()
        y = ts_img.get_data()
        ybar = y.mean(axis=-1)[..., np.newaxis]
        y -= ybar
        self.y = y

        # Store the image attributes
        self.affine = ts_aff
        self.header = ts_header

        # Load the parameter estimates, make 4D, and concatenate
        pes = [nib.load(f).get_data() for f in self.inputs.pe_files]
        pes = [pe[..., np.newaxis] for pe in pes]
        pes = np.concatenate(pes, axis=-1)

        # Compute and save the total sum of squares
        self.sstot = np.sum(np.square(y), axis=-1)
        self.save_image(self.sstot, "sstot")

        # Compute the full model r squared
        yhat_full = self.dot_by_slice(X, pes)
        ss_full, r2_full = self.compute_r2(yhat_full)
        self.save_image(ss_full, "ssres_full")
        self.save_image(r2_full, "r2_full")
        del yhat_full, r2_full

        # Compute the main model r squared
        yhat_main = self.dot_by_slice(X, pes, "main")
        ss_main, r2_main = self.compute_r2(yhat_main)
        self.save_image(ss_main, "ssres_main")
        self.save_image(r2_main, "r2_main")
        del yhat_main, r2_main

        # Compute the confound model r squared
        yhat_confound = self.dot_by_slice(X, pes, "confound")
        _, r2_confound = self.compute_r2(yhat_confound)
        self.save_image(r2_confound, "r2_confound")
        del yhat_confound, r2_confound

        # Compute and save the residual tSNR
        std = np.sqrt(ss_full / len(y))
        tsnr = np.squeeze(ybar) / std
        tsnr = np.nan_to_num(tsnr)
        self.save_image(tsnr, "tsnr")

        return runtime

    def save_image(self, data, fname):
        """Save data to the output structure."""
        img = nib.Nifti1Image(data, self.affine, self.header)
        img.to_filename(fname + ".nii.gz")

    def dot_by_slice(self, X, pes, component=None):
        """Broadcast a dot product by image slices to balance speed/memory."""
        if component is not None:
            pes = pes * getattr(X, component + "_vector").T[np.newaxis,
                                                            np.newaxis, :, :]
        # Set up the output data structure
        n_x, n_y, n_z, n_pe = pes.shape
        n_t = X.design_matrix.shape[0]
        out = np.empty((n_x, n_y, n_z, n_t))

        # Do the dot product, broadcasted for each Z slice
        for k in range(n_z):
            slice_pe = pes[:, :, k, :].reshape(-1, n_pe).T
            slice_dot = X.design_matrix.values.dot(slice_pe)
            out[:, :, k, :] = slice_dot.T.reshape(n_x, n_y, n_t)

        return out

    def compute_r2(self, yhat):
        """Efficiently compute the coefficient of variation."""
        ssres = np.zeros_like(self.sstot)
        n_frames = yhat.shape[-1]
        for tr in xrange(n_frames):
            ssres += np.square(yhat[..., tr] - self.y[..., tr])
        r2 = 1 - ssres / self.sstot
        return ssres, r2

    def _list_outputs(self):

        outputs = self._outputs().get()

        outputs["r2_files"] = [op.abspath("r2_full.nii.gz"),
                               op.abspath("r2_main.nii.gz"),
                               op.abspath("r2_confound.nii.gz")]
        outputs["ss_files"] = [op.abspath("sstot.nii.gz"),
                               op.abspath("ssres_full.nii.gz"),
                               op.abspath("ssres_main.nii.gz")]
        outputs["tsnr_file"] = op.abspath("tsnr.nii.gz")

        return outputs


class ModelReportInput(BaseInterfaceInputSpec):

    timeseries = File(exists=True)
    sigmasquareds_file = File(exists=True)
    tsnr_file = File(exists=True)
    zstat_files = InputMultiPath(File(exists=True))
    r2_files = InputMultiPath(File(exists=True))


class ModelReport(BaseInterface):

    input_spec = ModelReportInput
    output_spec = ManyOutFiles

    def _run_interface(self, runtime):

        # Load the sigmasquareds and use it to infer the model mask
        var_img = nib.load(self.inputs.sigmasquareds_file).get_data()
        self.mask = (var_img > 0).astype(np.int16)

        # Load the timeseries and take the mean over time for a background
        ts_img = nib.load(self.inputs.timeseries)
        self.mean = nib.Nifti1Image(ts_img.get_data().mean(axis=-1),
                                    ts_img.get_affine(),
                                    ts_img.get_header())

        # Set up the output list
        self.out_files = []

        # Plot the data
        self.plot_residuals()
        self.plot_rsquareds()
        self.plot_tsnr()
        if isdefined(self.inputs.zstat_files):
            self.plot_zstats()

        return runtime

    def plot_residuals(self):
        """Plot the variance of the model residuals across time."""
        ss = self.inputs.sigmasquareds_file
        m = Mosaic(self.mean, ss, self.mask, step=1)
        m.plot_overlay("cube:.8:.2", 0, alpha=.6, fmt="%d")
        png_name = nii_to_png(ss)
        m.savefig(png_name)
        m.close()
        self.out_files.append(png_name)

    def plot_tsnr(self):

        tsnr = self.inputs.tsnr_file
        m = Mosaic(self.mean, tsnr, self.mask, step=1)
        m.plot_overlay("cube:1.9:.5", 0, alpha=1, fmt="%d")
        png_name = nii_to_png(tsnr)
        m.savefig(png_name)
        m.close()
        self.out_files.append(png_name)

    def plot_rsquareds(self):
        """Plot the full, main, and confound R squared maps."""
        cmaps = ["cube:2:0", "cube:2.6:0", "cube:1.5:0"]
        for r2_file, cmap in zip(self.inputs.r2_files, cmaps):
            m = Mosaic(self.mean, r2_file, self.mask, step=1)
            m.plot_overlay(cmap, 0, alpha=.6)
            png_name = nii_to_png(r2_file)
            m.savefig(png_name)
            m.close()
            self.out_files.append(png_name)

    def plot_zstats(self):
        """Plot the positive and negative z stats with a low threshold."""
        for z_file in self.inputs.zstat_files:
            m = Mosaic(self.mean, z_file, self.mask, step=1)
            m.plot_activation(pos_cmap="Reds_r", neg_cmap="Blues",
                              thresh=1.7, alpha=.85)
            png_name = nii_to_png(z_file)
            m.savefig(png_name)
            m.close()
            self.out_files.append(png_name)

    def _list_outputs(self):

        outputs = self._outputs().get()
        outputs["out_files"] = self.out_files
        return outputs
