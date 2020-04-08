from __future__ import division
import os
import os.path as op

import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage

from nipype import Workflow, Node, JoinNode, IdentityInterface, DataSink
from nipype.interfaces.base import traits, TraitedSpec, Bunch

from .. import glm, signals
from ..utils import LymanInterface, SaveInfo, image_to_matrix, matrix_to_image
from ..visualizations import (Mosaic, CarpetPlot,
                              plot_design_matrix, plot_nuisance_variables)


def define_model_fit_workflow(info, subjects, sessions, qc=True):

    # --- Workflow parameterization and data input

    # We just need two levels of iterables here: one subject-level and
    # one "flat" run-level iterable (i.e. all runs collapsing over
    # sessions). But we want to be able to specify sessions to process.

    scan_info = info.scan_info
    experiment = info.experiment_name
    model = info.model_name

    iterables = generate_iterables(scan_info, experiment, subjects, sessions)
    subject_iterables, run_iterables = iterables

    subject_source = Node(IdentityInterface(["subject"]),
                          name="subject_source",
                          iterables=("subject", subject_iterables))

    run_source = Node(IdentityInterface(["subject", "run"]),
                      name="run_source",
                      itersource=("subject_source", "subject"),
                      iterables=("run", run_iterables))

    data_input = Node(ModelFitInput(experiment=experiment,
                                    model=model,
                                    proc_dir=info.proc_dir),
                      "data_input")

    # --- Data filtering and model fitting

    fit_model = Node(ModelFit(data_dir=info.data_dir,
                              info=info.trait_get()),
                     "fit_model")

    # --- Data output

    save_info = Node(SaveInfo(info_dict=info.trait_get()), "save_info")

    data_output = Node(DataSink(base_directory=info.proc_dir,
                                parameterization=False),
                       "data_output")

    # === Assemble pipeline

    cache_base = op.join(info.cache_dir, experiment)
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
             ("edge_file", "edge_file"),
             ("mask_file", "mask_file"),
             ("ts_file", "ts_file"),
             ("noise_file", "noise_file"),
             ("mc_file", "mc_file")]),

        (data_input, data_output,
            [("output_path", "container")]),
        (fit_model, data_output,
            [("mask_file", "@mask"),
             ("beta_file", "@beta"),
             ("error_file", "@error"),
             ("ols_file", "@ols"),
             ("resid_file", "@resid"),
             ("model_file", "@model")]),

    ]
    workflow.connect(processing_edges)

    qc_edges = [

        (run_source, save_info,
            [("run", "parameterization")]),
        (save_info, data_output,
            [("info_file", "qc.@info_json")]),

        (fit_model, data_output,
            [("model_plot", "qc.@model_plot"),
             ("nuisance_plot", "qc.@nuisance_plot"),
             ("resid_plot", "qc.@resid_plot"),
             ("error_plot", "qc.@error_plot")]),

    ]
    if qc:
        workflow.connect(qc_edges)

    return workflow


def define_model_results_workflow(info, subjects, qc=True):

    # TODO I am copying a lot from above ...

    # --- Workflow parameterization and data input

    # We just need two levels of iterables here: one subject-level and
    # one "flat" run-level iterable (i.e. all runs collapsing over
    # sessions). Unlike in the model fit workflow, we always want to process
    # all sessions.

    scan_info = info.scan_info
    experiment = info.experiment_name
    model = info.model_name

    iterables = generate_iterables(scan_info, experiment, subjects)
    subject_iterables, run_iterables = iterables

    subject_source = Node(IdentityInterface(["subject"]),
                          name="subject_source",
                          iterables=("subject", subject_iterables))

    run_source = Node(IdentityInterface(["subject", "run"]),
                      name="run_source",
                      itersource=("subject_source", "subject"),
                      iterables=("run", run_iterables))

    data_input = Node(ModelResultsInput(experiment=experiment,
                                        model=model,
                                        proc_dir=info.proc_dir),
                      "data_input")

    # --- Run-level contrast estimation

    estimate_contrasts = Node(EstimateContrasts(info=info.trait_get()),
                              "estimate_contrasts")

    # --- Subject-level contrast estimation

    model_results = JoinNode(ModelResults(info=info.trait_get()),
                             name="model_results",
                             joinsource="run_source",
                             joinfield=["contrast_files",
                                        "variance_files",
                                        "name_files"])

    # --- Data output

    save_info = Node(SaveInfo(info_dict=info.trait_get()), "save_info")

    run_output = Node(DataSink(base_directory=info.proc_dir,
                               parameterization=False),
                      "run_output")

    results_path = Node(ModelResultsPath(proc_dir=info.proc_dir,
                                         experiment=experiment,
                                         model=model),
                        "results_path")

    subject_output = Node(DataSink(base_directory=info.proc_dir,
                                   parameterization=False),
                          "subject_output")

    # === Assemble pipeline

    cache_base = op.join(info.cache_dir, experiment)
    workflow = Workflow(name="model_results", base_dir=cache_base)

    # Connect processing nodes

    processing_edges = [

        (subject_source, run_source,
            [("subject", "subject")]),
        (subject_source, data_input,
            [("subject", "subject")]),
        (run_source, data_input,
            [("run", "run_tuple")]),

        (data_input, estimate_contrasts,
            [("mask_file", "mask_file"),
             ("beta_file", "beta_file"),
             ("error_file", "error_file"),
             ("ols_file", "ols_file"),
             ("model_file", "model_file")]),

        (subject_source, model_results,
            [("subject", "subject")]),
        (data_input, model_results,
            [("anat_file", "anat_file")]),
        (estimate_contrasts, model_results,
            [("contrast_file", "contrast_files"),
             ("variance_file", "variance_files"),
             ("name_file", "name_files")]),

        (run_source, save_info,
            [("run", "parameterization")]),
        (save_info, run_output,
            [("info_file", "qc.@info_json")]),

        (data_input, run_output,
            [("output_path", "container")]),
        (estimate_contrasts, run_output,
            [("contrast_file", "@contrast"),
             ("variance_file", "@variance"),
             ("tstat_file", "@tstat"),
             ("name_file", "@names")]),

        (subject_source, results_path,
            [("subject", "subject")]),
        (results_path, subject_output,
            [("output_path", "container")]),
        (model_results, subject_output,
            [("result_directories", "@results")]),

    ]
    workflow.connect(processing_edges)

    return workflow


# =========================================================================== #
# Custom processing code
# =========================================================================== #


def generate_iterables(scan_info, experiment, subjects, sessions=None):
    """Return lists of variables for model workflow iterables.

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
    run_iterables : dict
        A dictionary where keys are subject ids and values of lists of
        (session id, run id) pairs.

    """
    subject_iterables = []
    run_iterables = {}

    for subject in subjects:

        subject_run_iterables = []

        for session in scan_info[subject]:

            if sessions is not None and session not in sessions:
                continue

            sess_runs = scan_info[subject][session].get(experiment, [])
            run_tuples = [(session, run) for run in sess_runs]
            subject_run_iterables.extend(run_tuples)

        if subject_run_iterables:
            subject_iterables.append(subject)
            run_iterables[subject] = subject_run_iterables

    return subject_iterables, run_iterables


# --- Workflow inputs


class ModelFitInput(LymanInterface):

    class input_spec(TraitedSpec):
        experiment = traits.Str()
        model = traits.Str()
        proc_dir = traits.Directory(exists=True)
        subject = traits.Str()
        run_tuple = traits.Tuple(traits.Str(), traits.Str())

    class output_spec(TraitedSpec):
        subject = traits.Str()
        session = traits.Str()
        run = traits.Str()
        seg_file = traits.File(exists=True)
        surf_file = traits.File(exists=True)
        edge_file = traits.File(exists=True)
        mask_file = traits.File(exists=True)
        ts_file = traits.File(exists=True)
        noise_file = traits.File(Exists=True)
        mc_file = traits.File(exists=True)
        output_path = traits.Directory()

    def _run_interface(self, runtime):

        subject = self.inputs.subject
        session, run = self.inputs.run_tuple
        run_key = "{}_{}".format(session, run)

        experiment = self.inputs.experiment
        model = self.inputs.model
        proc_dir = self.inputs.proc_dir

        template_path = op.join(proc_dir, subject, "template")
        timeseries_path = op.join(proc_dir, subject, experiment,
                                  "timeseries", run_key)

        results = dict(

            subject=subject,
            session=session,
            run=run,

            seg_file=op.join(template_path, "seg.nii.gz"),
            surf_file=op.join(template_path, "surf.nii.gz"),
            edge_file=op.join(template_path, "edge.nii.gz"),

            mask_file=op.join(timeseries_path, "mask.nii.gz"),
            ts_file=op.join(timeseries_path, "func.nii.gz"),
            noise_file=op.join(timeseries_path, "noise.nii.gz"),
            mc_file=op.join(timeseries_path, "mc.csv"),

            output_path=op.join(proc_dir, subject, experiment, model, run_key)
        )
        self._results.update(results)

        return runtime


class ModelResultsInput(LymanInterface):

    class input_spec(TraitedSpec):
        experiment = traits.Str()
        model = traits.Str()
        proc_dir = traits.Directory(exists=True)
        subject = traits.Str()
        run_tuple = traits.Tuple(traits.Str(), traits.Str())

    class output_spec(TraitedSpec):
        subject = traits.Str()
        session = traits.Str()
        run = traits.Str()
        anat_file = traits.File(exists=True)
        mask_file = traits.File(exists=True)
        beta_file = traits.File(exists=True)
        ols_file = traits.File(exists=True)
        error_file = traits.File(exists=True)
        model_file = traits.File(exists=True)
        output_path = traits.Directory()

    def _run_interface(self, runtime):

        subject = self.inputs.subject
        session, run = self.inputs.run_tuple
        run_key = "{}_{}".format(session, run)

        experiment = self.inputs.experiment
        model = self.inputs.model
        proc_dir = self.inputs.proc_dir

        template_path = op.join(proc_dir, subject, "template")
        model_path = op.join(proc_dir, subject, experiment, model, run_key)

        results = dict(

            subject=subject,
            session=session,
            run=run,

            anat_file=op.join(template_path, "anat.nii.gz"),

            mask_file=op.join(model_path, "mask.nii.gz"),
            beta_file=op.join(model_path, "beta.nii.gz"),
            ols_file=op.join(model_path, "ols.nii.gz"),
            error_file=op.join(model_path, "error.nii.gz"),
            model_file=op.join(model_path, "model.csv"),

            output_path=model_path,
        )
        self._results.update(results)

        return runtime


# --- Model estimation code


class ModelFit(LymanInterface):

    class input_spec(TraitedSpec):
        subject = traits.Str()
        session = traits.Str()
        run = traits.Str()
        data_dir = traits.Directory(exists=True)
        info = traits.Dict()
        seg_file = traits.File(exists=True)
        surf_file = traits.File(exists=True)
        ts_file = traits.File(exists=True)
        mask_file = traits.File(exists=True)
        edge_file = traits.File(exists=True)
        noise_file = traits.File(exists=True)
        mc_file = traits.File(exists=True)
        wm_erode = traits.Int(default=2)
        csf_erode = traits.Int(default=1)

    class output_spec(TraitedSpec):
        mask_file = traits.File(exists=True)
        beta_file = traits.File(exists=True)
        error_file = traits.File(exists=True)
        ols_file = traits.File(exists=True)
        resid_file = traits.File()
        model_file = traits.File(exists=True)
        resid_plot = traits.File(exists=True)
        model_plot = traits.File(exists=True)
        error_plot = traits.File(exists=True)
        nuisance_plot = traits.File(exists=True)

    def _run_interface(self, runtime):

        subject = self.inputs.subject
        session = self.inputs.session
        run = self.inputs.run
        info = Bunch(self.inputs.info)
        data_dir = self.inputs.data_dir
        qc_title = "{} {} {}".format(subject, session, run)

        # --- Design loading

        # Do this first because problems with the design are more
        # common than problems with the preprocessed image data

        design = None
        if info.task_model:
            # Get the design information for this run
            fname = "{}-{}.csv".format(info.experiment_name, info.model_name)
            design_file = op.join(data_dir, subject, "design", fname)
            design = pd.read_csv(design_file)
            run_rows = (design.session == session) & (design.run == run)
            design = design.loc[run_rows]
            assert len(design), "Design file has no rows"

        # --- Data loading

        # Load the timeseries
        ts_img = nib.load(self.inputs.ts_file)
        affine, header = ts_img.affine, ts_img.header
        n_tp = ts_img.shape[-1]

        # Load the anatomical segmentation and find analysis mask
        seg_img = nib.load(self.inputs.seg_file)
        mask_img = nib.load(self.inputs.mask_file)

        seg = seg_img.get_fdata()
        mask = mask_img.get_fdata()
        mask = (mask > 0) & (seg > 0) & (seg < 5)
        mask_img = nib.Nifti1Image(mask.astype(np.uint8), affine, header)
        n_vox = mask.sum()

        # Load the map of locally-noisy voxels
        noise_img = nib.load(self.inputs.noise_file)

        # --- Nuisance variable extraction

        # Erode the WM mask and extract data
        wm_mask = np.isin(seg, [5, 6, 7])
        erode = self.inputs.wm_erode
        wm_mask = (wm_mask if not erode
                   else ndimage.binary_erosion(wm_mask, iterations=erode))
        wm_comp = info.nuisance_components.get("wm", 0)
        if wm_mask.any() and wm_comp:
            wm_img = nib.Nifti1Image(wm_mask.astype(np.uint8), affine)
            wm_data = image_to_matrix(ts_img, wm_img)
            wm_pca = signals.pca_transform(wm_data, wm_comp)
        else:
            wm_pca = None

        # Erode the CSF mask and extract data
        csf_mask = seg == 8
        erode = self.inputs.csf_erode
        csf_mask = (csf_mask if not erode
                    else ndimage.binary_erosion(csf_mask, iterations=erode))
        csf_comp = info.nuisance_components.get("csf", 0)
        if csf_mask.any() and csf_comp:
            csf_img = nib.Nifti1Image(csf_mask.astype(np.uint8), affine)
            csf_data = image_to_matrix(ts_img, csf_img)
            csf_pca = signals.pca_transform(csf_data, csf_comp)
        else:
            csf_pca = None

        # Extract data from the "edge" of the brain
        edge_img = nib.load(self.inputs.edge_file)
        edge_data = image_to_matrix(ts_img, edge_img)
        edge_comp = info.nuisance_components.get("edge", 0)
        if edge_comp:
            edge_pca = signals.pca_transform(edge_data, edge_comp)
        else:
            edge_pca = None

        # Extract data from "noisy" voxels
        noise_data = image_to_matrix(ts_img, noise_img)
        noise_comp = info.nuisance_components.get("noise", 0)
        if noise_comp:
            noise_pca = signals.pca_transform(noise_data, noise_comp)
        else:
            noise_pca = None

        # TODO motion correction parameters (do we still want this?)
        mc_data = pd.read_csv(self.inputs.mc_file)

        # TODO Detect frames for censoring

        # --- Spatial filtering

        fwhm = info.smooth_fwhm

        smooth_noise = noise_img if info.interpolate_noise else None

        # Volumetric smoothing
        filt_img = signals.smooth_segmentation(ts_img, seg_img,
                                               fwhm, smooth_noise)

        # Cortical manifold smoothing
        if info.surface_smoothing:

            vert_img = nib.load(self.inputs.surf_file)
            signals.smooth_surface(
                ts_img, vert_img, fwhm, subject,
                noise_img=smooth_noise, inplace=True,
            )

            ribbon = vert_img.get_fdata().max(axis=-1) > -1
            filt_data = filt_img.get_fdata()
            filt_data[ribbon] = ts_img.get_fdata()[ribbon]
            filt_img = nib.Nifti1Image(filt_data, affine, header)

        ts_img = filt_img

        # Compute the mean image for later
        # TODO limit to gray matter voxels?
        data = ts_img.get_fdata()
        mean = data.mean(axis=-1)
        mean_img = nib.Nifti1Image(mean, affine, header)

        # --- Temporal filtering

        # Temporally filter the data
        hpf_matrix = glm.highpass_filter_matrix(n_tp,
                                                info.hpf_cutoff,
                                                info.tr)
        data[mask] = np.dot(hpf_matrix, data[mask].T).T
        data[mask] -= data[mask].mean(axis=-1, keepdims=True)

        # Temporally filter the nuisance regressors
        wm_pca = None if wm_pca is None else hpf_matrix.dot(wm_pca)
        csf_pca = None if csf_pca is None else hpf_matrix.dot(csf_pca)
        edge_pca = None if edge_pca is None else hpf_matrix.dot(edge_pca)
        noise_pca = None if noise_pca is None else hpf_matrix.dot(noise_pca)

        # --- Design matrix construction

        # Build the regressor sub-matrix
        tps = np.arange(0, n_tp * info.tr, info.tr)
        wm_cols = [f"wm{i+1}" for i in range(wm_comp)]
        csf_cols = [f"csf{i+1}" for i in range(csf_comp)]
        edge_cols = [f"edge{i+1}" for i in range(edge_comp)]
        noise_cols = [f"noise{i+1}" for i in range(noise_comp)]

        regressors = pd.concat([
            pd.DataFrame(wm_pca, tps, wm_cols),
            pd.DataFrame(csf_pca, tps, csf_cols),
            pd.DataFrame(edge_pca, tps, edge_cols),
            pd.DataFrame(noise_pca, tps, noise_cols),
        ], axis=1).dropna()

        # Build the full design matrix
        hrf_model = glm.GammaBasis(time_derivative=info.hrf_derivative,
                                   disp_derivative=False)  # TODO?
        X = glm.build_design_matrix(design, hrf_model,
                                    regressors=regressors,
                                    n_tp=n_tp, tr=info.tr,
                                    hpf_matrix=hpf_matrix)

        # Save out the design matrix
        model_file = self.define_output("model_file", "model.csv")
        X.to_csv(model_file, index=False)

        # --- Model estimation

        data[~mask] = 0  # TODO why is this needed?

        # Convert to percent signal change?
        if info.percent_change:
            # TODO standarize the representation of mean in this method
            remeaned_data = data + mean[..., np.newaxis]
            data[mask] = signals.percent_change(remeaned_data[mask])

        # Prewhiten the data
        ts_img = nib.Nifti1Image(data, affine)
        WY, WX = glm.prewhiten_image_data(ts_img, mask_img, X.values)

        # Fit the final model
        B, SS, XtXinv, E = glm.iterative_ols_fit(WY, WX)

        # TODO should we re-compute the tSNR on the residuals?

        # --- Results output

        # Convert outputs to image format
        beta_img = matrix_to_image(B.T, mask_img)
        error_img = matrix_to_image(SS, mask_img)
        XtXinv_flat = XtXinv.reshape(n_vox, -1)
        ols_img = matrix_to_image(XtXinv_flat.T, mask_img)
        resid_img = matrix_to_image(E, mask_img, ts_img)

        # Write out the results
        self.write_image("mask_file", "mask.nii.gz", mask_img)
        self.write_image("beta_file", "beta.nii.gz", beta_img)
        self.write_image("error_file", "error.nii.gz", error_img)
        self.write_image("ols_file", "ols.nii.gz", ols_img)
        if info.save_residuals:
            self.write_image("resid_file", "resid.nii.gz", resid_img)

        # --- Quality control visualization

        # Make some QC plots
        # We want a version of the resid data with an intact mean so that
        # the carpet plot can compute percent signal change.
        # (Maybe carpetplot should accept a mean image and handle that
        # internally)?
        # TODO standarize the representation of mean in this method
        resid_data = np.zeros(ts_img.shape, np.float32)
        if not info.percent_change:
            resid_data += np.expand_dims(mean * mask, axis=-1)
        resid_data[mask] += E.T
        resid_img = nib.Nifti1Image(resid_data, affine, header)

        p = CarpetPlot(resid_img, seg_img, mc_data, title=qc_title,
                       percent_change=not info.percent_change)
        self.write_visualization("resid_plot", "resid.png", p)

        # Plot the design matrix
        f = plot_design_matrix(X, title=qc_title)
        self.write_visualization("model_plot", "model.png", f)

        # Plot the sigma squares image for QC
        error_m = Mosaic(mean_img, error_img, mask_img, title=qc_title)
        error_m.plot_overlay("cube:.8:.2", 0, fmt=".0f")
        self.write_visualization("error_plot", "error.png", error_m)

        # Plot the nuisance variables
        f = plot_nuisance_variables(X, title=qc_title)
        self.write_visualization("nuisance_plot", "nuisance.png", f)

        return runtime


class EstimateContrasts(LymanInterface):

    class input_spec(TraitedSpec):
        info = traits.Dict()
        mask_file = traits.File(exists=True)
        beta_file = traits.File(exists=True)
        ols_file = traits.File(exists=True)
        error_file = traits.File(exists=True)
        model_file = traits.File(exists=True)

    class output_spec(TraitedSpec):
        contrast_file = traits.File(exists=True)
        variance_file = traits.File(exists=True)
        tstat_file = traits.File(exists=True)
        name_file = traits.File(exists=True)

    def _run_interface(self, runtime):

        # Load model fit outputs
        mask_img = nib.load(self.inputs.mask_file)
        beta_img = nib.load(self.inputs.beta_file)
        error_img = nib.load(self.inputs.error_file)
        ols_img = nib.load(self.inputs.ols_file)

        B = image_to_matrix(beta_img, mask_img)
        SS = image_to_matrix(error_img, mask_img)
        XtXinv = image_to_matrix(ols_img, mask_img)

        X = pd.read_csv(self.inputs.model_file)
        param_names = X.columns

        # Reshape the matrix form data to what the glm functions expect
        # TODO the shape/orientation of model parameter matrices needs some
        # more thinking / standardization
        B = B.T
        n_vox, n_ev = B.shape
        XtXinv = XtXinv.reshape(n_ev, n_ev, n_vox).T

        # Obtain list of contrast matrices
        C = []
        names = []
        for contrast_spec in self.inputs.info["contrasts"]:
            name, params, _ = contrast_spec
            if set(params) <= set(param_names):
                C.append(glm.contrast_matrix(contrast_spec, X))
                names.append(name)

        # Estimate the contrasts, variances, and statistics in each voxel
        G, V, T = glm.iterative_contrast_estimation(B, SS, XtXinv, C)
        contrast_img = matrix_to_image(G.T, mask_img)
        variance_img = matrix_to_image(V.T, mask_img)
        tstat_img = matrix_to_image(T.T, mask_img)

        # Write out the output files
        self.write_image("contrast_file", "contrast.nii.gz", contrast_img)
        self.write_image("variance_file", "variance.nii.gz", variance_img)
        self.write_image("tstat_file", "tstat.nii.gz", tstat_img)
        name_file = self.define_output("name_file", "contrast.txt")
        np.savetxt(name_file, names, "%s")

        return runtime


class ModelResults(LymanInterface):

    class input_spec(TraitedSpec):
        info = traits.Dict()
        subject = traits.Str()
        anat_file = traits.File(exists=True)
        contrast_files = traits.List(traits.File(exists=True))
        variance_files = traits.List(traits.File(exists=True))
        name_files = traits.List(traits.File(exists=True))

    class output_spec(TraitedSpec):
        result_directories = traits.List(traits.Directory(exists=True))

    def _run_interface(self, runtime):

        info = Bunch(self.inputs.info)

        # Load the anatomical template to get image geometry information
        anat_img = nib.load(self.inputs.anat_file)
        affine, header = anat_img.affine, anat_img.header

        result_directories = []
        for i, contrast_tuple in enumerate(info.contrasts):

            name, _, _ = contrast_tuple
            qc_title = "{} {}".format(self.inputs.subject, name)

            result_directories.append(op.abspath(name))
            os.makedirs(op.join(name, "qc"))

            con_frames = []
            var_frames = []

            # Load the parameter and variance data for each run/contrast.
            con_images = [nib.load(f) for f in self.inputs.contrast_files]
            var_images = [nib.load(f) for f in self.inputs.variance_files]
            name_lists = [np.loadtxt(f, str).tolist()
                          for f in self.inputs.name_files]

            # Files are input as a list of 4D images where list entries are
            # runs and the last axis is contrast; we want to concatenate runs
            # for each contrast, so we need to transpose" the ordering
            # while matching against the list of contrast names for that run.
            for run, run_names in enumerate(name_lists):
                if name in run_names:
                    idx = run_names.index(name)
                    con_frames.append(con_images[run].get_fdata()[..., idx])
                    var_frames.append(var_images[run].get_fdata()[..., idx])

            con_data = np.stack(con_frames, axis=-1)
            var_data = np.stack(var_frames, axis=-1)

            # Define a mask as voxels with nonzero variance in each run
            # and extract voxel data as arrays
            mask = (var_data > 0).all(axis=-1)
            mask_img = nib.Nifti1Image(mask.astype(np.int8), affine, header)
            con = con_data[mask]
            var = var_data[mask]

            # Compute the higher-level fixed effects parameters
            con_ffx, var_ffx, t_ffx = glm.contrast_fixed_effects(con, var)

            # Convert to image volume format
            con_img = matrix_to_image(con_ffx.T, mask_img)
            var_img = matrix_to_image(var_ffx.T, mask_img)
            t_img = matrix_to_image(t_ffx.T, mask_img)

            # Write out output images
            con_img.to_filename(op.join(name, "contrast.nii.gz"))
            var_img.to_filename(op.join(name, "variance.nii.gz"))
            t_img.to_filename(op.join(name, "tstat.nii.gz"))
            mask_img.to_filename(op.join(name, "mask.nii.gz"))

            # Contrast t statistic overlay
            stat_m = Mosaic(anat_img, t_img, mask_img,
                            show_mask=True, title=qc_title)
            stat_m.plot_overlay("coolwarm", -10, 10)
            stat_m.savefig(op.join(name, "qc", "tstat.png"), close=True)

            # Analysis mask
            mask_m = Mosaic(anat_img, mask_img, title=qc_title)
            mask_m.plot_mask()
            mask_m.savefig(op.join(name, "qc", "mask.png"), close=True)

        # Output a list of directories with results.
        # This makes the connections in the workflow more opaque, but it
        # simplifies placing files in subdirectories named after contrasts.
        self._results["result_directories"] = result_directories

        return runtime


class ModelResultsPath(LymanInterface):

    class input_spec(TraitedSpec):
        proc_dir = traits.Directory(exists=True)
        subject = traits.Str()
        experiment = traits.Str()
        model = traits.Str()

    class output_spec(TraitedSpec):
        output_path = traits.Directory()

    def _run_interface(self, runtime):

        self._results["output_path"] = op.join(
            self.inputs.proc_dir,
            self.inputs.subject,
            self.inputs.experiment,
            self.inputs.model,
            "results"
        )
        return runtime
