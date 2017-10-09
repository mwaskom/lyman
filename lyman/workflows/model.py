from __future__ import division
import os
import os.path as op

import numpy as np
import pandas as pd
import nibabel as nib

from nipype import Workflow, Node, JoinNode, IdentityInterface, DataSink
from nipype.interfaces.base import traits, TraitedSpec, Bunch

from .. import glm, signals, surface
from ..utils import LymanInterface, SaveInfo, image_to_matrix, matrix_to_image
from ..visualizations import Mosaic, CarpetPlot, plot_design_matrix


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
                                    data_dir=info.data_dir,
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
             ("mask_file", "mask_file"),
             ("ts_file", "ts_file"),
             ("noise_file", "noise_file"),
             ("mc_file", "mc_file"),
             ("mesh_files", "mesh_files")]),

        (data_input, data_output,
            [("output_path", "container")]),
        (fit_model, data_output,
            [("mask_file", "@mask"),
             ("beta_file", "@beta"),
             ("error_file", "@error"),
             ("ols_file", "@ols"),
             ("resid_file", "@resid"),
             ("design_file", "@design")]),

    ]
    workflow.connect(processing_edges)

    qc_edges = [

        (run_source, save_info,
            [("run", "parameterization")]),
        (save_info, data_output,
            [("info_file", "qc.@info_json")]),

        (fit_model, data_output,
            [("design_plot", "qc.@design_plot"),
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
             ("design_file", "design_file")]),

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
        data_dir = traits.Directory(exists=True)
        proc_dir = traits.Directory(exists=True)
        subject = traits.Str()
        run_tuple = traits.Tuple(traits.Str(), traits.Str())

    class output_spec(TraitedSpec):
        subject = traits.Str()
        session = traits.Str()
        run = traits.Str()
        seg_file = traits.File(exists=True)
        surf_file = traits.File(exists=True)
        mask_file = traits.File(exists=True)
        ts_file = traits.File(exists=True)
        noise_file = traits.File(Exists=True)
        mc_file = traits.File(exists=True)
        mesh_files = traits.Tuple(traits.File(exists=True),
                                  traits.File(exists=True))
        output_path = traits.Directory()

    def _run_interface(self, runtime):

        subject = self.inputs.subject
        session, run = self.inputs.run_tuple
        run_key = "{}_{}".format(session, run)

        experiment = self.inputs.experiment
        model = self.inputs.model
        data_dir = self.inputs.data_dir
        proc_dir = self.inputs.proc_dir

        surface_path = op.join(data_dir, subject, "surf")
        template_path = op.join(proc_dir, subject, "template")
        timeseries_path = op.join(proc_dir, subject, experiment,
                                  "timeseries", run_key)

        results = dict(

            subject=subject,
            session=session,
            run=run,

            seg_file=op.join(template_path, "seg.nii.gz"),
            surf_file=op.join(template_path, "surf.nii.gz"),

            mask_file=op.join(timeseries_path, "mask.nii.gz"),
            ts_file=op.join(timeseries_path, "func.nii.gz"),
            noise_file=op.join(timeseries_path, "noise.nii.gz"),
            mc_file=op.join(timeseries_path, "mc.csv"),

            mesh_files=(op.join(surface_path, "lh.graymid"),
                        op.join(surface_path, "rh.graymid")),

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
        design_file = traits.File(exists=True)
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
            design_file=op.join(model_path, "design.csv"),

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
        noise_file = traits.File(exists=True)
        mc_file = traits.File(exists=True)
        mesh_files = traits.Tuple(traits.File(exists=True),
                                  traits.File(exists=True))

    class output_spec(TraitedSpec):
        mask_file = traits.File(exists=True)
        beta_file = traits.File(exists=True)
        error_file = traits.File(exists=True)
        ols_file = traits.File(exists=True)
        resid_file = traits.File()
        design_file = traits.File(exists=True)
        resid_plot = traits.File(exists=True)
        design_plot = traits.File(exists=True)
        error_plot = traits.File(exists=True)

    def _run_interface(self, runtime):

        subject = self.inputs.subject
        session = self.inputs.session
        run = self.inputs.run
        info = Bunch(self.inputs.info)
        data_dir = self.inputs.data_dir

        # --- Data loading

        # Load the timeseries
        ts_img = nib.load(self.inputs.ts_file)
        affine, header = ts_img.affine, ts_img.header
        n_tp = ts_img.shape[-1]

        # Load the anatomical segmentation and fine analysis mask
        run_mask = nib.load(self.inputs.mask_file).get_data() > 0

        seg_img = nib.load(self.inputs.seg_file)
        seg = seg_img.get_data()
        gray_seg = (seg > 0) & (seg < 5)

        vert_img = nib.load(self.inputs.surf_file)
        ribbon = vert_img.get_data().max(axis=-1) > -1

        mask = (ribbon | gray_seg) & run_mask
        n_vox = mask.sum()
        mask_img = nib.Nifti1Image(mask.astype(np.int8), affine, header)

        # --- Spatial filtering

        fwhm = info.smooth_fwhm

        # Load the noise segmentation
        noise_img = None
        if info.interpolate_noise:
            noise_img = nib.load(self.inputs.noise_file)

        # Volumetric smoothing
        # TODO use smooth_segmentation instead?
        filt_img = signals.smooth_volume(ts_img, fwhm, mask_img, noise_img)

        # Cortical maniforl smoothing
        if info.surface_smoothing:

            vert_data = nib.load(self.inputs.surf_file).get_data()
            ribbon = (vert_data > -1).any(axis=-1)

            for i, mesh_file in enumerate(self.inputs.mesh_files):
                sm = surface.SurfaceMeasure.from_file(mesh_file)
                vert_img = nib.Nifti1Image(vert_data[..., i], affine)
                signals.smooth_surface(ts_img, vert_img, sm, fwhm, noise_img,
                                       inplace=True)

            filt_img.get_data()[ribbon] = ts_img.get_data()[ribbon]

        ts_img = filt_img

        # Compute the mean image for later
        # TODO limit to gray matter voxels?
        data = ts_img.get_data()
        mean = data.mean(axis=-1)
        mean_img = nib.Nifti1Image(mean, affine, header)

        # --- Temporal filtering

        # Temporally filter the data
        hpf_matrix = glm.highpass_filter_matrix(n_tp,
                                                info.hpf_cutoff,
                                                info.tr)
        data[mask] = np.dot(hpf_matrix, data[mask].T).T
        data[mask] -= data[mask].mean(axis=-1, keepdims=True)

        # TODO remove the mean from the data
        # data[gray_mask] += mean[gray_mask, np.newaxis]
        data[~mask] = 0

        # --- Confound extraction

        # Define confound regressons from various sources
        # TODO
        mc_data = pd.read_csv(self.inputs.mc_file)

        # Detect artifact frames
        # TODO

        # --- Design construction

        # Get the design information for this run
        design_file = op.join(data_dir, subject, "design",
                              info.model_name + ".csv")
        design = pd.read_csv(design_file)
        run_rows = (design.session == session) & (design.run == run)
        design = design.loc[run_rows]
        # TODO better error when this fails (maybe check earlier too)
        assert len(design) > 0

        # Build the design matrix
        hrf_model = glm.GammaHRF(derivative=info.hrf_derivative)
        X = glm.build_design_matrix(design, hrf_model,
                                    n_tp=n_tp, tr=info.tr,
                                    hpf_matrix=hpf_matrix)

        # Save out the design matrix
        design_file = self.define_output("design_file", "design.csv")
        X.to_csv(design_file, index=False)

        # --- Model estimation

        # Convert to percent signal change?
        # TODO

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
        resid_data += np.expand_dims(mean * mask, axis=-1)
        resid_data[mask] += E.T
        resid_img = nib.Nifti1Image(resid_data, affine, header)

        p = CarpetPlot(resid_img, seg_img, mc_data)
        self.write_visualization("resid_plot", "resid.png", p)

        # Plot the deisgn matrix
        f = plot_design_matrix(X)
        self.write_visualization("design_plot", "design.png", f)

        # Plot the sigma squares image for QC
        error_m = Mosaic(mean_img, error_img, mask_img)
        error_m.plot_overlay("cube:.8:.2", 0, fmt=".0f")
        self.write_visualization("error_plot", "error.png", error_m)

        return runtime


class EstimateContrasts(LymanInterface):

    class input_spec(TraitedSpec):
        info = traits.Dict()
        mask_file = traits.File(exists=True)
        beta_file = traits.File(exists=True)
        ols_file = traits.File(exists=True)
        error_file = traits.File(exists=True)
        design_file = traits.File(exists=True)

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

        X = pd.read_csv(self.inputs.design_file)
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
                    con_idx = run_names.index(name)
                    con_frames.append(con_images[run].get_data()[..., con_idx])
                    var_frames.append(var_images[run].get_data()[..., con_idx])

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
            stat_m = Mosaic(anat_img, t_img, mask_img, show_mask=True)
            stat_m.plot_overlay("coolwarm", -10, 10)
            stat_m.savefig(op.join(name, "qc", "tstat.png"), close=True)

            # Analysis mask
            mask_m = Mosaic(anat_img, mask_img)
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
