from __future__ import division
import os
import os.path as op

import numpy as np
import pandas as pd
import nibabel as nib

from nipype import Workflow, Node, JoinNode, IdentityInterface, DataSink
from nipype.interfaces.base import traits, TraitedSpec

from moss import Bunch  # move into lyman
from moss import glm as mossglm  # TODO move into lyman

from .. import glm, signals, surface
from ..utils import LymanInterface, image_to_matrix, matrix_to_image
from ..visualizations import Mosaic, CarpetPlot


def define_model_fit_workflow(proj_info, exp_info, model_info,
                              subjects, sessions, qc=True):

    # --- Workflow parameterization and data input

    # We just need two levels of iterables here: one subject-level and
    # one "flat" run-level iterable (i.e. all runs collapsing over
    # sessions). But we want to be able to specify sessions to process.

    scan_info = proj_info.scan_info
    experiment = exp_info.name
    model = model_info.name

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
                                    data_dir=proj_info.data_dir,
                                    proc_dir=proj_info.proc_dir),
                      "data_input")

    # --- Data filtering and model fitting

    fit_model = Node(ModelFit(data_dir=proj_info.data_dir,
                              exp_info=exp_info,
                              model_info=model_info),
                     "fit_model")

    # --- Data output

    data_output = Node(DataSink(base_directory=proj_info.proc_dir,
                                parameterization=False),
                       "data_output")

    # === Assemble pipeline

    cache_base = op.join(proj_info.cache_dir, exp_info.name)
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

        (fit_model, data_output,
            [("design_plot", "qc.@design_plot"),
             ("resid_plot", "qc.@resid_plot"),
             ("error_plot", "qc.@error_plot")]),

    ]
    if qc:
        workflow.connect(qc_edges)

    return workflow


def define_model_results_workflow(proj_info, exp_info, model_info,
                                  subjects, qc=True):

    # TODO I am copying a lot from above ...

    # --- Workflow parameterization and data input

    # We just need two levels of iterables here: one subject-level and
    # one "flat" run-level iterable (i.e. all runs collapsing over
    # sessions). Unlike in the model fit workflow, we always want to process
    # all sessions.

    scan_info = proj_info.scan_info
    experiment = exp_info.name
    model = model_info.name

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
                                        proc_dir=proj_info.proc_dir),
                      "data_input")

    # --- Run-level contrast estimation

    estimate_contrasts = Node(EstimateContrasts(model_info=model_info),
                              "estimate_contrasts")

    # --- Subject-level contrast estimation

    model_results = JoinNode(ModelResults(model_info=model_info),
                             name="model_results",
                             joinsource="run_source",
                             joinfield=["contrast_files",
                                        "variance_files"])

    # --- Data output

    run_output = Node(DataSink(base_directory=proj_info.proc_dir,
                               parameterization=False),
                      "run_output")

    results_path = Node(ModelResultsPath(proc_dir=proj_info.proc_dir,
                                         experiment=experiment,
                                         model=model),
                        "results_path")

    subject_output = Node(DataSink(base_directory=proj_info.proc_dir,
                                   parameterization=False),
                          "subject_output")

    # === Assemble pipeline

    cache_base = op.join(proj_info.cache_dir, exp_info.name)
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
             ("ols_file", "ols_file")]),

        (data_input, model_results,
            [("anat_file", "anat_file")]),
        (estimate_contrasts, model_results,
            [("contrast_file", "contrast_files"),
             ("variance_file", "variance_files")]),

        (data_input, run_output,
            [("output_path", "container")]),
        (estimate_contrasts, run_output,
            [("contrast_file", "@contrast"),
             ("variance_file", "@variance"),
             ("tstat_file", "@tstat")]),

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
        exp_info = traits.Dict()
        model_info = traits.Dict()
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
        exp_info = Bunch(self.inputs.exp_info)
        model_info = Bunch(self.inputs.model_info)
        data_dir = self.inputs.data_dir

        # Load the timeseries
        ts_img = nib.load(self.inputs.ts_file)
        affine, header = ts_img.affine, ts_img.header

        # Load the anatomical segmentation and fine analysis mask
        run_mask = nib.load(self.inputs.mask_file).get_data() > 0
        seg_img = nib.load(self.inputs.seg_file)
        seg = seg_img.get_data()
        mask = (seg > 0) & (seg < 5) & run_mask
        n_vox = mask.sum()
        mask_img = nib.Nifti1Image(mask.astype(np.int8), affine, header)

        # Load the noise segmentation
        # TODO implement noisy voxel removal
        noise_img = nib.load(self.inputs.noise_file)

        # Spatially filter the data
        fwhm = model_info.smooth_fwhm
        # TODO use smooth_segmentation instead?
        signals.smooth_volume(ts_img, fwhm, mask_img, noise_img, inplace=True)

        if model_info.surface_smoothing:
            vert_data = nib.load(self.inputs.surf_file).get_data()
            for i, mesh_file in enumerate(self.inputs.mesh_files):
                sm = surface.SurfaceMeasure.from_file(mesh_file)
                vert_img = nib.Nifti1Image(vert_data[..., i], affine)
                signals.smooth_surface(ts_img, vert_img, sm, fwhm, noise_img,
                                       inplace=True)

        # Compute the mean image for later
        # TODO limit to gray matter voxels?
        data = ts_img.get_data()
        mean = data.mean(axis=-1)
        mean_img = nib.Nifti1Image(mean, affine, header)

        # Temporally filter the data
        n_tp = ts_img.shape[-1]
        hpf_matrix = glm.highpass_filter_matrix(n_tp,
                                                model_info.hpf_cutoff,
                                                exp_info.tr)
        data[mask] = np.dot(hpf_matrix, data[mask].T).T

        # TODO remove the mean from the data
        # data[gray_mask] += mean[gray_mask, np.newaxis]
        data[~mask] = 0  # TODO this is done within smoothing actually

        # Define confound regressons from various sources
        # TODO
        mc_data = pd.read_csv(self.inputs.mc_file)

        # Detect artifact frames
        # TODO

        # Convert to percent signal change?
        # TODO

        # Build the design matrix
        # TODO move out of moss and simplify
        design_file = op.join(data_dir, subject, "design",
                              model_info.name + ".csv")
        design = pd.read_csv(design_file)
        run_rows = (design.session == session) & (design.run == run)
        design = design.loc[run_rows]
        # TODO better error when this fails (maybe check earlier too)
        assert len(design) > 0
        dmat = mossglm.DesignMatrix(design, ntp=n_tp, tr=exp_info.tr)
        X = dmat.design_matrix.values

        # Save out the design matrix
        design_file = self.define_output("design_file", "design.csv")
        dmat.design_matrix.to_csv(design_file, index=False)

        # Prewhiten the data
        ts_img = nib.Nifti1Image(data, affine)
        WY, WX = glm.prewhiten_image_data(ts_img, mask_img, X)

        # Fit the final model
        B, SS, XtXinv, E = glm.iterative_ols_fit(WY, WX)

        # TODO should we re-compute the tSNR on the residuals?

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
        if model_info.save_residuals:
            self.write_image("resid_file", "resid.nii.gz", resid_img)

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
        # TODO update when improving design matrix code
        design_plot = self.define_output("design_plot", "design.png")
        dmat.plot(fname=design_plot, close=True)

        # Plot the sigma squares image for QC
        error_m = Mosaic(mean_img, error_img, mask_img)
        error_m.plot_overlay("cube:.8:.2", 0, fmt=".0f")
        self.write_visualization("error_plot", "error.png", error_m)

        return runtime


class EstimateContrasts(LymanInterface):

    class input_spec(TraitedSpec):
        exp_info = traits.Dict()
        model_info = traits.Dict()
        mask_file = traits.File(exists=True)
        beta_file = traits.File(exists=True)
        ols_file = traits.File(exists=True)
        error_file = traits.File(exists=True)

    class output_spec(TraitedSpec):
        contrast_file = traits.File(exists=True)
        variance_file = traits.File(exists=True)
        tstat_file = traits.File(exists=True)

    def _run_interface(self, runtime):

        # Load model fit outputs
        mask_img = nib.load(self.inputs.mask_file)
        beta_img = nib.load(self.inputs.beta_file)
        error_img = nib.load(self.inputs.error_file)
        ols_img = nib.load(self.inputs.ols_file)

        B = image_to_matrix(beta_img, mask_img)
        SS = image_to_matrix(error_img, mask_img)
        XtXinv = image_to_matrix(ols_img, mask_img)

        # Reshape the matrix form data to what the glm functions expect
        # TODO the shape/orientation of model parameter matrices needs some
        # more thinking / standardization
        B = B.T
        n_vox, n_ev = B.shape
        XtXinv = XtXinv.reshape(n_ev, n_ev, n_vox).T

        # Obtain list of contrast matrices
        # C = model_info.contrasts
        # TODO how are we going to do this? Hardcode for now.
        # TODO do we want to enforce vectors or do we ever want F stats
        C = [np.array([1, 0, 0, 0]),
             np.array([0, 1, 0, 0]),
             np.array([0, 0, 1, 0]),
             np.array([0, 0, 0, 1]),
             np.array([1, -1, 0, 0]),
             np.array([0, 0, 1, -1])]

        # TODO to get tests to run make this dumber but more flexible
        C = np.eye(B.shape[1])

        # Estimate the contrasts, variances, and statistics in each voxel
        G, V, T = glm.iterative_contrast_estimation(B, SS, XtXinv, C)
        contrast_img = matrix_to_image(G.T, mask_img)
        variance_img = matrix_to_image(V.T, mask_img)
        tstat_img = matrix_to_image(T.T, mask_img)

        # Write out the output files
        self.write_image("contrast_file", "contrast.nii.gz", contrast_img)
        self.write_image("variance_file", "variance.nii.gz", variance_img)
        self.write_image("tstat_file", "tstat.nii.gz", tstat_img)

        return runtime


class ModelResults(LymanInterface):

    class input_spec(TraitedSpec):
        model_info = traits.Dict()
        anat_file = traits.File(exists=True)
        contrast_files = traits.List(traits.File(exists=True))
        variance_files = traits.List(traits.File(exists=True))

    class output_spec(TraitedSpec):
        result_directories = traits.List(traits.Directory(exists=True))

    def _run_interface(self, runtime):

        model_info = Bunch(self.inputs.model_info)

        # Load the anatomical template to get image geometry information
        anat_img = nib.load(self.inputs.anat_file)
        affine, header = anat_img.affine, anat_img.header

        # TODO define contrasts properly accounting for missing EVs
        result_directories = []
        for i, contrast in enumerate(model_info.contrasts):

            result_directories.append(op.abspath(contrast))
            os.makedirs(op.join(contrast, "qc"))

            con_frames = []
            var_frames = []

            # Load the parameter and variance data for each run/contrast.
            con_images = map(nib.load, self.inputs.contrast_files)
            var_images = map(nib.load, self.inputs.variance_files)

            # Files are input as a list of 4D images where list entries are
            # runs and the last axis is contrast; we want to concatenate runs
            # for each contrast, so we need to transpose" the ordering.
            for con_img, var_img in zip(con_images, var_images):
                con_frames.append(con_img.get_data()[..., i])
                var_frames.append(var_img.get_data()[..., i])

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
            con_img.to_filename(op.join(contrast, "contrast.nii.gz"))
            var_img.to_filename(op.join(contrast, "variance.nii.gz"))
            t_img.to_filename(op.join(contrast, "tstat.nii.gz"))
            mask_img.to_filename(op.join(contrast, "mask.nii.gz"))

            # Contrast t statistic overlay
            stat_m = Mosaic(anat_img, t_img, mask_img, show_mask=True)
            stat_m.plot_overlay("coolwarm", -10, 10)
            stat_m.savefig(op.join(contrast, "qc", "tstat.png"), close=True)

            # Analysis mask
            mask_m = Mosaic(anat_img, mask_img)
            mask_m.plot_mask()
            mask_m.savefig(op.join(contrast, "qc", "mask.png"), close=True)

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
