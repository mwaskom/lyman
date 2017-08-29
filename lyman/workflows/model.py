from __future__ import division
import os
import os.path as op

import numpy as np
import pandas as pd
import nibabel as nib

from nipype import Workflow, Node, JoinNode, IdentityInterface, DataSink
from nipype.interfaces.base import traits, TraitedSpec

from moss import Bunch
from moss import glm as mossglm  # TODO move into lyman

from .. import glm, signals  # TODO confusingly close to scipy.signal
from ..mosaic import Mosaic
from ..graphutils import SimpleInterface


def define_model_fit_workflow(proj_info, subjects, session,
                              exp_info, model_info, qc=True):

    # --- Workflow parameterization and data input

    # We just need two levels of iterables here: one subject-level and
    # one "flat" run-level iterable (i.e. all runs collapsing over
    # sessions). But we want to be able to specify sessions to process.

    scan_info = proj_info.scan_info
    experiment = exp_info.name
    model = model_info.name

    iterables = generate_iterables(scan_info, experiment, subjects, session)
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
                                    analysis_dir=proj_info.analysis_dir),
                      "data_input")

    # --- Data filtering and model fitting

    fit_model = Node(ModelFit(data_dir=proj_info.data_dir,
                              exp_info=exp_info,
                              model_info=model_info),
                     "fit_model")

    # --- Data output

    data_output = Node(DataSink(base_directory=proj_info.analysis_dir,
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
             ("mc_file", "mc_file")]),

        (data_input, data_output,
            [("output_path", "container")]),
        (fit_model, data_output,
            [("beta_file", "@beta"),
             ("ols_file", "@ols"),
             ("sigsqr_file", "@sigsqr"),
             ("resid_file", "@resid"),
             ("design_file", "@design")]),

    ]
    workflow.connect(processing_edges)

    qc_edges = [

    ]
    if qc:
        workflow.connect(qc_edges)

    return workflow


def define_model_results_workflow(proj_info, subjects, session,
                                  exp_info, model_info, qc=True):

    # TODO I am copying a lot from above ...
    # --- Workflow parameterization and data input

    # We just need two levels of iterables here: one subject-level and
    # one "flat" run-level iterable (i.e. all runs collapsing over
    # sessions). But we want to be able to specify sessions to process.

    scan_info = proj_info.scan_info
    experiment = exp_info.name
    model = model_info.name

    iterables = generate_iterables(scan_info, experiment, subjects, session)
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
                                        analysis_dir=proj_info.analysis_dir),
                      "data_input")

    # --- Run-level contrast estimation

    estimate_contrasts = Node(EstimateContrasts(model_info=model_info),
                              "estimate_contrasts")

    # --- Subject-level contrast estimation

    model_results = JoinNode(ModelResults(model_info=model_info),
                             name="model_results",
                             joinsource="run_source",
                             joinfield=["contrast_files",
                                        "variance_files",
                                        "design_files"])

    # --- Data output

    run_output = Node(DataSink(base_directory=proj_info.analysis_dir,
                               parameterization=False),
                      "run_output")

    subject_output = Node(DataSink(base_directory=proj_info.analysis_dir,
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
            [("beta_file", "beta_file"),
             ("ols_file", "ols_file"),
             ("sigsqr_file", "sigsqr_file")]),

        (data_input, model_results,
            [("design_file", "design_files")]),
        (estimate_contrasts, model_results,
            [("contrast_file", "contrast_files"),
             ("variance_file", "variance_files")]),

        (data_input, run_output,
            [("run_output_path", "container")]),
        (estimate_contrasts, run_output,
            [("contrast_file", "@contrast"),
             ("variance_file", "@variance"),
             ("tstat_file", "@tstat")]),

        (data_input, subject_output,
            [("subject_output_path", "container")]),
        (model_results, subject_output,
            [("result_directories", "@results")]),

    ]
    workflow.connect(processing_edges)

    qc_edges = [

    ]
    if qc:
        workflow.connect(qc_edges)

    return workflow


# =========================================================================== #
# Custom processing code
# =========================================================================== #


def generate_iterables(scan_info, experiment, subjects, sessions=None):

    subject_iterables = subjects
    run_iterables = {}
    for subject in subjects:
        run_iterables[subject] = []
        for session in scan_info[subject]:
            if sessions is not None and session not in sessions:
                continue
            sess_runs = scan_info[subject][session].get(experiment, [])
            run_tuples = [(session, run) for run in sess_runs]
            run_iterables[subject].extend(run_tuples)

    return subject_iterables, run_iterables


# --- Workflow inputs


class ModelFitInput(SimpleInterface):

    class input_spec(TraitedSpec):
        experiment = traits.Str()
        model = traits.Str()
        analysis_dir = traits.Directory(exists=True)
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
        output_path = traits.Directory()

    def _run_interface(self, runtime):

        subject = self.inputs.subject
        session, run = self.inputs.run_tuple
        run_key = "{}_{}".format(session, run)

        experiment = self.inputs.experiment
        model = self.inputs.model
        anal_dir = self.inputs.analysis_dir

        template_path = op.join(anal_dir, subject, "template")
        timeseries_path = op.join(anal_dir, subject, experiment,
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

            output_path=op.join(anal_dir, subject, experiment, model, run_key)
        )
        self._results.update(results)

        return runtime


class ModelResultsInput(SimpleInterface):

    class input_spec(TraitedSpec):
        experiment = traits.Str()
        model = traits.Str()
        analysis_dir = traits.Directory(exists=True)
        subject = traits.Str()
        run_tuple = traits.Tuple(traits.Str(), traits.Str())

    class output_spec(TraitedSpec):
        subject = traits.Str()
        session = traits.Str()
        run = traits.Str()
        beta_file = traits.File(exists=True)
        ols_file = traits.File(exists=True)
        sigsqr_file = traits.File(exists=True)
        design_file = traits.File(exists=True)
        run_output_path = traits.Directory()
        subject_output_path = traits.Directory()

    def _run_interface(self, runtime):

        subject = self.inputs.subject
        session, run = self.inputs.run_tuple
        run_key = "{}_{}".format(session, run)

        experiment = self.inputs.experiment
        model = self.inputs.model
        anal_dir = self.inputs.analysis_dir

        model_path = op.join(anal_dir, subject, experiment, model, run_key)
        result_path = op.join(anal_dir, subject, experiment, model, "results")

        results = dict(

            subject=subject,
            session=session,
            run=run,

            beta_file=op.join(model_path, "beta.nii.gz"),
            ols_file=op.join(model_path, "ols.nii.gz"),
            sigsqr_file=op.join(model_path, "sigsqr.nii.gz"),
            design_file=op.join(model_path, "design.csv"),

            run_output_path=model_path,
            subject_output_path=result_path,
        )
        self._results.update(results)

        return runtime


# --- Model estimation code


class ModelFit(SimpleInterface):

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

    class output_spec(TraitedSpec):
        beta_file = traits.File(exists=True)
        ols_file = traits.File(exists=True)  # best name?
        sigsqr_file = traits.File(exists=True)  # maybe call "error_file"?
        resid_file = traits.File()  # TODO do we want?
        sigsqr_file = traits.File(exists=True)  # maybe call "error_file"?
        design_file = traits.File(exists=True)
        design_plot = traits.File(exists=True)

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

        # Load the anatomical segmentation and restrict to gray matter
        run_mask = nib.load(self.inputs.mask_file).get_data().astype(np.bool)
        seg_img = nib.load(self.inputs.seg_file)
        seg = seg_img.get_data()
        seg[~run_mask] = 0
        seg[seg > 4] = 0

        # Load the noise segmentation
        # TODO this will probably be conditional
        noise = nib.load(self.inputs.noise_file)

        # Spatially filter the data
        # TODO implement surface smoothing
        # Using simple volumetric smoothing for now to get things running
        fwhm = model_info.smooth_fwhm
        gray_mask = seg > 0
        mask_img = nib.Nifti1Image(gray_mask.astype(np.int), affine)
        ts_img = signals.smooth_volume(ts_img, fwhm, mask_img, noise)
        data = ts_img.get_data()

        # Compute the mean image for later
        mean = data.mean(axis=-1)

        # Temporally filter the data
        ntp = ts_img.shape[-1]
        hpf_matrix = glm.highpass_matrix(ntp,
                                         model_info.hpf_cutoff,
                                         exp_info.tr)
        data[gray_mask] = np.dot(hpf_matrix, data[gray_mask].T).T

        # TODO remove the mean from the data
        # data[gray_mask] += mean[gray_mask, np.newaxis]

        data[~gray_mask] = 0

        # Define confound regressons from various sources

        # TODO

        # Detect artifact frames

        # TODO

        # Convert to percent signal change?

        # TODO

        # Build the design matrix
        design_file = op.join(data_dir, subject, "design",
                              model_info.name + ".csv")
        design = pd.read_csv(design_file)
        run_rows = (design.session == session) & (design.run == run)
        design = design.loc[run_rows]
        # TODO better error when thisfails (maybe check earlier too)
        assert len(design) > 0
        dmat = mossglm.DesignMatrix(design, ntp=ntp, tr=exp_info.tr)
        X = dmat.design_matrix.values

        # Save out the design matrix
        design_file = self.define_output("design_file", "design.csv")
        dmat.design_matrix.to_csv(design_file)

        # Prewhiten the data
        assert not np.isnan(data).any()
        ts_img = nib.Nifti1Image(data, affine)
        WY, WX = glm.prewhiten_image_data(ts_img, X, mask_img)

        # Fit the final model
        B, XtXinv, SS, E = glm.iterative_ols_fit(WY, WX)

        # Generate output images
        nx, ny, nz, _ = ts_img.shape
        nev = X.shape[1]

        B_data = np.zeros((nx, ny, nz, nev))
        B_data[gray_mask] = B
        B_img = nib.Nifti1Image(B_data, affine, header)

        XtXinv_data = np.zeros((nx, ny, nz, nev, nev))
        XtXinv_data[gray_mask] = XtXinv
        XtXinv_img = nib.Nifti1Image(XtXinv_data, affine, header)

        SS_data = np.zeros((nx, ny, nz))
        SS_data[gray_mask] = SS
        SS_img = nib.Nifti1Image(SS_data, affine, header)

        # Make some QC plots

        # Write out the results
        self.write_image("beta_file", "beta.nii.gz", B_img)
        # TODO better name for this?
        self.write_image("ols_file", "ols.nii.gz", XtXinv_img)
        self.write_image("sigsqr_file", "sigsqr.nii.gz", SS_img)
        # TODO optionally save residual

        return runtime


class EstimateContrasts(SimpleInterface):

    class input_spec(TraitedSpec):
        exp_info = traits.Dict()
        model_info = traits.Dict()
        beta_file = traits.File(exists=True)
        ols_file = traits.File(exists=True)
        sigsqr_file = traits.File(exists=True)

    class output_spec(TraitedSpec):
        contrast_file = traits.File(exists=True)
        variance_file = traits.File(exists=True)
        tstat_file = traits.File(exists=True)

    def _run_interface(self, runtime):

        # Load model parameters
        beta_img = nib.load(self.inputs.beta_file)
        affine, header = beta_img.affine, beta_img.header
        beta_data = beta_img.get_data()

        ols_img = nib.load(self.inputs.ols_file)
        ols_data = ols_img.get_data()

        sigsqr_img = nib.load(self.inputs.sigsqr_file)
        sigsqr_data = sigsqr_img.get_data()

        # Convert to matrix form
        gray_mask = beta_data.any(axis=-1)
        B = beta_data[gray_mask]
        XtXinv = ols_data[gray_mask]
        SS = sigsqr_data[gray_mask]

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

        # Estimate the contrasts, variances, and statistics in each voxel
        G, V, T = glm.iterative_contrast_estimation(B, XtXinv, SS, C)

        # Generate the output images
        nx, ny, nz = gray_mask.shape
        out_shape = nx, ny, nz, len(C)

        contrast_data = np.zeros(out_shape)
        contrast_data[gray_mask] = G
        contrast_img = nib.Nifti1Image(contrast_data, affine, header)

        variance_data = np.zeros(out_shape)
        variance_data[gray_mask] = V
        variance_img = nib.Nifti1Image(variance_data, affine, header)

        tstat_data = np.zeros(out_shape)
        tstat_data[gray_mask] = T
        tstat_img = nib.Nifti1Image(tstat_data, affine, header)

        # Write out the output files
        self.write_image("contrast_file", "contrast.nii.gz", contrast_img)
        self.write_image("variance_file", "variance.nii.gz", variance_img)
        self.write_image("tstat_file", "tstat.nii.gz", tstat_img)

        return runtime


class ModelResults(SimpleInterface):

    class input_spec(TraitedSpec):
        model_info = traits.Dict()
        contrast_files = traits.List(traits.File(exists=True))
        variance_files = traits.List(traits.File(exists=True))
        design_files = traits.List(traits.File(exists=True))

    class output_spec(TraitedSpec):
        result_directories = traits.List(traits.Directory(exists=True))

    def _run_interface(self, runtime):

        model_info = Bunch(self.inputs.model_info)

        # Load the designs to compute the first-level degrees of freedom
        first_level_dofs = []
        for f in self.inputs.design_files:
            X = pd.read_csv(f, index_col=0)
            first_level_dofs.append(X.shape[0] - X.shape[1])
        ffx_dof = np.sum(first_level_dofs)

        # Load the first file to get
        img = nib.load(self.inputs.contrast_files[0])
        affine, header = img.affine, img.header
        vol_shape = img.shape[:3]

        # TODO define contrasts properly accounting for missing EVs
        result_directories = []
        for i, contrast in enumerate(model_info.contrasts):

            result_directories.append(op.abspath(contrast))
            os.mkdir(contrast)

            con_frames = []
            var_frames = []

            # Load the parameter and variance data for each run/contrast.
            # Files are input as a list of 4D images where list entries are
            # runs and the last axis is contrast, but we want to combine over
            # runs for each contrast, so we need to in effect "transpose" the
            # ordering. We do this by loading the data multiple times, which is
            # cheap for models with reasonable numbers of contrasts.
            for con_file, var_file in zip(self.inputs.contrast_files,
                                          self.inputs.variance_files):

                con_frames.append(nib.load(con_file).get_data()[..., i])
                var_frames.append(nib.load(var_file).get_data()[..., i])

            con_data = np.stack(con_frames, axis=-1)
            var_data = np.stack(var_frames, axis=-1)

            # Define a mask as voxels with nonzero variance in each run
            # and extract voxel data as arrays
            mask = (var_data > 0).all(axis=-1)
            con = con_data[mask]
            var = var_data[mask]

            # Compute the higher-level fixed effects parameters
            con_ffx, var_ffx, t_ffx = glm.contrast_fixed_effects(con, var)

            # TODO make a zstat image?

            # Write out output images
            con_vol_out = np.zeros(vol_shape)
            con_vol_out[mask] = con_ffx
            con_img_out = nib.Nifti1Image(con_vol_out, affine, header)
            con_img_out.to_filename(op.join(contrast, "contrast.nii.gz"))

            var_vol_out = np.zeros(vol_shape)
            var_vol_out[mask] = var_ffx
            var_img_out = nib.Nifti1Image(var_vol_out, affine, header)
            var_img_out.to_filename(op.join(contrast, "variance.nii.gz"))

            t_vol_out = np.zeros(vol_shape)
            t_vol_out[mask] = t_ffx
            t_img_out = nib.Nifti1Image(t_vol_out, affine, header)
            t_img_out.to_filename(op.join(contrast, "tstat.nii.gz"))

            # TODO save out the mask?

        # Output a list of directories with results.
        # This makes the connections in the workflow more opaque, but it
        # simplifies placing files in subdirectories named after contrasts.
        self._results["result_directories"] = result_directories

        return runtime
