import os.path as op
from copy import deepcopy
import numpy as np
import pandas as pd
import nibabel as nib
import nipype
import pytest

from .. import model


class TestModelWorkflows(object):

    def test_model_fit_workflow_creation(self, lyman_info):

        info = lyman_info["info"]
        subjects = lyman_info["subjects"]
        sessions = lyman_info["sessions"]

        wf = model.define_model_fit_workflow(info, subjects, sessions)

        # Check basic information about the workflow
        assert isinstance(wf, nipype.Workflow)
        assert wf.name == "model_fit"
        assert wf.base_dir == op.join(info.cache_dir, info.experiment_name)

        # Check root directory of output
        data_out = wf.get_node("data_output")
        assert data_out.inputs.base_directory == info.proc_dir

        # Check the list of nodes we expect
        expected_nodes = ["subject_source", "run_source", "save_info",
                          "data_input", "fit_model", "data_output"]
        expected_nodes.sort()
        assert wf.list_node_names() == expected_nodes

    def test_model_results_workflow_creation(self, lyman_info):

        info = lyman_info["info"]
        subjects = lyman_info["subjects"]

        wf = model.define_model_results_workflow(info, subjects)

        # Check basic information about the workflow
        assert isinstance(wf, nipype.Workflow)
        assert wf.name == "model_results"
        assert wf.base_dir == op.join(info.cache_dir, info.experiment_name)

        # Check root directory of output
        run_out = wf.get_node("run_output")
        assert run_out.inputs.base_directory == info.proc_dir
        subject_out = wf.get_node("subject_output")
        assert subject_out.inputs.base_directory == info.proc_dir

        # Check the list of nodes we expect
        expected_nodes = ["subject_source", "run_source", "data_input",
                          "estimate_contrasts", "model_results", "save_info",
                          "run_output", "results_path", "subject_output"]
        expected_nodes.sort()
        assert wf.list_node_names() == expected_nodes

    def test_model_iterables(self, lyman_info):

        info = lyman_info["info"]
        scan_info = info.scan_info

        # -- Test full iterables

        iterables = model.generate_iterables(
            scan_info, "exp_alpha", ["subj01", "subj02"],
        )
        expected_iterables = (
            ["subj01", "subj02"],
            {"subj01":
                [("sess01", "run01"),
                 ("sess01", "run02"),
                 ("sess02", "run01")],
             "subj02":
                [("sess01", "run01"),
                 ("sess01", "run02"),
                 ("sess01", "run03")]},
        )
        assert iterables == expected_iterables

        # -- Test iterables as set in workflow

        wf = model.define_model_fit_workflow(info, ["subj01", "subj02"], None)

        subject_source = wf.get_node("subject_source")
        assert subject_source.iterables == ("subject", iterables[0])

        run_source = wf.get_node("run_source")
        assert run_source.iterables == ("run", iterables[1])

        wf = model.define_model_results_workflow(info, ["subj01", "subj02"])

        subject_source = wf.get_node("subject_source")
        assert subject_source.iterables == ("subject", iterables[0])

        run_source = wf.get_node("run_source")
        assert run_source.iterables == ("run", iterables[1])

        # --  Test single subject

        iterables = model.generate_iterables(
            scan_info, "exp_alpha", ["subj01"],
        )
        expected_iterables = (
            ["subj01"],
            {"subj01":
                [("sess01", "run01"),
                 ("sess01", "run02"),
                 ("sess02", "run01")]}
        )
        assert iterables == expected_iterables

        # -- Test different experiment

        iterables = model.generate_iterables(
            scan_info, "exp_beta", ["subj01", "subj02"],
        )
        expected_iterables = (
            ["subj01"],
            {"subj01":
                [("sess02", "run01"),
                 ("sess02", "run02"),
                 ("sess02", "run03")]},
        )
        assert iterables == expected_iterables

        # -- Test single subject, single session

        iterables = model.generate_iterables(
            scan_info, "exp_alpha", ["subj01"], ["sess02"],
        )
        expected_iterables = (
            ["subj01"],
            {"subj01":
                [("sess02", "run01")]},
        )
        assert iterables == expected_iterables

    def test_model_results_path(self):

        proc_dir = op.realpath(".")
        subject = "subj01"
        experiment = "exp_a"
        model_name = "model_alpha"

        ifc = model.ModelResultsPath(
            proc_dir=str(proc_dir),
            subject=subject,
            experiment=experiment,
            model=model_name,
        )

        out = ifc.run().outputs
        expected_path = op.join(proc_dir, subject,
                                experiment, model_name, "results")

        assert out.output_path == expected_path

    def test_model_fit_input(self, timeseries):

        subject = timeseries["subject"]
        run_tuple = session, run = timeseries["session"], timeseries["run"]

        exp_name = timeseries["info"].experiment_name
        model_name = timeseries["info"].model_name

        out = model.ModelFitInput(
            experiment=exp_name,
            model=model_name,
            proc_dir=str(timeseries["proc_dir"]),
            subject=subject,
            run_tuple=run_tuple,
        ).run().outputs

        assert out.subject == subject
        assert out.session == session
        assert out.run == run
        assert out.seg_file == timeseries["seg_file"]
        assert out.surf_file == timeseries["surf_file"]
        assert out.mask_file == timeseries["mask_file"]
        assert out.edge_file == timeseries["edge_file"]
        assert out.ts_file == timeseries["ts_file"]
        assert out.noise_file == timeseries["noise_file"]
        assert out.mc_file == timeseries["mc_file"]
        assert out.output_path == timeseries["model_dir"]

    def test_model_results_input(self, modelfit):

        subject = modelfit["subject"]
        run_tuple = session, run = modelfit["session"], modelfit["run"]

        exp_name = modelfit["info"].experiment_name
        model_name = modelfit["info"].model_name

        out = model.ModelResultsInput(
            experiment=exp_name,
            model=model_name,
            proc_dir=str(modelfit["proc_dir"]),
            subject=subject,
            run_tuple=run_tuple,
        ).run().outputs

        assert out.subject == subject
        assert out.session == session
        assert out.run == run
        assert out.anat_file == modelfit["anat_file"]
        assert out.mask_file == modelfit["mask_file"]
        assert out.beta_file == modelfit["beta_file"]
        assert out.ols_file == modelfit["ols_file"]
        assert out.error_file == modelfit["error_file"]
        assert out.output_path == modelfit["model_dir"]

    @pytest.mark.parametrize(
        "percent_change,nuisance_regression",
        [(True, True), (False, False)],
    )
    def test_model_fit(self, execdir, timeseries,
                       percent_change, nuisance_regression):

        info = timeseries["info"]
        info.percent_change = percent_change
        if not nuisance_regression:
            info.nuisance_components = {}

        out = model.ModelFit(
            subject=timeseries["subject"],
            session=timeseries["session"],
            run=timeseries["run"],
            data_dir=str(timeseries["data_dir"]),
            info=info.trait_get(),
            seg_file=timeseries["seg_file"],
            surf_file=timeseries["surf_file"],
            edge_file=timeseries["edge_file"],
            ts_file=timeseries["ts_file"],
            mask_file=timeseries["mask_file"],
            noise_file=timeseries["noise_file"],
            mc_file=timeseries["mc_file"],
        ).run().outputs

        # Test output file names
        assert out.mask_file == execdir.join("mask.nii.gz")
        assert out.beta_file == execdir.join("beta.nii.gz")
        assert out.error_file == execdir.join("error.nii.gz")
        assert out.ols_file == execdir.join("ols.nii.gz")
        assert out.resid_file == execdir.join("resid.nii.gz")
        assert out.model_file == execdir.join("model.csv")
        assert out.resid_plot == execdir.join("resid.png")
        assert out.model_plot == execdir.join("model.png")
        assert out.error_plot == execdir.join("error.png")

        if nuisance_regression:
            assert out.nuisance_plot == execdir.join("nuisance.png")

        n_x, n_y, n_z = timeseries["vol_shape"]
        n_tp = timeseries["n_tp"]

        X = pd.read_csv(out.model_file)
        n_params = X.shape[1]

        # Test output image shapes
        mask_img = nib.load(out.mask_file)
        assert mask_img.shape == (n_x, n_y, n_z)

        beta_img = nib.load(out.beta_file)
        assert beta_img.shape == (n_x, n_y, n_z, n_params)

        error_img = nib.load(out.error_file)
        assert error_img.shape == (n_x, n_y, n_z)

        ols_img = nib.load(out.ols_file)
        assert ols_img.shape == (n_x, n_y, n_z, n_params ** 2)

        resid_img = nib.load(out.resid_file)
        assert resid_img.shape == (n_x, n_y, n_z, n_tp)

        model_matrix = pd.read_csv(out.model_file)
        assert model_matrix.shape == (n_tp, n_params)

    def test_estimate_contrasts(self, execdir, modelfit):

        out = model.EstimateContrasts(
            info=modelfit["info"].trait_get(),
            mask_file=modelfit["mask_file"],
            beta_file=modelfit["beta_file"],
            ols_file=modelfit["ols_file"],
            error_file=modelfit["error_file"],
            model_file=modelfit["model_file"],
        ).run().outputs

        # Test output file names
        assert out.contrast_file == execdir.join("contrast.nii.gz")
        assert out.variance_file == execdir.join("variance.nii.gz")
        assert out.tstat_file == execdir.join("tstat.nii.gz")
        assert out.name_file == execdir.join("contrast.txt")

        # Test output image shapes
        n_contrasts = len(modelfit["info"].contrasts)
        assert nib.load(out.contrast_file).shape[-1] == n_contrasts
        assert nib.load(out.variance_file).shape[-1] == n_contrasts
        assert nib.load(out.tstat_file).shape[-1] == n_contrasts
        assert len(np.loadtxt(out.name_file, str)) == n_contrasts

    def test_missing_contrasts(self, execdir, modelfit):

        info = deepcopy(modelfit["info"].trait_get())
        n_contrasts = len(info["contrasts"])
        info["contrasts"].append(("d", ["d"], [1]))

        out = model.EstimateContrasts(
            info=info,
            mask_file=modelfit["mask_file"],
            beta_file=modelfit["beta_file"],
            ols_file=modelfit["ols_file"],
            error_file=modelfit["error_file"],
            model_file=modelfit["model_file"],
        ).run().outputs

        # Test output image shapes
        assert nib.load(out.contrast_file).shape[-1] == n_contrasts
        assert nib.load(out.variance_file).shape[-1] == n_contrasts
        assert nib.load(out.tstat_file).shape[-1] == n_contrasts
        assert len(np.loadtxt(out.name_file, str)) == n_contrasts

    def test_model_results(self, execdir, modelres):

        out = model.ModelResults(
            info=modelres["info"].trait_get(),
            anat_file=modelres["anat_file"],
            contrast_files=modelres["contrast_files"],
            variance_files=modelres["variance_files"],
            name_files=modelres["name_files"],
        ).run().outputs

        contrast_names = [c for c, _, _ in modelres["info"].contrasts]
        result_directories = [execdir.join(c) for c in contrast_names]
        assert out.result_directories == result_directories
