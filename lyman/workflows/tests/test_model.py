import os.path as op
import pandas as pd
import nibabel as nib
import nipype

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
            proc_dir=proc_dir,
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
            data_dir=str(timeseries["data_dir"]),
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
        assert out.ts_file == timeseries["ts_file"]
        assert out.noise_file == timeseries["noise_file"]
        assert out.mc_file == timeseries["mc_file"]
        assert out.mesh_files == timeseries["mesh_files"]
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

    def test_model_fit(self, execdir, timeseries):

        out = model.ModelFit(
            subject=timeseries["subject"],
            session=timeseries["session"],
            run=timeseries["run"],
            data_dir=timeseries["data_dir"],
            info=timeseries["info"].trait_get(),
            seg_file=timeseries["seg_file"],
            surf_file=timeseries["surf_file"],
            ts_file=timeseries["ts_file"],
            mask_file=timeseries["mask_file"],
            noise_file=timeseries["noise_file"],
            mc_file=timeseries["mc_file"],
            mesh_files=timeseries["mesh_files"],
        ).run().outputs

        # Test output file names
        assert out.mask_file == execdir.join("mask.nii.gz")
        assert out.beta_file == execdir.join("beta.nii.gz")
        assert out.error_file == execdir.join("error.nii.gz")
        assert out.ols_file == execdir.join("ols.nii.gz")
        assert out.resid_file == execdir.join("resid.nii.gz")
        assert out.design_file == execdir.join("design.csv")
        assert out.resid_plot == execdir.join("resid.png")
        assert out.design_plot == execdir.join("design.png")
        assert out.error_plot == execdir.join("error.png")

        n_x, n_y, n_z = timeseries["vol_shape"]
        n_tp = timeseries["n_tp"]
        n_params = timeseries["n_params"]

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

        design = pd.read_csv(out.design_file)
        assert design.shape == (n_tp, n_params)

    def test_estimate_contrasts(self, execdir, modelfit):

        out = model.EstimateContrasts(
            info=modelfit["info"].trait_get(),
            mask_file=modelfit["mask_file"],
            beta_file=modelfit["beta_file"],
            ols_file=modelfit["ols_file"],
            error_file=modelfit["error_file"],
            design_file=modelfit["design_file"],
        ).run().outputs

        # Test output file names
        assert out.contrast_file == execdir.join("contrast.nii.gz")
        assert out.variance_file == execdir.join("variance.nii.gz")
        assert out.tstat_file == execdir.join("tstat.nii.gz")

        # Test output image shapes
        # TODO this needs to be fixed once contrasts info is finished

        # TODO we should also test "missing" contrast behavior here

    def test_model_results(self, execdir, modelres):

        out = model.ModelResults(
            info=modelres["info"].trait_get(),
            anat_file=modelres["anat_file"],
            contrast_files=[modelres["contrast_file"]],
            variance_files=[modelres["variance_file"]],
        ).run().outputs

        result_directories = [
            execdir.join(c) for c, _, _ in modelres["info"].contrasts
        ]
        assert out.result_directories == result_directories
