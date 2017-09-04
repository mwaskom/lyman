import os.path as op
import nipype

from .. import preproc


class TestPreprocWorkflow(object):

    def test_preproc_workflow_creation(self, lyman_info):

        proj_info = lyman_info["proj_info"]
        subjects = lyman_info["subjects"]
        sessions = lyman_info["sessions"]
        exp_info = lyman_info["exp_info"]

        wf = preproc.define_preproc_workflow(
            proj_info, exp_info, subjects, sessions,
        )

        # Check basic information about the workflow
        assert isinstance(wf, nipype.Workflow)
        assert wf.name == "preproc"
        assert wf.base_dir == op.join(proj_info.cache_dir, exp_info.name)

        # Check root directory of output
        template_out = wf.get_node("template_output")
        assert template_out.inputs.base_directory == proj_info.analysis_dir
        timeseries_out = wf.get_node("timeseries_output")
        assert timeseries_out.inputs.base_directory == proj_info.analysis_dir

        # Check the list of nodes we expect
        expected_nodes = ["subject_source", "session_source", "run_source",
                          "session_input", "run_input",
                          "estimate_distortions", "finalize_unwarping",
                          "fm2anat", "fm2anat_qc",
                          "sb2fm", "sb2fm_qc",
                          "ts2sb", "ts2sb_qc",
                          "combine_premats", "combine_postmats",
                          "restore_timeseries", "restore_template",
                          "finalize_timeseries", "finalize_template",
                          "template_output", "timeseries_output"]
        expected_nodes.sort()
        assert wf.list_node_names() == expected_nodes

    def test_preproc_iterables(self, lyman_info):

        proj_info = lyman_info["proj_info"]
        scan_info = proj_info["scan_info"]
        exp_info = lyman_info["exp_info"]

        # -- Test full iterables

        iterables = preproc.generate_iterables(
            scan_info, "exp_alpha", ["subj01", "subj02"],
        )
        expected_iterables = (
            ["subj01", "subj02"],
            {"subj01": [("subj01", "sess01"), ("subj01", "sess02")],
             "subj02": [("subj02", "sess01")]},
            {("subj01", "sess01"):
                [("subj01", "sess01", "run01"),
                 ("subj01", "sess01", "run02")],
             ("subj01", "sess02"):
                [("subj01", "sess02", "run01")],
             ("subj02", "sess01"):
                [("subj02", "sess01", "run01"),
                 ("subj02", "sess01", "run02"),
                 ("subj02", "sess01", "run03")]},
        )
        assert iterables == expected_iterables

        # -- Test iterables as set in workflow

        wf = preproc.define_preproc_workflow(
            proj_info, exp_info, ["subj01", "subj02"], None,
        )

        subject_source = wf.get_node("subject_source")
        assert subject_source.iterables == ("subject", iterables[0])

        session_source = wf.get_node("session_source")
        assert session_source.iterables == ("session", iterables[1])

        run_source = wf.get_node("run_source")
        assert run_source.iterables == ("run", iterables[2])

        # --  Test single subject

        iterables = preproc.generate_iterables(
            scan_info, "exp_alpha", ["subj01"],
        )
        expected_iterables = (
            ["subj01"],
            {"subj01": [("subj01", "sess01"), ("subj01", "sess02")]},
            {("subj01", "sess01"):
                [("subj01", "sess01", "run01"),
                 ("subj01", "sess01", "run02")],
             ("subj01", "sess02"):
                [("subj01", "sess02", "run01")]}
        )
        assert iterables == expected_iterables

        # -- Test different experiment

        iterables = preproc.generate_iterables(
            scan_info, "exp_beta", ["subj01", "subj02"],
        )
        expected_iterables = (
            ["subj01"],
            {"subj01": [("subj01", "sess02")]},
            {("subj01", "sess02"):
                [("subj01", "sess02", "run01"),
                 ("subj01", "sess02", "run02"),
                 ("subj01", "sess02", "run03")]},
        )
        assert iterables == expected_iterables

        # -- Test single subject, single session

        iterables = preproc.generate_iterables(
            scan_info, "exp_alpha", ["subj01"], ["sess02"],
        )
        expected_iterables = (
            ["subj01"],
            {"subj01": [("subj01", "sess02")]},
            {("subj01", "sess02"):
                [("subj01", "sess02", "run01")]},
        )
        assert iterables == expected_iterables

    def test_combine_linaer_transforms(self, execdir):

        pass
