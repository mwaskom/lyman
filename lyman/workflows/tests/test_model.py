import nipype
from moss import Bunch  # TODO change to lyman version when implemented

import pytest

from .. import model


class TestModelFitWorkflow(object):

    @pytest.fixture
    def test_data(self, tmpdir):

        data_dir = tmpdir.mkdir("data")
        analysis_dir = tmpdir.mkdir("analysis")
        cache_dir = tmpdir.mkdir("cache")

        # TODO probably get these from default info functions
        scan_info = {
            "subj01": {
                "exp_alpha": {
                    "sess01": ["run01", "run02"],
                    "sess02": ["run01"],
                },
            },
            "subj02": {
                "exp_alpha": {
                    "sess01": ["run01", "run02", "run02"],
                }
            },
        }
        proj_info = Bunch(
            data_dir=str(data_dir),
            analysis_dir=str(analysis_dir),
            cache_dir=str(cache_dir),
            scan_info=scan_info,
        )

        subjects = ["subj01", "subj02"]
        session = None

        exp_info = Bunch(name="exp_alpha")
        model_info = Bunch(name="model_a")

        return dict(
            proj_info=proj_info,
            subjects=subjects,
            session=session,
            exp_info=exp_info,
            model_info=model_info,
        )

    def test_model_fit_workflow_creation(self, test_data):

        wf = model.define_model_fit_workflow(
            test_data["proj_info"],
            test_data["subjects"],
            test_data["session"],
            test_data["exp_info"],
            test_data["model_info"]
        )

        assert isinstance(wf, nipype.Workflow)

    def test_model_results_workflow_creation(self, test_data):

        wf = model.define_model_results_workflow(
            test_data["proj_info"],
            test_data["subjects"],
            test_data["exp_info"],
            test_data["model_info"]
        )

        assert isinstance(wf, nipype.Workflow)
