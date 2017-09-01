import nipype
from moss import Bunch  # TODO change to lyman version when implemented

import pytest

from .. import template


class TestTemplateWorkflow(object):

    @pytest.fixture
    def test_data(self, tmpdir):
        # TODO copied from model -- define centrally!
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

        return dict(
            proj_info=proj_info,
            subjects=subjects,
        )

    def test_template_workflow_creation(self, test_data):

        wf = template.define_template_workflow(
            test_data["proj_info"],
            test_data["subjects"],
        )

        assert isinstance(wf, nipype.Workflow)
