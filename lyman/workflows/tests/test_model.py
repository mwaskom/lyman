import nipype

from .. import model


class TestModelFitWorkflow(object):

    def test_model_fit_workflow_creation(self, lyman_info):

        wf = model.define_model_fit_workflow(
            lyman_info["proj_info"],
            lyman_info["subjects"],
            lyman_info["session"],
            lyman_info["exp_info"],
            lyman_info["model_info"]
        )

        assert isinstance(wf, nipype.Workflow)

    def test_model_results_workflow_creation(self, lyman_info):

        wf = model.define_model_results_workflow(
            lyman_info["proj_info"],
            lyman_info["subjects"],
            lyman_info["exp_info"],
            lyman_info["model_info"]
        )

        assert isinstance(wf, nipype.Workflow)
