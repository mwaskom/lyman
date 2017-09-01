import nipype

from .. import preproc


class TestPreprocWorkflow(object):

    def test_preproc_workflow_creation(self, lyman_info):

        wf = preproc.define_preproc_workflow(
            lyman_info["proj_info"],
            lyman_info["subjects"],
            lyman_info["session"],
            lyman_info["exp_info"],
        )

        assert isinstance(wf, nipype.Workflow)
