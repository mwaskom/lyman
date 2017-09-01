import nipype

from .. import template


class TestTemplateWorkflow(object):

    def test_template_workflow_creation(self, lyman_info):

        wf = template.define_template_workflow(
            lyman_info["proj_info"],
            lyman_info["subjects"],
        )

        assert isinstance(wf, nipype.Workflow)
