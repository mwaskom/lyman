import nipype

from .. import template


class TestTemplateWorkflow(object):

    def test_template_workflow_creation(self, lyman_info):

        proj_info = lyman_info["proj_info"]
        subjects = lyman_info["subjects"]

        wf = template.define_template_workflow(
            proj_info, subjects
        )

        # Check basic information about the workflow
        assert isinstance(wf, nipype.Workflow)
        assert wf.name == "template"
        assert wf.base_dir == proj_info.cache_dir

        # Check root directory of output
        template_out = wf.get_node("template_output")
        assert template_out.inputs.base_directory == proj_info.analysis_dir

        # Check the list of nodes we expect
        expected_nodes = ["subject_source", "template_input",
                          "crop_image", "zoom_image", "reorient_image",
                          "generate_reg", "invert_reg",
                          "transform_wmparc", "anat_segment",
                          "hemi_source", "tag_surf", "combine_hemis",
                          "template_qc", "template_path", "template_output"]
        expected_nodes.sort()
        assert wf.list_node_names() == expected_nodes

        # Check iterables
        subject_source = wf.get_node("subject_source")
        assert subject_source.iterables == ("subject", subjects)
