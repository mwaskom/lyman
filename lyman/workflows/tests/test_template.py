import nipype

from ..template import (define_template_workflow,
                        TemplateInput,
                        AnatomicalSegmentation,
                        TemplateReport)


class TestTemplateWorkflow(object):

    def test_template_workflow_creation(self, lyman_info):

        proj_info = lyman_info["proj_info"]
        subjects = lyman_info["subjects"]

        wf = define_template_workflow(
            proj_info, subjects
        )

        # Check basic information about the workflow
        assert isinstance(wf, nipype.Workflow)
        assert wf.name == "template"
        assert wf.base_dir == proj_info.cache_dir

        # Check root directory of output
        template_out = wf.get_node("template_output")
        assert template_out.inputs.base_directory == proj_info.proc_dir

        # Check the list of nodes we expect
        expected_nodes = ["subject_source", "template_input",
                          "crop_image", "zoom_image", "reorient_image",
                          "generate_reg", "invert_reg",
                          "transform_wmparc", "anat_segment",
                          "hemi_source", "tag_surf", "combine_hemis",
                          "template_qc", "template_output"]
        expected_nodes.sort()
        assert wf.list_node_names() == expected_nodes

        # Check iterables
        subject_source = wf.get_node("subject_source")
        assert subject_source.iterables == ("subject", subjects)

    def test_template_input(self, freesurfer):

        out = TemplateInput(
            data_dir=freesurfer["data_dir"],
            subject=freesurfer["subject"]
        ).run().outputs

        assert out.norm_file == freesurfer["norm_file"]
        assert out.wmparc_file == freesurfer["wmparc_file"]

        output_path = "{}/template".format(freesurfer["subject"])
        assert out.output_path == output_path

    def test_anatomical_segmentation(self, execdir, freesurfer):

        out = AnatomicalSegmentation(
            wmparc_file=freesurfer["wmparc_file"],
        ).run().outputs

        assert out.seg_file == execdir.join("seg.nii.gz")
        assert out.mask_file == execdir.join("mask.nii.gz")

    def test_template_report(self, execdir, template):

        out = TemplateReport(
            seg_file=template["seg_file"],
            mask_file=template["mask_file"],
            surf_file=template["surf_file"],
            anat_file=template["anat_file"],
        ).run().outputs

        assert out.seg_plot == execdir.join("seg.png")
        assert out.mask_plot == execdir.join("mask.png")
        assert out.surf_plot == execdir.join("surf.png")
        assert out.anat_plot == execdir.join("anat.png")
