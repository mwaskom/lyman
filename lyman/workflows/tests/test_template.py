import numpy as np
import pandas as pd
import nibabel as nib
import nipype

from ..template import (define_template_workflow,
                        TemplateInput,
                        AnatomicalSegmentation,
                        MaskWithLabel,
                        MakeRibbon,
                        TemplateReport)


class TestTemplateWorkflow(object):

    def test_template_workflow_creation(self, lyman_info):

        info = lyman_info["info"]
        subjects = lyman_info["subjects"]

        wf = define_template_workflow(info, subjects)

        # Check basic information about the workflow
        assert isinstance(wf, nipype.Workflow)
        assert wf.name == "template"
        assert wf.base_dir == info.cache_dir

        # Check root directory of output
        template_out = wf.get_node("template_output")
        assert template_out.inputs.base_directory == info.proc_dir

        # Check the list of nodes we expect
        expected_nodes = ["subject_source", "template_input",
                          "crop_image", "zoom_image", "reorient_image",
                          "generate_reg", "invert_reg",
                          "transform_wmparc", "anat_segment",
                          "hemi_source", "combine_hemis",
                          "tag_surf", "mask_cortex", "make_ribbon",
                          "save_info", "template_qc", "template_output"]
        expected_nodes.sort()
        assert wf.list_node_names() == expected_nodes

        # Check iterables
        subject_source = wf.get_node("subject_source")
        assert subject_source.iterables == ("subject", subjects)

    def test_template_input(self, freesurfer):

        out = TemplateInput(
            data_dir=str(freesurfer["data_dir"]),
            subject=freesurfer["subject"]
        ).run().outputs

        assert out.norm_file == freesurfer["norm_file"]
        assert out.wmparc_file == freesurfer["wmparc_file"]
        assert out.label_files == freesurfer["label_files"]

        output_path = "{}/template".format(freesurfer["subject"])
        assert out.output_path == output_path

    def test_anatomical_segmentation(self, execdir, template):

        out = AnatomicalSegmentation(
            anat_file=template["anat_file"],
            surf_file=template["surf_file"],
            wmparc_file=template["wmparc_file"],
        ).run().outputs

        assert out.lut_file == execdir.join("seg.lut")
        assert out.seg_file == execdir.join("seg.nii.gz")
        assert out.edge_file == execdir.join("edge.nii.gz")
        assert out.mask_file == execdir.join("mask.nii.gz")

        # Test size of the lookup table
        lut = pd.read_csv(out.lut_file, sep="\t", header=None)
        assert lut.shape == (9, 6)

        # Test that segmentation has integral values
        seg = nib.load(out.seg_file).get_fdata()
        assert np.array_equal(seg, seg.astype("uint8"))

        # Test that the segmentation cortical gray matches surface vertices
        surf = (nib.load(template["surf_file"]).get_fdata() > -1).any(axis=-1)
        assert np.all(seg[surf] == 1)

    def test_mask_label(self, execdir, template):

        out = MaskWithLabel(
            in_file=template["surf_file"],
            label_files=template["label_files"],
            hemi="lh",
            fill_value=-1,
        ).run().outputs

        assert out.out_file == execdir.join("masked.nii.gz")

    def test_make_ribbon(self, execdir, template):

        out = MakeRibbon(
            in_file=template["surf_file"],
        ).run().outputs

        assert out.out_file == execdir.join("ribbon.nii.gz")

    def test_template_report(self, execdir, template):

        out = TemplateReport(
            lut_file=template["lut_file"],
            seg_file=template["seg_file"],
            mask_file=template["mask_file"],
            edge_file=template["edge_file"],
            surf_file=template["surf_file"],
            anat_file=template["anat_file"],
        ).run().outputs

        assert out.seg_plot == execdir.join("seg.png")
        assert out.mask_plot == execdir.join("mask.png")
        assert out.surf_plot == execdir.join("surf.png")
        assert out.edge_plot == execdir.join("edge.png")
        assert out.anat_plot == execdir.join("anat.png")
