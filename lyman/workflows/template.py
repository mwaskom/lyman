import os.path as op

import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib as mpl
import nibabel as nib

from nipype import Workflow, Node, JoinNode, IdentityInterface, DataSink
from nipype.interfaces.base import traits, TraitedSpec
from nipype.interfaces import fsl, freesurfer as fs

from ..utils import LymanInterface, SaveInfo
from ..visualizations import Mosaic


def define_template_workflow(info, subjects, qc=True):

    # --- Workflow parameterization

    subject_source = Node(IdentityInterface(["subject"]),
                          name="subject_source",
                          iterables=("subject", subjects))

    # Data input
    template_input = Node(TemplateInput(data_dir=info.data_dir),
                          "template_input")

    # --- Definition of functional template space

    crop_image = Node(fs.ApplyMask(args="-bb 4"), "crop_image")

    zoom_image = Node(fs.MRIConvert(resample_type="cubic",
                                    out_type="niigz",
                                    vox_size=info.voxel_size,
                                    ),
                      "zoom_image")

    reorient_image = Node(fsl.Reorient2Std(out_file="anat.nii.gz"),
                          "reorient_image")

    generate_reg = Node(fs.Tkregister2(fsl_out="anat2func.mat",
                                       reg_file="anat2func.dat",
                                       reg_header=True),
                        "generate_reg")

    invert_reg = Node(fs.Tkregister2(reg_file="func2anat.dat",
                                     reg_header=True),
                      "invert_reg")

    # --- Identification of surface vertices

    hemi_source = Node(IdentityInterface(["hemi"]), "hemi_source",
                       iterables=("hemi", ["lh", "rh"]))

    tag_surf = Node(fs.Surface2VolTransform(surf_name="graymid",
                                            transformed_file="ribbon.nii.gz",
                                            vertexvol_file="vertices.nii.gz",
                                            mkmask=True),
                    "tag_surf")

    mask_cortex = Node(MaskWithLabel(fill_value=-1), "mask_cortex")

    combine_hemis = JoinNode(fsl.Merge(dimension="t",
                                       merged_file="surf.nii.gz"),
                             name="combine_hemis",
                             joinsource="hemi_source",
                             joinfield="in_files")

    make_ribbon = Node(MakeRibbon(), "make_ribbon")

    # --- Segementation of anatomical tissue in functional space

    transform_wmparc = Node(fs.ApplyVolTransform(inverse=True,
                                                 interp="nearest",
                                                 args="--keep-precision"),
                            "transform_wmparc")

    anat_segment = Node(AnatomicalSegmentation(), "anat_segment")

    # --- Template QC

    template_qc = Node(TemplateReport(), "template_qc")

    # --- Workflow ouptut

    save_info = Node(SaveInfo(info_dict=info.trait_get()), "save_info")

    template_output = Node(DataSink(base_directory=info.proc_dir,
                                    parameterization=False),
                           "template_output")

    # === Assemble pipeline

    workflow = Workflow(name="template", base_dir=info.cache_dir)

    processing_edges = [

        (subject_source, template_input,
            [("subject", "subject")]),
        (template_input, crop_image,
            [("norm_file", "in_file"),
             ("wmparc_file", "mask_file")]),
        (crop_image, zoom_image,
            [("out_file", "in_file")]),
        (zoom_image, reorient_image,
            [("out_file", "in_file")]),

        (subject_source, generate_reg,
            [("subject", "subject_id")]),
        (template_input, generate_reg,
            [("norm_file", "moving_image")]),
        (reorient_image, generate_reg,
            [("out_file", "target_image")]),

        (subject_source, invert_reg,
            [("subject", "subject_id")]),
        (template_input, invert_reg,
            [("norm_file", "target_image")]),
        (reorient_image, invert_reg,
            [("out_file", "moving_image")]),

        (hemi_source, tag_surf,
            [("hemi", "hemi")]),
        (invert_reg, tag_surf,
            [("reg_file", "reg_file")]),
        (reorient_image, tag_surf,
            [("out_file", "template_file")]),
        (template_input, mask_cortex,
            [("label_files", "label_files")]),
        (hemi_source, mask_cortex,
            [("hemi", "hemi")]),
        (tag_surf, mask_cortex,
            [("vertexvol_file", "in_file")]),
        (mask_cortex, combine_hemis,
            [("out_file", "in_files")]),
        (combine_hemis, make_ribbon,
            [("merged_file", "in_file")]),

        (reorient_image, transform_wmparc,
            [("out_file", "source_file")]),
        (template_input, transform_wmparc,
            [("wmparc_file", "target_file")]),
        (invert_reg, transform_wmparc,
            [("reg_file", "reg_file")]),
        (reorient_image, anat_segment,
            [("out_file", "anat_file")]),
        (transform_wmparc, anat_segment,
            [("transformed_file", "wmparc_file")]),
        (combine_hemis, anat_segment,
            [("merged_file", "surf_file")]),

        (template_input, template_output,
            [("output_path", "container")]),
        (reorient_image, template_output,
            [("out_file", "@anat")]),
        (generate_reg, template_output,
            [("fsl_file", "@anat2func")]),
        (anat_segment, template_output,
            [("seg_file", "@seg"),
             ("lut_file", "@lut"),
             ("edge_file", "@edge"),
             ("mask_file", "@mask")]),
        (combine_hemis, template_output,
            [("merged_file", "@surf")]),
        (make_ribbon, template_output,
            [("out_file", "@ribon")]),

    ]
    workflow.connect(processing_edges)

    # Optionally connect QC nodes

    qc_edges = [

        (reorient_image, template_qc,
            [("out_file", "anat_file")]),
        (combine_hemis, template_qc,
            [("merged_file", "surf_file")]),
        (anat_segment, template_qc,
            [("lut_file", "lut_file"),
             ("seg_file", "seg_file"),
             ("edge_file", "edge_file"),
             ("mask_file", "mask_file")]),

        (subject_source, save_info,
            [("subject", "parameterization")]),
        (save_info, template_output,
            [("info_file", "qc.@info_json")]),

        (template_qc, template_output,
            [("seg_plot", "qc.@seg_plot"),
             ("mask_plot", "qc.@mask_plot"),
             ("edge_plot", "qc.@edge_plot"),
             ("surf_plot", "qc.@surf_plot"),
             ("anat_plot", "qc.@anat_plot")]),

    ]
    if qc:
        workflow.connect(qc_edges)

    return workflow


# =========================================================================== #
# Custom processing nodes
# =========================================================================== #


class TemplateInput(LymanInterface):

    class input_spec(TraitedSpec):
        data_dir = traits.Directory(exists=True)
        subject = traits.Str()

    class output_spec(TraitedSpec):
        norm_file = traits.File(exists=True)
        wmparc_file = traits.File(exists=True)
        label_files = traits.Dict(key_trait=traits.Str,
                                  value_trait=traits.File(exists=True))
        output_path = traits.Directory()

    def _run_interface(self, runtime):

        output_path = "{}/template".format(self.inputs.subject)
        mri_dir = op.join(self.inputs.data_dir, self.inputs.subject, "mri")
        label_dir = op.join(self.inputs.data_dir, self.inputs.subject, "label")

        results = dict(

            norm_file=op.join(mri_dir, "norm.mgz"),
            wmparc_file=op.join(mri_dir, "wmparc.mgz"),

            label_files=dict(lh=op.join(label_dir, "lh.cortex.label"),
                             rh=op.join(label_dir, "rh.cortex.label")),

            output_path=output_path,

        )
        self._results.update(results)
        return runtime


class AnatomicalSegmentation(LymanInterface):

    class input_spec(TraitedSpec):
        anat_file = traits.File(exists=True)
        surf_file = traits.File(exists=True)
        wmparc_file = traits.File(exists=True)

    class output_spec(TraitedSpec):
        lut_file = traits.File(exists=True)
        seg_file = traits.File(exists=True)
        mask_file = traits.File(exists=True)
        edge_file = traits.File(exists=True)

    def _run_interface(self, runtime):

        # Load the template-space anatomy to define affine/header
        anat_img = nib.load(self.inputs.anat_file)
        affine, header = anat_img.affine, anat_img.header

        # Load the template-space wmparc files
        fs_img = nib.load(self.inputs.wmparc_file)
        fs_data = fs_img.get_fdata()

        # Remap the wmparc ids to more general classifications
        seg_data = np.zeros_like(fs_data, np.int8)

        seg_ids = [
            np.arange(1000, 3000),  # Cortical gray matter
            [10, 11, 12, 13, 17, 18, 49, 50, 51, 52, 53, 54],  # Subcortical
            [16, 28, 60],  # Brain stem and ventral diencepedgen
            [8, 47],  # Cerebellar gray matter
            np.arange(3000, 5000),  # Superficial ("cortical") white matter
            [5001, 5002],  # Deep white matter
            [7, 46],  # Cerebellar white matter
            [4, 43, 31, 63],  # Lateral ventricle CSF
        ]

        for seg_val, id_vals in enumerate(seg_ids, 1):
            mask = np.isin(fs_data, id_vals)
            seg_data[mask] = seg_val

        # Reclassify any surface voxel as cortical gray matter
        surf_img = nib.load(self.inputs.surf_file)
        surf_data = (surf_img.get_fdata() > -1).any(axis=-1)
        seg_data[surf_data] = 1

        # Generate a lookup table for the new segmentation
        lut = pd.DataFrame([
            ["Unknown", 0, 0, 0, 0],
            ["Cortical-gray-matter", 59, 95, 138, 255],
            ["Subcortical-gray-matter", 91, 129, 177, 255],
            ["Brain-stem", 126, 163, 209, 255],
            ["Cerebellar-gray-matter", 168, 197, 233, 255],
            ["Superficial-white-matter", 206, 129, 134, 255],
            ["Deep-white-matter", 184, 103, 109, 255],
            ["Cerebellar-white-matter", 155, 78, 73, 255],
            ["CSF", 251, 221, 122, 255]
        ])

        # Write out the new segmentation image and lookup table
        self.write_image("seg_file", "seg.nii.gz", seg_data, affine, header)

        # Write a the RGB lookup table as a text file
        lut_file = self.define_output("lut_file", "seg.lut")
        lut.to_csv(lut_file, sep="\t", header=False, index=True)

        # --- Whole brain mask

        # Binarize the segmentation and dilate to generate a brain mask

        brainmask = seg_data > 0
        brainmask = ndimage.binary_dilation(brainmask, iterations=3)
        brainmask = ndimage.binary_erosion(brainmask, iterations=2)
        brainmask = ndimage.binary_fill_holes(brainmask)
        brainmask = brainmask.astype(np.uint8)
        self.write_image("mask_file", "mask.nii.gz", brainmask, affine, header)

        # --- Edge of brain mask
        brainmask_dil = ndimage.binary_dilation(brainmask, iterations=2)
        edge = brainmask_dil - brainmask
        self.write_image("edge_file", "edge.nii.gz", edge, affine, header)

        return runtime


class MakeRibbon(LymanInterface):

    class input_spec(TraitedSpec):
        in_file = traits.File(exists=True)

    class output_spec(TraitedSpec):
        out_file = traits.File(exists=True)

    def _run_interface(self, runtime):

        img = nib.load(self.inputs.in_file)
        affine, header = img.affine, img.header
        vertices = img.get_fdata()
        ribbon = (vertices > -1).any(axis=-1).astype(np.int8)

        self.write_image("out_file", "ribbon.nii.gz", ribbon, affine, header)

        return runtime


class MaskWithLabel(LymanInterface):

    class input_spec(TraitedSpec):
        in_file = traits.File(exists=True)
        label_files = traits.Dict(traits.Str, traits.File)
        hemi = traits.Enum("lh", "rh")
        fill_value = traits.Float()

    class output_spec(TraitedSpec):
        out_file = traits.File(exists=True)

    def _run_interface(self, runtime):

        img = nib.load(self.inputs.in_file)
        affine, header = img.affine, img.header
        data = img.get_fdata()

        label_file = self.inputs.label_files[self.inputs.hemi]
        label_vertices = nib.freesurfer.read_label(label_file)

        data[~np.isin(data, label_vertices)] = self.inputs.fill_value
        self.write_image("out_file", "masked.nii.gz", data, affine, header)

        return runtime


class TemplateReport(LymanInterface):

    class input_spec(TraitedSpec):
        lut_file = traits.File(exists=True)
        seg_file = traits.File(exists=True)
        edge_file = traits.File(exists=True)
        mask_file = traits.File(exists=True)
        surf_file = traits.File(exists=True)
        anat_file = traits.File(exists=True)

    class output_spec(TraitedSpec):
        seg_plot = traits.File(exists=True)
        mask_plot = traits.File(exists=True)
        edge_plot = traits.File(exists=True)
        surf_plot = traits.File(exists=True)
        anat_plot = traits.File(exists=True)

    def _run_interface(self, runtime):

        # Anatomical template
        mask_img = nib.load(self.inputs.mask_file)
        anat_img = nib.load(self.inputs.anat_file)
        m_anat = Mosaic(anat_img, step=2, tight=True)
        self.write_visualization("anat_plot", "anat.png", m_anat)

        # Anatomical segmentation
        seg_img = nib.load(self.inputs.seg_file)
        seg_data = seg_img.get_fdata().astype(np.float)
        seg_lut = pd.read_csv(self.inputs.lut_file, sep="\t", header=None)
        seg_rgb = seg_lut.loc[1:, [2, 3, 4]].values / 255
        seg_cmap = mpl.colors.ListedColormap(seg_rgb)
        m_seg = Mosaic(anat_img, seg_data, mask_img,
                       step=2, tight=True, show_mask=False)
        m_seg.plot_overlay(seg_cmap, 1, 8, thresh=.5, fmt=None)
        self.write_visualization("seg_plot", "seg.png", m_seg)

        # Brain mask
        m_mask = Mosaic(anat_img, mask_img, mask_img,
                        step=2, tight=True, show_mask=False)
        m_mask.plot_mask()
        self.write_visualization("mask_plot", "mask.png", m_mask)

        # Brain edge
        edge_img = nib.load(self.inputs.edge_file)
        m_edge = Mosaic(anat_img, edge_img, mask_img,
                        step=2, tight=True, show_mask=False)
        m_edge.plot_mask()
        self.write_visualization("edge_plot", "edge.png", m_edge)

        # Surface ribbon
        surf_img = nib.load(self.inputs.surf_file)
        surf = surf_img.get_fdata()
        ribbon = np.zeros(anat_img.shape)
        ribbon[surf[..., 0] > 0] = 1
        ribbon[surf[..., 1] > 0] = 2
        ribbon_cmap = mpl.colors.ListedColormap(["#5ebe82", "#ec966f"])
        m_surf = Mosaic(anat_img, ribbon, mask_img,
                        step=2, tight=True, show_mask=False)
        m_surf.plot_overlay(ribbon_cmap, 1, 2, thresh=.5, fmt=None)
        self.write_visualization("surf_plot", "surf.png", m_surf)

        return runtime
