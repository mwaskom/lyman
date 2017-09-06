import os.path as op

import numpy as np
from scipy import ndimage
import matplotlib as mpl
import nibabel as nib

from nipype import (Workflow, Node, JoinNode,
                    IdentityInterface, Function, DataSink)
from nipype.interfaces.base import traits, TraitedSpec
from nipype.interfaces import fsl, freesurfer as fs

from ..utils import LymanInterface
from ..visualizations import Mosaic


def define_template_workflow(proj_info, subjects, qc=True):

    # --- Workflow parameterization

    subject_source = Node(IdentityInterface(["subject"]),
                          name="subject_source",
                          iterables=("subject", subjects))

    # Data input
    template_input = Node(TemplateInput(data_dir=proj_info.data_dir),
                          "template_input")

    # --- Definition of functional template space

    crop_image = Node(fs.ApplyMask(args="-bb 4"), "crop_image")

    zoom_image = Node(fs.MRIConvert(resample_type="cubic",
                                    out_type="niigz",
                                    vox_size=(2, 2, 2),  # TODO make proj param
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

    # --- Segementation of anatomical tissue in functional space

    transform_wmparc = Node(fs.ApplyVolTransform(inverse=True,
                                                 interp="nearest",
                                                 args="--keep-precision"),
                            "transform_wmparc")

    anat_segment = Node(AnatomicalSegmentation(), "anat_segment")

    # --- Identification of surface vertices

    hemi_source = Node(IdentityInterface(["hemi"]), "hemi_source",
                       iterables=("hemi", ["lh", "rh"]))

    tag_surf = Node(fs.Surface2VolTransform(surf_name="graymid",
                                            transformed_file="ribbon.nii.gz",
                                            vertexvol_file="vertices.nii.gz",
                                            mkmask=True),
                    "tag_surf")

    combine_hemis = JoinNode(fsl.Merge(dimension="t",
                                       merged_file="surf.nii.gz"),
                             name="combine_hemis",
                             joinsource="hemi_source",
                             joinfield="in_files")

    # --- Template QC

    template_qc = Node(TemplateReport(), "template_qc")

    # --- Workflow ouptut

    def define_template_path(subject):
        return "{}/template".format(subject)

    template_path = Node(Function("subject", "path",
                                  define_template_path),
                         "template_path")

    template_output = Node(DataSink(base_directory=proj_info.analysis_dir,
                                    parameterization=False),
                           "template_output")

    # === Assemble pipeline

    workflow = Workflow(name="template", base_dir=proj_info.cache_dir)

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

        (reorient_image, transform_wmparc,
            [("out_file", "source_file")]),
        (template_input, transform_wmparc,
            [("wmparc_file", "target_file")]),
        (invert_reg, transform_wmparc,
            [("reg_file", "reg_file")]),
        (transform_wmparc, anat_segment,
            [("transformed_file", "wmparc_file")]),

        (hemi_source, tag_surf,
            [("hemi", "hemi")]),
        (invert_reg, tag_surf,
            [("reg_file", "reg_file")]),
        (reorient_image, tag_surf,
            [("out_file", "template_file")]),
        (tag_surf, combine_hemis,
            [("vertexvol_file", "in_files")]),

        (subject_source, template_path,
            [("subject", "subject")]),
        (template_path, template_output,
            [("path", "container")]),
        (reorient_image, template_output,
            [("out_file", "@anat")]),
        (generate_reg, template_output,
            [("fsl_file", "@anat2func")]),
        (anat_segment, template_output,
            [("seg_file", "@seg"),
             ("mask_file", "@mask")]),
        (combine_hemis, template_output,
            [("merged_file", "@surf")]),

    ]
    workflow.connect(processing_edges)

    # Optionally connect QC nodes

    qc_edges = [

        (reorient_image, template_qc,
            [("out_file", "anat_file")]),
        (anat_segment, template_qc,
            [("seg_file", "seg_file"),
             ("mask_file", "mask_file")]),
        (combine_hemis, template_qc,
            [("merged_file", "surf_file")]),

        (template_qc, template_output,
            [("seg_plot", "qc.@seg_plot"),
             ("mask_plot", "qc.@mask_plot"),
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

    def _run_interface(self, runtime):

        mri_dir = op.join(self.inputs.data_dir, self.inputs.subject, "mri")
        results = dict(
            norm_file=op.join(mri_dir, "norm.mgz"),
            wmparc_file=op.join(mri_dir, "wmparc.mgz"),
        )
        self._results.update(results)
        return runtime


class AnatomicalSegmentation(LymanInterface):

    class input_spec(TraitedSpec):
        wmparc_file = traits.File(exists=True)

    class output_spec(TraitedSpec):
        seg_file = traits.File(exists=True)
        mask_file = traits.File(exists=True)

    def _run_interface(self, runtime):

        # Load the template-space wmparc and reclassify voxels
        fs_img = nib.load(self.inputs.wmparc_file)
        fs_data = fs_img.get_data()
        affine, header = fs_img.affine, fs_img.header

        seg_data = np.zeros_like(fs_data)

        seg_ids = [
            np.arange(1000, 3000),  # Cortical gray matter
            [10, 11, 12, 13, 17, 18, 49, 50, 51, 52, 53, 54],  # Subcortical
            [16, 28, 60],  # Brain stem and ventral diencephalon
            [8, 47],  # Cerebellar gray matter
            np.arange(3000, 5000),  # Superficial ("cortical") white matter
            [5001, 5002],  # Deep white matter
            [7, 46],  # Cerebellar white matter
            [4, 43, 31, 63],  # Lateral ventricle CSF
        ]

        for seg_val, id_vals in enumerate(seg_ids, 1):
            mask = np.in1d(fs_data.flat, id_vals).reshape(seg_data.shape)
            seg_data[mask] = seg_val

        self.write_image("seg_file", "seg.nii.gz", seg_data, affine, header)

        # TODO Write a seg.lut for Freeview

        # --- Whole brain mask

        # Binarize the segmentation and dilate to generate a brain mask

        brainmask = seg_data > 0
        brainmask = ndimage.binary_dilation(brainmask, iterations=2)
        brainmask = ndimage.binary_erosion(brainmask)
        brainmask = ndimage.binary_fill_holes(brainmask)
        brainmask = brainmask.astype(np.uint8)
        self.write_image("mask_file", "mask.nii.gz", brainmask, affine, header)

        return runtime


class TemplateReport(LymanInterface):

    class input_spec(TraitedSpec):
        seg_file = traits.File(exists=True)
        mask_file = traits.File(exists=True)
        surf_file = traits.File(exists=True)
        anat_file = traits.File(exists=True)

    class output_spec(TraitedSpec):
        seg_plot = traits.File(exists=True)
        mask_plot = traits.File(exists=True)
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
        seg_cmap = mpl.colors.ListedColormap(  # TODO get from seg lut
            ['#3b5f8a', '#5b81b1', '#7ea3d1', '#a8c5e9',
             '#ce8186', '#b8676d', '#9b4e53', '#fbdd7a']
         )
        m_seg = Mosaic(anat_img, seg_img, mask_img,
                       step=2, tight=True, show_mask=False)
        m_seg.plot_overlay(seg_cmap, 1, 8, thresh=.5, fmt=None)
        self.write_visualization("seg_plot", "seg.png", m_seg)

        # Brain mask
        m_mask = Mosaic(anat_img, mask_img, mask_img,
                        step=2, tight=True, show_mask=False)
        m_mask.plot_mask()
        self.write_visualization("mask_plot", "mask.png", m_mask)

        # Surface ribbon
        surf_img = nib.load(self.inputs.surf_file)
        surf = surf_img.get_data()
        ribbon = np.zeros(anat_img.shape)
        ribbon[surf[..., 0] > 0] = 1
        ribbon[surf[..., 1] > 0] = 2
        ribbon_cmap = mpl.colors.ListedColormap(["#5ebe82", "#ec966f"])
        m_surf = Mosaic(anat_img, ribbon, mask_img,
                        step=2, tight=True, show_mask=False)
        m_surf.plot_overlay(ribbon_cmap, 1, 2, thresh=.5, fmt=None)
        self.write_visualization("surf_plot", "surf.png", m_surf)

        return runtime
