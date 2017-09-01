import os
import os.path as op
import shutil

import numpy as np
from scipy import ndimage
import matplotlib as mpl
import nibabel as nib

from nipype import Workflow, Node, IdentityInterface, Function, DataSink
from nipype.interfaces.base import traits, TraitedSpec

from ..utils import LymanInterface
from ..plotting import Mosaic


def define_template_workflow(proj_info, subjects, qc=True):

    # --- Workflow parameterization

    subject_source = Node(IdentityInterface(["subject"]),
                          name="subject_source",
                          iterables=("subject", subjects))

    # --- Definition of functinoal template space

    define_template = Node(DefineTemplateSpace(data_dir=proj_info.data_dir,
                                               voxel_size=(2, 2, 2)),
                           "define_template")

    # --- Segementation of anatomical tissue in functional space

    anat_segment = Node(AnatomicalSegmentation(), "anat_segment")

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

        (subject_source, define_template,
            [("subject", "subject")]),
        (define_template, anat_segment,
            [("func2anat_dat", "reg_file"),
             ("t1w_file", "template_file")]),
        (subject_source, template_path,
            [("subject", "subject")]),
        (template_path, template_output,
            [("path", "container")]),
        (define_template, template_output,
            [("anat_file", "@anat"),
             ("t1w_file", "@t1w"),
             ("t2w_file", "@t2w"),
             ("func2anat_dat", "@func2anat_dat"),
             ("anat2func_dat", "@anat2func_dat"),
             ("func2anat_mat", "@func2anat_mat"),
             ("anat2func_mat", "@anat2func_mat")]),
        (anat_segment, template_output,
            [("seg_file", "@seg"),
             ("mask_file", "@mask"),
             ("surf_file", "@surf")]),

    ]
    workflow.connect(processing_edges)

    # Optionally connect QC nodes

    qc_edges = [

        (define_template, template_output,
            [("t1w_plot", "qc.@t1w_plot"),
             ("t2w_plot", "qc.@t2w_plot")]),
        (anat_segment, template_output,
            [("seg_plot", "qc.@seg_plot"),
             ("mask_plot", "qc.@mask_plot"),
             ("surf_plot", "qc.@surf_plot")]),

    ]
    if qc:
        workflow.connect(qc_edges)

    return workflow


# =========================================================================== #
# Custom processing nodes
# =========================================================================== #


class DefineTemplateSpace(LymanInterface):

    class input_spec(TraitedSpec):
        data_dir = traits.Directory(exists=True)
        subject = traits.Str()
        voxel_size = traits.Tuple(traits.Float(),
                                  traits.Float(),
                                  traits.Float())

    class output_spec(TraitedSpec):
        anat2func_dat = traits.File(exists=True)
        func2anat_dat = traits.File(exists=True)
        anat2func_mat = traits.File(exists=True)
        func2anat_mat = traits.File(exists=True)
        anat_file = traits.File(exists=True)
        t1w_file = traits.File(exists=True)
        t1w_plot = traits.File(exists=True)
        t2w_file = traits.File()
        t2w_plot = traits.File()

    def _run_interface(self, runtime):

        subjects_dir = os.environ["SUBJECTS_DIR"]
        self.mri_dir = op.join(subjects_dir, self.inputs.subject, "mri")

        # Transform the T1w image into template space
        t1w_file = self.transform_image(runtime, "norm.mgz",
                                        "t1w_file", "T1w.nii.gz")

        # Duplicate the T1w file with a different name
        anat_file = self.define_output("anat_file", "anat.nii.gz")
        anat_file = shutil.copyfile(t1w_file, anat_file)

        # Transform the T2w image into template space
        # TODO if we use the HCP recon enhancements instead of
        # recon-all -T2pial (which doesn't work well!) this file won't get made
        have_t2w = op.exists(op.join(self.mri_dir, "T2w.norm.mgz"))
        if have_t2w:
            t2w_file = self.transform_image(runtime, "T2w.norm.mgz",
                                            "t2w_file", "T2w.nii.gz")

        # Generate an anat -> func and inverse registration matrices
        # TODO do we need to write both tkreg and fsl format?
        anat2func_dat = self.define_output("anat2func_dat", "anat2func.dat")
        anat2func_mat = self.define_output("anat2func_mat", "anat2func.mat")
        cmdline = ["tkregister2",
                   "--mov", op.join(self.mri_dir, "norm.mgz"),
                   "--targ", t1w_file,
                   "--s", self.inputs.subject,
                   "--regheader",
                   "--reg", anat2func_dat,
                   "--fslregout", anat2func_mat,
                   "--noedit"]
        self.submit_cmdline(runtime, cmdline)

        func2anat_dat = self.define_output("func2anat_dat", "func2anat.dat")
        func2anat_mat = self.define_output("func2anat_mat", "func2anat.mat")
        cmdline = ["tkregister2",
                   "--mov", t1w_file,
                   "--targ", op.join(self.mri_dir, "norm.mgz"),
                   "--s", self.inputs.subject,
                   "--regheader",
                   "--reg", func2anat_dat,
                   "--fslregout", func2anat_mat,
                   "--noedit"]
        self.submit_cmdline(runtime, cmdline)

        # Write out QC mosaics
        t1w_plot = self.define_output("t1w_plot", "T1w.png")
        m = Mosaic(t1w_file)
        m.savefig(t1w_plot)
        m.close()

        if have_t2w:
            t2w_plot = self.define_output("t2w_plot", "T2w.png")
            m = Mosaic(t2w_file)
            m.savefig(t2w_plot)
            m.close()

        return runtime

    def transform_image(self, runtime, in_fname, out_field, out_fname):

        fstem, _ = op.splitext(in_fname)

        # Crop the image using the aseg
        cropped_fname = fstem + "_crop.nii.gz"
        cmdline = ["mri_mask",
                   "-bb", "3",
                   op.join(self.mri_dir, in_fname),
                   op.join(self.mri_dir, "aseg.mgz"),
                   cropped_fname]
        self.submit_cmdline(runtime, cmdline)

        # Downsample to the desired functional resolution
        zoomed_fname = fstem + "_crop_zoom.nii.gz"
        cmdline = ["mri_convert",
                   cropped_fname,
                   zoomed_fname,
                   "-rt", "cubic",
                   "-vs"]
        cmdline.extend(map(str, self.inputs.voxel_size))
        self.submit_cmdline(runtime, cmdline)

        # Reorient to cannonical orientation
        out_file = self.define_output(out_field, out_fname)
        cmdline = ["fslreorient2std", zoomed_fname, out_file]
        self.submit_cmdline(runtime, cmdline)

        return out_file


class AnatomicalSegmentation(LymanInterface):

    class input_spec(TraitedSpec):
        template_file = traits.File(exists=True)
        reg_file = traits.File(exists=True)

    class output_spec(TraitedSpec):
        seg_file = traits.File(exists=True)
        seg_plot = traits.File(exists=True)
        mask_file = traits.File(exists=True)
        mask_plot = traits.File(exists=True)
        surf_file = traits.File(exists=True)
        surf_plot = traits.File(exists=True)

    def _run_interface(self, runtime):

        # Define the template space geometry

        template_img = nib.load(self.inputs.template_file)

        # --- Coarse segmentation into anatomical components

        # Transform the wmparc image into functional space

        fs_fname = "wmparc.nii.gz"
        cmdline = ["mri_vol2vol",
                   "--nearest",
                   "--keep-precision",
                   "--inv",
                   "--mov", self.inputs.template_file,
                   "--fstarg", "wmparc.mgz",
                   "--reg", self.inputs.reg_file,
                   "--o", fs_fname]

        self.submit_cmdline(runtime, cmdline)

        # Load the template-space wmparc and reclassify voxels

        fs_img = nib.load(fs_fname)
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

        seg_img = self.write_image("seg_file", "seg.nii.gz",
                                   seg_data, affine, header)

        # TODO Write a seg.lut for Freeview

        # --- Whole brain mask

        # Binarize the segmentation and dilate to generate a brain mask

        brainmask = seg_data > 0
        brainmask = ndimage.binary_dilation(brainmask, iterations=2)
        brainmask = ndimage.binary_erosion(brainmask)
        brainmask = ndimage.binary_fill_holes(brainmask)
        mask_img = self.write_image("mask_file", "mask.nii.gz",
                                    brainmask.astype(np.int), affine, header)

        # --- Surface vertex mapping

        hemi_files = []
        for hemi in ["lh", "rh"]:
            hemi_mask = "{}.ribbon.nii.gz".format(hemi)
            hemi_file = "{}.surf.nii.gz".format(hemi)
            cmdline = ["mri_surf2vol",
                       "--template", self.inputs.template_file,
                       "--reg", self.inputs.reg_file,
                       "--surf", "graymid",
                       "--hemi", hemi,
                       "--mkmask",
                       "--o", hemi_mask,
                       "--vtxvol", hemi_file]

            self.submit_cmdline(runtime, cmdline)
            hemi_files.append(hemi_file)

        hemi_data = [nib.load(f).get_data() for f in hemi_files]
        surf = np.stack(hemi_data, axis=-1)
        self.write_image("surf_file", "surf.nii.gz", surf, affine, header)

        # --- Generate QC mosaics

        # Anatomical segmentation

        seg_plot = self.define_output("seg_plot", "seg.png")
        seg_cmap = mpl.colors.ListedColormap(
            ['#3b5f8a', '#5b81b1', '#7ea3d1', '#a8c5e9',
             '#ce8186', '#b8676d', '#9b4e53', '#fbdd7a']
         )
        m_seg = Mosaic(template_img, seg_img, mask_img,
                       step=2, tight=True, show_mask=False)
        m_seg.plot_overlay(seg_cmap, 1, 8, thresh=.5, fmt=None)
        m_seg.savefig(seg_plot)
        m_seg.close()

        # Brain mask

        mask_plot = self.define_output("mask_plot", "mask.png")
        m_mask = Mosaic(template_img, mask_img, mask_img,
                        step=2, tight=True, show_mask=False)
        m_mask.plot_mask()
        m_mask.savefig(mask_plot)
        m_mask.close()

        # Surface ribbon

        surf_plot = self.define_output("surf_plot", "surf.png")
        ribbon = np.zeros(template_img.shape)
        ribbon[surf[..., 0] > 0] = 1
        ribbon[surf[..., 1] > 0] = 2
        ribbon_cmap = mpl.colors.ListedColormap(["#5ebe82", "#ec966f"])
        m_surf = Mosaic(template_img, ribbon, mask_img,
                        step=2, tight=True, show_mask=False)
        m_surf.plot_overlay(ribbon_cmap, 1, 2, thresh=.5, fmt=None)
        m_surf.savefig(surf_plot)
        m_surf.close()

        return runtime
