import os
import os.path as op
import shutil

import numpy as np
from scipy import ndimage
import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib

from nipype import (Workflow, Node, MapNode, JoinNode,
                    IdentityInterface, Function, DataSink)
from nipype.interfaces.base import traits, TraitedSpec, isdefined
from nipype.interfaces import fsl, freesurfer as fs

from .preproc import TimeSeriesGIF  # TODO Put this somewhere independent
from ..mosaic import Mosaic
from ..graphutils import SimpleInterface


def define_template_workflow(proj_info, subjects, qc=True):

    # --- Workflow parameterization and data input

    subject_iterables = subjects

    subject_source = Node(IdentityInterface(["subject"]),
                          name="subject_source",
                          iterables=("subject", subject_iterables))

    session_iterables = {
        subj: [(subj, sess) for sess in proj_info.scan_info[subj]]
        for subj in subjects
    }
    session_source = Node(IdentityInterface(["subject", "session"]),
                          name="session_source",
                          itersource=("subject_source", "subject"),
                          iterables=("session", session_iterables))

    session_input = Node(SessionInput(data_dir=proj_info.data_dir,
                                      fm_template=proj_info.fm_template,
                                      phase_encoding=proj_info.phase_encoding),
                         "session_input")

    # --- Warpfield estimation using topup

    # Distortion warpfield estimation
    estimate_distortions = Node(fsl.TOPUP(config="b02b0.cnf"),
                                "estimate_distortions")

    fieldmap_qc = Node(FieldMapReport(), "fieldmap_qc")

    # Average distortion-corrected spin-echo images
    average_fm = Node(fsl.MeanImage(out_file="fm_restored.nii.gz"),
                      "average_fm")

    # Select the relevant warp image to apply to time series data
    # TODO maybe also save out the jacobian here
    finalize_warp = Node(FinalizeWarp(), "finalize_warp")

    # --- Registration of SE-EPI (without distortions) to Freesurfer anatomy

    fm2anat = Node(fs.BBRegister(init="fsl",
                                 contrast_type="t2",
                                 out_fsl_file="sess2anat.mat",
                                 out_reg_file="sess2anat.dat"),
                   "fm2anat")

    fm2anat_qc = Node(AnatRegReport(), "fm2anat_qc")

    # --- Definition of common cross-session space (template space)

    define_template = JoinNode(DefineTemplateSpace(),
                               name="define_template",
                               joinsource="session_source",
                               joinfield=["session_info",
                                          "in_matrices",
                                          "in_volumes"])

    func2anat_qc = Node(AnatRegReport(), "func2anat_qc")

    # --- Segementation of anatomical tissue in functional space

    anat_segment = Node(AnatomicalSegmentation(), "anat_segment")

    # --- Associate template-space transforms with data from correct session

    # The logic here is a little complex. The template creation node collapses
    # the session-wise iterables and returns list of files that then need
    # to hook back into the iterable parameterization so that they can be
    # associated with data from the correct session when they are applied.

    def select_transform_func(session_info, subject, session, in_matrices):

        for info, matrix in zip(session_info, in_matrices):
            if info == (subject, session):
                out_matrix = matrix
        return out_matrix

    select_sesswise = Node(Function(["session_info",
                                     "subject", "session",
                                     "in_matrices"],
                                    "out_matrix",
                                    select_transform_func),
                           "select_sesswise")

    # --- Restore each sessions SE image in template space then average

    restore_fm = MapNode(fsl.ApplyWarp(interp="spline", relwarp=True),
                         ["in_file", "premat", "field_file"],
                         "restore_fm")

    finalize_template = JoinNode(FinalizeTemplate(),
                                 name="finalize_template",
                                 joinsource="session_source",
                                 joinfield=["in_files"])

    # --- Workflow ouptut

    def define_common_container(subject):
        return "{}/template".format(subject)

    common_container = Node(Function("subject", "path",
                                     define_common_container),
                            "common_container")

    common_output = Node(DataSink(base_directory=proj_info.analysis_dir,
                                  parameterization=False),
                         "common_output")

    def define_session_container(subject, session):
        return "{}/template/{}".format(subject, session)

    session_container = Node(Function(["subject", "session"], "path",
                                      define_session_container),
                             "session_container")

    session_output = Node(DataSink(base_directory=proj_info.analysis_dir,
                                   parameterization=False),
                          "session_output")

    # === Assemble pipeline

    workflow = Workflow(name="template", base_dir=proj_info.cache_dir)

    processing_edges = [

        (subject_source, session_source,
            [("subject", "subject")]),
        (session_source, session_input,
            [("session", "session")]),

        # --- SE-EPI fieldmap processing and template creation

        # Phase-encode distortion estimation

        (session_input, estimate_distortions,
            [("fm", "in_file"),
             ("phase_encoding", "encoding_direction"),
             ("readout_times", "readout_times")]),
        (session_input, finalize_warp,
            [("fm", "fieldmap_file")]),
        (estimate_distortions, finalize_warp,
            [("out_warps", "warp_files")]),
        (estimate_distortions, average_fm,
            [("out_corrected", "in_file")]),

        # Registration of corrected SE-EPI to anatomy

        (session_input, fm2anat,
            [("subject", "subject_id")]),
        (average_fm, fm2anat,
            [("out_file", "source_file")]),
        (average_fm, finalize_warp,
            [("out_file", "func_file")]),

        # Creation of cross-session subject-specific template

        (subject_source, define_template,
            [("subject", "subject_id")]),
        (session_source, define_template,
            [("session", "session_info")]),
        (session_input, define_template,
            [("fm", "in_volumes")]),
        (fm2anat, define_template,
            [("out_fsl_file", "in_matrices")]),
        (define_template, anat_segment,
            [("subject_id", "subject_id"),
             ("reg_file", "reg_file"),
             ("out_template", "template_file")]),
        (define_template, select_sesswise,
            [("out_matrices", "in_matrices"),
             ("out_template", "in_templates"),
             ("session_info", "session_info")]),
        (session_input, select_sesswise,
            [("subject", "subject"),
             ("session", "session")]),
        (session_input, restore_fm,
            [("fm_frames", "in_file")]),
        (estimate_distortions, restore_fm,
            [("out_mats", "premat"),
             ("out_warps", "field_file")]),
        (define_template, restore_fm,
            [("out_template", "ref_file")]),
        (select_sesswise, restore_fm,
            [("out_matrix", "postmat")]),
        (select_sesswise, finalize_warp,
            [("out_matrix", "reg_file")]),
        (restore_fm, finalize_template,
            [("out_file", "in_files")]),
        (anat_segment, finalize_template,
            [("mask_file", "mask_file")]),

        # --- Persistent data storage

        # Ouputs associated with the subject-specific template

        (session_input, common_container,
            [("subject", "subject")]),
        (common_container, common_output,
            [("path", "container")]),
        (finalize_template, common_output,
            [("out_file", "@template")]),
        (define_template, common_output,
            [("reg_file", "@reg")]),
        (anat_segment, common_output,
            [("seg_file", "@seg"),
             ("mask_file", "@mask"),
             ("anat_file", "@anat"),
             ("surf_file", "@surf")]),

        # Ouputs associated with each session

        (session_input, session_container,
            [("subject", "subject"),
             ("session", "session")]),
        (session_container, session_output,
            [("path", "container")]),
        (finalize_warp, session_output,
            [("out_raw", "@raw"),
             ("out_func", "@func"),
             ("out_reg", "@sess2temp_reg"),
             ("out_warp", "@warp")]),
        (fm2anat, session_output,
            [("out_reg_file", "@sess2anat_reg")]),

    ]
    workflow.connect(processing_edges)

    # Optionally connect QC nodes

    qc_edges = [

        # Phase-encode distortion estimation

        (session_input, fieldmap_qc,
            [("fm", "orig_file"),
             ("session", "session"),
             ("phase_encoding", "phase_encoding")]),
        (estimate_distortions, fieldmap_qc,
            [("out_corrected", "corr_file")]),

        # Registration of corrected SE-EPI to anatomy

        (session_input, fm2anat_qc,
            [("subject", "subject_id")]),
        (average_fm, fm2anat_qc,
            [("out_file", "in_file")]),
        (fm2anat, fm2anat_qc,
            [("out_reg_file", "reg_file"),
             ("min_cost_file", "cost_file")]),

        # Creation of cross-session subject-specific template

        (session_input, func2anat_qc,
            [("subject", "subject_id")]),
        (define_template, func2anat_qc,
            [("reg_file", "reg_file")]),
        (finalize_template, func2anat_qc,
            [("out_file", "in_file")]),

        # Ouputs associated with the subject-specific template

        (anat_segment, common_output,
            [("seg_plot", "qc.@seg_plot"),
             ("mask_plot", "qc.@mask_plot"),
             ("anat_plot", "qc.@anat_plot"),
             ("surf_plot", "qc.@surf_plot")]),
        (func2anat_qc, common_output,
            [("out_file", "qc.@func2anat_plot")]),
        (finalize_template, common_output,
            [("out_png", "qc.@template_png"),
             ("out_gif", "qc.@template_gif")]),

        # Outputs associated with individual sessions

        (finalize_warp, session_output,
            [("out_plot", "qc.@warp_plot")]),
        (fieldmap_qc, session_output,
            [("out_file", "qc.@unwarp_gif")]),
        (fm2anat_qc, session_output,
            [("out_file", "qc.@fm2anat_plot")]),

    ]

    if qc:
        workflow.connect(qc_edges)

    return workflow


# =========================================================================== #
# Custom processing nodes
# =========================================================================== #


# ---- Data input


class SessionInput(SimpleInterface):

    class input_spec(TraitedSpec):
        session = traits.Tuple()
        data_dir = traits.Str()
        fm_template = traits.Str()
        phase_encoding = traits.Str()

    class output_spec(TraitedSpec):
        fm = traits.File(exists=True)
        fm_frames = traits.List(traits.File(exists=True))
        phase_encoding = traits.List(traits.Str())
        readout_times = traits.List(traits.Float())
        session_key = traits.Tuple()
        subject = traits.Str()
        session = traits.Str()

    def _run_interface(self, runtime):

        # Determine the phase encoding directions
        pe = self.inputs.phase_encoding
        if pe == "ap":
            pos_pe, neg_pe = "ap", "pa"
        elif pe == "pa":
            pos_pe, neg_pe = "pa", "ap"
        else:
            raise ValueError("Phase encoding must be 'ap' or 'pa'")

        # Determine the parameters
        subject, session = self.inputs.session
        self._results["session_key"] = self.inputs.session
        self._results["subject"] = str(subject)
        self._results["session"] = str(session)

        # Spec out full paths to the pair of fieldmap files
        keys = dict(subject=subject, session=session)
        template = self.inputs.fm_template
        func_dir = op.join(self.inputs.data_dir, subject, "func")
        pos_fname = op.join(func_dir,
                            template.format(encoding=pos_pe, **keys))
        neg_fname = op.join(func_dir,
                            template.format(encoding=neg_pe, **keys))

        # Load the two images in canonical orientation
        pos_img = nib.as_closest_canonical(nib.load(pos_fname))
        neg_img = nib.as_closest_canonical(nib.load(neg_fname))
        affine, header = pos_img.affine, pos_img.header

        # Concatenate the images into a single volume
        pos_data = pos_img.get_data()
        neg_data = neg_img.get_data()
        data = np.concatenate([pos_data, neg_data], axis=-1)
        assert len(data.shape) == 4

        # Convert image datatype to float
        header.set_data_dtype(np.float32)

        # Write out a 4D file
        fname = self.define_output("fm", "fieldmap.nii.gz")
        img = nib.Nifti1Image(data, affine, header)
        img.to_filename(fname)

        # Write out a set of 3D files for each frame
        fm_frames = []
        frames = nib.four_to_three(img)
        for i, frame in enumerate(frames):
            fname = op.abspath("fieldmap_{:02d}.nii.gz".format(i))
            fm_frames.append(fname)
            frame.to_filename(fname)
        self._results["fm_frames"] = fm_frames

        # Define phase encoding and readout times for TOPUP
        pe_dir = ["y"] * pos_img.shape[-1] + ["y-"] * neg_img.shape[-1]
        readout_times = [1 for _ in pe_dir]
        self._results["phase_encoding"] = pe_dir
        self._results["readout_times"] = readout_times

        return runtime


# --- Processing operations


class DefineTemplateSpace(SimpleInterface):

    class input_spec(TraitedSpec):
        subject_id = traits.Str()
        session_info = traits.List(traits.Tuple())
        in_matrices = traits.List(traits.File(exists=True))
        in_volumes = traits.List(traits.File(exists=True))

    class output_spec(TraitedSpec):
        subject_id = traits.Str()
        session_info = traits.List(traits.Tuple())
        out_template = traits.File(exists=True)
        reg_file = traits.File(exists=True)
        out_matrices = traits.List(traits.File(exists=True))

    def _run_interface(self, runtime):

        subjects_dir = os.environ["SUBJECTS_DIR"]

        assert all([s == self.inputs.subject_id
                    for s, _ in self.inputs.session_info])
        subj = self.inputs.subject_id

        self._results["subject_id"] = subj
        self._results["session_info"] = self.inputs.session_info

        # -- Convert the anatomical image to nifti
        anat_file = "orig.nii.gz"
        cmdline = ["mri_convert",
                   op.join(subjects_dir, subj, "mri/orig.mgz"),
                   anat_file]

        self.submit_cmdline(runtime, cmdline)

        # -- Compute the intermediate transform
        midtrans_file = "anat2func.mat"
        cmdline = ["midtrans",
                   "--template=" + anat_file,
                   "--separate=se2template_",
                   "--out=" + midtrans_file]
        cmdline.extend(self.inputs.in_matrices)
        out_matrices = [
            op.abspath("se2template_{:04d}.mat".format(i))
            for i, _ in enumerate(self.inputs.in_matrices, 1)
        ]

        self.submit_cmdline(runtime, cmdline, out_matrices=out_matrices)

        # -- Invert the anat2temp transformation
        flirt_file = "func2anat.mat"
        cmdline = ["convert_xfm",
                   "-omat", flirt_file,
                   "-inverse",
                   midtrans_file]

        self.submit_cmdline(runtime, cmdline)

        # -- Transform first fieldmap info template space to get geometry
        # (Note that we don't write out this image as the template because
        # we separately generate the final file with a single interpolation)
        out_template = self.define_output("out_template", "space.nii.gz")

        cmdline = ["flirt",
                   "-in", self.inputs.in_volumes[0],
                   "-ref", self.inputs.in_volumes[0],
                   "-init", out_matrices[0],
                   "-out", out_template,
                   "-applyxfm"]

        self.submit_cmdline(runtime, cmdline)

        # -- Average frames of fieldmap image in template space
        cmdline = ["fslmaths", out_template, "-Tmean", out_template]

        self.submit_cmdline(runtime, cmdline)

        # -- Convert the FSL matrices to tkreg matrix format
        reg_file = self.define_output("reg_file", "reg.dat")
        cmdline = ["tkregister2",
                   "--s", subj,
                   "--mov", out_template,
                   "--fsl", flirt_file,
                   "--reg", reg_file,
                   "--noedit"]

        self.submit_cmdline(runtime, cmdline)

        return runtime


class FinalizeTemplate(SimpleInterface, TimeSeriesGIF):

    class input_spec(TraitedSpec):
        in_files = traits.List(traits.List(traits.File(exists=True)))
        mask_file = traits.File(exists=True)

    class output_spec(TraitedSpec):
        out_file = traits.File(exists=True)
        out_gif = traits.File(exists=True)
        out_png = traits.File(exists=True)

    def _run_interface(self, runtime):

        # Load the mask image
        mask_img = nib.load(self.inputs.mask_file)
        mask = mask_img.get_data().astype(bool)

        # Load each frame and scale to a common mean
        template_imgs = []
        for sess_files in self.inputs.in_files:
            for fname in sess_files:
                img = nib.load(fname)
                data = img.get_data()

                target = 10000
                scale_value = target / data[mask].mean()
                data = data * scale_value

                img = nib.Nifti1Image(data, img.affine, img.header)
                template_imgs.append(img)

        # Take the temporal mean over all template frames
        frame_img = nib.concat_images(template_imgs)
        template_data = frame_img.get_data().mean(axis=-1)

        # Apply the mask to skullstrip
        template_data[~mask] = 0

        # Write out the 3D template image
        affine, header = frame_img.affine, frame_img.header
        out_file = self.define_output("out_file", "func.nii.gz")
        out_img = nib.Nifti1Image(template_data, affine, header)
        out_img.to_filename(out_file)

        # Make a GIF movie of the template_frames
        out_gif = self.define_output("out_gif", "func.gif")
        self.write_time_series_gif(runtime, frame_img, out_gif)

        # Make a static png of the final template
        out_png = self.define_output("out_png", "func.png")
        m = Mosaic(out_img, mask=mask_img)
        m.savefig(out_png)
        m.close()

        return runtime


class FinalizeWarp(SimpleInterface):

    class input_spec(TraitedSpec):
        fieldmap_file = traits.File(exists=True)
        func_file = traits.File(exists=True)
        reg_file = traits.File(exists=True)
        warp_files = traits.List(traits.File(exists=True))

    class output_spec(TraitedSpec):
        out_reg = traits.File(exists=True)
        out_raw = traits.File(exists=True)
        out_func = traits.File(exists=True)
        out_warp = traits.File(exists=True)
        out_plot = traits.File(exists=True)

    def _run_interface(self, runtime):

        # Copy the session -> template matrix to its output name
        out_reg = self.define_output("out_reg", "sess2temp.mat")
        shutil.copyfile(self.inputs.reg_file, out_reg)

        # Select the first frame of the 4D fieldmap image
        fieldmap_img_4d = nib.load(self.inputs.fieldmap_file)
        fieldmap_img = nib.four_to_three(fieldmap_img_4d)[0]

        # Write out the raw image to serve as a registration target
        out_raw = self.define_output("out_raw", "raw.nii.gz")
        fieldmap_img.to_filename(out_raw)

        # Select the first warpfield image
        # We combine the two fieldmap images so that the first one has
        # a phase encoding that matches the time series data.
        # Also note that when the fieldmap images have multiple frames,
        # the warps corresponding to those frames are identical.
        warp_file = self.inputs.warp_files[0]

        # Load in the the warp file and save out with the correct affine
        # (topup doesn't save the header geometry correctly for some reason)
        out_warp = self.define_output("out_warp", "warp.nii.gz")
        affine, header = fieldmap_img.affine, fieldmap_img.header
        warp_data = nib.load(warp_file).get_data()
        warp_img = nib.Nifti1Image(warp_data, affine, header)
        warp_img.to_filename(out_warp)

        # Select the warp along the phase encode direction
        # Note: we elsewhere currently require phase encoding to be AP or PA
        # so because the input note transforms the fieldmap to canonical
        # orientation this will work. But in the future we might want to ne
        # more flexible with what we accept and will need to change this.
        warp_data_y = warp_data[..., 1]

        # Copy the corrected functional image to the output
        out_func = self.define_output("out_func", "func.nii.gz")
        shutil.copyfile(self.inputs.func_file, out_func)

        # Generate a QC image of the warpfield
        out_plot = self.define_output("out_plot", "warp.png")
        m = Mosaic(fieldmap_img, warp_data_y)
        m.plot_overlay("coolwarm", vmin=-6, vmax=6, alpha=.75)
        m.savefig(out_plot)
        m.close()

        return runtime


class AnatomicalSegmentation(SimpleInterface):

    class input_spec(TraitedSpec):
        subject_id = traits.Str()
        template_file = traits.File(exists=True)
        reg_file = traits.File(exists=True)

    class output_spec(TraitedSpec):
        seg_file = traits.File(exists=True)
        seg_plot = traits.File(exists=True)
        anat_file = traits.File(exists=True)
        anat_plot = traits.File(exists=True)
        mask_file = traits.File(exists=True)
        mask_plot = traits.File(exists=True)
        surf_file = traits.File(exists=True)
        surf_plot = traits.File(exists=True)

    def _run_interface(self, runtime):

        # Define the template space geometry

        template_img = nib.load(self.inputs.template_file)
        affine, header = template_img.affine, template_img.header

        # --- Coarse segmentation into anatomical components

        # Transform the wmparc image into functional space

        fs_fname = "wmparc.nii.gz"
        cmdline = ["mri_vol2vol",
                   "--nearest",
                   "--inv",
                   "--mov", self.inputs.template_file,
                   "--fstarg", "wmparc.mgz",
                   "--reg", self.inputs.reg_file,
                   "--o", fs_fname]

        self.submit_cmdline(runtime, cmdline)

        # Load the template-space wmparc and reclassify voxels

        fs_img = nib.load(fs_fname)
        fs_data = fs_img.get_data()

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

        seg_file = self.define_output("seg_file", "seg.nii.gz")
        seg_img = nib.Nifti1Image(seg_data, affine, header)
        seg_img.to_filename(seg_file)

        # --- Whole brain mask

        # Binarize the segmentation and dilate to generate a brain mask

        brainmask = seg_data > 0
        brainmask = ndimage.binary_dilation(brainmask, iterations=2)
        brainmask = ndimage.binary_erosion(brainmask)
        brainmask = ndimage.binary_fill_holes(brainmask)

        mask_file = self.define_output("mask_file", "mask.nii.gz")
        mask_img = nib.Nifti1Image(brainmask, affine, header)
        mask_img.to_filename(mask_file)

        # --- T1w anatomical image in functional space

        anat_file = self.define_output("anat_file", "anat.nii.gz")
        cmdline = ["mri_vol2vol",
                   "--cubic",
                   "--inv",
                   "--mov", self.inputs.template_file,
                   "--fstarg", "brain.finalsurfs.mgz",
                   "--reg", self.inputs.reg_file,
                   "--o", anat_file]

        self.submit_cmdline(runtime, cmdline)

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

        surf_file = self.define_output("surf_file", "surf.nii.gz")
        surf_img = nib.Nifti1Image(surf, affine, header)
        surf_img.to_filename(surf_file)

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

        # Anatomical image

        anat_plot = self.define_output("anat_plot", "anat.png")
        m_mask = Mosaic(anat_file, mask=mask_img,
                        step=2, tight=True, show_mask=False)
        m_mask.savefig(anat_plot)
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


# --- Quality control image generation


class AnatRegReport(SimpleInterface):

    class input_spec(TraitedSpec):
        subject_id = traits.Str()
        in_file = traits.File(exists=True)
        reg_file = traits.File(exists=True)
        cost_file = traits.File(exists=True)
        out_file = traits.File()

    class output_spec(TraitedSpec):
        out_file = traits.File(exists=True)

    def _run_interface(self, runtime):

        # Use the registration to transform the input file
        registered_file = "func_in_anat.nii.gz"
        cmdline = ["mri_vol2vol",
                   "--mov", self.inputs.in_file,
                   "--reg", self.inputs.reg_file,
                   "--o", registered_file,
                   "--fstarg",
                   "--cubic"]

        self.submit_cmdline(runtime, cmdline)

        # Load the WM segmentation and a brain mask
        mri_dir = op.join(os.environ["SUBJECTS_DIR"],
                          self.inputs.subject_id, "mri")

        wm_file = op.join(mri_dir, "wm.mgz")
        wm_data = (nib.load(wm_file).get_data() > 0).astype(int)

        aseg_file = op.join(mri_dir, "aseg.mgz")
        mask = (nib.load(aseg_file).get_data() > 0).astype(int)

        # Read the final registration cost
        if isdefined(self.inputs.cost_file):
            cost = np.loadtxt(self.inputs.cost_file)[0]
        else:
            cost = None

        # Make a mosaic of the registration from func to wm seg
        # TODO this should be an OrthoMosaic when that is implemented
        out_file = self.define_output("out_file", "reg.png")

        m = Mosaic(registered_file, wm_data, mask, step=3, show_mask=False)
        m.plot_mask_edges()
        if cost is not None:
            m.fig.suptitle("Final cost: {:.2f}".format(cost),
                           size=10, color="white")
        m.savefig(out_file)
        m.close()

        return runtime


class FieldMapReport(SimpleInterface):

    class input_spec(TraitedSpec):
        orig_file = traits.File(exists=True)
        corr_file = traits.File(exists=True)
        phase_encoding = traits.List(traits.Str)
        session = traits.Str()

    class output_spec(TraitedSpec):
        out_file = traits.File(exists=True)

    def _run_interface(self, runtime):

        vol_data = dict(orig=nib.load(self.inputs.orig_file).get_data(),
                        corr=nib.load(self.inputs.corr_file).get_data())

        pe_data = dict(orig=[], corr=[])
        pe = np.array(self.inputs.phase_encoding)
        for enc in np.unique(pe):
            enc_trs = pe == enc
            for scan in ["orig", "corr"]:
                enc_data = vol_data[scan][..., enc_trs].mean(axis=-1)
                pe_data[scan].append(enc_data)

        r_vals = dict()
        for scan, (scan_pos, scan_neg) in pe_data.items():
            r_vals[scan] = np.corrcoef(scan_pos.flat, scan_neg.flat)[0, 1]

        nx, ny, nz, _ = vol_data["orig"].shape
        x_slc = (np.linspace(.2, .8, 8) * nx).astype(np.int)

        vmin, vmax = np.percentile(vol_data["orig"].flat, [2, 98])
        kws = dict(vmin=vmin, vmax=vmax, cmap="gray")

        width = len(x_slc)
        height = (nz / ny) * 2.75

        png_fnames = []
        for i, enc in enumerate(["pos", "neg"]):

            f = plt.figure(figsize=(width, height))
            gs = dict(
                orig=plt.GridSpec(1, len(x_slc), 0, .5, 1, .95, 0, 0),
                corr=plt.GridSpec(1, len(x_slc), 0, 0, 1, .45, 0, 0)
            )

            text_kws = dict(size=7, color="w", backgroundcolor="0",
                            ha="center", va="bottom")
            f.text(.5, .93,
                   "Original similarity: {:.2f}".format(r_vals["orig"]),
                   **text_kws)
            f.text(.5, .43,
                   "Corrected similarity: {:.2f}".format(r_vals["corr"]),
                   **text_kws)

            for scan in ["orig", "corr"]:
                axes = [f.add_subplot(pos) for pos in gs[scan]]
                vol = pe_data[scan][i]

                for ax, x in zip(axes, x_slc):
                    slice = np.rot90(vol[x])
                    ax.imshow(slice, **kws)
                    ax.set_axis_off()

                png_fname = "frame{}.png".format(i)
                png_fnames.append(png_fname)
                f.savefig(png_fname, facecolor="0", edgecolor="0")
                plt.close(f)

        out_file = self.define_output("out_file", "unwarp.gif")
        cmdline = ["convert", "-loop", "0", "-delay", "100"]
        cmdline.extend(png_fnames)
        cmdline.append(out_file)

        self.submit_cmdline(runtime, cmdline)

        return runtime
