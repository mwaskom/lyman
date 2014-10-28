"""
Estimate nonlinear normalization parameters from Freesurfer conformed space
to FSL's nonlinear MNI152 target using FLIRT and FNIRT.

See the docstring for the create_normalization_workflow function for more
information about the processing.

"""
import os
import os.path as op
from moss.mosaic import Mosaic

from nipype import (IdentityInterface,
                    SelectFiles, DataSink,
                    Node, Workflow)
from nipype.interfaces import fsl, freesurfer as fs
from nipype.interfaces.base import (BaseInterface,
                                    BaseInterfaceInputSpec,
                                    TraitedSpec, File)

from lyman.tools import submit_cmdline


def create_fsl_workflow(data_dir=None, subjects=None, name="fslwarp"):
    """Set up the anatomical normalzation workflow using FNIRT.

    Your anatomical data must have been processed in Freesurfer.
    Unlike most lyman workflows, the DataGrabber and DataSink
    nodes are hardwired within the returned workflow, as this
    tightly integrates with the Freesurfer subjects directory
    structure.

    Parameters
    ----------
    data_dir : path
        top level of data hierarchy/FS subjects directory
    subjects : list of strings
        list of subject IDs
    name : alphanumeric string, optional
        workflow name

    """
    if data_dir is None:
        data_dir = os.environ["SUBJECTS_DIR"]
    if subjects is None:
        subjects = []

    # Get target images
    target_brain = fsl.Info.standard_image("avg152T1_brain.nii.gz")
    target_head = fsl.Info.standard_image("avg152T1.nii.gz")
    hires_head = fsl.Info.standard_image("MNI152_T1_1mm.nii.gz")
    target_mask = fsl.Info.standard_image(
        "MNI152_T1_2mm_brain_mask_dil.nii.gz")
    fnirt_cfg = os.path.join(
        os.environ["FSLDIR"], "etc/flirtsch/T1_2_MNI152_2mm.cnf")

    # Subject source node
    subjectsource = Node(IdentityInterface(fields=["subject_id"]),
                         iterables=("subject_id", subjects),
                         name="subjectsource")

    # Grab recon-all outputs
    head_image = "T1"
    templates = dict(aseg="{subject_id}/mri/aparc+aseg.mgz",
                     head="{subject_id}/mri/" + head_image + ".mgz")
    datasource = Node(SelectFiles(templates,
                                  base_directory=data_dir),
                      "datasource")

    # Convert images to nifti storage and float representation
    cvtaseg = Node(fs.MRIConvert(out_type="niigz"), "convertaseg")

    cvthead = Node(fs.MRIConvert(out_type="niigz",
                                 out_datatype="float"),
                   "converthead")

    # Turn the aparc+aseg into a brainmask
    makemask = Node(fs.Binarize(dilate=1, min=0.5), "makemask")

    # Extract the brain from the orig.mgz using the mask
    skullstrip = Node(fsl.ApplyMask(), "skullstrip")

    # FLIRT brain to MNI152_brain
    flirt = Node(fsl.FLIRT(reference=target_brain), "flirt")

    sw = [-180, 180]
    for dim in ["x", "y", "z"]:
        setattr(flirt.inputs, "searchr_%s" % dim, sw)

    # FNIRT head to MNI152
    fnirt = Node(fsl.FNIRT(ref_file=target_head,
                           refmask_file=target_mask,
                           config_file=fnirt_cfg,
                           fieldcoeff_file=True),
                 "fnirt")

    # Warp and rename the images
    warpbrain = Node(fsl.ApplyWarp(ref_file=target_head,
                                   interp="spline",
                                   out_file="brain_warp.nii.gz"),
                     "warpbrain")

    warpbrainhr = Node(fsl.ApplyWarp(ref_file=hires_head,
                                     interp="spline",
                                     out_file="brain_warp_hires.nii.gz"),
                       "warpbrainhr")

    # Generate a png summarizing the registration
    warpreport = Node(WarpReport(), "warpreport")

    # Save relevant files to the data directory
    fnirt_subs = [(head_image + "_out_masked_flirt.mat", "affine.mat"),
                  (head_image + "_out_fieldwarp", "warpfield"),
                  (head_image + "_out_masked", "brain"),
                  (head_image + "_out", "T1")]
    datasink = Node(DataSink(base_directory=data_dir,
                             parameterization=False,
                             substitutions=fnirt_subs),
                    "datasink")

    # Define and connect the workflow
    # -------------------------------

    normalize = Workflow(name=name)

    normalize.connect([
        (subjectsource, datasource,
            [("subject_id", "subject_id")]),
        (datasource, cvtaseg,
            [("aseg", "in_file")]),
        (datasource, cvthead,
            [("head", "in_file")]),
        (cvtaseg, makemask,
            [("out_file", "in_file")]),
        (cvthead, skullstrip,
            [("out_file", "in_file")]),
        (makemask, skullstrip,
            [("binary_file", "mask_file")]),
        (skullstrip, flirt,
            [("out_file", "in_file")]),
        (flirt, fnirt,
            [("out_matrix_file", "affine_file")]),
        (cvthead, fnirt,
            [("out_file", "in_file")]),
        (skullstrip, warpbrain,
            [("out_file", "in_file")]),
        (fnirt, warpbrain,
            [("fieldcoeff_file", "field_file")]),
        (skullstrip, warpbrainhr,
            [("out_file", "in_file")]),
        (fnirt, warpbrainhr,
            [("fieldcoeff_file", "field_file")]),
        (warpbrain, warpreport,
            [("out_file", "in_file")]),
        (subjectsource, datasink,
            [("subject_id", "container")]),
        (skullstrip, datasink,
            [("out_file", "normalization.@brain")]),
        (cvthead, datasink,
            [("out_file", "normalization.@t1")]),
        (flirt, datasink,
            [("out_file", "normalization.@brain_flirted")]),
        (flirt, datasink,
            [("out_matrix_file", "normalization.@affine")]),
        (warpbrain, datasink,
            [("out_file", "normalization.@brain_warped")]),
        (warpbrainhr, datasink,
            [("out_file", "normalization.@brain_hires")]),
        (fnirt, datasink,
            [("fieldcoeff_file", "normalization.@warpfield")]),
        (warpreport, datasink,
            [("out_file", "normalization.@report")]),
    ])

    return normalize


def create_ants_workflow(data_dir=None, subjects=None, name="antswarp"):
    """Set up the anatomical normalzation workflow using ANTS.

    Your anatomical data must have been processed in Freesurfer.
    Unlike most lyman workflows, the DataGrabber and DataSink
    nodes are hardwired within the returned workflow, as this
    tightly integrates with the Freesurfer subjects directory
    structure.

    Parameters
    ----------
    data_dir : path
        top level of data hierarchy/FS subjects directory
    subjects : list of strings
        list of subject IDs
    name : alphanumeric string, optional
        workflow name

    """
    if data_dir is None:
        data_dir = os.environ["SUBJECTS_DIR"]
    if subjects is None:
        subjects = []

    # Subject source node
    subjectsource = Node(IdentityInterface(fields=["subject_id"]),
                         iterables=("subject_id", subjects),
                         name="subjectsource")

    # Grab recon-all outputs
    templates = dict(aseg="{subject_id}/mri/aparc+aseg.mgz",
                     head="{subject_id}/mri/brain.mgz")
    datasource = Node(SelectFiles(templates,
                                  base_directory=data_dir),
                      "datasource")

    # Convert images to nifti storage and float representation
    cvtaseg = Node(fs.MRIConvert(out_type="niigz"), "convertaseg")

    cvtbrain = Node(fs.MRIConvert(out_type="niigz",
                                  out_datatype="float"),
                    "convertbrain")

    # Turn the aparc+aseg into a brainmask
    makemask = Node(fs.Binarize(dilate=4, erode=3, min=0.5), "makemask")

    # Extract the brain from the orig.mgz using the mask
    skullstrip = Node(fsl.ApplyMask(), "skullstrip")

    # Normalize using ANTS
    antswarp = Node(ANTSIntroduction(), "antswarp")

    # Generate a png summarizing the registration
    warpreport = Node(WarpReport(), "warpreport")

    # Save relevant files to the data directory
    datasink = Node(DataSink(base_directory=data_dir,
                             parameterization=False),
                    name="datasink")

    # Define and connect the workflow
    # -------------------------------

    normalize = Workflow(name=name)

    normalize.connect([
        (subjectsource, datasource,
            [("subject_id", "subject_id")]),
        (datasource, cvtaseg,
            [("aseg", "in_file")]),
        (datasource, cvtbrain,
            [("head", "in_file")]),
        (cvtaseg, makemask,
            [("out_file", "in_file")]),
        (cvtbrain, skullstrip,
            [("out_file", "in_file")]),
        (makemask, skullstrip,
            [("binary_file", "mask_file")]),
        (skullstrip, antswarp,
            [("out_file", "in_file")]),
        (antswarp, warpreport,
            [("brain_file", "in_file")]),
        (subjectsource, datasink,
            [("subject_id", "container")]),
        (antswarp, datasink,
            [("warp_file", "normalization.@warpfield"),
             ("inv_warp_file", "normalization.@inverse_warpfield"),
             ("affine_file", "normalization.@affine"),
             ("brain_file", "normalization.@brain")]),
        (warpreport, datasink,
            [("out_file", "normalization.@report")]),
    ])

    return normalize


class ANTSIntroductionInputSpec(BaseInterfaceInputSpec):

    in_file = File(exists=True)


class ANTSIntroductionOutputSpec(TraitedSpec):

    brain_file = File(exitsts=True)
    warp_file = File(exitsts=True)
    inv_warp_file = File(exitsts=True)
    affine_file = File(exitsts=True)


class ANTSIntroduction(BaseInterface):

    input_spec = ANTSIntroductionInputSpec
    output_spec = ANTSIntroductionOutputSpec

    def _run_interface(self, runtime):

        target_brain = fsl.Info.standard_image("avg152T1_brain.nii.gz")
        cmdline = " ".join(["antsIntroduction.sh",
                            "-d", "3",
                            "-r", target_brain,
                            "-i", self.inputs.in_file,
                            "-o", "ants_"])

        runtime.environ["ITK_NUM_THREADS"] = "1"
        runtime = submit_cmdline(runtime, cmdline)

        os.rename("ants_Affine.txt", "affine.txt")
        os.rename("ants_Warp.nii.gz", "warpfield.nii.gz")
        os.rename("ants_deformed.nii.gz", "brain_warp.nii.gz")
        os.rename("ants_InverseWarp.nii.gz", "inverse_warpfield.nii.gz")

        return runtime

    def _list_outputs(self):

        outputs = self._outputs().get()
        outputs["affine_file"] = op.abspath("affine.txt")
        outputs["warp_file"] = op.abspath("warpfield.nii.gz")
        outputs["brain_file"] = op.abspath("brain_warp.nii.gz")
        outputs["inv_warp_file"] = op.abspath("inverse_warpfield.nii.gz")

        return outputs


class WarpReportInput(BaseInterfaceInputSpec):

    in_file = File(exists=True)


class WarpReportOutput(TraitedSpec):

    out_file = File(exitsts=True)


class WarpReport(BaseInterface):

    input_spec = WarpReportInput
    output_spec = WarpReportOutput

    def _run_interface(self, runtime):

        target_brain = fsl.Info.standard_image("avg152T1_brain.nii.gz")
        m = Mosaic(self.inputs.in_file, target_brain)
        m.plot_contours("Reds", 2)
        m.savefig("warp_report.png")
        m.close()

        return runtime

    def _list_outputs(self):

        outputs = self._outputs().get()
        outputs["out_file"] = op.realpath("warp_report.png")
        return outputs
