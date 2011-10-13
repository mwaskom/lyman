import os

from nipype.interfaces import io
from nipype.interfaces import fsl
from nipype.interfaces import freesurfer as fs
from nipype.interfaces import utility as util
from nipype.pipeline import engine as pe

from .interfaces import CheckReg



def create_normalization_workflow(data_dir, subjects, name="normalize"):

    # Get target images
    target_brain = fsl.Info.standard_image("avg152T1_brain.nii.gz")
    target_head =  fsl.Info.standard_image("avg152T1.nii.gz")
    target_mask = fsl.Info.standard_image("MNI152_T1_2mm_brain_mask_dil.nii.gz")
    fnirt_cfg = os.path.join(os.environ["FSLDIR"], "etc/flirtsch/T1_2_MNI152_2mm.cnf")

    # Subject source node
    subjectsource = pe.Node(util.IdentityInterface(fields=["subject_id"]),
                            iterables = ("subject_id", subjects),
                            name = "subjectsource")

    # Grab recon-all outputs
    datasource = pe.Node(io.DataGrabber(infields=["subject_id"],
                                        outfields=["brain", "head"],
                                        base_directory=data_dir,
                                        template="%s/mri/%s.mgz"),
                         name="datagrabber")
    datasource.inputs.template_args = dict(brain=[["subject_id", "norm"]],
                                           head=[["subject_id", "nu"]])

    # Convert images to nifti storage and float representation
    cvtbrain = pe.Node(fs.MRIConvert(out_type="niigz", out_datatype="float"),
                       name="convertbrain")

    cvthead = pe.Node(fs.MRIConvert(out_type="niigz", out_datatype="float"),
                      name="converthead")

    # FLIRT brain to MNI152_brain
    flirt = pe.Node(fsl.FLIRT(reference=target_brain),
                    name="flirt")
    sw = [-180, 180]
    for dim in ["x","y","z"]:
        setattr(flirt.inputs, "searchr_%s"%dim, sw)

    # FNIRT head to MNI152
    fnirt = pe.Node(fsl.FNIRT(ref_file=target_head,
                              refmask_file=target_mask,
                              config_file=fnirt_cfg,
                              fieldcoeff_file=True),
                    name="fnirt")

    # Warp the images
    warpbrain = pe.Node(fsl.ApplyWarp(ref_file=target_head,
                                      interp="spline"),
                        name="warpbrain")

    warphead = pe.Node(fsl.ApplyWarp(ref_file=target_head,
                                     interp="spline"),
                        name="warphead")

    # Generate a png summarizing the registration
    checkreg = pe.Node(CheckReg(),
                       name="checkreg")

    # Save relevant files to the data directory
    datasink = pe.Node(io.DataSink(base_directory=data_dir,
                                   parameterization = False,
                                   substitutions=[("norm_", "brain_"),
                                                  ("nu_", "T1_"),
                                                  ("_out", ""),
                                                  ("T1_fieldwarp", "warpfield"),
                                                  ("brain_flirt.mat", "affine.mat")]),
                       name="datasink")

    # Define and connect the workflow
    # -------------------------------

    normalize = pe.Workflow(name="normalize", 
                            base_dir=data_dir)

    normalize.connect([
        (subjectsource,   datasource,   [("subject_id", "subject_id")]),
        (datasource,      cvtbrain,     [("brain", "in_file")]),
        (datasource,      cvthead,      [("head", "in_file")]),
        (cvtbrain,        flirt,        [("out_file", "in_file")]),
        (flirt,           fnirt,        [("out_matrix_file", "affine_file")]),
        (cvthead,         fnirt,        [("out_file", "in_file")]),
        (cvtbrain,        warpbrain,    [("out_file", "in_file")]),
        (fnirt,           warpbrain,    [("fieldcoeff_file", "field_file")]),
        (cvthead,         warphead,     [("out_file", "in_file")]),
        (fnirt,           warphead,     [("fieldcoeff_file", "field_file")]),
        (warpbrain,       checkreg,     [("out_file", "in_file")]),
        (subjectsource,   datasink,     [("subject_id", "container")]),
        (cvtbrain,        datasink,     [("out_file", "normalization.@brain")]),
        (cvthead,         datasink,     [("out_file", "normalization.@t1")]),
        (flirt,           datasink,     [("out_file", "normalization.@brain_flirted")]),
        (flirt,           datasink,     [("out_matrix_file", "normalization.@affine")]),
        (warphead,        datasink,     [("out_file", "normalization.@t1_warped")]),
        (warpbrain,       datasink,     [("out_file", "normalization.@brain_warped")]),
        (fnirt,           datasink,     [("fieldcoeff_file", "normalization.@warpfield")]),
        (checkreg,        datasink,     [("out_file", "normalization.@reg_png")]),
        ])

    return normalize

