"""
Estimate nonlinear normalization parameters from Freesurfer conformed space
to FSL's nonlinear MNI152 target using FLIRT and FNIRT.

See the docstring for the create_normalization_workflow function for more
information about the processing.

"""
import os

from nipype.interfaces import io
from nipype.interfaces import fsl
from nipype.interfaces import freesurfer as fs
from nipype.interfaces import utility as util
from nipype.pipeline import engine as pe


def create_normalization_workflow(data_dir, subjects, name="normalize"):
    """Set up the anatomical normalzation workflow.

    Your anatomical data must have been processed in Freesurfer.
    This normalization workflow does not need the entire recon-all
    pipeline to finish; it can be run after the canorm stage.



    """

    # Get target images
    target_brain = fsl.Info.standard_image("avg152T1_brain.nii.gz")
    target_head = fsl.Info.standard_image("avg152T1.nii.gz")
    hires_head = fsl.Info.standard_image("MNI152_T1_1mm.nii.gz")
    target_mask = fsl.Info.standard_image(
        "MNI152_T1_2mm_brain_mask_dil.nii.gz")
    fnirt_cfg = os.path.join(
        os.environ["FSLDIR"], "etc/flirtsch/T1_2_MNI152_2mm.cnf")

    # Subject source node
    subjectsource = pe.Node(util.IdentityInterface(fields=["subject_id"]),
                            iterables=("subject_id", subjects),
                            name="subjectsource")

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
    for dim in ["x", "y", "z"]:
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
    
    warpbrainhr = pe.Node(fsl.ApplyWarp(ref_file=hires_head,
                                        interp="spline"),
                             name="warpbrainhr")

    warpheadhr = pe.Node(fsl.ApplyWarp(ref_file=hires_head,
                                       interp="spline"),
                         name="warpheadhr")

    namehrbrain = pe.Node(util.Rename(format_string="brain_warp_hires",
                                      keep_ext=True),
                          name="namehrbrain")
    
    namehrhead = pe.Node(util.Rename(format_string="T1_warp_hires",
                                     keep_ext=True),
                         name="namehrhead")

    # Generate a png summarizing the registration
    checkreg = pe.Node(util.Function(input_names=["in_file"],
                                     output_names=["out_file"],
                                     function=mni_reg_qc),
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
        (cvtbrain,        warpbrainhr,  [("out_file", "in_file")]),
        (fnirt,           warpbrainhr,  [("fieldcoeff_file", "field_file")]),
        (cvthead,         warpheadhr,   [("out_file", "in_file")]),
        (fnirt,           warpheadhr,   [("fieldcoeff_file", "field_file")]),
        (warpbrain,       checkreg,     [("out_file", "in_file")]),
        (warpheadhr,      namehrhead,   [("out_file", "in_file")]),
        (warpbrainhr,     namehrbrain,  [("out_file", "in_file")]),
        (subjectsource,   datasink,     [("subject_id", "container")]),
        (cvtbrain,        datasink,     [("out_file", "normalization.@brain")]),
        (cvthead,         datasink,     [("out_file", "normalization.@t1")]),
        (flirt,           datasink,     [("out_file", "normalization.@brain_flirted")]),
        (flirt,           datasink,     [("out_matrix_file", "normalization.@affine")]),
        (warphead,        datasink,     [("out_file", "normalization.@t1_warped")]),
        (warpbrain,       datasink,     [("out_file", "normalization.@brain_warped")]),
        (namehrhead,      datasink,     [("out_file", "normalization.@head_hires")]),
        (namehrbrain,     datasink,     [("out_file", "normalization.@brain_hires")]),
        (fnirt,           datasink,     [("fieldcoeff_file", "normalization.@warpfield")]),
        (checkreg,        datasink,     [("out_file", "normalization.@reg_png")]),
        ])

    return normalize


def mni_reg_qc(in_file):
    """Create a png image summarizing a normalization to the MNI152 brain."""
    import os.path as op
    from subprocess import call
    from nipype.interfaces import fsl

    mni_targ = fsl.Info.standard_image("avg152T1_brain.nii.gz")

    planes = ["x", "y", "z"]
    options = []
    for plane in planes:
        for slice in ["%.2f"%i for i in .15,.3,.45,.5,.55,.7,.85]:
            if not(plane == "x" and slice == "0.50"):
                options.append((plane,slice))

    shots = ["%s-%s.png" % i for i in options]

    for i, shot in enumerate(shots):
        cmd = ["slicer", 
               in_file,
               mni_targ,
               "-s 1.5",
               "-%s"%options[i][0],
               options[i][1],
               shot]

        call(cmd)

    for i in range(3):
        cmd = ["pngappend"]
        cmd.append(" + ".join([s for s in shots if op.split(s)[1].startswith(planes[i])]))
        rowimg = "row-%d.png" % i
        cmd.append(rowimg)
        shots.append(rowimg)
        call(" ".join(cmd), shell=True)

    cmd = ["pngappend"]
    cmd.append(" - ".join(["row-%d.png" % i for i in range(3)]))
    out_file = op.join(op.abspath("."), "brain_warp_to_mni.png")
    cmd.append(out_file)
    call(" ".join(cmd), shell=True)

    return out_file
