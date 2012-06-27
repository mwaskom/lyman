"""Registration module contains workflows to move all runs into a common space.

There are four possible spaces:
- epi: linear transform of several runs into the first run's space
- mni: nonlinear warp to FSL's MNI152 space
- cortex: transformation to native cortical surface representation
- fsaverage: transformation into Freesurfer's common surface space

"""
from nipype.interfaces import fsl
from nipype.interfaces import freesurfer as surf
from nipype.interfaces.utility import Function, IdentityInterface
from nipype.pipeline.engine import Workflow, Node, MapNode

spaces = ["epi", "mni"]

def create_epi_reg_workflow(name="epi_reg", interp="spline"):
    """Set up a workflow to register several runs into the first run space."""
    inputnode = Node(IdentityInterface(fields=["source_image",
                                               "fsl_affine"]),
                     name="inputnode")

    # Register runs 2+ into the space of the first
    # This combines the transfrom from each run to the anatomical
    # with  the inverse of the transform from run 1 to the anatomical
    applyreg = Node(Function(input_names=["source_images",
                                          "fsl_affines",
                                          "interp"],
                             output_names=["out_files"],
                             function=register_to_epi),
                    name="applyreg")
    applyreg.inputs.interp = interp

    outputnode = Node(IdentityInterface(fields=["out_file"]),
                      name="outputnode")

    xfmflow = Workflow(name=name)
    xfmflow.connect([
        (inputnode, applyreg,
            [("source_image", "source_images"),
             ("fsl_affine", "fsl_affines")]),
        (applyreg, outputnode,
            [("out_files", "out_file")])
        ])

    return xfmflow, inputnode, outputnode


def create_mni_reg_workflow(name="mni_reg", interp="spline"):
    """Set up a workflow to register an epi into FSL's MNI space."""
    inputnode = Node(IdentityInterface(fields=["source_image",
                                               "warpfield",
                                               "fsl_affine"]),
                     name="inputnode")

    target = fsl.Info.standard_image("avg152T1_brain.nii.gz")

    getinterp = MapNode(Function(input_names=["source_file",
                                              "default_interp"],
                                 output_names="interp",
                                 function=get_interp),
                        iterfield=["source_file"],
                        name="getinterp")
    getinterp.inputs.default_interp=interp

    applywarp = MapNode(fsl.ApplyWarp(ref_file=target),
                        iterfield=["in_file", "premat", "interp"],
                        name="applywarp")

    outputnode = Node(IdentityInterface(fields=["out_file"]),
                      name="outputnode")

    warpflow = Workflow(name=name)
    warpflow.connect([
        (inputnode, applywarp,
            [("source_image", "in_file"),
             ("warpfield", "field_file"),
             ("fsl_affine", "premat")]),
        (inputnode, getinterp,
            [("source_image", "source_file")]),
        (getinterp, applywarp,
            [("interp", "interp")]),
        (applywarp, outputnode,
            [("out_file", "out_file")])
        ])

    return warpflow, inputnode, outputnode


def create_cortex_reg_workflow():

    pass


def create_fsaverage_reg_workflow():

    pass


def get_interp(source_file, default_interp="spline"):
    """Determine what interpolation to use for volume resampling."""
    from nibabel import load
    img_data = load(source_file).get_data()
    is_bool = img_data.sum() == img_data.astype(bool).sum()
    interp = "nn" if is_bool else default_interp
    return interp


def register_to_epi(source_images, fsl_affines, interp="spline"):
    """Register a list of runs into the space given by the first."""
    from os import mkdir, symlink, remove
    from os.path import abspath, exists
    from subprocess import call
    from nibabel import load
    from nipype.utils.filemanip import fname_presuffix
    # Pop the target image and affine
    target_img = source_images[0]
    target_aff = fsl_affines[0]

    # Invert the transform from target space to anatomical
    target_aff_inv = fname_presuffix(target_aff, suffix="_inv")
    call(["convert_xfm", "-omat", target_aff_inv, "-inverse", target_aff])

    # For each of the other runs, combine their transform with the inverse
    xfm_mats = []
    for run, mat in enumerate(fsl_affines[1:], 2):
        out_xfm = fname_presuffix(mat, suffix="_%d_to_1" % run)
        xfm_mats.append(out_xfm)
        call(["convert_xfm", "-omat", out_xfm, "-concat", target_aff_inv, mat])

    # Apply this transformation to each of the non-target images
    out_files = []
    for i, img in enumerate(source_images[1:]):
        out_img = fname_presuffix(img, suffix="_xfm",
                                  newpath=abspath("./_run_%d" % (i + 2)))
        try:
            mkdir("_run_%d" % (i + 2))
        except OSError:
            pass
        out_files.append(out_img)

        # Check if we got a mask, in which case use nearest neighbor interp
        img_data = load(img).get_data()
        is_bool = img_data.sum() == img_data.astype(bool).sum()
        if is_bool:
            interp = "nn"

        # Execute the transformation using FSL's applywarp
        call(["applywarp",
              "-i", img,
              "-r", target_img,
              "-o", out_img,
              "--interp=%s" % interp,
              "--premat=%s" % xfm_mats[i]])

    # :ink the target image so it is named appropriately
    out_img = fname_presuffix(target_img, suffix="_xfm",
                              newpath=abspath("./_run_1"))
    if exists(out_img):
        remove(out_img)
    try:
        mkdir(abspath("./_run_1"))
    except OSError:
        pass
    out_files.insert(0, out_img)
    symlink(target_img, out_img)

    return out_files
