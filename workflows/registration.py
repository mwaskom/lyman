from nipype.interfaces import fsl
from nipype.interfaces import freesurfer as surf
from nipype.interfaces.utility import Function, IdentityInterface
from nipype.pipeline.engine import Workflow, Node, MapNode


def create_mni_reg_workflow(name="mni_reg"):

    inputnode = Node(IdentityInterface(fields=["source_image",
                                               "warpfield",
                                               "fsl_affine"]),
                     name="inputnode")

    target = fsl.Info.standard_image("avg152T1_brain.nii.gz")

    applywarp = MapNode(fsl.ApplyWarp(ref_file=target,
                                      interp="spline"),
                        iterfield=["in_file", "premat"],
                        name="applywarp")

    outputnode = Node(IdentityInterface(fields=["out_file"]),
                      name="outputnode")

    warpflow = Workflow(name=name)
    warpflow.connect([
        (inputnode, applywarp,
            [("source_image", "in_file"),
             ("warpfield", "field_file"),
             ("fsl_affine", "premat")]),
        (applywarp, outputnode,
            [("out_file", "out_file")])
        ])

    return warpflow, inputnode, outputnode


def create_epi_reg_workflow(name="epi_reg"):

    inputnode = Node(IdentityInterface(fields=["source_image",
                                               "fsl_affine"]),
                     name="inputnode")

    applyreg = Node(Function(input_names=["source_images",
                                          "fsl_affines"],
                             output_names=["out_files"],
                             function=register_to_epi),
                    name="applyreg")

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


def create_cortex_reg_workflow():

    pass


def create_fsaverage_reg_workflow():

    pass


def register_to_epi(source_images, fsl_affines):
    """Register a list of runs into the space given by the first."""
    # Pop the target image and affine
    from os import mkdir, symlink, remove
    from os.path import abspath, exists
    from subprocess import call
    from nipype.utils.filemanip import fname_presuffix
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

    # Now apply this transformation to each of the non-target images
    out_files = []
    for i, img in enumerate(source_images[1:]):
        out_img = fname_presuffix(img,
                                  suffix="_xfm",
                                  newpath=abspath("./_run_%d" % (i + 2)))
        try:
            mkdir("_run_%d" % (i + 2))
        except OSError:
            pass
        out_files.append(out_img)
        call(["applywarp",
              "-i", img,
              "-r", target_img,
              "-o", out_img,
              "--interp=spline",
              "--premat=%s" % xfm_mats[i]])

    # Now link the target image so it is named appropriately
    out_img = fname_presuffix(target_img, suffix="_xfm",
                              newpath=abspath("./_run_1"))
    if exists(out_img):
        remove(out_img)
    out_files.insert(0, out_img)
    mkdir(abspath("./_run_1"))
    symlink(target_img, out_img)

    return out_files
