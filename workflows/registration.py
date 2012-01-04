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


def create_epi_reg_workflow():

    pass


def create_cortex_reg_workflow():

    pass


def create_fsaverage_reg_workflow():

    pass
