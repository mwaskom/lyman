import nipype.interfaces.fsl as fsl
import nipype.interfaces.utility as util
import nipype.pipeline.engine as pe

from .utility import OutputConnector


def get_fsl_fixed_fx_workflow(name="fixed_fx", volume_report=True):

    # Define the workflow
    fixed_fx = pe.Workflow(name=name)

    # Set up the inputs
    inputnode = pe.Node(util.IdentityInterface(fields=["cope", 
                                                       "varcope",
                                                       "mask",
                                                       "dof_file"]),
                        name="inputspec")

    # Concatenate the Cope for each run
    copemerge = pe.Node(fsl.Merge(dimension="t"),
                        name="copemerge")

    # Concatenate the Varcope for each run
    varcopemerge = pe.Node(fsl.Merge(dimension="t"),
                           name="varcopemerge")
    
    # Get an image of the DOFs
    getdof = pe.MapNode(fsl.ImageMaths(suffix="_dof"),
                        iterfield=["in_file", "op_string"],
                        name="getdof")
    
    dofmerge = pe.Node(fsl.Merge(dimension="t"),
                       name="dofmerge")

    # Set up a FLAMEO model
    level2model = pe.Node(fsl.L2Model(),
                          name="l2model")

    # Run a fixed effects analysis in FLAMEO
    flameo = pe.Node(fsl.FLAMEO(run_mode="fe"),
                     name="flameo")

    if volume_report:
        # Display on the FSL template brain
        mni_brain = fsl.Info.standard_image("avg152T1_brain.nii.gz")

        # Overlay the stats onto a background image
        overlayflame = pe.Node(fsl.Overlay(stat_thresh=(2.3, 10),
                                                        auto_thresh_bg=True,
                                                        show_negative_stats=True,
                                                        background_image=mni_brain),
                                  name="overlayflame")

        # Slice the overlaid statistical images
        sliceflame = pe.Node(fsl.Slicer(image_width=872),
                                name="sliceflame")
        sliceflame.inputs.sample_axial = 2

    # Outputs
    outfields = ["stats"]
    if volume_report:
        outfields.append("zstat")
    outputnode = pe.Node(util.IdentityInterface(fields=outfields),
                         name="outputspec")


    fixed_fx.connect([
        (inputnode,    copemerge,     [("cope", "in_files")]),
        (inputnode,    varcopemerge,  [("varcope", "in_files")]),
        (inputnode,    getdof,        [("cope", "in_file")]),
        (inputnode,    getdof,        [("mask", "in_file2")]),
        (inputnode,    getdof,        [(("dof_file", get_dof_opstring), "op_string" )]),
        (getdof,       dofmerge,      [("out_file", "in_files")]),
        (copemerge,    flameo,        [("merged_file","cope_file")]),
        (varcopemerge, flameo,        [("merged_file","var_cope_file")]),
        (dofmerge,     flameo,        [("merged_file", "dof_var_cope_file")]),
        (inputnode,    flameo,        [("mask", "mask_file")]),
        (inputnode,    level2model,   [(("cope", get_length), "num_copes")]),
        (level2model,  flameo,        [("design_mat","design_file"),
                                       ("design_con","t_con_file"),
                                       ("design_grp","cov_split_file")]),
        (flameo,       outputnode,    [("stats_dir", "stats")]),
        ])
    
    if volume_report:
        fixed_fx.connect([
            (flameo,       overlayflame,  [("zstats","stat_image")]),
            (overlayflame, sliceflame,    [("out_file", "in_file")]),
            ])
    
        # Use a utility class (defined in utility module) to control renaming 
        # and connections to the output node
        rename = OutputConnector(fixed_fx, outputnode)
        rename.connect(sliceflame, "zstat")
        
    return fixed_fx, inputnode, outputnode
    

def get_dof_opstring(doffiles):
    """Generate an fslmaths opstring to fill a DOF volume."""
    opstrings = []
    for doffile in doffiles:
        with open(doffile) as f:
            dof = f.read().strip()
            opstrings.append("-mul 0 -add %s -mas"%dof)
    return opstrings

def get_length(x):
    return len(x)
