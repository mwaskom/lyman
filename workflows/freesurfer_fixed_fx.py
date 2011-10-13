import nipype.interfaces.freesurfer as fs
import nipype.interfaces.utility as util
import nipype.pipeline.engine as pe
#from .interfaces import MayaviShots

def get_freesurfer_fixed_fx_workflow(name="fixed_fx"):

    # Define the workflow
    fixed_fx = pe.Workflow(name=name)

    # Set up the inputs
    inputnode = pe.Node(util.IdentityInterface(fields=["hemi",
                                                       "cope", 
                                                       "varcope",
                                                       "dof_file",
                                                       "subject_id"]),
                        name="inputspec")

    # Concatenate the copes and varcopes for each run
    copemerge = pe.Node(fs.Concatenate(),
                        name="copemerge")
    
    varcopemerge = pe.Node(fs.Concatenate(),
                           name="varcopemerge")

    # Use the function defined below to figure out the fixed_effects variance
    getdof = pe.Node(util.Function(input_names=["doffiles"],output_names=["dof"],
                                   function=get_dof_func),
                     name="getdof")

    # Run a fixed-effects analysis using mri_glmfit
    glmfit = pe.Node(fs.GLMFit(one_sample=True,
                               surf=True,
                               cortex=True,
                               glm_dir="stats"),
                     name="glmfit")

    # Stats screenshots
#    snapshots = pe.Node(MayaviShots(),
#                        name="snapshots")

    outputnode = pe.Node(util.IdentityInterface(fields=["stats"]),#,"snapshots"]),
                         name="outputspec")

    fixed_fx.connect([
        (inputnode,    copemerge,    [("cope", "in_files")]),
        (inputnode,    varcopemerge, [("varcope", "in_files")]),
        (inputnode,    getdof,       [("dof_file", "doffiles")]),
        (copemerge,    glmfit,       [("concatenated_file", "in_file")]),
        (varcopemerge, glmfit,       [("concatenated_file", "fixed_fx_var")]),
        (getdof,       glmfit,       [("dof", "fixed_fx_dof")]),
        (inputnode,    glmfit,       [("hemi", "hemi"),
                                      ("subject_id", "subject_id")]),
#        (glmfit,       snapshots,    [("sig_file", "in_file")]),
#        (inputnode,    snapshots,    [("hemi", "hemi")]),
        (glmfit,       outputnode,   [("glm_dir", "stats")]),
#        (snapshots,    outputnode,   [("snapshots", "snapshots")]),
        ])

    return fixed_fx, inputnode, outputnode    

def get_dof_func(doffiles):

    if isinstance(doffiles, list):
        dof = sum([int(open(f).read().strip()) for f in doffiles])
    else:
        dof = int(open(doffiles).read().strip())
    return dof
