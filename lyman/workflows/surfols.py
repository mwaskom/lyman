import os
import os.path as op
import shutil

import numpy as np
from scipy import stats

from nipype import IdentityInterface, Function, Node, MapNode, Workflow
from nipype import freesurfer as fs

import lyman

imports = ["import os",
           "import os.path as op",
           "import shutil",
           "import numpy as np",
           "from scipy import stats",
           "import subprocess as sub",
           "import seaborn"]


def create_surface_ols_workflow(name="surface_group",
                                subject_list=None,
                                exp_info=None):
    """Workflow to project ffx copes onto surface and run ols."""
    if subject_list is None:
        subject_list = []
    if exp_info is None:
        exp_info = lyman.default_experiment_parameters()

    inputnode = Node(IdentityInterface(["l1_contrast",
                                        "copes",
                                        "reg_file",
                                        "subject_id"]),
                     "inputnode")

    hemisource = Node(IdentityInterface(["hemi"]), "hemisource")
    hemisource.iterables = ("hemi", ["lh", "rh"])

    # Sample the volume-encoded native data onto the fsaverage surface
    # manifold with projection + spherical transform
    # TODO probably expose the sampling to an experiment configurable
    # Also expose surface smoothing
    surfsample = MapNode(fs.SampleToSurface(sampling_method="average",
                                            sampling_range=(0, 1, .1),
                                            sampling_units="frac",
                                            target_subject="fsaverage",
                                            smooth_surf=5),
                          ["subject_id", "reg_file", "source_file"],
                          "surfsample")

    # Concatenate the subject files into a 4D image
    mergecope = Node(fs.Concatenate(), "mergecope")

    # Run the one-sample OLS model
    glmfit = Node(fs.GLMFit(one_sample=True,
                            surf=True,
                            cortex=True,
                            glm_dir="_glm_results",
                            subject_id="fsaverage"),
                  "glmfit")

    # Use the cached Monte-Carlo simulations for correction
    cluster = Node(Function(["y_file", "glm_dir",
                             "cluster_zthresh", "p_thresh"],
                            ["glm_dir"],
                            glm_corrections,
                            imports),
                   "cluster")
    cluster.inputs.cluster_zthresh = exp_info["cluster_zthresh"]
    cluster.inputs.p_thresh = exp_info["grf_pthresh"]

    # Return the outputs
    outputnode = Node(IdentityInterface(["glm_dir"]), "outputnode")

    # Define and connect the workflow
    group = Workflow(name)
    group.connect([
        (inputnode, surfsample,
            [("copes", "source_file"),
             ("reg_file", "reg_file"),
             ("subject_id", "subject_id")]),
        (hemisource, surfsample,
            [("hemi", "hemi")]),
        (surfsample, mergecope,
            [("out_file", "in_files")]),
        (mergecope, glmfit,
            [("concatenated_file", "in_file")]),
        (hemisource, glmfit,
            [("hemi", "hemi")]),
        (mergecope, cluster,
            [("concatenated_file", "y_file")]),
        (glmfit, cluster,
            [("glm_dir", "glm_dir")]),
        (glmfit, outputnode,
            [("glm_dir", "glm_dir")]),
                    ])

    return group, inputnode, outputnode


def glm_corrections(y_file, glm_dir, cluster_zthresh, p_thresh):
    """Use Freesurfer's cached simulations for monte-carlo correction."""
    # Convert from z to -log10(p)
    sig = -np.log10(stats.norm.sf(cluster_zthresh))

    shutil.copy(y_file, glm_dir)

    cmdline = ["mri_glmfit-sim",
               "--glmdir", glm_dir,
               "--cwpvalthresh", str(p_thresh),
               "--cache", str(sig), "pos",
               "--2spaces"]
    sub.check_output(cmdline)

    return glm_dir
