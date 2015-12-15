import os.path as op
import shutil
from glob import glob
import subprocess as sub

import numpy as np
from scipy import stats

import nibabel as nib
from nipype import IdentityInterface, Function, Node, MapNode, Workflow
from nipype.interfaces import freesurfer as fs
from nipype.interfaces.base import (BaseInterface,
                                    BaseInterfaceInputSpec, TraitedSpec,
                                    InputMultiPath, OutputMultiPath, File)

import lyman

imports = ["import os",
           "import os.path as op",
           "import shutil",
           "from glob import glob",
           "import numpy as np",
           "from scipy import stats",
           "import subprocess as sub"]


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
    surfsample = MapNode(fs.SampleToSurface(
        sampling_method=exp_info["sampling_method"],
        sampling_range=exp_info["sampling_range"],
        sampling_units=exp_info["sampling_units"],
        smooth_surf=exp_info["surf_smooth"],
        target_subject="fsaverage"),
        ["subject_id", "reg_file", "source_file"], "surfsample")

    # Remove subjects with completely empty images
    removeempty = Node(RemoveEmpty(), "removeempty")

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
    cluster = Node(Function(["y_file",
                             "glm_dir",
                             "sign",
                             "cluster_zthresh",
                             "p_thresh"],
                            ["glm_dir",
                             "thresholded_file"],
                            glm_corrections,
                            imports),
                   "cluster")
    cluster.inputs.cluster_zthresh = exp_info["cluster_zthresh"]
    cluster.inputs.p_thresh = exp_info["grf_pthresh"]
    cluster.inputs.sign = exp_info["surf_corr_sign"]

    # Return the outputs
    outputnode = Node(IdentityInterface(["glm_dir", "sig_file"]), "outputnode")

    # Define and connect the workflow
    group = Workflow(name)
    group.connect([
        (inputnode, surfsample,
            [("copes", "source_file"),
             ("reg_file", "reg_file"),
             ("subject_id", "subject_id")]),
        (hemisource, surfsample,
            [("hemi", "hemi")]),
        (surfsample, removeempty,
            [("out_file", "in_files")]),
        (removeempty, mergecope,
            [("out_files", "in_files")]),
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
        (cluster, outputnode,
            [("thresholded_file", "sig_file")]),
        ])

    return group, inputnode, outputnode


class RemoveEmptyInput(BaseInterfaceInputSpec):

    in_files = InputMultiPath(File(exists=True))


class RemoveEmptyOutput(TraitedSpec):

    out_files = OutputMultiPath(File(exists=True))


class RemoveEmpty(BaseInterface):

    input_spec = RemoveEmptyInput
    output_spec = RemoveEmptyOutput

    def _run_interface(self, runtime):

        good_images = [f for f in self.inputs.in_files
                       if not np.allclose(nib.load(f).get_data(), 0)]
        self.good_images = good_images
        return runtime

    def _list_outputs(self):

        outputs = self._outputs().get()
        outputs["out_files"] = self.good_images
        return outputs


def glm_corrections(y_file, glm_dir, sign, cluster_zthresh, p_thresh):
    """Use Freesurfer's cached simulations for monte-carlo correction."""
    # Convert from z to -log10(p)
    sig = -np.log10(stats.norm.sf(cluster_zthresh))

    shutil.copy(y_file, glm_dir)

    cmdline = ["mri_glmfit-sim",
               "--glmdir", glm_dir,
               "--cwpvalthresh", str(p_thresh),
               "--cache", str(sig), sign,
               "--2spaces"]
    sub.check_output(cmdline)

    thresholded_file = glob(op.join(glm_dir, "osgm/cache*masked.mgh"))[0]

    return glm_dir, thresholded_file
