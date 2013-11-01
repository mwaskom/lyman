import os
import os.path as op
import shutil
from glob import glob

import numpy as np
import matplotlib as mpl
from scipy import stats
import nibabel as nib

from nipype import IdentityInterface, Function, Node, MapNode, Workflow
from nipype import freesurfer as fs

import lyman

imports = ["import os",
           "import os.path as op",
           "import shutil",
           "from glob import glob",
           "import numpy as np",
           "import matplotlib as mpl",
           "from scipy import stats",
           "import nibabel as nib",
           "import subprocess as sub",
           "import seaborn"]


def create_surface_ols_workflow(name="surface_group",
                                subject_list=None,
                                exp_info=None,
                                surfviz=True):
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

    # Plot the results on the surface
    surfplot = Node(Function(["mask_file",
                              "sig_file",
                              "hemi",
                              "sign",
                              "cluster_zthresh",
                              "surf_name"],
                             ["surf_png"],
                             plot_surface_viz,
                             imports),
                    "surfplot")
    surfplot.inputs.cluster_zthresh = exp_info["cluster_zthresh"]
    surfplot.inputs.sign = exp_info["surf_corr_sign"]
    surfplot.inputs.surf_name = exp_info["surf_name"]

    # Return the outputs
    outputnode = Node(IdentityInterface(["glm_dir", "surf_png"]), "outputnode")

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

    # Optionally connect the surface visualization
    if surfviz:
        group.connect([
            (glmfit, surfplot,
                [("mask_file", "mask_file")]),
            (cluster, surfplot,
                [("thresholded_file", "sig_file")]),
            (hemisource, surfplot,
                [("hemi", "hemi")]),
            (surfplot, outputnode,
                [("surf_png", "surf_png")]),
            ])

    return group, inputnode, outputnode


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


def plot_surface_viz(mask_file, sig_file, hemi, sign,
                     cluster_zthresh, surf_name):
    """Use PySurfer to plot the inferential results."""

    # Delay the import so the workflow can run without Pysurfer installed
    from surfer import Brain
    from mayavi import mlab
    from traits.trait_errors import TraitError

    # Load the visualization
    b = Brain("fsaverage", hemi, surf_name,
              config_opts=dict(background="white",
                               width=800, height=500))

    # Read the mask file and flip the boolean
    mask_data = np.array(nib.load(mask_file).get_data().squeeze())
    mask_data = np.logical_not(mask_data).astype(np.float)

    # Plot the masked-out vertices
    b.add_data(mask_data, min=0, max=10, thresh=.5,
               colormap="bone", alpha=.6, colorbar=False)

    # Read the sig (-log10(p)) data and convert to z stats
    sig_data = np.array(nib.load(sig_file).get_data().squeeze())
    sig_sign = np.sign(sig_data)
    p_data = 10 ** -np.abs(sig_data)
    z_data = stats.norm.ppf(p_data)
    z_data[np.sign(z_data) != sig_sign] *= -1

    try:
        b.add_overlay(z_data, cluster_zthresh, sign=sign, name="zstat")
    except TraitError:
        print "No vertices above the threshold."

    # Save the plots
    view_temp = "zstat_threshold_%s.png"
    views = ["lat", "med", "ven"]
    for view in views:

        if view == "ven":
            b.overlays["zstat"].pos_bar.visible = True
        else:
            b.overlays["zstat"].pos_bar.visible = False

        b.show_view(view, distance=330)

        b.save_image(view_temp % view)

    frames = [mpl.image.imread(view_temp % v)) for v in views]
    full_img = np.concatenate(frames, axis=0)

    mpl.image.imsave("zstat_threshold.png", full_img)
    return op.abspath("zstat_threshold.png")
