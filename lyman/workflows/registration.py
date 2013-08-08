"""Registration module contains workflows to move all runs into a common space.

There are two possible spaces:
- epi: linear transform of several runs into the first run's space
- mni: nonlinear warp to FSL's MNI152 space

"""
import os
import os.path as op
import shutil
import subprocess as sub
from glob import glob
import numpy as np

from nipype import fsl, Workflow, Node, MapNode, Function, IdentityInterface

imports = ["import os",
           "import os.path as op",
           "import shutil",
           "import subprocess as sub",
           "from glob import glob",
           "import numpy as np",
           ]

spaces = ["epi", "mni"]


def create_epi_reg_workflow(name="epi_reg", regtype="model"):
    """Register model outputs into the first run's native space."""

    if regtype == "model":
        fields = ["copes", "varcopes", "masks", "affines"]
    elif regtype == "timeseries":
        fields = ["timeseries", "masks", "affines"]
    else:
        raise ValueError("regtype must be in {'model', 'timeseries'}")

    inputnode = Node(IdentityInterface(fields), "inputnode")

    func = dict(model=epi_model_transform,
                timeseries=epi_timeseries_transform)[regtype]

    transform = Node(Function(fields, ["out_files"],
                              func, imports),
                     "transform")

    regflow = Workflow(name=name)

    outputnode = Node(IdentityInterface(["out_files"]), "outputnode")
    for field in fields:
        regflow.connect(inputnode, field, transform, field)
    regflow.connect(transform, "out_files", outputnode, "out_files")

    return regflow, inputnode, outputnode


def create_mni_reg_workflow(name="mni_reg", regtype="model"):
    """Set up a workflow to register files into FSL's MNI space."""
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
    getinterp.inputs.default_interp = interp

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


# Interface functions
# ===================

def epi_model_transform(copes, varcopes, masks, affines):
    """Take model outputs into the 'epi' space in a workflow context."""
    n_runs = len(affines)

    ref_file = copes[0]
    copes = map(list, np.split(np.array(copes), n_runs))
    varcopes = map(list, np.split(np.array(varcopes), n_runs))

    # Iterate through the runs
    for n in range(n_runs):

        # Make the output directories
        out_dir = "run_%d" % (n + 1)
        os.mkdir(out_dir)

        run_copes = copes[n]
        run_varcopes = varcopes[n]
        run_mask = masks[n]
        run_affine = affines[n]

        files = [run_mask] + run_copes + run_varcopes

        if not n:
            # Just copy the first run files over
            for f in files:
                to = op.join(out_dir, op.basename(f).replace(".nii.gz",
                                                             "_xfm.nii.gz"))
                shutil.copyfile(f, to)

        else:
            # Otherwise apply the transformation
            inv_affine = op.basename(run_affine).replace(".mat", "_inv.mat")
            sub.check_output(["convert_xfm", "-omat", inv_affine,
                              "-inverse", run_affine])

            interps = ["nn"] + (["trilinear"] * (len(files) - 1))
            for f, interp in zip(files, interps):
                out_file = op.join(out_dir,
                                   op.basename(f).replace(".nii.gz",
                                                          "_xfm.nii.gz"))
                cmd = ["applywarp", "-i", f, "-r", ref_file, "-o", out_file,
                       "--interp=%s" % interp, "--premat=%s" % inv_affine]
                sub.check_output(cmd)

    out_files = [op.abspath(f) for f in glob("run_*/*.nii.gz")]
    return out_files


def epi_timeseries_transform(timeseries, masks, affines):
    """Take a set of timeseries files into the 'epi' space."""
    n_runs = len(affines)
    ref_file = timeseries[0]

    for n in range(n_runs):

        # Make the output directory
        out_dir = "run_%d" % (n + 1)
        os.mkdir(out_dir)

        run_timeseries = timeseries[n]
        run_mask = masks[n]
        run_affine = affines[n]

        if not n:
            # Just copy the first run files over
            to_timeseries = op.join(out_dir, "timeseries_xfm.nii.gz")
            shutil.copyfile(run_timeseries, to_timeseries)
            to_mask = op.join(out_dir, "functional_mask_xfm.nii.gz")
            shutil.copyfile(run_mask, to_mask)

        else:

            # Otherwise apply the transformation
            inv_affine = op.basename(run_affine).replace(".mat", "_inv.mat")
            sub.check_output(["convert_xfm", "-omat", inv_affine,
                              "-inverse", run_affine])

            files = [run_mask, run_timeseries]
            interps = ["nn", "spline"]
            for f, interp in zip(files, interps):
                out_file = op.join(out_dir,
                                   op.basename(f).replace(".nii.gz",
                                                          "_xfm.nii.gz"))
                cmd = ["applywarp", "-i", f, "-r", ref_file, "-o", out_file,
                       "--interp=%s" % interp, "--premat=%s" % inv_affine]
                sub.check_output(cmd)

    out_files = [op.abspath(f) for f in glob("run_*/*.nii.gz")]
    return out_files
