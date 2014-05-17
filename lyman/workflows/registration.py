"""Registration module contains workflows to move all runs into a common space.

There are two possible spaces:
- epi: linear transform of several runs into the first run's space
- mni: nonlinear warp to FSL's MNI152 space

"""
import os
import os.path as op
import shutil
import subprocess as sub
import numpy as np

from nipype import Workflow, Node, Function, IdentityInterface
from nipype.interfaces import fsl

imports = ["import os",
           "import os.path as op",
           "import shutil",
           "import subprocess as sub",
           "import numpy as np",
           "from nipype.interfaces import fsl",
           ]

spaces = ["epi", "mni"]


def create_reg_workflow(name="reg", space="mni", regtype="model"):
    """Flexibly register files into one of several common spaces."""
    if regtype == "model":
        fields = ["copes", "varcopes", "ss_files"]
    elif regtype == "timeseries":
        fields = ["timeseries"]
    fields.extend(["masks", "affines"])

    if space == "mni":
        fields.append("warpfield")

    inputnode = Node(IdentityInterface(fields), "inputnode")

    func = globals()["%s_%s_transform" % (space, regtype)]

    transform = Node(Function(fields, ["out_files"],
                              func, imports),
                     "transform")

    regflow = Workflow(name=name)

    outputnode = Node(IdentityInterface(["out_files"]), "outputnode")
    for field in fields:
        regflow.connect(inputnode, field, transform, field)
    regflow.connect(transform, "out_files", outputnode, "out_files")

    return regflow, inputnode, outputnode


# Interface functions
# ===================

def epi_model_transform(copes, varcopes, ss_files, masks, affines):
    """Take model outputs into the 'epi' space in a workflow context."""
    n_runs = len(affines)

    ref_file = copes[0]
    copes = map(list, np.split(np.array(copes), n_runs))
    varcopes = map(list, np.split(np.array(varcopes), n_runs))
    ss_files = map(list, np.split(np.array(ss_files), n_runs))

    # Invert the first run's affine
    run_1_inv_mat = op.basename(affines[0]).replace(".mat", "_inv.mat")
    sub.check_output(["convert_xfm", "-omat", run_1_inv_mat,
                      "-inverse", affines[0]])

    # Iterate through the runs
    for n in range(n_runs):

        # Make the output directories
        out_dir = "run_%d" % (n + 1)
        os.mkdir(out_dir)

        run_copes = copes[n]
        run_varcopes = varcopes[n]
        run_ss_files = ss_files[n]
        run_mask = masks[n]
        run_affine = affines[n]

        files = [run_mask] + run_copes + run_varcopes + run_ss_files

        if not n:
            # Just copy the first run files over
            for f in files:
                to = op.join(out_dir, op.basename(f).replace(".nii.gz",
                                                             "_xfm.nii.gz"))
                shutil.copyfile(f, to)

        else:
            # Otherwise apply the transformation
            full_affine = op.basename(run_affine).replace(".mat", "_to_epi.mat")
            sub.check_output(["convert_xfm", "-omat", full_affine,
                              "-concat", run_1_inv_mat, run_affine])

            interps = ["nn"] + (["trilinear"] * (len(files) - 1))
            for f, interp in zip(files, interps):
                out_file = op.join(out_dir,
                                   op.basename(f).replace(".nii.gz",
                                                          "_xfm.nii.gz"))
                cmd = ["applywarp", "-i", f, "-r", ref_file, "-o", out_file,
                       "--interp=%s" % interp, "--premat=%s" % full_affine]
                sub.check_output(cmd)

    out_files = [op.abspath("run_%d/" % (i + 1)) for i in range(n_runs)]
    return out_files


def epi_timeseries_transform(timeseries, masks, affines):
    """Take a set of timeseries files into the 'epi' space."""
    n_runs = len(affines)
    ref_file = timeseries[0]

    # Invert the first run's affine
    run_1_inv_mat = op.basename(affines[0]).replace(".mat", "_inv.mat")
    sub.check_output(["convert_xfm", "-omat", run_1_inv_mat,
                      "-inverse", affines[0]])

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
            full_affine = op.basename(run_affine).replace(".mat", "_to_epi.mat")
            sub.check_output(["convert_xfm", "-omat", full_affine,
                              "-concat", run_1_inv_mat, run_affine])

            files = [run_mask, run_timeseries]
            interps = ["nn", "spline"]
            for f, interp in zip(files, interps):
                out_file = op.join(out_dir, "timeseries_xfm.nii.gz")
                cmd = ["applywarp", "-i", f, "-r", ref_file, "-o", out_file,
                       "--interp=%s" % interp, "--premat=%s" % full_affine]
                sub.check_output(cmd)

    out_files = [op.abspath("run_%d/" % (i + 1)) for i in range(n_runs)]
    return out_files


def mni_model_transform(copes, varcopes, ss_files, masks, affines, warpfield):
    """Take model outputs into the FSL MNI space."""
    n_runs = len(affines)

    ref_file = fsl.Info.standard_image("avg152T1_brain.nii.gz")
    copes = map(list, np.split(np.array(copes), n_runs))
    varcopes = map(list, np.split(np.array(varcopes), n_runs))
    ss_files = map(list, np.split(np.array(ss_files), n_runs))

    # Iterate through the runs
    for n in range(n_runs):

        # Make the output directories
        out_dir = "run_%d" % (n + 1)
        os.mkdir(out_dir)

        run_copes = copes[n]
        run_varcopes = varcopes[n]
        run_ss_files = ss_files[n]
        run_mask = masks[n]
        run_affine = affines[n]

        files = [run_mask] + run_copes + run_varcopes + run_ss_files

        # Otherwise apply the transformation
        interps = ["nn"] + (["trilinear"] * (len(files) - 1))
        for f, interp in zip(files, interps):
            out_file = op.join(out_dir,
                               op.basename(f).replace(".nii.gz",
                                                      "_warp.nii.gz"))
            cmd = ["applywarp", "-i", f, "-r", ref_file, "-o", out_file,
                   "--interp=%s" % interp, "--premat=%s" % run_affine,
                   "-w", warpfield]
            sub.check_output(cmd)

    out_files = [op.abspath("run_%d/" % (i + 1)) for i in range(n_runs)]
    return out_files


def mni_timeseries_transform(timeseries, masks, affines, warpfield):
    """Take timeseries files into the FSL MNI space."""
    n_runs = len(affines)
    ref_file = fsl.Info.standard_image("avg152T1_brain.nii.gz")

    # Iterate through the runs
    for n in range(n_runs):

        # Make the output directory
        out_dir = "run_%d" % (n + 1)
        os.mkdir(out_dir)

        run_timeseries = timeseries[n]
        run_mask = masks[n]
        run_affine = affines[n]

        files = [run_mask, run_timeseries]
        interps = ["nn", "spline"]

        for f, interp in zip(files, interps):
            out_file = op.join(out_dir, "timeseries_warp.nii.gz")
            cmd = ["applywarp", "-i", f, "-r", ref_file, "-o", out_file,
                   "--interp=%s" % interp, "--premat=%s" % run_affine,
                   "-w", warpfield]
            sub.check_output(cmd)

    out_files = [op.abspath("run_%d/" % (i + 1)) for i in range(n_runs)]
    return out_files
