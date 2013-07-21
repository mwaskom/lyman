"""Preprocessing workflow definition.

See the docstring of create_preprocessing_workflow() for information
about the workflow itself. This module contains the main setup
function and assorted supporting functions for preprocessing.

"""
import os
import numpy as np
import scipy as sp
import nibabel as nib
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import moss
import seaborn

from nipype import fsl
from nipype import freesurfer as fs
from nipype import (Node, MapNode, Workflow,
                    IdentityInterface, Function, Rename)
from nipype.workflows.fmri.fsl import create_susan_smooth


def create_preprocessing_workflow(name="preproc",
                                  temporal_interp=True,
                                  frames_to_toss=6,
                                  interleaved=True,
                                  slice_order="up",
                                  TR=2,
                                  intensity_threshold=3,
                                  motion_threshold=1,
                                  smooth_fwhm=6,
                                  highpass_sigma=32,
                                  partial_brain=False):
    """Return a Nipype workflow for fMRI preprocessing.

    """
    preproc = Workflow(name)

    # Define the inputs for the preprocessing workflow
    in_fields = ["timeseries", "subject_id"]

    if partial_brain:
        in_fields.append("whole_brain_epi")

    inputnode = Node(IdentityInterface(in_fields), "inputnode")

    # Remove equilibrium frames and convert to float
    prepare = MapNode(Function(["in_file", "frames_to_toss"],
                               ["out_file"],
                               prep_timeseries,
                               ["import os",
                                "import numpy as np",
                                "import nibabel as nib"]),
                      "in_file",
                      "prep_timeseries")
    prepare.inputs.frames_to_toss = frames_to_toss

    # Motion and slice time correct
    realign = create_realignment_workflow()

    # Run a conservative skull strip and get a brain mask
    skullstrip = create_skullstrip_workflow()

    # Automatically detect motion and intensity outliers
    artifacts = MapNode(Function(["timeseries",
                                  "mask_file",
                                  "motion_file"
                                  "intensity_thresh",
                                  "motion_thresh"],
                                 ["artifact_report"],
                                 detect_artifacts,
                                 ["import os",
                                  "import pandas as pd",
                                  "import nibabel as nib",
                                  "import matplotlib.pyplot as plt",
                                  "import seaborn"]),
                        ["timeseries", "mask_file", "motion_file"],
                        "artifacts")
    artifacts.inputs.intensity_thresh = intensity_threshold
    artifacts.inputs.motion_thresh = motion_threshold

    # Estimate a registration from funtional to anatomical space
    func2anat = create_bbregister_workflow()

    # Smooth intelligently in the volume
    susan = create_susan_smooth()
    susan.inputs.inputnode.fwhm = smooth_fwhm

    # Scale the grand median of the timeserieses to 10000
    scale_smooth = MapNode(Function(input_names=["in_file",
                                                 "mask",
                                                 "statistic",
                                                 "target"],
                                     output_names=["out_file"],
                                     function=scale_timeseries),
                       iterfield=["in_file", "mask"],
                       name="scale_smooth")
    scale_smooth.inputs.statistic = "median"
    scale_smooth.inputs.target = 10000

    scale_rough = MapNode(util.Function(input_names=["in_file",
                                                      "mask",
                                                      "statistic",
                                                      "target"],
                                        output_names=["out_file"],
                                        function=scale_timeseries),
                           iterfield=["in_file", "mask"],
                           name="scale_rough")
    scale_rough.inputs.statistic = "median"
    scale_rough.inputs.target = 10000

    # High pass filter the two timeserieses
    hpfilt_smooth = MapNode(fsl.TemporalFilter(
                                highpass_sigma=highpass_sigma),
                            iterfield=["in_file"],
                            name="hpfilt_smooth")

    hpfilt_rough = MapNode(fsl.TemporalFilter(
                               highpass_sigma=highpass_sigma),
                           iterfield=["in_file"],
                           name="hpfilt_rough")

    # Rename the output timeserieses
    rename_smooth = MapNode(util.Rename(format_string="smoothed_timeseries",
                                        keep_ext=True),
                            iterfield=["in_file"],
                            name="rename_smooth")

    rename_rough = MapNode(util.Rename(format_string="unsmoothed_timeseries",
                                       keep_ext=True),
                           iterfield=["in_file"],
                           name="rename_rough")

    preproc.connect([
        (inputnode, prepare,
            [("timeseries", "in_file")]),
        (prepare, realign,
            [("out_file", "inputs.timeseries")]),
        (realign, skullstrip,
            [("outputs.timeseries", "inputs.timeseries")]),
        (realign, art,
            [("outputs.realign_parameters", "inputs.realignment_parameters")]),
        (skullstrip, art,
            [("outputs.timeseries", "inputs.realigned_timeseries"),
             ("outputs.mask_file", "inputs.mask_file")]),
        (skullstrip, func2anat,
            [("outputs.mean_file", "inputs.source_file")]),
        (inputnode, func2anat,
            [("subject_id", "inputs.subject_id")]),
        (skullstrip, susan,
            [("outputs.mask_file", "inputnode.mask_file"),
             ("outputs.timeseries", "inputnode.in_files")]),
        (susan, scale_smooth,
            [("outputnode.smoothed_files", "in_file")]),
        (skullstrip, scale_smooth,
            [("outputs.mask_file", "mask")]),
        (skullstrip, scale_rough,
            [("outputs.timeseries", "in_file")]),
        (skullstrip, scale_rough,
            [("outputs.mask_file", "mask")]),
        (scale_smooth, hpfilt_smooth,
            [("out_file", "in_file")]),
        (scale_rough, hpfilt_rough,
            [("out_file", "in_file")]),
        (hpfilt_smooth, rename_smooth,
            [("out_file", "in_file")]),
        (hpfilt_rough, rename_rough,
            [("out_file", "in_file")]),
        ])

    report = MapNode(util.Function(input_names=["subject_id",
                                                "input_timeseries",
                                                "realign_report",
                                                "mask_report",
                                                "intensity_plot",
                                                "outlier_volumes",
                                                "coreg_report"],
                                   output_names=["out_files"],
                                   function=write_preproc_report),
                     iterfield=["input_timeseries",
                                "realign_report",
                                "mask_report",
                                "intensity_plot",
                                "outlier_volumes",
                                "coreg_report"],
                     name="report")

    preproc.connect([
        (inputnode, report,
            [("subject_id", "subject_id"),
             ("timeseries", "input_timeseries")]),
        (realign, report,
            [("outputs.realign_report", "realign_report")]),
        (skullstrip, report,
            [("outputs.mask_report", "mask_report")]),
        (art, report,
            [("outputs.intensity_plot", "intensity_plot"),
             ("outputs.outlier_volumes", "outlier_volumes")]),
        (func2anat, report,
            [("outputs.report", "coreg_report")]),
        ])

    # Define the outputs of the top-level workflow
    output_fields = ["smoothed_timeseries",
                     "unsmoothed_timeseries",
                     "example_func",
                     "mean_file",
                     "functional_mask",
                     "realign_parameters",
                     "mean_func_slices",
                     "intensity_plot",
                     "outlier_volumes",
                     "realign_report",
                     "flirt_affine",
                     "tkreg_affine",
                     "coreg_report",
                     "report_files"]

    outputnode = Node(util.IdentityInterface(fields=output_fields),
                      name="outputnode")

    preproc.connect([
        (realign, outputnode,
            [("outputs.realign_report", "realign_report"),
             ("outputs.realign_parameters", "realign_parameters"),
             ("outputs.example_func", "example_func")]),
        (skullstrip, outputnode,
            [("outputs.mean_func", "mean_func"),
             ("outputs.mask_file", "functional_mask"),
             ("outputs.report_png", "mean_func_slices")]),
        (art, outputnode,
            [("outputs.intensity_plot", "intensity_plot"),
             ("outputs.outlier_volumes", "outlier_volumes")]),
        (func2anat, outputnode,
            [("outputs.tkreg_mat", "tkreg_affine"),
             ("outputs.flirt_mat", "flirt_affine"),
             ("outputs.report", "coreg_report")]),
        (rename_smooth, outputnode,
            [("out_file", "smoothed_timeseries")]),
        (rename_rough, outputnode,
            [("out_file", "unsmoothed_timeseries")]),
        (report, outputnode,
            [("out_files", "report_files")]),
        ])

    return preproc, inputnode, outputnode


def create_realignment_workflow(name="realignment", temporal_interp=True,
                                TR=2, slice_order="up", interleaved=True):
    """Motion and slice-time correct the timeseries and summarize."""

    # Define the workflow inputs
    inputnode = Node(IdentityInterface(["timeseries"], "inputs"))

    # Get the middle volume of each run for motion correction
    extractref = MapNode(fsl.ExtractROI(out_file="example_func.nii.gz",
                                        t_size=1),
                         ["in_file", "t_min"],
                         "extractref")

    # Motion correct to middle volume of each run
    mcflirt = MapNode(fsl.MCFLIRT(cost="normcorr",
                                  interpolation="spline",
                                  save_mats=True,
                                  save_rms=True,
                                  save_plots=True),
                       ["in_file", "ref_file"],
                       "mcflirt")

    # Optionally emoporally interpolate to correct for slice time differences
    if temporal_interp:
        slicetime = MapNode(fsl.SliceTimer(time_repetition=TR),
                            "in_file",
                            "slicetime")

        if slice_order == "down":
            slicetime.inputs.index_dir = True
        elif slice_order != "up":
            raise ValueError("slice_order must be 'up' or 'down'")

        if interleaved:
            slicetime.inputs.interleaved = True

    # Generate a report on the motion correction
    report_inputs = ["target_file", "realign_params", "displace_params"]
    report_outputs = ["realign_report", "motion_file"]
    mcreport = MapNode(Function(report_inputs,
                                report_outputs,
                                realign_report,
                                ["import os",
                                 "import moss",
                                 "import numpy as np",
                                 "import pandas as pd",
                                 "import nibabel as nib",
                                 "import matplotlib.pyplot as plt",
                                 "import seaborn"],
                                ),
                       report_inputs,
                       "mcreport")

    # Define the outputs
    outputnode = Node(IdentityInterface(["timeseries",
                                         "example_func",
                                         "realign_report"
                                         "motion_file"]),

                      "outputs")

    # Define and connect the sub workflow
    realignment = Workflow(name)

    realignment.connect([
        (inputnode, extractref,
            [("timeseries", "in_file"),
             (("timeseries", get_middle_volume), "t_min")]),
        (inputnode, mcflirt,
            [("timeseries", "in_file")]),
        (extractref, mcflirt,
            [("roi_file", "ref_file")]),
        (extractref, mcreport,
            [("roi_file", "target_file")]),
        (mcflirt, mcreport,
            [("par_file", "realign_params"),
             ("rms_files", "displace_params")]),
        (extractref, outputnode,
            [("roi_file", "example_func")]),
        (mcreport, outputnode,
            [("realign_report", "realign_report"),
             ("motion_file", "motion_file")]),
        ])

    if temporal_interp:
        realignment.connect([
            (mcflirt, slicetime,
                [("out_file", "in_filr")]),
            (slicetime, outputnode,
                [("slice_time_corrected_file", "timeseries")])
            ])
    else:
        realignment.connect([
            (realign, outputnode,
                [("out_file", "timeseries")])
            ])

    return realignment


def create_skullstrip_workflow(name="skullstrip"):
    """Remove non-brain voxels from the timeseries."""

    # Define the workflow inputs
    inputnode = Node(IdentityInterface(["timeseries"]), "inputs")

    # Mean the timeseries across the fourth dimension
    origmean = MapNode(fsl.MeanImage(), "in_file", name="origmean")

    # Skullstrip the mean functional image
    findmask = MapNode(fsl.BET(mask=True,
                               no_output=True,
                               frac=0.3),
                        "in_file",
                        "findmask")

    # Use the mask from skullstripping to strip each timeseries
    maskfunc = MapNode(fsl.ApplyMask(),
                       ["in_file", "mask_file"],
                       name="maskfunc")

    # Refine the brain mask
    refinemask = MapNode(Function(["in_file", "mask_file"],
                                  ["timeseries", "mask_file", "mean_file"],
                                  refine_brain_mask,
                                  ["import os",
                                   "import moss",
                                   "import scipy as sp",
                                   "import nibabel as nib"]),
                         ["in_file", "mask_file"],
                         "refinemask")

    # Generate images summarizing the skullstrip and resulting data
    reportmask = MapNode(Function(["mask_file", "orig_file", "mean_file"],
                                  ["mask_report"],
                                  write_mask_report,
                                  ["import os",
                                   "import numpy as np",
                                   "import matplotlib as mpl",
                                   "import matplotlib.pyplot as plt"]),
                         ["mask_file", "mean_file"],
                         "reportmask")

    # Define the workflow outputs
    outputnode = Node(IdentityInterface(["timeseries",
                                         "mean_file",
                                         "mask_file",
                                         "report_png"]),
                      "outputs")

    # Define and connect the workflow
    skullstrip = Workflow(name)

    skullstrip.connect([
        (inputnode, origmean,
            [("timeseries", "in_file")]),
        (origmean, findmask,
            [("out_file", "in_file")]),
        (inputnode, maskfunc,
            [("timeseries", "in_file")]),
        (findmask, maskfunc,
            [("mask_file", "mask_file")]),
        (maskfunc, refinemask,
            [("out_file", "timeseries")]),
        (findmask, refinemask,
            [("mask_file", "mask_file")]),
        (origmean, reportmask,
            [("out_file", "orig_file")]),
        (refinemask, reportmask,
            [("mask_file", "mask_file"),
             ("mean_file", "mean_file")]),
        (refinemask, outputnode,
            [("timeseries", "timeseries"),
             ("mask_file", "mask_file"),
             ("mean_file", "mean_file")]),
        (reportmask, outputnode,
            [("mask_report", "report")]),
        ])

    return skullstrip


def create_bbregister_workflow(name="bbregister", contrast_type="t2",
                               partial_fov=False):

    # Define the workflow inputs
    in_fields = ["subject_id", "source_file"]
    if partial_fov:
        in_fields.append("full_fov_epi")
    inputnode = Node(IdentityInterface(fields=in_fields),
                     name="inputs")

    mean_fullfov = Node(Function(input_names=["in_file"],
                                 output_names=["out_file"],
                                 function=maybe_mean),
                        name="mean_fullfov")

    # Estimate the registration to Freesurfer conformed space
    func2anat = MapNode(fs.BBRegister(contrast_type=contrast_type,
                                      init="fsl",
                                      epi_mask=True,
                                      registered_file=True,
                                      out_fsl_file=True),
                        iterfield=["source_file"],
                        name="func2anat")

    # Make an image for quality control on the registration
    regpng = MapNode(util.Function(input_names=["subject_id", "in_file"],
                                   output_names=["out_file"],
                                   function=write_coreg_plot),
                           iterfield=["in_file"],
                           name="regpng")

    # Rename some files
    costname = MapNode(Rename(format_string="func2anat_cost.dat"),
                       iterfield=["in_file"],
                       name="costname")

    tkregname = MapNode(Rename(format_string="func2anat_tkreg.dat"),
                        iterfield=["in_file"],
                        name="tkregname")

    flirtname = MapNode(util.Rename(format_string="func2anat_flirt.mat"),
                        iterfield=["in_file"],
                        name="flirtname")

    # Merge the slicer png and cost file into a report list
    report = Node(util.Merge(2, axis="hstack"),
                  name="report")

    # Define the workflow outputs
    outputnode = Node(util.IdentityInterface(fields=["tkreg_mat",
                                                     "flirt_mat",
                                                     "report"]),
                      name="outputs")

    bbregister = Workflow(name=name)

    # Connect the registration
    bbregister.connect([
        (inputnode,    func2anat,    [("subject_id", "subject_id"),
                                      ("source_file", "source_file")]),
        (inputnode,    regpng,       [("subject_id", "subject_id")]),
        (func2anat,    regpng,       [("registered_file", "in_file")]),
        (func2anat,    tkregname,    [("out_reg_file", "in_file")]),
        (func2anat,    flirtname,    [("out_fsl_file", "in_file")]),
        (func2anat,    costname,     [("min_cost_file", "in_file")]),
        (costname,     report,       [("out_file", "in1")]),
        (regpng,       report,       [("out_file", "in2")]),
        (tkregname,    outputnode,   [("out_file", "tkreg_mat")]),
        (flirtname,    outputnode,   [("out_file", "flirt_mat")]),
        (report,       outputnode,   [("out", "report")]),
        ])

    # Possibly connect the full_fov image
    if partial_fov:
        bbregister.connect([
            (inputnode, mean_fullfov,
                [("full_fov_epi", "in_file")]),
            (mean_fullfov, func2anat,
                [("out_file", "intermediate_file")]),
                ])

    return bbregister


# ------------------------
# Main interface functions
# ------------------------


def prep_timeseries(in_file, frames_to_toss):
    """Trim equilibrium TRs and change datatype to float."""
    img = nib.load(in_file)
    data = img.get_data()
    aff = img.get_affine()
    hdr = img.get_header()

    data = data[..., frames_to_toss:]
    hdr.set_data_dtype(np.float64)

    new_img = nib.Nifti1Image(data, aff, hdr)
    out_file = os.path.abspath("timeseries.nii.gz")
    new_img.to_filename(out_file)
    return out_file


def maybe_mean(in_file):
    """Mean over time if image is 4D otherwise pass it right back out."""
    from os.path import getcwd
    from nibabel import load, Nifti1Image
    from nipype.utils.filemanip import fname_presuffix

    # Load in the file and get the shape
    img = load(in_file)
    img_shape = img.get_shape()

    # If it's 3D just pass it right back out
    if len(img_shape) <= 3 or img_shape[3] == 1:
        return in_file

    # Otherwise, mean over time dimension and write a new file
    data = img.get_data()
    mean_data = data.mean(axis=3)
    mean_img = Nifti1Image(mean_data, img.get_affine(), img.get_header())

    # Write the mean image to disk and return the filename
    out_file = fname_presuffix(in_file, suffix="_mean", newpath=getcwd())
    mean_img.to_filename(out_file)
    return out_file


def realign_report(target_file, realign_params, displace_params):
    """Create files summarizing the motion correction."""

    # Create a DataFrame with the 6 motion parameters
    rot = ["rot_" + dim for dim in ["x", "y", "z"]]
    trans = ["trans_" + dim for dim in ["x", "y", "z"]]
    df = pd.DataFrame(np.loadtxt(realign_params),
                      columns=rot + trans)

    abs, rel = displace_params
    df["displace_abs"] = np.loadtxt(abs)
    df["displace_rel"] = 0
    df["displace_rel"][1:] = np.loadtxt(rel)
    motion_file = os.path.abspath("realignment_params.csv")
    df.to_csv(motion_file, index=False)

    # Write the motion plots
    seaborn.set()
    seaborn.set_color_palette("husl", 3)
    f, (ax_rot, ax_trans) = plt.subplots(2, 1,
                                         figsize=(8, 3.75),
                                         sharex=True)
    ax_rot.plot(df[rot] * 100)
    ax_rot.axhline(0, c="#444444", ls="--", zorder=1)
    ax_trans.plot(df[trans])
    ax_trans.axhline(0, c="#444444", ls="--", zorder=1)
    ax_rot.set_xlim(0, len(df) - 1)

    ax_rot.set_ylabel(r"Rotations (rad $\times$ 100)")
    ax_trans.set_ylabel("Translations (mm)")
    plt.tight_layout()

    plot_file = os.path.abspath("realignment_plots.png")
    f.savefig(plot_file, dpi=100, bbox_inches="tight")
    plt.close(f)

    # Write the example func plot
    data = nib.load(target_file).get_data()
    n_slices = data.shape[-1]
    n_row, n_col = n_slices // 8, 8
    start = n_slices % n_col // 2
    figsize = (10, 1.375 * n_row)
    f, axes = plt.subplots(n_row, n_col, figsize=figsize)
    f.set_facecolor("k")

    vmin, vmax = 0, moss.percentiles(data, 99)
    for i, ax in enumerate(axes.ravel(), start):
        ax.imshow(data[..., i].T, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
    f.subplots_adjust(hspace=1e-5, wspace=1e-5)
    target_file = os.path.abspath("example_func.png")
    f.savefig(target_file, dpi=100, bbox_inches="tight",
              facecolor="k", edgecolor="k")
    plt.close(f)

    return (motion_file, plot_file, target_file), motion_file


def refine_mask(timeseries, mask_file):
    """Improve brain mask by thresholding and dilating masked timeseries."""
    ts_img = nib.load(timeseries)
    ts_data = ts_img.get_data()

    mask_img = nib.load(mask_file)

    # Find a robust 10% threshold and apply it to the timeseries
    rmin, rmax = moss.percentiles(ts_data, [2, 98])
    thresh = rmin + 0.1 * (rmax + rmin)
    ts_data[ts_data < thresh] = 0
    ts_min = ts_data.min(axis=-1)
    mask = ts_min > 0

    # Dilate the resulting mask by one voxel
    dilator = sp.ndimage.generate_binary_structure(3, 3)
    mask = sp.ndimage.binary_dilation(mask, dilator)

    # Mask the timeseries and save it
    ts_data[:] = ts_data[mask]
    timeseries = os.path.abspath("timeseries_masked.nii.gz")
    new_ts = nib.Nifti1Image(ts_data,
                             ts_img.get_affine(),
                             ts_img.get_header())
    new_ts.to_filename(timeseries)

    # Save the mask image
    mask_file = os.path.abspath("functional_mask.nii.gz")
    new_mask = nib.Nifti1Image(mask,
                               mask_img.get_affine(),
                               mask_img.get_header())
    new_mask.to_filename(mask_file)

    # Make a new mean functional image and save it
    mean_file = os.path.abspath("mean_func.nii.gz")
    new_mean = nib.Nifti1Image(ts_data.mean(axis=-1),
                               ts_img.get_affine(),
                               ts_img.get_header())
    new_mean.to_filename(mean_file)

    return timeseries, mask_file, mean_file


def write_mask_report(mask_file, orig_file, mean_file):
    """Write pngs with the mask and mean iamges."""
    mean = nib.load(mean_file).get_data()
    orig = nib.load(orig_file).get_data()
    mask = nib.load(mask_file).get_data().astype(float)
    mask[mask == 0] = np.nan

    n_slices = mean.shape[-1]
    n_row, n_col = n_slices // 8, 8
    start = n_slices % n_col // 2
    figsize = (10, 1.375 * n_row)

    # Write the functional mask image
    f, axes = plt.subplots(n_row, n_col, figsize=figsize)
    f.set_facecolor("k")
    vmin, vmax = 0, moss.percentiles(orig, 98)

    for i, ax in enumerate(axes.ravel(), start):
        ax.imshow(orig[..., i].T, cmap="gray", vmin=vmin, vmax=vmax)
        ax.imshow(mask[..., i].T, alpha=.6,
                  cmap=mpl.colors.ListedColormap(["MediumSpringGreen"]))
        ax.set_xticks([])
        ax.set_yticks([])
    f.subplots_adjust(hspace=1e-5, wspace=1e-5)
    mask_png = os.path.abspath("functional_mask.png")
    f.savefig(mask_png, dpi=100, bbox_inches="tight",
              facecolor="k", edgecolor="k")
    plt.close(f)

    # Write the mean func image
    f, axes = plt.subplots(n_row, n_col, figsize=figsize)
    f.set_facecolor("k")
    vmin, vmax = 0, moss.percentiles(mean, 98)

    for i, ax in enumerate(axes.ravel(), start):
        ax.imshow(mean[..., i].T, cmap="gray", vmin=vmin, vmax=vmax)
        ax.imshow(mean[..., i].T, cmap="hot", alpha=.6,
                  vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
    f.subplots_adjust(hspace=1e-5, wspace=1e-5)
    mean_png = os.path.abspath("mean_func.png")
    f.savefig(mean_png, dpi=100, bbox_inches="tight",
              facecolor="k", edgecolor="k")
    plt.close(f)

    return mask_png, mean_png


def detect_artifacts(timeseries, mask_file, motion_file,
                     intensity_thresh, motion_thresh):
    """Find frames with exessive signal intensity or motion."""
    seaborn.set()

    # Load the timeseries and detect outliers
    ts = nib.load(timeseries).get_data()
    mask = nib.load(mask_file).get_data().astype(bool)
    ts = ts[mask].mean(axis=0)
    ts = (ts - ts.mean()) / ts.std()
    art_intensity = np.abs(ts) > intensity_thresh

    # Load the motion file and detect outliers
    df = pd.read_csv(motion_file)
    rel_motion = np.array(df["displace_rel"])
    art_motion = rel_motion > motion_thresh

    # Plot the timecourses with outliers
    blue, green, red, _, _ = seaborn.color_palette("deep")
    f, (ax_int, ax_mot) = plt.subplots(2, 1, sharex=True,
                                       figsize=(8, 3.75))
    ax_int.axhline(-intensity_thresh, c="gray", ls="--")
    ax_int.axhline(intensity_thresh, c="gray", ls="--")
    ax_int.plot(ts, c=blue)
    for tr in np.flatnonzero(art_intensity):
        ax_int.axvline(tr, color=red, lw=2.5, alpha=.8)
    ax_int.set_xlim(0, len(df))

    ax_mot.axhline(motion_thresh, c="gray", ls="--")
    ax_mot.plot(rel_motion, c=green)
    ymin, ymax = ax_mot.get_ylim()
    for tr in np.flatnonzero(art_motion):
        ax_mot.axvline(tr, color=red, lw=2.5, alpha=.8)

    ax_int.set_ylabel("Normalized Intensity")
    ax_mot.set_ylabel("Relative Motion (mm)")

    plt.tight_layout()
    plot_file = os.path.abspath("artifact_detection.png")
    f.savefig(plot_file, dpi=100, bbox_inches="tight")

    # Save the artifacts file as csv
    artifacts = pd.DataFrame(dict(intensity=art_intensity,
                                  motion=art_motion)).astype(int)
    art_file = os.path.abspath("artifacts.csv")
    artifacts.to_csv(art_file, index=False)

    return plot_file, art_file


def write_coreg_plot(subject_id, in_file):
    """Plot the wm surface edges on the mean functional."""
    import os
    import os.path as op
    import numpy as np
    import matplotlib.pyplot as plt
    import nibabel as nib
    from nipy.labs import viz
    bold_img = nib.load(in_file)
    bold_data = bold_img.get_data()
    aff = bold_img.get_affine()

    subj_dir = os.environ["SUBJECTS_DIR"]
    wm_file = op.join(subj_dir, subject_id, "mri/wm.mgz")
    wm_data = nib.load(wm_file).get_data()

    f = plt.figure(figsize=(12, 6))
    cut_coords = np.linspace(0, 60, 8).reshape(2, 4)
    ax_positions = [(0, 0, 1, .5), (0, .5, 1, .5)]
    for cc, ap in zip(cut_coords, ax_positions):
        slicer = viz.plot_anat(bold_data, aff, cut_coords=cc,
                               draw_cross=False, annotate=False,
                               slicer="z", axes=ap, figure=f)
        try:
            slicer.contour_map(wm_data, aff, colors="gold")
        except ValueError:
            pass

    out_file = op.abspath("func2anat.png")
    plt.savefig(out_file)
    return out_file


def scale_timeseries(in_file, mask, statistic="median", target=10000):

    import os.path as op
    import numpy as np
    import nibabel as nib
    from nipype.utils.filemanip import split_filename

    ts_img = nib.load(in_file)
    ts_data = ts_img.get_data()
    mask = nib.load(mask).get_data().astype(bool)

    stat_value = getattr(np, statistic)(ts_data[mask])

    scale_value = float(target) / stat_value
    scaled_ts = ts_data * scale_value
    scaled_img = nib.Nifti1Image(scaled_ts,
                                 ts_img.get_affine(),
                                 ts_img.get_header())

    pth, fname, ext = split_filename(in_file)
    out_file = op.abspath(fname + "_scaled.nii.gz")
    scaled_img.to_filename(out_file)

    return out_file


def write_preproc_report(subject_id, input_timeseries, realign_report,
                         mean_func_slices, intensity_plot, outlier_volumes,
                         coreg_report):

    import os.path as op
    import time
    from nibabel import load
    from lyman.tools import write_workflow_report
    from lyman.workflows.reporting import preproc_report_template

    # Gather some attributes of the input timeseries
    input_image = load(input_timeseries)
    image_dimensions = "%dx%dx%d" % input_image.get_shape()[:3]
    image_timepoints = input_image.get_shape()[-1]

    # Read in motion statistics
    motion_file = realign_report[0]
    motion_info = [l.strip() for l in open(motion_file)]
    max_abs_motion, max_rel_motion, total_motion = (motion_info[1],
                                                    motion_info[3],
                                                    motion_info[5])

    # Fill in the report template dictionary
    report_dict = dict(now=time.asctime(),
                       subject_id=subject_id,
                       timeseries_file=input_timeseries,
                       orig_timeseries_path=op.realpath(input_timeseries),
                       image_dimensions=image_dimensions,
                       image_timepoints=image_timepoints,
                       mean_func_slices=mean_func_slices,
                       intensity_plot=intensity_plot,
                       n_outliers=len(open(outlier_volumes).readlines()),
                       max_abs_motion=max_abs_motion,
                       max_rel_motion=max_rel_motion,
                       total_motion=total_motion,
                       realignment_plot=realign_report[1],
                       example_func_slices=realign_report[2],
                       func_to_anat_cost=open(
                           coreg_report[0]).read().split()[0],
                       func_to_anat_slices=coreg_report[1])

    # Write the reports (this is sterotyped for workflows from here
    out_files = write_workflow_report("preproc",
                                      preproc_report_template,
                                      report_dict)

    # Return both report files as a list
    return out_files
