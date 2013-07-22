"""Preprocessing workflow definition.

See the docstring of create_preprocessing_workflow() for information
about the workflow itself. This module contains the main setup
function and assorted supporting functions for preprocessing.

"""
import os
import numpy as np
import scipy as sp
import pandas as pd
import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt

import moss
import seaborn

from nipype import fsl
from nipype import freesurfer as fs
from nipype import (Node, MapNode, Workflow,
                    IdentityInterface, Function)
from nipype.workflows.fmri.fsl import create_susan_smooth

# For nipype Function interfaces
imports = ["import os",
           "import numpy as np",
           "import scipy as sp",
           "import pandas as pd",
           "import nibabel as nib",
           "import matplotlib as mpl",
           "import matplotlib.pyplot as plt",
           "import moss",
           "import seaborn"]


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

    This mostly follows the preprocessing in FSL, although some
    of the processing has been moved into pure Python.

    Parameters
    ----------
    name : string
        workflow object name
    temporal_interp : bool
        whether to perform slice-time correction
    frames_to_toss : int
        number of initial frames to remove
    interleaved : bool
        whether slice acquisition is interleaved/alternating
    slice_order : "up" | "down"
        direction of slice acquisition
    TR : float
        repetition time of the sequence
    intensity_threshold : float
        z-score threshold for whole-brain intensity artifacts
    motion_thresold : float
        mm threshold for scan-to-scan motion artifacts
    smooth_fwhm : float
        mm smoothing kernel for spatial smoothing
    highpass_sigma : float
        sigma (in s) of high-pass smoothing kernal
    partial_brain : bool
        protocol is partial brain/hi-res

    """
    preproc = Workflow(name)

    # Define the inputs for the preprocessing workflow
    in_fields = ["timeseries", "subject_id"]

    if partial_brain:
        in_fields.append("whole_brain_epi")

    inputnode = Node(IdentityInterface(in_fields), "inputs")

    # Remove equilibrium frames and convert to float
    prepare = MapNode(Function(["in_file", "frames_to_toss"],
                               ["out_file"],
                               prep_timeseries,
                               imports),
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
                                  "motion_file",
                                  "intensity_thresh",
                                  "motion_thresh"],
                                 ["artifact_report"],
                                 detect_artifacts,
                                 imports),
                        ["timeseries", "mask_file", "motion_file"],
                        "artifacts")
    artifacts.inputs.intensity_thresh = intensity_threshold
    artifacts.inputs.motion_thresh = motion_threshold

    # Estimate a registration from funtional to anatomical space
    coregister = create_bbregister_workflow()

    # Smooth intelligently in the volume
    susan = create_susan_smooth()
    susan.inputs.inputnode.fwhm = smooth_fwhm

    # Scale and filter the timeseries
    filter_smooth = create_filtering_workflow("filter_smooth",
                                              highpass_sigma,
                                              "smoothed_timeseries")

    filter_rough = create_filtering_workflow("filter_rough",
                                              highpass_sigma,
                                              "unsmoothed_timeseries")

    preproc.connect([
        (inputnode, prepare,
            [("timeseries", "in_file")]),
        (prepare, realign,
            [("out_file", "inputs.timeseries")]),
        (realign, skullstrip,
            [("outputs.timeseries", "inputs.timeseries")]),
        (realign, artifacts,
            [("outputs.motion_file", "motion_file")]),
        (skullstrip, artifacts,
            [("outputs.timeseries", "timeseries"),
             ("outputs.mask_file", "mask_file")]),
        (skullstrip, coregister,
            [("outputs.mean_file", "inputs.source_file")]),
        (inputnode, coregister,
            [("subject_id", "inputs.subject_id")]),
        (skullstrip, susan,
            [("outputs.mask_file", "inputnode.mask_file"),
             ("outputs.timeseries", "inputnode.in_files")]),
        (susan, filter_smooth,
            [("outputnode.smoothed_files", "inputs.timeseries")]),
        (skullstrip, filter_smooth,
            [("outputs.mask_file", "inputs.mask_file")]),
        (skullstrip, filter_rough,
            [("outputs.timeseries", "inputs.timeseries")]),
        (skullstrip, filter_rough,
            [("outputs.mask_file", "inputs.mask_file")]),
        ])

    # Define the outputs of the top-level workflow
    output_fields = ["smoothed_timeseries",
                     "unsmoothed_timeseries",
                     "example_func",
                     "mean_func",
                     "functional_mask",
                     "realign_report",
                     "mask_report",
                     "artifact_report",
                     "flirt_affine",
                     "tkreg_affine",
                     "coreg_report"]

    outputnode = Node(IdentityInterface(output_fields), "outputs")

    preproc.connect([
        (realign, outputnode,
            [("outputs.example_func", "example_func"),
             ("outputs.report", "realign_report")]),
        (skullstrip, outputnode,
            [("outputs.mean_file", "mean_func"),
             ("outputs.mask_file", "functional_mask"),
             ("outputs.report", "mask_report")]),
        (artifacts, outputnode,
            [("artifact_report", "artifact_report")]),
        (coregister, outputnode,
            [("outputs.tkreg_mat", "tkreg_affine"),
             ("outputs.flirt_mat", "flirt_affine"),
             ("outputs.report", "coreg_report")]),
        (filter_smooth, outputnode,
            [("outputs.timeseries", "smoothed_timeseries")]),
        (filter_rough, outputnode,
            [("outputs.timeseries", "unsmoothed_timeseries")]),
        ])

    return preproc, inputnode, outputnode


def create_realignment_workflow(name="realignment", temporal_interp=True,
                                TR=2, slice_order="up", interleaved=True):
    """Motion and slice-time correct the timeseries and summarize."""
    inputnode = Node(IdentityInterface(["timeseries"]), "inputs")

    # Get the middle volume of each run for motion correction
    extractref = MapNode(Function(["in_file"],
                                  ["out_file"],
                                  extract_mc_target,
                                  imports),
                         "in_file",
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
                                imports),
                       report_inputs,
                       "mcreport")

    # Define the outputs
    outputnode = Node(IdentityInterface(["timeseries",
                                         "example_func",
                                         "report",
                                         "motion_file"]),

                      "outputs")

    # Define and connect the sub workflow
    realignment = Workflow(name)

    realignment.connect([
        (inputnode, extractref,
            [("timeseries", "in_file")]),
        (inputnode, mcflirt,
            [("timeseries", "in_file")]),
        (extractref, mcflirt,
            [("out_file", "ref_file")]),
        (extractref, mcreport,
            [("out_file", "target_file")]),
        (mcflirt, mcreport,
            [("par_file", "realign_params"),
             ("rms_files", "displace_params")]),
        (extractref, outputnode,
            [("out_file", "example_func")]),
        (mcreport, outputnode,
            [("realign_report", "report"),
             ("motion_file", "motion_file")]),
        ])

    if temporal_interp:
        realignment.connect([
            (mcflirt, slicetime,
                [("out_file", "in_file")]),
            (slicetime, outputnode,
                [("slice_time_corrected_file", "timeseries")])
            ])
    else:
        realignment.connect([
            (mcflirt, outputnode,
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
    refinemask = MapNode(Function(["timeseries", "mask_file"],
                                  ["timeseries", "mask_file", "mean_file"],
                                  refine_mask,
                                  imports),
                         ["timeseries", "mask_file"],
                         "refinemask")

    # Generate images summarizing the skullstrip and resulting data
    reportmask = MapNode(Function(["mask_file", "orig_file", "mean_file"],
                                  ["mask_report"],
                                  write_mask_report,
                                  imports),
                         ["mask_file", "orig_file", "mean_file"],
                         "reportmask")

    # Define the workflow outputs
    outputnode = Node(IdentityInterface(["timeseries",
                                         "mean_file",
                                         "mask_file",
                                         "report"]),
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


def create_bbregister_workflow(name="bbregister",
                               contrast_type="t2",
                               partial_brain=False):
    """Find a linear transformation to align the EPI file with the anatomy."""
    in_fields = ["subject_id", "source_file"]
    if partial_brain:
        in_fields.append("whole_brain_epi")
    inputnode = Node(IdentityInterface(in_fields),
                     "inputs")

    # Estimate the registration to Freesurfer conformed space
    func2anat = MapNode(fs.BBRegister(contrast_type=contrast_type,
                                      init="fsl",
                                      epi_mask=True,
                                      registered_file=True,
                                      out_reg_file="func2anat_tkreg.dat",
                                      out_fsl_file="func2anat_flirt.mat"),
                        "source_file",
                        "func2anat")

    # Make an image for quality control on the registration
    report = MapNode(Function(["subject_id", "in_file"],
                              ["out_file"],
                              write_coreg_plot,
                              imports),
                           "in_file",
                           "coreg_report")

    # Define the workflow outputs
    outputnode = Node(IdentityInterface(["tkreg_mat", "flirt_mat", "report"]),
                      "outputs")

    bbregister = Workflow(name=name)

    # Connect the registration
    bbregister.connect([
        (inputnode, func2anat,
            [("subject_id", "subject_id"),
             ("source_file", "source_file")]),
        (inputnode, report,
            [("subject_id", "subject_id")]),
        (func2anat, report,
            [("registered_file", "in_file")]),
        (func2anat, outputnode,
            [("out_reg_file", "tkreg_mat")]),
        (func2anat, outputnode,
            [("out_fsl_file", "flirt_mat")]),
        (report, outputnode,
            [("out_file", "report")]),
        ])

    # Possibly connect the full_fov image
    if partial_brain:
        bbregister.connect([
            (inputnode, func2anat,
                [("whole_brain_epi", "intermediate_file")]),
                ])

    return bbregister


def create_filtering_workflow(name="filter",
                              highpass_sigma=32,
                              output_name="timeseries"):
    """Scale and high-pass filter the timeseries."""
    inputnode = Node(IdentityInterface(["timeseries", "mask_file"]),
                     "inputs")

    # Grand-median scale within the brain mask
    scale = MapNode(Function(["in_file",
                              "mask_file"],
                             ["out_file"],
                             scale_timeseries,
                             imports),
                    ["in_file", "mask_file"],
                    "scale")

    # Gaussian running-line filter
    filter = MapNode(fsl.TemporalFilter(highpass_sigma=highpass_sigma,
                                        out_file=output_name + ".nii.gz"),
                     "in_file",
                     "filter")

    outputnode = Node(IdentityInterface(["timeseries"]), "outputs")

    filtering = Workflow(name)
    filtering.connect([
        (inputnode, scale,
            [("timeseries", "in_file"),
             ("mask_file", "mask_file")]),
        (scale, filter,
            [("out_file", "in_file")]),
        (filter, outputnode,
            [("out_file", "timeseries")]),
        ])

    return filtering


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
    hdr.set_data_dtype(np.float32)

    new_img = nib.Nifti1Image(data, aff, hdr)
    out_file = os.path.abspath("timeseries.nii.gz")
    new_img.to_filename(out_file)
    return out_file


def extract_mc_target(in_file):
    """Extract the middle frame of a timeseries."""
    img = nib.load(in_file)
    data = img.get_data()

    middle_vol = data.shape[-1] // 2
    targ = np.empty(data.shape[:-1])
    targ[:] = data[..., middle_vol]

    targ_img = nib.Nifti1Image(targ, img.get_affine(), img.get_header())
    out_file = os.path.abspath("example_func.nii.gz")
    targ_img.to_filename(out_file)
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
    df["displace_rel"] = pd.Series(np.loadtxt(rel), index=df.index[1:])
    df.loc[0, "displace_rel"] = 0
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
    f, axes = plt.subplots(n_row, n_col, figsize=figsize, facecolor="k")

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

    return [motion_file, plot_file, target_file], motion_file


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
    ts_data[~mask] = 0
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
    f, axes = plt.subplots(n_row, n_col, figsize=figsize, facecolor="k")
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
    f, axes = plt.subplots(n_row, n_col, figsize=figsize, facecolor="k")
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

    return [mask_png, mean_png]


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

    return [plot_file, art_file]


def write_coreg_plot(subject_id, in_file):
    """Plot the wm surface edges on the mean functional."""
    bold = nib.load(in_file).get_data()

    # Load the white matter volume from recon-all
    subj_dir = os.environ["SUBJECTS_DIR"]
    wm_file = os.path.join(subj_dir, subject_id, "mri/wm.mgz")
    wm = nib.load(wm_file).get_data()

    # Find the limits of the data
    # note that FS conformed space is not (x, y, z)
    xdata = np.flatnonzero(bold.any(axis=1).any(axis=1))
    xmin, xmax = xdata.min(), xdata.max()
    ydata = np.flatnonzero(bold.any(axis=0).any(axis=0))
    ymin, ymax = ydata.min(), ydata.max()
    zdata = np.flatnonzero(wm.any(axis=0).any(axis=1))
    zmin, zmax = zdata.min() + 10, zdata.max() - 25

    # Figure out the plot parameters
    n_slices = (zmax - zmin) // 3
    n_row, n_col = n_slices // 8, 8
    start = n_slices % n_col // 2 + zmin
    figsize = (10, 1.375 * n_row)
    slices = (start + np.arange(zmax - zmin))[::3][:n_slices]

    # Draw the slices and save
    vmin, vmax = 0, moss.percentiles(bold, 99)
    f, axes = plt.subplots(n_row, n_col, figsize=figsize, facecolor="k")
    for i, ax in enumerate(axes.ravel()):
        i = slices[i]
        ax.imshow(bold[xmin:xmax, i, ymin:ymax].T, cmap="gray",
                  vmin=vmin, vmax=vmax)
        ax.contour(wm[xmin:xmax, i, ymin:ymax].T, linewidths=.5,
                   cmap=mpl.colors.ListedColormap(["#C41E3A"]))
        ax.set_xticks([])
        ax.set_yticks([])

    out_file = os.path.abspath("func2anat.png")
    plt.savefig(out_file, dpi=100, bbox_inches="tight",
                facecolor="k", edgecolor="k")
    return out_file


def scale_timeseries(in_file, mask_file, statistic="median", target=10000):
    """Scale an entire series with a single number."""
    ts_img = nib.load(in_file)
    ts_data = ts_img.get_data()
    mask = nib.load(mask_file).get_data().astype(bool)

    # Flexibly get the statistic value.
    # This has to be stringly-typed because nipype
    # can't pass around functions
    stat_value = getattr(np, statistic)(ts_data[mask])

    scale_value = float(target) / stat_value
    scaled_ts = ts_data * scale_value
    scaled_img = nib.Nifti1Image(scaled_ts,
                                 ts_img.get_affine(),
                                 ts_img.get_header())

    out_file = os.path.abspath("timeseries_scaled.nii.gz")
    scaled_img.to_filename(out_file)

    return out_file
