"""Preprocessing workflow definition."""
import os
import os.path as op
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import decomposition as decomp


from moss.mosaic import Mosaic
import seaborn as sns

from nipype.interfaces import io, fsl, freesurfer as fs
from nipype import Node, MapNode, Workflow, IdentityInterface
from nipype.interfaces.base import (BaseInterface,
                                    BaseInterfaceInputSpec,
                                    InputMultiPath, OutputMultiPath,
                                    TraitedSpec, File, traits)
from nipype.workflows.fmri.fsl import create_susan_smooth

import lyman
from lyman.tools import (SingleInFile, SingleOutFile, ManyOutFiles,
                         SaveParameters, list_out_file)


def create_preprocessing_workflow(name="preproc", exp_info=None):
    """Return a Nipype workflow for fMRI preprocessing.

    This mostly follows the preprocessing in FSL, although some
    of the processing has been moved into pure Python.

    Parameters
    ----------
    name : string
        workflow object name
    exp_info : dict
        dictionary with experimental information

    """
    preproc = Workflow(name)

    if exp_info is None:
        exp_info = lyman.default_experiment_parameters()

    # Define the inputs for the preprocessing workflow
    in_fields = ["timeseries", "subject_id"]

    if exp_info["whole_brain_template"]:
        in_fields.append("whole_brain")

    if exp_info["fieldmap_template"]:
        in_fields.append("fieldmap")

    inputnode = Node(IdentityInterface(in_fields), "inputs")

    # Remove equilibrium frames and convert to float
    prepare = MapNode(PrepTimeseries(), "in_file", "prep_timeseries")
    prepare.inputs.frames_to_toss = exp_info["frames_to_toss"]

    # Unwarp using fieldmap images
    if exp_info["fieldmap_template"]:
        unwarp = create_unwarp_workflow(fieldmap_pe=exp_info["fieldmap_pe"])

    # Motion and slice time correct
    realign = create_realignment_workflow(
        temporal_interp=exp_info["temporal_interp"],
        TR=exp_info["TR"],
        slice_order=exp_info["slice_order"],
        interleaved=exp_info["interleaved"])

    # Estimate a registration from funtional to anatomical space
    coregister = create_bbregister_workflow(
        partial_brain=bool(exp_info["whole_brain_template"]),
        init_with=exp_info["coreg_init"])

    # Skullstrip the brain using the Freesurfer segmentation
    skullstrip = create_skullstrip_workflow()

    # Smooth intelligently in the volume
    susan = create_susan_smooth()
    susan.inputs.inputnode.fwhm = exp_info["smooth_fwhm"]

    # Scale and filter the timeseries
    filter_smooth = create_filtering_workflow("filter_smooth",
                                              exp_info["hpf_cutoff"],
                                              exp_info["TR"],
                                              "smoothed_timeseries")

    filter_rough = create_filtering_workflow("filter_rough",
                                             exp_info["hpf_cutoff"],
                                             exp_info["TR"],
                                             "unsmoothed_timeseries")

    # Automatically detect motion and intensity outliers
    artifacts = MapNode(ArtifactDetection(),
                        ["timeseries", "mask_file", "motion_file"],
                        "artifacts")
    artifacts.inputs.intensity_thresh = exp_info["intensity_threshold"]
    artifacts.inputs.motion_thresh = exp_info["motion_threshold"]
    artifacts.inputs.spike_thresh = exp_info["spike_threshold"]

    # Extract nuisance variables from anatomical sources
    confounds = create_confound_extraction_workflow("confounds",
                                                    exp_info["wm_components"])

    # Save the experiment info for this run
    saveparams = MapNode(SaveParameters(exp_info=exp_info),
                         "in_file", "saveparams")

    preproc.connect([
        (inputnode, prepare,
            [("timeseries", "in_file")]),
        (realign, artifacts,
            [("outputs.motion_file", "motion_file")]),
        (realign, coregister,
            [("outputs.timeseries", "inputs.timeseries")]),
        (inputnode, coregister,
            [("subject_id", "inputs.subject_id")]),
        (realign, skullstrip,
            [("outputs.timeseries", "inputs.timeseries")]),
        (inputnode, skullstrip,
            [("subject_id", "inputs.subject_id")]),
        (coregister, skullstrip,
            [("outputs.tkreg_mat", "inputs.reg_file")]),
        (skullstrip, artifacts,
            [("outputs.mask_file", "mask_file")]),
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
        (filter_rough, artifacts,
            [("outputs.timeseries", "timeseries")]),
        (filter_rough, confounds,
            [("outputs.timeseries", "inputs.timeseries")]),
        (inputnode, confounds,
            [("subject_id", "inputs.subject_id")]),
        (skullstrip, confounds,
            [("outputs.mask_file", "inputs.brain_mask")]),
        (coregister, confounds,
            [("outputs.tkreg_mat", "inputs.reg_file")]),
        (inputnode, saveparams,
            [("timeseries", "in_file")]),
        ])

    # Optionally add a connection for unwarping
    if bool(exp_info["fieldmap_template"]):
        preproc.connect([
            (inputnode, unwarp,
                [("fieldmap", "inputs.fieldmap")]),
            (prepare, unwarp,
                [("out_file", "inputs.timeseries")]),
            (unwarp, realign,
                [("outputs.timeseries", "inputs.timeseries")])
        ])
    else:
        preproc.connect([
            (prepare, realign,
                [("out_file", "inputs.timeseries")]),
        ])

    # Optionally connect the whole brain template
    if bool(exp_info["whole_brain_template"]):
        preproc.connect([
            (inputnode, coregister,
                [("whole_brain_template", "inputs.whole_brain_template")])
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
                     "confound_file",
                     "flirt_affine",
                     "tkreg_affine",
                     "coreg_report",
                     "json_file"]

    if bool(exp_info["fieldmap_template"]):
        output_fields.append("unwarp_report")

    outputnode = Node(IdentityInterface(output_fields), "outputs")

    preproc.connect([
        (realign, outputnode,
            [("outputs.example_func", "example_func"),
             ("outputs.report", "realign_report")]),
        (skullstrip, outputnode,
            [("outputs.mask_file", "functional_mask"),
             ("outputs.report", "mask_report")]),
        (artifacts, outputnode,
            [("out_files", "artifact_report")]),
        (coregister, outputnode,
            [("outputs.tkreg_mat", "tkreg_affine"),
             ("outputs.flirt_mat", "flirt_affine"),
             ("outputs.report", "coreg_report")]),
        (filter_smooth, outputnode,
            [("outputs.timeseries", "smoothed_timeseries")]),
        (filter_rough, outputnode,
            [("outputs.timeseries", "unsmoothed_timeseries"),
             ("outputs.mean_file", "mean_func")]),
        (confounds, outputnode,
            [("outputs.confound_file", "confound_file")]),
        (saveparams, outputnode,
            [("json_file", "json_file")]),
        ])

    if bool(exp_info["fieldmap_template"]):
        preproc.connect([
            (unwarp, outputnode,
                [("outputs.report", "unwarp_report")]),
        ])

    return preproc, inputnode, outputnode


# =========================================================================== #


def create_unwarp_workflow(name="unwarp", fieldmap_pe=("y", "y-")):
    """Unwarp functional timeseries using reverse phase-blipped images."""
    inputnode = Node(IdentityInterface(["timeseries", "fieldmap"]), "inputs")

    # Calculate the shift field
    # Note that setting readout_times to 1 will give a fine
    # map of the field, but the units will be off
    # Since we don't write out the map of the field itself, it does
    # not seem worth it to add another parameter for the readout times.
    # (It does require that they are the same, but when wouldn't they be?)
    topup = MapNode(fsl.TOPUP(encoding_direction=fieldmap_pe,
                              readout_times=[1] * len(fieldmap_pe)),
                    ["in_file"], "topup")

    # Unwarp the timeseries
    applytopup = MapNode(fsl.ApplyTOPUP(method="jac", in_index=[1]),
                         ["in_files",
                          "in_topup_fieldcoef",
                          "in_topup_movpar",
                          "encoding_file"],
                         "applytopup")

    # Make a figure summarize the unwarping
    report = MapNode(UnwarpReport(),
                     ["orig_file", "corrected_file"], "unwarp_report")

    # Define the outputs
    outputnode = Node(IdentityInterface(["timeseries", "report"]), "outputs")

    # Define and connect the workflow
    unwarp = Workflow(name)
    unwarp.connect([
        (inputnode, topup,
            [("fieldmap", "in_file")]),
        (inputnode, applytopup,
            [("timeseries", "in_files")]),
        (topup, applytopup,
            [("out_fieldcoef", "in_topup_fieldcoef"),
             ("out_movpar", "in_topup_movpar"),
             ("out_enc_file", "encoding_file")]),
        (inputnode, report,
            [("fieldmap", "orig_file")]),
        (topup, report,
            [("out_corrected", "corrected_file")]),
        (applytopup, outputnode,
            [("out_corrected", "timeseries")]),
        (report, outputnode,
            [("out_file", "report")]),
        ])

    return unwarp


def create_realignment_workflow(name="realignment", temporal_interp=True,
                                TR=2, slice_order="up", interleaved=True):
    """Motion and slice-time correct the timeseries and summarize."""
    inputnode = Node(IdentityInterface(["timeseries"]), "inputs")

    # Get the middle volume of each run for motion correction
    extractref = MapNode(ExtractRealignmentTarget(), "in_file", "extractref")

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
    mcreport = MapNode(RealignmentReport(),
                       ["target_file", "realign_params", "displace_params"],
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
    inputnode = Node(IdentityInterface(["subject_id",
                                        "timeseries",
                                        "reg_file"]),
                     "inputs")

    # Mean the timeseries across the fourth dimension
    origmean = MapNode(fsl.MeanImage(), "in_file", "origmean")

    # Grab the Freesurfer aparc+aseg file as an anatomical brain mask
    getaseg = Node(io.SelectFiles({"aseg": "{subject_id}/mri/aparc+aseg.mgz"},
                                  base_directory=os.environ["SUBJECTS_DIR"]),
                   "getaseg")

    # Threshold the aseg volume to get a boolean mask
    makemask = Node(fs.Binarize(dilate=4, min=0.5), "makemask")

    # Transform the brain mask into functional space
    transform = MapNode(fs.ApplyVolTransform(inverse=True,
                                             interp="nearest"),
                        ["reg_file", "source_file"],
                        "transform")

    # Convert the mask to nifti and rename
    convertmask = MapNode(fs.MRIConvert(out_file="functional_mask.nii.gz"),
                          "in_file", "convertmask")

    # Use the mask to skullstrip the timeseries
    stripts = MapNode(fs.ApplyMask(), ["in_file", "mask_file"], "stripts")

    # Use the mask to skullstrip the mean image
    stripmean = MapNode(fs.ApplyMask(), ["in_file", "mask_file"], "stripmean")

    # Generate images summarizing the skullstrip and resulting data
    reportmask = MapNode(MaskReport(), ["mask_file", "orig_file", "mean_file"],
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
        (inputnode, getaseg,
            [("subject_id", "subject_id")]),
        (origmean, transform,
            [("out_file", "source_file")]),
        (getaseg, makemask,
            [("aseg", "in_file")]),
        (makemask, transform,
            [("binary_file", "target_file")]),
        (inputnode, transform,
            [("reg_file", "reg_file")]),
        (transform, stripts,
            [("transformed_file", "mask_file")]),
        (transform, stripmean,
            [("transformed_file", "mask_file")]),
        (inputnode, stripts,
            [("timeseries", "in_file")]),
        (origmean, stripmean,
            [("out_file", "in_file")]),
        (stripmean, reportmask,
            [("out_file", "mean_file")]),
        (origmean, reportmask,
            [("out_file", "orig_file")]),
        (transform, reportmask,
            [("transformed_file", "mask_file")]),
        (transform, convertmask,
            [("transformed_file", "in_file")]),
        (stripts, outputnode,
            [("out_file", "timeseries")]),
        (stripmean, outputnode,
            [("out_file", "mean_file")]),
        (convertmask, outputnode,
            [("out_file", "mask_file")]),
        (reportmask, outputnode,
            [("out_files", "report")]),
        ])

    return skullstrip


def create_bbregister_workflow(name="bbregister",
                               contrast_type="t2",
                               partial_brain=False,
                               init_with="fsl"):
    """Find a linear transformation to align the EPI file with the anatomy."""
    in_fields = ["subject_id", "timeseries"]
    if partial_brain:
        in_fields.append("whole_brain_template")
    inputnode = Node(IdentityInterface(in_fields), "inputs")

    # Take the mean over time to get a target volume
    meanvol = MapNode(fsl.MeanImage(), "in_file", "meanvol")

    # Do a rough skullstrip using BET
    skullstrip = MapNode(fsl.BET(), "in_file", "bet")

    # Estimate the registration to Freesurfer conformed space
    func2anat = MapNode(fs.BBRegister(contrast_type=contrast_type,
                                      init=init_with,
                                      epi_mask=True,
                                      registered_file=True,
                                      out_reg_file="func2anat_tkreg.dat",
                                      out_fsl_file="func2anat_flirt.mat"),
                        "source_file",
                        "func2anat")

    # Make an image for quality control on the registration
    report = MapNode(CoregReport(), "in_file", "coreg_report")

    # Define the workflow outputs
    outputnode = Node(IdentityInterface(["tkreg_mat", "flirt_mat", "report"]),
                      "outputs")

    bbregister = Workflow(name=name)

    # Connect the registration
    bbregister.connect([
        (inputnode, func2anat,
            [("subject_id", "subject_id")]),
        (inputnode, report,
            [("subject_id", "subject_id")]),
        (inputnode, meanvol,
            [("timeseries", "in_file")]),
        (meanvol, skullstrip,
            [("out_file", "in_file")]),
        (skullstrip, func2anat,
            [("out_file", "source_file")]),
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
                [("whole_brain_template", "intermediate_file")]),
        ])

    return bbregister


def create_filtering_workflow(name="filter",
                              hpf_cutoff=128,
                              TR=2,
                              output_name="timeseries"):
    """Scale and high-pass filter the timeseries."""
    inputnode = Node(IdentityInterface(["timeseries", "mask_file"]),
                     "inputs")

    # Grand-median scale within the brain mask
    scale = MapNode(ScaleTimeseries(statistic="median", target=10000),
                    ["in_file", "mask_file"],
                    "scale")

    # Gaussian running-line filter
    hpf_sigma = (hpf_cutoff / 2.0) / TR
    filter = MapNode(fsl.TemporalFilter(highpass_sigma=hpf_sigma),
                     "in_file",
                     "filter")

    # Possibly replace the mean
    # (In later versions of FSL, the highpass filter removes the
    # mean component. Put it back, but be flexible so this isn't
    # broken on older versions of FSL).
    replacemean = MapNode(ReplaceMean(output_name=output_name),
                          ["orig_file", "filtered_file"],
                          "replacemean")

    # Compute a final mean functional volume
    meanfunc = MapNode(fsl.MeanImage(out_file="mean_func.nii.gz"),
                       "in_file", "meanfunc")

    outputnode = Node(IdentityInterface(["timeseries",
                                         "mean_file"]), "outputs")

    filtering = Workflow(name)
    filtering.connect([
        (inputnode, scale,
            [("timeseries", "in_file"),
             ("mask_file", "mask_file")]),
        (scale, filter,
            [("out_file", "in_file")]),
        (scale, replacemean,
            [("out_file", "orig_file")]),
        (filter, replacemean,
            [("out_file", "filtered_file")]),
        (replacemean, meanfunc,
            [("out_file", "in_file")]),
        (replacemean, outputnode,
            [("out_file", "timeseries")]),
        (meanfunc, outputnode,
            [("out_file", "mean_file")]),
        ])

    return filtering


def create_confound_extraction_workflow(name="confounds", wm_components=6):
    """Extract nuisance variables from anatomical sources."""
    inputnode = Node(IdentityInterface(["timeseries",
                                        "brain_mask",
                                        "reg_file",
                                        "subject_id"]),
                     "inputs")

    # Find the subject's Freesurfer segmentation
    # Grab the Freesurfer aparc+aseg file as an anatomical brain mask
    getaseg = Node(io.SelectFiles({"aseg": "{subject_id}/mri/aseg.mgz"},
                                  base_directory=os.environ["SUBJECTS_DIR"]),
                   "getaseg")

    # Select and erode the white matter to get deep voxels
    selectwm = Node(fs.Binarize(erode=3, wm=True), "selectwm")

    # Transform the mask into functional space
    transform = MapNode(fs.ApplyVolTransform(inverse=True,
                                             interp="nearest"),
                        ["reg_file", "source_file"],
                        "transform")

    # Extract eigenvariates of the timeseries from WM and whole brain
    extract = MapNode(ExtractConfounds(n_components=wm_components),
                      ["timeseries", "brain_mask", "wm_mask"],
                      "extract")

    outputnode = Node(IdentityInterface(["confound_file"]), "outputs")

    confounds = Workflow(name)
    confounds.connect([
        (inputnode, getaseg,
            [("subject_id", "subject_id")]),
        (getaseg, selectwm,
            [("aseg", "in_file")]),
        (selectwm, transform,
            [("binary_file", "target_file")]),
        (inputnode, transform,
            [("reg_file", "reg_file"),
             ("timeseries", "source_file")]),
        (transform, extract,
            [("transformed_file", "wm_mask")]),
        (inputnode, extract,
            [("timeseries", "timeseries"),
             ("brain_mask", "brain_mask")]),
        (extract, outputnode,
            [("out_file", "confound_file")]),
        ])

    return confounds


# =========================================================================== #


class PrepTimeseriesInput(BaseInterfaceInputSpec):

    in_file = File(exists=True)
    frames_to_toss = traits.Int()


class PrepTimeseries(BaseInterface):

    input_spec = PrepTimeseriesInput
    output_spec = SingleOutFile

    def _run_interface(self, runtime):

        # Load the input timeseries
        img = nib.load(self.inputs.in_file)
        data = img.get_data()
        aff = img.get_affine()
        hdr = img.get_header()

        # Trim off the equilibrium TRs
        data = self.trim_timeseries(data)

        # Save the output timeseries as float32
        hdr.set_data_dtype(np.float32)
        new_img = nib.Nifti1Image(data, aff, hdr)
        new_img.to_filename("timeseries.nii.gz")

        return runtime

    def trim_timeseries(self, data):
        """Remove frames from beginning of timeseries."""
        return data[..., self.inputs.frames_to_toss:]

    _list_outputs = list_out_file("timeseries.nii.gz")


class UnwarpReportInput(BaseInterfaceInputSpec):

    orig_file = File(exists=True)
    corrected_file = File(exists=True)


class UnwarpReport(BaseInterface):

    input_spec = UnwarpReportInput
    output_spec = SingleOutFile

    def _run_interface(self, runtime):

        # Make a discrete colormap
        cmap = mpl.colors.ListedColormap(["black", "#d65f5f", "white"])

        # Initialize the figure
        f, axes = plt.subplots(1, 2, figsize=(9, 2.75),
                               facecolor="black", edgecolor="black")

        for ax, fname in zip(axes, [self.inputs.orig_file,
                                    self.inputs.corrected_file]):

            # Combine the frames from this image and plot
            img = nib.load(fname)
            ax.imshow(self.combine_frames(img), cmap=cmap, vmin=0, vmax=2)
            ax.set_axis_off()

        # Save the figure and close
        f.subplots_adjust(0, 0, 1, 1, 0, 0)
        f.savefig("unwarping.png", facecolor="black", edgecolor="black")
        plt.close(f)

        return runtime

    def combine_frames(self, img):

        # Find a value to loosely segment the brain
        d = img.get_data()
        counts, bins = np.histogram(d[d > 0], 50)
        thresh = bins[np.diff(counts) > 0][0]

        # Show the middle slice
        middle = d.shape[0] // 2

        # Combine a binary mask for each phase direction
        a = np.rot90(d[middle, ..., 0] > thresh)
        b = np.rot90(d[middle, ..., 1] > thresh)

        # Make an image showing overlap and divergence
        c = np.zeros_like(a, int)
        c[a ^ b] = 1
        c[a & b] = 2

        return c

    _list_outputs = list_out_file("unwarping.png")


class ExtractRealignmentTarget(BaseInterface):

    input_spec = SingleInFile
    output_spec = SingleOutFile

    def _run_interface(self, runtime):

        # Load the input timeseries
        img = nib.load(self.inputs.in_file)

        # Extract the target volume
        targ = self.extract_target(img)

        # Save a new 3D image
        targ_img = nib.Nifti1Image(targ,
                                   img.get_affine(),
                                   img.get_header())
        targ_img.to_filename("example_func.nii.gz")

        return runtime

    def extract_target(self, img):
        """Return a 3D array with data from the middle TR."""
        middle_vol = img.shape[-1] // 2
        targ = np.empty(img.shape[:-1])
        targ[:] = img.dataobj[..., middle_vol]

        return targ

    _list_outputs = list_out_file("example_func.nii.gz")


class RealignmentReportInput(BaseInterfaceInputSpec):

    target_file = File(exists=True)
    realign_params = File(exists=True)
    displace_params = InputMultiPath(File(exists=True))


class RealignmentReportOutput(TraitedSpec):

    realign_report = OutputMultiPath(File(exists=True))
    motion_file = File(exists=True)


class RealignmentReport(BaseInterface):

    input_spec = RealignmentReportInput
    output_spec = RealignmentReportOutput

    def _run_interface(self, runtime):

        self.out_files = []

        # Load the realignment parameters
        rot = ["rot_" + dim for dim in ["x", "y", "z"]]
        trans = ["trans_" + dim for dim in ["x", "y", "z"]]
        df = pd.DataFrame(np.loadtxt(self.inputs.realign_params),
                          columns=rot + trans)

        # Load the RMS displacement parameters
        abs, rel = self.inputs.displace_params
        df["displace_abs"] = np.loadtxt(abs)
        df["displace_rel"] = pd.Series(np.loadtxt(rel), index=df.index[1:])
        df.loc[0, "displace_rel"] = 0

        # Write the motion file to csv
        self.motion_file = op.abspath("realignment_params.csv")
        df.to_csv(self.motion_file)

        # Plot the motion timeseries
        f = self.plot_motion(df)
        self.plot_file = op.abspath("realignment_plots.png")
        f.savefig(self.plot_file, dpi=100)
        plt.close(f)

        # Plot the target image
        m = self.plot_target()
        self.target_file = op.abspath("example_func.png")
        m.savefig(self.target_file)
        m.close()

        return runtime

    def plot_motion(self, df):
        """Plot the timecourses of realignment parameters."""
        with sns.axes_style("whitegrid"):
            fig, axes = plt.subplots(3, 1, figsize=(9, 5), sharex=True)

        # Trim off all but the axis name
        def axis(s):
            return s[-1]

        # Plot rotations
        pal = sns.color_palette("Reds_d", 3)
        rot_df = np.rad2deg(df.filter(like="rot")).rename(columns=axis)
        rot_df.plot(ax=axes[0], color=pal, lw=1.5)

        # Plot translations
        pal = sns.color_palette("Blues_d", 3)
        trans_df = df.filter(like="trans").rename(columns=axis)
        trans_df.plot(ax=axes[1], color=pal, lw=1.5)

        # Plot displacement
        def ref(s):
            return s[-3:]
        pal = sns.color_palette("Greens_d", 2)
        disp_df = df.filter(like="displace").rename(columns=ref)
        disp_df.plot(ax=axes[2], color=pal, lw=1.5)

        # Label the graphs
        axes[0].set_xlim(0, len(df) - 1)
        axes[0].axhline(0, c=".4", ls="--", zorder=1)
        axes[1].axhline(0, c=".4", ls="--", zorder=1)

        for ax in axes:
            ax.legend(frameon=True, ncol=3, loc="best")
            ax.legend_.get_frame().set_color("white")

        axes[0].set_ylabel("Rotations (degrees)")
        axes[1].set_ylabel("Translations (mm)")
        axes[2].set_ylabel("Displacement (mm)")
        fig.tight_layout()
        return fig

    def plot_target(self):
        """Plot a mosaic of the motion correction target image."""
        m = Mosaic(self.inputs.target_file, step=1)
        return m

    def _list_outputs(self):

        outputs = self._outputs().get()
        outputs["realign_report"] = [self.target_file,
                                     self.motion_file,
                                     self.plot_file]
        outputs["motion_file"] = self.motion_file
        return outputs


class CoregReportInput(BaseInterfaceInputSpec):

    in_file = File(exists=True)
    subject_id = traits.Str()


class CoregReport(BaseInterface):

    input_spec = CoregReportInput
    output_spec = SingleOutFile

    def _run_interface(self, runtime):

        subjects_dir = os.environ["SUBJECTS_DIR"]
        wm_file = op.join(subjects_dir, self.inputs.subject_id, "mri/wm.mgz")
        wm_data = nib.load(wm_file).get_data().astype(bool).astype(int)

        m = Mosaic(self.inputs.in_file, wm_data, step=3)
        m.plot_contours(["#DD2222"])
        m.savefig("func2anat.png")
        m.close()

        return runtime

    _list_outputs = list_out_file("func2anat.png")


class MaskReportInput(BaseInterfaceInputSpec):

    mask_file = File(exists=True)
    orig_file = File(exists=True)
    mean_file = File(exsits=True)


class MaskReport(BaseInterface):

    input_spec = MaskReportInput
    output_spec = ManyOutFiles

    def _run_interface(self, runtime):

        self.out_files = []
        self.plot_mean_image()
        self.plot_mask_image()

        return runtime

    def plot_mean_image(self):

        cmap = sns.cubehelix_palette(as_cmap=True, reverse=True,
                                     light=1, dark=0)
        m = Mosaic(self.inputs.mean_file, self.inputs.mean_file,
                   self.inputs.mask_file, step=1)
        m.plot_overlay(vmin=0, cmap=cmap, fmt="%d")
        m.savefig("mean_func.png")
        m.close()

    def plot_mask_image(self):

        m = Mosaic(self.inputs.orig_file, self.inputs.mask_file,
                   self.inputs.mask_file, show_mask=False, step=1)
        m.plot_mask()
        m.savefig("functional_mask.png")
        m.close()

    def _list_outputs(self):

        outputs = self._outputs().get()
        outputs["out_files"] = [op.abspath("mean_func.png"),
                                op.abspath("functional_mask.png")]
        return outputs


class ArtifactDetectionInput(BaseInterfaceInputSpec):

    timeseries = File(exists=True)
    mask_file = File(exists=True)
    motion_file = File(exists=True)
    intensity_thresh = traits.Float()
    motion_thresh = traits.Float()
    spike_thresh = traits.Either(traits.Float(), None)


class ArtifactDetection(BaseInterface):

    input_spec = ArtifactDetectionInput
    output_spec = ManyOutFiles

    def _run_interface(self, runtime):

        # Load the timeseries and mask files
        ts = nib.load(self.inputs.timeseries).get_data()
        mask = nib.load(self.inputs.mask_file).get_data().astype(bool)

        # Normalize the timeseries using robust statistics
        norm_ts = self.normalize_timeseries(ts, mask)

        # Find the intensity artifacts
        art_intensity = np.abs(norm_ts) > self.inputs.intensity_thresh

        # Load the motion files and find motion artifacts
        df = pd.read_csv(self.inputs.motion_file)
        rel_motion = df["displace_rel"].values
        art_motion = rel_motion > self.inputs.motion_thresh

        # Extract the residual timeseries from each slice
        slices = self.slice_timeseries(ts, mask)
        spike_thresh = self.inputs.spike_thresh
        if spike_thresh is None:
            art_spike = np.zeros_like(art_motion)
        else:
            # Spike threshold is unidirectional
            if spike_thresh < 0:
                art_spike = (slices < spike_thresh).any(axis=0)
            else:
                art_spike = (slices > spike_thresh).any(axis=0)

        # Make a DataFrame of the artifacts
        art_df = pd.DataFrame(dict(intensity=art_intensity,
                                   motion=art_motion,
                                   spikes=art_spike)).astype(int)
        art_df.to_csv("artifacts.csv", index=False)

        # Plot the artifact data
        art_fig = self.plot_artifacts(norm_ts, slices, rel_motion, art_df)
        art_fig.savefig("artifact_detection.png", dpi=100)
        plt.close(art_fig)

        return runtime

    def normalize_timeseries(self, ts, mask):
        """Compute a robust zscore using median and MAD."""
        brain_ts = ts[mask]
        med_ts = np.median(brain_ts, axis=0)
        mad = np.median(np.abs(med_ts - np.median(brain_ts)))
        norm_ts = (med_ts - np.median(brain_ts)) / mad

        return norm_ts

    def plot_artifacts(self, norm, slices, rel, art):

        b, g, r, p, y, c = sns.color_palette("deep")
        with sns.axes_style("whitegrid"):
            f, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 5.5))

        # Plot the main timeseries
        axes[0].plot(norm, color=p, zorder=3)

        palette = sns.color_palette("GnBu_d", len(slices))
        for slice_ts, color in zip(slices, palette):
            axes[1].plot(slice_ts, linewidth=.75, color=color, zorder=3)

        axes[2].plot(rel, color=b, zorder=3)
        axes[0].set_xlim(0, len(norm))

        # Plot the artifacts
        art_kws = dict(color=r, linewidth=2, alpha=.8)
        for t in np.flatnonzero(art["intensity"]):
            axes[0].axvline(t, **art_kws)
        for t in np.flatnonzero(art["spikes"]):
            axes[1].axvline(t, color=r, linewidth=2, alpha=.8)
        for t in np.flatnonzero(art["motion"]):
            axes[2].axvline(t, **art_kws)

        # Plot the thresholds
        thresh_kws = dict(color=".6", linestyle="--")
        axes[0].axhline(self.inputs.intensity_thresh, **thresh_kws)
        axes[0].axhline(-self.inputs.intensity_thresh, **thresh_kws)
        if self.inputs.spike_thresh is not None:
            axes[1].axhline(self.inputs.spike_thresh, **thresh_kws)
        axes[2].axhline(self.inputs.motion_thresh, **thresh_kws)

        # Label the axes
        axes[0].set_ylabel("Normalized Intensity")
        axes[1].set_ylabel("Normalized Intensity")
        axes[2].set_ylabel("Relative Motion (mm)")

        f.tight_layout()
        return f

    def slice_timeseries(self, ts, mask):
        """Get the residual timeseries within the mask from each slice."""
        med_ts = np.median(ts[mask], axis=0)

        slices = []
        for k in range(mask.shape[-1]):
            if not mask[..., k].any():
                continue
            slice_ts = np.median(ts[:, :, k, :][mask[:, :, k]], axis=0)
            slice_ts = slice_ts - med_ts

            slice_med = np.median(slice_ts)
            slice_mad = np.median(np.abs(slice_ts - slice_med))

            norm_ts = (slice_ts - slice_med) / slice_mad
            slices.append(norm_ts)
        slices = np.asarray(slices)

        return slices

    def _list_outputs(self):

        outputs = self._outputs().get()
        outputs["out_files"] = [op.abspath("artifacts.csv"),
                                op.abspath("artifact_detection.png")]
        return outputs


class ScaleTimeseriesInput(BaseInterfaceInputSpec):

    in_file = File(exists=True)
    mask_file = File(exists=True)
    statistic = traits.Str()
    target = traits.Float()


class ScaleTimeseries(BaseInterface):

    input_spec = ScaleTimeseriesInput
    output_spec = SingleOutFile

    def _run_interface(self, runtime):

        ts_img = nib.load(self.inputs.in_file)
        ts_data = ts_img.get_data()
        mask = nib.load(self.inputs.mask_file).get_data().astype(bool)

        # Flexibly get the statistic value.
        # This has to be stringly-typed because nipype
        # can't pass around functions
        stat_func = getattr(np, self.inputs.statistic)

        # Do the scaling
        scaled_ts = self.scale_timeseries(stat_func, ts_data,
                                          mask, self.inputs.target)

        # Save the resulting image
        scaled_img = nib.Nifti1Image(scaled_ts,
                                     ts_img.get_affine(),
                                     ts_img.get_header())
        scaled_img.to_filename("timeseries_scaled.nii.gz")

        return runtime

    def scale_timeseries(self, stat_func, data, mask, target):
        """Make scale timeseries across four dimensions to a target."""
        stat_value = stat_func(data[mask])
        scale_value = target / stat_value
        scaled_data = data * scale_value
        return scaled_data

    _list_outputs = list_out_file("timeseries_scaled.nii.gz")


class ReplaceMeanInput(BaseInterfaceInputSpec):

    orig_file = File(exists=True)
    filtered_file = File(exists=True)
    output_name = traits.String()


class ReplaceMean(BaseInterface):
    """Ensure that filtered timeseries mean is same as before filtering.

    This works around some changes in later version of FSL (as of 11/2014)
    that return a de-meaned timeseries from the FSL highpass filter.
    In FEAT, the mean is replaced, and the rest of the processing carries
    on as usual. Because I don't want to break compatability with older
    versions of FSL, this adds back in the mean but only if it looks
    like the filtered timeseries has been de-meaned.

    """
    input_spec = ReplaceMeanInput
    output_spec = SingleOutFile

    def _run_interface(self, runtime):

        # Load the original and filtered timeseries data
        orig_img = nib.load(self.inputs.orig_file)
        orig_mean = orig_img.get_data().mean(axis=-1)

        filtered_img = nib.load(self.inputs.filtered_file)
        filtered_data = filtered_img.get_data()
        filtered_mean = filtered_data.mean(axis=-1)

        # Simple heuristic: if the maximum value in the mean image
        # is less than 1, it looks like this timeseries has been
        # de-meaned. In practice these values seem to be around
        # 1e-5, so this should be safe.
        if filtered_mean.max() < 1:
            replacement = orig_mean
        else:
            replacement = np.zeros_like(orig_mean)
        replacement = replacement[..., np.newaxis]

        # Add back in what is needed and write out the image
        new_data = filtered_data + replacement
        new_img = nib.Nifti1Image(new_data,
                                  filtered_img.get_affine(),
                                  filtered_img.get_header())
        out_fname = "{}.nii.gz".format(self.inputs.output_name)
        new_img.to_filename(out_fname)

        return runtime

    def _list_outputs(self):

        outputs = self._outputs().get()
        out_fname = "{}.nii.gz".format(self.inputs.output_name)
        outputs["out_file"] = op.abspath(out_fname)
        return outputs


class ExtractConfoundsInput(BaseInterfaceInputSpec):

    timeseries = File(Exists=True)
    wm_mask = File(Exists=True)
    brain_mask = File(Exists=True)
    n_components = traits.Int()


class ExtractConfounds(BaseInterface):
    """Extract nuisance variables from anatomical sources."""
    input_spec = ExtractConfoundsInput
    output_spec = SingleOutFile

    def _run_interface(self, runtime):

        # Load the brain images
        ts_data = nib.load(self.inputs.timeseries).get_data()
        wm_mask = nib.load(self.inputs.wm_mask).get_data()
        brain_mask = nib.load(self.inputs.brain_mask).get_data()

        # Set up the output dataframe
        wm_cols = ["wm_{:d}".format(i)
                   for i in range(self.inputs.n_components)]
        cols = wm_cols + ["brain"]
        index = np.arange(ts_data.shape[-1])
        out_df = pd.DataFrame(index=index, columns=cols, dtype=np.float)

        # Extract eigenvariates of the white matter timeseries
        wm_ts = ts_data[wm_mask.astype(bool)].T
        wm_pca = decomp.PCA(self.inputs.n_components)
        wm_comp = wm_pca.fit_transform(wm_ts)
        out_df[wm_cols] = wm_comp

        # Extract the mean whole-brain timeseries
        brain_ts = ts_data[brain_mask.astype(bool)].mean(axis=0)
        out_df["brain"] = brain_ts

        # Write out the resulting data to disk
        out_df.to_csv("nuisance_variables.csv", index=False)

        return runtime

    _list_outputs = list_out_file("nuisance_variables.csv")
