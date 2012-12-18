"""Preprocessing workflow definition.

See the docstring of create_preprocessing_workflow() for information
about the workflow itself. This module contains the main setup
function and assorted supporting functions for preprocessing.

"""
import nipype.interfaces.io as io
import nipype.interfaces.fsl as fsl
import nipype.interfaces.freesurfer as fs
import nipype.interfaces.utility as util
import nipype.algorithms.rapidart as ra
from nipype.interfaces.utility import IdentityInterface, Rename, Function
from nipype.pipeline.engine import Node, MapNode, Workflow

try:
    from nipype.workflows.fmri.fsl import create_susan_smooth
except ImportError:
    from nipype.workflows.fsl import create_susan_smooth


def create_preprocessing_workflow(name="preproc",
                                  do_slice_time_cor=True,
                                  frames_to_toss=6,
                                  interleaved=True,
                                  slice_order="up",
                                  TR=2,
                                  smooth_fwhm=6,
                                  highpass_sigma=32,
                                  partial_fov=False):
    """Return a Nipype workflow for fMRI preprocessing.


    """
    preproc = Workflow(name=name)

    # Define the inputs for the preprocessing workflow
    in_fields = ["timeseries", "subject_id"]
    if partial_fov:
        in_fields.append("full_fov_epi")
    inputnode = Node(IdentityInterface(fields=in_fields),
                     name="inputnode")

    # Remove early frames to account for T1 stabalization
    trimmer = MapNode(fsl.ExtractROI(t_min=frames_to_toss),
                      iterfield=["in_file", "t_size"],
                      name="trimmer")

    # Convert functional images to float representation
    img2float = MapNode(fsl.ChangeDataType(output_datatype="float"),
                        iterfield=["in_file"],
                        name="img2float")

    # Correct for slice-time acquisition differences
    # We handle the logic of slice timing and realignment order
    # below in the connections
    if do_slice_time_cor:
        slicetime = MapNode(fsl.SliceTimer(time_repetition=TR),
                            iterfield=["in_file"],
                            name="slicetime")
        if slice_order == "down":
            slicetime.inputs.index_dir = True
        elif slice_order != "up":
            raise ValueError("slice_order must be 'up' or 'down'")
        if interleaved:
            slicetime.inputs.interleaved = True

    # Realign each timeseries to the middle volume of that run
    realign = create_realignment_workflow()

    # Run a conservative skull strip and get a brain mask
    skullstrip = create_skullstrip_workflow()

    # Automatically detect motion and intensity outliers
    art = create_art_workflow()

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
        (inputnode, trimmer,
            [("timeseries", "in_file"),
            (("timeseries", get_trimmed_length, frames_to_toss), "t_size")]),
        (trimmer, img2float,
            [("roi_file", "in_file")]),
        (img2float, realign,
            [("out_file", "inputs.timeseries")]),
        (realign, art,
            [("outputs.realign_parameters", "inputs.realignment_parameters")]),
        (img2float, art,
            [("out_file", "inputs.raw_timeseries")]),
        (skullstrip, art,
            [("outputs.timeseries", "inputs.realigned_timeseries"),
             ("outputs.mask_file", "inputs.mask_file")]),
        (skullstrip, func2anat,
            [("outputs.mean_func", "inputs.source_file")]),
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

    # Possibly hook-up slice time correction
    if do_slice_time_cor:
        preproc.connect([
            (realign,   slicetime,
                [("outputs.timeseries", "in_file")]),
            (slicetime, skullstrip,
                [("slice_time_corrected_file", "inputs.timeseries")]),
            ])
    else:
        preproc.connect([
            (realign,     skullstrip,
                [("outputs.timeseries", "inputs.timeseries")]),
            ])

    report = MapNode(util.Function(input_names=["subject_id",
                                                "input_timeseries",
                                                "realign_report",
                                                "mean_func_slices",
                                                "intensity_plot",
                                                "outlier_volumes",
                                                "coreg_report"],
                                   output_names=["out_files"],
                                   function=write_preproc_report),
                     iterfield=["input_timeseries",
                                "realign_report",
                                "mean_func_slices",
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
            [("outputs.report_png", "mean_func_slices")]),
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
                     "mean_func",
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


def create_realignment_workflow(name="realignment"):

    # Define the workflow inputs
    inputnode = Node(util.IdentityInterface(fields=["timeseries"]),
                     name="inputs")

    # Get the middle volume of each run for motion correction
    extractref = MapNode(fsl.ExtractROI(t_size=1),
                         iterfield=["in_file", "t_min"],
                         name="extractref")

    # Slice the example func for reporting
    exampleslice = MapNode(fsl.Slicer(image_width=800,
                                      all_axial=True,
                                      show_orientation=False,
                                      label_slices=False),
                           iterfield=["in_file"],
                           name="exampleslice")

    # Motion correct to middle volume of each run
    mcflirt = MapNode(fsl.MCFLIRT(save_plots=True,
                                  save_rms=True,
                                  save_mats=True,
                                  args="-spline_final"),
                       name="mcflirt",
                       iterfield=["in_file", "ref_file"])

    # Generate a report on the motion correction
    report_inputs = ["realign_params", "rms_files"]
    report_outputs = ["motion_file", "disp_plot", "rot_plot", "trans_plot"]
    mcreport = MapNode(util.Function(input_names=report_inputs,
                                     output_names=report_outputs,
                                     function=write_realign_report),
                       iterfield=report_inputs,
                       name="mcreport")

    # Rename some things
    exfuncname = MapNode(util.Rename(format_string="example_func",
                                     keep_ext=True),
                         iterfield=["in_file"],
                         name="exfuncname")

    exslicename = MapNode(util.Rename(format_string="example_func",
                                      keep_ext=True),
                          iterfield=["in_file"],
                          name="exslicename")

    parname = MapNode(util.Rename(format_string="realignment_parameters.par"),
                      iterfield=["in_file"],
                      name="parname")

    # Send out all the report data as one list
    mergereport = Node(util.Merge(numinputs=5, axis="hstack"),
                       name="mergereport")

    # Define the outputs
    outputnode = Node(util.IdentityInterface(fields=["timeseries",
                                                     "example_func",
                                                     "realign_parameters",
                                                     "realign_report"]),
                      name="outputs")

    # Define and connect the sub workflow
    realignment = Workflow(name=name)

    realignment.connect([
        (inputnode, extractref,
            [("timeseries", "in_file"),
             (("timeseries", get_middle_volume), "t_min")]),
        (extractref, exampleslice,
            [("roi_file", "in_file")]),
        (inputnode, mcflirt,
            [("timeseries", "in_file")]),
        (extractref, mcflirt,
            [("roi_file", "ref_file")]),
        (mcflirt, mcreport,
            [("par_file", "realign_params"),
             ("rms_files", "rms_files")]),
        (exampleslice, exslicename,
            [("out_file", "in_file")]),
        (mcreport, mergereport,
            [("motion_file", "in1"),
             ("rot_plot", "in2"),
             ("disp_plot", "in3"),
             ("trans_plot", "in4")]),
        (exslicename, mergereport,
            [("out_file", "in5")]),
        (mcflirt, parname,
            [("par_file", "in_file")]),
        (parname, outputnode,
            [("out_file", "realign_parameters")]),
        (extractref, exfuncname,
            [("roi_file", "in_file")]),
        (mcflirt, outputnode,
            [("out_file", "timeseries")]),
        (exfuncname, outputnode,
            [("out_file", "example_func")]),
        (mergereport, outputnode,
            [("out", "realign_report")]),
        ])

    return realignment


def create_skullstrip_workflow(name="skullstrip"):

    # Define the workflow inputs
    inputnode = Node(util.IdentityInterface(fields=["timeseries"]),
                     name="inputs")

    # Mean the timeseries across the fourth dimension
    meanfunc1 = MapNode(fsl.MeanImage(),
                        iterfield=["in_file"],
                        name="meanfunc1")

    # Skullstrip the mean functional image
    stripmean = MapNode(fsl.BET(mask=True,
                                no_output=True,
                                frac=0.3),
                        iterfield=["in_file"],
                        name="stripmean")

    # Use the mask from skullstripping to strip each timeseries
    maskfunc1 = MapNode(fsl.ApplyMask(),
                        iterfield=["in_file", "mask_file"],
                        name="maskfunc1")

    # Determine the 2nd and 98th centile intensities of each run
    getthresh = MapNode(fsl.ImageStats(op_string="-p 2 -p 98"),
                        iterfield=["in_file"],
                        name="getthreshold")

    # Threshold functional data at 10% of the 98th percentile
    threshold = MapNode(fsl.ImageMaths(out_data_type="char",
                                       suffix="_thresh"),
                        iterfield=["in_file"],
                        name="threshold")

    # Dilate the mask
    dilatemask = MapNode(fsl.DilateImage(operation="max"),
                         iterfield=["in_file"],
                         name="dilatemask")

    # Mask the runs again with this new mask
    maskfunc2 = MapNode(fsl.ApplyMask(),
                        iterfield=["in_file", "mask_file"],
                        name="maskfunc2")

    # Get a new mean image from each functional run
    meanfunc2 = MapNode(fsl.MeanImage(),
                        iterfield=["in_file"],
                        name="meanfunc2")

    # Slice the mean func for reporting
    meanslice = MapNode(fsl.Slicer(image_width=800,
                                   all_axial=True,
                                   show_orientation=False,
                                   label_slices=False),
                        iterfield=["in_file", "image_edges"],
                        name="meanslice")

    # Rename the outputs
    meanname = MapNode(util.Rename(format_string="mean_func",
                                   keep_ext=True),
                       iterfield=["in_file"],
                       name="meanname")

    maskname = MapNode(util.Rename(format_string="functional_mask",
                                   keep_ext=True),
                       iterfield=["in_file"],
                       name="maskname")

    pngname = MapNode(util.Rename(format_string="mean_func.png"),
                      iterfield=["in_file"],
                      name="pngname")

    # Define the workflow outputs
    outputnode = Node(util.IdentityInterface(fields=["timeseries",
                                                     "mean_func",
                                                     "mask_file",
                                                     "report_png"]),
                      name="outputs")

    # Define and connect the workflow
    skullstrip = Workflow(name=name)

    skullstrip.connect([
        (inputnode,  meanfunc1,     [("timeseries", "in_file")]),
        (meanfunc1,  stripmean,     [("out_file", "in_file")]),
        (inputnode,  maskfunc1,     [("timeseries", "in_file")]),
        (stripmean,  maskfunc1,     [("mask_file", "mask_file")]),
        (maskfunc1,  getthresh,     [("out_file", "in_file")]),
        (getthresh,  threshold,
            [(("out_stat", get_thresh_op), "op_string")]),
        (maskfunc1,  threshold,     [("out_file", "in_file")]),
        (threshold,  dilatemask,    [("out_file", "in_file")]),
        (inputnode,  maskfunc2,     [("timeseries", "in_file")]),
        (dilatemask, maskfunc2,     [("out_file", "mask_file")]),
        (maskfunc2,  meanfunc2,     [("out_file", "in_file")]),
        (meanfunc2,  meanslice,     [("out_file", "in_file")]),
        (dilatemask, meanslice,     [("out_file", "image_edges")]),
        (meanslice,  pngname,       [("out_file", "in_file")]),
        (meanfunc2,  meanname,      [("out_file", "in_file")]),
        (dilatemask, maskname,      [("out_file", "in_file")]),
        (maskfunc2,  outputnode,    [("out_file", "timeseries")]),
        (pngname,    outputnode,    [("out_file", "report_png")]),
        (maskname,   outputnode,    [("out_file", "mask_file")]),
        (meanname,   outputnode,    [("out_file", "mean_func")]),
        ])

    return skullstrip


def create_art_workflow(name="art"):

    # Define the workflow inputs
    inputnode = Node(util.IdentityInterface(fields=["raw_timeseries",
                                                    "realigned_timeseries",
                                                    "mask_file",
                                                    "realignment_parameters"]),
                     name="inputs")

    # Use RapidART to detect motion/intensity outliers
    art = MapNode(ra.ArtifactDetect(use_differences=[True, False],
                                    use_norm=True,
                                    zintensity_threshold=3,
                                    norm_threshold=1,
                                    parameter_source="FSL",
                                    mask_type="file"),
                  iterfield=["realignment_parameters",
                             "realigned_files",
                             "mask_file"],
                  name="art")

    # Plot a timeseries of the global mean intensity
    art_plot_inputs = ["intensity_file", "outlier_file"]
    plotmean = MapNode(util.Function(input_names=art_plot_inputs,
                                     output_names="intensity_plot",
                                     function=write_art_plot),
                       iterfield=art_plot_inputs,
                       name="plotmean")

    outliername = MapNode(util.Rename(format_string="outlier_volumes.txt"),
                          iterfield=["in_file"],
                          name="outliername")

    # Define the workflow outputs
    out_fields = ["outlier_volumes", "intensity_plot"]

    outputnode = Node(util.IdentityInterface(fields=out_fields),
                      name="outputs")

    # Define and connect the workflow
    artifact = Workflow(name=name)
    artifact.connect([
        (inputnode, art,
            [("realignment_parameters", "realignment_parameters"),
             ("realigned_timeseries", "realigned_files"),
             ("mask_file", "mask_file")]),
        (art, plotmean,
            [("intensity_files", "intensity_file"),
             ("outlier_files", "outlier_file")]),
        (art, outliername,
            [("outlier_files", "in_file")]),
        (outliername, outputnode,
            [("out_file", "outlier_volumes")]),
        (plotmean, outputnode,
            [("intensity_plot", "intensity_plot")]),
        ])

    return artifact


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

    # Set up a node to grab the target from the subjects directory
    fssource = Node(io.FreeSurferSource(
                    subjects_dir=fs.Info.subjectsdir()),
                    name="fssource")
    # Always overwrite the grab; shouldn't cascade
    fssource.overwrite = True

    # Convert the target to nifti
    convert = Node(fs.MRIConvert(out_type="niigz"), name="convertbrain")

    # Swap dimensions so stuff looks nice in the report
    flipbrain = Node(fsl.SwapDimensions(new_dims=("RL", "PA", "IS")),
                     name="flipbrain")

    flipfunc = MapNode(fsl.SwapDimensions(new_dims=("RL", "PA", "IS")),
                       iterfield=["in_file"],
                       name="flipfunc")

    # Slice up the registration
    func2anatpng = MapNode(fsl.Slicer(middle_slices=True,
                                      show_orientation=False,
                                      scaling=.6,
                                      label_slices=False),
                           iterfield=["in_file"],
                           name="func2anatpng")

    # Rename some files
    pngname = MapNode(Rename(format_string="func2anat.png"),
                      iterfield=["in_file"],
                      name="pngname")

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
        (inputnode,    fssource,     [("subject_id", "subject_id")]),
        (func2anat,    flipfunc,     [("registered_file", "in_file")]),
        (flipfunc,     func2anatpng, [("out_file", "in_file")]),
        (fssource,     convert,      [("brain", "in_file")]),
        (convert,      flipbrain,    [("out_file", "in_file")]),
        (flipbrain,    func2anatpng, [("out_file", "image_edges")]),
        (func2anatpng, pngname,      [("out_file", "in_file")]),
        (func2anat,    tkregname,    [("out_reg_file", "in_file")]),
        (func2anat,    flirtname,    [("out_fsl_file", "in_file")]),
        (func2anat,    costname,     [("min_cost_file", "in_file")]),
        (costname,     report,       [("out_file", "in1")]),
        (pngname,      report,       [("out_file", "in2")]),
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

def get_trimmed_length(in_files, frames_to_toss):
    from nibabel import load
    lengths = []
    if not isinstance(in_files, list):
        in_files = [in_files]
    for f in in_files:
        lengths.append(load(f).shape[-1] - frames_to_toss)
    return lengths


def max_motion_func(rms_files):
    """Determine the maximum absolute and relative motion values."""
    from os import getcwd
    from os.path import join
    from numpy import loadtxt, max
    motion = map(loadtxt, rms_files)
    maxima = map(max, motion)
    out_file = join(getcwd(), "max_motion.txt")
    with open(out_file, "w") as f:
        f.write("#Absolute:\n%.4f\n#Relative\n%.4f" % tuple(maxima))
    return out_file


def spline_reslice(timeseries, ref_file, aff_mats):
    """Reslice with applywarp using mcflirt affine mats."""
    from os import getcwd
    from glob import glob
    from subprocess import call
    from nipype.utils.filemanip import fname_presuffix

    # Split the timeseries by frame
    cmd = ["fslsplit", timeseries, "frame", "-t"]
    call(cmd)

    # Grab the list of 3D images
    frames = sorted(glob("frame*.nii.gz"))

    # Reslice each image with its matrix
    resliced_frames = []
    for i, mat in enumerate(aff_mats):

        out_frame = "resliced%04d.nii.gz" % i
        cmd = ["applywarp", "-i", frames[i],
               "--premat=%s" % mat, "-r", ref_file,
               "-o", out_frame, "--interp=sinc"]
        call(cmd)
        resliced_frames.append(out_frame)

    # Merge the 3D resliced images back into one file
    out_file = fname_presuffix(timeseries, suffix="_mcf", newpath=getcwd())
    cmd = ["fslmerge", "-t", out_file]
    cmd.extend(resliced_frames)
    call(cmd)

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


def write_realign_report(realign_params, rms_files):
    from os.path import abspath
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    # Set some visual defaults on the fly
    mpl.rcParams.update({'figure.figsize': (8, 2.5),
                         'figure.subplot.left': .075,
                         'figure.subplot.right': .95,
                         'font.size': 8,
                         'legend.labelspacing': .2})

    # Open up the RMS files and get the max motion
    displace = map(np.loadtxt, rms_files)
    motion = map(max, displace)
    motion.append(displace[1].sum())
    displace[1] = np.concatenate(([0], displace[1]))
    motion_file = abspath("max_motion.txt")
    with open(motion_file, "w") as f:
        f.write(
            "#Absolute:\n%.4f\n#Relative\n%.4f\nTotal\n%.4f" % tuple(motion))

    # Write the displacement plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.array(displace).T)
    ax.set_xlim((0, len(displace[0]) - 1))
    ax.legend(['abs', 'rel'], ncol=2)
    ax.set_title('MCFLIRT estimated mean displacement (mm)')

    disp_plot = abspath("displacement_plot.png")
    plt.savefig(disp_plot)
    plt.close()

    # Open up the realignment parameters
    params = np.loadtxt(realign_params)
    xyz = ['x', 'y', 'z']

    # Write the rotation plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(params[:, :3])
    ax.plot(ax.get_xlim(), (0, 0), "k--")
    ax.set_xlim((0, params.shape[0] - 1))
    ax.legend(xyz, ncol=3)
    ax.set_title('MCFLIRT estimated rotations (rad)')

    rot_plot = abspath("rotation_plot.png")
    plt.savefig(rot_plot)
    plt.close()

    # Write the translation plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(params[:, 3:])
    ax.set_xlim((0, params.shape[0] - 1))
    ax.plot(ax.get_xlim(), (0, 0), "k--")
    ax.legend(xyz, ncol=3)
    ax.set_title('MCFLIRT estimated translations (mm)')

    trans_plot = abspath("translation_plot.png")
    plt.savefig(trans_plot)

    plt.close()

    return motion_file, disp_plot, rot_plot, trans_plot


def write_art_plot(intensity_file, outlier_file):
    from os.path import abspath
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    # Set some visual defaults on the fly
    mpl.rcParams.update({'figure.figsize': (8, 2.5),
                         'figure.subplot.left': .075,
                         'figure.subplot.right': .95,
                         'font.size': 8,
                         'legend.labelspacing': .2})

    # Build the figure
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Plot the global timecourse
    global_intensity = np.loadtxt(intensity_file)
    ax.plot(global_intensity)
    ax.set_xlim((0, len(global_intensity)))

    # Plot the mean intensity value
    mean = global_intensity.mean()
    ax.plot(ax.get_xlim(), (mean, mean), "k--")

    # Find the high and low intensity outlier thresh and plot
    # Could add thresh as parameter -- assume 3 for now
    std = global_intensity.std()
    high = mean + (3 * std)
    low = mean - (3 * std)
    ax.plot(ax.get_xlim(), [high, high], "k--")
    ax.plot(ax.get_xlim(), [low, low], "k--")

    ax.set_ylim((min(global_intensity.min(), (low - 1)),
                 max(global_intensity.max(), (high + 1))))

    # Plot the outliers
    try:
        outliers = np.loadtxt(outlier_file, ndmin=1)
        for out in outliers:
            ax.axvline(out, color="r")
    except IOError:
        pass

    # Title and write the plot
    ax.set_title("Global mean timecourse and ART outliers")
    intensity_plot = abspath("intensity_plot.png")
    plt.savefig(intensity_plot)
    plt.close()
    return intensity_plot


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
    motion_info = [l.strip() for l in open(motion_file).readlines()]
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
                       example_func_slices=realign_report[4],
                       mean_func_slices=mean_func_slices,
                       intensity_plot=intensity_plot,
                       n_outliers=len(open(outlier_volumes).readlines()),
                       max_abs_motion=max_abs_motion,
                       max_rel_motion=max_rel_motion,
                       total_motion=total_motion,
                       displacement_plot=realign_report[2],
                       rotation_plot=realign_report[1],
                       translation_plot=realign_report[3],
                       func_to_anat_cost=open(
                           coreg_report[0]).read().split()[0],
                       func_to_anat_slices=coreg_report[1])

    # Write the reports (this is sterotyped for workflows from here
    out_files = write_workflow_report("preproc",
                                      preproc_report_template,
                                      report_dict)

    # Return both report files as a list
    return out_files


def scale_timeseries(in_file, mask, statistic="median", target=10000):

    import os.path as op
    import numpy as np
    import nibabel as nib
    from nipype.utils.filemanip import split_filename

    ts_img = nib.load(in_file)
    ts_data = ts_img.get_data()
    mask = nib.load(mask).get_data().astype(bool)

    if statistic == "median":
        stat_value = np.median(ts_data[mask])
    elif statistic == "mean":
        stat_value = np.mean(ts_data[mask])
    else:
        raise ValueError("Statistic must be either 'mean' or 'median'")

    scale_value = float(target) / stat_value
    scaled_ts = ts_data * scale_value
    scaled_img = nib.Nifti1Image(scaled_ts,
                                 ts_img.get_affine(),
                                 ts_img.get_header())

    pth, fname, ext = split_filename(in_file)
    out_file = op.abspath(fname + "_scaled.nii.gz")
    scaled_img.to_filename(out_file)

    return out_file


# ----------------
# Helper functions
# ----------------


def get_middle_volume(func):
    """Return the middle volume index."""
    from nibabel import load
    return [(load(f).get_shape()[3] / 2) - 1 for f in func]


def get_thresh_op(thresh):
    """Return an fslmaths op string to get 10% of the intensity"""
    return "-thr %.10f -Tmin -bin" % (0.1 * thresh[0][1])


def get_scale_value(medianvals):
    """Get the scale value to set the grand mean of the timeseries ~10000."""
    return [10000. / val for val in medianvals]


def get_bright_thresh(medianvals):
    """Get the brightness threshold for SUSAN."""
    return [0.75 * val for val in medianvals]


def get_usans(inlist):
    """Return the usans at the right threshold."""
    return [[tuple([val[0], 0.75 * val[1]])] for val in inlist]


def divide_by_two(x):
    return float(x) / 2
