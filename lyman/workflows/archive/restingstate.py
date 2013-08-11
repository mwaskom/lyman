"""
Preprocessing for resting-state anaylsis
See the docstring for get_preproc_workflow() for more information.

"""

import nipype.interfaces.io as io
import nipype.interfaces.fsl as fsl       
import nipype.interfaces.freesurfer as fs
import nipype.interfaces.utility as util   
import nipype.pipeline.engine as pe         
import nipype.algorithms.rapidart as ra      

from nipype.workflows.fsl import create_susan_smooth

from .interfaces import TimeSeriesMovie

#---------------#
# Main Workflow #
#---------------#

def create_resting_workflow(name="resting_state"):
    """Return a preprocessing workflow.

    Input spec node takes these three inputs:
        - Timeseries (image files)
        - FWHM of smoothing kernel for (in mms)
        - FNIRT warp coefficient image
        - Freesurfer Subject ID
 
    Output node returns these files:
        - Smoothed timeseries (fully preprocessed and smoothed timeseries in native space)
        - Unsmoothed timeseries (identical steps except no smoothing in the volume)
        - Example func (target volume for MCFLIRT realignment)
        - Mean func (unsmoothed mean functional)
        - Funcational mask (binary dilated brainmask in functional space)
        - Realignment parameters (text files from MCFLIRT)
        - Outlier Files (outlier text files from ART)
        - Plotted estimated rotations from MCFLIRT
        - Plotted estimated translastion from MCFLIRT
        - Plotted estimated relative and absolute displacement from MCFLIRT
        - Plotted global mean intensity value
        - Sliced png of the example func (MCFLIRT target)
        - Sliced png of the unsmoothed mean functional volume
        - Tkregister-style affine matrix
        - FSL-style affine matrix
        - Sliced png summarizing the functional to anatomical transform
        - Optimization cost file quantitatively summarizing the transformation

    """
    resting = pe.Workflow(name=name)

    # Define the inputs for the preprocessing workflow
    inputnode = pe.Node(util.IdentityInterface(fields=["timeseries",
                                                       "subject_id",
                                                       "warpfield",
                                                       "smooth_fwhm"]),
                        name="inputspec")

    # Remove the first two frames to account for T1 stabalization
    trimmer = pe.MapNode(fsl.ExtractROI(t_min=6),
                         iterfield=["in_file"],
                         name="trimmer")

    # Convert functional images to float representation
    img2float = pe.MapNode(fsl.ChangeDataType(output_datatype="float"),
                           iterfield=["in_file"],
                           name="img2float")


    # Perform slice-timing correction
    slicetime = pe.MapNode(fsl.SliceTimer(interleaved=True,
                                          time_repetition=6),
                           iterfield=["in_file"],
                           name="slicetime")

    # Motion correct
    realign = create_realignment_workflow()

    skullstrip = create_skullstrip_workflow()

    art = create_art_workflow(make_movie=False)

    func2anat = create_bbregister_workflow()

    confounds = create_confound_removal_workflow()
    
    susan = create_susan_smooth()

    normalize = create_normalize_workflow()

    tosurf = create_surface_projection_workflow()

    rename = pe.MapNode(util.Rename(format_string="timeseries",
                                    keep_ext=True),
                        iterfield=["in_file"],
                        name="rename")
    
    resting.connect([
        (inputnode,   trimmer,       [("timeseries", "in_file"),
                                     (("timeseries", get_trimmed_length), "t_size")]),
        (trimmer,     img2float,     [("roi_file", "in_file")]),
        (img2float,   slicetime,     [("out_file", "in_file")]),
        (slicetime,   realign,       [("slice_time_corrected_file", "inputs.timeseries")]),
        (realign,     skullstrip,    [("outputs.timeseries", "inputs.timeseries")]),

        (realign,     art,           [("outputs.realign_parameters", "inputs.realignment_parameters")]),
        (img2float,   art,           [("out_file", "inputs.raw_timeseries")]),
        (skullstrip,  art,           [("outputs.timeseries", "inputs.realigned_timeseries"),
                                      ("outputs.mask_file", "inputs.mask_file")]),
        (skullstrip,  func2anat,     [("outputs.mean_func", "inputs.source_file")]),
        (inputnode,   func2anat,     [("subject_id", "inputs.subject_id")]),
        (inputnode,   confounds,     [("subject_id", "inputs.subject_id")]),
        (skullstrip,  confounds,     [("outputs.timeseries", "inputs.timeseries")]),
        (realign,     confounds,     [("outputs.realign_parameters", "inputs.motion_parameters")]),
        (func2anat,   confounds,     [("outputs.tkreg_mat", "inputs.reg_file")]),
        (confounds,   susan,         [("outputs.timeseries", "inputnode.in_files")]),
        (skullstrip,  susan,         [("outputs.mask_file", "inputnode.mask_file")]),
        (inputnode,   susan,         [("smooth_fwhm", "inputnode.fwhm")]),
        (susan,       rename,        [("outputnode.smoothed_files", "in_file")]),
        (susan,       normalize,     [("outputnode.smoothed_files", "inputs.timeseries")]),
        (inputnode,   normalize,     [("warpfield", "inputs.warpfield")]),
        (func2anat,   normalize,     [("outputs.flirt_mat", "inputs.flirt_affine")]),
        (confounds,   tosurf,        [("outputs.timeseries", "inputs.timeseries")]),
        (func2anat,   tosurf,        [("outputs.tkreg_mat", "inputs.tkreg_affine")]),
        (inputnode,   tosurf,        [("subject_id", "inputs.subject_id"),
                                      ("smooth_fwhm", "inputs.smooth_fwhm")]),
        ])

    # Define the outputs of the top-level workflow
    output_fields = ["volume_timeseries",
                     "surface_timeseries",
                     "native_timeseries",
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
                     "confound_sources"]


    outputnode = pe.Node(util.IdentityInterface(fields=output_fields),
                         name="outputspec")

    resting.connect([
        (realign,     outputnode,     [("outputs.realign_report", "realign_report"),
                                       ("outputs.realign_parameters", "realign_parameters"),
                                       ("outputs.example_func", "example_func")]),
        (skullstrip,  outputnode,     [("outputs.mean_func", "mean_func"),
                                       ("outputs.mask_file", "functional_mask"),
                                       ("outputs.report_png", "mean_func_slices")]),
        (art,         outputnode,     [("outputs.intensity_plot", "intensity_plot"),
                                       ("outputs.outlier_volumes", "outlier_volumes")]),
        (func2anat,   outputnode,     [("outputs.tkreg_mat", "tkreg_affine"),
                                       ("outputs.flirt_mat", "flirt_affine"),
                                       ("outputs.report", "coreg_report")]),
        (confounds,    outputnode,     [("outputs.confound_sources", "confound_sources")]),
        (tosurf,       outputnode,    [("outputs.timeseries", "surface_timeseries")]),
        (normalize,    outputnode,    [("outputs.timeseries", "volume_timeseries")]),
        (rename,       outputnode,    [("out_file", "native_timeseries")]),
        ])

    return resting, inputnode, outputnode

#---------------#
# Sub Workflows #
#---------------#


def create_realignment_workflow(name="realignment", interp_type="trilinear"):
    
    # Define the workflow inputs
    inputnode = pe.Node(util.IdentityInterface(fields=["timeseries"]),
                        name="inputs")

    # Get the middle volume of each run for motion correction 
    extractref = pe.MapNode(fsl.ExtractROI(t_size=1),
                             iterfield=["in_file", "t_min"],
                             name = "extractref")

    # Slice the example func for reporting
    exampleslice = pe.MapNode(fsl.Slicer(image_width=800,
                                         label_slices=False),
                              iterfield=["in_file"],
                              name="exampleslice")
    exampleslice.inputs.sample_axial=2

    # Motion correct to middle volume of each run
    mcflirt =  pe.MapNode(fsl.MCFLIRT(save_plots=True,
                                      save_rms=True,
                                      interpolation=interp_type),
                          name="mcflirt",
                          iterfield = ["in_file", "ref_file"])
  
    report_inputs = ["realign_params", "rms_files"]
    report_outputs = ["max_motion_file", "disp_plot", "rot_plot", "trans_plot"]
    mcreport = pe.MapNode(util.Function(input_names=report_inputs,
                                        output_names=report_outputs,
                                        function=write_realign_report),
                          iterfield=report_inputs,
                          name="mcreport")
                           
    # Rename some things
    exfuncname = pe.MapNode(util.Rename(format_string="example_func",
                                        keep_ext=True),
                            iterfield=["in_file"],
                            name="exfuncname")

    exslicename = pe.MapNode(util.Rename(format_string="example_func",
                                         keep_ext=True),
                             iterfield=["in_file"],
                             name="exslicename")

    parname = pe.MapNode(util.Rename(format_string="realignment_parameters.par"),
                         iterfield=["in_file"],
                         name="parname")

    # Send out all the report data as one list
    mergereport = pe.Node(util.Merge(numinputs=5, axis="hstack"),
                          name="mergereport")   

    # Define the outputs
    outputnode = pe.Node(util.IdentityInterface(fields=["timeseries",
                                                        "example_func",
                                                        "realign_parameters",
                                                        "realign_report"]),
                         name="outputs")
    
    # Define and connect the sub workflow
    realignment = pe.Workflow(name=name)

    realignment.connect([
        (inputnode,     extractref,       [("timeseries", "in_file"), 
                                           (("timeseries", get_middle_volume), "t_min")]),
        (extractref,    exampleslice,     [("roi_file", "in_file")]),
        (inputnode,     mcflirt,          [("timeseries", "in_file")]),
        (extractref,    mcflirt,          [("roi_file", "ref_file")]),
        (mcflirt,       mcreport,         [("par_file", "realign_params"),
                                           ("rms_files", "rms_files")]),
        (exampleslice,  exslicename,      [("out_file", "in_file")]),
        (mcreport,      mergereport,      [("max_motion_file", "in1"),
                                           ("rot_plot", "in2"),
                                           ("disp_plot", "in3"),
                                           ("trans_plot", "in4")]),
        (exslicename,   mergereport,      [("out_file", "in5")]),
        (mcflirt,       parname,          [("par_file", "in_file")]),
        (parname,       outputnode,       [("out_file", "realign_parameters")]),
        (extractref,    exfuncname,       [("roi_file", "in_file")]),
        (mcflirt,       outputnode,       [("out_file", "timeseries")]),
        (exfuncname,    outputnode,       [("out_file", "example_func")]),  
        (mergereport,   outputnode,       [("out", "realign_report")]),
        ])

    return realignment


def create_skullstrip_workflow(name="skullstrip"):

    # Define the workflow inputs
    inputnode = pe.Node(util.IdentityInterface(fields=["timeseries"]),
                        name="inputs")
    
    # Mean the timeseries across the fourth dimension
    meanfunc1 = pe.MapNode(fsl.MeanImage(),
                           iterfield=["in_file"],
                           name="meanfunc1")

    # Skullstrip the mean functional image
    stripmean = pe.MapNode(fsl.BET(mask = True,
                                   no_output=True,
                                   frac = 0.3),
                           iterfield = ["in_file"],
                           name = "stripmean")

    # Use the mask from skullstripping to strip each timeseries
    maskfunc1 = pe.MapNode(fsl.ApplyMask(),
                           iterfield=["in_file", "mask_file"],
                           name = "maskfunc1")

    # Determine the 2nd and 98th percentile intensities of each run
    getthresh = pe.MapNode(fsl.ImageStats(op_string="-p 2 -p 98"),
                           iterfield = ["in_file"],
                           name="getthreshold")

    # Threshold functional data at 10% of the 98th percentile
    threshold = pe.MapNode(fsl.ImageMaths(out_data_type="char",
                                          suffix="_thresh"),
                           iterfield = ["in_file"],
                           name="threshold")

    # Dilate the mask
    dilatemask = pe.MapNode(fsl.DilateImage(operation="max"),
                            iterfield=["in_file"],
                            name="dilatemask")

    # Mask the runs again with this new mask
    maskfunc2 = pe.MapNode(fsl.ApplyMask(),
                           iterfield=["in_file", "mask_file"],
                           name="maskfunc2")

    # Get a new mean image from each functional run
    meanfunc2 = pe.MapNode(fsl.MeanImage(),
                           iterfield=["in_file"],
                           name="meanfunc2")

    # Slice the mean func for reporting
    meanslice = pe.MapNode(fsl.Slicer(image_width = 800,
                                      label_slices = False),
                           iterfield=["in_file", "image_edges"],
                           name="meanslice")
    meanslice.inputs.sample_axial = 2

    # Rename the outputs
    meanname = pe.MapNode(util.Rename(format_string="mean_func",
                                      keep_ext=True),
                          iterfield=["in_file"],
                          name="meanname")

    maskname = pe.MapNode(util.Rename(format_string="functional_mask",
                                      keep_ext=True),
                          iterfield=["in_file"],
                          name="maskname")

    pngname = pe.MapNode(util.Rename(format_string="mean_func.png"),
                         iterfield=["in_file"],
                         name="pngname")

    # Define the workflow outputs
    outputnode = pe.Node(util.IdentityInterface(fields=["timeseries",
                                                        "mean_func",
                                                        "mask_file",
                                                        "report_png"]),
                         name="outputs")

    # Define and connect the workflow
    skullstrip = pe.Workflow(name=name)

    skullstrip.connect([
        (inputnode,  meanfunc1,     [("timeseries", "in_file")]),
        (meanfunc1,  stripmean,     [("out_file", "in_file")]),
        (inputnode,  maskfunc1,     [("timeseries", "in_file")]),
        (stripmean,  maskfunc1,     [("mask_file", "mask_file")]),
        (maskfunc1,  getthresh,     [("out_file", "in_file")]),
        (getthresh,  threshold,     [(("out_stat", get_thresh_op), "op_string")]),
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


def create_art_workflow(name="art", make_movie=True):

    # Define the workflow inputs
    inputnode = pe.Node(util.IdentityInterface(fields=["raw_timeseries",
                                                       "realigned_timeseries",
                                                       "mask_file",
                                                       "realignment_parameters"]),
                        name="inputs")

    # Use RapidART to detect motion/intensity outliers
    art = pe.MapNode(ra.ArtifactDetect(use_differences=[True, False],
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
    plotmean = pe.MapNode(util.Function(input_names=art_plot_inputs,
                                        output_names="intensity_plot",
                                        function=write_art_plot),
                          iterfield=art_plot_inputs,
                          name="plotmean")

    # Use our very own ts_movie script to generate a movie of the timeseries
    if make_movie:
        tsmovie = pe.MapNode(TimeSeriesMovie(ref_type="mean"),
                                             iterfield=["in_file", "plot_file"],
                                             name="tsmovie")
        
        # Rename the outputs
        moviename = pe.MapNode(util.Rename(format_string="timeseries_movie.gif"),
                               iterfield=["in_file"],
                               name="moviename")

    outliername = pe.MapNode(util.Rename(format_string="outlier_volumes.txt"),
                             iterfield=["in_file"],
                             name="outliername")

    # Define the workflow outputs
    out_fields = ["outlier_volumes", "intensity_plot"]
    if make_movie:
        out_fields.append("timeseries_movie")

    outputnode = pe.Node(util.IdentityInterface(fields=out_fields),
                         name="outputs")

    # Define and connect the workflow
    artifact = pe.Workflow(name=name)
    artifact.connect([
        (inputnode,    art,          [("realignment_parameters", "realignment_parameters"),
                                      ("realigned_timeseries", "realigned_files"),
                                      ("mask_file", "mask_file")]),
        (art,          plotmean,     [("intensity_files", "intensity_file"),
                                      ("outlier_files", "outlier_file")]),   
        (art,          outliername,  [("outlier_files" ,"in_file")]),
        (outliername,  outputnode,   [("out_file", "outlier_volumes")]),
        (plotmean,     outputnode,   [("intensity_plot", "intensity_plot")]),
        ])

    if make_movie:
        artifact.connect([
            (inputnode,    tsmovie,      [("raw_timeseries", "in_file")]),
            (art,          tsmovie,      [("intensity_files", "plot_file")]),
            (tsmovie,      moviename,    [("out_file", "in_file")]),
            (moviename,    outputnode,   [("out_file", "timeseries_movie")]),
            ])


    return artifact

def create_bbregister_workflow(name="bbregister", contrast_type="t2"):

    # Define the workflow inputs
    inputnode = pe.Node(util.IdentityInterface(fields=["subject_id",
                                                       "source_file"]),
                        name="inputs")
    
    # Estimate the registration to Freesurfer conformed space
    func2anat = pe.MapNode(fs.BBRegister(contrast_type=contrast_type,
                                         init="fsl",
                                         epi_mask=True,
                                         registered_file=True,
                                         out_fsl_file=True),
                           iterfield=["source_file"],
                           name="func2anat")

    # Set up a node to grab the target from the subjects directory
    fssource  = pe.Node(io.FreeSurferSource(subjects_dir=fs.Info.subjectsdir()),
                        name="fssource") 
    # Always overwrite the grab; shouldn't cascade unless the underlying image changes
    fssource.overwrite = True
    
    # Convert the target to nifti
    convert = pe.Node(fs.MRIConvert(out_type="niigz"), name="convertbrain")

    # Swap dimensions so stuff looks nice in the report
    flipbrain = pe.Node(fsl.SwapDimensions(new_dims=("RL","PA","IS")),
                        name="flipbrain")

    flipfunc = pe.MapNode(fsl.SwapDimensions(new_dims=("RL","PA","IS")),
                          iterfield=["in_file"],
                          name="flipfunc")

    # Slice up the registration
    func2anatpng = pe.MapNode(fsl.Slicer(middle_slices=True,
                                         show_orientation=False,
                                         scaling=.6,
                                         label_slices=False),
                              iterfield=["in_file"],
                              name="func2anatpng")

    # Rename some files
    pngname = pe.MapNode(util.Rename(format_string="func2anat.png"),
                         iterfield=["in_file"],
                         name="pngname")

    costname = pe.MapNode(util.Rename(format_string="func2anat_cost.dat"),
                          iterfield=["in_file"],
                          name="costname")

    tkregname = pe.MapNode(util.Rename(format_string="func2anat_tkreg.dat"),
                           iterfield=["in_file"],
                           name="tkregname")

    flirtname = pe.MapNode(util.Rename(format_string="func2anat_flirt.mat"),
                           iterfield=["in_file"],
                           name="flirtname")

    # Merge the slicer png and cost file into a report list
    report = pe.Node(util.Merge(2, axis="hstack"),
                     name="report")

    # Define the workflow outputs
    outputnode = pe.Node(util.IdentityInterface(fields=["tkreg_mat",
                                                        "flirt_mat",
                                                        "report"]),
                         name="outputs")
    
    bbregister = pe.Workflow(name=name)

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

    return bbregister

def create_confound_removal_workflow(workflow_name="confound_removal"):

    inputnode = pe.Node(util.IdentityInterface(fields=["subject_id",
                                                       "timeseries",
                                                       "reg_file",
                                                       "motion_parameters"]),
                        name="inputs")

    # Get the Freesurfer aseg volume from the Subjects Directory
    getaseg = pe.Node(io.FreeSurferSource(subjects_dir=fs.Info.subjectsdir()),
                      name="getaseg")

    # Binarize the Aseg to use as a whole brain mask
    asegmask = pe.Node(fs.Binarize(min=0.5,
                                   dilate=2),
                       name="asegmask")

    # Extract and erode a mask of the deep cerebral white matter
    extractwm = pe.Node(fs.Binarize(match=[2, 41], 
                                    erode=3),
                        name="extractwm")

    # Extract and erode a mask of the ventricles and CSF
    extractcsf = pe.Node(fs.Binarize(match=[4, 5, 14, 15, 24, 31, 43, 44, 63],
                                     erode=1),
                         name="extractcsf")

    # Mean the timeseries across the fourth dimension
    meanfunc = pe.MapNode(fsl.MeanImage(),
                          iterfield=["in_file"],
                          name="meanfunc")

    # Invert the anatomical coregistration and resample the masks
    regwm = pe.MapNode(fs.ApplyVolTransform(inverse=True,
                                            interp="nearest"),
                       iterfield=["source_file", "reg_file"],
                       name="regwm")

    regcsf = pe.MapNode(fs.ApplyVolTransform(inverse=True,
                                             interp="nearest"),
                        iterfield=["source_file", "reg_file"],
                        name="regcsf")

    regbrain = pe.MapNode(fs.ApplyVolTransform(inverse=True,
                                             interp="nearest"),
                        iterfield=["source_file", "reg_file"],
                        name="regbrain")

    # Convert to Nifti for FSL tools
    convertwm = pe.MapNode(fs.MRIConvert(out_type="niigz"),
                           iterfield=["in_file"],
                           name="convertwm")

    convertcsf = pe.MapNode(fs.MRIConvert(out_type="niigz"),
                            iterfield=["in_file"],
                            name="convertcsf")
    
    convertbrain= pe.MapNode(fs.MRIConvert(out_type="niigz"),
                            iterfield=["in_file"],
                            name="convertbrain")
    
    # Add the mask images together for a report image
    addconfmasks = pe.MapNode(fsl.ImageMaths(suffix="conf", 
                                             op_string="-mul 2 -add",
                                             out_data_type="char"),
                              iterfield=["in_file", "in_file2"],
                              name="addconfmasks")

    # Overlay and slice the confound mask overlaied on mean func for reporting
    confoverlay = pe.MapNode(fsl.Overlay(auto_thresh_bg=True,
                                         stat_thresh=(.7, 2)),
                             iterfield=["background_image", "stat_image"],
                             name="confoverlay")

    confslice = pe.MapNode(fsl.Slicer(image_width = 800,
                                      label_slices = False),
                           iterfield=["in_file"],
                           name="confslice")
    confslice.inputs.sample_axial = 2

    # Extract the mean signal from white matter and CSF masks
    wmtcourse = pe.MapNode(fs.SegStats(exclude_id=0, avgwf_txt_file=True),
                           iterfield=["segmentation_file", "in_file"],
                           name="wmtcourse")

    csftcourse = pe.MapNode(fs.SegStats(exclude_id=0, avgwf_txt_file=True),
                            iterfield=["segmentation_file", "in_file"],
                            name="csftcourse")

    # Extract the mean signal from over the whole brain
    globaltcourse = pe.MapNode(fs.SegStats(exclude_id=0, avgwf_txt_file=True),
                               iterfield=["segmentation_file", "in_file"],
                               name="globaltcourse")

    # Build the confound design matrix
    conf_inputs = ["motion_params", "global_waveform", "wm_waveform", "csf_waveform"]
    confmatrix = pe.MapNode(util.Function(input_names=conf_inputs,
                                          output_names=["confound_matrix"],
                                          function=make_confound_matrix),
                           iterfield=conf_inputs,
                           name="confmatrix")

    # Regress the confounds out of the timeseries
    confregress = pe.MapNode(fsl.FilterRegressor(filter_all=True),
                             iterfield=["in_file", "design_file", "mask"],
                             name="confregress")

    # Rename the confound mask png
    renamepng = pe.MapNode(util.Rename(format_string="confound_sources.png"),
                           iterfield=["in_file"],
                           name="renamepng")

    # Define the outputs
    outputnode = pe.Node(util.IdentityInterface(fields=["timeseries",
                                                        "confound_sources"]),
                         name="outputs")

    # Define and connect the confound workflow
    confound = pe.Workflow(name=workflow_name)

    confound.connect([
        (inputnode,       meanfunc,        [("timeseries", "in_file")]),
        (inputnode,       getaseg,         [("subject_id", "subject_id")]),
        (getaseg,         extractwm,       [("aseg", "in_file")]),
        (getaseg,         extractcsf,      [("aseg", "in_file")]),
        (getaseg,         asegmask,        [("aseg", "in_file")]),
        (extractwm,       regwm,           [("binary_file", "target_file")]),
        (extractcsf,      regcsf,          [("binary_file", "target_file")]),
        (asegmask,        regbrain,        [("binary_file", "target_file")]),
        (meanfunc,        regwm,           [("out_file", "source_file")]),
        (meanfunc,        regcsf,          [("out_file", "source_file")]),
        (meanfunc,        regbrain,        [("out_file", "source_file")]),
        (inputnode,       regwm,           [("reg_file", "reg_file")]),
        (inputnode,       regcsf,          [("reg_file", "reg_file")]),
        (inputnode,       regbrain,        [("reg_file", "reg_file")]),
        (regwm,           convertwm,       [("transformed_file", "in_file")]),
        (regcsf,          convertcsf,      [("transformed_file", "in_file")]),
        (regbrain,        convertbrain,    [("transformed_file", "in_file")]),
        (convertwm,       addconfmasks,    [("out_file", "in_file")]),
        (convertcsf,      addconfmasks,    [("out_file", "in_file2")]),
        (addconfmasks,    confoverlay,     [("out_file", "stat_image")]),
        (meanfunc,        confoverlay,     [("out_file", "background_image")]),
        (confoverlay,     confslice,       [("out_file", "in_file")]),
        (confslice,       renamepng,       [("out_file", "in_file")]),
        (regwm,           wmtcourse,       [("transformed_file", "segmentation_file")]),
        (inputnode,       wmtcourse,       [("timeseries", "in_file")]),
        (regcsf,          csftcourse,      [("transformed_file", "segmentation_file")]),
        (inputnode,       csftcourse,      [("timeseries", "in_file")]),
        (regbrain,        globaltcourse,   [("transformed_file", "segmentation_file")]),
        (inputnode,       globaltcourse,   [("timeseries", "in_file")]),
        (inputnode,       confmatrix,      [("motion_parameters", "motion_params")]),
        (wmtcourse,       confmatrix,      [("avgwf_txt_file", "wm_waveform")]),
        (csftcourse,      confmatrix,      [("avgwf_txt_file", "csf_waveform")]),
        (globaltcourse,   confmatrix,      [("avgwf_txt_file", "global_waveform")]),
        (confmatrix,      confregress,     [("confound_matrix", "design_file")]),
        (inputnode,       confregress,     [("timeseries", "in_file")]),
        (convertbrain,    confregress,     [("out_file", "mask")]),
        (confregress,     outputnode,      [("out_file", "timeseries")]),
        (renamepng,       outputnode,      [("out_file", "confound_sources")]),
        ])

    return confound


def create_normalize_workflow(name="normalize"):

    # Define the workflow inputs
    inputnode = pe.Node(util.IdentityInterface(fields=["timeseries",
                                                       "flirt_affine",
                                                       "warpfield"]),
                        name="inputs")

    # Define the target space and warp to it
    mni152 = fsl.Info.standard_image("avg152T1_brain.nii.gz")

    applywarp = pe.MapNode(fsl.ApplyWarp(ref_file=mni152,
                                         interp="spline"),
                           iterfield=["in_file", "premat"],
                           name="applywarp")

    # Rename the timeseries
    rename = pe.MapNode(util.Rename(format_string="timeseries_warped",
                                    keep_ext=True),
                        iterfield=["in_file"],
                        name="rename")

    # Define the outputs
    outputnode = pe.Node(util.IdentityInterface(fields=["timeseries"]),
                         name="outputs")


    normalize = pe.Workflow(name=name)
    normalize.connect([
        (inputnode, applywarp, [("timeseries", "in_file"),
                                ("warpfield", "field_file"),
                                ("flirt_affine", "premat")]),
        (applywarp, rename,    [("out_file", "in_file")]),
        (rename,    outputnode,[("out_file", "timeseries")]),
         ])
    
    return normalize

def create_surface_projection_workflow(name="surface_projection"):

    # Define the workflow inputs
    inputnode = pe.Node(util.IdentityInterface(fields=["subject_id",
                                                       "timeseries",
                                                       "tkreg_affine",
                                                       "smooth_fwhm"]),
                        name="inputs")

    # Set up a hemisphere iterable
    hemisource = pe.Node(util.IdentityInterface(fields=["hemi"]),
                         iterables=("hemi",["lh","rh"]),
                         name="hemisource")

    # Project data onto the surface mesh
    surfproject = pe.MapNode(fs.SampleToSurface(sampling_range=(0,1,.1),
                                                sampling_units="frac",
                                                cortex_mask=True),
                             iterfield=["source_file", "reg_file"],
                             name="surfproject")
    surfproject.inputs.sampling_method="average"

    # Apply the spherical warp to the data to bring into fsaverage space
    surftransform = pe.MapNode(fs.SurfaceTransform(target_subject="fsaverage",
                                                   reshape=True),
                               iterfield=["source_file"],
                               name="surftransform")

    # Smooth the data along the surface
    smoothnormsurf = pe.MapNode(fs.SurfaceSmooth(subject_id="fsaverage",
                                                 reshape=True),
                                iterfield=["in_file"],
                                name="smoothnormsurf")

    # Convert the fsaverage surface to nifti
    cvtnormsurf = pe.MapNode(fs.MRIConvert(out_type="niigz"),
                             iterfield=["in_file"],
                             name="convertnormsurf")

    # Rename the timeseries
    rename = pe.MapNode(util.Rename(format_string="%(hemi)s.timeseries.fsaverage",
                                    keep_ext=True),
                        iterfield=["in_file"],
                        name="rename")

    # Define the outputs
    outputnode = pe.Node(util.IdentityInterface(fields=["timeseries"]),
                         name="outputs")

    
    # Define and connect the workflow
    tosurf = pe.Workflow(name=name)
    tosurf.connect([
        (inputnode,       surfproject,    [("timeseries", "source_file"),
                                           ("subject_id", "subject_id"),
                                           ("tkreg_affine", "reg_file")]),
        (hemisource,      surfproject,    [("hemi", "hemi")]),
        (surfproject,     surftransform,  [("out_file", "source_file")]),
        (inputnode,       surftransform,  [("subject_id", "source_subject")]),
        (hemisource,      surftransform,  [("hemi", "hemi")]),
        (surftransform,   smoothnormsurf, [("out_file", "in_file")]),
        (hemisource,      smoothnormsurf, [("hemi", "hemi")]),
        (inputnode,       smoothnormsurf, [("smooth_fwhm", "fwhm")]),
        (smoothnormsurf,  cvtnormsurf,    [("out_file", "in_file")]),
        (cvtnormsurf,     rename,         [("out_file", "in_file")]),
        (hemisource,      rename,         [("hemi", "hemi")]),
        (rename,          outputnode,     [("out_file", "timeseries")]),
        ])

    return tosurf


#----------------------#
# Supporting functions #
#----------------------#

def get_trimmed_length(func):
    """Return the desired length after removing two frames."""
    from nibabel import load
    funcfile = func
    if isinstance(func, list):
        funcfile = func[0]
    _,_,_,timepoints = load(funcfile).get_shape()
    return timepoints-2

def max_motion_func(rms_files):
    """Determine the maximum absolute and relative motion values."""
    from os import getcwd
    from os.path import join
    from numpy import loadtxt, max
    motion = map(loadtxt, rms_files)
    maxima = map(max, motion)
    out_file = join(getcwd(), "max_motion.txt")
    with open(out_file, "w") as f:
        f.write("#Absolute:\n%.4f\n#Relative\n%.4f"%tuple(maxima))
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
    maxima = map(max, displace)
    displace[1] = np.concatenate(([0], displace[1]))
    max_motion_file = abspath("max_motion.txt")
    with open(max_motion_file, "w") as f:
       f.write("#Absolute:\n%.4f\n#Relative\n%.4f"%tuple(maxima))

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
    ax.plot(params[:,:3])
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
    ax.plot(params[:,3:])
    ax.set_xlim((0, params.shape[0] - 1))
    ax.plot(ax.get_xlim(), (0, 0), "k--")
    ax.legend(xyz, ncol=3)
    ax.set_title('MCFLIRT estimated translations (mm)')
    
    trans_plot = abspath("translation_plot.png")
    plt.savefig(trans_plot)
    
    plt.close()
    
    return max_motion_file, disp_plot, rot_plot, trans_plot

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
                 max(global_intensity.max(),(high + 1))))

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


def make_confound_matrix(motion_params, global_waveform, wm_waveform, csf_waveform):
    """Make a confound matrix from a set of text file waveform sources"""
    import numpy as np
    from os.path import abspath

    # Load the confound source files
    wf_files = [motion_params, global_waveform, wm_waveform, csf_waveform]
    wf_arrays = map(np.loadtxt, wf_files)

    # Make sure the arrays are stackable and concatenate them
    wf_arrays[1:] = map(lambda a: a.reshape(-1, 1), wf_arrays[1:])
    confound_matrix = np.hstack(wf_arrays)

    # Demean the design matrix columns
    confound_matrix -= confound_matrix.mean(axis=0)

    # Add in temporal derivatives
    confound_derivs = np.vstack((np.zeros((1, confound_matrix.shape[1])),
                                 np.diff(confound_matrix, axis=0)))
    confound_matrix = np.hstack((confound_matrix, confound_derivs))

    # Write out the confound matrix
    out_file = abspath("confound_matrix.dat")
    np.savetxt(out_file, confound_matrix)

    return out_file

def get_middle_volume(func):
    """Return the middle volume index."""
    from nibabel import load
    return [(load(f).get_shape()[3]/2)-1 for f in func]

def get_thresh_op(thresh):
    """Return an fslmaths op string to get 10% of the intensity"""
    return "-thr %.10f -Tmin -bin"%(0.1*thresh[0][1])

def get_scale_value(medianvals):
    """Get the scale value to set the grand mean of the timeseries ~10000.""" 
    return [10000./val for val in medianvals]

def get_bright_thresh(medianvals):
    """Get the brightness threshold for SUSAN."""
    return [0.75*val for val in medianvals]

def get_usans(inlist):
    """Return the usans at the right threshold."""
    return [[tuple([val[0],0.75*val[1]])] for val in inlist]

def divide_by_two(x):
    return float(x)/2

