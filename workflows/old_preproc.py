"""
Contains an FSL/Freesurfer preprocessing workflow. 
See the docstring for get_workflow() for more information.

"""
from warnings import warn

import nipype.interfaces.fsl as fsl
import nipype.interfaces.freesurfer as fs
import nipype.interfaces.utility as util
import nipype.interfaces.io as nio
import nipype.pipeline.engine as pe    
import nipype.algorithms.rapidart as ra

from .interfaces import TimeSeriesMovie
from .utility import OutputConnector

def get_preproc_workflow(name="preproc", anat_reg=True, mcflirt_sinc_search=True, b0_unwarp=False):
    """Return a preprocessing workflow.

    The workflow primarily uses FSL, and mostly replicates FEAT preprocessing. By default,
    it uses Freesurfer's bbregister program to generate a mapping between functional space
    and Freesufer anatomical space, but that can be turned off at the function-call level.
    It also will offer optional support for B0 unwarping using fieldmaps, but that is not
    yet fully implemented.

    If mcflirt_sinc_search is set to True (as it is by default), the MCFLIRT motion-correction
    algorithm will run with four stages of searching, the final stage using sinc interpolation.
    Although this will lead to more accurate motion-correction, it will take considerably
    more time and computational resources. Set as false to use MCFLIRT with the default search
    schedule. 

    Input spec node always takes these three inputs:
        - Timeseries (image files)
        - Highpass filter cutoff (in TRs)
        - FWHM of smoothing kernel for SUSAN (in mms)
 
    If anatomical registration is set to True (as it is by default), the following input must be added
        - Freesurfer Subject ID

    If B0 unwarping is set to True (false by default), the following inputs must be added
        - Voxel shift map
        - Forward-warped magnitude volume

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

    If anatomical registration is performed, the following are added:
        - Tkregister-style affine matrix
        - FSL-style affine matrix
        - Sliced png summarizing the functional to anatomical transform
        - Optimization cost file quantitatively summarizing the transformation
    """

    if b0_unwarp:
        warn("B0 unwarping not actually implemented.\n\n")

    preproc = pe.Workflow(name=name)

    # Define the inputs for the preprocessing workflow
    input_fields = ["timeseries",
                    "hpf_cutoff",
                    "smooth_fwhm"]
    if anat_reg:
        input_fields.extend(["subject_id"])
    if b0_unwarp:
        input_fields.extend(["voxel_shift_map",
                             "fieldmap_brain"])

    inputnode = pe.Node(util.IdentityInterface(fields=input_fields),
                        name="inputspec")

    # Convert functional images to float representation
    img2float = pe.MapNode(fsl.ChangeDataType(output_datatype="float"),
                           iterfield=["in_file"],
                           name="img2float")

    # Get the middle volume of each run for motion correction 
    extractref = pe.MapNode(fsl.ExtractROI(t_size=1),
                            iterfield=["in_file", "t_min"],
                            name = "extractref")

    # Slice the example func for reporting
    exampleslice = pe.MapNode(fsl.Slicer(image_width = 572,
                                         label_slices = False),
                              iterfield=["in_file"],
                              name="exampleslice")
    exampleslice.inputs.sample_axial=2

    # Motion correct to middle volume of each run
    realign =  pe.MapNode(fsl.MCFLIRT(save_mats = True,
                                      save_plots = True,
                                      save_rms = True),
                          name="realign",
                          iterfield = ["in_file", "ref_file"])
    if mcflirt_sinc_search:
        realign.inputs.stages=4

    # Plot the rotations, translations, and displacement parameters from MCFLIRT
    plotrot = pe.MapNode(fsl.PlotMotionParams(in_source="fsl",
                                              plot_type="rotations"),
                         name="plotrotation", 
                         iterfield=["in_file"])

    plottrans = pe.MapNode(fsl.PlotMotionParams(in_source="fsl",
                                                plot_type="translations"),
                           name="plottranslation", 
                           iterfield=["in_file"])

    plotdisp = pe.MapNode(fsl.PlotMotionParams(in_source="fsl",
                                               plot_type="displacement"),
                          name="plotdisplacement",
                          iterfield=["in_file"])

    maxmotion = pe.MapNode(util.Function(input_names=["rms_files"],
                                         output_names=["out_file"],
                                         function=max_motion_func),
                           iterfield=["rms_files"],
                           name="maxmotion")

    # Get a mean image of the realigned timeseries
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

    # Connect the nodes for the first stage of preprocessing
    preproc.connect([
        (inputnode,  img2float,     [("timeseries", "in_file")]),
        (img2float,  extractref,    [("out_file", "in_file"), 
                                    (("out_file", get_middle_volume), "t_min")]),
        (extractref, exampleslice,  [("roi_file", "in_file")]),
        (img2float,  realign,       [("out_file", "in_file")]),
        (extractref, realign,       [("roi_file", "ref_file")]),
        (realign,    plotrot,       [("par_file", "in_file")]),
        (realign,    plottrans,     [("par_file", "in_file")]),
        (realign,    plotdisp,      [("rms_files","in_file")]),
        (realign,    maxmotion,     [("rms_files","rms_files")]),
        (realign,    meanfunc1,     [("out_file", "in_file")]),
        (meanfunc1,  stripmean,     [("out_file", "in_file")]),
        (realign,    maskfunc1,     [("out_file", "in_file")]),
        (stripmean,  maskfunc1,     [("mask_file", "mask_file")]),
        ])

    # Optional B0 Inhomogeneity Unwarping
    # -----------------------------------
    if b0_unwarp:

        # Use the mask to strip the example func
        maskexample = pe.MapNode(fsl.ApplyMask(),
                                 iterfield=["in_file", "mask_file"],
                                 name = "maskexample")

        # Register the fieldmap magnitude volume to the example func
        regfieldmap = pe.MapNode(fsl.FLIRT(bins=256,
                                           cost="corratio",
                                           searchr_x=[-20, 20],
                                           searchr_y=[-20, 20],
                                           searchr_z=[-20, 20],
                                           dof=6,
                                           interp="trilinear"),
                                 iterfield=["reference"],
                                 name="regfieldmap")

        # Resample the voxel-shift map to match the timeseries
        resampvsm = pe.MapNode(fsl.ApplyXfm(apply_xfm=True,
                                            interp="trilinear"),
                               iterfield=["reference", "in_matrix_file"],
                               name="resamplevsm")

        # Dewarp the example func
        dewarpexfunc = pe.MapNode(fsl.FUGUE(),
                                  iterfield=["in_file", "shift_in_file", "mask_file"],
                                  name="dewarpexfunc")

        # Dewarp the timeseries
        dewarpts = pe.MapNode(fsl.FUGUE(),
                              iterfield=["in_file", "shift_in_file", "mask_file"],
                              name="dewarptimeseries")


        # Connect up the dewarping stages
        preproc.connect([
            (extractref,   maskexample,  [("roi_file", "in_file")]),
            (stripmean,    maskexample,  [("mask_file", "mask_file")]),
            (inputnode,    regfieldmap,  [("fieldmap_brain", "in_file")]),
            (maskexample,  regfieldmap,  [("out_file", "reference")]),
            (regfieldmap,  resampvsm,    [("out_matrix_file", "in_matrix_file")]),
            (inputnode,    resampvsm,    [("voxel_shift_map", "in_file")]),
            (maskexample,  resampvsm,    [("out_file", "reference")]),
            (resampvsm,    dewarpexfunc, [("out_file", "shift_in_file")]),
            (maskexample,  dewarpexfunc, [("out_file", "in_file")]),
            (stripmean,    dewarpexfunc, [("mask_file", "mask_file")]),
            (resampvsm,    dewarpts,     [("out_file", "shift_in_file")]),
            (maskfunc1,    dewarpts,     [("out_file", "in_file")]),
            (stripmean,    dewarpts,     [("mask_file", "mask_file")]),
            ])

    # Determine the 2nd and 98th percentile intensities of each run
    getthresh = pe.MapNode(fsl.ImageStats(op_string="-p 2 -p 98"),
                           iterfield = ["in_file"],
                           name="getthreshold")

    # Threshold the functional data at 10% of the 98th percentile
    threshold = pe.MapNode(fsl.ImageMaths(out_data_type="char",
                                          suffix="_thresh"),
                           iterfield = ["in_file"],
                           name="threshold")


    # Determine the median value of the functional runs using the mask
    medianval = pe.MapNode(fsl.ImageStats(op_string="-k %s -p 50"),
                           iterfield = ["in_file", "mask_file"],
                           name="medianval")

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

    # Connect the pipeline up through the new masked runs
    preproc.connect([
        (maskfunc1,  getthresh,  [("out_file", "in_file")]),
        (getthresh,  threshold,  [(("out_stat", get_thresh_op), "op_string")]),
        (maskfunc1,  threshold,  [("out_file", "in_file")]),
        (realign,    medianval,  [("out_file", "in_file")]),
        (threshold,  medianval,  [("out_file", "mask_file")]),
        (threshold,  dilatemask, [("out_file", "in_file")]),
        (realign,    maskfunc2,  [("out_file", "in_file")]),
        (dilatemask, maskfunc2,  [("out_file", "mask_file")]),
        (maskfunc2,  meanfunc2,  [("out_file", "in_file")]),
        ])

    # Use RapidART to detect motion/intensity outliers
    art = pe.MapNode(ra.ArtifactDetect(use_differences = [True, False],
                                       use_norm = True,
                                       zintensity_threshold = 3,
                                       norm_threshold = 1,
                                       parameter_source = "FSL",
                                       mask_type = "file"),
                     iterfield=["realignment_parameters","realigned_files","mask_file"],
                     name="art")

    plotmean = pe.MapNode(fsl.PlotTimeSeries(title="Global Mean Intensity"),
                          iterfield=["in_file"],
                          name="plotmean")

    # Use our very own ts_movie script to generate a movie of the timeseries
    tsmovie = pe.MapNode(TimeSeriesMovie(ref_type="mean"),
                                         iterfield=["in_file", "plot_file"],
                                         name="tsmovie")

    # Make connections to ART and movie
    preproc.connect([
        (realign,    art,      [("par_file", "realignment_parameters")]),
        (maskfunc2,  art,      [("out_file", "realigned_files")]),
        (dilatemask, art,      [("out_file", "mask_file")]),
        (art,        plotmean, [("intensity_files", "in_file")]),
        (inputnode,  tsmovie,  [("timeseries", "in_file")]),
        (art,        tsmovie,  [("intensity_files", "plot_file")]),
        ])

    # Scale the median value each voxel in the run to 10000
    medianscale = pe.MapNode(fsl.BinaryMaths(operation="mul"),
                             iterfield=["in_file","operand_value"],
                             name="medianscale")

    # High-pass filter the timeseries
    highpass = pe.MapNode(fsl.TemporalFilter(),
                          iterfield=["in_file"],
                          name="highpass")

    # Get a new mean image from each functional run
    meanfunc3 = pe.MapNode(fsl.MeanImage(),
                           iterfield=["in_file"],
                           name="meanfunc3")

    # Slice the example func for reporting
    meanslice = pe.MapNode(fsl.Slicer(image_width = 572,
                                      label_slices = False),
                           iterfield=["in_file"],
                           name="meanslice")
    meanslice.inputs.sample_axial = 2

    # Get the median value of the filtered timeseries
    medianval2 = pe.MapNode(fsl.ImageStats(op_string="-k %s -p 50"),
                            iterfield = ["in_file", "mask_file"],
                            name="medianval2")

    # Make connections to the intensity normalization and temporal filtering
    preproc.connect([
        (maskfunc2,   medianscale,  [("out_file", "in_file")]),
        (medianval,   medianscale,  [(("out_stat", get_scale_value), "operand_value")]),
        (inputnode,   highpass,     [(("hpf_cutoff", divide_by_two), "highpass_sigma")]),
        (medianscale, highpass,     [("out_file", "in_file")]),
        (highpass,    meanfunc3,    [("out_file", "in_file")]),
        (meanfunc3,   meanslice,    [("out_file", "in_file")]),
        (highpass,    medianval2,   [("out_file", "in_file")]),
        (threshold,   medianval2,   [("out_file", "mask_file")]),
        ])

    # Merge the median values with the mean functional images into a coupled list
    mergenode = pe.Node(util.Merge(2, axis="hstack"),
                        name="merge")

    # Smooth in the volume with SUSAN
    smooth = pe.MapNode(fsl.SUSAN(),
                        iterfield=["in_file", "brightness_threshold", "usans"],
                        name="smooth")

    # Mask the smoothed data with the dilated mask
    masksmoothfunc = pe.MapNode(fsl.ApplyMask(),
                                iterfield=["in_file","mask_file"],
                                name="masksmoothfunc")
    # Make the smoothing connections
    preproc.connect([
        (meanfunc3,  mergenode,      [("out_file", "in1")]),
        (medianval2, mergenode,      [("out_stat", "in2")]),
        (highpass,   smooth,         [("out_file", "in_file")]),
        (inputnode,  smooth,         [("smooth_fwhm", "fwhm")]),
        (medianval2, smooth,         [(("out_stat", get_bright_thresh), "brightness_threshold")]),
        (mergenode,  smooth,         [(("out", get_usans), "usans")]),
        (smooth,     masksmoothfunc, [("smoothed_file", "in_file")]),
        (dilatemask, masksmoothfunc, [("out_file", "mask_file")]),
        ])

    # Optional Anatomical Coregistration
    # ----------------------------------
    if anat_reg:
        
        # Estimate the registration to Freesurfer conformed space
        func2anat = pe.MapNode(fs.BBRegister(contrast_type="t2",
                                  init="fsl",
                                  epi_mask=True,
                                  registered_file=True,
                                  out_fsl_file=True),
                       iterfield=["source_file"],
                       name="func2anat")
   
        # Set up a node to grab the target from the subjects directory
        fssource  = pe.Node(nio.FreeSurferSource(subjects_dir=fs.Info.subjectsdir()),
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
                                             label_slices=False),
                                  iterfield=["in_file"],
                                  name="func2anatpng")
        
        # Connect the registration
        preproc.connect([
            (inputnode,    func2anat,    [("subject_id", "subject_id")]),
            (inputnode,    fssource,     [("subject_id", "subject_id")]),
            (meanfunc3,    func2anat,    [("out_file", "source_file")]),
            (func2anat,    flipfunc,     [("registered_file", "in_file")]),
            (flipfunc,     func2anatpng, [("out_file", "in_file")]),
            (fssource,     convert,      [("brain", "in_file")]),
            (convert,      flipbrain,    [("out_file", "in_file")]),
            (flipbrain,    func2anatpng, [("out_file", "image_edges")]),
            ])

    # Define the outputs of the preprocessing that will be used by the model
    output_fields = ["smoothed_timeseries",
                     "unsmoothed_timeseries",
                     "example_func",
                     "mean_func",
                     "functional_mask",
                     "realignment_parameters",
                     "timeseries_movie",
                     "example_func_slices",
                     "mean_func_slices",
                     "intensity_plot",
                     "outlier_volumes",
                     "max_motion",
                     "rotation_plot",
                     "translation_plot",
                     "displacement_plot"]
    if anat_reg:
        output_fields.extend(["func2anat", # ends up as .mincost file
                              "func2anat_flirt",
                              "func2anat_tkreg",
                              "func2anat_slices",
                              ])

    outputnode = pe.Node(util.IdentityInterface(fields=output_fields),
                         name="outputspec")

    # Use a utility class (defined in utility module) to control renaming 
    # and connections to the output node
    rename = OutputConnector(preproc, outputnode)

    rename.connect(highpass,       "unsmoothed_timeseries")
    rename.connect(masksmoothfunc, "smoothed_timeseries")
    rename.connect(extractref,     "example_func", "roi_file")
    rename.connect(meanfunc3,      "mean_func")
    rename.connect(dilatemask,     "functional_mask")
    rename.connect(realign,        "realignment_parameters", "par_file")
    rename.connect(art,            "outlier_volumes", "outlier_files")
    rename.connect(tsmovie,        "timeseries_movie")
    rename.connect(plotmean,       "intensity_plot")
    rename.connect(plotrot,        "rotation_plot")
    rename.connect(plottrans,      "translation_plot")
    rename.connect(plotdisp,       "displacement_plot")
    rename.connect(maxmotion,      "max_motion")
    rename.connect(exampleslice,   "example_func_slices")
    rename.connect(meanslice,      "mean_func_slices")

    
    if anat_reg:
        rename.connect(func2anatpng, "func2anat_slices")
        rename.connect(func2anat,    "func2anat", "min_cost_file")
        rename.connect(func2anat,    "func2anat_tkreg", "out_reg_file")
        rename.connect(func2anat,    "func2anat_flirt", "out_fsl_file")

    return preproc, inputnode, outputnode

# Connecting functions
# --------------------

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

def get_middle_volume(func):
    """Return the middle volume index."""
    from nibabel import load
    return [(load(f).get_shape()[3]/2)-1 for f in func]

def get_thresh_op(thresh):
    """Return an fslmaths op string to get10% of the intensity"""
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

