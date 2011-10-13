from nipype.algorithms import  modelgen
from nipype.interfaces import fsl
import nipype.interfaces.utility as util
import nipype.pipeline.engine as pe

from .interfaces import XCorrCoef
from .utility import OutputConnector

def get_model_workflow(name="model"):

    # Model Workflow Definition
    model = pe.Workflow(name=name)

    inputnode = pe.Node(util.IdentityInterface(fields=["subject_info",
                                                       "TR", 
                                                       "units", 
                                                       "hpf_cutoff",
                                                       "HRF_bases", 
                                                       "contrasts",
                                                       "outlier_files",
                                                       "overlay_background",
                                                       "realignment_parameters",
                                                       "timeseries"]),
                        name="inputspec")

    # Generate Nipype-style model information
    modelspec = pe.Node(modelgen.SpecifyModel(),
                        name="modelspec")

    # Generate FSL-style model information
    level1design = pe.Node(fsl.Level1Design(model_serial_correlations=True),
                           name="level1design")

    # Use model information to create fsf files
    featmodel = pe.MapNode(fsl.FEATModel(), 
                          iterfield = ["fsf_file","ev_files"],
                          name="featmodel")

    # Generate a plot of stimulus correlation
    xcorrcoef = pe.MapNode(XCorrCoef(),
                           iterfield = ["design_matrix"],
                   name="xcorrcoef")

    # Use film_gls to estimate the model
    modelestimate = pe.MapNode(fsl.FILMGLS(smooth_autocorr=True,
                                           mask_size=5,
                                           threshold=1000),
                               name="modelestimate",
                               iterfield = ["design_file","in_file"])

    # Get the robust threshold of the sigmasquareds volume
    threshres = pe.MapNode(fsl.ImageStats(op_string = "-r"),
                           name="threshresidual",
                           iterfield=["in_file"])

    # Overlay the sigmasquareds in color
    overlayres = pe.MapNode(fsl.Overlay(auto_thresh_bg=True),
                            name="overlayresidual",
                            iterfield = ["stat_image", "stat_thresh", "background_image"])

    # Slice the sigmasquareds into a png
    sliceres = pe.MapNode(fsl.Slicer(),
                          name="sliceresidual",
                          iterfield=["in_file"])

    # Estimate contrasts from the parameter effect size images
    contrastestimate = pe.MapNode(fsl.ContrastMgr(), name="contrastestimate",
                                  iterfield = ["tcon_file","dof_file", "corrections",
                                               "param_estimates", "sigmasquareds"])

    # This node will iterate over each contrast for reporting
    # (The iterables must be set elsewhere, as this workflow is 
    # agnostic to model information)
    selectcontrast = pe.MapNode(util.Select(), 
                                name="selectcontrast", 
                                iterfield=["inlist"])

    # Overlay the zstats
    overlaystats = pe.MapNode(fsl.Overlay(stat_thresh=(2.3,10),
                                          auto_thresh_bg=True,
                                          show_negative_stats=True),
                              name="overlaystats",
                              iterfield = ["stat_image","background_image"])

    # Slice the zstats for reporting
    slicestats = pe.MapNode(fsl.Slicer(),
                            name="slicestats",
                            iterfield=["in_file"])

    # Define the workflow outputs
    outputnode = pe.Node(util.IdentityInterface(fields=["results",
                                                        "design_image",
                                                        "stimulus_correlation",
                                                        "sigmasquareds",
                                                        "copes",
                                                        "varcopes",
                                                        "zstats",
                                                        "zstat"]),
                         name="outputspec")



    # Connect up the model workflow
    model.connect([
        (inputnode,         modelspec,         [("subject_info", "subject_info"),
                                                ("TR", "time_repetition"),
                                                ("units", "input_units"),
                                                ("hpf_cutoff", "high_pass_filter_cutoff"),
                                                ("timeseries", "functional_runs"),
                                                ("outlier_files", "outlier_files"),
                                                ("realignment_parameters","realignment_parameters")]),
        (inputnode,         level1design,      [("contrasts", "contrasts"),
                                                ("TR", "interscan_interval"),
                                                ("HRF_bases", "bases")]),
        (inputnode,         modelestimate,     [("timeseries", "in_file")]),
        (modelspec,         level1design,      [("session_info","session_info")]),
        (level1design,      featmodel,         [("fsf_files","fsf_file"),
                                                ("ev_files", "ev_files")]),
        (featmodel,         modelestimate,     [("design_file","design_file")]),
        (featmodel,         contrastestimate,  [("con_file","tcon_file")]),
        (featmodel,         xcorrcoef,         [("design_file", "design_matrix")]),
        (contrastestimate,  selectcontrast,    [(("zstats", con_sort),"inlist")]),
        (modelestimate,     threshres,         [("sigmasquareds","in_file")]),
        (inputnode,         overlayres,        [("overlay_background", "background_image")]),
        (modelestimate,     overlayres,        [("sigmasquareds","stat_image")]),
        (threshres,         overlayres,        [(("out_stat", make_tuple), "stat_thresh")]),
        (inputnode,         sliceres,          [(("overlay_background", get_sampling_rate), "sample_axial"),
                                                (("overlay_background", get_image_width), "image_width")]),
        (overlayres,        sliceres,          [("out_file", "in_file")]),
        (modelestimate,     contrastestimate,  [("dof_file", "dof_file"),
                                                ("corrections", "corrections"),
                                                ("param_estimates", "param_estimates"),
                                                ("sigmasquareds", "sigmasquareds")]),
        (selectcontrast,    overlaystats,      [("out","stat_image")]),
        (inputnode,         overlaystats,      [("overlay_background", "background_image")]),
        (overlaystats,      slicestats,        [("out_file", "in_file")]),
        (inputnode,         slicestats,        [(("overlay_background", get_sampling_rate), "sample_axial"),
                                                (("overlay_background", get_image_width), "image_width")]),
        (modelestimate,     outputnode,        [("results_dir", "results")]),
        (contrastestimate,  outputnode,        [("copes", "copes"),
                                                ("varcopes", "varcopes"),
                                                ("zstats", "zstats")]),
        ])

    # Use a utility class (defined in utility module) to control renaming 
    # and connections to the output node
    rename = OutputConnector(model, outputnode)
    
    rename.connect(featmodel, "design_image", "design_image")
    rename.connect(xcorrcoef, "stimulus_correlation", "corr_png")
    rename.connect(sliceres,  "sigmasquareds")
    rename.connect(slicestats, "zstat")

    return model, inputnode, outputnode


def con_sort(files):
    """Take a list, sort it, and return it."""
    files.sort()
    return files

def get_sampling_rate(bg_image):
    """Sample overlay images every 2 slices if in MNI space, otherwise show every slice."""
    import os
    if isinstance(bg_image, list):
        bg_image = bg_image[0]
    try:
        # This heurstic is not perfect, but will do for us
        if bg_image.startswith(os.environ["FSLDIR"]):
            return 2
    except KeyError:
        return 1
    return 1

def get_image_width(bg_image):
    """Set the image width of the slicer png based on what space the background image is in."""
    import os
    if isinstance(bg_image, list):
        bg_image = bg_image[0]
    try:
        # This heurstic is not perfect, but will do for us
        if bg_image.startswith(os.environ["FSLDIR"]):
            return 872
    except KeyError:
        return 750
    return 750

def make_tuple(x):
    return [tuple(i) for i in x]
