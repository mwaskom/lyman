from nipype.interfaces import fsl
from nipype.algorithms.modelgen import  SpecifyModel
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.pipeline.engine import Node, MapNode, Workflow


def create_timeseries_model_workflow(name="model", exp_info={}):

    inputnode = Node(IdentityInterface(fields=["subject_id",
                                               "outlier_files",
                                               "mean_func",
                                               "realignment_parameters",
                                               "timeseries"]),
                     name="inputnode")

    # Read in par files to determine model information
    modelinfo = Node(Function(input_names=["subject_id",
                                           "functional_runs",
                                           "exp_info"],
                              output_names=["subject_info"],
                              function=build_model_info),
                     name="modelinfo")
    modelinfo.inputs.exp_info = exp_info

    # Generate Nipype-style model specification
    modelspec = Node(SpecifyModel(
        time_repetition=exp_info["TR"],
        input_units=exp_info["units"],
        high_pass_filter_cutoff=exp_info["hpf_cutoff"]),
                     name="modelspec")

    # Generate FSL fsf model specifcations
    level1design = Node(fsl.Level1Design(model_serial_correlations=True,
                                         bases=exp_info["hrf_bases"],
                                         interscan_interval=exp_info["TR"],
                                         contrasts=exp_info["contrasts"]),
                        name="level1design")

    # Generate design and contrast matrix files
    featmodel = MapNode(fsl.FEATModel(),
                        iterfield=["fsf_file", "ev_files"],
                        name="featmodel")

    # Generate a plot of regressor correlation
    designcorr = MapNode(Function(input_names=["in_file"],
                                  output_names=["out_file"],
                                  function=design_corr),
                         iterfield=["in_file]"],
                         name="designcorr")

    # Use film_gls to estimate the timeseries model
    modelestimate = MapNode(fsl.FILMGLS(smooth_autocorr=True,
                                        mask_size=5,
                                        threshold=1000),
                            iterfield=["design_file", "in_file"],
                            name="modelestimate")

    # Plot the residual error map for quality control
    plotresidual = MapNode(Function(input_names=["resid_file",
                                                 "background_file"],
                                   output_names=["out_file"],
                                   function=plot_residual),
                          iterfield=["resid_file", "background_file"],
                          name="plotresidual")

    # Run the contrast estimation routine
    contrastestimate = MapNode(fsl.ContrastMgr(),
                               iterfield=["tcon_file",
                                          "dof_file",
                                          "corrections",
                                          "param_estimates",
                                          "sigmasquareds"],
                               name="contrastestimate")

    # Plot the zstat images
    plotzstats = MapNode(Function(input_names=["background_file",
                                               "zstat_files",
                                               "contrasts"],
                                  output_names=["out_dir"],
                                  function=plot_zstats),
                         iterfield=["background_file", "zstat_files"],
                         name="plotzstats")
    plotzstats.inputs.contrasts = exp_info["contrasts"]

    # Define the workflow outputs
    outputnode = Node(IdentityInterface(fields=["results",
                                                "design_image",
                                                "design_corr",
                                                "sigmasquareds",
                                                "copes",
                                                "varcopes",
                                                "zstats",
                                                "zstat_dir"]),
                      name="outputnode")

    # Define the workflow and connect the nodes
    model = Workflow(name=name)
    model.connect([
        (inputnode, modelinfo,
            [("subject_id", "subject_id"),
             ("timeseries", "functional_runs")]),
        (modelinfo, modelspec,
            [("subject_info", "subject_info")]),
        (inputnode, modelspec,
            [("timeseries", "functional_runs"),
             ("outlier_files", "outlier_files"),
             ("realignment_parameters", "realignment_parameters")]),
        (inputnode, modelestimate,
            [("timeseries", "in_file")]),
        (modelspec, level1design,
            [("session_info", "session_info")]),
        (level1design, featmodel,
            [("fsf_files", "fsf_file"),
             ("ev_files", "ev_files")]),
        (featmodel, modelestimate,
            [("design_file", "design_file")]),
        (featmodel, contrastestimate,
            [("con_file", "tcon_file")]),
        (featmodel, designcorr,
            [("design_file", "in_file")]),
        (modelestimate, plotresidual,
            [("sigmasquareds", "resid_file")]),
        (inputnode, plotresidual,
            [("mean_func", "background_file")]),
        (modelestimate, contrastestimate,
            [("dof_file", "dof_file"),
             ("corrections", "corrections"),
             ("param_estimates", "param_estimates"),
             ("sigmasquareds", "sigmasquareds")]),
        (contrastestimate, plotzstats,
            [("zstats", "zstat_files")]),
        (inputnode, plotzstats,
            [("mean_func", "background_file")]),
        (featmodel, outputnode,
            [("design_image", "design_image")]),
        (designcorr, outputnode,
            [("out_file", "design_corr")]),
        (plotresidual, outputnode,
            [("out_file", "sigmasquareds")]),
        (modelestimate, outputnode,
            [("results_dir", "results")]),
        (contrastestimate, outputnode,
            [("copes", "copes"),
             ("varcopes", "varcopes"),
             ("zstats", "zstats")]),
        (plotzstats, outputnode,
            [("out_dir", "zstat_dir")]),
        ])

    return model, inputnode, outputnode


def build_model_info(subject_id, functional_runs, exp_info):

    from copy import deepcopy
    from numpy import loadtxt
    from nipype.interfaces.base import Bunch

    events = exp_info["events"]
    regressor_names = exp_info["regressors"]

    model_info = []

    n_runs = len(functional_runs)
    for run in range(1, n_runs + 1):
        onsets, durations, amplitudes, regressors = [], [], [], []
        for event in events:
            event_info = dict(event=event, run=run, subject_id=subject_id)
            parfile = exp_info['parfile_template'] % event_info
            o, d, a = loadtxt(parfile, unpack=True)
            onsets.append(o)
            durations.append(d)
            amplitudes.append(a)
        for regressor in regressors:
            regress_info = dict(regressor=regressor,
                                run=run,
                                subject_id=subject_id)
            regressor_file = exp_info['regressor_template'] % regress_info
            regressors.append(loadtxt(regressor_file))

        model_info.append(
            Bunch(conditions=events,
                  regressor_names=regressor_names,
                  regressors=deepcopy(regressors),
                  onsets=deepcopy(onsets),
                  durations=deepcopy(durations),
                  amplitudes=deepcopy(amplitudes)))

    return model_info


def design_corr(in_file):
    from os.path import abspath
    import numpy as np
    import matplotlib.pyplot as plt

    X = np.loadtxt(in_file, skiprows=5)
    f = plt.figure(figsize=(5, 5))
    ax = f.add_subplot(111)
    ax.matshow(np.corrcoef(X.T), vmin=-1, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    out_file = abspath("design_correlation.png")
    plt.savefig(out_file)
    return out_file


def plot_residual(resid_file, background_file):
    """Plot the model residual. Use FSL for now; should move to pylab."""
    from os.path import abspath
    from nibabel import load
    from subprocess import call
    from scipy.stats import scoreatpercentile
    ov_nii = "sigmasquareds_overlay.nii.gz"
    out_file = abspath("sigmasquareds.png")
    err_img = load(resid_file)
    err_data = err_img.get_data()
    low = scoreatpercentile(err_data, 2)
    high = scoreatpercentile(err_data, 98)
    n_slice = err_img.shape[-1]
    native = n_slice < 50  # arbitrary but should work fine
    width = 750 if native else 872
    sample = 1 if native else 2
    del err_data
    call(["overlay", "1", "0", background_file, "-a",
          resid_file, low, high, ov_nii])
    call(["slicer", ov_nii, "-S", sample, width, out_file])
    return out_file


def plot_zstats(background_file, zstat_files, contrasts):
    """Plot the zstats and return a directory of pngs."""
    from os import mkdir
    from os.path import join, abspath
    from nibabel import load
    from subprocess import call
    out_dir = abspath("zstat_pngs")
    mkdir(out_dir)
    bg_img = load(background_file)
    n_slice = bg_img.shape[-1]
    native = n_slice < 50
    width = 750 if native else 872
    sample = 1 if native else 2
    for i, contrast in enumerate(contrasts):
        zstat_file = zstat_files[i]
        zstat_png = join(out_dir, contrast + ".png")
        ov_nii = contrast + ".nii.gz"
        call(["overlay", "1", "0", background_file, "-a",
              zstat_file, "2.3", "10",
              zstat_file, "-2.3", "-10", ov_nii])
        call(["slicer", ov_nii, "-S", sample, width, zstat_png])
    return out_dir
