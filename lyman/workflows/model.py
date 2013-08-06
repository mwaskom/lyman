"""Timeseries model using FSL's gaussian least squares."""
import os
import os.path as op
import re
import numpy as np
import scipy as sp
from scipy import stats, signal
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import moss
from moss import glm
import seaborn as sns

from nipype import fsl, Node, MapNode, Workflow, IdentityInterface, Function

from lyman import default_experiment_parameters

imports = ["import os",
           "import os.path as op",
           "import re",
           "import numpy as np",
           "import scipy as sp",
           "from scipy import stats, signal",
           "import pandas as pd",
           "import matplotlib as mpl",
           "import matplotlib.pyplot as plt",
           "import moss",
           "from moss import glm",
           "import seaborn as sns"]


def create_timeseries_model_workflow(name="model", exp_info=None):

    # Node inputs
    inputnode = Node(IdentityInterface(["subject_id",
                                        "design_file",
                                        "realign_file",
                                        "artifact_file",
                                        "mean_file",
                                        "timeseries"]),
                     "inputs")

    # Default experiment parameters for generating graph inamge, testing, etc.
    if exp_info is None:
        exp_info = default_experiment_parameters()

    # Set up the experimental design
    modeldesign = MapNode(Function(["exp_info",
                                    "design_file",
                                    "realign_file",
                                    "artifact_file",
                                    "run"],
                                   ["design_matrix_file",
                                    "contrast_file",
                                    "report"],
                                   design_model,
                                   imports),
                          [],
                          "modeldesign")

    # Use film_gls to estimate the timeseries model
    modelestimate = MapNode(fsl.FILMGLS(smooth_autocorr=True,
                                        mask_size=5,
                                        threshold=1000),
                            ["design_file", "in_file"],
                            "modelestimate")

    # Run the contrast estimation routine
    contrastestimate = MapNode(fsl.ContrastMgr(),
                               ["tcon_file",
                                "dof_file",
                                "corrections",
                                "param_estimates",
                                "sigmasquareds"],
                               "contrastestimate")

    # Define the workflow outputs
    outputnode = Node(IdentityInterface(["results",
                                         "design_image",
                                         "design_corr",
                                         "sigmasquareds",
                                         "copes",
                                         "varcopes",
                                         "zstats",
                                         "reports",
                                         "design_mat",
                                         "contrast_mat",
                                         "json_file",
                                         "zstat_pngs"]),
                      "outputs")

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
             ("realign_params", "realignment_parameters")]),
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
        (featmodel, rename_x_mat,
            [("design_file", "in_file")]),
        (featmodel, rename_c_mat,
            [("con_file", "in_file")]),
        (level1design, designcorr,
            [("ev_files", "ev_files")]),
        (featmodel, rename_design,
            [("design_image", "in_file")]),
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
        (inputnode, dumpjson,
            [("timeseries", "timeseries")]),
        (inputnode, report,
            [("subject_id", "subject_id")]),
        (rename_design, report,
            [("out_file", "design_image")]),
        (designcorr, report,
            [("out_file", "design_corr")]),
        (plotresidual, report,
            [("out_file", "residual")]),
        (plotzstats, report,
            [("out_files", "zstat_pngs")]),
        (report, outputnode,
            [("reports", "reports")]),
        (dumpjson, outputnode,
            [("json_file", "json_file")]),
        (rename_design, outputnode,
            [("out_file", "design_image")]),
        (rename_x_mat, outputnode,
            [("out_file", "design_mat")]),
        (rename_c_mat, outputnode,
            [("out_file", "contrast_mat")]),
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
            [("out_files", "zstat_pngs")]),
        ])

    return model, inputnode, outputnode


# Main Interface Functions
# ========================


def design_model(design_file, realign_file, artifact_file, exp_info, run):
    """Build the model design."""
    design = pd.read_csv(design_file)
    design = design[design.run == run]

    realign = pd.read_csv(realign_file)
    realign = realign.filter(regex="rot|trans").apply(stats.zscore)

    artifacts = pd.read_csv(artifact_file).max(axis=1)
    ntp = len(artifacts)
    tr = exp_info["TR"]

    # Set up the HRF model
    hrf = getattr(glm, exp_info["hrf_model"])
    hrf = hrf(exp_info["temporal_deriv"], tr, **exp_info["hrf_params"])

    # Keep tabs on the keyword arguments for the design matrix
    design_kwargs = dict(confounds=realign,
                         artifacts=artifacts,
                         tr=tr,
                         condition_names=exp_info["condition_names"],
                         confound_pca=exp_info["confound_pca"],
                         hpf_cutoff=exp_info["hpf_cutoff"])

    # Create the main design matrix object
    X = glm.DesignMatrix(design, hrf, ntp, **design_kwargs)

    # Also create one without high pass filtering
    design_kwargs["hpf_cutoff"] = None
    X_unfiltered = glm.DesignMatrix(design, hrf, ntp, **design_kwargs)

    # Now we build up the report, with lots of lovely images
    sns.set()
    design_png = op.abspath("design.png")
    X.plot(fname=design_png)

    corr_png = op.abspath("design_correlation.png")
    X.plot_confound_correlation(fname=corr_png)

    svd_png = op.abspath("design_singular_values.png")
    X.plot_singular_values(fname=svd_png)

    report = [design_png, corr_png, svd_png]

    # Now plot the information loss from the filter
    for i, (name, cols, weights) in enumerate(exp_info["contrasts"], 1):
        C = X.contrast_vector(cols, weights)

        y_filt = X.design_matrix.dot(C)
        y_unfilt = X_unfiltered.design_matrix.dot(C)

        fs, pxx_filt = signal.welch(y_filt, 1. / tr, nperseg=ntp)
        fs, pxx_unfilt = signal.welch(y_unfilt, 1. / tr, nperseg=ntp)

        f, ax = plt.subplots(1, 1)
        ax.fill_between(fs, pxx_unfilt, color="#C41E3A")
        ax.fill_between(fs, pxx_filt, color="#444444")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Spectral Density")
        plt.tight_layout()
        fname = op.abspath("cope%d_filter.png" % i)
        f.savefig(fname)
        report.append(fname)

    # Finally, write out the design files in FSL format
    design_matrix_file = op.abspath("design.mat")
    contrast_file = op.abspath("design.con")
    X.to_fsl_files("design", exp_info["contrasts"])
        
    return design_matrix_file, contrast_file, report


def build_model_info(subject_id, functional_runs, exp_info):
    import os.path as op
    from copy import deepcopy
    from numpy import loadtxt, atleast_1d, atleast_2d
    from nipype.interfaces.base import Bunch

    events = exp_info["events"]
    regressor_names = exp_info["regressors"]

    model_info = []

    if not isinstance(functional_runs, list):
        functional_runs = [functional_runs]
    n_runs = len(functional_runs)
    for run in range(1, n_runs + 1):
        onsets, durations, amplitudes, regressors = [], [], [], []
        for event in events:
            event_info = dict(event=event, run=run, subject_id=subject_id)
            parfile = op.join(exp_info["parfile_base_dir"],
                              exp_info['parfile_template'] % event_info)
            o, d, a = atleast_2d(loadtxt(parfile)).T
            onsets.append(o)
            durations.append(d)
            amplitudes.append(a)
        for regressor in regressor_names:
            regress_info = dict(regressor=regressor,
                                run=run,
                                subject_id=subject_id)
            regressor_file = op.join(
                exp_info["regressor_base_dir"],
                exp_info["regressor_template"] % regress_info)
            regressors.append(atleast_1d(loadtxt(regressor_file)))

        model_info.append(
            Bunch(conditions=events,
                  regressor_names=regressor_names,
                  regressors=deepcopy(regressors),
                  onsets=deepcopy(onsets),
                  durations=deepcopy(durations),
                  amplitudes=deepcopy(amplitudes)))

    return model_info


def design_corr(in_file, ev_files, n_runs):
    import re
    from os.path import abspath, basename
    import numpy as np
    import matplotlib.pyplot as plt
    X = np.loadtxt(in_file, skiprows=5)
    f = plt.figure(figsize=(5, 5))
    ax = f.add_subplot(111)
    ax.matshow(np.abs(np.corrcoef(X.T)), vmin=0, vmax=1, cmap="hot")
    run = int(re.match(r"run(\d+).mat", basename(in_file)).group(1))
    pat = "ev_(\w+)_%d_\d+.txt" % run
    if n_runs > 1:
        ev_files = ev_files[run]
    ev_names = [re.match(pat, basename(f)).group(1) for f in ev_files]
    ev_names = map(lambda x: x.lower(), ev_names)
    ax.set_xticks(range(len(ev_names)))
    ax.set_yticks(range(len(ev_names)))
    ax.set_xticklabels(ev_names, fontsize=6, rotation="vertical")
    ax.set_yticklabels(ev_names, fontsize=6)
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
    low = scoreatpercentile(err_data.ravel(), 2)
    high = scoreatpercentile(err_data.ravel(), 98)
    n_slice = err_img.shape[-1]
    native = n_slice < 50  # arbitrary but should work fine
    width = 750 if native else 872
    sample = 1 if native else 2
    del err_data
    call(["overlay", "1", "0", background_file, "-a",
          resid_file, str(low), str(high), ov_nii])
    call(["slicer", ov_nii, "-S", str(sample), str(width), out_file])
    return out_file


def plot_zstats(background_file, zstat_files, contrasts):
    """Plot the zstats and return a directory of pngs."""
    from os.path import abspath
    from nibabel import load
    from subprocess import call
    bg_img = load(background_file)
    n_slice = bg_img.shape[-1]
    native = n_slice < 50
    width = 750 if native else 872
    sample = 1 if native else 2
    out_files = []
    if not isinstance(zstat_files, list):
        zstat_files = [zstat_files]
    for i, contrast in enumerate(contrasts):
        zstat_file = zstat_files[i]
        zstat_png = "zstat%d.png" % (i + 1)
        ov_nii = "zstat%d_overlay.nii.gz" % (i + 1)
        call(["overlay", "1", "0", background_file, "-a",
              zstat_file, "2.3", "10",
              zstat_file, "-2.3", "-10", ov_nii])
        call(["slicer", ov_nii, "-S", str(sample), str(width), zstat_png])
        out_files.append(abspath(zstat_png))
    return out_files


def dump_exp_info(exp_info, timeseries):
    """Dump the exp_info dict into a json file."""
    from copy import deepcopy
    from os.path import abspath
    import json
    json_file = abspath("experiment_info.json")
    json_info = deepcopy(exp_info)
    del json_info["contrasts"]
    with open(json_file, "w") as fp:
        json.dump(json_info, fp, sort_keys=True, indent=4)
    return json_file


def write_model_report(subject_id, design_image, design_corr,
                       residual, zstat_pngs, contrast_names):
    """Write model report info to rst and convert to pdf/html."""
    import time
    from lyman.tools import write_workflow_report
    from lyman.workflows.reporting import model_report_template

    # Fill in the initial report template dict
    report_dict = dict(now=time.asctime(),
                       subject_id=subject_id,
                       design_image=design_image,
                       design_corr=design_corr,
                       residual=residual)

    # Add the zstat image templates and update the dict
    for i, con in enumerate(contrast_names, 1):
        report_dict["con%d_name" % i] = con
        report_dict["zstat%d_png" % i] = zstat_pngs[i - 1]
        header = "Zstat %d: %s" % (i, con)
        model_report_template = "\n".join(
            [model_report_template,
             header,
             "".join(["^" for l in header]),
             "",
             "".join([".. image:: %(zstat", str(i), "_png)s"]),
             "    :width: 6.5in",
             ""])

    out_files = write_workflow_report("model",
                                      model_report_template,
                                      report_dict)
    return out_files


# Smaller helper functions
# ========================


def run_indices(design_files):
    """Given a list of files, return a list of 1-based integers."""
    return range(len(design_files))
