"""Timeseries model using FSL's gaussian least squares."""
import os.path as op
import json
import numpy as np
from scipy import stats, signal
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import moss
from moss import glm
import seaborn as sns

from nipype import Node, MapNode, Workflow, IdentityInterface, Function
from nipype.interfaces import fsl

from lyman import default_experiment_parameters

imports = ["import os.path as op",
           "import json",
           "import numpy as np",
           "from scipy import stats, signal",
           "import pandas as pd",
           "import nibabel as nib",
           "import matplotlib.pyplot as plt",
           "import moss",
           "from moss import glm",
           "import seaborn as sns"]


def create_timeseries_model_workflow(name="model", exp_info=None):

    # Default experiment parameters for generating graph inamge, testing, etc.
    if exp_info is None:
        exp_info = default_experiment_parameters()

    # Define constant inputs
    inputs = ["design_file", "realign_file", "artifact_file", "timeseries"]

    # Possibly add the regressor file to the inputs
    if exp_info["regressor_file"] is not None:
        inputs.append("regressor_file")

    # Define the workflow inputs
    inputnode = Node(IdentityInterface(inputs), "inputs")

    # Set up the experimental design
    modelsetup = MapNode(Function(["exp_info",
                                   "design_file",
                                   "realign_file",
                                   "artifact_file",
                                   "regressor_file",
                                   "run"],
                                  ["design_matrix_file",
                                   "contrast_file",
                                   "design_matrix_pkl",
                                   "report"],
                                  setup_model,
                                  imports),
                          ["realign_file", "artifact_file", "run"],
                          "modelsetup")
    modelsetup.inputs.exp_info = exp_info
    if exp_info["regressor_file"] is None:
        modelsetup.inputs.regressor_file = None

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

    calcrsquared = MapNode(Function(["design_matrix_pkl",
                                     "timeseries",
                                     "pe_files"],
                                    ["r2_files",
                                     "ss_files"],
                                    compute_rsquareds,
                                    imports),
                           ["design_matrix_pkl",
                            "timeseries",
                            "pe_files"],
                           "calcrsquared")
    calcrsquared.plugin_args = dict(qsub_args="-l h_vmem=8G")

    # Save the experiment info for this run
    dumpjson = MapNode(Function(["exp_info", "timeseries"], ["json_file"],
                                dump_exp_info, imports),
                    "timeseries",
                    "dumpjson")
    dumpjson.inputs.exp_info = exp_info

    # Report on the results of the model
    modelreport = MapNode(Function(["timeseries",
                                    "sigmasquareds_file",
                                    "zstat_files",
                                    "r2_files"],
                                   ["report"],
                                   report_model,
                                   imports),
                          ["timeseries", "sigmasquareds_file",
                           "zstat_files", "r2_files"],
                          "modelreport")

    # Define the workflow outputs
    outputnode = Node(IdentityInterface(["results",
                                         "copes",
                                         "varcopes",
                                         "zstats",
                                         "r2_files",
                                         "ss_files",
                                         "report",
                                         "design_mat",
                                         "contrast_mat",
                                         "design_pkl",
                                         "design_report",
                                         "json_file"]),
                      "outputs")

    # Define the workflow and connect the nodes
    model = Workflow(name=name)
    model.connect([
        (inputnode, modelsetup,
            [("design_file", "design_file"),
             ("realign_file", "realign_file"),
             ("artifact_file", "artifact_file"),
             (("timeseries", run_indices), "run")]),
        (inputnode, modelestimate,
            [("timeseries", "in_file")]),
        (inputnode, dumpjson,
            [("timeseries", "timeseries")]),
        (modelsetup, modelestimate,
            [("design_matrix_file", "design_file")]),
        (modelestimate, contrastestimate,
            [("dof_file", "dof_file"),
             ("corrections", "corrections"),
             ("param_estimates", "param_estimates"),
             ("sigmasquareds", "sigmasquareds")]),
        (modelsetup, contrastestimate,
            [("contrast_file", "tcon_file")]),
        (modelsetup, calcrsquared,
            [("design_matrix_pkl", "design_matrix_pkl")]),
        (inputnode, calcrsquared,
            [("timeseries", "timeseries")]),
        (modelestimate, calcrsquared,
            [("param_estimates", "pe_files")]),
        (inputnode, modelreport,
            [("timeseries", "timeseries")]),
        (modelestimate, modelreport,
            [("sigmasquareds", "sigmasquareds_file")]),
        (contrastestimate, modelreport,
            [("zstats", "zstat_files")]),
        (calcrsquared, modelreport,
            [("r2_files", "r2_files")]),
        (modelsetup, outputnode,
            [("design_matrix_file", "design_mat"),
             ("contrast_file", "contrast_mat"),
             ("design_matrix_pkl", "design_pkl"),
             ("report", "design_report")]),
        (dumpjson, outputnode,
            [("json_file", "json_file")]),
        (modelestimate, outputnode,
            [("results_dir", "results")]),
        (contrastestimate, outputnode,
            [("copes", "copes"),
             ("varcopes", "varcopes"),
             ("zstats", "zstats")]),
        (calcrsquared, outputnode,
            [("r2_files", "r2_files"),
             ("ss_files", "ss_files")]),
        (modelreport, outputnode,
            [("report", "report")]),
        ])

    if exp_info["regressor_file"] is not None:
        model.connect([
            (inputnode, modelsetup,
                [("regressor_file", "regressor_file")])
                       ])

    return model, inputnode, outputnode


# Main Interface Functions
# ========================


def setup_model(design_file, realign_file, artifact_file, regressor_file,
                exp_info, run):
    """Build the model design."""
    design = pd.read_csv(design_file)
    design = design[design["run"] == run]

    realign = pd.read_csv(realign_file)
    realign = realign.filter(regex="rot|trans").apply(stats.zscore)

    artifacts = pd.read_csv(artifact_file).max(axis=1)
    ntp = len(artifacts)
    tr = exp_info["TR"]

    if regressor_file is None:
        regressors = None
    else:
        regressors = pd.read_csv(regressor_file)
        regressors = regressors[regressors["run"] == run].drop("run", axis=1)
        if exp_info["regressor_names"] is not None:
            regressors = regressors[exp_info["regressor_names"]]
        regressors.index = np.arange(ntp) * tr

    # Set up the HRF model
    hrf = getattr(glm, exp_info["hrf_model"])
    hrf = hrf(exp_info["temporal_deriv"], tr, **exp_info["hrf_params"])

    # Keep tabs on the keyword arguments for the design matrix
    design_kwargs = dict(confounds=realign,
                         artifacts=artifacts,
                         regressors=regressors,
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

    # Build a list of images sumarrizing the model
    report = [design_png, corr_png, svd_png]

    # Now plot the information loss from the filter
    for i, (name, cols, weights) in enumerate(exp_info["contrasts"], 1):
        C = X.contrast_vector(cols, weights)

        y_filt = X.design_matrix.dot(C)
        y_unfilt = X_unfiltered.design_matrix.dot(C)

        fs, pxx_filt = signal.welch(y_filt, 1. / tr, nperseg=ntp)
        fs, pxx_unfilt = signal.welch(y_unfilt, 1. / tr, nperseg=ntp)

        f, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.fill_between(fs, pxx_unfilt, color="#C41E3A")
        ax.axvline(1.0 / exp_info["hpf_cutoff"], c="#222222", ls=":", lw=1.5)
        ax.fill_between(fs, pxx_filt, color="#444444")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Spectral Density")
        ax.set_xlim(0, .15)
        plt.tight_layout()
        fname = op.abspath("cope%d_filter.png" % i)
        f.savefig(fname, dpi=100)
        plt.close(f)
        report.append(fname)

    # Write out the X object as a pkl to pass to the report function
    design_matrix_pkl = op.abspath("design.pkl")
    X.to_pickle(design_matrix_pkl)

    # Finally, write out the design files in FSL format
    design_matrix_file = op.abspath("design.mat")
    contrast_file = op.abspath("design.con")
    X.to_fsl_files("design", exp_info["contrasts"])

    # Close the open figures
    plt.close("all")

    return design_matrix_file, contrast_file, design_matrix_pkl, report


def compute_rsquareds(design_matrix_pkl, timeseries, pe_files):
    """Compute partial r2 for various parts of the design matrix."""
    X = glm.DesignMatrix.from_pickle(design_matrix_pkl)

    # Load the timeseries
    ts_img = nib.load(timeseries)
    ts_aff, ts_header = ts_img.get_affine(), ts_img.get_header()
    y = ts_img.get_data().reshape(-1, ts_img.shape[-1]).T
    y -= y.mean(axis=0)

    outshape = ts_img.shape[:3]

    # Load the parameter estimates
    pes = [nib.load(f).get_data()[..., np.newaxis] for f in pe_files]
    pes = np.concatenate(pes, axis=3).reshape(-1, len(pe_files)).T

    # Get the sum of squares of the data
    ybar = y.mean(axis=0)
    sstot = np.square(y - ybar).sum(axis=0)

    # Store the sum of squares
    sstot_img = nib.Nifti1Image(sstot.reshape(outshape),
                                ts_aff, ts_header)
    sstot_file = op.abspath("sstot.nii.gz")
    sstot_img.to_filename(sstot_file)

    # Now get the r2 for the full model
    yhat_full = X.design_matrix.dot(pes)
    ssres_full = np.square(yhat_full - y).sum(axis=0)
    r2_full = (1 - ssres_full / sstot).reshape(outshape)

    full_img = nib.Nifti1Image(r2_full, ts_aff, ts_header)
    full_file = op.abspath("r2_full.nii.gz")
    full_img.to_filename(full_file)

    ssres_full_img = nib.Nifti1Image(ssres_full.reshape(outshape),
                                     ts_aff, ts_header)
    ssres_full_file = op.abspath("ssres_full.nii.gz")
    ssres_full_img.to_filename(ssres_full_file)

    # Next just the "main" submatrix(conditions and regressors)
    yhat_main = X.design_matrix.dot(pes * X.main_vector)
    ssres_main = np.square(yhat_main - y).sum(axis=0)
    r2_main = (1 - ssres_main / sstot).reshape(outshape)

    main_img = nib.Nifti1Image(r2_main, ts_aff, ts_header)
    main_file = op.abspath("r2_main.nii.gz")
    main_img.to_filename(main_file)

    ssres_main_img = nib.Nifti1Image(ssres_main.reshape(outshape),
                                     ts_aff, ts_header)
    ssres_main_file = op.abspath("ssres_main.nii.gz")
    ssres_main_img.to_filename(ssres_main_file)

    # Finally the confound submatrix
    yhat_conf = X.design_matrix.dot(pes * X.confound_vector)
    ssres_conf = np.square(yhat_conf - y).sum(axis=0)
    r2_conf = (1 - ssres_conf / sstot).reshape(outshape)

    conf_img = nib.Nifti1Image(r2_conf, ts_aff, ts_header)
    conf_file = op.abspath("r2_confound.nii.gz")
    conf_img.to_filename(conf_file)

    return ([full_file, main_file, conf_file],
            [sstot_file, ssres_full_file, ssres_main_file])


def report_model(timeseries, sigmasquareds_file, zstat_files, r2_files):
    """Build the model report images, mostly from axial montages."""
    sns.set()

    # Load the timeseries, get a mean image
    ts_img = nib.load(timeseries)
    ts_aff, ts_header = ts_img.get_affine(), ts_img.get_header()
    ts_data = ts_img.get_data()
    mean_data = ts_data.mean(axis=-1)
    mlow, mhigh = 0, moss.percentiles(mean_data, 98)

    # Get the plot params. OMG I need a general function for this
    n_slices = mean_data.shape[-1]
    n_row, n_col = n_slices // 8, 8
    start = n_slices % n_col // 2
    figsize = (10, 1.4 * n_row)
    spkws = dict(nrows=n_row, ncols=n_col, figsize=figsize, facecolor="k")
    savekws = dict(dpi=100, bbox_inches="tight", facecolor="k", edgecolor="k")

    def add_colorbar(f, cmap, low, high, left, width, fmt):
        cbar = np.outer(np.arange(0, 1, .01), np.ones(10))
        cbar_ax = f.add_axes([left, 0, width, .03])
        cbar_ax.imshow(cbar.T, aspect="auto", cmap=cmap)
        cbar_ax.axis("off")
        f.text(left - .01, .018, fmt % low, ha="right", va="center",
               color="white", size=13, weight="demibold")
        f.text(left + width + .01, .018, fmt % high, ha="left",
               va="center", color="white", size=13, weight="demibold")

    report = []

    # Plot the residual image (sigmasquareds)
    ss = nib.load(sigmasquareds_file).get_data()
    sslow, sshigh = moss.percentiles(ss, [2, 98])
    ss[mean_data == 0] = np.nan
    f, axes = plt.subplots(**spkws)
    for i, ax in enumerate(axes.ravel(), start):
        ax.imshow(mean_data[..., i].T, cmap="gray",
                  vmin=mlow, vmax=mhigh, interpolation="nearest")
        ax.imshow(ss[..., i].T, cmap="PuRd_r",
                  vmin=sslow, vmax=sshigh, alpha=.7)
        ax.axis("off")
    add_colorbar(f, "PuRd_r", sslow, sshigh, .35, .3, "%d")
    ss_png = op.abspath("sigmasquareds.png")
    f.savefig(ss_png, **savekws)
    plt.close(f)
    report.append(ss_png)

    # Now plot each zstat file
    for z_i, zname in enumerate(zstat_files, 1):
        zdata = nib.load(zname).get_data()
        pos = zdata.copy()
        pos[pos < 2.3] = np.nan
        neg = zdata.copy()
        neg[neg > -2.3] = np.nan
        zlow = 2.3
        zhigh = max(np.abs(zdata).max(), 3.71)
        f, axes = plt.subplots(**spkws)
        for i, ax in enumerate(axes.ravel()):
            ax.imshow(mean_data[..., i].T, cmap="gray",
                      vmin=mlow, vmax=mhigh)
            ax.imshow(pos[..., i].T, cmap="Reds_r",
                      vmin=zlow, vmax=zhigh)
            ax.imshow(neg[..., i].T, cmap="Blues",
                      vmin=-zhigh, vmax=-zlow)
            ax.axis("off")
        add_colorbar(f, "Blues", -zhigh, -zlow, .15, .3, "%.1f")
        add_colorbar(f, "Reds_r", zlow, zhigh, .55, .3, "%.1f")

        fname = op.abspath("zstat%d.png" % z_i)
        f.savefig(fname, **savekws)
        plt.close(f)
        report.append(fname)

    # Now the r_2 files
    for rname, cmap in zip(r2_files, ["GnBu_r", "YlGn_r", "OrRd_r"]):
        data = nib.load(rname).get_data()
        rhigh = moss.percentiles(np.nan_to_num(data), 99)
        f, axes = plt.subplots(**spkws)
        for i, ax in enumerate(axes.ravel(), start):
            ax.imshow(mean_data[..., i].T, cmap="gray",
                      vmin=mlow, vmax=mhigh, interpolation="nearest")
            ax.imshow(data[..., i].T, cmap=cmap, vmin=0, vmax=rhigh, alpha=.7)
            ax.axis("off")
        add_colorbar(f, cmap, 0, rhigh, .35, .3, "%.2f")

        fname = op.abspath(op.basename(rname).replace(".nii.gz", ".png"))
        f.savefig(fname, **savekws)
        plt.close(f)
        report.append(fname)

    return report


def dump_exp_info(exp_info, timeseries):
    """Dump the exp_info dict into a json file."""
    json_file = op.abspath("experiment_info.json")
    with open(json_file, "w") as fp:
        json.dump(exp_info, fp, sort_keys=True, indent=2)
    return json_file


# Smaller helper functions
# ========================


def run_indices(ts_files):
    """Find the run numbers associated with timeseries files."""
    import re
    if not isinstance(ts_files, list):
        ts_files = [ts_files]
    runs = [re.search("run_(\d+)", f).group(1) for f in ts_files]
    return map(int, runs)
