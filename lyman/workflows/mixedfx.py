import os
import os.path as op

import numpy as np
import scipy as sp
import pandas as pd
import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt

from nipype import fsl, IdentityInterface, Function, Node, MapNode, Workflow

import seaborn
from moss import locator

imports = ["import os",
           "import os.path as op",
           "import numpy as np",
           "import scipy as sp",
           "import pandas as pd",
           "import nibabel as nib",
           "import matplotlib as mpl",
           "import matplotlib.pyplot as plt",
           "from nipype import fsl",
           "from moss import locator",
           "import seaborn"]


def create_volume_mixedfx_workflow(name="volume_group",
                                   subject_list=[],
                                   regressors=[],
                                   contrasts=[],
                                   flame_mode="flame1",
                                   cluster_zthresh=2.3,
                                   grf_pthresh=0.05):

    inputnode = Node(IdentityInterface(["l1_contrast",
                                        "copes",
                                        "varcopes",
                                        "dofs"]),
                     "inputnode")

    mergecope = Node(fsl.Merge(dimension="t"), "mergecope")

    mergevarcope = Node(fsl.Merge(dimension="t"), "mergevarcope")

    mergedof = Node(fsl.Merge(dimension="t"), "mergedof")

    design = Node(fsl.MultipleRegressDesign(regressors=regressors,
                                            contrasts=contrasts),
                  "design")

    makemask = Node(Function(["varcope_file"],
                             ["mask_file"],
                             make_group_mask,
                             imports),
                    "makemask")

    flameo = Node(fsl.FLAMEO(run_mode=flame_mode), "flameo")

    smoothest = MapNode(fsl.SmoothEstimate(), "zstat_file", "smoothest")

    cluster = MapNode(fsl.Cluster(threshold=cluster_zthresh,
                                  pthreshold=0.05,
                                  out_threshold_file=True,
                                  out_index_file=True,
                                  out_localmax_txt_file=True,
                                  peak_distance=15,
                                  use_mm=True),
                      ["in_file", "dlh", "volume"],
                      "cluster")

    peaktable = MapNode(Function(["localmax_file"],
                                 ["out_file"],
                                 imports=imports,
                                 function=cluster_table),
                        "localmax_file",
                        "peaktable")

    report = MapNode(Function(["mask_file",
                               "zstat_file",
                               "localmax_file",
                               "cope_file"],
                              ["report"],
                              mfx_report,
                              imports),
                     ["zstat_file", "localmax_file"],
                     "report")

    outputnode = Node(IdentityInterface(["mask_file",
                                         "flameo_stats",
                                         "thresh_zstat",
                                         "cluster_image",
                                         "cluster_peaks",
                                         "report"]),
                      "outputnode")

    group = Workflow(name)
    group.connect([
        (inputnode, mergecope,
            [("copes", "in_files")]),
        (inputnode, mergevarcope,
            [("varcopes", "in_files")]),
        (inputnode, mergedof,
            [("dofs", "in_files")]),
        (mergecope, flameo,
            [("merged_file", "cope_file")]),
        (mergevarcope, flameo,
            [("merged_file", "var_cope_file")]),
        (mergevarcope, makemask,
            [("merged_file", "varcope_file")]),
        (mergedof, flameo,
            [("merged_file", "dof_var_cope_file")]),
        (makemask, flameo,
            [("mask_file", "mask_file")]),
        (design, flameo,
            [("design_con", "t_con_file"),
             ("design_grp", "cov_split_file"),
             ("design_mat", "design_file")]),
        (flameo, smoothest,
            [("zstats", "zstat_file")]),
        (makemask, smoothest,
            [("mask_file", "mask_file")]),
        (smoothest, cluster,
            [("dlh", "dlh"),
             ("volume", "volume")]),
        (flameo, cluster,
            [("zstats", "in_file")]),
        (makemask, report,
            [("mask_file", "mask_file")]),
        (cluster, report,
            [("threshold_file", "zstat_file"),
             ("localmax_txt_file", "localmax_file")]),
        (mergecope, report,
            [("merged_file", "cope_file")]),
        (cluster, peaktable,
            [("localmax_txt_file", "localmax_file")]),
        (makemask, outputnode,
            [("mask_file", "mask_file")]),
        (flameo, outputnode,
            [("stats_dir", "flameo_stats")]),
        (cluster, outputnode,
            [("threshold_file", "thresh_zstat"),
             ("index_file", "cluster_image")]),
        (peaktable, outputnode,
             [("out_file", "cluster_peaks")]),
        (report, outputnode,
            [("report", "report")]),
        ])

    return group, inputnode, outputnode


def make_group_mask(varcope_file):
    """Find the intersection of the MNI brain and var > 0 voxels."""
    mni_mask = fsl.Info.standard_image("MNI152_T1_2mm_brain_mask.nii.gz")
    mni_img = nib.load(mni_mask)
    mask_data = mni_img.get_data().astype(bool)

    # Find the voxels with positive variance
    var_data = nib.load(varcope_file).get_data()
    good_var = var_data.all(axis=-1)

    # Find the intersection
    mask_data *= good_var

    # Save the mask file
    new_img = nib.Nifti1Image(mask_data,
                              mni_img.get_affine(),
                              mni_img.get_header())
    new_img.set_data_dtype(np.int16)
    mask_file = os.path.abspath("group_mask.nii.gz")
    new_img.to_filename(mask_file)
    return mask_file


def mfx_report(mask_file, zstat_file, localmax_file, cope_file):
    """Plot various information related to the results."""
    mni_brain = fsl.Info.standard_image("avg152T1_brain.nii.gz")
    mni_data = nib.load(mni_brain).get_data()

    mask_data = nib.load(mask_file).get_data()
    mask = np.where(mask_data, 1, np.nan)

    z_data = nib.load(zstat_file).get_data()
    z_data[z_data < 2.3] = np.nan

    peaks = pd.read_table(localmax_file,
                          delimiter="\t")[["x", "y", "z"]].values

    # Find the plot parameters
    xdata = np.flatnonzero(mni_data.any(axis=1).any(axis=1))
    xmin, xmax = xdata.min(), xdata.max() + 1
    ydata = np.flatnonzero(mni_data.any(axis=0).any(axis=1))
    ymin, ymax = ydata.min(), ydata.max() + 1
    zdata = np.flatnonzero(mni_data.any(axis=0).any(axis=0))
    zmin, zmax = zdata.min(), zdata.max() + 1

    n_slices = (zmax - zmin) // 2
    n_row, n_col = n_slices // 8, 8
    start = n_slices % n_col // 2 + zmin + 4
    figsize = (10, 1.375 * n_row)
    slices = (start + np.arange(zmax - zmin))[::2][:n_slices]
    pltkws = dict(nrows=n_row, ncols=n_col, figsize=figsize, facecolor="k")
    pngkws = dict(dpi=100, bbox_inches="tight", facecolor="k", edgecolor="k")

    vmin, vmax = 0, mni_data.max()

    # First plot the mask image
    f, axes = plt.subplots(**pltkws)
    cmap = mpl.colors.ListedColormap(["MediumSpringGreen"])
    for i, ax in zip(slices, axes.ravel()):
        ax.imshow(mni_data[xmin:xmax, ymin:ymax, i].T,
                  cmap="gray", vmin=vmin, vmax=vmax)
        ax.imshow(mask[xmin:xmax, ymin:ymax, i].T,
                  alpha=.7, cmap=cmap, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
    mask_png = os.path.abspath("group_mask.png")
    plt.savefig(mask_png, **pngkws)

    # Now plot the zstat image
    mask = np.where(mask_data, np.nan, 1)
    mask_cmap = mpl.colors.ListedColormap(["#160016"])
    f, axes = plt.subplots(**pltkws)
    for i, ax in zip(slices, axes.ravel()):
        ax.imshow(mni_data[xmin:xmax, ymin:ymax, i].T,
                  cmap="gray", vmin=vmin, vmax=vmax)
        ax.imshow(z_data[xmin:xmax, ymin:ymax, i].T,
                  cmap="YlOrRd_r", vmin=2.3, vmax=4.26)
        ax.imshow(mask[xmin:xmax, ymin:ymax, i].T,
                  cmap=mask_cmap, alpha=.5, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
    zname = os.path.basename(zstat_file).replace(".nii.gz", "")
    zstat_png = os.path.abspath("%s.png" % zname)
    plt.savefig(zstat_png, **pngkws)

    # Now plot the peak centroids
    peak_data = np.zeros_like(z_data)
    y, x = np.ogrid[-4: 4 + 1, -4:4 + 1]
    disk = x ** 2 + y ** 2 <= 4 ** 2
    dilator = np.dstack([disk, disk, np.zeros_like(disk)])
    for i, peak in enumerate(peaks, 1):
        spot = np.zeros_like(z_data)
        spot[tuple(peak)] = 1
        spot = sp.ndimage.binary_dilation(spot, dilator)
        peak_data[spot] = i
    peak_data[peak_data == 0] = np.nan

    husl_colors = reversed(seaborn.husl_palette(len(peaks)))
    peak_cmap = mpl.colors.ListedColormap(list(husl_colors))
    f, axes = plt.subplots(**pltkws)
    bg = np.zeros(mni_data.shape[:2])
    for i, ax in zip(slices, axes.ravel()):
        ax.imshow(bg, cmap="gray", vmin=0, vmax=vmax)
        ax.imshow(mni_data[xmin:xmax, ymin:ymax, i].T,
                  cmap="gray", alpha=.6, vmin=vmin, vmax=vmax)
        ax.imshow(peak_data[xmin:xmax, ymin:ymax, i].T,
                  cmap=peak_cmap, vmin=1, vmax=len(peaks) + 1)
        ax.set_xticks([])
        ax.set_yticks([])
    peaks_png = os.path.abspath("%s_peaks.png" % zname)
    plt.savefig(peaks_png, **pngkws)

    # Now make a boxplot of the peaks
    seaborn.set()
    cope_data = nib.load(cope_file).get_data()
    peak_dists = []
    for coords in reversed(peaks):
        peak_dists.append(cope_data[tuple(coords)])
    peak_dists.reverse()
    n_peaks = len(peak_dists)
    fig = plt.figure(figsize=(8, float(n_peaks) / 3 + 0.33))
    ax = fig.add_subplot(111)
    seaborn.boxplot(np.transpose(peak_dists),
                    color="husl",
                    vert=0, ax=ax)
    ax.axvline(0, c="#222222", ls="--")
    labels = range(1, n_peaks + 1)
    labels.reverse()
    ax.set_yticklabels(labels)
    ax.set_ylabel("Local Maximum")
    ax.set_xlabel("COPE Value")
    plt.tight_layout()
    boxplot_png = os.path.abspath("peak_boxplot.png")
    plt.savefig(boxplot_png, dpi=100, bbox_inches="tight")

    return [mask_png, zstat_png, peaks_png, boxplot_png]

def cluster_table(localmax_file):
    """Add some info to an FSL cluster file and format it properly."""
    df = pd.read_table(localmax_file, delimiter="\t")
    df = df[["Cluster Index", "Value", "x", "y", "z"]]
    df.columns = ["Cluster", "Value", "x", "y", "z"]
    df.index.name = "Peak"

    # Find out where the peaks most likely are
    coords = ["x", "y", "z"]
    loc_df = locator.locate_peaks(np.array(df[coords]))
    df = pd.concat([df, loc_df], axis=1)
    mni_coords = locator.vox_to_mni(np.array(df[coords])).T
    for i, ax in enumerate(coords):
        df[ax] = mni_coords[i]

    out_file = op.abspath(op.basename(localmax_file[:-3] + "csv"))
    df.to_csv(out_file)
    return out_file
