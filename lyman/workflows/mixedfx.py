import os
import os.path as op

import numpy as np
import scipy as sp
import pandas as pd
import nibabel as nib
from skimage import morphology
import matplotlib as mpl
import matplotlib.pyplot as plt

from nipype import (IdentityInterface, Function, Rename,
                    Node, MapNode, Workflow)
from nipype.interfaces import fsl, freesurfer

import seaborn
from moss import locator

import lyman

imports = ["import os",
           "import os.path as op",
           "import numpy as np",
           "import scipy as sp",
           "import pandas as pd",
           "import nibabel as nib",
           "import matplotlib as mpl",
           "import matplotlib.pyplot as plt",
           "from skimage import morphology",
           "from nipype.interfaces import fsl",
           "from moss import locator",
           "import seaborn"]


def create_volume_mixedfx_workflow(name="volume_group",
                                   subject_list=None,
                                   regressors=None,
                                   contrasts=None,
                                   exp_info=None):

    # Handle default arguments
    if subject_list is None:
        subject_list = []
    if regressors is None:
        regressors = dict(group_mean=[])
    if contrasts is None:
        contrasts = [["group_mean", "T", ["group_mean"], [1]]]
    if exp_info is None:
        exp_info = lyman.default_experiment_parameters()

    # Define workflow inputs
    inputnode = Node(IdentityInterface(["l1_contrast",
                                        "copes",
                                        "varcopes",
                                        "dofs"]),
                     "inputnode")

    # Merge the fixed effect summary images into one 4D image
    mergecope = Node(fsl.Merge(dimension="t"), "mergecope")
    mergevarcope = Node(fsl.Merge(dimension="t"), "mergevarcope")
    mergedof = Node(fsl.Merge(dimension="t"), "mergedof")

    # Make a simple design
    design = Node(fsl.MultipleRegressDesign(regressors=regressors,
                                            contrasts=contrasts),
                  "design")

    # Find the intersection of masks across subjects
    makemask = Node(Function(["varcope_file"],
                             ["mask_file"],
                             make_group_mask,
                             imports),
                    "makemask")

    # Fit the mixed effects model
    flameo = Node(fsl.FLAMEO(run_mode=exp_info["flame_mode"]), "flameo")

    # Estimate the smoothness of the data
    smoothest = Node(fsl.SmoothEstimate(), "smoothest")

    # Correct for multiple comparisons
    cluster = Node(fsl.Cluster(threshold=exp_info["cluster_zthresh"],
                               pthreshold=exp_info["grf_pthresh"],
                               out_threshold_file=True,
                               out_index_file=True,
                               out_localmax_txt_file=True,
                               peak_distance=exp_info["peak_distance"],
                               use_mm=True),
                   "cluster")

    # Deal with FSL's poorly formatted table of peaks
    peaktable = Node(Function(["localmax_file"],
                              ["out_file"],
                              imports=imports,
                              function=cluster_table),
                     "peaktable")

    # Segment the z stat image with a watershed algorithm
    watershed = Node(Function(["zstat_file", "localmax_file"],
                              ["seg_file", "peak_file", "lut_file"],
                              watershed_segment,
                              imports),
                     "watershed")

    # Sample the zstat image to the surface
    hemisource = Node(IdentityInterface(["mni_hemi"]), "hemisource")
    hemisource.iterables = ("mni_hemi", ["lh", "rh"])

    zstatproj = Node(freesurfer.SampleToSurface(
        sampling_method=exp_info["sampling_method"],
        sampling_range=exp_info["sampling_range"],
        sampling_units=exp_info["sampling_units"],
        smooth_surf=exp_info["surf_smooth"],
        subject_id="fsaverage",
        mni152reg=True,
        target_subject="fsaverage"),
        "zstatproj")

    # Sample the mask to the surface
    maskproj = Node(freesurfer.SampleToSurface(
        sampling_method="max",
        sampling_range=exp_info["sampling_range"],
        sampling_units=exp_info["sampling_units"],
        smooth_surf=exp_info["surf_smooth"],
        subject_id="fsaverage",
        mni152reg=True,
        target_subject="fsaverage"),
        "maskproj")

    # Make static report images in the volume
    report = Node(Function(["mask_file",
                            "zstat_file",
                            "localmax_file",
                            "cope_file",
                            "seg_file",
                            "subjects"],
                           ["report"],
                           mfx_report,
                           imports),
                  "report")
    report.inputs.subjects = subject_list

    # Define the workflow outputs
    outputnode = Node(IdentityInterface(["copes",
                                         "varcopes",
                                         "mask_file",
                                         "flameo_stats",
                                         "thresh_zstat",
                                         "surf_zstat",
                                         "surf_mask",
                                         "cluster_image",
                                         "cluster_peaks",
                                         "seg_file",
                                         "peak_file",
                                         "lut_file",
                                         "report"]),
                      "outputnode")

    # Define and connect up the workflow
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
        (cluster, watershed,
            [("threshold_file", "zstat_file"),
             ("localmax_txt_file", "localmax_file")]),
        (makemask, report,
            [("mask_file", "mask_file")]),
        (cluster, report,
            [("threshold_file", "zstat_file"),
             ("localmax_txt_file", "localmax_file")]),
        (mergecope, report,
            [("merged_file", "cope_file")]),
        (watershed, report,
            [("seg_file", "seg_file")]),
        (cluster, peaktable,
            [("localmax_txt_file", "localmax_file")]),
        (cluster, zstatproj,
            [("threshold_file", "source_file")]),
        (hemisource, zstatproj,
            [("mni_hemi", "hemi")]),
        (makemask, maskproj,
            [("mask_file", "source_file")]),
        (hemisource, maskproj,
            [("mni_hemi", "hemi")]),
        (mergecope, outputnode,
            [("merged_file", "copes")]),
        (mergevarcope, outputnode,
            [("merged_file", "varcopes")]),
        (makemask, outputnode,
            [("mask_file", "mask_file")]),
        (flameo, outputnode,
            [("stats_dir", "flameo_stats")]),
        (cluster, outputnode,
            [("threshold_file", "thresh_zstat"),
             ("index_file", "cluster_image")]),
        (peaktable, outputnode,
            [("out_file", "cluster_peaks")]),
        (watershed, outputnode,
            [("seg_file", "seg_file"),
             ("peak_file", "peak_file"),
             ("lut_file", "lut_file")]),
        (zstatproj, outputnode,
            [("out_file", "surf_zstat")]),
        (maskproj, outputnode,
            [("out_file", "surf_mask")]),
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


def watershed_segment(zstat_file, localmax_file):
    """Segment the thresholded zstat image."""
    z_img = nib.load(zstat_file)
    z_data = z_img.get_data()

    # Set up the output filenames
    seg_file = op.basename(zstat_file).replace(".nii.gz", "_seg.nii.gz")
    seg_file = op.abspath(seg_file)
    peak_file = op.basename(zstat_file).replace(".nii.gz", "_peaks.nii.gz")
    peak_file = op.abspath(peak_file)
    lut_file = seg_file.replace(".nii.gz", ".txt")

    # Read in the peak txt file from FSL cluster
    peaks = pd.read_table(localmax_file, "\t")[["x", "y", "z"]].values
    markers = np.zeros_like(z_data)

    # Do the watershed, or not, depending on whether we had peaks
    if len(peaks):
        markers[tuple(peaks.T)] = np.arange(len(peaks)) + 1
        seg = morphology.watershed(-z_data, markers, mask=z_data > 0)
    else:
        seg = np.zeros_like(z_data)

    # Create a Nifti image with the segmentation and save it
    seg_img = nib.Nifti1Image(seg.astype(np.int16),
                              z_img.get_affine(),
                              z_img.get_header())
    seg_img.set_data_dtype(np.int16)
    seg_img.to_filename(seg_file)

    # Create a Nifti image with just the peaks and save it
    peak_img = nib.Nifti1Image(markers.astype(np.int16),
                               z_img.get_affine(),
                               z_img.get_header())
    peak_img.set_data_dtype(np.int16)
    peak_img.to_filename(peak_file)

    # Write a lookup-table in Freesurfer format so we can
    # view the segmentation in Freeview
    n = int(markers.max())
    colors = [[0, 0, 0]] + seaborn.husl_palette(n)
    colors = np.hstack([colors, np.zeros((n + 1, 1))])
    lut_data = pd.DataFrame(columns=["#ID", "ROI", "R", "G", "B", "A"],
                            index=np.arange(n + 1))
    names = ["Unknown"] + ["roi_%d" % i for i in range(1, n + 1)]
    lut_data["ROI"] = np.array(names)
    lut_data["#ID"] = np.arange(n + 1)
    lut_data.loc[:, "R":"A"] = (colors * 255).astype(int)
    lut_data.to_csv(lut_file, "\t", index=False)

    return seg_file, peak_file, lut_file


def mfx_report(mask_file, zstat_file, localmax_file,
               cope_file, seg_file, subjects):
    """Plot various information related to the results."""
    mni_brain = fsl.Info.standard_image("avg152T1_brain.nii.gz")
    mni_data = nib.load(mni_brain).get_data()

    mask_data = nib.load(mask_file).get_data()
    mask = np.where(mask_data, 1, np.nan)

    z_stat = nib.load(zstat_file).get_data()
    z_plot = z_stat.copy()
    z_plot[z_stat == 0] = np.nan

    peaks = pd.read_table(localmax_file, "\t")[["x", "y", "z"]].values

    seg_data = nib.load(seg_file).get_data().astype(float)

    # Set up the output names
    mask_png = os.path.abspath("group_mask.png")
    zname = os.path.basename(zstat_file).replace(".nii.gz", "")
    zstat_png = os.path.abspath("%s.png" % zname)
    peaks_png = os.path.abspath("%s_peaks.png" % zname)
    boxplot_png = os.path.abspath("peak_boxplot.png")
    seg_png = os.path.abspath(seg_file.replace(".nii.gz", ".png"))

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
    pltkws = dict(nrows=int(n_row), ncols=int(n_col),
                  figsize=figsize, facecolor="k")
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
        ax.axis("off")
    plt.savefig(mask_png, **pngkws)
    plt.close(f)

    def add_colorbar(f, cmap, low, high, left, width, fmt):
        cbar = np.outer(np.arange(0, 1, .01), np.ones(10))
        cbar_ax = f.add_axes([left, 0, width, .03])
        cbar_ax.imshow(cbar.T, aspect="auto", cmap=cmap)
        cbar_ax.axis("off")
        f.text(left - .01, .018, fmt % low, ha="right", va="center",
               color="white", size=13, weight="demibold")
        f.text(left + width + .01, .018, fmt % high, ha="left",
               va="center", color="white", size=13, weight="demibold")

    # Now plot the zstat image
    mask = np.where(mask_data, np.nan, 1)
    mask_cmap = mpl.colors.ListedColormap(["#160016"])
    zlow = 2.3
    zhigh = max(3.71, z_stat.max())
    f, axes = plt.subplots(**pltkws)
    for i, ax in zip(slices, axes.ravel()):
        ax.imshow(mni_data[xmin:xmax, ymin:ymax, i].T,
                  cmap="gray", vmin=vmin, vmax=vmax)
        ax.imshow(z_plot[xmin:xmax, ymin:ymax, i].T,
                  cmap="Reds_r", vmin=zlow, vmax=zhigh)
        ax.imshow(mask[xmin:xmax, ymin:ymax, i].T,
                  cmap=mask_cmap, alpha=.5, interpolation="nearest")
        ax.axis("off")
    add_colorbar(f, "Reds_r", zlow, zhigh, .35, .3, "%.1f")
    plt.savefig(zstat_png, **pngkws)
    plt.close(f)

    # Everything else is dependent on there being some peak data
    if len(peaks):

        # Now plot the peak centroids
        peak_data = np.zeros_like(z_stat)
        disk_data = np.zeros_like(z_stat)
        y, x = np.ogrid[-4: 4 + 1, -4:4 + 1]
        disk = x ** 2 + y ** 2 <= 4 ** 2
        dilator = np.dstack([disk, disk, np.zeros_like(disk)])
        for i, peak in enumerate(peaks, 1):
            spot = np.zeros_like(z_stat)
            spot[tuple(peak)] = 1
            peak_data += spot
            disk = sp.ndimage.binary_dilation(spot, dilator)
            disk_data[disk] = i
        disk_data[disk_data == 0] = np.nan

        husl_colors = seaborn.husl_palette(len(peaks))
        peak_cmap = mpl.colors.ListedColormap(husl_colors)
        f, axes = plt.subplots(**pltkws)
        bg = np.zeros(mni_data.shape[:2])
        for i, ax in zip(slices, axes.ravel()):
            ax.imshow(bg, cmap="gray", vmin=0, vmax=vmax)
            ax.imshow(mni_data[xmin:xmax, ymin:ymax, i].T,
                      cmap="gray", alpha=.6, vmin=vmin, vmax=vmax)
            ax.imshow(disk_data[xmin:xmax, ymin:ymax, i].T,
                      cmap=peak_cmap, vmin=1, vmax=len(peaks) + 1)
            ax.axis("off")
        plt.savefig(peaks_png, **pngkws)
        plt.close(f)

        # Now make a boxplot of the peaks
        seaborn.set()
        cope_data = nib.load(cope_file).get_data()
        peak_dists = list(cope_data[tuple(peaks.T)])
        peak_dists.reverse()
        n_peaks = len(peak_dists)
        f, ax = plt.subplots(figsize=(8, float(n_peaks) / 3 + 0.33))
        colors = list(reversed(husl_colors))
        seaborn.boxplot(np.transpose(peak_dists),
                        color=colors, vert=False, ax=ax)
        ax.axvline(0, c="#222222", ls="--")
        labels = np.arange(1, n_peaks + 1)[::-1]
        ax.set_yticklabels(labels)
        ax.set_ylabel("Local Maximum")
        ax.set_xlabel("COPE Value")
        plt.savefig(boxplot_png, dpi=100, bbox_inches="tight")
        plt.close(f)

        # Watershed segmentation image
        seg_data[seg_data == 0] = np.nan
        f, axes = plt.subplots(**pltkws)
        for i, ax in zip(slices, axes.ravel()):
            ax.imshow(bg, cmap="gray", vmin=0, vmax=vmax)
            ax.imshow(mni_data[xmin:xmax, ymin:ymax, i].T,
                      cmap="gray", alpha=.6, vmin=vmin, vmax=vmax)
            ax.imshow(mask[xmin:xmax, ymin:ymax, i].T,
                      cmap=mask_cmap, alpha=.5, interpolation="nearest")
            ax.imshow(seg_data[xmin:xmax, ymin:ymax, i].T,
                      interpolation="nearest",
                      cmap=peak_cmap, vmin=1, vmax=len(peaks) + 1)
            ax.axis("off")
        plt.savefig(seg_png, **pngkws)
        plt.close(f)

    else:
        for fname in [peaks_png, boxplot_png, seg_png]:
            with open(fname, "wb"):
                pass

    # Save the list of subjects in this analysis
    subj_file = op.abspath("subjects.txt")
    np.savetxt(subj_file, subjects, "%s")

    return [mask_png, zstat_png, peaks_png, boxplot_png, seg_png, subj_file]


def cluster_table(localmax_file):
    """Add some info to an FSL cluster file and format it properly."""
    df = pd.read_table(localmax_file, delimiter="\t")
    df = df[["Cluster Index", "Value", "x", "y", "z"]]
    df.columns = ["Cluster", "Value", "x", "y", "z"]
    df.index.name = "Peak"

    # Find out where the peaks most likely are
    if len(df):
        coords = df[["x", "y", "z"]].values
        loc_df = locator.locate_peaks(coords)
        df = pd.concat([df, loc_df], axis=1)
        mni_coords = locator.vox_to_mni(coords).T
        for i, ax in enumerate(["x", "y", "z"]):
            df[ax] = mni_coords[i]

    out_file = op.abspath(op.basename(localmax_file[:-3] + "csv"))
    df.to_csv(out_file)
    return out_file
