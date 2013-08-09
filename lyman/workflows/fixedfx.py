"""Fixed effects model to combine across runs for a single subject."""
import os
import os.path as op
import subprocess as sub
import numpy as np
import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import moss

from nipype import fsl, Workflow, Node, IdentityInterface, Function

imports = ["import os",
           "import os.path as op",
           "import subprocess as sub",
           "import numpy as np",
           "import nibabel as nib",
           "import matplotlib as mpl",
           "import matplotlib.pyplot as plt",
           "import seaborn as sns",
           "import moss"]


def create_ffx_workflow(name="mni_ffx", space="mni", contrasts=None):
    """Return a workflow object to execute a fixed-effects mode."""
    if contrasts is None:
        contrasts = []

    inputnode = Node(IdentityInterface(["copes",
                                        "varcopes",
                                        "masks",
                                        "dofs",
                                        "ss_files",
                                        "anatomy"]),
                     name="inputnode")

    # Set up a fixed effects FLAMEO model
    ffxdesign = Node(fsl.L2Model(num_copes=len(contrasts)), "ffxdesign")

    # Fit the fixedfx model for each contrast
    ffxmodel = Node(Function(["contrasts",
                              "copes",
                              "varcopes",
                              "dofs",
                              "masks",
                              "design_mat",
                              "design_con",
                              "design_grp"],
                             ["flame_results",
                              "zstat_files"],
                             fixedfx_model,
                             imports),
                    "ffxmodel")
    ffxmodel.inputs.contrasts = contrasts

    # Calculate the fixed effects Rsquared maps
    ffxr2 = Node(Function(["ss_files"], ["r2_files"],
                          fixedfx_r2, imports),
                 "ffxr2")

    # Plot the fixedfx results
    report = Node(Function(["space",
                            "anatomy",
                            "zstat_files",
                            "r2_files",
                            "masks"],
                           ["report"],
                           fixedfx_report,
                           imports),
                  "report")
    report.inputs.space = space

    outputnode = Node(IdentityInterface(["flame_results",
                                         "r2_files",
                                         "report"]),
                      "outputs")

    ffx = Workflow(name=name)
    ffx.connect([
        (inputnode, ffxmodel,
            [("copes", "copes"),
             ("varcopes", "varcopes"),
             ("dofs", "dofs"),
             ("masks", "masks")]),
        (ffxdesign, ffxmodel,
            [("design_mat", "design_mat"),
             ("design_con", "design_con"),
             ("design_grp", "design_grp")]),
        (inputnode, ffxr2,
            [("ss_files", "ss_files")]),
        (inputnode, report,
            [("anatomy", "anatomy"),
             ("masks", "masks")]),
        (ffxmodel, report,
            [("zstat_files", "zstat_files")]),
        (ffxr2, report,
            [("r2_files", "r2_files")]),
        (ffxmodel, outputnode,
            [("flame_results", "flame_results")]),
        (ffxr2, outputnode,
            [("r2_files", "r2_files")]),
        (report, outputnode,
            [("report", "report")]),
                 ])

    return ffx, inputnode, outputnode


# Main interface functions
# ------------------------


def fixedfx_model(contrasts, copes, varcopes, dofs, masks,
                  design_mat, design_con, design_grp):
    """Fit the fixed effects model for each contrast."""
    n_runs = len(copes) / len(contrasts)

    # Find the basic geometry of the image
    img = nib.load(copes[0]).shape
    x, y, z = img.shape
    aff, hdr = img.get_affine(), img.get_header()

    # Get lists of files for each contrast
    copes = zip(*map(list, np.split(copes, n_runs)))
    varcopes = zip(*map(list, np.split(varcopes, n_runs)))

    # Make an image with the DOF for each run
    dofs = np.concatenate(map(np.loadtxt, dofs))
    dof_data = np.ones((x, y, z, len(dofs))) * dofs
    nib.Nifti1Image(dof_data, aff, hdr).to_filename("dof.nii.gz")

    # Find the intersection of the masks
    mask_data = [nib.load(f).get_data() for f in masks]
    common_mask = np.all(mask_data, axis=0)
    nib.Nifti1Image(common_mask, aff, hdr).to_filename("mask.nii.gz")

    # Run the flame models
    flame_results = []
    zstat_files = []
    for i, contrast in enumerate(contrasts):
        os.mkdir(contrast)
        os.chdir(contrast)

        # Concatenate the copes and varcopes
        for kind, files in zip(["cope", "varcope"],
                               [copes, varcopes]):
            data = [nib.load(f).get_data()[..., np.newaxis] for f in files]
            data = np.concatenate(data, axis=-1)
            nib.Nifti1Image(data, aff, hdr).to_filename(kind + ".nii.gz")


        flamecmd = ["flameo",
                    "--cope=cope.nii.gz",
                    "--varcope=varcope.nii.gz",
                    "--mask=../mask.nii.gz",
                    "--dvc=../dof.nii.gz",
                    "--runmode=fe",
                    "--dm=" + design_mat,
                    "--tc=" + design_con,
                    "--cs=" + design_grp,
                    "--ld=flamestats",
                    "--npo"]
        sub.check_output(flamecmd)
        os.chdir("..")
        flame_results.append(contrast + "/flamestats/")
        zstat_files.append(op.abspath(contrast + "/zstat1.nii.gz"))

    return flame_results, zstat_files


def fixedfx_r2(ss_files):
    """Find the R2 for the full fixedfx model."""
    ss_tot = [f for f in ss_files if "sstot" in f]
    ss_res_full = [f for f in ss_files if "ssres_full" in f]
    ss_res_main = [f for f in ss_files if "ssres_main" in f]

    tot_data = [nib.load(f).get_data() for f in ss_tot]
    main_data = [nib.load(f).get_data() for f in ss_res_full]
    full_data = [nib.load(f).get_data() for f in ss_res_main]

    tot_data = np.sum(tot_data, axis=0)
    main_data = np.sum(main_data, axis=0)
    full_data = np.sum(full_data, axis=0)

    r2_full = 1 - full_data / tot_data
    r2_main = 1 - main_data / tot_data

    img = nib.load(ss_tot[0])
    aff, header = img.get_affine(), img.get_header()

    r2_full_img = nib.Nifti1Image(r2_full, aff, header)
    r2_full_file = op.abspath("r2_full.nii.gz")
    r2_full_img.to_filename(r2_full_file)

    r2_main_img = nib.Nifti1Image(r2_main, aff, header)
    r2_main_file = op.abspath("r2_main.nii.gz")
    r2_main_img.to_filename(r2_main_file)

    return [r2_full_file, r2_main_file]


def fixedfx_report(space, anatomy, zstat_files, r2_files, masks):
    """Plot the resulting data."""
    sns.set()
    bg = nib.load(anatomy).get_data()

    mask_data = [nib.load(f).get_data() for f in masks]
    mask = np.where(np.all(mask_data, axis=0), np.nan, 1)
    mask_count = np.sum(mask, axis=0)

    # Find the plot parameters
    xdata = np.flatnonzero(bg.any(axis=1).any(axis=1))
    xslice = slice(xdata.min(), xdata.max() + 1)
    ydata = np.flatnonzero(bg.any(axis=0).any(axis=1))
    yslice = slice(ydata.min(), ydata.max() + 1)
    zdata = np.flatnonzero(bg.any(axis=0).any(axis=0))
    zmin, zmax = zdata.min(), zdata.max() + 1

    step = 2 if space == "mni" else 1

    n_slices = (zmax - zmin) // step
    n_row, n_col = n_slices // 8, 8
    start = n_slices % n_col // step + zmin + 4
    figsize = (10, 1.375 * n_row)
    slices = (start + np.arange(zmax - zmin))[::step][:n_slices]
    pltkws = dict(nrows=n_row, ncols=n_col, figsize=figsize, facecolor="k")
    pngkws = dict(dpi=100, bbox_inches="tight", facecolor="k", edgecolor="k")

    vmin, vmax = 0, moss.percentiles(bg, 95)
    mask_cmap = mpl.colors.ListedColormap(["#160016"])

    report = []

    # Plot the mask counts
    f, axes = plt.subplots(**pltkws)
    mask_count[bg == 0] = np.nan
    for i, ax in zip(slices, axes.ravel()):
        ax.imshow(bg[xslice, yslice, i].T,
                  cmap="gray", vmin=vmin, vmax=vmax)
        ax.imshow(mask_count[xslice, yslice, i].T, alpha=.7,
                  cmap="gist_heat", interpolation="nearest")
        ax.axis("off")
    mask_png = op.abspath("mask_overlap.png")
    plt.savefig(mask_png, **pngkws)
    report.append(mask_png)

    def add_colorbar(f, cmap, low, high, left, width, fmt):
        cbar = np.outer(np.arange(0, 1, .01), np.ones(10))
        cbar_ax = f.add_axes([left, 0, width, .03])
        cbar_ax.imshow(cbar.T, aspect="auto", cmap=cmap)
        cbar_ax.axis("off")
        f.text(left - .01, .018, fmt % low, ha="right", va="center",
               color="white", size=13, weight="demibold")
        f.text(left + width + .01, .018, fmt % high, ha="left",
               va="center", color="white", size=13, weight="demibold")

    # Now plot the R2 images
    for fname, cmap in zip(r2_files, ["GnBu_r", "YlGn_r"]):
        f, axes = plt.subplots(**pltkws)
        r2data = nib.load(fname).get_data()
        r2data[bg == 0] = np.nan
        for i, ax in zip(slices, axes.ravel()):
            ax.imshow(bg[xslice, yslice, i].T,
                      cmap="gray", vmin=vmin, vmax=vmax)
            ax.imshow(r2data[xslice, yslice, i].T, cmap=cmap,
                      vmin=0, vmax=r2data.max())
            ax.imshow(mask[xslice, yslice, i].T, alpha=.5,
                      cmap=mask_cmap, interpolation="nearest")
            ax.axis("off")
        add_colorbar(f, cmap, 0, r2data.max(), .35, .3, "%.2f")
        savename = op.abspath(op.basename(fname).replace(".nii.gz", ".png"))
        plt.savefig(savename, **pngkws)
        report.append(savename)

    # Finally plot each zstat image
    for fname in zstat_files:
        zdata = nib.load(fname).get_data()
        zpos = zdata.copy()
        zneg = zdata.copy()
        zpos[zdata < 2.3] = np.nan
        zneg[zdata > -2.3] = np.nan
        zlow = 2.3
        zhigh = max(np.abs(zdata).max(), 2.3)
        f, axes = plt.subplots(**pltkws)
        for i, ax in zip(slices, axes.ravel()):
            ax.imshow(bg[xslice, yslice, i].T, cmap="gray",
                      vmin=vmin, vmax=vmax)
            ax.imshow(zpos[..., i].T, cmap="Reds_r",
                      vmin=zlow, vmax=zhigh)
            ax.imshow(zneg[..., i].T, cmap="Blues",
                      vmin=-zhigh, vmax=-zlow)
            ax.imshow(mask[xslice, yslice, i].T, alpha=.5,
                      cmap=mask_cmap, interpolation="nearest")
            ax.axis("off")
        add_colorbar(f, "Blues", -zhigh, -zlow, .15, .3, "%.1f")
        add_colorbar(f, "Reds_r", zlow, zhigh, .55, .3, "%.1f")

        savename = op.abspath(op.basename(fname).replace(".nii.gz", ".png"))
        f.savefig(savename, **pngkws)
        report.append(fname)

    return report
