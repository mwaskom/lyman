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

from nipype import Workflow, Node, IdentityInterface, Function
from nipype.interfaces import fsl

imports = ["import os",
           "import os.path as op",
           "import subprocess as sub",
           "import numpy as np",
           "import nibabel as nib",
           "import matplotlib as mpl",
           "import matplotlib.pyplot as plt",
           "import seaborn as sns",
           "import moss",
           "from nipype.interfaces import fsl"]


def create_ffx_workflow(name="mni_ffx", space="mni", contrasts=None):
    """Return a workflow object to execute a fixed-effects mode."""
    if contrasts is None:
        contrasts = []

    inputnode = Node(IdentityInterface(["copes",
                                        "varcopes",
                                        "masks",
                                        "dofs",
                                        "ss_files",
                                        "anatomy",
                                        "reg_file"]),
                     name="inputnode")

    # Fit the fixedfx model for each contrast
    ffxmodel = Node(Function(["contrasts",
                              "copes",
                              "varcopes",
                              "dofs",
                              "masks",
                              "reg_file"],
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
             ("masks", "masks"),
             ("reg_file", "reg_file")]),
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


def fixedfx_model(contrasts, copes, varcopes, dofs, masks, reg_file):
    """Fit the fixed effects model for each contrast."""
    n_con = len(contrasts)

    # Find the basic geometry of the image
    img = nib.load(copes[0])
    x, y, z = img.shape
    aff, hdr = img.get_affine(), img.get_header()

    # Get lists of files for each contrast
    copes = [[f for f in copes if "cope%d_" % (i + 1) in f]
             for i in range(n_con)]
    varcopes = [[f for f in varcopes if "varcope%d_" % (i + 1) in f]
                for i in range(n_con)]

    # Make an image with the DOF for each run
    dofs = np.array([np.loadtxt(f) for f in dofs])
    dof_data = np.ones((x, y, z, len(dofs))) * dofs

    # Find the intersection of the masks
    mask_data = [nib.load(f).get_data() for f in masks]
    common_mask = np.all(mask_data, axis=0)
    nib.Nifti1Image(common_mask, aff, hdr).to_filename("mask.nii.gz")

    # Run the flame models
    flame_results = []
    zstat_files = []
    for i, contrast in enumerate(contrasts):

        # Load each run of cope and varcope files into a list
        cs = [nib.load(f).get_data()[..., np.newaxis] for f in copes[i]]
        vs = [nib.load(f).get_data()[..., np.newaxis] for f in varcopes[i]]

        # Find all of the nonzero copes
        # This handles cases where there were no events for some of
        # the runs for the contrast we're currently dealing with
        good_cs = [not np.allclose(d, 0) for d in cs]
        good_vs = [not np.allclose(d, 0) for d in vs]
        good = np.all([good_cs, good_vs], axis=0)

        # Concatenate the cope and varcope data, saving only the good frames
        c_data = np.concatenate(cs, axis=-1)[:, :, :, good]
        v_data = np.concatenate(vs, axis=-1)[:, :, :, good]

        # Write out the concatenated copes and varcopes
        nib.Nifti1Image(c_data, aff, hdr).to_filename("cope_4d.nii.gz")
        nib.Nifti1Image(v_data, aff, hdr).to_filename("varcope_4d.nii.gz")

        # Write out a correctly sized design for this contrast
        fsl.L2Model(num_copes=int(good.sum())).run()

        # Mask the DOF data and write it out for this run
        contrast_dof = dof_data[:, :, :, good]
        nib.Nifti1Image(contrast_dof, aff, hdr).to_filename("dof.nii.gz")

        # Build the flamo commandline and run
        flamecmd = ["flameo",
                    "--cope=cope_4d.nii.gz",
                    "--varcope=varcope_4d.nii.gz",
                    "--mask=mask.nii.gz",
                    "--dvc=dof.nii.gz",
                    "--runmode=fe",
                    "--dm=design.mat",
                    "--tc=design.con",
                    "--cs=design.grp",
                    "--ld=" + contrast,
                    "--npo"]
        sub.check_output(flamecmd)

        # Rename the written file and append to the outputs
        for kind in ["cope", "varcope"]:
            os.rename(kind + "_4d.nii.gz",
                      "%s/%s_4d.nii.gz" % (contrast, kind))

        # Put the zstats and mask on the surface
        for hemi in ["lh", "rh"]:
            projcmd = ["mri_vol2surf",
                       "--mov", "%s/zstat1.nii.gz" % contrast,
                       "--reg", reg_file,
                       "--surf-fwhm", "5",
                       "--hemi", hemi,
                       "--projfrac-avg", "0", "1", ".1",
                       "--o", "%s/%s.zstat1.mgz" % (contrast, hemi)]
            sub.check_output(projcmd)

            # Mask image
            projcmd = ["mri_vol2surf",
                       "--mov", "%s/mask.nii.gz" % contrast,
                       "--reg", reg_file,
                       "--hemi", hemi,
                       "--projfrac-max", "0", "1", ".1",
                       "--o", "%s/%s.mask.mgz" % (contrast, hemi)]
            sub.check_output(projcmd)

        flame_results.append(op.abspath(contrast))
        zstat_files.append(op.abspath("%s/zstat1.nii.gz" % contrast))

    return flame_results, zstat_files


def fixedfx_r2(ss_files):
    """Find the R2 for the full fixedfx model."""
    out_files = []

    # First read the total sum of squares for each run
    ss_tot = [f for f in ss_files if "sstot" in f]
    tot_data = [nib.load(f).get_data() for f in ss_tot]

    # Sum across runs
    tot_sum = np.sum(tot_data, axis=0)

    # Get basic info about the image
    img = nib.load(ss_tot[0])
    aff, header = img.get_affine(), img.get_header()

    # Do the same processing for the full and main model
    for comp in ["full", "main"]:

        # Read in the residual sum of squares and take grand sum
        ss_res = [f for f in ss_files if "ssres_%s" % comp in f]
        res_data = [nib.load(f).get_data() for f in ss_res]
        res_sum = np.sum(res_data, axis=0)

        # Calculate the full model R2
        r2 = 1 - res_sum / tot_sum

        # Save an image with these data
        r2_img = nib.Nifti1Image(r2, aff, header)
        r2_file = op.abspath("r2_%s.nii.gz" % comp)
        r2_img.to_filename(r2_file)
        out_files.append(r2_file)

    return out_files


def fixedfx_report(space, anatomy, zstat_files, r2_files, masks):
    """Plot the resulting data."""
    sns.set()
    bg = nib.load(anatomy).get_data()

    mask_data = [nib.load(f).get_data() for f in masks]
    mask = np.where(np.all(mask_data, axis=0), np.nan, 1)
    mask[bg < moss.percentiles(bg, 5)] = np.nan

    # Find the plot parameters
    xdata = np.flatnonzero(bg.any(axis=1).any(axis=1))
    xslice = slice(xdata.min(), xdata.max() + 1)
    ydata = np.flatnonzero(bg.any(axis=0).any(axis=1))
    yslice = slice(ydata.min(), ydata.max() + 1)
    zdata = np.flatnonzero(bg.any(axis=0).any(axis=0))
    zmin, zmax = zdata.min(), zdata.max()

    step = 2 if space == "mni" else 1
    offset = 4 if space == "mni" else 0

    n_slices = (zmax - zmin) // step
    n_row, n_col = n_slices // 8, 8
    start = n_slices % n_col // step + zmin + offset
    figsize = (10, 1.375 * n_row)
    slices = (start + np.arange(zmax - zmin))[::step][:n_slices]
    pltkws = dict(nrows=int(n_row), ncols=int(n_col),
                  figsize=figsize, facecolor="k")
    pngkws = dict(dpi=100, bbox_inches="tight", facecolor="k", edgecolor="k")

    vmin, vmax = 0, moss.percentiles(bg, 95)
    mask_cmap = mpl.colors.ListedColormap(["#160016"])

    report = []

    def add_colorbar(f, cmap, low, high, left, width, fmt):
        cbar = np.outer(np.arange(0, 1, .01), np.ones(10))
        cbar_ax = f.add_axes([left, 0, width, .03])
        cbar_ax.imshow(cbar.T, aspect="auto", cmap=cmap)
        cbar_ax.axis("off")
        f.text(left - .01, .018, fmt % low, ha="right", va="center",
               color="white", size=13, weight="demibold")
        f.text(left + width + .01, .018, fmt % high, ha="left",
               va="center", color="white", size=13, weight="demibold")

    # Plot the mask edges
    f, axes = plt.subplots(**pltkws)
    mask_colors = sns.husl_palette(len(mask_data))
    mask_colors.reverse()
    cmaps = [mpl.colors.ListedColormap([c]) for c in mask_colors]
    for i, ax in zip(slices, axes.ravel()):
        ax.imshow(bg[xslice, yslice, i].T,
                  cmap="gray", vmin=vmin, vmax=vmax)
        for j, m in enumerate(mask_data):
            if m[xslice, yslice, i].any():
                ax.contour(m[xslice, yslice, i].T,
                           cmap=cmaps[j], linewidths=.75)
        ax.axis("off")
    text_min = max(.15, .5 - len(mask_data) * .05)
    text_max = min(.85, .5 + len(mask_data) * .05)
    text_pos = np.linspace(text_min, text_max, len(mask_data))
    for i, color in enumerate(mask_colors):
        f.text(text_pos[i], .03, "Run %d" % (i + 1), color=color,
               size=11, weight="demibold", ha="center", va="center")
    mask_png = op.abspath("mask_overlap.png")
    plt.savefig(mask_png, **pngkws)
    report.append(mask_png)
    plt.close(f)

    # Now plot the R2 images
    for fname, cmap in zip(r2_files, ["GnBu_r", "YlGn_r"]):
        f, axes = plt.subplots(**pltkws)
        r2data = nib.load(fname).get_data()
        rmax = r2data[~np.isnan(r2data)].max()
        r2data[bg == 0] = np.nan
        for i, ax in zip(slices, axes.ravel()):
            ax.imshow(bg[xslice, yslice, i].T,
                      cmap="gray", vmin=vmin, vmax=vmax)
            ax.imshow(r2data[xslice, yslice, i].T, cmap=cmap,
                      vmin=0, vmax=rmax, alpha=.8)
            ax.imshow(mask[xslice, yslice, i].T, alpha=.5,
                      cmap=mask_cmap, interpolation="nearest")
            ax.axis("off")
        savename = op.abspath(op.basename(fname).replace(".nii.gz", ".png"))
        add_colorbar(f, cmap, 0, rmax, .35, .3, "%.2f")
        plt.savefig(savename, **pngkws)
        report.append(savename)
        plt.close(f)

    # Finally plot each zstat image
    for fname in zstat_files:
        zdata = nib.load(fname).get_data()
        zpos = zdata.copy()
        zneg = zdata.copy()
        zpos[zdata < 2.3] = np.nan
        zneg[zdata > -2.3] = np.nan
        zlow = 2.3
        zhigh = max(np.abs(zdata).max(), 3.71)
        f, axes = plt.subplots(**pltkws)
        for i, ax in zip(slices, axes.ravel()):
            ax.imshow(bg[xslice, yslice, i].T, cmap="gray",
                      vmin=vmin, vmax=vmax)
            ax.imshow(zpos[xslice, yslice, i].T, cmap="Reds_r",
                      vmin=zlow, vmax=zhigh)
            ax.imshow(zneg[xslice, yslice, i].T, cmap="Blues",
                      vmin=-zhigh, vmax=-zlow)
            ax.imshow(mask[xslice, yslice, i].T, alpha=.5,
                      cmap=mask_cmap, interpolation="nearest")
            ax.axis("off")
        add_colorbar(f, "Blues", -zhigh, -zlow, .18, .23, "%.1f")
        add_colorbar(f, "Reds_r", zlow, zhigh, .59, .23, "%.1f")

        contrast = fname.split("/")[-2]
        os.mkdir(contrast)
        savename = op.join(contrast, "zstat1.png")
        f.savefig(savename, **pngkws)
        report.append(op.abspath(contrast))
        plt.close(f)

    return report
