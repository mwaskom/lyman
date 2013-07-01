from nipype.interfaces import fsl
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.pipeline.engine import Node, MapNode, Workflow
from .. import tools


def create_volume_mixedfx_workflow(name="volume_group",
                                   subject_list=[],
                                   regressors=[], contrasts=[],
                                   flame_mode="flame1",
                                   overlay_thresh=(2.3, 4.265),
                                   cluster_zthresh=2.3, grf_pthresh=0.05):

    inputnode = Node(IdentityInterface(fields=["l1_contrast",
                                               "copes",
                                               "varcopes",
                                               "dofs"]),
                     name="inputnode")

    mergecope = Node(fsl.Merge(dimension="t"),
                     name="mergecope")

    mergevarcope = Node(fsl.Merge(dimension="t"),
                        name="mergevarcope")

    mergedof = Node(fsl.Merge(dimension="t"),
                    name="mergedof")

    design = Node(fsl.MultipleRegressDesign(regressors=regressors,
                                            contrasts=contrasts),
                  name="design")

    brain_mask = fsl.Info.standard_image("MNI152_T1_2mm_brain_mask.nii.gz")
    bg_image = fsl.Info.standard_image("avg152T1_brain.nii.gz")

    flameo = Node(fsl.FLAMEO(run_mode=flame_mode,
                             mask_file=brain_mask),
                  name="flameo")

    maskpng = Node(Function(input_names=["varcope_file", 
                                         "background_file",
                                         "brain_mask"],
                            output_names=["out_file"],
                            function=mfx_mask_func),
                   name="maskpng")
    maskpng.inputs.brain_mask = brain_mask
    maskpng.inputs.background_file = bg_image

    smoothest = MapNode(fsl.SmoothEstimate(mask_file=brain_mask),
                        iterfield=["zstat_file"],
                        name="smoothest")

    cluster = MapNode(fsl.Cluster(threshold=cluster_zthresh,
                                  pthreshold=0.05,
                                  out_threshold_file=True,
                                  out_index_file=True,
                                  out_localmax_txt_file=True,
                                  use_mm=True),
                      iterfield=["in_file", "dlh", "volume"],
                      name="cluster")

    overlay = MapNode(fsl.Overlay(auto_thresh_bg=True,
                                  stat_thresh=overlay_thresh,
                                  background_image=bg_image),
                      iterfield=["stat_image"],
                      name="overlay")

    slicer = MapNode(fsl.Slicer(image_width=872),
                     iterfield=["in_file"],
                     name="slicer")
    slicer.inputs.sample_axial = 2

    boxplot = MapNode(Function(input_names=["cope_file", "localmax_file"],
                               output_names=["out_file"],
                               function=mfx_boxplot),
                      iterfield=["localmax_file"],
                      name="boxplot")

    peaktable = MapNode(Function(input_names=["localmax_file"],
                                 output_names=["out_file"],
                                 function=tools.cluster_to_rst),
                        iterfield=["localmax_file"],
                        name="peaktable")

    # Build pdf and html reports
    report = Node(Function(input_names=["subject_list",
                                        "l1_contrast",
                                        "zstat_pngs",
                                        "boxplots",
                                        "peak_tables",
                                        "mask_png",
                                        "contrasts"],
                              output_names=["reports"],
                              function=write_mfx_report),
                     name="report")
    report.inputs.subject_list = subject_list
    report.inputs.contrasts = contrasts

    outputnode = Node(IdentityInterface(fields=["flameo_stats",
                                                "thresh_zstat",
                                                "cluster_image",
                                                "cluster_peaks",
                                                "zstat_pngs",
                                                "mask_png",
                                                "boxplots",
                                                "reports"]),
                      name="outputnode")

    group_anal = Workflow(name=name)

    group_anal.connect([
        (inputnode, mergecope,
            [("copes", "in_files")]),
        (inputnode, mergevarcope,
            [("varcopes", "in_files")]),
        (inputnode, mergedof,
            [("dofs", "in_files")]),
        (inputnode, report,
            [("l1_contrast", "l1_contrast")]),
        (mergecope, flameo,
            [("merged_file", "cope_file")]),
        (mergevarcope, flameo,
            [("merged_file", "var_cope_file")]),
        (mergevarcope, maskpng,
            [("merged_file", "varcope_file")]),
        (mergedof, flameo,
            [("merged_file", "dof_var_cope_file")]),
        (design, flameo,
            [("design_con", "t_con_file"),
             ("design_grp", "cov_split_file"),
             ("design_mat", "design_file")]),
        (flameo, smoothest,
            [("zstats", "zstat_file")]),
        (smoothest, cluster,
            [("dlh", "dlh"),
             ("volume", "volume")]),
        (flameo, cluster,
            [("zstats", "in_file")]),
        (mergecope, boxplot,
            [("merged_file", "cope_file")]),
        (cluster, boxplot,
            [("localmax_txt_file", "localmax_file")]),
        (cluster, peaktable,
            [("localmax_txt_file", "localmax_file")]),
        (cluster, overlay,
            [("threshold_file", "stat_image")]),
        (overlay, slicer,
            [("out_file", "in_file")]),
        (slicer, outputnode,
            [("out_file", "zstat_pngs")]),
        (slicer, report,
            [("out_file", "zstat_pngs")]),
        (maskpng, report,
            [("out_file", "mask_png")]),
        (boxplot, report,
            [("out_file", "boxplots")]),
        (peaktable, report,
            [("out_file", "peak_tables")]),
        (report, outputnode,
            [("reports", "reports")]),
        (cluster, outputnode,
            [("threshold_file", "thresh_zstat"),
             ("index_file", "cluster_image"),
             ("localmax_txt_file", "cluster_peaks")]),
        (boxplot, outputnode,
            [("out_file", "boxplots")]),
        (maskpng, outputnode,
            [("out_file", "mask_png")]),
        (flameo, outputnode,
            [("stats_dir", "flameo_stats")])
        ])

    return group_anal, inputnode, outputnode


def mfx_mask_func(varcope_file, brain_mask, background_file):
    from os.path import abspath
    from nibabel import load, Nifti1Image
    from subprocess import check_output
    import numpy as np
    varcope_img = load(varcope_file)
    varcope_data = varcope_img.get_data()
    mask_img = load(brain_mask)
    mask_data = mask_img.get_data()

    pos_vars = np.all(varcope_data, axis=-1)
    pos_vars = np.logical_and(pos_vars, mask_data)

    pos_img = Nifti1Image(pos_vars,
                          mask_img.get_affine(),
                          mask_img.get_header())
    pos_file = "pos_var.nii.gz"
    pos_img.to_filename(pos_file)

    overlay_cmd = ["overlay", "1", "0", background_file,
                   "-a", pos_file, "0.5", "1.5", "pos_overlay.nii.gz"]
    check_output(overlay_cmd)

    out_file = abspath("pos_variance.png")
    slicer_cmd = ["slicer", "pos_overlay.nii.gz", "-S", "2", "872", out_file]
    check_output(slicer_cmd)

    return out_file


def mfx_boxplot(cope_file, localmax_file):
    """Plot the distribution of fixed effects COPEs at each local maximum."""
    from os.path import abspath
    from nibabel import load
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    sns.set()

    out_file = abspath("peak_boxplot.png")
    peak_array = np.loadtxt(localmax_file, int, skiprows=1, usecols=(2, 3, 4))

    if not peak_array.any():
        # If there wre no significant peaks, return an empty text file
        with open(out_file, "w") as f:
            f.write("")
        return out_file

    cope_data = load(cope_file).get_data()
    peak_dists = []
    for coords in peak_array:
        peak_dists.append(cope_data[tuple(coords)])
    peak_dists.reverse()
    n_peaks = len(peak_dists)
    fig = plt.figure(figsize=(7, float(n_peaks) / 2 + 0.5))
    ax = fig.add_subplot(111)
    sns.boxplot(peak_dists, color="PaleGreen", vert=0, widths=0.5, ax=ax)
    ax.axvline(0, c="#222222", ls="--")
    labels = range(1, n_peaks + 1)
    labels.reverse()
    ax.set_yticklabels(labels)
    ax.set_ylabel("Local Maximum")
    ax.set_xlabel("COPE Value")
    ax.set_title("COPE Distributions")
    plt.savefig(out_file)

    return out_file


def write_mfx_report(subject_list, l1_contrast, mask_png,
                     zstat_pngs, peak_tables, boxplots, contrasts):
    import time
    from lyman.tools import write_workflow_report
    from lyman.workflows.reporting import mfx_report_template

    # Fill in the initial report template dict
    report_dict = dict(now=time.asctime(),
                       l1_contrast=l1_contrast,
                       subject_list=", ".join(subject_list),
                       mask_png=mask_png,
                       n_subs=len(subject_list))

    # Add the zstat image templates and update the dict
    for i, con in enumerate(contrasts, 1):
        report_dict["con%d_name" % i] = con[0]
        report_dict["zstat%d_png" % i] = zstat_pngs[i - 1]
        report_dict["boxplot%d_png" % i] = boxplots[i - 1]
        header = "Zstat %d: %s" % (i, con[0])
        mfx_report_template = "\n".join([
             mfx_report_template,
             header,
             "".join(["^" for l in header]),
             "",
             "".join([".. image:: %(zstat", str(i), "_png)s"]),
             "    :width: 6.5in",
             "",
             "Local Maxima",
             "^^^^^^^^^^^^",
             "",
             ])
        mfx_report_template = "\n".join([
             mfx_report_template,
             open(peak_tables[i - 1]).read()])

        mfx_report_template = "\n".join([
            mfx_report_template,
            "\n".join([
                "",
                "COPE Distributions",
                "^^^^^^^^^^^^^^^^^^",
                "",
                "".join([".. image:: %(boxplot", str(i), "_png)s"]),
                "",
                ])
            ])

    out_files = write_workflow_report("mfx",
                                      mfx_report_template,
                                      report_dict)
    return out_files
