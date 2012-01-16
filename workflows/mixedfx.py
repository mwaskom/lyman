from nipype.interfaces import fsl
from nipype.interfaces.utility import IdentityInterface
from nipype.pipeline.engine import Node, MapNode, Workflow


def create_volume_mixedfx_workflow(name="volume_group",
                                   regressors=[], contrasts=[],
                                   flame_mode="flame1",
                                   overlay_thresh=(2.3, 4.265),
                                   cluster_zthresh=2.3, grf_pthresh=0.05):

    inputnode = Node(IdentityInterface(fields=["copes",
                                               "varcopes",
                                               "dofs"]),
                     name="inputnode")

    mergecope = Node(fsl.Merge(dimension="t"),
                     name="mergecope")

    mergevarcope = Node(fsl.Merge(dimension="t"),
                        name="mergevarcope")

    mergedof = Node(fsl.Merge(dimension="t"),
                    name="mergedof")

    design = Node(fsl.MultipleRegressDesign(),
                  name="design")

    brain_mask = fsl.Info.standard_image("MNI152_T1_2mm_brain_mask.nii.gz")

    flameo = Node(fsl.FLAMEO(run_mode=flame_mode,
                             mask_file=brain_mask),
                  name="flameo")

    smoothest = MapNode(fsl.SmoothEstimate(mask_file=brain_mask),
                        iterfield=["zstat_file"],
                        name="smoothest")

    cluster = MapNode(fsl.Cluster(threshold=cluster_zthresh,
                                  pthreshold=0.05,
                                  out_threshold_file=True,
                                  out_index_file=True,
                                  out_localmax_txt_file=True),
                      iterfield=["in_file", "dlh", "volume"],
                      name="cluster")

    bg_image = fsl.Info.standard_image("avg152T1_brain.nii.gz")
    overlay = MapNode(fsl.Overlay(auto_thresh_bg=True,
                                  stat_thresh=overlay_thresh,
                                  background_image=bg_image),
                      iterfield=["stat_image"],
                      name="overlay")

    slicer = MapNode(fsl.Slicer(image_width=991),
                     iterfield=["in_file"],
                     name="slicer")
    slicer.inputs.sample_axial = 2

    outputnode = Node(IdentityInterface(fields=["flameo_stats",
                                                "thresh_zstat",
                                                "cluster_image",
                                                "cluster_peaks",
                                                "stat_png"]),
                      name="outputnode")

    group_anal = Workflow(name=name)

    group_anal.connect([
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
        (cluster, overlay,
            [("threshold_file", "stat_image")]),
        (overlay, slicer,
            [("out_file", "in_file")]),
        (slicer, outputnode,
            [("out_file", "stat_png")]),
        (cluster, outputnode,
            [("threshold_file", "thresh_zstat"),
             ("index_file", "cluster_image"),
             ("localmax_txt_file", "cluster_peaks")]),
        (flameo, outputnode,
            [("stats_dir", "flameo_stats")])
        ])

    return group_anal, inputnode, outputnode
