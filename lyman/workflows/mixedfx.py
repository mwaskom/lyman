import os
import re
import os.path as op

import numpy as np
import pandas as pd
import nibabel as nib

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
                                 imports=[
                    "import os.path as op",
                    "import pandas as pd",
                    "import numpy as np",
                    "from lyman.tools import locate_peaks, vox_to_mni"],
                                 function=tools.cluster_table),
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
             ("index_file", "cluster_image")]),
        (peaktable, outputnode,
             [("out_file", "cluster_peaks")]),
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
    sns.boxplot(np.transpose(peak_dists), color="PaleGreen",
                vert=0, widths=0.5, ax=ax)
    ax.axvline(0, c="#222222", ls="--")
    labels = range(1, n_peaks + 1)
    labels.reverse()
    ax.set_yticklabels(labels)
    ax.set_ylabel("Local Maximum")
    ax.set_xlabel("COPE Value")
    ax.set_title("COPE Distributions")
    plt.tight_layout()
    plt.savefig(out_file)

    return out_file


def write_mfx_report(subject_list, l1_contrast, mask_png,
                     zstat_pngs, peak_tables, boxplots, contrasts):
    import time
    import pandas as pd
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
             "",
             ])
        peak_csv = peak_tables[i - 1]
        peak_df = pd.read_csv(peak_csv, index_col="Peak").reset_index()
        peak_str = peak_df.to_string(index=False)
        peak_lines = ["    " + l for l in peak_str.split("\n")]
        peak_str = "::\n\n" + "\n".join(peak_lines) + "\n"

        mfx_report_template += peak_str

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

def cluster_table(localmax_file):
    """Add some info to an FSL cluster file and format it properly."""
    df = pd.read_table(localmax_file, delimiter="\t")
    df = df[["Cluster Index", "Value", "x", "y", "z"]]
    df.columns = ["Cluster", "Value", "x", "y", "z"]
    df.index.name = "Peak"

    # Find out where the peaks most likely are
    coords = ["x", "y", "z"]
    loc_df = locate_peaks(np.array(df[coords]))
    df = pd.concat([df, loc_df], axis=1)
    mni_coords = vox_to_mni(np.array(df[coords])).T
    for i, ax in enumerate(coords):
        df[ax] = mni_coords[i]

    out_file = op.abspath(op.basename(localmax_file[:-3] + "csv"))
    df.to_csv(out_file)
    return out_file


def locate_peaks(vox_coords):
    """Find most probable region in HarvardOxford Atlas of a vox coord."""
    sub_names = harvard_oxford_sub_names
    ctx_names = harvard_oxford_ctx_names
    at_dir = op.join(os.environ["FSLDIR"], "data", "atlases")
    ctx_data = nib.load(op.join(at_dir, "HarvardOxford",
                            "HarvardOxford-cort-prob-2mm.nii.gz")).get_data()
    sub_data = nib.load(op.join(at_dir, "HarvardOxford",
                            "HarvardOxford-sub-prob-2mm.nii.gz")).get_data()

    loc_list = []
    for coord in vox_coords:
        coord = tuple(coord)
        ctx_index = np.argmax(ctx_data[coord])
        ctx_prob = ctx_data[coord][ctx_index]
        sub_index = np.argmax(sub_data[coord])
        sub_prob = sub_data[coord][sub_index]

        if not max(sub_prob, ctx_prob):
            loc_list.append(("Unknown", 0))
            continue
        if not ctx_prob and sub_index in [0, 11]:
            loc_list.append((sub_names[sub_index], sub_prob))
            continue
        if sub_prob > ctx_prob and sub_index not in [0, 1, 11, 12]:
            loc_list.append((sub_names[sub_index], sub_prob))
            continue
        loc_list.append((ctx_names[ctx_index], ctx_prob))

    return pd.DataFrame(loc_list, columns=["MaxProb Region", "Prob"])


def shorten_name(region_name, atlas):
    """Implement regexp sub for verbose Harvard Oxford Atlas region."""
    sub_list = dict(ctx=harvard_oxford_ctx_subs,
                    sub=harvard_oxford_sub_subs)
    for pat, rep in sub_list[atlas]:
        region_name = re.sub(pat, rep, region_name).strip()
    return region_name


def vox_to_mni(vox_coords):
    """Given ijk voxel coordinates, return xyz from image affine.

    The _to_mni part is rather a misnomer, although this at the moment
    only gets used in the group volume workflows.

    """
    import numpy as np
    from nibabel import load
    from nipype.interfaces.fsl import Info

    mni_file = Info.standard_image("avg152T1.nii.gz")
    aff = load(mni_file).get_affine()
    mni_coords = np.zeros_like(vox_coords)
    for i, coord in enumerate(vox_coords):
        coord = coord.astype(float)
        mni_coords[i] = np.dot(aff, np.r_[coord, 1])[:3].astype(int)
    return mni_coords


harvard_oxford_sub_subs = [
    ("Left", "L"),
    ("Right", "R"),
    ("Cerebral Cortex", "Ctx"),
    ("Cerebral White Matter", "Cereb WM"),
    ("Lateral Ventrica*le*", "LatVent"),
]

harvard_oxford_ctx_subs = [
    ("Superior", "Sup"),
    ("Middle", "Mid"),
    ("Inferior", "Inf"),
    ("Lateral", "Lat"),
    ("Medial", "Med"),
    ("Frontal", "Front"),
    ("Parietal", "Par"),
    ("Temporal", "Temp"),
    ("Occipital", "Occ"),
    ("Cingulate", "Cing"),
    ("Cortex", "Ctx"),
    ("Gyrus", "G"),
    ("Sup Front G", "SFG"),
    ("Mid Front G", "MFG"),
    ("Inf Front G", "IFG"),
    ("Sup Temp G", "STG"),
    ("Mid Temp G", "MTG"),
    ("Inf Temp G", "ITG"),
    ("Parahippocampal", "Parahip"),
    ("Juxtapositional", "Juxt"),
    ("Intracalcarine", "Intracalc"),
    ("Supramarginal", "Supramarg"),
    ("Supracalcarine", "Supracalc"),
    ("Paracingulate", "Paracing"),
    ("Fusiform", "Fus"),
    ("Orbital", "Orb"),
    ("Opercul[ua][mr]", "Oper"),
    ("temporooccipital", "tempocc"),
    ("triangularis", "triang"),
    ("opercularis", "oper"),
    ("division", ""),
    ("par[st] *", ""),
    ("anterior", "ant"),
    ("posterior", "post"),
    ("superior", "sup"),
    ("inferior", "inf"),
    (" +", " "),
    ("\(.+\)", ""),
]

harvard_oxford_sub_names = [
    'L Cereb WM',
    'L Ctx',
    'L LatVent',
    'L Thalamus',
    'L Caudate',
    'L Putamen',
    'L Pallidum',
    'Brain-Stem',
    'L Hippocampus',
    'L Amygdala',
    'L Accumbens',
    'R Cereb WM',
    'R Ctx',
    'R LatVent',
    'R Thalamus',
    'R Caudate',
    'R Putamen',
    'R Pallidum',
    'R Hippocampus',
    'R Amygdala',
    'R Accumbens']

harvard_oxford_ctx_names = [
    'Front Pole',
    'Insular Ctx',
    'SFG',
    'MFG',
    'IFG, triang',
    'IFG, oper',
    'Precentral G',
    'Temp Pole',
    'STG, ant',
    'STG, post',
    'MTG, ant',
    'MTG, post',
    'MTG, tempocc',
    'ITG, ant',
    'ITG, post',
    'ITG, tempocc',
    'Postcentral G',
    'Sup Par Lobule',
    'Supramarg G, ant',
    'Supramarg G, post',
    'Angular G',
    'Lat Occ Ctx, sup',
    'Lat Occ Ctx, inf',
    'Intracalc Ctx',
    'Front Med Ctx',
    'Juxt Lobule Ctx',
    'Subcallosal Ctx',
    'Paracing G',
    'Cing G, ant',
    'Cing G, post',
    'Precuneous Ctx',
    'Cuneal Ctx',
    'Front Orb Ctx',
    'Parahip G, ant',
    'Parahip G, post',
    'Lingual G',
    'Temp Fus Ctx, ant',
    'Temp Fus Ctx, post',
    'Temp Occ Fus Ctx',
    'Occ Fus G',
    'Front Oper Ctx',
    'Central Oper Ctx',
    'Par Oper Ctx',
    'Planum Polare',
    'Heschl"s G',
    'Planum Tempe',
    'Supracalc Ctx',
    'Occ Pole']
