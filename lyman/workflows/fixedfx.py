"""Fixed effects model to combine across runs for a single subject."""
import os
import os.path as op
import numpy as np
from scipy import stats
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns

from nipype import fsl, Workflow, Node, IdentityInterface, Function

imports = ["import os",
           "import os.path as op",
           "import json",
           "import numpy as np",
           "from scipy import stats",
           "import pandas as pd",
           "import nibabel as nib",
           "import matplotlib.pyplot as plt",
           "import seaborn as sns"]


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
    ffxdesign = Node(fsl.L2Model(), "ffxdesign")

    # Fit the fixedfx model for each contrast
    ffxmodel = Node(Function(["contrasts",
                              "copes",
                              "varcopes",
                              "dofs",
                              "masks",
                              "design_mat",
                              "design_con",
                              "design_grp"],
                             ["flameo_results",
                              "zstat_files"],
                             fixedfx_model,
                             imports),
                    "ffxmodel")

    # Calculate the fixed effects Rsquared maps
    ffxr2 = Node(Function(["ss_files"], ["r2_files"],
                          fixedfx_r2, imports),
                 "ffxr2")

    # Plot the fixedfx results
    report = Node(Function(["contrasts",
                            "anatomy",
                            "zstat_files",
                            "r2_files",
                            "masks"],
                           ["report"],
                           fixedfx_report,
                           imports),
                  "report")

    ffx = Workflow(name=name)
    ffx.connect([
        (inputnode, copemerge,
            [(("copes", force_list), "in_files")]),
        (inputnode, varcopemerge,
            [(("varcopes", force_list), "in_files")]),
        (inputnode, createdof,
            [("copes", "copes"),
             ("dof_files", "dof_files")]),
        (copemerge, flameo,
            [("merged_file", "cope_file")]),
        (varcopemerge, flameo,
            [("merged_file", "var_cope_file")]),
        (createdof, flameo,
            [("out_file", "dof_var_cope_file")]),
        (inputnode, getmask,
            [("masks", "masks"),
             ("background_file", "background_file")]),
        (getmask, flameo,
            [("mask_file", "mask_file")]),
        (inputnode, ffxmodel,
            [(("copes", length), "num_copes")]),
        (ffxmodel, flameo,
            [("design_mat", "design_file"),
             ("design_con", "t_con_file"),
             ("design_grp", "cov_split_file")]),
        (inputnode, report,
            [("subject_id", "subject_id"),
             ("contrast", "contrast")]),
        (getmask, report,
            [("mask_png", "mask_png")]),
        (flameo, outputnode,
            [("stats_dir", "stats")]),
        (getmask, outputnode,
            [("mask_png", "mask_png")]),
        (report, outputnode,
            [("reports", "report")]),
        ])

    return ffx, inputnode, outputnode


def create_dof_image(copes, dof_files):
    """Create images where voxel values are DOF for that run."""
    from os.path import abspath
    from numpy import ones, int16
    from nibabel import load, Nifti1Image
    from lyman.workflows.fixedfx import force_list
    copes = force_list(copes)
    dof_files = force_list(dof_files)
    cope_imgs = map(load, copes)
    data_shape = list(cope_imgs[0].shape) + [len(cope_imgs)]
    dof_data = ones(data_shape, int16)
    for i, img in enumerate(cope_imgs):
        dof_val = int(open(dof_files[i]).read().strip())
        dof_data[..., i] *= dof_val

    template_img = load(copes[0])
    dof_hdr = template_img.get_header()
    dof_hdr.set_data_dtype(int16)
    dof_img = Nifti1Image(dof_data,
                          template_img.get_affine(),
                          dof_hdr)
    out_file = abspath("dof.nii.gz")
    dof_img.to_filename(out_file)
    return out_file


def create_ffx_mask(masks, background_file):
    """Create a mask for areas that are nonzero in all masks"""
    import os
    from os.path import abspath, join
    from nibabel import load, Nifti1Image
    from numpy import zeros
    from subprocess import call
    from lyman.workflows.fixedfx import force_list
    masks = force_list(masks)
    mask_imgs = map(load, masks)
    data_shape = list(mask_imgs[0].shape) + [len(mask_imgs)]
    mask_data = zeros(data_shape, int)
    for i, img in enumerate(mask_imgs):
        data = img.get_data()
        data[data < 1e-4] = 0
        mask_data[..., i] = img.get_data()

    out_mask_data = mask_data.all(axis=-1)
    sum_data = mask_data.astype(bool).sum(axis=-1)

    template_img = load(masks[0])
    template_affine = template_img.get_affine()
    template_hdr = template_img.get_header()

    mask_img = Nifti1Image(out_mask_data, template_affine, template_hdr)
    mask_file = abspath("fixed_effects_mask.nii.gz")
    mask_img.to_filename(mask_file)

    sum_img = Nifti1Image(sum_data, template_affine, template_hdr)
    sum_file = abspath("mask_overlap.nii.gz")
    sum_img.to_filename(sum_file)

    overlay_file = abspath("mask_overlap_overlay.nii.gz")
    call(["overlay", "1", "0", background_file, "-a",
          sum_file, "1", str(len(masks)), overlay_file])
    native = sum_img.shape[-1] < 50
    width = 750 if native else 872
    sample = 1 if native else 2
    mask_png = abspath("mask_overlap.png")
    lut = join(os.environ["FSLDIR"], "etc/luts/renderjet.lut")
    call(["slicer", overlay_file, "-l", lut, "-S",
          str(sample), str(width), mask_png])

    return mask_file, mask_png


def write_ffx_report(subject_id, mask_png, zstat_pngs, contrast):
    import time
    from lyman.tools import write_workflow_report
    from lyman.workflows.reporting import ffx_report_template

    # Fill in the report template dict
    report_dict = dict(now=time.asctime(),
                       subject_id=subject_id,
                       contrast_name=contrast,
                       mask_png=mask_png,
                       zstat_png=zstat_pngs[0])

    out_files = write_workflow_report("ffx",
                                      ffx_report_template,
                                      report_dict)
    return out_files


def force_list(f):
    if not isinstance(f, list):
        return [f]
    return f


def length(x):
    if isinstance(x, list):
        return len(x)
    return 1


# Main interface functions
# ------------------------


def fixedfx_model(contrasts, copes, varcopes, dofs, masks,
                  design_mat, design_con, design_grp):
    """Fit the fixed effects model for each contrast."""
    pass


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


def fixedfx_report(contrasts, anatomy, zstat_files, r2_files, masks):
    """Plot the resulting data."""
    pass
