import os
from copy import deepcopy
import numpy as np
import pandas as pd
import nibabel as nib

import pytest

from lyman.frontend import LymanInfo


@pytest.fixture()
def execdir(tmpdir):

    origdir = tmpdir.chdir()
    yield tmpdir
    origdir.chdir()


@pytest.fixture()
def lyman_info(tmpdir):

    data_dir = tmpdir.mkdir("data")
    proc_dir = tmpdir.mkdir("proc")
    cache_dir = tmpdir.mkdir("cache")

    os.environ["SUBJECTS_DIR"] = str(data_dir)

    # TODO probably get these from default info functions
    scan_info = {
        "subj01": {
            "sess01":
                {"exp_alpha": ["run01", "run02"]},
            "sess02":
                {"exp_alpha": ["run01"],
                 "exp_beta": ["run01", "run02", "run03"]},
        },
        "subj02": {
            "sess01":
                {"exp_alpha": ["run01", "run02", "run03"]}
        },
    }

    contrasts = [
        ("a", ["a", "b"], [1, 0]),
        ("b", ["b"], [1]),
        ("a-b", ["a", "b"], [1, -1])
    ]

    info = LymanInfo().trait_set(
        data_dir=str(data_dir),
        proc_dir=str(proc_dir),
        cache_dir=str(cache_dir),
        scan_info=scan_info,
        phase_encoding="ap",
        fm_template="{session}_{encoding}.nii.gz",
        ts_template="{session}_{experiment}_{run}.nii.gz",
        sb_template="{session}_{experiment}_{run}_sbref.nii.gz",
        experiment_name="exp_alpha",
        crop_frames=2,
        tr=1.5,
        model_name="model_a",
        smooth_fwhm=4,
        surface_smoothing=True,
        interpolate_noise=True,
        hpf_cutoff=10,
        hrf_derivative=False,
        save_residuals=True,
        contrasts=contrasts,
    )

    subjects = ["subj01", "subj02"]
    sessions = None

    design = pd.DataFrame(dict(
        onset=[0, 6, 12, 18, 24],
        condition=["a", "b", "c", "b", "a"],
        session="sess01",
        run="run01",
    ))

    for subject in subjects:

        subject_dir = data_dir.mkdir(subject)
        subject_dir.mkdir("mri")
        subject_dir.mkdir("surf")
        subject_dir.mkdir("label")
        subject_dir.mkdir("func")
        design_dir = subject_dir.mkdir("design")
        design.to_csv(design_dir.join(
            "{experiment_name}-{model_name}.csv".format(**info.trait_get())
        ))

    vol_shape = 12, 8, 4
    n_tp = 20
    n_params = len(design["condition"].unique())

    return dict(
        info=info,
        subjects=subjects,
        sessions=sessions,
        proc_dir=proc_dir,
        data_dir=data_dir,

        vol_shape=vol_shape,
        n_tp=n_tp,
        n_params=n_params,
        design=design,
    )


@pytest.fixture()
def freesurfer(lyman_info):

    subject = "subj01"
    mri_dir = lyman_info["data_dir"].join(subject).join("mri")
    label_dir = lyman_info["data_dir"].join(subject).join("label")

    seed = sum(map(ord, "freesurfer"))
    rs = np.random.RandomState(seed)
    affine = np.eye(4)
    vol_shape = lyman_info["vol_shape"]

    mask = rs.choice([0, 1], vol_shape, p=[.2, .8])

    norm_data = rs.randint(0, 110, vol_shape) * mask
    norm_file = str(mri_dir.join("norm.mgz"))
    nib.save(nib.MGHImage(norm_data.astype("uint8"), affine), norm_file)

    orig_file = str(mri_dir.join("orig.mgz"))
    nib.save(nib.MGHImage(norm_data.astype("uint8"), affine), orig_file)

    wmparc_vals = [1000, 10, 11, 16, 8, 3000, 5001, 7, 46, 4]
    wmparc_data = rs.choice(wmparc_vals, vol_shape) * mask
    wmparc_file = str(mri_dir.join("wmparc.mgz"))
    nib.save(nib.MGHImage(wmparc_data.astype("int16"), affine), wmparc_file)

    n = 10
    fmt = ["%d", "%.3f", "%.3f", "%.3f", "%.9f"]
    label_data = np.c_[np.arange(n), np.zeros((n, 4))]
    label_files = {}
    for hemi in ["lh", "rh"]:
        fname = str(label_dir.join("{}.cortex.label".format(hemi)))
        label_files[hemi] = fname
        np.savetxt(fname, label_data, fmt=fmt, header=str(n))

    lyman_info.update(
        subject=subject,
        norm_file=norm_file,
        orig_file=orig_file,
        wmparc_file=wmparc_file,
        label_files=label_files,
    )
    return lyman_info


@pytest.fixture()
def template(freesurfer):

    subject = "subj01"
    template_dir = (freesurfer["proc_dir"]
                    .mkdir(subject)
                    .mkdir("template"))

    seed = sum(map(ord, "template"))
    rs = np.random.RandomState(seed)

    vol_shape = freesurfer["vol_shape"]
    affine = np.array([[-2, 0, 0, 10],
                       [0, -2, -1, 10],
                       [0, 1, 2, 5],
                       [0, 0, 0, 1]])

    reg_file = str(template_dir.join("anat2func.mat"))
    np.savetxt(reg_file, np.random.randn(4, 4))

    lut = pd.DataFrame([
            ["Unknown", 0, 0, 0, 0],
            ["Cortical-gray-matter", 59, 95, 138, 255],
            ["Subcortical-gray-matter", 91, 129, 129, 255],
            ["Brain-stem", 126, 163, 209, 255],
            ["Cerebellar-gray-matter", 168, 197, 233, 255],
            ["Superficial-white-matter", 206, 129, 134, 255],
            ["Deep-white-matter", 184, 103, 109, 255],
            ["Cerebellar-white-matter", 155, 78, 73, 255],
            ["CSF", 251, 221, 122, 255]
        ])
    lut_file = str(template_dir.join("seg.lut"))
    lut.to_csv(lut_file, sep="\t", header=False, index=True)

    seg_data = rs.randint(0, 7, vol_shape)
    seg_file = str(template_dir.join("seg.nii.gz"))
    nib.save(nib.Nifti1Image(seg_data, affine), seg_file)

    anat_data = rs.randint(0, 100, vol_shape)
    anat_file = str(template_dir.join("anat.nii.gz"))
    nib.save(nib.Nifti1Image(anat_data, affine), anat_file)

    mask_data = (seg_data > 0).astype(np.uint8)
    mask_file = str(template_dir.join("mask.nii.gz"))
    nib.save(nib.Nifti1Image(mask_data, affine), mask_file)

    n_verts = (seg_data == 1).sum()
    surf_ids = np.arange(n_verts)
    surf_data = np.full(vol_shape + (2,), -1, np.int)
    surf_data[seg_data == 1, 0] = surf_ids
    surf_data[seg_data == 1, 1] = surf_ids
    surf_file = str(template_dir.join("surf.nii.gz"))
    nib.save(nib.Nifti1Image(surf_data, affine), surf_file)

    mesh_name = "graymid"
    verts = rs.uniform(-1, 1, (n_verts, 3))
    faces = np.array([(i, i + 1, i + 2) for i in range(n_verts - 2)])
    surf_dir = freesurfer["data_dir"].join(subject).join("surf")
    mesh_files = (str(surf_dir.join("lh." + mesh_name)),
                  str(surf_dir.join("rh." + mesh_name)))
    for fname in mesh_files:
        nib.freesurfer.write_geometry(fname, verts, faces)

    freesurfer.update(
        vol_shape=vol_shape,
        subject=subject,
        lut_file=lut_file,
        seg_file=seg_file,
        reg_file=reg_file,
        anat_file=anat_file,
        mask_file=mask_file,
        surf_file=surf_file,
        mesh_name=mesh_name,
        mesh_files=mesh_files,
    )
    return freesurfer


@pytest.fixture()
def timeseries(template):

    seed = sum(map(ord, "timeseries"))
    rs = np.random.RandomState(seed)

    session = "sess01"
    run = "run01"

    exp_name = template["info"].experiment_name
    model_name = template["info"].model_name

    vol_shape = template["vol_shape"]
    n_tp = template["n_tp"]
    affine = np.eye(4)
    affine[:3, :3] *= 2

    timeseries_dir = (template["proc_dir"]
                      .join(template["subject"])
                      .mkdir(exp_name)
                      .mkdir("timeseries")
                      .mkdir("{}_{}".format(session, run)))

    model_dir = (template["proc_dir"]
                 .join(template["subject"])
                 .join(exp_name)
                 .mkdir(model_name)
                 .mkdir("{}_{}".format(session, run)))

    mask_data = nib.load(template["seg_file"]).get_data() > 0
    mask_data &= rs.uniform(0, 1, vol_shape) > .05
    mask_file = str(timeseries_dir.join("mask.nii.gz"))
    nib.save(nib.Nifti1Image(mask_data.astype(np.int), affine), mask_file)

    noise_data = mask_data & rs.choice([False, True], vol_shape, p=[.95, .05])
    noise_file = str(timeseries_dir.join("noise.nii.gz"))
    nib.save(nib.Nifti1Image(noise_data.astype(np.int), affine), noise_file)

    ts_shape = vol_shape + (n_tp,)
    ts_data = rs.normal(100, 5, ts_shape) * mask_data[..., np.newaxis]
    ts_file = str(timeseries_dir.join("func.nii.gz"))
    nib.save(nib.Nifti1Image(ts_data, affine), ts_file)

    mc_data = rs.normal(0, 1, (n_tp, 6))
    mc_file = str(timeseries_dir.join("mc.csv"))
    cols = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
    pd.DataFrame(mc_data, columns=cols).to_csv(mc_file)

    template.update(
        n_tp=n_tp,
        affine=affine,
        session=session,
        run=run,
        mask_file=mask_file,
        noise_file=noise_file,
        ts_file=ts_file,
        mc_file=mc_file,
        timeseries_dir=timeseries_dir,
        model_dir=model_dir,
    )
    return template


@pytest.fixture()
def modelfit(timeseries):

    seed = sum(map(ord, "modelfit"))
    rs = np.random.RandomState(seed)

    vol_shape = timeseries["vol_shape"]
    affine = timeseries["affine"]
    n_params = timeseries["n_params"]
    n_tp = timeseries["n_tp"]
    n_vox = np.product(vol_shape)

    model_dir = timeseries["model_dir"]

    seg_data = nib.load(timeseries["seg_file"]).get_data()
    mask_data = nib.load(timeseries["mask_file"]).get_data()
    mask_data = ((seg_data == 1) & (mask_data == 1)).astype(np.int)
    mask_file = str(model_dir.join("mask.nii.gz"))
    nib.save(nib.Nifti1Image(mask_data, affine), mask_file)

    beta_data = rs.normal(0, 1, vol_shape + (n_params,))
    beta_file = str(model_dir.join("beta.nii.gz"))
    nib.save(nib.Nifti1Image(beta_data, affine), beta_file)

    ols_data = np.empty((n_vox, n_params, n_params))
    for i in range(n_vox):
        X = rs.normal(0, 1, (n_tp, n_params))
        ols_data[i] = np.linalg.pinv(np.dot(X.T, X))
    ols_data = ols_data.reshape(vol_shape + (n_params ** 2,))
    ols_file = str(model_dir.join("ols.nii.gz"))
    nib.save(nib.Nifti1Image(ols_data, affine), ols_file)

    error_data = rs.uniform(0, 5, vol_shape)
    error_file = str(model_dir.join("error.nii.gz"))
    nib.save(nib.Nifti1Image(error_data, affine), error_file)

    design_data = rs.normal(0, 1, (n_tp, n_params))
    columns = np.sort(timeseries["design"]["condition"].unique())
    design_file = str(model_dir.join("design.csv"))
    pd.DataFrame(design_data, columns=columns).to_csv(design_file, index=False)

    timeseries.update(
        n_params=n_params,
        mask_file=mask_file,
        beta_file=beta_file,
        ols_file=ols_file,
        error_file=error_file,
        design_file=design_file,
    )
    return timeseries


@pytest.fixture()
def modelres(modelfit):

    seed = sum(map(ord, "modelres"))
    rs = np.random.RandomState(seed)

    vol_shape = modelfit["vol_shape"]
    affine = modelfit["affine"]

    name_lists = [
        ["a", "b", "c", "a-b"],
        ["a", "b", "a-b"],
    ]
    run_ns = [len(n_list) for n_list in name_lists]

    info = deepcopy(modelfit["info"])
    info.contrasts.insert(2, ("c", ["c"], [1]))

    exp_name = modelfit["info"].experiment_name
    model_name = modelfit["info"].model_name
    session = "s1"
    runs = ["r1", "r2"]

    model_dir_base = (modelfit["proc_dir"]
                      .join(modelfit["subject"])
                      .join(exp_name)
                      .join(model_name))
    model_dirs = [
        model_dir_base.mkdir("{}_{}".format(session, run))
        for run in runs
    ]

    con_data = [rs.normal(0, 5, vol_shape + (n,)) for n in run_ns]
    con_files = [str(d.join("contrast.nii.gz")) for d in model_dirs]
    for d, f in zip(con_data, con_files):
        nib.save(nib.Nifti1Image(d, affine), f)

    var_data = [rs.uniform(0, 5, vol_shape + (n,)) for n in run_ns]
    var_files = [str(d.join("variance.nii.gz")) for d in model_dirs]
    for d, f in zip(var_data, var_files):
        nib.save(nib.Nifti1Image(d, affine), f)

    name_files = [str(d.join("contrast.txt")) for d in model_dirs]
    for l, f in zip(name_lists, name_files):
        np.savetxt(f, l, "%s")

    modelfit.update(
        info=info,
        contrast_files=con_files,
        variance_files=var_files,
        name_files=name_files,
    )
    return modelfit


@pytest.fixture
def meshdata(execdir):

    verts = np.array([[0, 0, 0],
                      [1, 0, 0],
                      [1, 1, 1],
                      [2, 0, 0],
                      [2, 2, 2]], np.float)

    faces = np.array([[0, 1, 2],
                      [0, 2, 3],
                      [2, 3, 4]], np.int)

    sqrt2 = np.sqrt(2)
    sqrt3 = np.sqrt(3)
    sqrt8 = np.sqrt(8)

    neighbors = {0: {1: 1.0, 2: sqrt3, 3: 2.0},
                 1: {0: 1.0, 2: sqrt2},
                 2: {0: sqrt3, 1: sqrt2, 3: sqrt3, 4: sqrt3},
                 3: {0: 2.0, 2: sqrt3, 4: sqrt8},
                 4: {2: sqrt3, 3: sqrt8}}

    subj = "subj01"
    surf = "white"

    surf_dir = execdir.mkdir(subj).mkdir("surf")
    for hemi in ["lh", "rh"]:
        fname = str(surf_dir.join("{}.{}".format(hemi, surf)))
        nib.freesurfer.write_geometry(fname, verts, faces)

    meshdata = dict(
        verts=verts,
        faces=faces,
        neighbors=neighbors,
        fname=fname,
        subj=subj,
        surf=surf,
        hemi=hemi,
    )

    orig_subjects_dir = os.environ.get("SUBJECTS_DIR", None)
    os.environ["SUBJECTS_DIR"] = str(execdir)

    yield meshdata

    if orig_subjects_dir is None:
        del os.environ["SUBJECTS_DIR"]
    else:
        os.environ["SUBJECTS_DIR"] = orig_subjects_dir
