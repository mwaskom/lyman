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
    proc_dir = tmpdir.mkdir("analysis")
    cache_dir = tmpdir.mkdir("cache")

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
        hpf_cutoff=10,
        save_residuals=True,
        # TODO FIX
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
        subject_dir.mkdir("func")
        design_dir = subject_dir.mkdir("design")
        design.to_csv(design_dir.join("model_a.csv"))

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
    )


@pytest.fixture()
def freesurfer(lyman_info):

    subject = "subj01"
    mri_dir = lyman_info["data_dir"].join(subject).join("mri")

    seed = sum(map(ord, "freesurfer"))
    rs = np.random.RandomState(seed)
    affine = np.eye(4)
    vol_shape = lyman_info["vol_shape"]

    mask = rs.choice([0, 1], vol_shape, p=[.2, .8])

    norm_data = rs.randint(0, 110, vol_shape) * mask
    norm_file = str(mri_dir.join("norm.mgz"))
    nib.save(nib.MGHImage(norm_data.astype("uint8"), affine), norm_file)

    wmparc_vals = [1000, 10, 11, 16, 8, 3000, 5001, 7, 46, 4]
    wmparc_data = rs.choice(wmparc_vals, vol_shape) * mask
    wmparc_file = str(mri_dir.join("wmparc.mgz"))
    nib.save(nib.MGHImage(wmparc_data.astype("int16"), affine), wmparc_file)

    lyman_info.update(
        subject=subject,
        norm_file=norm_file,
        wmparc_file=wmparc_file,
    )
    return lyman_info


@pytest.fixture()
def template(lyman_info):

    subject = "subj01"
    template_dir = (lyman_info["proc_dir"]
                    .mkdir(subject)
                    .mkdir("template"))

    seed = sum(map(ord, "template"))
    rs = np.random.RandomState(seed)

    vol_shape = lyman_info["vol_shape"]
    affine = np.array([[-2, 0, 0, 10],
                       [0, -2, -1, 10],
                       [0, 1, 2, 5],
                       [0, 0, 0, 1]])

    reg_file = str(template_dir.join("anat2func.mat"))
    np.savetxt(reg_file, np.random.randn(4, 4))

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

    verts = rs.uniform(-1, 1, (n_verts, 3))
    faces = np.array([(i, i + 1, i + 2) for i in range(n_verts - 2)])
    surf_dir = lyman_info["data_dir"].join(subject).join("surf")
    mesh_files = (str(surf_dir.join("lh.graymid")),
                  str(surf_dir.join("rh.graymid")))
    for fname in mesh_files:
        nib.freesurfer.write_geometry(fname, verts, faces)

    lyman_info.update(
        vol_shape=vol_shape,
        subject=subject,
        reg_file=reg_file,
        seg_file=seg_file,
        anat_file=anat_file,
        mask_file=mask_file,
        surf_file=surf_file,
        mesh_files=mesh_files,
    )
    return lyman_info


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

    model_dir = timeseries["model_dir"]

    seg_data = nib.load(timeseries["seg_file"]).get_data()
    mask_data = nib.load(timeseries["mask_file"]).get_data()
    mask_data = ((seg_data == 1) & (mask_data == 1)).astype(np.int)
    mask_file = str(model_dir.join("mask.nii.gz"))
    nib.save(nib.Nifti1Image(mask_data, affine), mask_file)

    beta_data = rs.normal(0, 1, vol_shape + (n_params,))
    beta_file = str(model_dir.join("beta.nii.gz"))
    nib.save(nib.Nifti1Image(beta_data, affine), beta_file)

    ols_data = rs.uniform(0, 1, vol_shape + (n_params, n_params))
    ols_data += ols_data.transpose(0, 1, 2, 4, 3)
    ols_data = ols_data.reshape(vol_shape + (n_params ** 2,))
    ols_file = str(model_dir.join("ols.nii.gz"))
    nib.save(nib.Nifti1Image(ols_data, affine), ols_file)

    error_data = rs.uniform(0, 5, vol_shape)
    error_file = str(model_dir.join("error.nii.gz"))
    nib.save(nib.Nifti1Image(error_data, affine), error_file)

    timeseries.update(
        n_params=n_params,
        mask_file=mask_file,
        beta_file=beta_file,
        ols_file=ols_file,
        error_file=error_file,
    )
    return timeseries


@pytest.fixture()
def modelres(modelfit):

    seed = sum(map(ord, "modelfit"))
    rs = np.random.RandomState(seed)

    vol_shape = modelfit["vol_shape"]
    affine = modelfit["affine"]
    n_params = modelfit["n_params"]
    # TODO Fix this when constrast definition is done
    n_contrasts = n_params

    model_dir = modelfit["model_dir"]

    contrast_data = rs.normal(0, 5, vol_shape + (n_contrasts,))
    contrast_file = str(model_dir.join("contrast.nii.gz"))
    nib.save(nib.Nifti1Image(contrast_data, affine), contrast_file)

    variance_data = rs.uniform(0, 5, vol_shape + (n_contrasts,))
    variance_file = str(model_dir.join("variance.nii.gz"))
    nib.save(nib.Nifti1Image(variance_data, affine), variance_file)

    tstat_data = rs.normal(0, 2, vol_shape + (n_contrasts,))
    tstat_file = str(model_dir.join("tstat.nii.gz"))
    nib.save(nib.Nifti1Image(tstat_data, affine), tstat_file)

    modelfit.update(
        contrast_file=contrast_file,
        variance_file=variance_file,
        tstat_file=tstat_file,
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

    fname = execdir.join("test.mesh")
    nib.freesurfer.write_geometry(fname, verts, faces)

    meshdata = dict(
        verts=verts,
        faces=faces,
        neighbors=neighbors,
        fname=fname,
    )
    return meshdata
