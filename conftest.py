import numpy as np
import pandas as pd
import nibabel as nib

import pytest

from moss import Bunch  # TODO change to lyman version when implemented


@pytest.fixture()
def execdir(tmpdir):

    origdir = tmpdir.chdir()
    yield tmpdir
    origdir.chdir()


@pytest.fixture()
def lyman_info(tmpdir):

    data_dir = tmpdir.mkdir("data")
    analysis_dir = tmpdir.mkdir("analysis")
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

    proj_info = Bunch(
        data_dir=str(data_dir),
        analysis_dir=str(analysis_dir),
        cache_dir=str(cache_dir),
        scan_info=scan_info,
        phase_encoding="ap",
        fm_template="{session}_{encoding}.nii.gz",
        ts_template="{session}_{experiment}_{run}.nii.gz",
        sb_template="{session}_{experiment}_{run}_sbref.nii.gz",
    )

    exp_info = Bunch(
        name="exp_alpha",
        tr=1.5,
    )

    model_info = Bunch(
        name="model_a",
        smooth_fwhm=2,
        hpf_cutoff=10,
        save_residuals=True,

        # TODO FIX
        contrasts=["a", "b", "a-b"]
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
        subject_dir.mkdir("func")
        design_dir = subject_dir.mkdir("design")
        design.to_csv(design_dir.join("model_a.csv"))

    vol_shape = 12, 8, 4
    n_tp = 20
    n_params = len(design["condition"].unique())

    return dict(
        proj_info=proj_info,
        subjects=subjects,
        sessions=sessions,
        exp_info=exp_info,
        model_info=model_info,
        analysis_dir=analysis_dir,
        data_dir=data_dir,

        vol_shape=vol_shape,
        n_tp=n_tp,
        n_params=n_params,
    )


@pytest.fixture()
def template(lyman_info):

    subject = "subj01"
    template_dir = (lyman_info["analysis_dir"]
                    .mkdir(subject)
                    .mkdir("template"))

    random_seed = sum(map(ord, "template"))
    rs = np.random.RandomState(random_seed)

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

    surf_ids = np.arange(1, (seg_data == 1).sum() + 1)
    surf_data = np.zeros(vol_shape + (2,), np.int)
    surf_data[seg_data == 1, 0] = surf_ids
    surf_data[seg_data == 1, 1] = surf_ids
    surf_file = str(template_dir.join("surf.nii.gz"))
    nib.save(nib.Nifti1Image(surf_data, affine), surf_file)

    lyman_info.update(
        vol_shape=vol_shape,
        subject=subject,
        reg_file=reg_file,
        seg_file=seg_file,
        anat_file=anat_file,
        mask_file=mask_file,
        surf_file=surf_file,
    )
    return lyman_info


@pytest.fixture()
def timeseries(template):

    random_seed = sum(map(ord, "timeseries"))
    rs = np.random.RandomState(random_seed)

    session = "sess01"
    run = "run01"

    exp_name = template["exp_info"].name
    model_name = template["model_info"].name

    vol_shape = template["vol_shape"]
    n_tp = template["n_tp"]
    affine = np.eye(4)
    affine[:3, :3] *= 2

    timeseries_dir = (template["analysis_dir"]
                      .join(template["subject"])
                      .mkdir(exp_name)
                      .mkdir("timeseries")
                      .mkdir("{}_{}".format(session, run)))

    model_dir = (template["analysis_dir"]
                 .join(template["subject"])
                 .join(exp_name)
                 .mkdir(model_name)
                 .mkdir("{}_{}".format(session, run)))

    mask_data = nib.load(template["seg_file"]).get_data() > 0
    mask_data &= rs.uniform(0, 1, vol_shape) > .05
    mask_file = str(timeseries_dir.join("mask.nii.gz"))
    nib.save(nib.Nifti1Image(mask_data.astype(np.int), affine), mask_file)

    ts_shape = vol_shape + (n_tp,)
    ts_data = rs.normal(100, 5, ts_shape) * mask_data[..., np.newaxis]
    ts_file = str(timeseries_dir.join("func.nii.gz"))
    nib.save(nib.Nifti1Image(ts_data, affine), ts_file)

    noise_data = rs.choice([0, 1], vol_shape, p=[.95, .05])
    noise_file = str(timeseries_dir.join("noise.nii.gz"))
    nib.save(nib.Nifti1Image(noise_data, affine), noise_file)

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
        ts_file=ts_file,
        noise_file=noise_file,
        mc_file=mc_file,
        timeseries_dir=timeseries_dir,
        model_dir=model_dir,
    )
    return template


@pytest.fixture()
def modelfit(timeseries):

    random_seed = sum(map(ord, "modelfit"))
    rs = np.random.RandomState(random_seed)

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

    random_seed = sum(map(ord, "modelfit"))
    rs = np.random.RandomState(random_seed)

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
