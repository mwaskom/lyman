import numpy as np
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
        phase_encoding="pa",
        fm_template="{session}_fieldmap_{encoding}.nii.gz",
        ts_template="{session}_{experiment}_{run}.nii.gz",
        sb_template="{session}_{experiment}_{run}_sbref.nii.gz",
    )

    exp_info = Bunch(name="exp_alpha")

    model_info = Bunch(name="model_info")

    subjects = ["subj01", "subj02"]
    sessions = None

    return dict(
        proj_info=proj_info,
        subjects=subjects,
        sessions=sessions,
        exp_info=exp_info,
        model_info=model_info,
    )


@pytest.fixture()
def template(tmpdir):

    subject = "subj01"
    analysis_dir = tmpdir.mkdir("analysis")
    template_dir = analysis_dir.mkdir(subject).mkdir("template")

    random_seed = sum(map(ord, "template"))
    rs = np.random.RandomState(random_seed)

    shape = 12, 8, 4
    affine = np.array([[-2, 0, 0, 10],
                       [0, -2, -1, 10],
                       [0, 1, 2, 5],
                       [0, 0, 0, 1]])

    reg_file = str(template_dir.join("anat2func.mat"))
    np.savetxt(reg_file, np.random.randn(4, 4))

    seg_data = rs.randint(0, 7, shape)
    seg_file = str(template_dir.join("seg.nii.gz"))
    nib.save(nib.Nifti1Image(seg_data, affine), seg_file)

    anat_data = rs.randint(0, 100, shape)
    anat_file = str(template_dir.join("anat.nii.gz"))
    nib.save(nib.Nifti1Image(anat_data, affine), anat_file)

    mask_data = (seg_data > 0).astype(np.uint8)
    mask_file = str(template_dir.join("mask.nii.gz"))
    nib.save(nib.Nifti1Image(mask_data, affine), mask_file)

    surf_ids = np.arange(1, (seg_data == 1).sum() + 1)
    surf_data = np.zeros(shape + (2,), np.int)
    surf_data[seg_data == 1, 0] = surf_ids
    surf_data[seg_data == 1, 1] = surf_ids
    surf_file = str(template_dir.join("surf.nii.gz"))
    nib.save(nib.Nifti1Image(surf_data, affine), surf_file)

    return dict(
        analysis_dir=analysis_dir,
        reg_file=reg_file,
        seg_file=seg_file,
        anat_file=anat_file,
        mask_file=mask_file,
        surf_file=surf_file,
    )
