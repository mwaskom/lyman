import os.path as op
import shutil
from tempfile import mkdtemp
import numpy as np
from nipype.testing import assert_equal, assert_raises
from nipype.interfaces.base import Bunch

from .. import model


def test_model_info():

    test_dir = mkdtemp()

    subject_id = "subj"
    exp_info = dict(
        events=["ev1", "ev2"],
        regressors=["reg1", "reg2"],
        parfile_base_dir=test_dir,
        regressor_base_dir=test_dir,
        parfile_template="%(subject_id)s_%(event)s_%(run)s.txt",
        regressor_template="%(subject_id)s_%(regressor)s_%(run)s.txt")
    functional_runs = ["run1.nii", "run2.nii"]

    args = [subject_id, functional_runs, exp_info]

    ev1r1 = np.array([[5., 0., 1.], [10., 0., 1.]])
    ev2r1 = np.array([2.5, 2.5, 0.5])
    np.savetxt(op.join(test_dir, "subj_ev1_1.txt"), ev1r1)
    np.savetxt(op.join(test_dir, "subj_ev2_1.txt"), ev2r1)

    assert_raises(IOError, model.build_model_info, *args)

    ev1r2 = np.array([[6., 0., 1.], [11., 0., 1.]])
    ev2r2 = np.array([[3.5, 2.5, 0.5], [8.5, 2.5, 1.5]])
    np.savetxt(op.join(test_dir, "subj_ev1_2.txt"), ev1r2)
    np.savetxt(op.join(test_dir, "subj_ev2_2.txt"), ev2r2)

    reg1r1 = np.arange(5, dtype=float)
    reg2r1 = np.array([5.])
    reg1r2 = np.arange(1, 6, dtype=float)
    reg2r2 = np.arange(1, 4, .5, dtype=float)
    np.savetxt(op.join(test_dir, "subj_reg1_1.txt"), reg1r1)
    np.savetxt(op.join(test_dir, "subj_reg2_1.txt"), reg2r1)
    np.savetxt(op.join(test_dir, "subj_reg1_2.txt"), reg1r2)
    np.savetxt(op.join(test_dir, "subj_reg2_2.txt"), reg2r2)

    model_info = [Bunch(conditions=exp_info["events"],
                        regressor_names=exp_info["regressors"],
                        regressors=[reg1r1, reg2r1],
                        onsets=[ev1r1[:, 0], np.array([ev2r1[0]])],
                        durations=[ev1r1[:, 1], np.array([ev2r1[1]])],
                        amplitudes=[ev1r1[:, 2], np.array([ev2r1[2]])]),
                  Bunch(conditions=exp_info["events"],
                        regressor_names=exp_info["regressors"],
                        regressors=[reg1r2, reg2r2],
                        onsets=[ev1r2[:, 0], ev2r2[:, 0]],
                        durations=[ev1r2[:, 1], ev2r2[:, 1]],
                        amplitudes=[ev1r2[:, 2], ev2r2[:, 2]])]

    got = model.build_model_info(*args)

    for i, wanted in enumerate(model_info):
        yield assert_equal, wanted.__dict__, got[i].__dict__

    shutil.rmtree(test_dir)
