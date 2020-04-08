import os.path as op
import json
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

from nipype.interfaces.base import traits, TraitedSpec, Bunch

import pytest

from .. import utils


class TestLymanInterface(object):

    def test_inheriting_interface_behavior(self):

        class TestInterface(utils.LymanInterface):

            class input_spec(TraitedSpec):
                a = traits.Int()
                b = traits.Int()

            class output_spec(TraitedSpec):
                c = traits.Int()

            def _run_interface(self, runtime):
                c = self.inputs.a + self.inputs.b
                self._results["c"] = c
                return runtime

        a, b = 2, 3
        c = a + b

        ifc = TestInterface(a=a, b=b)
        assert ifc._results == {}

        res = ifc.run()
        assert ifc._results == {"c": c}
        assert res.outputs.c == c

    def test_output_definition(self, execdir):

        ifc = utils.LymanInterface()
        field_name = "out_file"
        file_name = "out_file.txt"

        abspath_file_name = execdir.join(file_name)
        out = ifc.define_output(field_name, file_name)

        assert out == abspath_file_name
        assert ifc._results == {field_name: abspath_file_name}

    def test_write_image(self, execdir):

        field_name = "out_file"
        file_name = "out_file.nii"
        abspath_file_name = execdir.join(file_name)

        data = np.random.randn(12, 8, 4)
        affine = np.eye(4)
        img = nib.Nifti1Image(data, affine)

        # Test writing with an image
        ifc = utils.LymanInterface()
        img_out = ifc.write_image(field_name, file_name, img)

        assert ifc._results == {field_name: abspath_file_name}
        assert isinstance(img_out, nib.Nifti1Image)
        assert np.array_equal(img_out.get_fdata(), data)
        assert np.array_equal(img_out.affine, affine)

        # Test writing with data and affine
        ifc = utils.LymanInterface()
        img_out = ifc.write_image(field_name, file_name, data, affine)

        assert ifc._results == {field_name: abspath_file_name}
        assert isinstance(img_out, nib.Nifti1Image)
        assert np.array_equal(img_out.get_fdata(), data)
        assert np.array_equal(img_out.affine, affine)

    def test_write_visualization(self, execdir):

        class Visualization(object):
            self.closed = False

            def savefig(self, fname, close):
                with open(fname, "w"):
                    pass
                if close:
                    self.closed = True

        out_field = "test_file"
        out_path = "test.png"

        viz = Visualization()
        ifc = utils.LymanInterface()
        ifc.write_visualization(out_field, out_path, viz)

        assert op.exists(out_path)
        assert ifc._results == {out_field: op.join(execdir, out_path)}
        assert viz.closed

        out_field = "test_figure"
        out_path = "test_figure.png"

        f = plt.figure()
        ifc = utils.LymanInterface()
        ifc.write_visualization(out_field, out_path, f)

        assert op.exists(out_path)
        assert ifc._results == {out_field: op.join(execdir, out_path)}

        viz = None
        ifc.write_visualization(out_field, out_path, viz)

        with pytest.raises(RuntimeError):
            ifc.write_visualization(out_field, out_path, "bad type")

    def test_submit_cmdline(self, execdir):

        msg = "test"
        runtime = Bunch(returncode=None,
                        cwd=str(execdir),
                        environ={"msg": msg})

        ifc = utils.LymanInterface()
        cmdline_a = ["echo", "$msg"]

        runtime = ifc.submit_cmdline(runtime, cmdline_a)

        stdout = "\n{}\n".format(msg + "\n")
        assert runtime.stdout == stdout

        stderr = "\n\n"
        assert runtime.stderr == stderr

        cmdline = "\n{}\n".format(" ".join(cmdline_a))
        assert runtime.cmdline == cmdline
        assert runtime.returncode == 0

        with pytest.raises(RuntimeError):

            ifc = utils.LymanInterface()
            fname = "not_a_file"
            cmdline_b = ["cat", fname]

            runtime = ifc.submit_cmdline(runtime, cmdline_b)

        stdout = stdout + "\n\n"
        assert runtime.stdout == stdout

        stderr = stderr + ("\ncat: {}: No such file or directory\n\n"
                           .format(fname))
        assert runtime.stderr == stderr

        cmdline = cmdline + "\n{}\n".format(" ".join(cmdline_b))
        assert runtime.cmdline == cmdline

        assert runtime.returncode == 1


class TestSaveInfo(object):

    def test_save_info(self, execdir):

        info = {"proc_dir": "../proc",
                "data_dir": "../data"}

        res = utils.SaveInfo(info_dict=info).run()

        with open(res.outputs.info_file) as fid:
            info_roundtrip = json.load(fid)

        assert info == info_roundtrip


class TestImageMatrixConversion(object):

    @pytest.fixture
    def test_data(self):

        seed = sum(map(ord, "image_to_matrix"))
        rs = np.random.RandomState(seed)

        vol_shape = 12, 8, 4
        n_x, n_y, n_z = vol_shape
        n_tp = 20

        mask = np.arange(n_x * n_y * n_z).reshape(vol_shape) % 4
        n_vox = mask.astype(bool).sum()

        seg = rs.randint(0, 4, vol_shape)

        data_4d = rs.normal(0, 1, (n_x, n_y, n_z, n_tp))
        data_3d = data_4d[..., 0]
        data_2d = data_4d[mask.astype(bool)].T
        data_1d = data_3d[mask.astype(bool)].T

        data_4d_masked = data_4d * mask.astype(bool)[:, :, :, None]
        data_3d_masked = data_4d_masked[..., 0]

        affine = np.array([[-2, 0, 0, 100],
                           [0, 1.9, -.6, 120],
                           [0, -.6, 1.9, -40],
                           [0, 0, 0, 1]])

        img_4d = nib.Nifti1Image(data_4d, affine)

        tr = 1.5
        z_x, z_y, z_z, _ = img_4d.header.get_zooms()
        img_4d.header.set_zooms((z_x, z_y, z_z, tr))

        img_3d = nib.Nifti1Image(data_3d, affine)
        mask_img = nib.Nifti1Image(mask, affine)
        seg_img = nib.Nifti1Image(seg, affine)

        test_data = dict(
            mask=mask,
            mask_img=mask_img,
            seg=seg,
            seg_img=seg_img,
            affine=affine,
            img_4d=img_4d,
            img_3d=img_3d,
            data_4d=data_4d,
            data_4d_masked=data_4d_masked,
            data_3d=data_3d,
            data_3d_masked=data_3d_masked,
            data_2d=data_2d,
            data_1d=data_1d,
            vol_shape=vol_shape,
            n_vox=n_vox,
            n_tp=n_tp,
            tr=tr,
        )

        return test_data

    def test_image_to_matrix(self, test_data):

        img_4d = test_data["img_4d"]
        img_3d = test_data["img_3d"]
        mask_img = test_data["mask_img"]
        seg_img = test_data["seg_img"]

        # Test 4D image > 2D matrix with a mask
        data_2d = utils.image_to_matrix(img_4d, mask_img)
        assert np.array_equal(data_2d, test_data["data_2d"])
        assert data_2d.shape == (test_data["n_tp"], test_data["n_vox"])

        # Test 3D image > 1D matrix
        data_1d = utils.image_to_matrix(img_3d, mask_img)
        assert np.array_equal(data_1d, test_data["data_1d"])
        assert data_1d.shape == (test_data["n_vox"],)

        # Test segmentation value(s)
        for use in [1, [1, 2]]:
            data_seg = utils.image_to_matrix(img_4d, seg_img, use=use)
            within_seg = np.isin(test_data["seg"], use).sum()
            assert data_seg.shape == (test_data["n_tp"], within_seg)

    def test_matrix_to_image(self, test_data):

        data_2d = test_data["data_2d"]
        data_1d = test_data["data_1d"]
        mask_img = test_data["mask_img"]
        seg_img = test_data["seg_img"]
        n_x, n_y, n_z = test_data["vol_shape"]
        n_tp = test_data["n_tp"]

        # Test 2D matrix > 4D image
        img_4d = utils.matrix_to_image(data_2d, mask_img)
        assert np.array_equal(img_4d.get_fdata(), test_data["data_4d_masked"])
        assert np.array_equal(img_4d.affine, mask_img.affine)
        assert img_4d.shape == (n_x, n_y, n_z, n_tp)

        # Test 1d matrix > 3D image
        img_3d = utils.matrix_to_image(data_1d, mask_img)
        assert np.array_equal(img_3d.get_fdata(), test_data["data_3d_masked"])
        assert np.array_equal(img_3d.affine, mask_img.affine)
        assert img_3d.shape == (n_x, n_y, n_z)

        # Test segmentation value(s)
        for use in [1, [1, 2]]:
            within_seg = np.isin(test_data["seg"], use)
            seg_mask_img = nib.Nifti1Image(within_seg.astype(np.uint8),
                                           test_data["affine"])
            data_seg = test_data["data_3d"][within_seg]
            img_mask = utils.matrix_to_image(data_seg, seg_mask_img)
            img_seg = utils.matrix_to_image(data_seg, seg_img, use=use)
            assert np.array_equal(img_mask.get_fdata(), img_seg.get_fdata())

        # Test affine and header from template image are used
        img_template = test_data["img_4d"]
        mask_img_nohdr = nib.Nifti1Image(test_data["mask"], np.eye(4))
        img_4d = utils.matrix_to_image(data_2d, mask_img_nohdr, img_template)
        assert np.array_equal(img_4d.affine, img_template.affine)
        assert img_template.header == img_template.header

    def test_check_mask(self):

        data_shape = 12, 8, 4
        x = np.random.normal(0, 1, data_shape)

        mask_shape = data_shape[:-1]
        mask_correct_3d = np.random.uniform(0, 1, data_shape) > .5
        mask_correct_2d = np.random.uniform(0, 1, mask_shape) > .5
        mask_bad_shape = np.random.uniform(0, 1, mask_shape).T > .5
        mask_bad_dtype = np.random.uniform(0, 1, mask_shape)

        utils.check_mask(mask_correct_3d, x)
        utils.check_mask(mask_correct_2d, x)

        with pytest.raises(ValueError):
            utils.check_mask(mask_bad_shape, x)

        with pytest.raises(TypeError):
            utils.check_mask(mask_bad_dtype, x)
