import numpy as np
import nibabel as nib

import pytest

from .. import utils


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

        test_data = dict(
            mask=mask,
            mask_img=mask_img,
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

        # Test 4D image > 2D matrix with a mask
        data_2d = utils.image_to_matrix(img_4d, mask_img)
        assert np.array_equal(data_2d, test_data["data_2d"])
        assert data_2d.shape == (test_data["n_tp"], test_data["n_vox"])

        # Test 3D image > 1D matrix
        data_1d = utils.image_to_matrix(img_3d, mask_img)
        assert np.array_equal(data_1d, test_data["data_1d"])
        assert data_1d.shape == (test_data["n_vox"],)

    def test_matrix_to_image(self, test_data):

        data_2d = test_data["data_2d"]
        data_1d = test_data["data_1d"]
        mask_img = test_data["mask_img"]
        n_x, n_y, n_z = test_data["vol_shape"]
        n_tp = test_data["n_tp"]

        # Test 2D matrix > 4D image
        img_4d = utils.matrix_to_image(data_2d, mask_img)
        assert np.array_equal(img_4d.get_data(), test_data["data_4d_masked"])
        assert np.array_equal(img_4d.affine, mask_img.affine)
        assert img_4d.shape == (n_x, n_y, n_z, n_tp)

        # Test 1d matrix > 3D image
        img_3d = utils.matrix_to_image(data_1d, mask_img)
        assert np.array_equal(img_3d.get_data(), test_data["data_3d_masked"])
        assert np.array_equal(img_3d.affine, mask_img.affine)
        assert img_3d.shape == (n_x, n_y, n_z)

        # Test affine and header from template image are used
        img_template = test_data["img_4d"]
        mask_img_nohdr = nib.Nifti1Image(test_data["mask"], np.eye(4))
        img_4d = utils.matrix_to_image(data_2d, mask_img_nohdr, img_template)
        assert np.array_equal(img_4d.affine, img_template.affine)
        assert img_template.header == img_template.header
