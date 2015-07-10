import numpy as np
import nibabel as nib
import nose.tools as nt
import numpy.testing as npt
from nipype.interfaces import fsl

from .. import mixedfx


class TestGroupMerge(object):

    rs = np.random.RandomState(77)

    def test_group_mask(self):

        # Load the common-space mask
        mni_mask = fsl.Info.standard_image("MNI152_T1_2mm_brain_mask.nii.gz")
        mni_img = nib.load(mni_mask)
        mask_data = mni_img.get_data().astype(bool)

        # Create some "variance" data with sparsity
        i, j, k = mask_data.shape
        var_data = self.rs.gamma(3, size=(i, j, k, 10))
        zero_voxels = self.rs.uniform(size=(i, j, k, 10)) < .5
        var_data[zero_voxels] = 0
        var_img = nib.Nifti1Image(var_data, mni_img.get_affine())

        merge = mixedfx.MergeAcrossSubjects()
        out_mask = merge._create_group_mask(var_img)
        out_data = out_mask.get_data().astype(bool)
        nt.assert_is_instance(out_mask, nib.Nifti1Image)

        # Test that the final mask is a subset of both the MNI space mask
        # and of a mask derived from all voxels with nonzero variance
        nt.assert_equal(np.sum(~mask_data & out_data), 0)
        nt.assert_equal(np.sum(~(var_data > 0).all(axis=-1) & out_data), 0)

    def test_group_merge(self):

        img_size = (91, 109, 91)
        in_data = [self.rs.normal(size=img_size) for _ in range(10)]
        in_imgs = [nib.Nifti1Image(d, np.eye(4)) for d in in_data]

        merge = mixedfx.MergeAcrossSubjects()
        out_img = merge._merge_subject_images(in_imgs)
        nt.assert_is_instance(out_img, nib.Nifti1Image)
        nt.assert_equal(out_img.shape, (91, 109, 91, 10))
        out_data = out_img.get_data()
        in_data_reordered = np.asarray(in_data).transpose(1, 2, 3, 0)
        npt.assert_array_equal(in_data_reordered, out_data)

        merge = mixedfx.MergeAcrossSubjects()
        good_indices = range(8)
        out_img = merge._merge_subject_images(in_imgs, good_indices)
        nt.assert_is_instance(out_img, nib.Nifti1Image)
        nt.assert_equal(out_img.shape, (91, 109, 91, 8))
