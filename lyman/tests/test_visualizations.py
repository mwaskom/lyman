import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

import pytest

from .. import visualizations as viz


class VisualizationFixture(object):

    @pytest.fixture
    def closefig(self):

        yield
        plt.close("all")


class TestMosaic(VisualizationFixture):

    @pytest.fixture
    def images(self, execdir):

        seed = sum(map(ord, "images"))
        rs = np.random.RandomState(seed)
        shape = 30, 30, 8
        aff = np.eye(4)

        anat_data = rs.randint(0, 100, shape)
        stat_data = rs.normal(0, 2, shape)
        mask_data = rs.uniform(0, 1, shape) > .1

        anat_img = nib.Nifti1Image(anat_data, aff)
        stat_img = nib.Nifti1Image(stat_data, aff)
        mask_img = nib.Nifti1Image(mask_data.astype(np.int), aff)

        anat_fname = "anat.nii"
        stat_fname = "stat.nii"
        mask_fname = "mask.nii"

        anat_img.to_filename(anat_fname)
        stat_img.to_filename(stat_fname)
        mask_img.to_filename(mask_fname)

        images = dict(

            anat_data=anat_data,
            anat_fname=anat_fname,
            anat_img=anat_img,

            stat_data=stat_data,
            stat_fname=stat_fname,
            stat_img=stat_img,

            mask_data=mask_data,
            mask_fname=mask_fname,
            mask_img=mask_img,

        )

        return images

    @pytest.mark.parametrize("src", ["fname", "img", "data"])
    def test_mosaic_inputs(self, images, closefig, src):

        viz.Mosaic(images["anat_" + src],
                   images["stat_" + src],
                   images["mask_" + src])

        viz.Mosaic(images["anat_" + src],
                   images["mask_" + src])

    def test_mosaic_bad_inputs(self, images, closefig):

        with pytest.raises(TypeError):
            viz.Mosaic(2)

        with pytest.raises(TypeError):
            viz.Mosaic(images["anat_img"], stat=2)

        with pytest.raises(TypeError):
            viz.Mosaic(images["anat_img"], mask=2)

    @pytest.mark.parametrize("slice_dir",
                             ["s", "sag", "c", "cor", "a", "axial"])
    def test_slice_dir(self, images, closefig, slice_dir):

        viz.Mosaic(images["anat_img"], slice_dir=slice_dir)

    def test_bad_slice_dir(self, images, closefig):

        with pytest.raises(ValueError):
            viz.Mosaic(images["anat_img"], slice_dir="horiz")

    @pytest.mark.parametrize("tight", [True, False])
    def test_tight(self, images, closefig, tight):

        viz.Mosaic(images["anat_img"], tight=tight)
        viz.Mosaic(images["anat_img"], mask=images["mask_img"], tight=tight)

    def test_anat_lims(self, images, closefig):

        viz.Mosaic(images["anat_img"], anat_lims=(0, 10))

    def test_plot_activation(self, images, closefig):

        m = viz.Mosaic(images["anat_img"], images["stat_img"])
        m.plot_activation()

        m = viz.Mosaic(images["anat_img"], images["stat_img"])
        m.plot_activation(thresh=10)

        m = viz.Mosaic(images["anat_img"], images["stat_img"])
        m.plot_activation(neg_cmap="Blues")

    def test_plot_overlay(self, images, closefig):

        m = viz.Mosaic(images["anat_img"], images["stat_img"])
        m.plot_overlay("cube:0:0")

        m = viz.Mosaic(images["anat_img"], images["stat_img"])
        m.plot_overlay("cube:2:-1_r", center=True)

        m = viz.Mosaic(images["anat_img"], images["stat_img"])
        m.plot_overlay("Reds", vmin=1, vmax=5, thresh=0)

        m = viz.Mosaic(images["anat_img"],
                       images["stat_img"],
                       images["mask_img"])
        m.plot_overlay("Reds_r")

        m = viz.Mosaic(images["anat_img"],
                       images["stat_data"].astype(np.int32),
                       images["mask_img"])
        m.plot_overlay("Reds_r")

        m = viz.Mosaic(images["anat_img"], images["stat_img"])
        with pytest.raises(ValueError):
            m.plot_overlay("not_a_colormap")

    def test_plot_mask(self, images, closefig):

        m = viz.Mosaic(images["anat_img"], images["mask_img"])
        m.plot_mask()

    def test_plot_mask_edges(self, images, closefig):

        m = viz.Mosaic(images["anat_img"], images["mask_img"])
        m.plot_mask_edges("red")

    def test_title(self, images, closefig):

        title = "test title"
        m = viz.Mosaic(images["anat_img"], title=title)
        title_text = m.fig.texts[0]
        assert title == title_text.get_text()

    def test_savefig(self, images, closefig):

        m = viz.Mosaic(images["anat_img"])
        m.savefig("mosaic.png", close=True)


class TestCarpetPlot(VisualizationFixture):

    @pytest.fixture
    def images(self, execdir):

        seed = sum(map(ord, "images"))
        rs = np.random.RandomState(seed)
        shape = 30, 30, 8, 10
        aff = np.eye(4)

        func_data = rs.randint(0, 100, shape)
        seg_data = rs.randint(0, 9, shape[:-1])

        func_img = nib.Nifti1Image(func_data, aff)
        seg_img = nib.Nifti1Image(seg_data, aff)

        func_fname = "func.nii"
        seg_fname = "seg.nii"

        func_img.to_filename(func_fname)
        seg_img.to_filename(seg_fname)

        mc_data = pd.DataFrame(
            rs.normal(0, 1, (shape[-1], 6)),
            columns=["rot_x", "rot_y", "rot_z",
                     "trans_x", "trans_y", "trans_z"]
        )

        mc_fname = "mc.csv"
        mc_data.to_csv(mc_fname, index=False)

        images = dict(

            func_img=func_img,
            func_fname=func_fname,

            seg_img=seg_img,
            seg_fname=seg_fname,

            mc_data=mc_data,
            mc_fname=mc_fname,

        )

        return images

    @pytest.mark.parametrize("src", ["fname", "img"])
    def test_carpetplot_inputs(self, images, closefig, src):
        viz.CarpetPlot(images["func_" + src], images["seg_" + src])

    @pytest.mark.parametrize("src", ["fname", "data"])
    def test_carpetplot_fd(self, images, closefig, src):
        viz.CarpetPlot(images["func_img"], images["seg_img"],
                       images["mc_" + src])

    @pytest.mark.parametrize("fwhm", [None, 0, 5])
    def test_carpetplot_smoothing(self, images, closefig, fwhm):
        viz.CarpetPlot(images["func_img"], images["seg_img"], smooth_fwhm=fwhm)

    @pytest.mark.parametrize("title", [None, "Run 1"])
    def test_carpetplot_title(self, images, closefig, title):
        viz.CarpetPlot(images["func_img"], images["seg_img"], title=title)

    def test_carpetplot_savefig(self, images, closefig):
        p = viz.CarpetPlot(images["func_img"], images["seg_img"])
        p.savefig("carpetplot.png", close=True)


class TestDesignMatrixPlots(VisualizationFixture):

    def test_plot_design_matrix(self, closefig):

        X = pd.DataFrame(np.random.randn(20, 3), columns=list("abc"))
        viz.plot_design_matrix(X)

    def test_plot_nuisance_variables(self, closefig):

        X = pd.DataFrame(np.random.randn(20, 5),
                         columns=["wm1", "wm2", "edge1", "edge2", "edge3"])
        viz.plot_nuisance_variables(X)
