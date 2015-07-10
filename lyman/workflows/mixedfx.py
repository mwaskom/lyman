import os.path as op

import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import binary_dilation
from skimage import morphology
import matplotlib as mpl
import matplotlib.pyplot as plt

from nipype import IdentityInterface, Node, Workflow
from nipype.interfaces.base import (BaseInterface,
                                    BaseInterfaceInputSpec,
                                    InputMultiPath, OutputMultiPath,
                                    TraitedSpec, File, traits)
from nipype.interfaces import fsl, freesurfer

import seaborn as sns
from moss import locator
from moss.mosaic import Mosaic

import lyman
from lyman.tools import SaveParameters, add_suffix, nii_to_png


def create_volume_mixedfx_workflow(name="volume_group",
                                   subject_list=None,
                                   regressors=None,
                                   contrasts=None,
                                   exp_info=None):

    # Handle default arguments
    if subject_list is None:
        subject_list = []
    if regressors is None:
        regressors = dict(group_mean=[])
    if contrasts is None:
        contrasts = [["group_mean", "T", ["group_mean"], [1]]]
    if exp_info is None:
        exp_info = lyman.default_experiment_parameters()

    # Define workflow inputs
    inputnode = Node(IdentityInterface(["l1_contrast",
                                        "copes",
                                        "varcopes",
                                        "dofs"]),
                     "inputnode")

    # Merge the fixed effect summary images into one 4D image
    merge = Node(MergeAcrossSubjects(regressors=regressors), "merge")

    # Make a simple design
    design = Node(fsl.MultipleRegressDesign(contrasts=contrasts),
                  "design")

    # Fit the mixed effects model
    flameo = Node(fsl.FLAMEO(run_mode=exp_info["flame_mode"]), "flameo")

    # Estimate the smoothness of the data
    smoothest = Node(fsl.SmoothEstimate(), "smoothest")

    # Correct for multiple comparisons
    cluster = Node(fsl.Cluster(threshold=exp_info["cluster_zthresh"],
                               pthreshold=exp_info["grf_pthresh"],
                               out_threshold_file=True,
                               out_index_file=True,
                               out_localmax_txt_file=True,
                               peak_distance=exp_info["peak_distance"],
                               use_mm=True),
                   "cluster")

    # Project the mask and thresholded zstat onto the surface
    surfproj = create_surface_projection_workflow(exp_info=exp_info)

    # Segment the z stat image with a watershed algorithm
    watershed = Node(Watershed(), "watershed")

    # Make static report images in the volume
    report = Node(MFXReport(), "report")
    report.inputs.subjects = subject_list

    # Save the experiment info
    saveparams = Node(SaveParameters(exp_info=exp_info), "saveparams")

    # Define the workflow outputs
    outputnode = Node(IdentityInterface(["copes",
                                         "varcopes",
                                         "mask_file",
                                         "flameo_stats",
                                         "thresh_zstat",
                                         "surf_zstat",
                                         "surf_mask",
                                         "cluster_image",
                                         "seg_file",
                                         "peak_file",
                                         "lut_file",
                                         "report",
                                         "json_file"]),
                      "outputnode")

    # Define and connect up the workflow
    group = Workflow(name)
    group.connect([
        (inputnode, merge,
            [("copes", "cope_files"),
             ("varcopes", "varcope_files"),
             ("dofs", "dof_files")]),
        (inputnode, saveparams,
            [("copes", "in_file")]),
        (merge, flameo,
            [("cope_file", "cope_file"),
             ("varcope_file", "var_cope_file"),
             ("dof_file", "dof_var_cope_file"),
             ("mask_file", "mask_file")]),
        (merge, design,
            [("regressors", "regressors")]),
        (design, flameo,
            [("design_con", "t_con_file"),
             ("design_grp", "cov_split_file"),
             ("design_mat", "design_file")]),
        (flameo, smoothest,
            [("zstats", "zstat_file")]),
        (merge, smoothest,
            [("mask_file", "mask_file")]),
        (smoothest, cluster,
            [("dlh", "dlh"),
             ("volume", "volume")]),
        (flameo, cluster,
            [("zstats", "in_file")]),
        (cluster, watershed,
            [("threshold_file", "zstat_file"),
             ("localmax_txt_file", "localmax_file")]),
        (merge, report,
            [("mask_file", "mask_file"),
             ("cope_file", "cope_file")]),
        (flameo, report,
            [("zstats", "zstat_file")]),
        (cluster, report,
            [("threshold_file", "zstat_thresh_file"),
             ("localmax_txt_file", "localmax_file")]),
        (watershed, report,
            [("seg_file", "seg_file")]),
        (merge, surfproj,
            [("mask_file", "inputs.mask_file")]),
        (cluster, surfproj,
            [("threshold_file", "inputs.zstat_file")]),
        (merge, outputnode,
            [("cope_file", "copes"),
             ("varcope_file", "varcopes"),
             ("mask_file", "mask_file")]),
        (flameo, outputnode,
            [("stats_dir", "flameo_stats")]),
        (cluster, outputnode,
            [("threshold_file", "thresh_zstat"),
             ("index_file", "cluster_image")]),
        (watershed, outputnode,
            [("seg_file", "seg_file"),
             ("peak_file", "peak_file"),
             ("lut_file", "lut_file")]),
        (surfproj, outputnode,
            [("outputs.surf_zstat", "surf_zstat"),
             ("outputs.surf_mask", "surf_mask")]),
        (report, outputnode,
            [("out_files", "report")]),
        (saveparams, outputnode,
            [("json_file", "json_file")]),
        ])

    return group, inputnode, outputnode


def create_surface_projection_workflow(name="surfproj", exp_info=None):
    """Project the group mask and thresholded zstat file onto the surface."""
    if exp_info is None:
        exp_info = lyman.default_experiment_parameters()

    inputnode = Node(IdentityInterface(["zstat_file", "mask_file"]), "inputs")

    # Sample the zstat image to the surface
    hemisource = Node(IdentityInterface(["mni_hemi"]), "hemisource")
    hemisource.iterables = ("mni_hemi", ["lh", "rh"])

    zstatproj = Node(freesurfer.SampleToSurface(
        sampling_method=exp_info["sampling_method"],
        sampling_range=exp_info["sampling_range"],
        sampling_units=exp_info["sampling_units"],
        smooth_surf=exp_info["surf_smooth"],
        subject_id="fsaverage",
        mni152reg=True,
        target_subject="fsaverage"),
        "zstatproj")

    # Sample the mask to the surface
    maskproj = Node(freesurfer.SampleToSurface(
        sampling_range=exp_info["sampling_range"],
        sampling_units=exp_info["sampling_units"],
        subject_id="fsaverage",
        mni152reg=True,
        target_subject="fsaverage"),
        "maskproj")
    if exp_info["sampling_method"] == "point":
        maskproj.inputs.sampling_method = "point"
    else:
        maskproj.inputs.sampling_method = "max"

    outputnode = Node(IdentityInterface(["surf_zstat",
                                         "surf_mask"]), "outputs")

    # Define and connect the workflow
    proj = Workflow(name)
    proj.connect([
        (inputnode, zstatproj,
            [("zstat_file", "source_file")]),
        (inputnode, maskproj,
            [("mask_file", "source_file")]),
        (hemisource, zstatproj,
            [("mni_hemi", "hemi")]),
        (hemisource, maskproj,
            [("mni_hemi", "hemi")]),
        (zstatproj, outputnode,
            [("out_file", "surf_zstat")]),
        (maskproj, outputnode,
            [("out_file", "surf_mask")]),
        ])

    return proj


# =========================================================================== #


class MergeInput(BaseInterfaceInputSpec):

    cope_files = InputMultiPath(File(exists=True))
    varcope_files = InputMultiPath(File(exists=True))
    dof_files = InputMultiPath(File(exists=True))
    regressors = traits.Dict()


class MergeOutput(TraitedSpec):

    cope_file = File(exists=True)
    varcope_file = File(exists=True)
    dof_file = File(exists=True)
    mask_file = File(exists=True)
    regressors = traits.Dict()


class MergeAcrossSubjects(BaseInterface):

    input_spec = MergeInput
    output_spec = MergeOutput

    def _run_interface(self, runtime):

        # Find indices for subjects with non-empty varcopes
        good_indices = self._find_good_images(self.inputs.varcope_files)

        for ftype in ["cope", "varcope", "dof"]:

            in_files = getattr(self.inputs, ftype + "_files")
            in_imgs = [nib.load(f) for f in in_files]
            out_img = self._merge_subject_images(in_imgs, good_indices)
            out_img.to_filename(ftype + "_merged.nii.gz")

            # Use the varcope image to make a group mask
            if ftype == "varcope":
                mask_img = self._create_group_mask(out_img)
                mask_img.to_filename("group_mask.nii.gz")

        # Filter the regressor columns
        filtered_regressors = {}
        for name, col in self.inputs.regressors.items():
            filtered_col = [r for i, r in enumerate(col)
                            if i in good_indices]
            filtered_regressors[name] = filtered_col
        self.filtered_regressors = filtered_regressors

        return runtime

    def _find_good_images(self, filenames):
        """Find indices of non-empty images."""
        return [i for i, f in enumerate(filenames)
                if not np.allclose(nib.load(f).get_data(), 0)]

    def _merge_subject_images(self, images, good_indices=None):
        """Stack a list of 3D images into 4D image."""
        if good_indices is None:
            good_indices = range(len(images))
        images = [img for i, img in enumerate(images) if i in good_indices]
        out_img = nib.concat_images(images)
        return out_img

    def _create_group_mask(self, var_img):

        mni_mask = fsl.Info.standard_image("MNI152_T1_2mm_brain_mask.nii.gz")
        mni_img = nib.load(mni_mask)
        mask_data = mni_img.get_data().astype(bool)

        # Find the voxels with positive variance
        var_data = var_img.get_data()
        good_var = (var_data > 0).all(axis=-1)

        # Find the intersection
        mask_data &= good_var

        # Create and return the 3D mask image
        mask_img = nib.Nifti1Image(mask_data,
                                   mni_img.get_affine(),
                                   mni_img.get_header())
        mask_img.set_data_dtype(np.int16)
        return mask_img

    def _list_outputs(self):

        outputs = self._outputs().get()
        for ftype in ["cope", "varcope", "dof"]:
            outputs[ftype + "_file"] = op.realpath(ftype + "_merged.nii.gz")
        outputs["mask_file"] = op.realpath("group_mask.nii.gz")
        outputs["regressors"] = self.filtered_regressors
        return outputs


class WatershedInput(BaseInterfaceInputSpec):

    zstat_file = File(exists=True)
    localmax_file = File(exsits=True)


class WatershedOutput(TraitedSpec):

    seg_file = File(exists=True)
    peak_file = File(exists=True)
    lut_file = File(exists=True)


class Watershed(BaseInterface):

    input_spec = WatershedInput
    output_spec = WatershedOutput

    def _run_interface(self, runtime):

        # Load the zstat data
        z_img = nib.load(self.inputs.zstat_file)
        z_data = z_img.get_data()

        # Load the peaks
        peaks_df = pd.read_table(self.inputs.localmax_file, "\t")
        peaks = peaks_df[["x", "y", "z"]].values

        # Possibly do the segmentation, if there are peaks
        if len(peaks):
            seg, markers = self.segment(z_data, peaks)
        else:
            seg, markers = np.zeros_like(z_data), np.zeros_like(z_data)

        # Set up the output names
        seg_fname = op.basename(add_suffix(self.inputs.zstat_file, "seg"))
        peak_fname = op.basename(add_suffix(self.inputs.zstat_file, "peaks"))
        lut_fname = seg_fname.replace(".nii.gz", ".txt")

        self._seg_fname = seg_fname
        self._peak_fname = peak_fname
        self._lut_fname = lut_fname

        # Write the output images
        self.write_output(seg, seg_fname)
        self.write_output(markers, peak_fname)

        # Create a LUT and save it
        lut_data = self.create_lut(markers)
        lut_data.to_csv(lut_fname, "\t", index=False)

        return runtime

    def segment(self, data, peaks):
        """Perform a watershed segmentation based on local maxima."""
        markers = np.zeros_like(data)
        markers[tuple(peaks.T)] = np.arange(len(peaks)) + 1
        seg = morphology.watershed(-data, markers, mask=data > 0)

        return seg, markers

    def create_lut(self, markers):
        """Create a Freesurfer-style lookup tabke for the segmentation."""
        n = int(markers.max())
        colors = [[0, 0, 0]] + sns.husl_palette(n)
        colors = np.hstack([colors, np.zeros((n + 1, 1))])
        lut_data = pd.DataFrame(columns=["#ID", "ROI", "R", "G", "B", "A"],
                                index=np.arange(n + 1))
        names = ["Unknown"] + ["roi_%d" % i for i in range(1, n + 1)]
        lut_data["ROI"] = np.array(names)
        lut_data["#ID"] = np.arange(n + 1)
        lut_data.loc[:, "R":"A"] = (colors * 255).astype(int)

        return lut_data

    def write_output(self, data, fname):
        """Save an array of data to an image like the input z stat."""
        template = nib.load(self.inputs.zstat_file)
        out_img = nib.Nifti1Image(data.astype(np.int16),
                                  template.get_affine(),
                                  template.get_header())
        out_img.set_data_dtype(np.int16)
        out_img.to_filename(fname)

    def _list_outputs(self):

        outputs = self._outputs().get()
        outputs["seg_file"] = op.realpath(self._seg_fname)
        outputs["peak_file"] = op.realpath(self._peak_fname)
        outputs["lut_file"] = op.realpath(self._lut_fname)
        return outputs


class MFXReportInput(BaseInterfaceInputSpec):

    mask_file = File(exits=True)
    zstat_file = File(exsts=True)
    zstat_thresh_file = File(exsts=True)
    localmax_file = File(exists=True)
    cope_file = File(exists=True)
    seg_file = File(exists=True)
    subjects = traits.List()


class MFXReportOutput(TraitedSpec):

    out_files = OutputMultiPath(File(exists=True))


class MFXReport(BaseInterface):

    input_spec = MFXReportInput
    output_spec = MFXReportOutput

    def _run_interface(self, runtime):

        self.out_files = []

        self.save_subject_list()
        self.plot_mask()
        self.plot_full_zstat()
        self.plot_thresh_zstat()
        self.reformat_cluster_table()

        self.peaks = peaks = self._load_peaks()
        if len(peaks):

            self.plot_watershed(peaks)
            self.plot_peaks(peaks)
            self.plot_boxes(peaks)

        else:
            fnames = [nii_to_png(self.inputs.seg_file),
                      nii_to_png(self.inputs.zstat_thresh_file, "_peaks"),
                      op.realpath("peak_boxplot.png")]
            self.out_files.extend(fnames)
            for name in fnames:
                with open(name, "wb"):
                    pass

        return runtime

    def save_subject_list(self):
        """Save the subject list for this analysis."""
        subjects_fname = op.realpath("subjects.txt")
        np.savetxt(subjects_fname, self.inputs.subjects, "%s")
        self.out_files.append(subjects_fname)

    def plot_mask(self):
        """Plot the analysis mask."""
        m = Mosaic(stat=self.inputs.mask_file)
        m.plot_mask()
        out_fname = nii_to_png(self.inputs.mask_file)
        self.out_files.append(out_fname)
        m.savefig(out_fname)
        m.close()

    def plot_full_zstat(self):
        """Plot the unthresholded zstat."""
        m = Mosaic(stat=self.inputs.zstat_file, mask=self.inputs.mask_file)
        m.plot_overlay(cmap="coolwarm", center=True, alpha=.9)
        out_fname = nii_to_png(self.inputs.zstat_file)
        self.out_files.append(out_fname)
        m.savefig(out_fname)
        m.close()

    def plot_thresh_zstat(self):
        """Plot the thresholded zstat."""
        m = Mosaic(stat=self.inputs.zstat_thresh_file,
                   mask=self.inputs.mask_file)
        m.plot_activation(pos_cmap="OrRd_r", vfloor=3.3, alpha=.9)
        out_fname = nii_to_png(self.inputs.zstat_thresh_file)
        self.out_files.append(out_fname)
        m.savefig(out_fname)
        m.close()

    def plot_watershed(self, peaks):
        """Plot the watershed segmentation."""
        palette = sns.husl_palette(len(peaks))
        cmap = mpl.colors.ListedColormap(palette)

        m = Mosaic(stat=self.inputs.seg_file, mask=self.inputs.mask_file)
        m.plot_overlay(thresh=.5, cmap=cmap, vmin=1, vmax=len(peaks))
        out_fname = nii_to_png(self.inputs.seg_file)
        self.out_files.append(out_fname)
        m.savefig(out_fname)
        m.close()

    def plot_peaks(self, peaks):
        """Plot the peaks."""
        palette = sns.husl_palette(len(peaks))
        cmap = mpl.colors.ListedColormap(palette)

        disk_img = self._peaks_to_disks(peaks)
        m = Mosaic(stat=disk_img, mask=self.inputs.mask_file)
        m.plot_overlay(thresh=.5, cmap=cmap, vmin=1, vmax=len(peaks))
        out_fname = nii_to_png(self.inputs.zstat_thresh_file, "_peaks")
        self.out_files.append(out_fname)
        m.savefig(out_fname)
        m.close()

    def plot_boxes(self, peaks):
        """Draw a boxplot to show the distribution of copes at peaks."""
        cope_data = nib.load(self.inputs.cope_file).get_data()
        peak_spheres = self._peaks_to_spheres(peaks).get_data()
        peak_dists = np.zeros((cope_data.shape[-1], len(peaks)))
        for i, peak in enumerate(peaks, 1):
            sphere_mean = cope_data[peak_spheres == i].mean(axis=(0))
            peak_dists[:, i - 1] = sphere_mean

        with sns.axes_style("whitegrid"):
            f, ax = plt.subplots(figsize=(9, float(len(peaks)) / 3 + 0.33))

        try:
            # seaborn >= 0.6
            sns.boxplot(data=peak_dists, palette="husl", orient="h", ax=ax)
            labels = np.arange(len(peaks)) + 1
        except TypeError:
            # seaborn < 0.6
            pal = sns.husl_palette(peak_dists.shape[1])[::-1]
            sns.boxplot(peak_dists[:, ::-1], color=pal, ax=ax, vert=False)
            labels = np.arange(len(peaks))[::-1] + 1

        sns.despine(left=True, bottom=True)
        ax.axvline(0, c=".3", ls="--")
        ax.set(yticklabels=labels, ylabel="Local Maximum", xlabel="COPE Value")

        out_fname = op.realpath("peak_boxplot.png")
        self.out_files.append(out_fname)
        f.savefig(out_fname, bbox_inches="tight")
        plt.close(f)

    def reformat_cluster_table(self):
        """Add some info to an FSL cluster file and format it properly."""
        in_fname = self.inputs.localmax_file
        df = pd.read_table(in_fname, delimiter="\t")
        df = df[["Cluster Index", "Value", "x", "y", "z"]]
        df.columns = ["Cluster", "Value", "x", "y", "z"]
        df.index.name = "Peak"

        # Find out where the peaks most likely are
        if len(df):
            coords = df[["x", "y", "z"]].values
            loc_df = locator.locate_peaks(coords)
            df = pd.concat([df, loc_df], axis=1)
            mni_coords = locator.vox_to_mni(coords).T
            for i, ax in enumerate(["x", "y", "z"]):
                df[ax] = mni_coords[i]

        out_fname = op.abspath(op.basename(in_fname[:-3] + "csv"))
        self.out_files.append(out_fname)
        df.to_csv(out_fname)

    def _load_peaks(self):

        peak_df = pd.read_table(self.inputs.localmax_file, "\t")
        peaks = peak_df[["x", "y", "z"]].values
        return peaks

    def _peaks_to_disks(self, peaks, r=4):

        zstat_img = nib.load(self.inputs.zstat_file)

        x, y = np.ogrid[-r:r + 1, -r:r + 1]
        disk = x ** 2 + y ** 2 <= r ** 2
        dilator = np.dstack([disk, disk, np.zeros_like(disk)])
        disk_data = self._dilate_peaks(zstat_img.shape, peaks, dilator)

        disk_img = nib.Nifti1Image(disk_data, zstat_img.get_affine())
        return disk_img

    def _peaks_to_spheres(self, peaks, r=4):

        zstat_img = nib.load(self.inputs.zstat_file)

        x, y, z = np.ogrid[-r:r + 1, -r:r + 1, -r:r + 1]
        dilator = x ** 2 + y ** 2 + z ** 2 <= r ** 2
        sphere_data = self._dilate_peaks(zstat_img.shape, peaks, dilator)

        sphere_img = nib.Nifti1Image(sphere_data, zstat_img.get_affine())
        return sphere_img

    def _dilate_peaks(self, shape, peaks, dilator):

        peak_data = np.zeros(shape)
        disk_data = np.zeros(shape)

        for i, peak in enumerate(peaks, 1):
            spot = np.zeros(shape, np.bool)
            spot[tuple(peak)] = 1
            peak_data += spot
            disk = binary_dilation(spot, dilator)
            disk_data[disk] = i

        return disk_data

    def _list_outputs(self):

        outputs = self._outputs().get()
        outputs["out_files"] = self.out_files
        return outputs
