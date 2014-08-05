"""Defines a class for flexible functional mask generation."""
import os
import os.path as op
import shutil
from tempfile import mkdtemp
from subprocess import check_output
from IPython.parallel import Client
from IPython.parallel.error import TimeoutError

import numpy as np
import nibabel as nib
import matplotlib as mpl
mpl.use("Agg")

from moss.mosaic import Mosaic
import lyman


class MaskFactory(object):
    """Class for the rapid and flexible creation of functional masks.

    This class can make appropriate calls to external (Freesurfer and
    FSL) command-line programs to take ROIs defined in a variety of
    sources and generate binary mask images in native EPI space.

    """
    def __init__(self, subject_list, experiment, roi_name,
                 orig_type, force_serial=False, debug=False):

        # Set up basic info
        self.subject_list = lyman.determine_subjects(subject_list)
        project = lyman.gather_project_info()
        self.experiment = experiment
        self.roi_name = roi_name
        self.orig_type = orig_type
        self.debug = debug
        if debug:
            print "Setting up for %d subjects" % len(subject_list)
            print "Experiment name:", experiment
            print "ROI name:", roi_name

        # Set up directories
        if project["default_exp"] is not None and experiment is None:
            experiment = project["default_exp"]
        self.experiment = experiment
        self.data_dir = project["data_dir"]
        self.anal_dir = project["analysis_dir"]

        # Set up temporary output
        self.temp_dir = mkdtemp()

        # Set the SUBJECTS_DIR variable for Freesurfer
        os.environ["SUBJECTS_DIR"] = self.data_dir

        # Set up parallel execution
        self.parallel = False
        if force_serial:
            self.map = map
        else:
            try:
                rc = Client()
                self.dv = rc[:]
                self.map = self.dv.map_async
                # Push SUBJECTS_DIR to engines
                self.dv.execute("import os")
                self.dv["data_dir"] = self.data_dir
                self.dv.execute("os.environ['SUBJECTS_DIR'] = data_dir")
                self.parallel = True

            except (TimeoutError, IOError):
                self.map = map
        if debug:
            print "Set to run in %s" % (
                "parallel" if self.parallel else "serial")

        # Set up some persistent templates
        self.epi_template = op.join(self.anal_dir, self.experiment,
                                    "%(subj)s",
                                    "preproc/run_1/mean_func.nii.gz")
        self.fov_template = op.join(self.anal_dir, self.experiment,
                                    "%(subj)s",
                                    "preproc/run_1/functional_mask.nii.gz")
        self.reg_template = op.join(self.anal_dir, self.experiment,
                                    "%(subj)s",
                                    "preproc/run_1/func2anat_tkreg.dat")
        self.out_template = op.join(self.data_dir,
                                    "%(subj)s",
                                    "masks/%s.nii.gz" % self.roi_name)
        if debug:
            print "EPI template: %s" % self.epi_template
            print "Reg template: %s" % self.reg_template
            print "Output template: %s" % self.out_template

        # Ensure the output directory will exist
        for subj in self.subject_list:
            mask_dir = op.join(self.data_dir, subj, "masks")
            if not op.exists(mask_dir):
                os.mkdir(mask_dir)

    def __del__(self):

        if self.debug:
            print "Debug mode: not removing output directory:"
            print self.temp_dir
        else:
            shutil.rmtree(self.temp_dir)

    def from_common_label(self, label_template, hemis, proj_args,
                          save_native=False):
        """Reverse normalize possibly bilateral labels to native space."""
        native_label_temp = op.join(self.temp_dir,
                                    "%(hemi)s.%(subj)s_native_label.label")

        # Transform by subject and hemi
        warp_cmds = []
        for subj in self.subject_list:
            for hemi in hemis:
                args = dict(hemi=hemi, subj=subj)
                cmd = ["mri_label2label",
                       "--srcsubject", "fsaverage",
                       "--trgsubject", subj,
                       "--hemi", hemi,
                       "--srclabel", label_template % args,
                       "--trglabel", native_label_temp % args,
                       "--regmethod", "surface"]
                warp_cmds.append(cmd)

        # Execute the transformation
        self.execute(warp_cmds, native_label_temp)

        # Possibly copy the resulting native space label to
        # the subject's label directory
        if save_native:
            save_temp = op.join(self.data_dir, "%(subj)s", "label",
                                "%(hemi)s." + self.roi_name + ".label")
            for subj in self.subject_list:
                for hemi in hemis:
                    args = dict(subj=subj, hemi=hemi)
                    shutil.copyfile(native_label_temp % args, save_temp % args)

        # Carry on with the native label stage
        self.from_native_label(native_label_temp, hemis, proj_args)

    def from_native_label(self, label_template, hemis, proj_args):
        """Given possibly bilateral native labels, make epi masks."""
        indiv_mask_temp = op.join(self.temp_dir,
                                  "%(hemi)s.%(subj)s_mask.nii.gz")
        # Command list for this step
        proj_cmds = []
        for subj in self.subject_list:
            for hemi in hemis:
                args = dict(hemi=hemi, subj=subj)
                cmd = ["mri_label2vol",
                       "--label", label_template % args,
                       "--temp", self.epi_template % args,
                       "--reg", self.reg_template % args,
                       "--hemi", hemi,
                       "--subject", subj,
                       "--o", indiv_mask_temp % args,
                       "--proj"]
                cmd.extend(proj_args)
                proj_cmds.append(cmd)

        # Execute the projection from a surface label
        self.execute(proj_cmds, indiv_mask_temp)

        # Combine the bilateral masks into the final mask
        combine_cmds = []
        for subj in self.subject_list:
            cmd = ["mri_concat"]
            for hemi in hemis:
                args = dict(hemi=hemi, subj=subj)
                cmd.append(indiv_mask_temp % args)
            cmd.extend(["--max",
                        "--o", self.out_template % dict(subj=subj)])
            combine_cmds.append(cmd)

        # Execute the final step
        self.execute(combine_cmds, self.out_template)

    def from_hires_atlas(self, hires_atlas_template, region_ids, erode):
        """Create epi space mask from index volume (e.g. aseg.mgz"""
        hires_mask_template = op.join(self.temp_dir,
                                      "%(subj)s_hires_mask.nii.gz")

        # First run mri_binarize
        bin_cmds = []
        for subj in self.subject_list:
            args = dict(subj=subj)
            cmd_list = ["mri_binarize",
                        "--i", hires_atlas_template % args,
                        "--o", hires_mask_template % args]
            for id in region_ids:
                cmd_list.extend(["--match", str(id)])
            if erode is not None:
                cmd_list.extend(["--erode", str(erode)])
            bin_cmds.append(cmd_list)
        self.execute(bin_cmds, hires_mask_template)

        self.from_hires_mask(hires_mask_template)

    def from_hires_mask(self, hires_mask_template):
        """Create epi space mask from hires mask (binary) volume."""
        xfm_cmds = []
        for subj in self.subject_list:
            args = dict(subj=subj)
            xfm_cmds.append(
                ["mri_vol2vol",
                 "--mov", self.epi_template % args,
                 "--targ", hires_mask_template % args,
                 "--inv",
                 "--o", self.out_template % args,
                 "--reg", self.reg_template % args,
                 "--no-save-reg",
                 "--nearest"])
        self.execute(xfm_cmds, self.out_template)

    def apply_statistical_mask(self, stat_file_temp,
                               thresh=None, n_voxels=None):
        """Create a mask by binarizing an epi-space fixed effects zstat map."""
        if n_voxels is not None:
            return self.take_top_voxels(stat_file_temp, n_voxels)

        bin_cmds = []
        for subj in self.subject_list:
            args = dict(subj=subj)
            cmd = ["fslmaths",
                   stat_file_temp % args,
                   "-thr", thresh,
                   "-bin",
                   "-mul",
                   self.out_template % args,
                   self.out_template % args]
            bin_cmds.append(cmd)

        self.execute(bin_cmds, self.out_template)

    def take_top_voxels(self, stat_file_temp, n_voxels):
        """Create take the top n voxels within a mask based on a stat map."""
        for subj in self.subject_list:
            args = dict(subj=subj)
            mask_file = self.out_template % args
            stat_file = stat_file_temp % args
            img = nib.load(mask_file)
            mask = img.get_data().astype(bool)
            stat = nib.load(stat_file).get_data()
            stat[~mask] = -np.inf
            not_mask = np.argsort(stat.flat).argsort() < (mask.size - n_voxels)
            mask.flat[not_mask] = 0
            new_img = nib.Nifti1Image(mask, img.get_affine(), img.get_header())
            new_img.to_filename(mask_file)

            if self.debug:
                print "Mask file %s sum: %d" % (mask_file, mask.sum())

    def write_png(self):
        """Write a mosiac png showing the masked voxels."""
        slices_temp = op.join(self.data_dir, "%(subj)s/masks",
                              self.roi_name + ".png")

        for subj in self.subject_list:
            args = dict(subj=subj)
            m = Mosaic(self.epi_template % args,
                       self.out_template % args,
                       self.fov_template % args,
                       step=1, show_mask=False,
                       stat_interp="nearest")
            cmap = mpl.colors.ListedColormap(["#C41E3A"])
            m.plot_overlay(cmap, thresh=.5, alpha=.9, colorbar=False)
            m.savefig(slices_temp % args)
            m.close()

    def execute(self, cmd_list, out_temp):
        """Exceute a list of commands and verify output file existence."""
        res = self.map(check_output, cmd_list)
        if self.parallel:
            if self.debug:
                res.wait_interactive()
            else:
                res.wait()
        if self.parallel:
            if not res.successful():
                raise RuntimeError(res.pyerr)
