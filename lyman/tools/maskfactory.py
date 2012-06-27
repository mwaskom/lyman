import os
import os.path as op
import shutil
from tempfile import mkdtemp
from subprocess import call, check_output
from IPython.parallel import Client, error
import numpy as np


class MaskFactory(object):

    def __init__(self, study_dir, subject_list, experiment,
                 mask_name, let_fail=False, force_serial=False):

        self.study_dir = study_dir
        if isinstance(subject_list, list):
            self.subject_list = subject_list
        else:
            self.subject_list = np.loadtxt(subject_list, str)
        self.experiment = experiment
        self.mask_name = mask_name
        self.let_fail = let_fail

        self.data_dir = op.join(study_dir, "data")
        self.anal_dir = op.join(study_dir, "analysis")
        self.temp_dir = mkdtemp()

        os.environ["SUBJECTS_DIR"] = self.data_dir

        if force_serial:
            print "Set to run in serial"
            self.map = map
        else:
            try:
                rc = Client()
                print "Set to run in parallel"
                self.map = rc[:].map_sync
            except error.TimeoutError:
                print "Set to run in serial"
                self.map = map

    def __del__(self):

        shutil.rmtree(self.temp_dir)

    def from_common_label(self, label_file, hemi, dilate=True, keep_going=True):

        subj_label_temp = op.join(self.temp_dir,
                                  hemi + ".%s_label.label")
        label_cmds = []
        for subj in self.subject_list:
            label_cmds.append(
                ["mri_label2label",
                 "--srclabel", label_file,
                 "--srcsubject", "fsaverage",
                 "--trgsubject", subj,
                 "--trglabel", subj_label_temp % subj,
                 "--regmethod", "surface",
                 "--hemi", hemi])

        self.execute(label_cmds, subj_label_temp)

        if keep_going:
            self.from_label(subj_label_temp, hemi, dilate=dilate)

    def from_common_bilat_label(self, label_template, dilate=True):

        lateral_label_temp = op.join(self.temp_dir, "%(hemi)s.%(subj)s_label.label")
        for hemi in ["lh", "rh"]:
            self.from_common_label(label_template % hemi, hemi,
                                  dilate=dilate, keep_going=False)

        self.from_bilat_labels(lateral_label_temp, dilate=dilate)

    def from_label(self, label_template, hemi, dilate=True, keep_going=True):

        hires_mask_template = op.join(self.temp_dir,
                                      hemi + ".%s_hires_mask.nii.gz")
        ref_template = op.join(self.data_dir, "%s/mri/orig.mgz")
        proj_cmds = []
        for subj in self.subject_list:
            proj_cmds.append(
                       ["mri_label2vol",
                        "--label", label_template % subj,
                        "--temp", ref_template % subj,
                        "--identity",
                        "--proj", "frac", "0", "1", ".1",
                        "--hemi", hemi,
                        "--subject", subj,
                        "--o", hires_mask_template % subj])
        self.execute(proj_cmds, hires_mask_template)

        if dilate:
            dil_cmds = []
            for subj in self.subject_list:
                dil_cmds.append(
                          ["fslmaths",
                           hires_mask_template % subj,
                           "-dilF",
                           hires_mask_template % subj])
            self.execute(dil_cmds, hires_mask_template)

        if keep_going:
            self.from_hires_mask(hires_mask_template)

    def from_bilat_labels(self, label_template, dilate=True):

        lateral_mask_temp = op.join(self.temp_dir, "%(hemi)s.%(subj)s_hires_mask.nii.gz")
        hires_mask_template = op.join(self.temp_dir, "%s_hires_mask.nii.gz")

        for hemi in ["lh", "rh"]:
            lateral_temp = label_template % dict(hemi=hemi, subj="%s")
            self.from_label(lateral_temp, hemi, dilate=dilate, keep_going=False)

        combine_cmds = []
        for subj in self.subject_list:
            combine_cmds.append(
                          ["fslmaths",
                           lateral_mask_temp % dict(hemi="lh", subj=subj),
                           "-add",
                           lateral_mask_temp % dict(hemi="rh", subj=subj),
                           "-bin",
                           hires_mask_template % subj])
        self.execute(combine_cmds, hires_mask_template)

        self.from_hires_mask(hires_mask_template)

    def from_hires_atlas(self, hires_atlas_template, region_ids):

        hires_mask_template = op.join(self.temp_dir,
                                      "%s_hires_mask.nii.gz")
        bin_cmds = []
        for subj in self.subject_list:
            cmd_list = ["mri_binarize",
                        "--i", hires_atlas_template % subj,
                        "--dilate", "1",
                        "--o", hires_mask_template % subj]
            try:
                for id in region_ids:
                    cmd_list.extend(["--match", str(id)])
            except TypeError:
                cmd_list.extend(["--match", str(id)])
            bin_cmds.append(cmd_list)
        self.execute(bin_cmds, hires_mask_template)

        self.from_hires_mask(hires_mask_template)

    def from_hires_mask(self, hires_mask_template):

        self.epi_template = op.join(self.anal_dir, self.experiment,
                                    "%s/preproc/run_1/mean_func.nii.gz")
        self.reg_template = op.join(self.anal_dir, self.experiment,
                                    "%s/preproc/run_1/func2anat_tkreg.dat")
        self.epi_mask_template = op.join(self.data_dir, "%s/masks",
                                         self.mask_name + ".nii.gz")
        xfm_cmds = []
        for subj in self.subject_list:
            mask_dir = op.join(self.data_dir, subj, "masks")
            if not op.exists(mask_dir):
                os.mkdir(mask_dir)
            xfm_cmds.append(
                      ["mri_vol2vol",
                       "--mov", self.epi_template % subj,
                       "--targ", hires_mask_template % subj,
                       "--inv",
                       "--o", self.epi_mask_template % subj,
                       "--reg", self.reg_template % subj,
                       "--no-save-reg",
                       "--nearest"])
        self.execute(xfm_cmds, self.epi_mask_template)

        self.make_mask_png()

    def make_mask_png(self):

        overlay_temp = op.join(self.temp_dir, "%s_overlay.nii.gz")
        slices_temp = op.join(self.data_dir, "%s/masks",
                              self.mask_name + ".png")

        overlay_cmds = []
        for subj in self.subject_list:
            overlay_cmds.append(
                          ["overlay", "1", "0",
                           self.epi_template % subj, "-a",
                           self.epi_mask_template % subj, "0.8", "2",
                           overlay_temp % subj])
        self.execute(overlay_cmds, overlay_temp)

        slicer_cmds = []
        for subj in self.subject_list:
            slicer_cmds.append(
                          ["slicer",
                           overlay_temp % subj,
                           "-A", "750",
                           slices_temp % subj])
        self.execute(slicer_cmds, slices_temp)

    def execute(self, cmd_list, out_temp):

        self.map(check_output, cmd_list)
        self.check_exists(out_temp)

    def check_exists(self, fpath_temp):

        fail_list = []
        for subj in self.subject_list:
            if not op.exists(fpath_temp % subj):
                fail_list.append(fpath_temp % subj)
        if fail_list and not self.let_fail:
            print "Failed to write files:"
            print "\n".join(fail_list)
            raise RuntimeError
