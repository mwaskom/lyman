import os

from nipype import (Workflow, Node, MapNode, JoinNode,
                    Function, IdentityInterface, SelectFiles, DataSink)
from nipype.interfaces.base import (traits, BaseInterface,
                                    BaseInterfaceInputSpec, TraitedSpec,
                                    File, InputMultiPath, OutputMultiPath)
from nipype.interfaces import fsl, freesurfer as fs, utility as pipeutil

from ..tools.submission import submit_cmdline


def define_preproc_workflow(proj_info, sess_info, exp_info):

    # --- Workflow parameterization

    subject_iterables = list(sess_info.keys())
    subject_source = Node(IdentityInterface(["subject"]),
                          name="subject_source",
                          iterables=("subject", subject_iterables))

    session_iterables = {
        subj: [(subj, sess) for sess in sess_info[subj]]
        for subj in sess_info
    }
    session_source = Node(IdentityInterface(["subject", "session"]),
                          name="session_source",
                          itersource=("subject_source", "subject"),
                          iterables=("session", session_iterables))

    run_iterables = {
        (subj, sess): [(subj, sess, run) for run in sess_info[subj][sess]]
        for subj, sess in sess_info.items()
    }
    run_source = Node(IdentityInterface(["subject", "session", "run"]),
                      name="run_source",
                      itersource=("session_source", "session"),
                      iterables=("run", run_iterables))

    # --- Semantic information

    def info_func(info_tuple):
        try:
            subject, session = info_tuple
            return subject, session
        except ValueError:
            subject, session, run = info_tuple
            return subject, session, run

    sesswise_info = Node(Function("info_tuple",
                                  ["subject", "session"],
                                  info_func),
                         "sesswise_info")

    runwise_info = Node(Function("info_tuple",
                                 ["subject", "session", "run"],
                                 info_func),
                        "runwise_info")

    # --- Input file selection

    session_templates = dict(se=exp_info.se_template)
    sesswise_input = Node(SelectFiles(session_templates,
                                      base_directory=proj_info.data_dir),
                          "sesswise_input")

    run_templates = dict(ts=exp_info.ts_template, sb=exp_info.sb_template)
    runwise_input = Node(SelectFiles(run_templates,
                                     base_directory=proj_info.data_dir),
                         "runwise_input")

    # --- Warpfield estimation using topup

    # Distortion warpfield estimation
    phase_encoding = ["y", "y", "y", "y-", "y-", "y-"]
    readout_times = [1, 1, 1, 1, 1, 1]
    estimate_distortions = Node(fsl.TOPUP(encoding_direction=phase_encoding,
                                          readout_times=readout_times,
                                          config="b02b0.cnf"),
                                "estimate_distortions")

    # Average distortion-corrected spin-echo images
    average_se = Node(fsl.MeanImage(out_file="se_restored.nii.gz"),
                      "average_se")

    # Select first warpfield image from output list
    select_warp = Node(pipeutil.Select(index=[0]), "select_warp")

    # Define a mask of areas with large distortions
    thresh_ops = "-abs -thr 4 -Tmax -binv"
    mask_distortions = Node(fsl.ImageMaths(op_string=thresh_ops),
                            "mask_distortions")

    # --- Registration of SBRef to SE-EPI (with distortions)

    sb2se = Node(fsl.FLIRT(dof=6), "sb2se")

    # --- Registration of SE-EPI (without distortions) to Freesurfer anatomy

    se2anat = Node(fs.BBRegister(init="fsl",
                                 contrast_type="t2",
                                 out_fsl_file="se2anat_flirt.mat",
                                 out_reg_file="se2anat_tkreg.dat"),
                   "se2anat")

    # --- Definition of common cross-session space

    se2native = JoinNode(NativeTransform(),
                         name="se2native",
                         joinsource="session_source",
                         joinfield=["session_info",
                                    "in_matrices", "in_volumes"])

    # --- Associate native-space transforms with data from correct session

    def select_transform_func(session_info, subject, session, in_matrices):

        for info, matrix in zip(session_info, in_matrices):
            if info == (subject, session):
                out_matrix = matrix
        return out_matrix

    select_sesswise = Node(Function(["session_info",
                                     "subject", "session",
                                     "in_matrices"],
                                    "out_matrix",
                                    select_transform_func),
                           "select_sesswise")

    select_runwise = select_sesswise.clone("select_runwise")

    # --- Restore each sessions SE image in native space then average

    split_se = Node(fsl.Split(dimension="t"), "split_se")

    restore_se = MapNode(fsl.ApplyWarp(interp="spline", relwarp=True),
                         ["in_file", "premat", "field_file"],
                         "restore_se")

    def flatten_file_list(in_files):
        out_files = [item for sublist in in_files for item in sublist]
        return out_files

    combine_se = JoinNode(Function("in_files", "out_files", flatten_file_list),
                          name="combine_se",
                          joinsource="session_source",
                          joinfield=["in_files"])

    merge_se = Node(fsl.Merge(dimension="t"), name="merge_se")

    average_native = Node(fsl.MeanImage(out_file="se_native.nii.gz"),
                          "average_native")

    # --- Motion correction of timeseries to SBRef (with distortions)

    ts2sb = Node(fsl.MCFLIRT(save_mats=True, save_plots=True),
                 "ts2sb")

    # --- Combined motion correction and unwarping of timeseries

    # Split the timeseries into each frame
    split_ts = Node(fsl.Split(dimension="t"), "split_ts")

    # Concatenation ts2sb and sb2se rigid transform
    combine_rigids = MapNode(fsl.ConvertXFM(concat_xfm=True),
                             "in_file", "combine_rigids")

    # Simultaneously apply rigid transform and nonlinear warpfield
    restore_ts_frames = MapNode(fsl.ApplyWarp(interp="spline", relwarp=True),
                                ["in_file", "premat"],
                                "restore_ts")

    # Recombine the timeseries frames into a 4D image
    merge_ts = Node(fsl.Merge(merged_file="ts_restored.nii.gz",
                              dimension="t"), "merge_ts")

    # Take a temporal average of the timeseries
    average_ts = Node(fsl.MeanImage(out_file="mean_restored.nii.gz"),
                      "average_ts")

    # --- Workflow ouptut

    output_dir = os.path.realpath("python_script_outputs")
    file_output = Node(DataSink(base_directory=output_dir),
                       "file_output")

    # === Assemble pipeline

    workflow = Workflow(name="prisma_preproc_multirun",
                        base_dir="nipype_cache")

    workflow.connect([
        (subject_source, session_source,
            [("subject", "subject")]),
        (subject_source, run_source,
            [("subject", "subject")]),
        (session_source, run_source,
            [("session", "session")]),
        (session_source, sesswise_info,
            [("session", "info_tuple")]),
        (run_source, runwise_info,
            [("run", "info_tuple")]),
        (sesswise_info, sesswise_input,
            [("subject", "subject"),
             ("session", "session")]),
        (runwise_info, runwise_input,
            [("subject", "subject"),
             ("session", "session"),
             ("run", "run")]),
        (sesswise_input, estimate_distortions,
            [("se", "in_file")]),
        (estimate_distortions, select_warp,
            [("out_warps", "inlist")]),
        (select_warp, mask_distortions,
            [("out", "in_file")]),
        (estimate_distortions, average_se,
            [("out_corrected", "in_file")]),
        (sesswise_info, se2anat,
            [("subject", "subject_id")]),
        (average_se, se2anat,
            [("out_file", "source_file")]),
        (session_source, se2native,
            [("session", "session_info")]),
        (sesswise_input, se2native,
            [("se", "in_volumes")]),
        (se2anat, se2native,
            [("out_fsl_file", "in_matrices")]),
        (se2native, select_sesswise,
            [("out_matrices", "in_matrices"),
             ("out_template", "in_templates"),
             ("session_info", "session_info")]),
        (sesswise_info, select_sesswise,
            [("subject", "subject"),
             ("session", "session")]),
        (sesswise_input, split_se,
            [("se", "in_file")]),
        (split_se, restore_se,
            [("out_files", "in_file")]),
        (estimate_distortions, restore_se,
            [("out_mats", "premat"),
             ("out_warps", "field_file")]),
        (se2native, restore_se,
            [("out_template", "ref_file")]),
        (select_sesswise, restore_se,
            [("out_matrix", "postmat")]),
        (restore_se, combine_se,
            [("out_file", "in_files")]),
        (combine_se, merge_se,
            [("out_files", "in_files")]),
        (merge_se, average_native,
            [("merged_file", "in_file")]),
        (runwise_input, ts2sb,
            [("ts", "in_file"),
             ("sb", "ref_file")]),
        (runwise_input, split_ts,
            [("ts", "in_file")]),
        (runwise_input, sb2se,
            [("sb", "in_file")]),
        (sesswise_input, sb2se,
            [("se", "reference")]),
        (mask_distortions, sb2se,
            [("out_file", "ref_weight")]),
        (ts2sb, combine_rigids,
            [("mat_file", "in_file")]),
        (sb2se, combine_rigids,
            [("out_matrix_file", "in_file2")]),
        (split_ts, restore_ts_frames,
            [("out_files", "in_file")]),
        (combine_rigids, restore_ts_frames,
            [("out_file", "premat")]),
        (select_warp, restore_ts_frames,
            [("out", "field_file")]),
        (se2native, select_runwise,
            [("out_matrices", "in_matrices"),
             ("session_info", "session_info")]),
        (runwise_info, select_runwise,
            [("subject", "subject"),
             ("session", "session")]),
        (se2native, restore_ts_frames,
            [("out_template", "ref_file")]),
        (select_runwise, restore_ts_frames,
            [("out_matrix", "postmat")]),
        (restore_ts_frames, merge_ts,
            [("out_file", "in_files")]),
        (merge_ts, average_ts,
            [("merged_file", "in_file")]),
        (merge_ts, file_output,
            [("merged_file", "@restored_timeseries")]),
        (average_ts, file_output,
            [("out_file", "@mean_func")]),
        (ts2sb, file_output,
            [("par_file", "@realign_params")]),
        (average_native, file_output,
            [("out_file", "@se_native")]),
        (se2native, file_output,
            [("out_tkreg_file", "@tkreg_file")]),
    ])


class NativeTransformInput(BaseInterfaceInputSpec):

    session_info = traits.List(traits.Tuple())
    in_matrices = InputMultiPath(File(exists=True))
    in_volumes = InputMultiPath(File(exists=True))


class NativeTransformOutput(TraitedSpec):

    session_info = traits.List(traits.Tuple())
    out_template = File(exists=True)
    out_matrices = OutputMultiPath(File(exists=True))
    out_flirt_file = File(exists=True)
    out_tkreg_file = File(exists=True)


class NativeTransform(BaseInterface):

    input_spec = NativeTransformInput
    output_spec = NativeTransformOutput

    def _list_outputs(self):

        outputs = self._outputs().get()

        outputs["session_info"] = self.inputs.session_info

        out_matrices = [
            os.path.abspath("se2native_{:04d}.mat".format(i))
            for i, _ in enumerate(self.inputs.in_matrices, 1)
            ]

        outputs["out_matrices"] = out_matrices

        outputs["out_template"] = os.path.abspath("native_template.nii.gz")
        outputs["out_flirt_file"] = os.path.abspath("native2anat_flirt.mat")
        outputs["out_tkreg_file"] = os.path.abspath("native2anat_tkreg.dat")

        return outputs

    def _run_interface(self, runtime):

        subjects_dir = os.environ["SUBJECTS_DIR"]
        subj = set([s for s, _ in self.inputs.session_info]).pop()

        # Convert the anatomical image to nifti
        cmdline = ["mri_convert",
                   os.path.join(subjects_dir, subj, "mri/orig.mgz"),
                   "orig.nii.gz"]

        runtime = submit_cmdline(runtime, cmdline)

        # Compute the intermediate transform
        cmdline = ["midtrans",
                   "--template=orig.nii.gz",
                   "--separate=se2native_",
                   "--out=anat2native_flirt.mat"]
        cmdline.extend(self.inputs.in_matrices)

        runtime = submit_cmdline(runtime, cmdline)

        # Invert the anat2native transformation
        cmdline = ["convert_xfm",
                   "-omat", "native2anat_flirt.mat",
                   "-inverse",
                   "anat2native_flirt.mat"]

        runtime = submit_cmdline(runtime, cmdline)

        # Transform the first volume into the native space to get the geometry
        cmdline = ["flirt",
                   "-in", self.inputs.in_volumes[0],
                   "-ref", self.inputs.in_volumes[0],
                   "-init", "se2native_0001.mat",
                   "-out", "native_template.nii.gz",
                   "-applyxfm"]

        runtime = submit_cmdline(runtime, cmdline)

        # Convert the FSL matrices to tkreg matrix format
        cmdline = ["tkregister2",
                   "--s", subj,
                   "--mov", "native_template.nii.gz",
                   "--fsl", "native2anat_flirt.mat",
                   "--reg", "native2anat_tkreg.dat",
                   "--noedit"]

        runtime = submit_cmdline(runtime, cmdline)

        return runtime
