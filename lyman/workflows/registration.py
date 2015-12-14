"""Registration module contains workflows to move all runs into a common space.

There are two possible spaces:
- epi: linear transform of several runs into the first run's space
- mni: nonlinear warp to FSL's MNI152 space

A registration can be performed over two different classes of inputs
- timeseries: the preprocessed 4D timeseries
- model: 3D images of first-level model outputs

For normalizations to mni space, two different warps can be used
- fsl: a standard FLIRT and FNIRT based normalization
- ants: a superior normalization using SyN

"""
import os
import os.path as op
import shutil
import numpy as np

from nipype import Workflow, Node, IdentityInterface
from nipype.interfaces import fsl
from nipype.interfaces.base import (BaseInterface,
                                    BaseInterfaceInputSpec,
                                    InputMultiPath, OutputMultiPath,
                                    TraitedSpec, File, Directory,
                                    traits, isdefined)

from lyman.tools import add_suffix, submit_cmdline

spaces = ["epi", "mni"]


def create_reg_workflow(name="reg", space="mni",
                        regtype="model", method="fsl",
                        residual=False, cross_exp=False):
    """Flexibly register files into one of several common spaces."""

    # Define the input fields flexibly
    if regtype == "model":
        fields = ["copes", "varcopes", "sumsquares"]
    elif regtype == "timeseries":
        fields = ["timeseries"]

    if cross_exp:
        fields.extend(["first_rigid"])

    fields.extend(["means", "masks", "rigids"])

    if space == "mni":
        fields.extend(["affine", "warpfield"])
    else:
        fields.extend(["tkreg_rigid"])

    inputnode = Node(IdentityInterface(fields), "inputnode")

    # Grap the correct interface class dynamically
    interface_name = "{}{}Registration".format(space.upper(),
                                               regtype.capitalize())
    reg_interface = globals()[interface_name]
    transform = Node(reg_interface(method=method), "transform")

    # Sanity check on inputs
    if regtype == "model" and residual:
        raise ValueError("residual and regtype=model does not make sense")

    # Set the kind of timeseries
    if residual:
        transform.inputs.residual = True

    outputnode = Node(IdentityInterface(["out_files"]), "outputnode")

    # Define the workflow
    regflow = Workflow(name=name)

    # Connect the inputs programatically
    for field in fields:
        regflow.connect(inputnode, field, transform, field)

    # The transform node only ever has one output
    regflow.connect(transform, "out_files", outputnode, "out_files")

    return regflow, inputnode, outputnode


# =============================================================================#
# This is perhaps too complex, but it handles a rather tricky bit of logic

class RegistrationInput(BaseInterfaceInputSpec):

    means = InputMultiPath(File(exists=True))
    masks = InputMultiPath(File(exists=True))
    rigids = InputMultiPath(File(exists=True))
    method = traits.Enum("ants", "fsl")


class ModelRegInput(BaseInterfaceInputSpec):

    copes = InputMultiPath(File(exists=True))
    varcopes = InputMultiPath(File(exists=True))
    sumsquares = InputMultiPath(File(exists=True))


class TimeseriesRegInput(BaseInterfaceInputSpec):

    timeseries = InputMultiPath(File(exists=True))
    residual = traits.Bool(default=False)


class MNIRegInput(BaseInterfaceInputSpec):

    warpfield = File(exists=True)
    affine = File(exists=True)


class EPIRegInput(BaseInterfaceInputSpec):

    rigids = InputMultiPath(File(exists=True))
    first_rigid = File(exists=True)
    tkreg_rigid = File(exists=True)


class MNIModelRegInput(MNIRegInput,
                       ModelRegInput,
                       RegistrationInput):

    pass


class EPIModelRegInput(EPIRegInput,
                       ModelRegInput,
                       RegistrationInput):

    pass


class MNITimeseriesRegInput(MNIRegInput,
                            TimeseriesRegInput,
                            RegistrationInput):

    pass


class EPITimeseriesRegInput(EPIRegInput,
                            TimeseriesRegInput,
                            RegistrationInput):

    pass


class RegistrationOutput(TraitedSpec):

    out_files = OutputMultiPath(Directory(exists=True))


class Registration(BaseInterface):

    output_spec = RegistrationOutput

    def _list_outputs(self):

        outputs = self._outputs().get()
        outputs["out_files"] = self.out_files
        return outputs

    def apply_ants_warp(self, runtime, in_file, out_file, rigid):

        out_rigid = op.basename(add_suffix(out_file, "anat"))

        continuous_interp = dict(trilinear="trilin",
                                 spline="cubic")[self.interp]
        interp = "nearest" if "mask" in in_file else continuous_interp
        cmdline_rigid = ["mri_vol2vol",
                         "--mov", in_file,
                         "--reg", rigid,
                         "--fstarg",
                         "--" + interp,
                         "--o", out_rigid,
                         "--no-save-reg"]
        runtime = submit_cmdline(runtime, cmdline_rigid)

        continuous_interp = dict(trilinear="trilin",
                                 spline="BSpline")[self.interp]
        interp = "NN" if "mask" in in_file else continuous_interp
        cmdline_warp = ["WarpImageMultiTransform",
                        "3",
                        out_rigid,
                        out_file,
                        self.inputs.warpfield,
                        self.inputs.affine,
                        "-R", self.ref_file]
        if interp != "trilin":
            cmdline_warp.append("--use-" + interp)
        runtime = submit_cmdline(runtime, cmdline_warp)
        return runtime

    def apply_fsl_warp(self, runtime, in_file, out_file, rigid):

        interp = "nn" if "mask" in in_file else self.interp
        cmdline = ["applywarp",
                   "-i", in_file,
                   "-o", out_file,
                   "-r", self.ref_file,
                   "-w", self.inputs.warpfield,
                   "--interp={}".format(interp),
                   "--premat={}".format(rigid)]
        runtime = submit_cmdline(runtime, cmdline)
        return runtime

    def apply_fsl_rigid(self, runtime, in_file, out_file, rigid):

        interp = "nn" if "mask" in in_file else self.interp
        cmdline = ["applywarp",
                   "-i", in_file,
                   "-o", out_file,
                   "-r", self.ref_file,
                   "--interp={}".format(interp),
                   "--premat={}".format(rigid)]
        runtime = submit_cmdline(runtime, cmdline)
        return runtime


class MNIRegistration(object):

    ref_file = fsl.Info.standard_image("avg152T1_brain.nii.gz")


class EPIRegistration(object):

    def combine_rigids(self, runtime, first_rigid, second_rigid):
        """Invert the first rigid and combine with the second.

        This creates a transformation from run n to run 1 space,
        through the anatomical coregistration.

        """
        first_inv = op.basename(add_suffix(first_rigid, "inv"))
        cmdline_inv = ["convert_xfm",
                       "-omat", first_inv,
                       "-inverse",
                       first_rigid]
        runtime = submit_cmdline(runtime, cmdline_inv)

        full_rigid = op.basename(add_suffix(second_rigid, "to_epi"))
        cmdline_full = ["convert_xfm",
                        "-omat", full_rigid,
                        "-concat", first_inv,
                        second_rigid]
        runtime = submit_cmdline(runtime, cmdline_full)

        return runtime, full_rigid


class ModelRegistration(object):

    interp = "trilinear"

    def unpack_files(self, field, n_runs):
        """Transform a long list into a list of lists.

        The model outputs are grabbed as a list of file_ij where i is
        the run number and j is the contrast number. j indexes fastest.

        """
        files = getattr(self.inputs, field)
        files = map(list, np.split(np.array(files), n_runs))
        return files


class TimeseriesRegistration(object):

    interp = "spline"


class MNIModelRegistration(MNIRegistration,
                           ModelRegistration,
                           Registration):

    input_spec = MNIModelRegInput

    def _run_interface(self, runtime):

        # Get a reference to either the ANTS or FSL function
        method_name = "apply_{}_warp".format(self.inputs.method)
        warp_func = getattr(self, method_name)

        # Unpack the long file lists to be ordered by run
        n_runs = len(self.inputs.rigids)
        copes = self.unpack_files("copes", n_runs)
        varcopes = self.unpack_files("varcopes", n_runs)
        sumsquares = self.unpack_files("sumsquares", n_runs)

        out_files = []
        for i in range(n_runs):

            # Set up the output directory
            out_dir = "run_{}/".format(i + 1)
            os.mkdir(out_dir)
            out_files.append(op.realpath(out_dir))

            # Select the files for this run
            run_rigid = self.inputs.rigids[i]
            run_copes = copes[i]
            run_varcopes = varcopes[i]
            run_sumsquares = sumsquares[i]
            run_mask = [self.inputs.masks[i]]
            run_mean = [self.inputs.means[i]]
            all_files = (run_copes + run_varcopes +
                         run_sumsquares + run_mask + run_mean)

            # Apply the transformation to each file
            for in_file in all_files:

                out_fname = op.basename(add_suffix(in_file, "warp"))
                out_file = op.join(out_dir, out_fname)
                runtime = warp_func(runtime, in_file, out_file, run_rigid)

        self.out_files = out_files
        return runtime


class EPIModelRegistration(EPIRegistration,
                           ModelRegistration,
                           Registration):

    input_spec = EPIModelRegInput

    @property
    def ref_file(self):

        return self.inputs.copes[0]

    def _run_interface(self, runtime):

        n_runs = len(self.inputs.rigids)
        if isdefined(self.inputs.first_rigid):
            first_rigid = self.inputs.first_rigid
        else:
            first_rigid = self.inputs.rigids[0]

        # Unpack the long file lists to be ordered by run
        copes = self.unpack_files("copes", n_runs)
        varcopes = self.unpack_files("varcopes", n_runs)
        sumsquares = self.unpack_files("sumsquares", n_runs)

        out_files = []
        for i in range(n_runs):

            # Set up the output directory
            out_dir = "run_{}/".format(i + 1)
            os.mkdir(out_dir)
            out_files.append(op.realpath(out_dir))

            # Combine the transformations
            run_rigid = self.inputs.rigids[i]
            runtime, full_rigid = self.combine_rigids(runtime,
                                                      first_rigid,
                                                      run_rigid)

            # Select the files for this run
            run_copes = copes[i]
            run_varcopes = varcopes[i]
            run_sumsquares = sumsquares[i]
            run_mask = [self.inputs.masks[i]]
            run_mean = [self.inputs.means[i]]
            all_files = (run_copes + run_varcopes +
                         run_sumsquares + run_mask + run_mean)

            # Apply the transformation to each file
            for in_file in all_files:

                out_fname = op.basename(add_suffix(in_file, "xfm"))
                out_file = op.join(out_dir, out_fname)
                runtime = self.apply_fsl_rigid(runtime, in_file,
                                               out_file, full_rigid)

            # Copy the matrix to go from this space to the anatomy
            if not i:
                tkreg_fname = op.basename(self.inputs.tkreg_rigid)
                out_tkreg = op.join(out_dir, tkreg_fname)
                shutil.copyfile(self.inputs.tkreg_rigid, out_tkreg)

        self.out_files = out_files
        return runtime


class MNITimeseriesRegistration(MNIRegistration,
                                TimeseriesRegistration,
                                Registration):

    input_spec = MNITimeseriesRegInput

    def _run_interface(self, runtime):

        # Get a reference to either the ANTS or FSL function
        method_name = "apply_{}_warp".format(self.inputs.method)
        warp_func = getattr(self, method_name)

        n_runs = len(self.inputs.rigids)
        out_files = []
        for i in range(n_runs):

            # Set up the output directory
            out_dir = "run_{}/".format(i + 1)
            os.mkdir(out_dir)
            out_files.append(op.realpath(out_dir))

            # Warp the timeseries files
            run_timeseries = self.inputs.timeseries[i]
            name = "res4d" if self.inputs.residual else "timeseries"
            out_timeseries = op.join(out_dir, name + "_warp.nii.gz")
            run_rigid = self.inputs.rigids[i]
            runtime = warp_func(runtime, run_timeseries,
                                out_timeseries, run_rigid)

            # Warp the mask file
            run_mask = self.inputs.masks[i]
            out_mask_fname = op.basename(add_suffix(run_mask, "warp"))
            out_mask = op.join(out_dir, out_mask_fname)
            runtime = warp_func(runtime, run_mask, out_mask, run_rigid)

            # Warp the mean file
            run_mean = self.inputs.means[i]
            out_mean_fname = op.basename(add_suffix(run_mean, "warp"))
            out_mean = op.join(out_dir, out_mean_fname)
            runtime = warp_func(runtime, run_mean, out_mean, run_rigid)

        self.out_files = out_files
        return runtime


class EPITimeseriesRegistration(EPIRegistration,
                                TimeseriesRegistration,
                                Registration):

    input_spec = EPITimeseriesRegInput

    @property
    def ref_file(self):

        return self.inputs.timeseries[0]

    def _run_interface(self, runtime):

        n_runs = len(self.inputs.rigids)
        if isdefined(self.inputs.first_rigid):
            first_rigid = self.inputs.first_rigid
        else:
            first_rigid = self.inputs.rigids[0]

        out_files = []
        for i in range(n_runs):

            # Set up the output directory
            out_dir = "run_{}/".format(i + 1)
            os.mkdir(out_dir)
            out_files.append(op.realpath(out_dir))

            # Combine the transformations
            run_rigid = self.inputs.rigids[i]
            runtime, full_rigid = self.combine_rigids(runtime,
                                                      first_rigid,
                                                      run_rigid)

            # Warp the timeseries files
            run_timeseries = self.inputs.timeseries[i]
            name = "res4d" if self.inputs.residual else "timeseries"
            out_timeseries = op.join(out_dir, name + "_xfm.nii.gz")
            runtime = self.apply_fsl_rigid(runtime, run_timeseries,
                                           out_timeseries, full_rigid)

            # Warp the mask file
            run_mask = self.inputs.masks[i]
            out_mask_fname = op.basename(add_suffix(run_mask, "xfm"))
            out_mask = op.join(out_dir, out_mask_fname)
            runtime = self.apply_fsl_rigid(runtime, run_mask,
                                           out_mask, full_rigid)

            # Warp the mean file
            run_mean = self.inputs.means[i]
            out_mean_fname = op.basename(add_suffix(run_mean, "xfm"))
            out_mean = op.join(out_dir, out_mean_fname)
            runtime = self.apply_fsl_rigid(runtime, run_mean,
                                           out_mean, full_rigid)

            # Copy the matrix to go from this space to the anatomy
            if not i:
                tkreg_fname = op.basename(self.inputs.tkreg_rigid)
                out_tkreg = op.join(out_dir, tkreg_fname)
                shutil.copyfile(self.inputs.tkreg_rigid, out_tkreg)

        self.out_files = out_files
        return runtime
