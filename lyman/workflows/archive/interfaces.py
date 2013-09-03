import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nipype.interfaces.base import (CommandLineInputSpec, CommandLine, TraitedSpec, BaseInterface,
                                    OutputMultiPath, File, traits)
from nipype.interfaces.fsl.base import FSLCommand, FSLCommandInputSpec
from nipype.utils.filemanip import fname_presuffix

class CheckRegInput(FSLCommandInputSpec):
    
    in_file = File(exists=True, argstr="%s", position=1)
    out_file = File(genfile=True, argstr="%s", position=2)

class CheckRegOutput(TraitedSpec):

    out_file = File(exists=True)

class CheckReg(FSLCommand):

    _cmd = "check_mni_reg"
    input_spec = CheckRegInput
    output_spec = CheckRegOutput

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["out_file"] = fname_presuffix(self.inputs.in_file,
                                              suffix="_to_mni.png",
                                              use_ext=False,
                                              newpath=os.getcwd())
        return outputs

    def _gen_filename(self, name):
        if name == "out_file":
            return self._list_outputs()[name]
        return None

class TimeSeriesMovieInput(FSLCommandInputSpec):

    in_file = File(exists=True,argstr="-ts %s")
    ref_type = traits.String(argstr="-ref %s")
    plot_file = File(exists=True,argstr="-plot %s")
    norm_plot = traits.Bool(argstr="-normplot")
    art_min = traits.Float(argstr="-min %.3f")
    art_max = traits.Float(argstr="-max %.3f")
    out_file = traits.File(genfile=True, argstr="-out %s")

class TimeSeriesMovieOutput(TraitedSpec):

    out_file = File(exists=True)

class TimeSeriesMovie(FSLCommand):

    _cmd = "ts_movie"
    input_spec = TimeSeriesMovieInput
    output_spec = TimeSeriesMovieOutput

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["out_file"] = fname_presuffix(self.inputs.in_file,
                                              suffix=".gif",
                                              use_ext=False,
                                              newpath=os.getcwd())
        return outputs

    def _gen_filename(self, name):
        if name == "out_file":
            return self._list_outputs()[name]
        return None


class XCorrCoefInput(TraitedSpec):

    design_matrix = File(exists=True,mandatory=True,desc="FEAT design matrix")

class XCorrCoefOutput(TraitedSpec):

    corr_png = File(exists=True,desc="graphical representation of design correlaton matrix")

class XCorrCoef(BaseInterface):

    input_spec = XCorrCoefInput
    output_spec = XCorrCoefOutput

    def _run_interface(self, runtime):
        
        X = np.loadtxt(self.inputs.design_matrix, skiprows=5)
        f = plt.figure()
        ax = f.add_subplot(111)
        ax.matshow(np.corrcoef(X.T), vmin=-1, vmax=1)
        plt.savefig("xcorrcoef.png")

        runtime.returncode=0
        return runtime

    def _list_outputs(self):

        outputs = self._outputs().get()
        outputs["corr_png"] = os.path.join(os.getcwd(), "xcorrcoef.png")
        return outputs


class MayaviShotsInputSpec(CommandLineInputSpec):
    
    server_args = traits.Str('"-screen 0 1024x768x24"', position=-4, usedefault=True, argstr="--server-args=%s")
    mayavi_script = File("/mindhive/gablab/u/mwaskom/mayavi_shots.py", position=-3, usedefault=True, argstr="%s")
    hemi = traits.Str(position=-2, argstr="%s")
    in_file = File(exists=True, mandatory=True, position=-1, argstr="%s")

class MayaviShotsOutputSpec(TraitedSpec):

    snapshots = OutputMultiPath()

class MayaviShots(CommandLine):

    _cmd = "/mindhive/gablab/u/mwaskom/xvfb-run"
    input_spec = MayaviShotsInputSpec
    output_spec = MayaviShotsOutputSpec

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['snapshots'] = [os.path.join(os.getcwd(),
                                             "%s-%s.png"%(self.inputs.hemi, v)) for v in ["ant","lat",
                                                                                          "post","med"]]
        return outputs
