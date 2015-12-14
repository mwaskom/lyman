import os
import json
import os.path as op
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (BaseInterface, BaseInterfaceInputSpec,
                                    TraitedSpec, InputMultiPath, File, traits)


class SaveParametersInput(BaseInterfaceInputSpec):

    exp_info = traits.Dict()
    in_file = traits.Either(InputMultiPath(File(exists=True)),
                            File(exists=True))


class SaveParametersOutput(TraitedSpec):

    json_file = File(exists=True)


class SaveParameters(BaseInterface):

    input_spec = SaveParametersInput
    output_spec = SaveParametersOutput
    _always_run = True

    def _run_interface(self, runtime):

        with open("experiment_info.json", "w") as fp:
            json.dump(self.inputs.exp_info, fp, sort_keys=True, indent=2)

        return runtime

    def _list_outputs(self):

        outputs = self._outputs().get()
        outputs["json_file"] = op.abspath("experiment_info.json")
        return outputs


def dump_exp_info(exp_info, timeseries):
    """Dump the exp_info dict into a json file."""
    json_file = op.abspath("experiment_info.json")
    with open(json_file, "w") as fp:
        json.dump(exp_info, fp, sort_keys=True, indent=2)
        return json_file


def add_suffix(fname, suffix):
    """Insert a suffix into a filename before the extension."""
    out_fname = fname_presuffix(fname, suffix="_" + suffix,
                                use_ext=True)
    return out_fname


def nii_to_png(fname, suffix=""):
    """Return a path to write a local png based on an image."""
    out_fname = fname_presuffix(fname, suffix=suffix + ".png",
                                newpath=os.getcwd(),
                                use_ext=False)
    return out_fname
