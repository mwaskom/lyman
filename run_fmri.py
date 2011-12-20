#! /usr/bin/env python
import os
import re
import sys
import shutil
import os.path as op

import matplotlib as mpl
mpl.use("Agg")
from nipype.pipeline.engine import Node
from nipype.interfaces.io import DataGrabber, DataSink

import tools
from tools.commandline import parser
from workflows.preproc import create_preprocessing_workflow


def main(arglist):

    args = parse_args(arglist)

    project = tools.gather_project_info()
    exp = gather_experiment_info(args.experiment)

    os.environ["SUBJECTS_DIR"] = project["data_dir"]

    sys.path.insert(0, os.path.abspath("."))

    subject_list = tools.determine_subjects(args.subjects)

    subj_source = tools.make_subject_source(subject_list)

    preproc, preproc_input, preproc_output = create_preprocessing_workflow(
                              do_slice_time_cor=exp["slice_time_correction"],
                              frames_to_toss=exp["frames_to_toss"],
                              interleaved=exp["interleaved"],
                              slice_order=exp["slice_order"],
                              TR=exp["TR"],
                              smooth_fwhm=exp["smooth_fwhm"],
                              highpass_sigma=exp["highpass_sigma"])

    preproc_source = Node(DataGrabber(infields=["subject_id"],
                                      outfields=["timeseries"],
                                      base_directory=project["data_dir"],
                                      template=exp["source_template"],
                                      sort_filelist=True),
                          name="preproc_source")

    preproc_source.inputs.template_args = dict(timeseries=[["subject_id"]])

    preproc_inwrap = tools.InputWrapper(preproc, subj_source,
                                        preproc_source, preproc_input)

    preproc_inwrap.connect_inputs()

    preproc_sink = Node(DataSink(base_directory=op.join(
                                 project["analysis_dir"], args.experiment)),
                        name="preproc_sink")

    preproc_outwrap = tools.OutputWrapper(preproc, subj_source,
                                          preproc_sink, preproc_output)

    preproc_outwrap.set_subject_container()
    preproc_outwrap.set_mapnode_substitutions(exp["n_runs"])
    preproc_outwrap.sink_outputs("preproc")

    preproc.base_dir = op.join(project["working_dir"], args.experiment)

    preproc.config = dict(crashdump_dir="/tmp")

    run_workflow(preproc, "preproc", args)

    if project["rm_working_dir"]:
        shutil.rmtree(
            op.join(project["working_dir"], args.experiment, "preproc"))


def gather_experiment_info(experiment_name, altmodel=None):

    try:
        if altmodel is not None:
            experiment_name = "%s-%s" % (experiment_name, altmodel)
        exp = __import__("experiments." + experiment_name,
                         fromlist=["experiments"])
    except ImportError:
        print "ERROR: Could not import experiments/%s.py" % experiment_name
        sys.exit()

    return dict(
        [(k, v) for k, v in exp.__dict__.items() if not re.match("__.*__", k)])


def run_workflow(wf, name, args):

    plugin, plugin_args = tools.determine_engine(args)
    if name in args.workflows:
        wf.run(plugin, plugin_args)


def parse_args(arglist):

    parser.add_argument("-experiment", help="experimental paradigm")
    parser.add_argument("-altmodel", help="alternate model to estimate")
    parser.add_argument("-workflows", nargs="*",
                        choices=["all", "preproc", "model", "reg", "ffx"],
                        help="which workflos to run")
    parser.add_argument("-surf", action="store_true",
                        help="run processing for surface analysis")
    parser.add_argument("-native", action="store_true",
                        help="run fixed effect analysis on native surface")
    return parser.parse_args(arglist)

if __name__ == "__main__":
    main(sys.argv[1:])
