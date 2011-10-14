#! /usr/bin/env python
import os
import re
import sys

import matplotlib as mpl
mpl.use("Agg")

import nipype.pipeline.engine as pe

import nipype.interfaces.io as nio
from nipype.interfaces import fsl
from nipype.interfaces import utility 

from workflows.preproc import create_preprocessing_workflow

import util
from util.commandline import parser

def main(arglist):

    args = parse_args(arglist)

    project = util.gather_project_info()
    exp = gather_experiment_info(args.experiment)
    
    os.environ["SUBJECTS_DIR"] = project["data_dir"]

    sys.path.insert(0, os.path.abspath("."))


    subject_list = util.determine_subjects(args.subjects)
    
    # Subject source node
    # -------------------
    subjectsource = pe.Node(utility.IdentityInterface(fields=["subject_id"]),
                            iterables = ("subject_id", subject_list),
                            overwrite=True,
                            name = "subjectsource")

    preproc, preproc_input, preproc_output = create_preprocessing_workflow(
                                      do_slice_time_cor=exp["slice_time_correction"],
                                      frames_to_toss=exp["frames_to_toss"],
                                      interleaved=exp["interleaved"],
                                      slice_order=exp["slice_order"],
                                      TR=exp["TR"],
                                      smooth_fwhm=exp["smooth_fwhm"],
                                      highpass_sigma=exp["highpass_sigma"])


    # Preprocessing
    # =============

    # Preproc datasource node
    preprocsource = pe.Node(nio.DataGrabber(infields=["subject_id"],
                                            outfields=["timeseries"],
                                            base_directory=project["data_dir"],
                                            template=exp["source_template"],
                                            sort_filelist=True),
                            name="preprocsource")

    preprocsource.inputs.template_args = exp["template_args"]

    # Preproc node substitutions
    preprocsinksubs = util.get_mapnode_substitutions(preproc, exp["nruns"])

    # Preproc Datasink nodes
    preprocsink = pe.Node(nio.DataSink(base_directory=project["analysis_dir"],
                                       substitutions=preprocsinksubs),
                          name="preprocsink")

    # Preproc connections
    preproc.connect(subjectsource, "subject_id", preprocsource, "subject_id")
    preproc.connect(subjectsource, "subject_id", preproc_input, "subject_id")

    # Input connections
    util.connect_inputs(preproc, preprocsource, preproc_input)

    # Set up the subject containers
    util.subject_container(preproc, subjectsource, preprocsink)

    # Connect the heuristic outputs to the datainks
    util.sink_outputs(preproc, preproc_output, preprocsink, "preproc")

    # Set up the working output
    preproc.base_dir = project["working_dir"]

    # Archive crashdumps
    util.archive_crashdumps(preproc)

    run_workflow(preproc, "preproc", args)

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
        [(k,v) for k,v in exp.__dict__.items() if not re.match("__.*__", k)])

def run_workflow(wf, name, args):

    plugin, plugin_args = util.determine_engine(args)
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
