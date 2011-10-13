#! /usr/bin/env python
import os
import re
import sys
import shutil
import argparse
import inspect
import os.path as op
from tempfile import mkdtemp
"""
import matplotlib as mpl
mpl.use("Agg")

import nipype.pipeline.engine as pe

import nipype.interfaces.io as nio
from nipype.interfaces import fsl
import nipype.interfaces.utility as util

from workflows.preproc import get_preproc_workflow
from workflows.fsl_model import get_model_workflow
from workflows.registration import get_registration_workflow
from workflows.fsl_fixed_fx import get_fsl_fixed_fx_workflow
from workflows.freesurfer_fixed_fx import get_freesurfer_fixed_fx_workflow
"""

from util.commandline import parser

def main(argslist):

    pass

def gather_project_info():

    # This seems safer than just catching an import error, since maybe
    # someone will copy another set of scripts and just delete the 
    # project.py without knowing anything about .pyc files
    if op.exists("project.py"):
        import project
        return dict(
            [(k,v) for k,v in project.__dict__.items() if not re.match("__.*__", k)])

    print "ERROR: Did not find a project.py file in this directory."
    print "You must run setup_project.py before using the analysis scripts."
    sys.exit()

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

def parse_args(arglist):

    parser.add_argument("-experiment", help="experimental paradigm")
    parser.add_argument("-altmodel", help="alternate model to estimate")
    parser.add_argument("-workflows", 
                        choices=["all", "preproc", "model", "reg", "ffx"],
                        help="which workflos to run")
    parser.add_argument("-surf", action="store_true",
                        help="run processing for surface analysis")
    parser.add_argument("-native", action="store_true",
                        help="run fixed effect analysis on native surface")
    return parser.parse_args(arglist)

if __name__ == "__main__":
    main(sys.argv[1:])
