import re
import sys
import os.path as op

import numpy as np
import networkx as nx

from nipype.pipeline.engine import Workflow, MapNode, Node
from nipype.interfaces.base import isdefined
from nipype.interfaces.utility import IdentityInterface


class InputWrapper(object):

    def __init__(self, workflow, subject_node, grabber_node, input_node):

        self.wf = workflow
        self.subj_node = subject_node
        self.grab_node = grabber_node
        self.in_node = input_node

    def connect_inputs(self):
        """Connect stereotyped inputs to the input IdentityInterface."""

        # Connect subject_id to input and grabber nodes
        self.wf.connect(self.subj_node, "subject_id",
                        self.grab_node, "subject_id")
        if hasattr(self.in_node.inputs, "subject_id"):
            self.wf.connect(self.subj_node, "subject_id",
                            self.in_node, "subject_id")

        # Connect the datagrabber outputs to the workflow inputs
        grabbed = self.grab_node.outputs.get()
        inputs = self.in_node.inputs.get()
        for field in grabbed:
            if field in inputs:
                self.wf.connect(self.grab_node, field,
                                self.in_node, field)


class OutputWrapper(object):

    def __init__(self, workflow, subject_node, sink_node, output_node):

        self.wf = workflow
        self.subj_node = subject_node
        self.sink_node = sink_node
        self.out_node = output_node

    def set_mapnode_substitutions(self, n_runs):
        """Find mapnode names and add datasink substitutions to sort by run."""

        # First determine top-level mapnode names
        mapnode_names = find_mapnodes(self.wf)

        # Then determine mapnode names for each nested workflow
        # Note that this currently only works for one level of nesting
        nested_workflows = find_nested_workflows(self.wf)
        for wf in nested_workflows:
            mapnode_names.extend(find_mapnodes(wf))

        # Build a list of substitution tuples
        substitutions = []
        for r in reversed(range(n_runs)):
            for name in mapnode_names:
                substitutions.append(("_%s%d" % (name, r), "run_%d" % (r + 1)))

        # Set the substitutions attribute on the DataSink node
        if isdefined(self.sink_node.inputs.substitutions):
            self.sink_node.inputs.substitutions.extend(substitutions)
        else:
            self.sink_node.inputs.substitutions = substitutions

    def add_regexp_substitutions(self, sub_list):

        if isdefined(self.sink_node.inputs.regexp_substitutions):
            self.sink_node.inputs.regexp_substitutions.extend(sub_list)
        else:
            self.sink_node.inputs.regexp_substitutions = sub_list

    def set_subject_container(self):

        # Connect the subject_id value as the container
        self.wf.connect(self.subj_node, "subject_id",
                        self.sink_node, "container")

        subj_subs = []
        for s in self.subj_node.iterables[1]:
            subj_subs.append(("/_subject_id_%s" % s, "/"))

        # Strip the subject_id iterable from the path
        if isdefined(self.sink_node.inputs.substitutions):
            self.sink_node.inputs.substitutions.extend(subj_subs)
        else:
            self.sink_node.inputs.substitutions = subj_subs

    def sink_outputs(self, dir_name):
        """Connect the outputs of a workflow to a datasink."""

        outputs = self.out_node.outputs.get()
        for field in outputs:
            self.wf.connect(self.out_node, field,
                            self.sink_node, dir_name + ".@" + field)


def find_mapnodes(workflow):
    """Given a workflow, return a list of MapNode names."""

    mapnode_names = []
    wf_nodes = nx.nodes(workflow._graph)
    for node in wf_nodes:
        if isinstance(node, MapNode):
            mapnode_names.append(node.name)

    return mapnode_names


def find_nested_workflows(workflow):
    """Given a workflow, find nested workflow objects."""

    nested_workflows = []
    wf_nodes = nx.nodes(workflow._graph)
    for node in wf_nodes:
        if isinstance(node, Workflow):
            nested_workflows.append(node)

    return nested_workflows


def gather_project_info():

    # This seems safer than just catching an import error, since maybe
    # someone will copy another set of scripts and just delete the
    # project.py without knowing anything about .pyc files
    if op.exists("project.py"):
        import project
        return dict(
            [(k, v) for k, v in project.__dict__.items()
                if not re.match("__.*__", k)])

    print "ERROR: Did not find a project.py file in this directory."
    print "You must run setup_project.py before using the analysis scripts."
    sys.exit()

def gather_experiment_info(experiment_name, altmodel=None):
    """Import an experiment module and add some formatted information."""
    module_name = experiment_name
    if altmodel is not None:
        module_name = "-".join([experiment_name, altmodel])
    try:
        exp = __import__("experiments." + module_name,
                         fromlist=["experiments"])
    except ImportError:
        print "ERROR: Could not import experiments/%s.py" % module_name
        sys.exit()

    # Create an experiment dict stripping the OOP hooks
    exp_dict = dict(
        [(k, v) for k, v in exp.__dict__.items() if not re.match("__.*__", k)])

    # Verify some experiment dict attributes
    verify_experiment_info(exp_dict)

    # Save the __doc__ attribute to the dict
    exp_dict["comments"] = exp.__doc__

    # Convert HPF cutoff to sigma for fslmaths
    exp_dict["TR"] = float(exp_dict["TR"])
    exp_dict["hpf_cutoff"] = float(exp_dict["hpf_cutoff"])
    exp_dict["hpf_sigma"] = (exp_dict["hpf_cutoff"] / 2.35) / exp_dict["TR"]

    # Setup the hrf_bases dictionary
    exp_dict["hrf_bases"] = {exp_dict["hrf_model"]:
                                {"derivs": exp_dict["hrf_derivs"]}}

    # Build contrasts list if neccesary
    if "contrasts" not in exp_dict:
        conkeys = sorted([k for k in exp_dict if re.match("cont\d+", k)])
        exp_dict["contrasts"] = [exp_dict[key] for key in conkeys]
    exp_dict["contrast_names"] = [c[0] for c in exp_dict["contrasts"]]

    if "regressors" not in exp_dict:
        exp_dict["regressors"] = []

    return exp_dict


def verify_experiment_info(exp_dict):
    """Catch setup errors that might lead to confusing workflow crashes."""
    if exp_dict["units"] not in ["secs", "scans"]:
        raise ValueError("units must be 'secs' or 'scans'")

    if (exp_dict["slice_time_correction"]
        and exp_dict["slice_order"] not in ["up", "down"]):
        raise ValueError("slice_order must be 'up' or 'down'")


def determine_subjects(subject_arg):

    if op.isfile(subject_arg[0]):
        return np.loadtxt(subject_arg[0], str).tolist()
    return subject_arg


def determine_engine(args):

    plugin_dict = dict(linear="Linear", multiproc="MultiProc",
                       ipython="IPython", torque="PBS")

    plugin = plugin_dict[args.plugin]

    plugin_args = dict()
    if plugin == "MultiProc":
        plugin_args['n_procs'] = args.nprocs

    return plugin, plugin_args


def make_subject_source(subject_list):

    return Node(IdentityInterface(fields=["subject_id"]),
                iterables=("subject_id", subject_list),
                overwrite=True,
                name="subj_source")


def run_workflow(wf, name=None, args=None):
    """Run a workflow, if we asked to do so on the command line."""
    plugin, plugin_args = determine_engine(args)
    if name is None or name in args.workflows:
        wf.run(plugin, plugin_args)


def find_contrast_number(contrast_name, contrast_names):

    if contrast_name == "_mask":
        return 0
    return contrast_names.index(contrast_name) + 1


def reg_template(contrast, mask_template, model_template):

    return mask_template if contrast == "_mask" else model_template


def reg_template_args(contrast, mask_args, model_args):

    return mask_args if contrast == "_mask" else model_args


def write_workflow_report(workflow_name, report_template, report_dict):
    from os.path import exists, basename
    from subprocess import call

    # Plug the values into the template for the pdf file
    report_rst_text = report_template % report_dict

    # Write the rst file and convert to pdf
    report_pdf_rst_file = "%s_pdf.rst" % workflow_name
    report_pdf_file = op.abspath("%s_report.pdf" % workflow_name)
    open(report_pdf_rst_file, "w").write(report_rst_text)
    call(["rst2pdf", report_pdf_rst_file, "-o", report_pdf_file])
    if not exists(report_pdf_file):
        raise RuntimeError

    # For images going into the html report, we want the path to be relative
    # (We expect to read the html page from within the datasink directory
    # containing the images.  So iteratate through and chop off leading path.
    for k, v in report_dict.items():
        if v.endswith(".png"):
            report_dict[k] = basename(v)

    # Write the another rst file and convert it to html
    report_html_rst_file = "%s_html.rst" % workflow_name
    report_html_file = op.abspath("%s_report.html" % workflow_name)
    report_rst_text = report_template % report_dict
    open(report_html_rst_file, "w").write(report_rst_text)
    call(["rst2html.py", report_html_rst_file, report_html_file])
    if not exists(report_html_file):
        raise RuntimeError

    # Return both report files as a list
    return [report_pdf_file, report_html_file]
