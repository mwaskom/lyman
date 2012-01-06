import os
import re
import sys
import os.path as op
from datetime import datetime

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


def find_contrast_number(contrast_name, contrast_names):

    if contrast_name == "_mask":
        return 0
    return contrast_names.index(contrast_name) + 1


def reg_template(contrast, mask_template, model_template):

    return mask_template if contrast == "_mask" else model_template


def reg_template_args(contrast, mask_args, model_args):

    return mask_args if contrast == "_mask" else model_args
