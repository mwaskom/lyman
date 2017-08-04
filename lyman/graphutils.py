import os.path as op
import subprocess as sp
import networkx as nx
from nipype import Workflow, MapNode, Node, IdentityInterface
from nipype.interfaces.base import (BaseInterface, BaseInterfaceInputSpec,
                                    TraitedSpec, File, OutputMultiPath,
                                    isdefined)


class InputWrapper(object):
    """Implements connections between DataGrabber and workflow inputs."""
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
    """Implements connections between workflow outputs and DataSink."""
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
        """Safely set subsitutions implemented with regular expressions."""
        if isdefined(self.sink_node.inputs.regexp_substitutions):
            self.sink_node.inputs.regexp_substitutions.extend(sub_list)
        else:
            self.sink_node.inputs.regexp_substitutions = sub_list

    def set_subject_container(self):
        """Store results by subject at highest level."""
        # Connect the subject_id value as the container
        self.wf.connect(self.subj_node, "subject_id",
                        self.sink_node, "container")

        subj_subs = []
        for s in self.subj_node.iterables[1]:
            subj_subs.append(("/_subject_id_%s/" % s, "/"))

        # Strip the subject_id iterable from the path
        if isdefined(self.sink_node.inputs.substitutions):
            self.sink_node.inputs.substitutions.extend(subj_subs)
        else:
            self.sink_node.inputs.substitutions = subj_subs

    def sink_outputs(self, dir_name=None):
        """Connect the outputs of a workflow to a datasink."""
        outputs = self.out_node.outputs.get()
        prefix = "@" if dir_name is None else dir_name + ".@"
        for field in outputs:
            self.wf.connect(self.out_node, field,
                            self.sink_node, prefix + field)


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


def make_subject_source(subject_list):
    """Generate a source node with iterables over a subject_id list."""
    return Node(IdentityInterface(fields=["subject_id"]),
                iterables=("subject_id", subject_list),
                overwrite=True,
                name="subj_source")


class SingleInFile(TraitedSpec):

    in_file = File(exists=True)


class SingleOutFile(TraitedSpec):

    out_file = File(exists=True)


class ManyOutFiles(TraitedSpec):

    out_files = OutputMultiPath(File(exists=True))


def list_out_file(fname):
    """Return a _list_outputs function for a single out_file."""
    def _list_outputs(self):

        outputs = self._outputs().get()
        outputs["out_file"] = op.abspath(fname)
        return outputs

    return _list_outputs


class SimpleInterface(BaseInterface):

    def __init__(self, **inputs):

        if isinstance(self.input_spec, dict):
            self.input_spec = type("AnonymousInputSpec",
                                   (BaseInterfaceInputSpec,),
                                   self.input_spec)

        if isinstance(self.output_spec, dict):
            self.output_spec = type("AnonymousOutputSpec",
                                    (TraitedSpec,),
                                    self.output_spec)

        super(SimpleInterface, self).__init__(**inputs)

        self._results = {}

    def _list_outputs(self):

        return self._results

    def submit_cmdline(self, runtime, cmdline, **results):
        """Submit a command-line job and capture the output."""

        for attr in ["stdout", "stderr", "cmdline"]:
            if not hasattr(runtime, attr):
                setattr(runtime, attr, "")
        if not hasattr(runtime, "returncode"):
            runtime.returncode = 0
        elif runtime.returncode is None:
            runtime.returncode = 0

        if isinstance(cmdline, list):
            cmdline = " ".join(cmdline)

        proc = sp.Popen(cmdline,
                        stdout=sp.PIPE,
                        stderr=sp.PIPE,
                        shell=True,
                        cwd=runtime.cwd,
                        env=runtime.environ,
                        universal_newlines=True)

        stdout, stderr = proc.communicate()

        runtime.stdout += "\n" + stdout + "\n"
        runtime.stderr += "\n" + stderr + "\n"
        runtime.cmdline += "\n" + cmdline + "\n"
        runtime.returncode += proc.returncode

        if proc.returncode is None or proc.returncode != 0:
            message = "\n\nCommand:\n" + runtime.cmdline + "\n"
            message += "Standard output:\n" + runtime.stdout + "\n"
            message += "Standard error:\n" + runtime.stderr + "\n"
            message += "Return code: " + str(runtime.returncode)
            raise RuntimeError(message)

        for field, fname in results.items():
            self._results[field] = fname

        return runtime
