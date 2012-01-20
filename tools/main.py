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
        if isinstance(v, str) and v.endswith(".png"):
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


def cluster_to_rst(localmax_file):
    """Convert the localmax text file from Cluster to RST formatting.

    Also convert the voxel coordinates to MNI coordinates and add in
    the most likely location and probability from Harvard Oxford atlas.

    """
    from os.path import abspath
    import numpy as np
    from tools import vox_to_mni, locate_peaks

    out_file = abspath("localmax_table.txt")
    # Localmax files seem to be badly formed
    # So geting them into array form is pretty annoying
    with open(localmax_file) as f:
        clust_head = f.readline().split()
        _ = clust_head.pop(1)
    clust_a = np.loadtxt(localmax_file, str,
                        delimiter="\t", skiprows=1)
    if not clust_a.size:
        with open(out_file, "w") as f:
            f.write("")
        return out_file
    clust_a = np.vstack((clust_head, clust_a))
    index_v = np.atleast_2d(np.array(
        ["Peak"] + range(1, clust_a.shape[0]))).T
    clust_a = np.hstack((index_v, clust_a))

    # Find out where the peaks most likely are
    loc_a = np.array(locate_peaks(clust_a[1:, 3:6].astype(int)))
    peak_a = np.hstack((clust_a, loc_a))

    # Convert the voxel coordinates to MNI
    vox_coords = peak_a[1:, 3:6]
    peak_a[1:, 3:6] = vox_to_mni(vox_coords)

    # Insert the column-defining dash rows
    len_a = np.array([[len(c) for c in r] for r in peak_a])
    max_lens = len_a.max(axis=0)
    hyphen_v = ["".join(["=" for i in range(l)]) for l in max_lens]
    peak_l = peak_a.tolist()
    for pos in [0, 2]:
        peak_l.insert(pos, hyphen_v)
    peak_l.append(hyphen_v)
    peak_a = np.array(peak_l)

    # Write the rows out to a text file with padding
    len_a = np.array([[len(c) for c in r] for r in peak_a])
    len_diff = max_lens - len_a
    pad_a = np.array(
        [["".join([" " for i in range(l)]) for l in row] for row in len_diff])
    with open(out_file, "w") as f:
        for i, row in enumerate(peak_a):
            for j, word in enumerate(row):
                f.write("%s%s " % (word, pad_a[i, j]))
            f.write("\n")

    return out_file


def locate_peaks(vox_coords):
    from os import environ
    import os.path as op
    import numpy as np
    from lxml import etree
    from nibabel import load
    from tools import shorten_name
    at_dir = op.join(environ["FSLDIR"], "data", "atlases")
    ctx_xml = op.join(at_dir, "HarvardOxford-Cortical.xml")
    ctx_labels = etree.parse(ctx_xml).find("data").findall("label")
    sub_xml = op.join(at_dir, "HarvardOxford-SubCortical.xml")
    sub_labels = etree.parse(sub_xml).find("data").findall("label")
    ctx_data = load(op.join(at_dir, "HarvardOxford",
                            "HarvardOxford-cort-prob-2mm.nii.gz")).get_data()
    sub_data = load(op.join(at_dir, "HarvardOxford",
                            "HarvardOxford-sub-prob-2mm.nii.gz")).get_data()

    loc_list = [("MaxProb Region", "Prob")]
    for coord in vox_coords:
        coord = tuple(coord)
        ctx_index = np.argmax(ctx_data[coord])
        ctx_prob = ctx_data[coord][ctx_index]
        sub_index = np.argmax(sub_data[coord])
        sub_prob = sub_data[coord][sub_index]

        if not max(sub_prob, ctx_prob):
            loc_list.append(("Unknown", 0))
            continue
        if not ctx_prob and sub_index in [0, 11]:
            loc_list.append(
                (shorten_name(sub_labels[sub_index].text, "sub"), sub_prob))
            continue
        if sub_prob > ctx_prob and sub_index not in [0, 1, 11, 12]:
            loc_list.append(
                (shorten_name(sub_labels[sub_index].text, "sub"), sub_prob))
            continue
        loc_list.append(
            (shorten_name(ctx_labels[ctx_index].text, "ctx"), ctx_prob))

    return loc_list


def shorten_name(region_name, atlas):
    import re
    from . import (harvard_oxford_ctx_subs,
                   harvard_oxford_sub_subs)
    sub_list = dict(ctx=harvard_oxford_ctx_subs,
                    sub=harvard_oxford_sub_subs)
    for pat, rep in sub_list[atlas]:
        region_name = re.sub(pat, rep, region_name)
    return region_name


def vox_to_mni(vox_coords):
    import numpy as np
    from nibabel import load
    from nipype.interfaces.fsl import Info

    mni_file = Info.standard_image("avg152T1.nii.gz")
    aff = load(mni_file).get_affine()
    for i, coord in enumerate(vox_coords):
        coord = coord.astype(float)
        vox_coords[i] = np.dot(aff, np.r_[coord, 1])[:3].astype(int)
    return vox_coords

harvard_oxford_sub_subs = [
    ("Left", "L"),
    ("Right", "R"),
    ("Cerebral Cortex", "Ctx"),
    ("Cerebral White Matter", "Cereb WM"),
    ("Lateral Ventrica*le*", "LatVent"),
]

harvard_oxford_ctx_subs = [
    ("Superior", "Sup"),
    ("Middle", "Mid"),
    ("Inferior", "Inf"),
    ("Lateral", "Lat"),
    ("Medial", "Med"),
    ("Frontal", "Front"),
    ("Parietal", "Par"),
    ("Temporal", "Temp"),
    ("Occipital", "Occ"),
    ("Cingulate", "Cing"),
    ("Cortex", "Ctx"),
    ("Gyrus", "G"),
    ("Sup Front G", "SFG"),
    ("Mid Front G", "MFG"),
    ("Inf Front G", "IFG"),
    ("Sup Temp G", "STG"),
    ("Mid Temp G", "MTG"),
    ("Inf Temp G", "ITG"),
    ("Parahippocampal", "Parahip"),
    ("Juxtapositional", "Juxt"),
    ("Intracalcarine", "Intracalc"),
    ("Supramarginal", "Supramarg"),
    ("Supracalcarine", "Supracalc"),
    ("Paracingulate", "Paracing"),
    ("Fusiform", "Fus"),
    ("Orbital", "Orb"),
    ("Opercul[ua][mr]", "Oper"),
    ("temporooccipital", "tempocc"),
    ("triangularis", "triang"),
    ("opercularis", "oper"),
    ("division", ""),
    ("par[st] *", ""),
    ("anterior", "ant"),
    ("posterior", "post"),
    ("superior", "sup"),
    ("inferior", "inf"),
    (" +", " "),
    ("\(.+\)", ""),
]
