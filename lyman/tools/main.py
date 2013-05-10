import os
import re
import sys
import imp
import os.path as op

import numpy as np
import networkx as nx

import nipype
from nipype.pipeline.engine import Workflow, MapNode, Node
from nipype.interfaces.base import isdefined
from nipype.interfaces.utility import IdentityInterface


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
    """Import project information based on environment settings."""
    proj_file = op.join(os.environ["LYMAN_DIR"], "project.py")
    try:
        project = sys.modules["project"]
    except KeyError:
        project = imp.load_source("project", proj_file)
    return dict(
        [(k, v) for k, v in project.__dict__.items()
            if not re.match("__.*__", k)])


def gather_experiment_info(experiment_name, altmodel=None):
    """Import an experiment module and add some formatted information."""
    if altmodel is None:
        module_name = experiment_name
    else:
        module_name = "-".join([experiment_name, altmodel])
    exp_file = op.join(os.environ["LYMAN_DIR"], module_name + ".py")
    try:
        exp = sys.modules[module_name]
    except KeyError:
        exp = imp.load_source(module_name, exp_file)

    # Create an experiment dict stripping the OOP hooks
    exp_dict = dict(
        [(k, v) for k, v in exp.__dict__.items() if not re.match("__.*__", k)])

    # Verify some experiment dict attributes
    verify_experiment_info(exp_dict)

    # Save the __doc__ attribute to the dict
    exp_dict["comments"] = exp.__doc__

    # Check if it looks like this is a partial FOV acquisition
    parfov = True if "full_fov_epi" in exp_dict else False
    exp_dict["partial_fov"] = parfov

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
    """Given list of names or file with list of names, return the list."""
    if op.isfile(subject_arg[0]):
        return np.loadtxt(subject_arg[0], str).tolist()
    return subject_arg


def determine_engine(args):
    """Read command line args and return Workflow.run() args."""
    plugin_dict = dict(linear="Linear", multiproc="MultiProc",
                       ipython="IPython", torque="PBS", sge="SGE")

    plugin = plugin_dict[args.plugin]

    plugin_args = dict()
    if plugin == "MultiProc":
        plugin_args['n_procs'] = args.nprocs
    elif plugin == "PBS":
        plugin_args["qsub_args"] = "-V -e /dev/null -o /dev/null"

    return plugin, plugin_args


def make_subject_source(subject_list):
    """Generate a source node with iterables over a subject_id list."""
    return Node(IdentityInterface(fields=["subject_id"]),
                iterables=("subject_id", subject_list),
                overwrite=True,
                name="subj_source")


def crashdump_config(wf, dump_dir):
    """Configure workflow to dump crashfiles somewhere."""
    version = nipype.__version__
    if version > "0.4.1":
        wf.config["execution"]["crashdump_dir"] = dump_dir
    else:
        wf.config = dict(crashdump_dir=dump_dir)


def run_workflow(wf, name=None, args=None):
    """Run a workflow, if we asked to do so on the command line."""
    plugin, plugin_args = determine_engine(args)
    if name is None or name in args.workflows:
        wf.run(plugin, plugin_args)


def find_contrast_number(contrast_name, contrast_names):
    """Find index in contrast list for given contrast name.

    Contains a hack to handle mask registration.

    """
    if contrast_name == "_mask":
        return 0
    return contrast_names.index(contrast_name) + 1


def reg_template(contrast, mask_template, model_template):
    """Implement a hack to grab mask files for registration."""
    return mask_template if contrast == "_mask" else model_template


def reg_template_args(contrast, mask_args, model_args):
    """Implement a hack to grab mask files for registration."""
    return mask_args if contrast == "_mask" else model_args


def write_workflow_report(workflow_name, report_template, report_dict):
    """Generic function to take write .rst files and convert to pdf/html.

    Accepts a report template and dictionary. Writes rst once with
    full paths for image files and generates a pdf, then strips
    leading path components and writes again, generating an html
    file that exepects to live in the same directory as report images.

    """
    from os.path import exists, basename
    from subprocess import check_output

    # Plug the values into the template for the pdf file
    report_rst_text = report_template % report_dict

    # Write the rst file and convert to pdf
    report_pdf_rst_file = "%s_pdf.rst" % workflow_name
    report_pdf_file = op.abspath("%s_report.pdf" % workflow_name)
    open(report_pdf_rst_file, "w").write(report_rst_text)
    check_output(["rst2pdf", report_pdf_rst_file, "-o", report_pdf_file])
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
    check_output(["rst2html.py", report_html_rst_file, report_html_file])
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
    from lyman.tools import vox_to_mni, locate_peaks

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
    """Find most probable region in HarvardOxford Atlas of a vox coord."""
    from os import environ
    import os.path as op
    import numpy as np
    from nibabel import load
    from lyman.tools.main import (harvard_oxford_sub_names,
                              harvard_oxford_ctx_names)
    sub_names = harvard_oxford_sub_names
    ctx_names = harvard_oxford_ctx_names
    at_dir = op.join(environ["FSLDIR"], "data", "atlases")
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
            loc_list.append((sub_names[sub_index], sub_prob))
            continue
        if sub_prob > ctx_prob and sub_index not in [0, 1, 11, 12]:
            loc_list.append((sub_names[sub_index], sub_prob))
            continue
        loc_list.append((ctx_names[ctx_index], ctx_prob))

    return loc_list


def shorten_name(region_name, atlas):
    """Implement regexp sub for verbose Harvard Oxford Atlas region."""
    import re
    from . import (harvard_oxford_ctx_subs,
                   harvard_oxford_sub_subs)
    sub_list = dict(ctx=harvard_oxford_ctx_subs,
                    sub=harvard_oxford_sub_subs)
    for pat, rep in sub_list[atlas]:
        region_name = re.sub(pat, rep, region_name).strip()
    return region_name


def vox_to_mni(vox_coords):
    """Given ijk voxel coordinates, return xyz from image affine.

    The _to_mni part is rather a misnomer, although this at the moment
    only gets used in the group volume workflows.

    """
    import numpy as np
    from nibabel import load
    from nipype.interfaces.fsl import Info

    mni_file = Info.standard_image("avg152T1.nii.gz")
    aff = load(mni_file).get_affine()
    mni_coords = np.zeros_like(vox_coords)
    for i, coord in enumerate(vox_coords):
        coord = coord.astype(float)
        mni_coords[i] = np.dot(aff, np.r_[coord, 1])[:3].astype(int)
    return mni_coords

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

harvard_oxford_sub_names = [
    'L Cereb WM',
    'L Ctx',
    'L LatVent',
    'L Thalamus',
    'L Caudate',
    'L Putamen',
    'L Pallidum',
    'Brain-Stem',
    'L Hippocampus',
    'L Amygdala',
    'L Accumbens',
    'R Cereb WM',
    'R Ctx',
    'R LatVent',
    'R Thalamus',
    'R Caudate',
    'R Putamen',
    'R Pallidum',
    'R Hippocampus',
    'R Amygdala',
    'R Accumbens']

harvard_oxford_ctx_names = [
    'Front Pole',
    'Insular Ctx',
    'SFG',
    'MFG',
    'IFG, triang',
    'IFG, oper',
    'Precentral G',
    'Temp Pole',
    'STG, ant',
    'STG, post',
    'MTG, ant',
    'MTG, post',
    'MTG, tempocc',
    'ITG, ant',
    'ITG, post',
    'ITG, tempocc',
    'Postcentral G',
    'Sup Par Lobule',
    'Supramarg G, ant',
    'Supramarg G, post',
    'Angular G',
    'Lat Occ Ctx, sup',
    'Lat Occ Ctx, inf',
    'Intracalc Ctx',
    'Front Med Ctx',
    'Juxt Lobule Ctx',
    'Subcallosal Ctx',
    'Paracing G',
    'Cing G, ant',
    'Cing G, post',
    'Precuneous Ctx',
    'Cuneal Ctx',
    'Front Orb Ctx',
    'Parahip G, ant',
    'Parahip G, post',
    'Lingual G',
    'Temp Fus Ctx, ant',
    'Temp Fus Ctx, post',
    'Temp Occ Fus Ctx',
    'Occ Fus G',
    'Front Oper Ctx',
    'Central Oper Ctx',
    'Par Oper Ctx',
    'Planum Polare',
    'Heschl"s G',
    'Planum Tempe',
    'Supracalc Ctx',
    'Occ Pole']
