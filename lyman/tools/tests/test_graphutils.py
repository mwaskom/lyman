from nipype.testing import assert_equal, assert_true

from nipype import Workflow, Node, MapNode, IdentityInterface, DataGrabber

from .. import graphutils as gu


def make_simple_workflow():

    wf = Workflow(name="test")

    node1 = Node(IdentityInterface(fields=["foo"]), name="node1")
    node2 = MapNode(IdentityInterface(fields=["foo"]),
                    name="node2", iterfield=["foo"])
    node3 = Node(IdentityInterface(fields=["foo"]), name="node3")

    wf.connect([
        (node1, node2, [("foo", "foo")]),
        (node2, node3, [("foo", "foo")]),
        ])

    return wf, node1, node2, node3


def test_input_wrapper():

    wf, node1, node2, node3 = make_simple_workflow()

    s_list = ['s1', 's2']
    s_node = gu.make_subject_source(s_list)

    g_node = Node(DataGrabber(in_fields=["foo"],
                              out_fields=["bar"]),
                  name="g_node")

    iw = gu.InputWrapper(wf, s_node, g_node, node1)

    yield assert_equal, iw.wf, wf
    yield assert_equal, iw.subj_node, s_node
    yield assert_equal, iw.grab_node, g_node
    yield assert_equal, iw.in_node, node1

    iw.connect_inputs()

    g = wf._graph
    yield assert_true, s_node in g.nodes()
    yield assert_true, g_node in g.nodes()
    yield assert_true, (s_node, g_node) in g.edges()


def test_find_mapnodes():

    wf = make_simple_workflow()[0]
    mapnodes = gu.find_mapnodes(wf)
    yield assert_equal, mapnodes, ["node2"]


def test_find_nested_workflows():

    wf, node1, node2, node3 = make_simple_workflow()
    inner_wf = make_simple_workflow()[0]

    wf.connect(node3, "foo", inner_wf, "node1.foo"),

    workflows = gu.find_nested_workflows(wf)

    yield assert_equal, workflows, [inner_wf]


def test_make_subject_source():

    subj_list = ['s1', 's2', 's3']
    node = gu.make_subject_source(subj_list)
    iterable_name, iterable_val = node.iterables
    yield assert_equal, iterable_name, "subject_id"
    yield assert_equal, iterable_val, subj_list
