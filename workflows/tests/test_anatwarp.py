from nipype.testing import assert_equal, assert_true
from nipype.pipeline.engine import Workflow

from .. import anatwarp


def test_anatwap():

    data_dir = "/tmp"
    subjects = ["s1", "s2"]
    name = "normalize"

    wf = anatwarp.create_anatwarp_workflow(data_dir, subjects, name)

    yield assert_true, isinstance(wf, Workflow)
    yield assert_equal, wf.name, name

    subj_source = wf.get_node("subjectsource")
    yield assert_equal, subj_source.iterables, ("subject_id", subjects)

    data_grabber = wf.get_node("datagrabber")
    yield assert_equal, data_grabber.inputs.base_directory, data_dir
