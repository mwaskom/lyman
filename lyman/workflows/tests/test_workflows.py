"""Very basic tests the workflow factories are importable and callable."""
import re
from nipype import Workflow, Node, IdentityInterface

from lyman import workflows as wf


def test_workflow_functions():

    pat = re.compile("create_(.+)_workflow")
    funcs = [getattr(wf, k) for k in dir(wf) if pat.match(k)]

    for f in funcs:

        def case():
            ret = f()

            try:
                w, i, o = ret
            except TypeError:
                assert isinstance(ret, Workflow)
                return 
                
            assert isinstance(w, Workflow)
            assert isinstance(i, Node)
            assert isinstance(o, Node)
            assert isinstance(i.interface, IdentityInterface)
            assert isinstance(o.interface, IdentityInterface)

        name = pat.match(f.__name__).group(1)
        case.description = "Test %s workflow function" % name
        yield case
