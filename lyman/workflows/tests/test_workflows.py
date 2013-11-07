"""Very basic tests that workflow factories are importable and callable."""
import re
from nose.tools import assert_is_instance
from nipype import Workflow, Node, IdentityInterface

import warnings
warnings.simplefilter("ignore", UserWarning)

from lyman import workflows as wf


def test_workflow_functions():
    """Written as generator for clarity about which tests are failing."""
    pat = re.compile("create_(.+)_workflow")
    funcs = [getattr(wf, k) for k in dir(wf) if pat.match(k)]

    for f in funcs:

        def case():
            ret = f()

            try:
                w, i, o = ret
            except TypeError:
                assert_is_instance(ret, Workflow)
                return

            assert_is_instance(w, Workflow)
            assert_is_instance(i, Node)
            assert_is_instance(o, Node)
            assert_is_instance(i.interface, IdentityInterface)
            assert_is_instance(o.interface, IdentityInterface)

        name = pat.match(f.__name__).group(1)
        case.description = "Test %s workflow function" % name
        yield case
