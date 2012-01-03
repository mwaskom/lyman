import os.path as op
import sys
sys.path.insert(0, op.abspath(op.pardir))
import run_fmri


def test_run_fmri():
    """Not a great test, but should catch some dumb things."""
    try:
        run_fmri.main([])
    except TypeError:
        pass
