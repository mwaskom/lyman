import os.path as op
import sys
sys.path.insert(0, op.abspath(op.pardir))
import run_warp


def test_run_warp():
    """Not a great test, but should catch dumb things."""
    try:
        run_warp.main([])
    except TypeError:
        pass
