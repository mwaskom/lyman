"""Save graphical representations of all the lyman workflows."""
import os
import re
import sys
from glob import glob
from lyman import workflows as wf
from nipype import config


def main(arglist):

    config.set('logging', 'workflow_level', 'CRITICAL')

    # Find the functions that create workflows
    wf_funcs = [k for k in dir(wf) if re.match("create_.*_workflow", k)]

    for func in wf_funcs:
        try:
            out = getattr(wf, func)()
        except:
            print "ERROR: call to %s failed" % func

        # Some of the workflow functions return (flow, inputs, outputs)
        try:
            flow, _, _ = out
        except TypeError:
            flow = out

        # Write the graphs
        name = flow.name
        if arglist:
            if name in arglist:
                flow.write_graph("graphs/%s.dot" % name, "orig")
        else:
            flow.write_graph("graphs/%s.dot" % name, "orig")

    # Remove the .dot files as they are not of use to us
    files = glob("graphs/*")
    for f in files:
        if f.endswith(".dot"):
            os.remove(f)


if __name__ == "__main__":
    main(sys.argv[1:])
