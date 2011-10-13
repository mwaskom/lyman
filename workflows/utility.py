import nipype.pipeline.engine as pe    
import nipype.interfaces.utility as util


class OutputConnector(object):

    def __init__(self, workflow, outnode):
        self.workflow = workflow
        self.outnode = outnode

    def connect(self, procnode, outfield, procfield=None, name=None):
        if procfield is None:
            procfield = "out_file"
        if name is None:
            name = outfield
        renamenode = pe.MapNode(util.Rename(format_string=name, keep_ext=True),
                                iterfield=["in_file"],
                                name="rename_%s"%outfield)
        self.workflow.connect([
            (procnode,   renamenode,      [(procfield, "in_file")]),
            (renamenode, self.outnode,    [("out_file", outfield)])
            ])


