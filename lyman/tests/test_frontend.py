from argparse import Namespace
from nose.tools import assert_equal

from .. import frontend


def test_determine_engine():

    plugin_dict = dict(linear="Linear",
                       multiproc="MultiProc",
                       ipython="IPython",
                       torque="PBS")

    for arg, plugin_str in plugin_dict.items():
        args = Namespace(plugin=arg, queue=None)
        if arg == "multiproc":
            args.nprocs = 4
        plugin, plugin_args = frontend.determine_engine(args)
        yield assert_equal, plugin, plugin_str

        if arg == "multiproc":
            yield assert_equal, plugin_args, dict(n_procs=4, qsub_args="")
