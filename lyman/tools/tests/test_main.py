import numpy as np
import pandas as pd
from argparse import Namespace
from nipype.testing import assert_equal

from .. import main


def test_determine_engine():

    plugin_dict = dict(linear="Linear",
                       multiproc="MultiProc",
                       ipython="IPython",
                       torque="PBS")

    for arg, plugin_str in plugin_dict.items():
        args = Namespace(plugin=arg, queue=None)
        if arg == "multiproc":
            args.nprocs = 4
        plugin, plugin_args = main.determine_engine(args)
        yield assert_equal, plugin, plugin_str

        if arg == "multiproc":
            yield assert_equal, plugin_args, dict(n_procs=4, qsub_args="")


def test_locate_peaks():

    challenge = [
        ([(60, 60, 50)], ("L Cereb WM", 95)),
        ([(62, 69, 50)], ("MFG", 20)),
        ([(31, 50, 27)], ("Parahip G, post", 30)),
        ([(26, 55, 27)], ("Temp Fus Ctx, post", 3)),
        ([(29, 50, 30)], ("R Hippocampus", 95))]

    for coord, res in challenge:
        res = dict(zip(["MaxProb Region", "Prob"], list(res)))
        print res
        res = pd.DataFrame(res, index=[0])
        yield assert_equal, np.array(res), np.array(main.locate_peaks(coord))


def test_shorten_name():

    names = [("Parahippocampal Gyrus, anterior division",
              "Parahip G, ant",
              "ctx"),
             ("Middle Frontal Gyrus", "MFG", "ctx"),
             ("Right Hippocampus", "R Hippocampus", "sub")]

    for orig, new, atlas in names:
        yield assert_equal, new, main.shorten_name(orig, atlas)


def test_vox_to_mni():

    coords = [((29, 68, 57), (32, 10, 42)),
              ((70, 38, 42), (-50, -50, 12)),
              ((45, 63, 36), (0, 0, 0))]

    for vox, mni in coords:
        vox = np.atleast_2d(vox)
        mni = np.atleast_2d(mni)
        yield assert_equal, mni, main.vox_to_mni(vox)
