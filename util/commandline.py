import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-s", "-subjects", nargs="*",
                    help="subject ids or path to text file")
parser.add_argument("-plugin", default="multiproc",
                    choices=["linear", "multiproc", "ipython", "torque"],
                    help="worklow execution plugin to use")
parser.add_argument("-nprocs", default=4, type=int,
                    help="number of MultiProc processes to use")

