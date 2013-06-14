import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-s", "-subjects", nargs="*", dest="subjects",
                    help="subject ids or path to text file")
parser.add_argument("-plugin", default="multiproc",
                    choices=["linear", "multiproc", "ipython", "torque", "sge"],
                    help="worklow execution plugin to use")
parser.add_argument("-nprocs", default=4, type=int,
                    help="number of MultiProc processes to use")
parser.add_argument("-queue", help"which cue for PBS/SGE execution")
