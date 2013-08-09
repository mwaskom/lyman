import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-s", "-subjects", nargs="*", dest="subjects",
                    help=("list of subject ids, name of file in lyman directory, "
                          "or full path to text file with subject ids"))
parser.add_argument("-plugin", default="multiproc",
                    choices=["linear", "multiproc", "ipython", "torque", "sge"],
                    help="worklow execution plugin")
parser.add_argument("-nprocs", default=4, type=int,
                    help="number of MultiProc processes to use")
parser.add_argument("-queue", help="which queue for PBS/SGE execution")
