#! /usr/bin/env python
import sys
import util
from util.commandline import parser
from workflows import normalization


def main(arglist):

    args = parser.parse_args(arglist)

    subject_list = util.determine_subjects(args.subjects)
    project_info = util.gather_project_info()

    normalize = normalization.create_normalization_workflow(project_info['data_dir'], subject_list)
    
    plugin, plugin_args = util.determine_engine(args)

    normalize.config = dict(crashdump_dir="/tmp")
    normalize.run(plugin=plugin, plugin_args=plugin_args)

if __name__ == "__main__":
    main(sys.argv[1:])
