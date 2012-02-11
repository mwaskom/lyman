#! /usr/bin/env python
import sys
import tools
from tools.commandline import parser
from workflows import anatwarp


def main(arglist):

    args = parser.parse_args(arglist)

    subject_list = tools.determine_subjects(args.subjects)
    project_info = tools.gather_project_info()

    normalize = anatwarp.create_anatwarp_workflow(
                    project_info['data_dir'], subject_list)

    plugin, plugin_args = tools.determine_engine(args)

    tools.crashdump_config(normalize, "/tmp")
    normalize.run(plugin=plugin, plugin_args=plugin_args)

if __name__ == "__main__":
    main(sys.argv[1:])
