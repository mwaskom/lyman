#! /usr/bin/env python
import sys
import shutil
import lyman
from lyman import tools
from lyman.workflows import anatwarp


def main(arglist):

    # Process cmdline args
    args = tools.parser.parse_args(arglist)
    plugin, plugin_args = lyman.determine_engine(args)

    # Load up the lyman info
    subject_list = lyman.determine_subjects(args.subjects)
    project = lyman.gather_project_info()

    # Create the workflow object
    normalize = anatwarp.create_anatwarp_workflow(
                    project["data_dir"], subject_list)
    normalize.base_dir = project["working_dir"]
    normalize.config["execution"]["crashdump_dir"] = "/tmp"

    # Execute the workflow
    lyman.run_workflow(normalize, args=args)

    # Clean up
    if project["rm_working_dir"]:
        shutil.rmtree(normalize.base_dir)

if __name__ == "__main__":
   main(sys.argv[1:])
