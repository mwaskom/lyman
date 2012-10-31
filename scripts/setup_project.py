#! /usr/bin/env python
"""
Script to guide project setup.

The basic idea here is borrowed from the Sphinx project, and
in fact a significant amount of the code in this file was
borrowed from there too.

"""
import sys, os, time
import os.path as op

from lyman.tools.console import (purple, bold, red, teal, turquoise,
                                 color_terminal, nocolor)

TERM_ENCODING = getattr(sys.stdin, 'encoding', None)

PROMPT_PREFIX = "> "

PROJECT_CONF = """\
# -*- coding: utf-8 -*-
#
# Configuration file for %(project_name)s built on %(now)s

# Default experiment name
default_exp = %(default_exp)s

# Data directory is where the input data lives
data_dir = '%(data_dir)s'

# Analysis directory is where anything written by these scripts will live
analysis_dir = '%(analysis_dir)s'

# Working directory is where data lives during workflow execution
working_dir = '%(working_dir)s'

# Set this to True to remove the working directory after each excecution
rm_working_dir = %(rm_work_dir)s

"""

def mkdir_p(dir):
    if op.isdir(dir):
        return
    os.makedirs(dir)


class ValidationError(Exception):
    """Raised for validation errors."""

def is_path(x):
    if op.exists(x) and not op.isdir(x):
        raise ValidationError("Please enter a valid path name.")
    return op.abspath(x)

def nonnull_string(s):
    if s is not None:
        return "'s'"

def nonempty(x):
    if not x:
        raise ValidationError("Please enter some text.")
    return x

def choice(*l):
    def val(x):
        if x not in l:
            raise ValidationError('Please enter one of %s.' % ', '.join(l))
        return x
    return val

def boolean(x):
    if x.upper() not in ('Y', 'YES', 'N', 'NO'):
        raise ValidationError("Please enter either 'y' or 'n'.")
    return x.upper() in ('Y', 'YES')

def suffix(x):
    if not (x[0:1] == '.' and len(x) > 1):
        raise ValidationError("Please enter a file suffix, "
                              "e.g. '.rst' or '.txt'.")
    return x

def ok(x):
    return x


def do_prompt(d, key, text, default=None, validator=nonempty):
    while True:
        if default:
            prompt = purple(PROMPT_PREFIX + '%s [%s]: ' % (text, default))
        else:
            prompt = purple(PROMPT_PREFIX + text + ': ')
        x = raw_input(prompt)
        if default and not x:
            x = default
        if x.decode('ascii', 'replace').encode('ascii', 'replace') != x:
            if TERM_ENCODING:
                x = x.decode(TERM_ENCODING)
            else:
                print turquoise('* Note: non-ASCII characters entered '
                                'and terminal encoding unknown -- assuming '
                                'UTF-8 or Latin-1.')
                try:
                    x = x.decode('utf-8')
                except UnicodeDecodeError:
                    x = x.decode('latin1')
        try:
            x = validator(x)
        except ValidationError, err:
            print red('* ' + str(err))
            continue
        break
    d[key] = x




def main():
    
    d = dict()

    if not color_terminal():
        nocolor()

    # Check if a project file already exists
    if op.exists("project.py"):

        # But let's make sure it's clean
        try:
            import project
            clean_import = True
        except Exception:
            clean_import = False
        import_notes = "" if clean_import else ", but it did not import cleanly"
        
        # Maybe give a heads up about it
        print red("Warning:"), """\
project.py file found in current directory%s.

Do you wish to generate a new project file?
(Note that you can always edit the existing file).
""" % import_notes

        # And let the user choose whether to overwrite it
        do_prompt(d, "overwrite", "Overwrite existing file? (y/N)",
                  "n", boolean)
        
        if not d["overwrite"]:
            print teal("Aborting project setup.")
            sys.exit(0)
        os.remove("project.py")

    # Now go through the prompted setup procedure
    print bold("Let's set up your project.")

    print '''
Please enter values for the following settings (just press Enter to
accept a default value, if one is given in brackets).'''

    do_prompt(d, "project_name", "Project name")

    do_prompt(d, "default_exp", "Default experiment", None, nonnull_string)

    do_prompt(d, "data_dir", "Data tree path", "../data", is_path)

    do_prompt(d, "analysis_dir", "Analysis tree path", "../analysis", is_path)
    
    do_prompt(d, "working_dir", "Working tree path", 
              op.join(d['analysis_dir'], 'workingdir'), is_path)

    do_prompt(d, "rm_work_dir", "Remove working directory after execution? (y/N)",
              "n", boolean)


    # Record the time this happened
    d['now'] = time.asctime()

    # Write the project file
    f = open("project.py", "w")
    conf_text = PROJECT_CONF % d
    f.write(conf_text.encode("utf-8"))
    f.close()

if __name__ == "__main__":
    main()
