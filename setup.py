#! /usr/bin/env python
#
# Copyright (C) 2012-2014 Michael Waskom <mwaskom@stanford.edu>

descr = """Lyman: Tools for analyzing neuroimaging data."""

import os
from setuptools import setup

DISTNAME = 'lyman'
DESCRIPTION = descr
MAINTAINER = 'Michael Waskom'
MAINTAINER_EMAIL = 'mwaskom@stanford.edu'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/mwaskom/lyman'
VERSION = '0.1.dev'

def check_dependencies():

    # Just make sure dependencies exist, I haven't rigorously
    # tested what the minimal versions that will work are
    needed_deps = ["IPython", "numpy", "scipy", "matplotlib",
                   "sklearn", "skimage", "pandas", "statsmodels", 
                   "nibabel", "nipype", "nipy", "seaborn", "moss"]
    missing_deps = []
    for dep in needed_deps:
        try:
            __import__(dep)
        except ImportError:
            missing_deps.append(dep)

    if missing_deps:
        missing = (", ".join(missing_deps)
                   .replace("sklearn", "scikit-learn")
                   .replace("skimage", "scikit-image"))
        raise ImportError("Missing dependencies: %s" % missing)

if __name__ == "__main__":

    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    check_dependencies()

    setup(name=DISTNAME,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        version=VERSION,
        download_url=DOWNLOAD_URL,
        packages=['lyman', 'lyman.tests',
                  'lyman.workflows', 'lyman.workflows.tests',
                  'lyman.tools', 'lyman.tools.tests'],
        scripts=['scripts/run_fmri.py', 'scripts/run_group.py',
                 'scripts/run_warp.py', 'scripts/setup_project.py',
                 'scripts/make_masks.py', 'scripts/anatomy_snapshots.py',
                 'scripts/surface_snapshots.py'],
    )
