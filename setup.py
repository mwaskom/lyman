#! /usr/bin/env python
#
# Copyright (C) 2012 Michael Waskom <mwaskom@stanford.edu>

descr = """Lyman: Tools for analyzing neuroimaging data."""

import os


DISTNAME = 'lyman'
DESCRIPTION = descr
MAINTAINER = 'Michael Waskom'
MAINTAINER_EMAIL = 'mwaskom@stanford.edu'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/mwaskom/lyman'
VERSION = '0.1.dev'

from numpy.distutils.core import setup


if __name__ == "__main__":
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    setup(name=DISTNAME,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        version=VERSION,
        download_url=DOWNLOAD_URL,
        packages=['lyman',
                  'lyman.workflows', 'lyman.workflows.tests',
                  'lyman.tools', 'lyman.tools.tests'],
        scripts=['scripts/run_fmri.py', 'scripts/run_group.py',
                 'scripts/run_warp.py', 'scripts/setup_project.py',
                 'scripts/make_masks.py'],
    )
