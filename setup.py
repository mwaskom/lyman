#! /usr/bin/env python

DISTNAME = 'lyman'
DESCRIPTION = 'lyman: neuroimaging analysis in Python'
MAINTAINER = 'Michael Waskom'
MAINTAINER_EMAIL = 'mwaskom@nyu.edu'
LICENSE = 'BSD (3-clause)'
URL = 'http://www.cns.nyu.edu/~mwaskom/software/lyman/'
DOWNLOAD_URL = 'https://github.com/mwaskom/lyman'
VERSION = '2.0.0.dev'
INSTALL_REQUIRES = [
    'numpy',
    'scipy',
    'pandas',
    'matplotlib',
    'pyyaml',
    'traits',
    'nipype',
    'nibabel',
]
PACKAGES = [
    'lyman',
    'lyman.tests',
    'lyman.workflows',
    'lyman.workflows.tests',
]
SCRIPTS = [
    'scripts/lyman',
]
CLASSIFIERS = [
    'Intended Audience :: Science/Research',
    'Programming Language :: Python :: 3.6',
    'License :: OSI Approved :: BSD License',
    'Operating System :: POSIX',
    'Operating System :: Unix',
    'Operating System :: MacOS',
]


if __name__ == '__main__':

    from setuptools import setup

    setup(
        name=DISTNAME,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        version=VERSION,
        url=URL,
        download_url=DOWNLOAD_URL,
        install_requires=INSTALL_REQUIRES,
        packages=PACKAGES,
        scripts=SCRIPTS,
        classifiers=CLASSIFIERS,
    )
