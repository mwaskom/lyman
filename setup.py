#! /usr/bin/env python

DISTNAME = 'lyman'
DESCRIPTION = 'lyman: neuroimaging analysis in Python'
MAINTAINER = 'Michael Waskom'
MAINTAINER_EMAIL = 'mwaskom@nyu.edu'
LICENSE = 'BSD (3-clause)'
URL = 'http://www.cns.nyu.edu/~mwaskom/software/lyman/'
DOWNLOAD_URL = 'https://github.com/mwaskom/lyman'
VERSION = '2.0.0'
PYTHON_REQUIRES = ">=3.7"
INSTALL_REQUIRES = [
    'numpy',
    'scipy',
    'pandas',
    'matplotlib',
    'pyyaml',
    'traits',
    'nibabel',
    'nipype>=1.0',
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
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
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
        python_requires=PYTHON_REQUIRES,
        install_requires=INSTALL_REQUIRES,
        packages=PACKAGES,
        scripts=SCRIPTS,
        classifiers=CLASSIFIERS,
    )
