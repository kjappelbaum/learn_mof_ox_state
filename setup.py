#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=invalid-name, missing-docstring, line-too-long

from __future__ import absolute_import
import io
import os
import sys
import subprocess

from setuptools import find_packages, setup

git_rpmfile = (
    "git+https://github.com/kjappelbaum/hyperopt-sklearn.git, git+#egg=hyperopt==0.2.2"
)

try:
    import hyperopt  # pylint:disable=unused-import
    import hpsklearn  # pylint:disable=unused-import
except Exception:  # pylint:disable=broad-except
    if "--user" in sys.argv:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--user",
                git_rpmfile,
            ],
            check=False,
        )
    else:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", git_rpmfile],
            check=False,
        )

# Package meta-data.
NAME = "learnmofox"
DESCRIPTION = "machine learn MOF oxidation states"
URL = "https://github.com/kjappelbaum/learn_mof_ox_state"
EMAIL = "kevin.jablonka@epfl.ch"
AUTHOR = "Kevin M. Jablonka, Daniele Ongari, Seyed Mohamad Moosavi, Berend Smit"
REQUIRES_PYTHON = ">=3.5.0"
VERSION = "0.2.0-alpha"

# What packages are required for this module to be executed?
REQUIRED = [
    "sklearn",
    "imblearn",
    "ml_insights",
    "mlxtend",
    "pandas",
    "click",
    "comet",
    "dvc",
    "numpy",
    "tqdm",
    "dask",
    "shap",
    "scipy",
]

# What packages are optional?
EXTRAS = {
    "testing": ["pytest"],
    "linting": ["prospector", "pre-commit", "pylint"],
    "documentation": ["sphinx", "sphinx_rtd_theme", "sphinx-autodoc-typehints"],
}

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
)
