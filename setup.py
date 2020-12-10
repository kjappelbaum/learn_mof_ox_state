#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=invalid-name, missing-docstring, line-too-long

import io
import os

from setuptools import find_packages, setup

import versioneer

# Package meta-data.
NAME = "learnmofox"
DESCRIPTION = "machine learn MOF oxidation states"
URL = "https://github.com/kjappelbaum/learn_mof_ox_state"
EMAIL = "kevin.jablonka@epfl.ch"
AUTHOR = "Kevin M. Jablonka, Daniele Ongari, Seyed Mohamad Moosavi, Berend Smit"
REQUIRES_PYTHON = ">=3.6.0"

# What packages are required for this module to be executed?
with open("requirements.txt", "r") as fh:
    _REQUIRED = fh.readlines()

REQUIRED = []

for line in _REQUIRED:
    line = line.strip().replace("\n", "")
    if "git" in line:
        package = line.split("#")[-1].replace("egg=", "")
        required = package + " @ " + line
        REQUIRED.append(required)
    else:
        REQUIRED.append(line)

# What packages are optional?
EXTRAS = {
    "testing": ["pytest~=6.1.0", "pytest-cov~=2.10"],
    "docs": [
        "sphinx~=3.2.1",
        "sphinx-book-theme~=0.0.39",
        "sphinx-autodoc-typehints~=1.11.0",
        "sphinx-copybutton~=0.3.0",
    ],
    "pre-commit": [
        "pre-commit~=2.7.1",
        "pylint~=2.6",
        "isort~=5.5.3",
    ],
    "dev": [
        "versioneer~=0.18",
        "black~=20.8b1",
    ],
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
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
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
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
)
