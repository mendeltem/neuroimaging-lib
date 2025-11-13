#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup configuration for neuroimaging directory utilities library.
Install with: pip install .
Or in development mode: pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="neuroimaging-lib",
    version="0.1.0",
    author="temuuleu",
    author_email="your.email@example.com",
    description="Utilities for neuroimaging directory management and file handling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/neuroimaging-lib",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=[
        "nibabel>=5.0.0",
        "nipype>=1.8.0",
        "ants>=0.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "flake8>=5.0",
            "mypy>=0.990",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)