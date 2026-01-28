"""
MetaTrajectory: Optimal Transport-based Metabolic Trajectory Analysis

A mathematically rigorous framework for analyzing metabolic state transitions
in single-cell RNA-seq data using optimal transport theory.
"""

from setuptools import setup, find_packages

with open("../README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="metatrajectory",
    version="1.0.0",
    author="MetaTrajectory Team",
    author_email="metatrajectory@example.com",
    description="Optimal Transport-based Metabolic Trajectory Analysis for Single-Cell Data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/metatrajectory/metatrajectory",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
        "pandas>=1.3",
        "scipy>=1.7",
        "scanpy>=1.9",
        "anndata>=0.8",
        "POT>=0.8",
        "networkx>=2.8",
        "scikit-learn>=1.0",
        "matplotlib>=3.5",
        "seaborn>=0.11",
    ],
    extras_require={
        "velocity": ["scvelo>=0.2"],
        "dev": ["pytest>=7.0", "pytest-cov>=3.0", "black>=22.0", "flake8>=4.0"],
        "docs": ["sphinx>=4.0", "sphinx-rtd-theme>=1.0"],
    },
)
