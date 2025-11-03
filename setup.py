"""
Setup script for PINN C2P Solver package.
"""
from setuptools import setup, find_packages

setup(
    name="pinn-c2p-solver",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
)