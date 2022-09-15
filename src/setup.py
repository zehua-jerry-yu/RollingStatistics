"""
python setup.py install
"""

from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup
import os
import shutil


ext_modules = [
    Pybind11Extension(
        'rolling_statistics_py',
        sources=['rolling_statistics_py.cpp'],
        language='c++',
        cxx_std=11
    ),
]

setup(
    name='rolling_statistics_py',
    author='Zehua Yu',
    ext_modules=ext_modules
)

# copy the package
for filename in os.listdir("dist"):
    if filename.endswith(".egg"):
        shutil.copy(os.path.join("dist", filename), ".")
