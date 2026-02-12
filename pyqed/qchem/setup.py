#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 30 21:56:03 2025

@author: bingg
"""

from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("basis.py")
)