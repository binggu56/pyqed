#!/usr/bin/env python3

from setuptools import Extension, setup

import numpy as np
from Cython.Build import cythonize


extensions = [
    Extension(
        name="_basis_cy",
        sources=["_basis_cy.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        name="_rys_cy",
        sources=["_rys_cy.pyx"],
        include_dirs=[np.get_include()],
    ),
]


setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    )
)
