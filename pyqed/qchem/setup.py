#!/usr/bin/env python3

import glob
import os
import sys

from setuptools import Extension, setup

import numpy as np
from Cython.Build import cythonize


def _cpp_include_dirs():
    include_dirs = [np.get_include()]
    if sys.platform == "darwin":
        candidates = []
        sdkroot = os.environ.get("SDKROOT")
        if sdkroot:
            candidates.append(f"{sdkroot}/usr/include/c++/v1")
        candidates.extend(
            sorted(
                glob.glob("/Library/Developer/CommandLineTools/SDKs/MacOSX*.sdk/usr/include/c++/v1"),
                reverse=True,
            )
        )
        candidates.extend(
            sorted(glob.glob(f"{sys.prefix}/pkgs/libcxx-*/include/c++/v1"), reverse=True)
        )
        for path in candidates:
            if os.path.isdir(path) and path not in include_dirs:
                include_dirs.append(path)
                break
        return include_dirs
    for pattern in (
        f"{sys.prefix}/pkgs/libcxx-*/include/c++/v1",
        f"{sys.prefix}/include/c++/v1",
    ):
        for path in sorted(glob.glob(pattern), reverse=True):
            if path not in include_dirs:
                include_dirs.append(path)
                return include_dirs
    return include_dirs


cpp_include_dirs = _cpp_include_dirs()
accelerate_link_args = ["-framework", "Accelerate"] if sys.platform == "darwin" else []

extensions = [
    Extension(
        name="_integrals_cpp",
        sources=["_integrals.cpp"],
        include_dirs=cpp_include_dirs,
        language="c++",
        extra_compile_args=["-std=c++17", "-O3"],
    ),
    Extension(
        name="_casscf_cpp",
        sources=["_casscf_cpp.cpp"],
        include_dirs=cpp_include_dirs,
        language="c++",
        extra_compile_args=["-std=c++17", "-O3"],
        extra_link_args=accelerate_link_args,
    ),
    Extension(
        name="_gdf_cpp",
        sources=["_gdf_cpp.cpp"],
        include_dirs=cpp_include_dirs,
        language="c++",
        extra_compile_args=["-std=c++17", "-O3"],
    ),
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
