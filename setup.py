#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

import glob
import os
from pathlib import Path
import sys

from setuptools import Extension, setup
from setuptools.command.build_py import build_py as _build_py


def _is_sync_conflict_copy(module_path):
    """Exclude OneDrive conflict copies without deleting a developer's files."""
    filename = Path(module_path).name.lower()
    return "gugroup" in filename or "bing" in filename and "mac" in filename


class _CleanBuildPy(_build_py):
    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        return [
            module
            for module in modules
            if not _is_sync_conflict_copy(module[2])
        ]


def _cpp_include_dirs(np):
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


def _mps_openmp_flags():
    """Return optional OpenMP compile and link flags for native MPS kernels."""
    requested = os.environ.get("PYQED_MPS_OPENMP", "auto").strip().lower()
    if requested in {"0", "false", "no", "off"}:
        return [], []
    if sys.platform == "win32":
        return ["/openmp"], []
    if sys.platform != "darwin":
        return ["-fopenmp"], ["-fopenmp"]

    prefixes = []
    explicit = os.environ.get("PYQED_OPENMP_PREFIX")
    if explicit:
        prefixes.append(Path(explicit))
    # Apple clang should use the matching Homebrew LLVM runtime when it is
    # available.  Conda may inject an older libomp and its RPATH ahead of
    # extension-specific flags, which can otherwise break module import.
    prefixes.extend((Path("/opt/homebrew/opt/libomp"), Path("/usr/local/opt/libomp")))
    prefixes.extend(
        Path(value)
        for value in (sys.prefix, os.environ.get("CONDA_PREFIX"))
        if value
    )
    prefixes = list(dict.fromkeys(prefixes))
    for prefix in prefixes:
        include = prefix / "include"
        library = prefix / "lib"
        dylib = library / "libomp.dylib"
        archive = library / "libomp.a"
        runtime = dylib if dylib.exists() else archive
        if (include / "omp.h").exists() and runtime.exists():
            link_args = [str(runtime)]
            if runtime.suffix == ".dylib":
                link_args.append("-Wl,-rpath," + str(library))
            return (
                ["-Xpreprocessor", "-fopenmp", "-I" + str(include)],
                link_args,
            )
    if requested in {"1", "true", "yes", "on", "required"}:
        raise RuntimeError(
            "OpenMP was requested but libomp was not found; set PYQED_OPENMP_PREFIX"
        )
    return [], []


def _optional_extensions():
    enabled = os.environ.get("PYQED_BUILD_EXTENSIONS", "0")
    if enabled.strip().lower() not in {"1", "true", "yes", "on"}:
        return []
    groups = {
        item.strip().lower()
        for item in os.environ.get("PYQED_EXTENSION_GROUPS", "all").split(",")
        if item.strip()
    }
    valid_groups = {"all", "qchem", "heom", "mps", "letta", "ldr"}
    unknown_groups = groups.difference(valid_groups)
    if unknown_groups:
        names = ", ".join(sorted(unknown_groups))
        raise ValueError(f"unknown PYQED_EXTENSION_GROUPS entries: {names}")

    extensions = []
    try:
        import numpy as np
    except Exception:
        np = None

    if np is not None:
        cpp_include_dirs = _cpp_include_dirs(np)
        cpp_compile_args = (
            ["/std:c++17", "/O2"]
            if sys.platform == "win32"
            else ["-std=c++17", "-O3"]
        )
        c_compile_args = ["/O2"] if sys.platform == "win32" else ["-O3"]
        lto_enabled = os.environ.get("PYQED_LTO", "1").strip().lower() in {
            "1", "true", "yes", "on",
        }
        native_cpu = os.environ.get("PYQED_NATIVE_CPU", "0").strip().lower() in {
            "1", "true", "yes", "on",
        }
        qchem_integral_compile_args = list(cpp_compile_args)
        qchem_integral_link_args = []
        if lto_enabled:
            if sys.platform == "win32":
                qchem_integral_compile_args.append("/GL")
                qchem_integral_link_args.append("/LTCG")
            else:
                qchem_integral_compile_args.append("-flto")
                qchem_integral_link_args.append("-flto")
        if native_cpu and sys.platform != "win32":
            qchem_integral_compile_args.append("-march=native")
        accelerate_link_args = (
            ["-framework", "Accelerate"] if sys.platform == "darwin" else []
        )
        mps_openmp_compile_args, mps_openmp_link_args = _mps_openmp_flags()
        casscf_libraries = ["blas"] if sys.platform.startswith("linux") else []
        casscf_macros = (
            [("PYQED_USE_CBLAS", "1")]
            if sys.platform.startswith("linux")
            else []
        )
        extensions.append(
            Extension(
                "pyqed.qchem._integrals_cpp",
                ["pyqed/qchem/_integrals.cpp"],
                include_dirs=cpp_include_dirs,
                language="c++",
                extra_compile_args=qchem_integral_compile_args,
                extra_link_args=qchem_integral_link_args,
                optional=True,
            )
        )
        extensions.append(
            Extension(
                "pyqed.qchem._casscf_cpp",
                ["pyqed/qchem/_casscf_cpp.cpp"],
                include_dirs=cpp_include_dirs,
                language="c++",
                extra_compile_args=cpp_compile_args,
                extra_link_args=accelerate_link_args,
                libraries=casscf_libraries,
                define_macros=casscf_macros,
                optional=True,
            )
        )
        extensions.append(
            Extension(
                "pyqed.qchem._gdf_cpp",
                ["pyqed/qchem/_gdf_cpp.cpp"],
                include_dirs=cpp_include_dirs,
                language="c++",
                extra_compile_args=cpp_compile_args,
                optional=True,
            )
        )
        extensions.append(
            Extension(
                "pyqed.heom._heom_cpp",
                ["pyqed/heom/_heom_cpp.cpp"],
                depends=["pyqed/_dop853.hpp"],
                include_dirs=cpp_include_dirs,
                language="c++",
                extra_compile_args=cpp_compile_args,
                optional=True,
            )
        )

    if np is not None:
        extensions.append(
            Extension(
                "pyqed.qchem._basis_cy",
                ["pyqed/qchem/_basis_cy.pyx"],
                include_dirs=[np.get_include()],
                extra_compile_args=c_compile_args,
                optional=True,
            )
        )
        extensions.append(
            Extension(
                "pyqed.qchem._rys_cy",
                ["pyqed/qchem/_rys_cy.pyx"],
                include_dirs=[np.get_include()],
                extra_compile_args=c_compile_args,
                optional=True,
            )
        )
        extensions.append(
            Extension(
                "pyqed.mps.nonabelian._su2_kernel",
                [
                    "pyqed/mps/nonabelian/_su2_kernel.cpp",
                    "pyqed/mps/nonabelian/su2_dmrg_engine.cpp",
                ],
                depends=[
                    "pyqed/mps/nonabelian/su2_dmrg_engine.hpp",
                    "pyqed/mps/nonabelian/su2_coupling_core.hpp",
                    "pyqed/mps/dmrg_linalg_core.hpp",
                ],
                include_dirs=cpp_include_dirs,
                language="c++",
                extra_compile_args=cpp_compile_args + mps_openmp_compile_args,
                extra_link_args=accelerate_link_args + mps_openmp_link_args + (
                    ["-ldl"] if sys.platform.startswith("linux") else []
                ),
                optional=True,
            )
        )
        extensions.append(
            Extension(
                "pyqed.ldr._kernels_cpp",
                ["pyqed/ldr/_kernels.cpp"],
                include_dirs=cpp_include_dirs,
                language="c++",
                extra_compile_args=cpp_compile_args,
                optional=True,
            )
        )
        extensions.append(
            Extension(
                "pyqed.letta._physical_blocks_cpp",
                ["pyqed/letta/_physical_blocks_cpp.cpp"],
                include_dirs=cpp_include_dirs,
                language="c++",
                extra_compile_args=cpp_compile_args,
                optional=True,
            )
        )
        extensions.append(
            Extension(
                "pyqed.letta._support_kernels_cpp",
                ["pyqed/letta/_support_kernels_cpp.cpp"],
                include_dirs=cpp_include_dirs,
                language="c++",
                extra_compile_args=cpp_compile_args,
                optional=True,
            )
        )
        extensions.append(
            Extension(
                "pyqed.letta._conditional_gauge_cpp",
                ["pyqed/letta/_conditional_gauge_cpp.cpp"],
                include_dirs=cpp_include_dirs,
                language="c++",
                extra_compile_args=cpp_compile_args,
                optional=True,
            )
        )
        extensions.append(
            Extension(
                "pyqed.letta._copy_einsum_cpp",
                ["pyqed/letta/_copy_einsum_cpp.cpp"],
                include_dirs=cpp_include_dirs,
                language="c++",
                extra_compile_args=cpp_compile_args,
                optional=True,
            )
        )
    if "all" in groups:
        return extensions
    prefixes = tuple(f"pyqed.{group}." for group in sorted(groups))
    return [
        extension
        for extension in extensions
        if extension.name.startswith(prefixes)
    ]


extensions = _optional_extensions()
if any(
    source.endswith(".pyx")
    for extension in extensions
    for source in extension.sources
):
    from Cython.Build import cythonize

    extensions = cythonize(
        extensions,
        build_dir="build/cython",
        compiler_directives={"language_level": "3"},
    )


# Project metadata lives in pyproject.toml.  This small compatibility hook is
# retained solely for the optional native extensions.
setup(
    cmdclass={"build_py": _CleanBuildPy},
    ext_modules=extensions,
)
