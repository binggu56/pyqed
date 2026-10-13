#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

import glob
import os
from pathlib import Path
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.archive_util import unpack_archive
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.install_egg_info import install_egg_info as _install_egg_info


def _is_sync_conflict_copy(module_path):
    """Exclude OneDrive conflict copies without deleting a developer's files."""
    filename = Path(module_path).name.lower()
    return "gugroup" in filename or "bing" in filename and "mac" in filename


class _CleanBuildPy(_build_py):
    def run(self):
        super().run()
        build_root = Path(self.build_lib)
        for path in build_root.rglob("*"):
            if path.is_file() and _is_sync_conflict_copy(path):
                path.unlink()

    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        return [
            module
            for module in modules
            if not _is_sync_conflict_copy(module[2])
        ]

    def find_data_files(self, package, src_dir):
        return [
            path
            for path in super().find_data_files(package, src_dir)
            if not _is_sync_conflict_copy(path)
        ]


class _LinkedBuildExt(_build_ext):
    @staticmethod
    def _openmp_runtime(ext):
        return next((Path(arg) for arg in ext.extra_link_args or ()
                     if Path(arg).is_absolute() and Path(arg).name == "libomp.dylib"), None)

    def build_extensions(self):
        force, compiler_force = self.force, self.compiler.force
        try:
            if sys.platform == "darwin":
                for ext in self.extensions:
                    runtime = self._openmp_runtime(ext)
                    output = Path(self.get_ext_fullpath(ext.name))
                    if runtime is None or not output.exists():
                        continue
                    linked = subprocess.run(["otool", "-L", str(output)], check=True,
                                            capture_output=True, text=True).stdout
                    libraries = [line.strip().split(" (", 1)[0] for line in linked.splitlines()[1:]]
                    if str(runtime) not in libraries:
                        # Decide before parallel compilation: timestamps do not
                        # account for a changed OpenMP runtime or link flags.
                        self.force = self.compiler.force = True
                        break
            super().build_extensions()
        finally:
            self.force, self.compiler.force = force, compiler_force

    def build_extension(self, ext):
        super().build_extension(ext)
        runtime = self._openmp_runtime(ext)
        if sys.platform == "darwin" and runtime is not None:
            # libomp's install name is @rpath even when linked by full path.
            # Conda's earlier RPATH can select an incompatible ABI.
            subprocess.run(
                ["install_name_tool", "-change", "@rpath/libomp.dylib",
                 str(runtime), self.get_ext_fullpath(ext.name)], check=True,
            )


class _CleanInstallEggInfo(_install_egg_info):
    def copytree(self):
        def keep(src, dst):
            if _is_sync_conflict_copy(src):
                return None
            for marker in ".svn/", "CVS/":
                if src.startswith(marker) or "/" + marker in src:
                    return None
            self.outputs.append(dst)
            return dst

        unpack_archive(self.source, self.target, keep)


def _cpp_include_dirs(np):
    include_dirs = [np.get_include()]
    if sys.platform == "darwin":
        sdkroot = os.environ.get("SDKROOT")
        if not sdkroot:
            try:
                sdkroot = subprocess.run(
                    ["xcrun", "--show-sdk-path"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            except (OSError, subprocess.CalledProcessError):
                sdkroot = None
        if sdkroot:
            path = f"{sdkroot}/usr/include/c++/v1"
            if os.path.isdir(path):
                include_dirs.append(path)
        # xcrun selects the libc++ headers matching the active Xcode toolchain.
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


def _mps_openmp_flags(requested=None):
    """Return optional OpenMP compile and link flags for native MPS kernels."""
    requested = (os.environ.get("PYQED_MPS_OPENMP", "auto") if requested is None else requested).strip().lower()
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


def _extensions_to_build():
    enabled = os.environ.get("PYQED_BUILD_EXTENSIONS", "1").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        return []
    if enabled not in {"1", "true", "yes", "on"}:
        raise ValueError(
            "PYQED_BUILD_EXTENSIONS must be an explicit true or false value."
        )
    groups = {
        item.strip().lower()
        for item in os.environ.get("PYQED_EXTENSION_GROUPS", "qchem").split(",")
        if item.strip()
    }
    valid_groups = {"all", "qchem", "gw", "heom", "mps", "letta", "ldr"}
    unknown_groups = groups.difference(valid_groups)
    if unknown_groups:
        names = ", ".join(sorted(unknown_groups))
        raise ValueError(f"unknown PYQED_EXTENSION_GROUPS entries: {names}")
    if not groups:
        raise ValueError("PYQED_EXTENSION_GROUPS must not be empty.")
    groups.add("qchem")
    groups.add("gw")

    extensions = []
    try:
        import numpy as np
    except Exception:
        np = None

    if np is None:
        raise RuntimeError("NumPy is required to build the qchem accelerators.")

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
        gw_openmp_requested = os.environ.get("PYQED_GW_OPENMP", "auto").strip().lower()
        gw_openmp_compile_args, gw_openmp_link_args = _mps_openmp_flags(gw_openmp_requested)
        import pybind11
        extensions.append(
            Extension(
                "pyqed.gw._dyson",
                ["pyqed/gw/dyson.cpp"],
                include_dirs=cpp_include_dirs + [pybind11.get_include()],
                language="c++",
                extra_compile_args=cpp_compile_args + gw_openmp_compile_args,
                extra_link_args=gw_openmp_link_args,
                optional=gw_openmp_requested not in {"required", "1", "true", "yes", "on"},
            )
        )
        bse_openmp_requested = os.environ.get("PYQED_BSE_OPENMP", "auto").strip().lower()
        bse_compile_args, bse_link_args = _mps_openmp_flags(bse_openmp_requested)
        extensions.append(
            Extension(
                "pyqed.gw._bse_contractions",
                ["pyqed/gw/bse_contractions.cpp"],
                depends=["pyqed/linalg/row_product.hpp"],
                include_dirs=cpp_include_dirs + [pybind11.get_include()],
                language="c++",
                extra_compile_args=cpp_compile_args + bse_compile_args,
                extra_link_args=bse_link_args,
                optional=bse_openmp_requested not in {"required", "1", "true", "yes", "on"},
            )
        )
        contour_openmp_requested = os.environ.get("PYQED_CONTOUR_OPENMP", "auto").strip().lower()
        contour_compile_args, contour_link_args = _mps_openmp_flags(contour_openmp_requested)
        extensions.append(
            Extension(
                "pyqed.gw._contour_contractions",
                ["pyqed/gw/contour_contractions.cpp"],
                depends=["pyqed/linalg/row_product.hpp"],
                include_dirs=cpp_include_dirs + [pybind11.get_include()],
                language="c++",
                extra_compile_args=cpp_compile_args + contour_compile_args,
                extra_link_args=contour_link_args,
                optional=contour_openmp_requested not in {"required", "1", "true", "yes", "on"},
            )
        )
        # Accelerate is available on macOS.  Generic Linux wheels must retain
        # the extension's internal fallback instead of assuming a system BLAS
        # development library is installed on the target machine.
        casscf_libraries = []
        casscf_macros = []
        extensions.append(
            Extension(
                "pyqed.qchem._integrals_cpp",
                ["pyqed/qchem/_integrals.cpp"],
                depends=["pyqed/qchem/ri_derivatives.hpp"],
                include_dirs=cpp_include_dirs,
                language="c++",
                extra_compile_args=qchem_integral_compile_args,
                extra_link_args=qchem_integral_link_args,
                optional=False,
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
                optional=False,
            )
        )
        extensions.append(
            Extension(
                "pyqed.qchem._gdf_cpp",
                ["pyqed/qchem/_gdf_cpp.cpp"],
                include_dirs=cpp_include_dirs,
                language="c++",
                extra_compile_args=cpp_compile_args,
                optional=False,
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
                optional=False,
            )
        )
        extensions.append(
            Extension(
                "pyqed.qchem._rys_cy",
                ["pyqed/qchem/_rys_cy.pyx"],
                include_dirs=[np.get_include()],
                extra_compile_args=c_compile_args,
                optional=False,
            )
        )
        extensions.append(
            Extension(
                "pyqed.mps.nonabelian._su2_kernel",
                [
                    "pyqed/mps/nonabelian/_su2_kernel.pyx",
                    "pyqed/mps/nonabelian/su2_dmrg_engine.cpp",
                ],
                depends=[
                    "pyqed/mps/nonabelian/su2_dmrg_engine.hpp",
                    "pyqed/mps/nonabelian/su2_coupling_core.hpp",
                    "pyqed/linalg/davidson.hpp",
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


extensions = _extensions_to_build()
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


# Project metadata lives in pyproject.toml. This hook builds the required qchem
# accelerators plus explicitly requested optional extension groups.
setup(
    cmdclass={
        "build_ext": _LinkedBuildExt,
        "build_py": _CleanBuildPy,
        "install_egg_info": _CleanInstallEggInfo,
    },
    ext_modules=extensions,
)
