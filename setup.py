#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

import os
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_py import build_py as _build_py


def _is_sync_conflict_copy(module_path):
    """Exclude OneDrive conflict copies without deleting a developer's files."""
    filename = Path(module_path).name.lower()
    return "gugroup" in filename or "bing" in filename and "mac" in filename


_EXCLUDED_LEGACY_MODULES = {
    "pyqed/dvr/sd.py",
    "pyqed/gw/dmft.py",
    "pyqed/integral.py",
    "pyqed/models/ShinMetiuTBE.py",
    "pyqed/mps/results.py",
    "pyqed/namd/eckart.py",
    "pyqed/namd/gmat.py",
    "pyqed/qchem/dvr/sd.py",
}


def _is_excluded_legacy_module(module_path):
    normalized = Path(module_path).as_posix()
    return any(normalized.endswith(path) for path in _EXCLUDED_LEGACY_MODULES)


class _CleanBuildPy(_build_py):
    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        return [
            module
            for module in modules
            if not _is_sync_conflict_copy(module[2])
            and not _is_excluded_legacy_module(module[2])
        ]


def _optional_extensions():
    enabled = os.environ.get("PYQED_BUILD_EXTENSIONS", "0")
    if enabled.strip().lower() not in {"1", "true", "yes", "on"}:
        return []

    try:
        import numpy as np
    except Exception:
        return []

    return [
        Extension(
            "pyqed.qchem._basis_cy",
            ["pyqed/qchem/_basis_cy.c"],
            include_dirs=[np.get_include()],
            optional=True,
        ),
        Extension(
            "pyqed.qchem._rys_cy",
            ["pyqed/qchem/_rys_cy.c"],
            include_dirs=[np.get_include()],
            optional=True,
        ),
    ]


# Project metadata lives in pyproject.toml.  This small compatibility hook is
# retained solely for the optional native extensions.
setup(
    cmdclass={"build_py": _CleanBuildPy},
    ext_modules=_optional_extensions(),
)
