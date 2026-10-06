"""Regression checks for incompatible OpenMP runtimes and build linkage."""

import builtins
from pathlib import Path
import runpy

import numpy as np
import pytest

from pyqed.qchem.dmrg.backends.reduced import _build_su2_moving_environment


def test_broken_su2_extension_does_not_silently_select_fallback(monkeypatch):
    original = builtins.__import__

    def importing(name, *args, **kwargs):
        if name == "pyqed.mps.nonabelian._su2_kernel":
            raise ImportError("Symbol not found: ___kmpc_dispatch_deinit")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importing)
    with pytest.raises(ImportError, match="refusing to silently change") as caught:
        _build_su2_moving_environment(
            np.eye(2), None, n_elec=2, spin=0, ecore=0,
            orb_sym=None, cutoff=1e-12,
        )
    assert "___kmpc_dispatch_deinit" in str(caught.value.__cause__)


def test_macos_build_pins_selected_openmp_runtime(monkeypatch):
    import setuptools
    from setuptools.command.build_ext import build_ext

    monkeypatch.setenv("PYQED_BUILD_EXTENSIONS", "0")
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: None)
    namespace = runpy.run_path(str(Path(__file__).parents[1] / "setup.py"))
    monkeypatch.setattr(namespace["sys"], "platform", "darwin")
    monkeypatch.setattr(build_ext, "build_extension", lambda *args: None)
    calls = []
    monkeypatch.setattr(namespace["subprocess"], "run", lambda *a, **kw: calls.append((a, kw)))
    command = namespace["_LinkedBuildExt"](setuptools.Distribution())
    monkeypatch.setattr(command, "get_ext_fullpath", lambda name: "/tmp/kernel.so")
    extension = setuptools.Extension(
        "test.kernel", [],
        extra_link_args=["/selected/lib/libomp.dylib", "-Wl,-rpath,/selected/lib"],
    )
    command.build_extension(extension)
    assert calls == [(([
        "install_name_tool", "-change", "@rpath/libomp.dylib",
        "/selected/lib/libomp.dylib", "/tmp/kernel.so",
    ],), {"check": True})]
