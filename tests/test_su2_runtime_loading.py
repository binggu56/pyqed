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


@pytest.mark.parametrize('reverse_imports', [False, True])
def test_installed_openmp_kernels_coexist_in_fresh_process(reverse_imports):
    """Exercise a real OpenMP team after loading integral and SU(2) kernels.

    Keep this in a subprocess: mismatched runtimes can segfault in a barrier
    even when each extension imports and its serial contractions work.
    """
    import importlib.util
    import os
    import subprocess
    import sys
    import textwrap

    modules = ['pyqed.qchem._integrals_cpp',
               'pyqed.qchem._casscf_cpp',
               'pyqed.mps.nonabelian._su2_kernel']
    for module in modules:
        if importlib.util.find_spec(module) is None:
            pytest.skip(f'{module} is not built')
    if reverse_imports:
        modules.reverse()
    script = f'import importlib\nfor name in {modules!r}: importlib.import_module(name)\n'
    script += textwrap.dedent('''
        import numpy as np
        from pyqed.qchem import _casscf_cpp
        # 96 rows exceeds the parallel kernel's 64-row threshold.
        rows = np.tile(np.array([[2, 0, 0]], dtype=np.uint64), (96, 1))
        refs = np.array([[1, 0, 0]], dtype=np.uint64)
        ci = np.ones(1)
        h1 = np.ones((2, 2))
        offsets = np.array([0, 1], dtype=np.intp)
        indices = np.array([0], dtype=np.int32)
        groups = np.zeros(96, dtype=np.int32)
        for eri in (np.zeros((2, 2, 2, 2)), np.zeros((1, 2, 2))):
            for workers in (1, 2, 4):
                for repeat in range(3):
                    result = _casscf_cpp.caspt2_direct_couplings_words(
                        rows, refs, ci, h1, eri, 2,
                        offsets, indices, groups, workers)
                    np.testing.assert_allclose(result, 1., atol=1e-14)
    ''')
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1]))
    for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                     'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[variable] = '1'
    completed = subprocess.run([sys.executable, '-X', 'faulthandler', '-c', script],
                               env=env, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
