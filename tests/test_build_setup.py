import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import setuptools
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _setup_namespace(monkeypatch):
    monkeypatch.setenv("PYQED_BUILD_EXTENSIONS", "0")
    monkeypatch.setattr(setuptools, "setup", lambda **_kwargs: None)
    return runpy.run_path(str(ROOT / "setup.py"))


def _cpp_davidson_namespace(monkeypatch):
    monkeypatch.setenv("PYQED_MPS_DISABLE_CPP_DAVIDSON", "1")
    return runpy.run_path(str(ROOT / "pyqed" / "mps" / "cpp_davidson.py"))


def test_darwin_openmp_links_the_selected_runtime_by_path(monkeypatch, tmp_path):
    prefix = tmp_path / "libomp"
    (prefix / "include").mkdir(parents=True)
    (prefix / "lib").mkdir()
    (prefix / "include" / "omp.h").touch()
    runtime = prefix / "lib" / "libomp.dylib"
    runtime.touch()

    namespace = _setup_namespace(monkeypatch)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("PYQED_MPS_OPENMP", "1")
    monkeypatch.setenv("PYQED_OPENMP_PREFIX", str(prefix))

    compile_args, link_args = namespace["_mps_openmp_flags"]()

    assert "-Xpreprocessor" in compile_args
    assert "-fopenmp" in compile_args
    assert str(runtime) in link_args
    assert "-lomp" not in link_args
    assert not any(arg.startswith("-L") for arg in link_args)


def test_darwin_openmp_prefers_homebrew_to_conda(monkeypatch, tmp_path):
    conda = tmp_path / "conda"
    homebrew = Path("/opt/homebrew/opt/libomp")
    available = {
        homebrew / "include" / "omp.h",
        homebrew / "lib" / "libomp.dylib",
        conda / "include" / "omp.h",
        conda / "lib" / "libomp.dylib",
    }

    namespace = _setup_namespace(monkeypatch)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(sys, "prefix", str(conda))
    monkeypatch.delenv("PYQED_OPENMP_PREFIX", raising=False)
    monkeypatch.setenv("CONDA_PREFIX", str(conda))
    monkeypatch.setattr(Path, "exists", lambda path: path in available)

    _compile_args, link_args = namespace["_mps_openmp_flags"]()

    assert str(homebrew / "lib" / "libomp.dylib") in link_args
    assert str(conda / "lib" / "libomp.dylib") not in link_args


def test_cpp_davidson_links_the_selected_runtime_by_path(monkeypatch, tmp_path):
    prefix = tmp_path / "libomp"
    (prefix / "include").mkdir(parents=True)
    (prefix / "lib").mkdir()
    (prefix / "include" / "omp.h").touch()
    runtime = prefix / "lib" / "libomp.dylib"
    runtime.touch()

    namespace = _cpp_davidson_namespace(monkeypatch)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("PYQED_MPS_OPENMP", "1")
    monkeypatch.setenv("PYQED_OPENMP_PREFIX", str(prefix))

    compile_args, link_args, signature = namespace["_openmp_build_setup"]()

    assert "-fopenmp" in compile_args
    assert str(runtime) in link_args
    assert "-lomp" not in link_args
    assert not any(arg.startswith("-L") for arg in link_args)
    assert signature == str(prefix)


def test_cpp_davidson_prefers_homebrew_to_conda(monkeypatch, tmp_path):
    conda = tmp_path / "conda"
    homebrew = Path("/opt/homebrew/opt/libomp")
    available = {
        homebrew / "include" / "omp.h",
        homebrew / "lib" / "libomp.dylib",
        conda / "include" / "omp.h",
        conda / "lib" / "libomp.dylib",
    }

    namespace = _cpp_davidson_namespace(monkeypatch)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(sys, "prefix", str(conda))
    monkeypatch.delenv("PYQED_OPENMP_PREFIX", raising=False)
    monkeypatch.setenv("CONDA_PREFIX", str(conda))
    monkeypatch.setattr(Path, "exists", lambda path: path in available)

    _compile_args, link_args, signature = namespace["_openmp_build_setup"]()

    assert str(homebrew / "lib" / "libomp.dylib") in link_args
    assert str(conda / "lib" / "libomp.dylib") not in link_args
    assert signature == str(homebrew)


def test_openmp_can_be_disabled(monkeypatch):
    namespace = _setup_namespace(monkeypatch)
    monkeypatch.setenv("PYQED_MPS_OPENMP", "0")

    assert namespace["_mps_openmp_flags"]() == ([], [])


def test_qchem_extensions_are_required_by_default(monkeypatch):
    namespace = _setup_namespace(monkeypatch)
    monkeypatch.delenv("PYQED_BUILD_EXTENSIONS")
    monkeypatch.delenv("PYQED_EXTENSION_GROUPS", raising=False)

    extensions = namespace["_extensions_to_build"]()

    assert {extension.name for extension in extensions} == {
        "pyqed.qchem._basis_cy",
        "pyqed.qchem._casscf_cpp",
        "pyqed.qchem._gdf_cpp",
        "pyqed.qchem._integrals_cpp",
        "pyqed.qchem._rys_cy",
    }
    assert all(not extension.optional for extension in extensions)


def test_reference_only_install_can_disable_extensions(monkeypatch):
    namespace = _setup_namespace(monkeypatch)

    assert namespace["_extensions_to_build"]() == []


def test_extra_extension_groups_keep_qchem_required(monkeypatch):
    namespace = _setup_namespace(monkeypatch)
    monkeypatch.setenv("PYQED_BUILD_EXTENSIONS", "1")
    monkeypatch.setenv("PYQED_EXTENSION_GROUPS", "heom,mps")

    extensions = namespace["_extensions_to_build"]()

    assert {extension.name for extension in extensions} == {
        "pyqed.qchem._basis_cy",
        "pyqed.qchem._casscf_cpp",
        "pyqed.qchem._gdf_cpp",
        "pyqed.qchem._integrals_cpp",
        "pyqed.qchem._rys_cy",
        "pyqed.heom._heom_cpp",
        "pyqed.mps.nonabelian._su2_kernel",
    }
    assert all(
        extension.optional == (not extension.name.startswith("pyqed.qchem."))
        for extension in extensions
    )


def test_extension_switch_rejects_typos(monkeypatch):
    namespace = _setup_namespace(monkeypatch)
    monkeypatch.setenv("PYQED_BUILD_EXTENSIONS", "maybe")

    with pytest.raises(ValueError, match="explicit true or false"):
        namespace["_extensions_to_build"]()


def test_sync_conflict_package_data_is_excluded(monkeypatch):
    namespace = _setup_namespace(monkeypatch)

    assert namespace["_is_sync_conflict_copy"](
        "pyqed/qchem/kernel-gugroup's Mac Pro.hpp"
    )
    assert not namespace["_is_sync_conflict_copy"]("pyqed/qchem/kernel.hpp")


def test_darwin_cpp_headers_follow_the_active_sdk(monkeypatch, tmp_path):
    namespace = _setup_namespace(monkeypatch)
    sdk = tmp_path / "MacOSX.sdk"
    headers = sdk / "usr" / "include" / "c++" / "v1"
    headers.mkdir(parents=True)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.delenv("SDKROOT", raising=False)
    monkeypatch.setattr(
        namespace["subprocess"],
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=f"{sdk}\n"),
    )

    assert str(headers) in namespace["_cpp_include_dirs"](
        SimpleNamespace(get_include=lambda: "numpy/include")
    )
