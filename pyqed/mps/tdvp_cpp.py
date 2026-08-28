"""Optional C++ kernels for dense-layout TDVP local evolution."""

from __future__ import annotations

import importlib.util
import os
import shlex
import subprocess
import sys
import sysconfig
from pathlib import Path

CPP_TDVP_AVAILABLE = False
CPP_TDVP_BUILD_ERROR = None
CPP_TDVP_HAS_BLAS = False
site_lanczos = None
site_lanczos_sum = None
two_site_lanczos = None
two_site_lanczos_sum = None
bond_lanczos = None
bond_lanczos_sum = None
one_site_lanczos_sum_sweep = None
reset_kernel_stats = None
kernel_stats = None


def _disabled(value):
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _enabled(value):
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def _load_extension(path):
    spec = importlib.util.spec_from_file_location("pyqed.mps._cpp_tdvp", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _command_output(cmd):
    try:
        return subprocess.check_output(
            cmd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def _darwin_compile_setup():
    if sys.platform != "darwin":
        return None, []
    cxx = _command_output(["xcrun", "--find", "clang++"])
    sdk = _command_output(["xcrun", "--show-sdk-path"])
    flags = []
    if sdk:
        flags.extend(["-isysroot", sdk])
        libcxx_include = Path(sdk) / "usr" / "include" / "c++" / "v1"
        if libcxx_include.exists():
            flags.extend(["-isystem", str(libcxx_include)])
    return cxx, flags


def _compile_extension():
    global CPP_TDVP_BUILD_ERROR

    try:
        import numpy as np
        import pybind11
    except Exception as exc:
        CPP_TDVP_BUILD_ERROR = f"missing build dependency: {exc}"
        return None

    source = Path(__file__).with_name("tdvp_kernels.cpp")
    if not source.exists():
        CPP_TDVP_BUILD_ERROR = f"source file not found: {source}"
        return None

    build_dir = Path(os.environ.get("PYQED_MPS_CPP_BUILD", "/private/tmp/pyqed-mps-cpp"))
    build_dir.mkdir(parents=True, exist_ok=True)
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    ext_path = build_dir / ("_cpp_tdvp" + suffix)
    stamp_path = build_dir / "_cpp_tdvp.stamp"
    fail_stamp_path = build_dir / "_cpp_tdvp.failed"
    source_mtime = str(source.stat().st_mtime_ns)
    compile_signature = "|".join(
        [
            source.name,
            source_mtime,
            sys.version.split()[0],
            sysconfig.get_config_var("CXX") or "",
            os.environ.get("CXX", ""),
        ]
    )
    force_rebuild = _enabled(os.environ.get("PYQED_MPS_FORCE_CPP_TDVP_REBUILD", "0"))
    if ext_path.exists() and stamp_path.exists():
        try:
            if stamp_path.read_text().strip() == compile_signature:
                return _load_extension(ext_path)
        except Exception:
            pass
    if not force_rebuild and fail_stamp_path.exists():
        try:
            lines = fail_stamp_path.read_text().splitlines()
            if lines and lines[0].strip() == compile_signature:
                CPP_TDVP_BUILD_ERROR = "\n".join(lines[1:])
                return None
        except Exception:
            pass

    darwin_cxx, darwin_flags = _darwin_compile_setup()
    cxx = os.environ.get("CXX") or darwin_cxx or sysconfig.get_config_var("CXX") or "c++"
    cmd = shlex.split(cxx)
    cmd.extend(
        [
            "-O3",
            "-std=c++17",
            "-shared",
            "-fPIC",
            *darwin_flags,
            "-I" + sysconfig.get_paths()["include"],
            "-I" + pybind11.get_include(),
            "-I" + np.get_include(),
            str(source),
            "-o",
            str(ext_path),
        ]
    )
    if sys.platform == "darwin":
        cmd.extend(["-undefined", "dynamic_lookup", "-framework", "Accelerate"])

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        stamp_path.write_text(compile_signature)
        try:
            fail_stamp_path.unlink()
        except FileNotFoundError:
            pass
        CPP_TDVP_BUILD_ERROR = None
        return _load_extension(ext_path)
    except Exception as exc:
        stderr = getattr(exc, "stderr", None)
        stdout = getattr(exc, "stdout", None)
        details = "\n".join(
            part.strip()
            for part in (
                f"command: {' '.join(shlex.quote(str(x)) for x in cmd)}",
                str(exc),
                stdout or "",
                stderr or "",
            )
            if part and str(part).strip()
        )
        CPP_TDVP_BUILD_ERROR = details
        try:
            fail_stamp_path.write_text(compile_signature + "\n" + details)
        except Exception:
            pass
        return None


def _initialize():
    global CPP_TDVP_AVAILABLE
    global CPP_TDVP_HAS_BLAS
    global site_lanczos
    global site_lanczos_sum
    global two_site_lanczos
    global two_site_lanczos_sum
    global bond_lanczos
    global bond_lanczos_sum
    global one_site_lanczos_sum_sweep
    global reset_kernel_stats
    global kernel_stats

    if _disabled(os.environ.get("PYQED_MPS_DISABLE_CPP_TDVP", "0")):
        return
    source = Path(__file__).with_name("tdvp_kernels.cpp")
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    packaged = Path(__file__).with_name("_cpp_tdvp" + suffix)
    source_is_newer = source.exists() and (
        not packaged.exists() or source.stat().st_mtime_ns > packaged.stat().st_mtime_ns
    )
    if source_is_newer and _enabled(
        os.environ.get("PYQED_MPS_AUTO_CPP_TDVP", "1")
    ):
        module = _compile_extension()
        if module is None:
            return
    else:
        try:
            from . import _cpp_tdvp as module
        except Exception:
            if not _enabled(os.environ.get("PYQED_MPS_AUTO_CPP_TDVP", "1")):
                return
            module = _compile_extension()
            if module is None:
                return

    site_lanczos = getattr(module, "site_lanczos", None)
    site_lanczos_sum = getattr(module, "site_lanczos_sum", None)
    two_site_lanczos = getattr(module, "two_site_lanczos", None)
    two_site_lanczos_sum = getattr(module, "two_site_lanczos_sum", None)
    bond_lanczos = getattr(module, "bond_lanczos", None)
    bond_lanczos_sum = getattr(module, "bond_lanczos_sum", None)
    one_site_lanczos_sum_sweep = getattr(
        module, "one_site_lanczos_sum_sweep", None
    )
    reset_kernel_stats = getattr(module, "reset_kernel_stats", None)
    kernel_stats = getattr(module, "kernel_stats", None)
    CPP_TDVP_HAS_BLAS = bool(getattr(module, "HAS_BLAS", False))
    CPP_TDVP_AVAILABLE = (
        site_lanczos is not None
        and two_site_lanczos is not None
        and bond_lanczos is not None
    )


_initialize()
