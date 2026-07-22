"""Optional C++ kernels for SU(2)-NARG reduced tensor products."""

from __future__ import annotations

import importlib.util
import os
import shlex
import subprocess
import sys
import sysconfig
from pathlib import Path


CPP_PRODUCT_AVAILABLE = False
CPP_ANGULAR_AVAILABLE = False
CPP_PRODUCT_BUILD_ERROR = None
reduced_product_block_sum = None
reduced_product_block_sum_batch = None
product_tensor_pair_entries = None
product_tensor_group_indices = None
accumulate_bilinear = None


def _disabled(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _enabled(value) -> bool:
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def _load_extension(path: Path):
    spec = importlib.util.spec_from_file_location("pyqed.narg.qchem._su2_native", path)
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
    global CPP_PRODUCT_BUILD_ERROR

    try:
        import numpy as np
        import pybind11
    except Exception as exc:
        CPP_PRODUCT_BUILD_ERROR = f"missing build dependency: {exc}"
        return None

    source = Path(__file__).with_name("su2_native.cpp")
    if not source.exists():
        CPP_PRODUCT_BUILD_ERROR = f"source file not found: {source}"
        return None

    build_dir = Path(os.environ.get("SU2_NARG_CPP_BUILD", "/private/tmp/su2-narg-cpp"))
    build_dir.mkdir(parents=True, exist_ok=True)
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    ext_path = build_dir / ("_su2_native" + suffix)
    stamp_path = build_dir / "_su2_native.stamp"
    fail_stamp_path = build_dir / "_su2_native.failed"
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
    force_rebuild = _enabled(os.environ.get("SU2_NARG_FORCE_CPP_PRODUCT_REBUILD", "0"))
    if ext_path.exists() and stamp_path.exists():
        try:
            if stamp_path.read_text().strip() == compile_signature:
                return _load_extension(ext_path)
        except Exception:
            pass
    if not force_rebuild and fail_stamp_path.exists():
        try:
            if fail_stamp_path.read_text().splitlines()[0].strip() == compile_signature:
                CPP_PRODUCT_BUILD_ERROR = "\n".join(
                    fail_stamp_path.read_text().splitlines()[1:]
                )
                return None
        except Exception:
            pass

    darwin_cxx, darwin_flags = _darwin_compile_setup()
    cxx = (
        os.environ.get("CXX")
        or darwin_cxx
        or sysconfig.get_config_var("CXX")
        or "c++"
    )
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
        CPP_PRODUCT_BUILD_ERROR = None
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
        CPP_PRODUCT_BUILD_ERROR = details
        try:
            fail_stamp_path.write_text(compile_signature + "\n" + details)
        except Exception:
            pass
        return None


def _initialize():
    global CPP_PRODUCT_AVAILABLE
    global CPP_ANGULAR_AVAILABLE
    global reduced_product_block_sum
    global reduced_product_block_sum_batch
    global product_tensor_pair_entries
    global product_tensor_group_indices
    global accumulate_bilinear

    if _disabled(os.environ.get("SU2_NARG_DISABLE_CPP_PRODUCT", "0")):
        return
    module = _compile_extension()
    if module is None:
        return
    reduced_product_block_sum = getattr(module, "reduced_product_block_sum", None)
    reduced_product_block_sum_batch = getattr(
        module,
        "reduced_product_block_sum_batch",
        None,
    )
    product_tensor_pair_entries = getattr(module, "product_tensor_pair_entries", None)
    product_tensor_group_indices = getattr(module, "product_tensor_group_indices", None)
    accumulate_bilinear = getattr(module, "accumulate_bilinear", None)
    CPP_PRODUCT_AVAILABLE = reduced_product_block_sum is not None
    CPP_ANGULAR_AVAILABLE = (
        product_tensor_pair_entries is not None
        and product_tensor_group_indices is not None
    )


_initialize()
