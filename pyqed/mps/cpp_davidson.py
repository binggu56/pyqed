"""Optional C++ Davidson backends for dense and packed-Abelian local solves."""

from __future__ import annotations

import importlib.util
import os
import shlex
import subprocess
import sys
import sysconfig
from pathlib import Path

CPP_DAVIDSON_AVAILABLE = False
CPP_DAVIDSON_BUILD_ERROR = None
block_table_davidson = None
block_table_matvec = None
dense_two_site_matvec = None
dense_coarse_grain_mpo = None
dense_coarse_grain_mps = None
dense_environment_update_left = None
dense_environment_update_right = None
DenseDavidsonWorkspace = None
DenseSweepWorkspace = None
lapack_svd = None
lapack_qr = None
abelian_two_site_svd_from_permuted_data = None
abelian_split_two_site_svd_data = None
abelian_split_flat_two_site_svd_data = None
abelian_merge_adjacent_site_tensors_data = None
abelian_merge_normalize_adjacent_site_tensors_data = None
abelian_merge_normalize_flatten_adjacent_site_tensors_data = None
abelian_flatten_data_to_layout = None
abelian_block_data_norm = None
abelian_scale_block_data = None
abelian_left_environment_advance_data = None
abelian_right_environment_advance_data = None
abelian_tdvp_site_heff_data = None
abelian_tdvp_bond_heff_data = None
abelian_tdvp_two_site_lanczos = None
AbelianTDVPSiteHeffPlan = None
AbelianTDVPBondHeffPlan = None
AbelianTDVPTwoSiteHeffPlan = None
AbelianEnvironmentAdvancePlan = None
BlockTable = None
RenormalizedTable = None
SparseRenormalizedTable = None
GroupedRenormalizedTable = None
GroupedFactorizedTable = None
CompactPlan = None
RawPayloadBuilder = None
RawRoutePlan = None
NamedRawPayloadPlan = None
PlannedDirectPayloadPlan = None
DirectFamilyRoutePlan = None
MovingEnvironment = None
SU2ParentBlockTable = None
SU2FactorizedFamilyTable = None
SU2PackedFactorizedFamilyTable = None
direct_left_stack = None
direct_right_stack = None
identity_channel_left_stack = None
identity_channel_right_stack = None
build_direct_family_payload = None
build_direct_family_payload_fastkeys = None
dict_resolve_many = None
dict_resolve_values_many = None
dict_resolve_current_ids_many = None
dict_put_many_values = None
dict_put_many_packed = None
packed_site_operator_from_left_payload = None
packed_site_operator_from_right_payload = None
packed_initial_left_environment_payload = None
packed_initial_right_environment_payload = None
contextual_left_prefix_closure = None
contextual_right_suffix_closure = None
contextual_prepare_boundary_build_wave = None
contextual_execute_boundary_build_wave = None
contextual_execute_boundary_build_wave_packed = None
contextual_probe_local_table_cache = None
contextual_fill_local_table_cache_misses = None
contextual_partition_pending_rows = None
packed_left_boundary_advance_payload = None
packed_left_identity_boundary_advance_payload = None
packed_right_boundary_advance_payload = None
packed_right_identity_boundary_advance_payload = None
contextual_left_finalize_batch = None
contextual_right_finalize_batch = None
contextual_left_prepare_local_table_batch = None
contextual_right_prepare_local_table_batch = None
contextual_left_finalize_prebuilt_batch = None
contextual_right_finalize_prebuilt_batch = None
build_spatial_qchem_family_entries = None
build_spatial_qchem_family_term_maps = None
build_spatial_qchem_family_mpos = None
build_spatial_block2_carrier_mpo = None
build_spatial_qchem_block2_setup = None
pack_rank_coupled_factor_routes = None
rank_coupled_reduced_actions = None


def _disabled(value):
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _enabled(value):
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def _load_extension(path):
    spec = importlib.util.spec_from_file_location("pyqed.mps._cpp_davidson", path)
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
    global CPP_DAVIDSON_BUILD_ERROR

    try:
        import numpy as np
        import pybind11
    except Exception as exc:
        CPP_DAVIDSON_BUILD_ERROR = f"missing build dependency: {exc}"
        return None

    source = Path(__file__).with_name("davidson.cpp")
    core_header = Path(__file__).with_name("dmrg_linalg_core.hpp")
    if not source.exists():
        CPP_DAVIDSON_BUILD_ERROR = f"source file not found: {source}"
        return None
    build_dir = Path(
        os.environ.get("PYQED_MPS_CPP_BUILD", "/private/tmp/pyqed-mps-cpp")
    )
    build_dir.mkdir(parents=True, exist_ok=True)
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    ext_path = build_dir / ("_cpp_davidson" + suffix)
    stamp_path = build_dir / "_cpp_davidson.stamp"
    fail_stamp_path = build_dir / "_cpp_davidson.failed"
    source_mtime = str(source.stat().st_mtime_ns)
    core_mtime = str(core_header.stat().st_mtime_ns) if core_header.exists() else "missing"
    compile_signature = "|".join(
        [
            source.name,
            source_mtime,
            core_header.name,
            core_mtime,
            sys.version.split()[0],
            sysconfig.get_config_var("CXX") or "",
            os.environ.get("CXX", ""),
        ]
    )
    force_rebuild = _enabled(
        os.environ.get("PYQED_MPS_FORCE_CPP_DAVIDSON_REBUILD", "0")
    )
    if ext_path.exists() and stamp_path.exists():
        try:
            if stamp_path.read_text().strip() == compile_signature:
                return _load_extension(ext_path)
        except Exception:
            pass
    if not force_rebuild and fail_stamp_path.exists():
        try:
            if fail_stamp_path.read_text().splitlines()[0].strip() == compile_signature:
                CPP_DAVIDSON_BUILD_ERROR = "\n".join(
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
        CPP_DAVIDSON_BUILD_ERROR = None
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
        CPP_DAVIDSON_BUILD_ERROR = details
        try:
            fail_stamp_path.write_text(compile_signature + "\n" + details)
        except Exception:
            pass
        return None


def _initialize():
    global CPP_DAVIDSON_AVAILABLE
    global CPP_DAVIDSON_BUILD_ERROR
    global block_table_davidson
    global block_table_matvec
    global dense_two_site_matvec
    global dense_coarse_grain_mpo
    global dense_coarse_grain_mps
    global dense_environment_update_left
    global dense_environment_update_right
    global DenseDavidsonWorkspace
    global DenseSweepWorkspace
    global lapack_svd
    global lapack_qr
    global abelian_two_site_svd_from_permuted_data
    global abelian_split_two_site_svd_data
    global abelian_split_flat_two_site_svd_data
    global abelian_merge_adjacent_site_tensors_data
    global abelian_merge_normalize_adjacent_site_tensors_data
    global abelian_merge_normalize_flatten_adjacent_site_tensors_data
    global abelian_flatten_data_to_layout
    global abelian_block_data_norm
    global abelian_scale_block_data
    global abelian_left_environment_advance_data
    global abelian_right_environment_advance_data
    global abelian_tdvp_site_heff_data
    global abelian_tdvp_bond_heff_data
    global abelian_tdvp_two_site_lanczos
    global AbelianTDVPSiteHeffPlan
    global AbelianTDVPBondHeffPlan
    global AbelianTDVPTwoSiteHeffPlan
    global AbelianEnvironmentAdvancePlan
    global BlockTable
    global RenormalizedTable
    global SparseRenormalizedTable
    global GroupedRenormalizedTable
    global GroupedFactorizedTable
    global CompactPlan
    global RawPayloadBuilder
    global RawRoutePlan
    global NamedRawPayloadPlan
    global PlannedDirectPayloadPlan
    global DirectFamilyRoutePlan
    global MovingEnvironment
    global SU2ParentBlockTable
    global SU2FactorizedFamilyTable
    global SU2PackedFactorizedFamilyTable
    global direct_left_stack
    global direct_right_stack
    global identity_channel_left_stack
    global identity_channel_right_stack
    global build_direct_family_payload
    global build_direct_family_payload_fastkeys
    global dict_resolve_many
    global dict_resolve_values_many
    global dict_resolve_current_ids_many
    global dict_put_many_values
    global dict_put_many_packed
    global packed_site_operator_from_left_payload
    global packed_site_operator_from_right_payload
    global packed_initial_left_environment_payload
    global packed_initial_right_environment_payload
    global contextual_left_prefix_closure
    global contextual_right_suffix_closure
    global contextual_prepare_boundary_build_wave
    global contextual_execute_boundary_build_wave
    global contextual_execute_boundary_build_wave_packed
    global contextual_probe_local_table_cache
    global contextual_fill_local_table_cache_misses
    global contextual_partition_pending_rows
    global packed_left_boundary_advance_payload
    global packed_left_identity_boundary_advance_payload
    global packed_right_boundary_advance_payload
    global packed_right_identity_boundary_advance_payload
    global contextual_left_finalize_batch
    global contextual_right_finalize_batch
    global contextual_left_prepare_local_table_batch
    global contextual_right_prepare_local_table_batch
    global contextual_left_finalize_prebuilt_batch
    global contextual_right_finalize_prebuilt_batch
    global build_spatial_qchem_family_entries
    global build_spatial_qchem_family_term_maps
    global build_spatial_qchem_family_mpos
    global build_spatial_block2_carrier_mpo
    global build_spatial_qchem_block2_setup
    global pack_rank_coupled_factor_routes
    global rank_coupled_reduced_actions

    if _disabled(os.environ.get("PYQED_MPS_DISABLE_CPP_DAVIDSON", "0")):
        return
    try:
        from . import _cpp_davidson as module
    except Exception:
        if not _enabled(os.environ.get("PYQED_MPS_AUTO_CPP_DAVIDSON", "1")):
            CPP_DAVIDSON_BUILD_ERROR = "auto build disabled"
            return
        module = _compile_extension()
        if module is None:
            return
    block_table_davidson = module.block_table_davidson
    block_table_matvec = getattr(module, "block_table_matvec", None)
    dense_two_site_matvec = getattr(module, "dense_two_site_matvec", None)
    dense_coarse_grain_mpo = getattr(module, "dense_coarse_grain_mpo", None)
    dense_coarse_grain_mps = getattr(module, "dense_coarse_grain_mps", None)
    dense_environment_update_left = getattr(
        module,
        "dense_environment_update_left",
        None,
    )
    dense_environment_update_right = getattr(
        module,
        "dense_environment_update_right",
        None,
    )
    DenseDavidsonWorkspace = getattr(module, "DenseDavidsonWorkspace", None)
    DenseSweepWorkspace = getattr(module, "DenseSweepWorkspace", None)
    lapack_svd = getattr(module, "lapack_svd", None)
    lapack_qr = getattr(module, "lapack_qr", None)
    abelian_two_site_svd_from_permuted_data = getattr(
        module,
        "abelian_two_site_svd_from_permuted_data",
        None,
    )
    abelian_split_two_site_svd_data = getattr(
        module,
        "abelian_split_two_site_svd_data",
        None,
    )
    abelian_split_flat_two_site_svd_data = getattr(
        module,
        "abelian_split_flat_two_site_svd_data",
        None,
    )
    abelian_merge_adjacent_site_tensors_data = getattr(
        module,
        "abelian_merge_adjacent_site_tensors_data",
        None,
    )
    abelian_merge_normalize_adjacent_site_tensors_data = getattr(
        module,
        "abelian_merge_normalize_adjacent_site_tensors_data",
        None,
    )
    abelian_merge_normalize_flatten_adjacent_site_tensors_data = getattr(
        module,
        "abelian_merge_normalize_flatten_adjacent_site_tensors_data",
        None,
    )
    abelian_flatten_data_to_layout = getattr(
        module,
        "abelian_flatten_data_to_layout",
        None,
    )
    abelian_block_data_norm = getattr(
        module,
        "abelian_block_data_norm",
        None,
    )
    abelian_scale_block_data = getattr(
        module,
        "abelian_scale_block_data",
        None,
    )
    abelian_left_environment_advance_data = getattr(
        module,
        "abelian_left_environment_advance_data",
        None,
    )
    abelian_right_environment_advance_data = getattr(
        module,
        "abelian_right_environment_advance_data",
        None,
    )
    abelian_tdvp_site_heff_data = getattr(
        module,
        "abelian_tdvp_site_heff_data",
        None,
    )
    abelian_tdvp_bond_heff_data = getattr(
        module,
        "abelian_tdvp_bond_heff_data",
        None,
    )
    abelian_tdvp_two_site_lanczos = getattr(
        module,
        "abelian_tdvp_two_site_lanczos",
        None,
    )
    AbelianTDVPSiteHeffPlan = getattr(
        module,
        "AbelianTDVPSiteHeffPlan",
        None,
    )
    AbelianTDVPBondHeffPlan = getattr(
        module,
        "AbelianTDVPBondHeffPlan",
        None,
    )
    AbelianTDVPTwoSiteHeffPlan = getattr(
        module,
        "AbelianTDVPTwoSiteHeffPlan",
        None,
    )
    AbelianEnvironmentAdvancePlan = getattr(
        module,
        "AbelianEnvironmentAdvancePlan",
        None,
    )
    BlockTable = getattr(module, "BlockTable", None)
    RenormalizedTable = getattr(module, "RenormalizedTable", BlockTable)
    SparseRenormalizedTable = getattr(module, "SparseRenormalizedTable", None)
    GroupedRenormalizedTable = getattr(module, "GroupedRenormalizedTable", None)
    GroupedFactorizedTable = getattr(module, "GroupedFactorizedTable", None)
    CompactPlan = getattr(module, "CompactPlan", None)
    RawPayloadBuilder = getattr(module, "RawPayloadBuilder", None)
    RawRoutePlan = getattr(module, "RawRoutePlan", None)
    NamedRawPayloadPlan = getattr(module, "NamedRawPayloadPlan", None)
    PlannedDirectPayloadPlan = getattr(module, "PlannedDirectPayloadPlan", None)
    DirectFamilyRoutePlan = getattr(
        module,
        "DirectFamilyRoutePlan",
        PlannedDirectPayloadPlan,
    )
    MovingEnvironment = getattr(module, "MovingEnvironment", None)
    direct_left_stack = getattr(module, "direct_left_stack", None)
    direct_right_stack = getattr(module, "direct_right_stack", None)
    identity_channel_left_stack = getattr(
        module,
        "identity_channel_left_stack",
        None,
    )
    identity_channel_right_stack = getattr(
        module,
        "identity_channel_right_stack",
        None,
    )
    build_direct_family_payload = getattr(module, "build_direct_family_payload", None)
    build_direct_family_payload_fastkeys = getattr(
        module,
        "build_direct_family_payload_fastkeys",
        None,
    )
    dict_resolve_many = getattr(
        module,
        "dict_resolve_many",
        None,
    )
    dict_resolve_values_many = getattr(
        module,
        "dict_resolve_values_many",
        None,
    )
    dict_resolve_current_ids_many = getattr(
        module,
        "dict_resolve_current_ids_many",
        None,
    )
    dict_put_many_values = getattr(
        module,
        "dict_put_many_values",
        None,
    )
    dict_put_many_packed = getattr(
        module,
        "dict_put_many_packed",
        None,
    )
    packed_site_operator_from_left_payload = getattr(
        module,
        "packed_site_operator_from_left_payload",
        None,
    )
    packed_site_operator_from_right_payload = getattr(
        module,
        "packed_site_operator_from_right_payload",
        None,
    )
    packed_initial_left_environment_payload = getattr(
        module,
        "packed_initial_left_environment_payload",
        None,
    )
    packed_initial_right_environment_payload = getattr(
        module,
        "packed_initial_right_environment_payload",
        None,
    )
    contextual_left_prefix_closure = getattr(
        module,
        "contextual_left_prefix_closure",
        None,
    )
    contextual_right_suffix_closure = getattr(
        module,
        "contextual_right_suffix_closure",
        None,
    )
    contextual_prepare_boundary_build_wave = getattr(
        module,
        "contextual_prepare_boundary_build_wave",
        None,
    )
    contextual_execute_boundary_build_wave = getattr(
        module,
        "contextual_execute_boundary_build_wave",
        None,
    )
    contextual_execute_boundary_build_wave_packed = getattr(
        module,
        "contextual_execute_boundary_build_wave_packed",
        None,
    )
    contextual_probe_local_table_cache = getattr(
        module,
        "contextual_probe_local_table_cache",
        None,
    )
    contextual_fill_local_table_cache_misses = getattr(
        module,
        "contextual_fill_local_table_cache_misses",
        None,
    )
    contextual_partition_pending_rows = getattr(
        module,
        "contextual_partition_pending_rows",
        None,
    )
    packed_left_boundary_advance_payload = getattr(
        module,
        "packed_left_boundary_advance_payload",
        None,
    )
    packed_left_identity_boundary_advance_payload = getattr(
        module,
        "packed_left_identity_boundary_advance_payload",
        None,
    )
    packed_right_boundary_advance_payload = getattr(
        module,
        "packed_right_boundary_advance_payload",
        None,
    )
    packed_right_identity_boundary_advance_payload = getattr(
        module,
        "packed_right_identity_boundary_advance_payload",
        None,
    )
    contextual_left_finalize_batch = getattr(
        module,
        "contextual_left_finalize_batch",
        None,
    )
    contextual_right_finalize_batch = getattr(
        module,
        "contextual_right_finalize_batch",
        None,
    )
    contextual_left_prepare_local_table_batch = getattr(
        module,
        "contextual_left_prepare_local_table_batch",
        None,
    )
    contextual_right_prepare_local_table_batch = getattr(
        module,
        "contextual_right_prepare_local_table_batch",
        None,
    )
    contextual_left_finalize_prebuilt_batch = getattr(
        module,
        "contextual_left_finalize_prebuilt_batch",
        None,
    )
    contextual_right_finalize_prebuilt_batch = getattr(
        module,
        "contextual_right_finalize_prebuilt_batch",
        None,
    )
    build_spatial_qchem_family_entries = getattr(
        module,
        "build_spatial_qchem_family_entries",
        None,
    )
    build_spatial_qchem_family_term_maps = getattr(
        module,
        "build_spatial_qchem_family_term_maps",
        None,
    )
    build_spatial_qchem_family_mpos = getattr(
        module,
        "build_spatial_qchem_family_mpos",
        None,
    )
    build_spatial_block2_carrier_mpo = getattr(
        module,
        "build_spatial_block2_carrier_mpo",
        None,
    )
    build_spatial_qchem_block2_setup = getattr(
        module,
        "build_spatial_qchem_block2_setup",
        None,
    )
    pack_rank_coupled_factor_routes = getattr(
        module,
        "pack_rank_coupled_factor_routes",
        None,
    )
    rank_coupled_reduced_actions = getattr(
        module,
        "rank_coupled_reduced_actions",
        None,
    )
    SU2ParentBlockTable = getattr(module, "SU2ParentBlockTable", None)
    SU2FactorizedFamilyTable = getattr(
        module,
        "SU2FactorizedFamilyTable",
        None,
    )
    SU2PackedFactorizedFamilyTable = getattr(
        module,
        "SU2PackedFactorizedFamilyTable",
        None,
    )
    CPP_DAVIDSON_AVAILABLE = True


_initialize()
