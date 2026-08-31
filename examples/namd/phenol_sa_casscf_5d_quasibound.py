#!/usr/bin/env python3
"""Compute the bound S1 parent state for the phenol 5D coupled dynamics."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import time

if os.environ.get("PYQED_NO_MATPLOTLIB") == "1":
    plt = None
else:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
import numpy as np

from examples.namd.phenol_sa_casscf_paths import (
    DEFAULT_PHENOL_5D_CHECKPOINT,
    DEFAULT_PHENOL_5D_DATA,
    DEFAULT_PHENOL_5D_OPERATOR_CACHE,
    DEFAULT_PHENOL_5D_RADIAL_CORRECTION,
    PHENOL_5D_PRODUCTION,
)
from examples.namd.phenol_sa_casscf_5d_ftt_ttldr import build_dvrs
from pyqed.ml import (
    CorrectedMatrixField,
    MACE,
    RadialMatrixCorrection,
    ReflectionScalarMLP,
)
from pyqed.models.phenol_coordinates import PhenolReactiveChart
from pyqed.mps import MPS, MPO
from pyqed.mps.cross import tt_cross
from pyqed.mps.functional import FunctionalTT
from pyqed.namd.phenol import _reflection_indices, build_phenol_5d_keo_mpo
from pyqed.mps.tdvp import one_site_tdvp_sum_step


HARTREE_TO_EV = 27.211386245988
HARTREE_TO_WAVENUMBER = 219474.6313632
DEFAULT_OUTPUT = PHENOL_5D_PRODUCTION / "states" / "s1_origin_5d_range_probe"
DEFAULT_SCALAR_KEO_CACHE = (
    PHENOL_5D_PRODUCTION / "cache" / "quasibound_scalar_keo_65x21x23x21x17"
)
DEFAULT_PARENT_POTENTIAL_CACHE = (
    PHENOL_5D_PRODUCTION / "cache" / "quasibound_s1_potential_65x21x23x21x17"
)
DEFAULT_RESIDUAL_POTENTIAL_CACHE = (
    PHENOL_5D_PRODUCTION / "cache" / "quasibound_s1_residual_65x21x23x21x17"
)
DEFAULT_GRID_SHAPE = (65, 21, 23, 21, 17)
DEFAULT_WIDE_BOUNDS = np.asarray(
    (
        (0.75, 3.00),
        (-1.00, 1.00),
        (np.deg2rad(90.0), np.deg2rad(134.0)),
        (-1.00, 1.00),
        (-0.40, 0.40),
    )
)
PACKET_WIDTHS = np.asarray((0.060, 0.13, np.deg2rad(2.8), 0.10, 0.055))


def _shape(value):
    shape = tuple(int(item) for item in str(value).split(","))
    if len(shape) != 5 or any(item < 2 for item in shape):
        raise argparse.ArgumentTypeError("grid shape must contain five integers >= 2")
    return shape


def _file_signature(path):
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def project_electronic_component(factors, state):
    """Return the nuclear MPO diagonal block ``<state|H|state>``."""
    factors = tuple(factors)
    if len(factors) < 2:
        raise ValueError("a vibronic MPO requires nuclear and electronic sites")
    electronic = np.asarray(factors[-1])
    if electronic.shape[1] != 1 or electronic.shape[2] != electronic.shape[3]:
        raise ValueError(
            "the electronic MPO site must be the right boundary of a square operator"
        )
    if not 0 <= int(state) < electronic.shape[2]:
        raise ValueError("electronic state is outside the cached Hamiltonian")
    boundary = electronic[:, 0, int(state), int(state)]
    previous = np.asarray(factors[-2])
    last = np.einsum("lrpq,r->lpq", previous, boundary, optimize=True)
    projected = [np.asarray(factor) for factor in factors[:-2]]
    projected.append(last[:, None, :, :])
    return MPO(projected)


def load_projected_components(directory, state=1):
    directory = Path(directory).expanduser()
    metadata = json.loads((directory / "metadata.json").read_text())
    axes = tuple(np.load(directory / name) for name in metadata["axes"])
    components = []
    for entries in metadata["components"]:
        factors = [
            np.load(directory / entry["path"], mmap_mode="r") for entry in entries
        ]
        components.append(project_electronic_component(factors, state))
    return axes, tuple(components), metadata


def load_parent_potential(directory, state=1):
    """Load only the potential component and select its parent-state diagonal."""
    directory = Path(directory).expanduser()
    metadata = json.loads((directory / "metadata.json").read_text())
    axes = tuple(np.load(directory / name) for name in metadata["axes"])
    entries = metadata["components"][-1]
    factors = [
        np.load(directory / entry["path"], mmap_mode="r") for entry in entries
    ]
    return axes, project_electronic_component(factors, state), metadata


def _scalar_keo_spec(source_metadata, axes, periodic_axes=()):
    source_spec = source_metadata["spec"]
    return {
        "grid_shape": [len(axis) for axis in axes],
        "grid_bounds": [[float(axis[0]), float(axis[-1])] for axis in axes],
        "data": source_spec["data"],
        "cross_max_rank": source_spec["keo_cross_rank"],
        "cross_sweeps": source_spec["keo_cross_sweeps"],
        "cross_rtol": source_spec["keo_cross_rtol"],
        "cross_validation": source_spec["keo_cross_validation"],
        "mpo_max_rank": source_spec["keo_mpo_rank"],
        "seed": source_spec["seed"],
        "split": True,
        "reflection_symmetrized": True,
        "metric_derivative": "dvr-kinetic-complete-procrustes-v2",
        "periodic_axes": [int(axis) for axis in periodic_axes],
    }


def _parent_potential_spec(args, axes):
    source = (
        {"scalar_potential": _file_signature(args.scalar_potential)}
        if args.scalar_potential is not None
        else {
            "checkpoint": _file_signature(args.checkpoint),
            "radial_correction": _file_signature(args.radial_correction),
        }
    )
    return {
        "representation": "centered-scalar-p-gauge-grid-cross-v2",
        **source,
        "electronic_state": int(args.electronic_state),
        "grid_shape": [len(axis) for axis in axes],
        "grid_bounds": [[float(axis[0]), float(axis[-1])] for axis in axes],
        "rank": int(args.potential_tt_rank),
        "sweeps": int(args.potential_cross_sweeps),
        "rtol": float(args.potential_cross_rtol),
        "validation": int(args.potential_cross_validation),
        "start_rank": int(args.potential_cross_start_rank),
        "kick_rank": int(args.potential_cross_kick_rank),
        "seed": int(args.seed),
    }


def add_constant_to_tt(cores, value):
    """Add a scalar offset exactly without asking TT-cross to resolve it."""
    cores = [np.asarray(core) for core in cores]
    if len(cores) == 1:
        result = cores[0].copy()
        result[0, :, 0] += value
        return [result]
    dtype = np.result_type(*cores, value)
    result = [
        np.concatenate(
            (
                np.asarray(cores[0], dtype=dtype),
                np.full((1, cores[0].shape[1], 1), value, dtype=dtype),
            ),
            axis=2,
        )
    ]
    for core in cores[1:-1]:
        left, physical, right = core.shape
        expanded = np.zeros((left + 1, physical, right + 1), dtype=dtype)
        expanded[:left, :, :right] = core
        expanded[-1, :, -1] = 1.0
        result.append(expanded)
    result.append(
        np.concatenate(
            (
                np.asarray(cores[-1], dtype=dtype),
                np.ones((1, cores[-1].shape[1], 1), dtype=dtype),
            ),
            axis=0,
        )
    )
    return result


def tt_cores_to_diagonal_mpo(cores):
    factors = []
    for core in cores:
        left, physical, right = core.shape
        factor = np.zeros((left, right, physical, physical), dtype=core.dtype)
        diagonal = np.arange(physical)
        factor[:, :, diagonal, diagonal] = core.transpose(0, 2, 1)
        factors.append(factor)
    return MPO(factors)


def reflection_pair(potential, reflection_sites=(1, 3), reflection_maps=None):
    """Return half-weighted original/reflected MPOs whose sum is exactly even."""
    original = [np.asarray(factor).copy() for factor in potential.factors]
    reflected = [np.asarray(factor).copy() for factor in potential.factors]
    original[0] *= 0.5
    reflected[0] *= 0.5
    reflection_maps = {} if reflection_maps is None else reflection_maps
    for site in reflection_sites:
        site = int(site)
        mapping = np.asarray(
            reflection_maps.get(site, np.arange(reflected[site].shape[2] - 1, -1, -1)),
            dtype=int,
        )
        reflected[site] = np.take(
            np.take(reflected[site], mapping, axis=2), mapping, axis=3
        )
    return MPO(original), MPO(reflected)


def mpo_diagonal_values(components, indices, *, batch_size=2048):
    """Evaluate a sum of diagonal MPOs at integer product-grid indices."""
    components = tuple(components)
    indices = np.asarray(indices, dtype=int)
    if indices.ndim == 1:
        indices = indices[None, :]
    if indices.ndim != 2 or any(term.L != indices.shape[1] for term in components):
        raise ValueError("indices and MPO components have incompatible dimensions")
    output = np.zeros(len(indices), dtype=float)
    for start in range(0, len(indices), int(batch_size)):
        block = indices[start : start + int(batch_size)]
        values = np.zeros(len(block), dtype=complex)
        for term in components:
            environment = np.ones((len(block), 1, 1), dtype=complex)
            for site, factor in enumerate(term.factors):
                positions = block[:, site]
                local = np.moveaxis(
                    np.asarray(factor)[:, :, positions, positions], -1, 0
                )
                environment = np.einsum(
                    "nab,nbc->nac", environment, local, optimize=True
                )
            values += environment[:, 0, 0]
        output[start : start + len(block)] = values.real
    return output


def sample_mps_indices(state, count, seed):
    """Draw exact Born samples from a finite MPS in right-canonical gauge."""
    state = state.copy().right_canonicalize().normalize()
    rng = np.random.default_rng(seed)
    count = int(count)
    left = np.ones((count, 1), dtype=complex)
    samples = np.empty((count, state.L), dtype=int)
    rows = np.arange(count)
    for site in range(state.L):
        factor = state._get_std_B(site)
        amplitudes = np.einsum("na,apb->npb", left, factor, optimize=True)
        probabilities = np.sum(np.abs(amplitudes) ** 2, axis=2).real
        probabilities = np.maximum(probabilities, 0.0)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        choices = np.sum(
            rng.random(count)[:, None] > np.cumsum(probabilities, axis=1), axis=1
        )
        choices = np.minimum(choices, factor.shape[1] - 1)
        samples[:, site] = choices
        selected = amplitudes[rows, choices]
        norms = np.linalg.norm(selected, axis=1)
        left = selected / norms[:, None]
    return samples


def symmetrize_state(
    state, reflection_sites=(1, 3), reflection_maps=None, max_bond=None
):
    """Project an MPS into the even simultaneous-reflection sector."""
    factors = [state._get_std_B(site).copy() for site in range(state.L)]
    reflection_maps = {} if reflection_maps is None else reflection_maps
    for site in reflection_sites:
        site = int(site)
        mapping = np.asarray(
            reflection_maps.get(site, np.arange(factors[site].shape[1] - 1, -1, -1)),
            dtype=int,
        )
        factors[site] = factors[site][:, mapping, :]
    reflected = MPS(factors, sites=state.sites)
    projected = state + reflected
    if max_bond is not None:
        projected = projected.compress(int(max_bond))
    projected.right_canonicalize().normalize()
    return projected


def load_parent_potential_cache(directory, axes, spec):
    directory = Path(directory).expanduser()
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("spec") != spec:
        return None
    cached_axes = tuple(np.load(directory / name) for name in metadata["axes"])
    if len(cached_axes) != len(axes) or any(
        not np.array_equal(left, right) for left, right in zip(cached_axes, axes)
    ):
        return None
    factors = [
        np.load(directory / entry["path"], mmap_mode="r")
        for entry in metadata["factors"]
    ]
    return MPO(factors), metadata["distillation"]


def save_parent_potential_cache(directory, axes, potential, spec, distillation):
    directory = Path(directory).expanduser()
    stage = directory.with_name(directory.name + ".tmp")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    axis_names = []
    for index, axis in enumerate(axes):
        name = f"axis_{index:02d}.npy"
        np.save(stage / name, axis)
        axis_names.append(name)
    factors = []
    for site, factor in enumerate(potential.factors):
        name = f"factor_{site:02d}.npy"
        values = np.asarray(factor)
        np.save(stage / name, values)
        factors.append({"path": name, "shape": list(values.shape)})
    metadata = {
        "version": 1,
        "spec": spec,
        "axes": axis_names,
        "factors": factors,
        "distillation": distillation,
    }
    (stage / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if directory.exists():
        shutil.rmtree(directory)
    stage.replace(directory)


def _residual_potential_spec(args, axes, base_spec, state_path):
    spec = {
        "representation": "adaptive-reflection-residual-sum-mpo-v2",
        "scalar_potential": _file_signature(args.scalar_potential),
        "base": base_spec,
        "state": None if state_path is None else _file_signature(state_path),
        "grid_shape": [len(axis) for axis in axes],
        "levels": int(args.potential_residual_levels),
        "rank": int(args.potential_residual_rank),
        "sweeps": int(args.potential_residual_sweeps),
        "start_rank": int(args.potential_residual_start_rank),
        "kick_rank": int(args.potential_residual_kick_rank),
        "cross_validation": int(args.potential_residual_cross_validation),
        "support_validation": int(args.potential_residual_support_validation),
        "guard_validation": int(args.potential_residual_guard_validation),
        "support_target_ev": float(args.potential_residual_target_ev),
        "guard_target_ev": float(args.potential_residual_guard_target_ev),
        "weighted_levels": int(args.potential_residual_weighted_levels),
        "weighted_rank": int(args.potential_residual_weighted_rank),
        "weighted_degree": int(args.potential_residual_weighted_degree),
        "weighted_samples": int(args.potential_residual_weighted_samples),
        "weighted_guard_samples": int(
            args.potential_residual_weighted_guard_samples
        ),
        "weighted_sweeps": int(args.potential_residual_weighted_sweeps),
        "weighted_regularization": float(
            args.potential_residual_weighted_regularization
        ),
        "weighted_window_quantile": float(
            args.potential_residual_weighted_window_quantile
        ),
        "seed": int(args.seed),
    }
    if args.potential_residual_discrete_levels:
        spec.update(
            {
                "discrete_levels": int(args.potential_residual_discrete_levels),
                "discrete_rank": int(args.potential_residual_discrete_rank),
                "discrete_samples": int(args.potential_residual_discrete_samples),
                "discrete_guard_samples": int(
                    args.potential_residual_discrete_guard_samples
                ),
                "discrete_sweeps": int(args.potential_residual_discrete_sweeps),
                "discrete_regularization": float(
                    args.potential_residual_discrete_regularization
                ),
            }
        )
    return spec


def load_residual_potential_cache(
    directory, axes, spec, *, allow_state_signature_change=False
):
    directory = Path(directory).expanduser()
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text())
    cached_spec = metadata.get("spec")
    if allow_state_signature_change and isinstance(cached_spec, dict):
        cached_spec = dict(cached_spec)
        current_spec = dict(spec)
        cached_state = cached_spec.get("state")
        current_state = current_spec.get("state")
        if isinstance(cached_state, dict) and isinstance(current_state, dict):
            cached_spec["state"] = {"path": cached_state.get("path")}
            current_spec["state"] = {"path": current_state.get("path")}
        matches = cached_spec == current_spec
    else:
        matches = cached_spec == spec
    if not matches:
        return None
    cached_axes = tuple(np.load(directory / name) for name in metadata["axes"])
    if len(cached_axes) != len(axes) or any(
        not np.array_equal(left, right) for left, right in zip(cached_axes, axes)
    ):
        return None
    corrections = []
    for entries in metadata["corrections"]:
        corrections.append(
            MPO(
                [
                    np.load(directory / entry["path"], mmap_mode="r")
                    for entry in entries
                ]
            )
        )
    return tuple(corrections), metadata["qualification"]


def save_residual_potential_cache(
    directory, axes, corrections, spec, qualification
):
    directory = Path(directory).expanduser()
    stage = directory.with_name(directory.name + ".tmp")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    axis_names = []
    for index, axis in enumerate(axes):
        name = f"axis_{index:02d}.npy"
        np.save(stage / name, axis)
        axis_names.append(name)
    saved_corrections = []
    for level, correction in enumerate(corrections):
        entries = []
        for site, factor in enumerate(correction.factors):
            name = f"correction_{level:02d}_factor_{site:02d}.npy"
            values = np.asarray(factor)
            np.save(stage / name, values)
            entries.append({"path": name, "shape": list(values.shape)})
        saved_corrections.append(entries)
    metadata = {
        "version": 1,
        "spec": spec,
        "axes": axis_names,
        "corrections": saved_corrections,
        "qualification": qualification,
    }
    (stage / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if directory.exists():
        shutil.rmtree(directory)
    stage.replace(directory)


def _potential_error_statistics(error):
    absolute = np.abs(np.asarray(error, dtype=float)) * HARTREE_TO_EV
    return {
        "rms_ev": float(np.sqrt(np.mean(absolute**2))),
        "mae_ev": float(np.mean(absolute)),
        "p99_ev": float(np.quantile(absolute, 0.99)),
        "maximum_ev": float(np.max(absolute)),
    }


def support_plateau_windows(
    indices,
    shape,
    quantile=0.999,
    reflection_sites=(1, 3),
    periodic_sites=(),
):
    """Build separable unit plateaus with smooth zero-boundary tapers."""
    indices = np.asarray(indices, dtype=int)
    reflection_sites = {int(site) for site in reflection_sites}
    periodic_sites = {int(site) for site in periodic_sites}
    windows = []
    plateaus = []
    for site, size in enumerate(shape):
        if site in periodic_sites:
            windows.append(np.ones(int(size), dtype=float))
            plateaus.append((0, int(size) - 1))
            continue
        if site in reflection_sites:
            center = (int(size) - 1) / 2.0
            radius = int(
                np.ceil(np.quantile(np.abs(indices[:, site] - center), quantile))
            ) + 1
            lower = max(0, int(np.floor(center - radius)))
            upper = min(int(size) - 1, int(np.ceil(center + radius)))
        else:
            lower = max(
                0,
                int(np.floor(np.quantile(indices[:, site], 1.0 - quantile))) - 1,
            )
            upper = min(
                int(size) - 1,
                int(np.ceil(np.quantile(indices[:, site], quantile))) + 1,
            )
        window = np.ones(int(size), dtype=float)
        if lower > 0:
            coordinate = np.arange(lower, dtype=float) / float(lower)
            window[:lower] = 0.5 - 0.5 * np.cos(np.pi * coordinate)
        if upper < int(size) - 1:
            coordinate = (
                np.arange(upper + 1, int(size), dtype=float) - upper
            ) / float(int(size) - 1 - upper)
            window[upper + 1 :] = 0.5 + 0.5 * np.cos(np.pi * coordinate)
        windows.append(window)
        plateaus.append((lower, upper))
    return tuple(windows), tuple(plateaus)


def discrete_tt_values(cores, indices):
    indices = np.asarray(indices, dtype=int)
    values = np.ones((len(indices), 1), dtype=np.result_type(*cores))
    for site, core in enumerate(cores):
        selected = np.moveaxis(core[:, indices[:, site], :], 1, 0)
        values = np.einsum("na,nab->nb", values, selected, optimize=True)
    return values[:, 0]


def fit_discrete_tt(
    indices,
    values,
    shape,
    *,
    rank,
    sweeps,
    regularization,
    seed,
    validation=None,
):
    """Fit categorical TT cores by ridge-regularized alternating least squares."""
    indices = np.asarray(indices, dtype=int)
    values = np.asarray(values, dtype=float)
    shape = tuple(int(size) for size in shape)
    if indices.shape != (len(values), len(shape)):
        raise ValueError("discrete TT samples have incompatible dimensions")
    ranks = [1]
    for split in range(1, len(shape)):
        ranks.append(
            min(
                int(rank),
                int(np.prod(shape[:split])),
                int(np.prod(shape[split:])),
            )
        )
    ranks.append(1)
    rng = np.random.default_rng(seed)
    cores = [
        rng.normal(scale=1.0 / np.sqrt(max(left, right)), size=(left, size, right))
        for left, size, right in zip(ranks[:-1], shape, ranks[1:])
    ]
    initial = discrete_tt_values(cores, indices)
    scale = np.std(initial)
    if scale > 1.0e-14:
        cores[0] *= max(np.std(values), 1.0e-12) / scale

    def environments(sample_indices):
        selected = [
            np.moveaxis(core[:, sample_indices[:, site], :], 1, 0)
            for site, core in enumerate(cores)
        ]
        lefts = [np.ones((len(sample_indices), 1))]
        for local in selected:
            lefts.append(
                np.einsum("na,nab->nb", lefts[-1], local, optimize=True)
            )
        rights = [None] * (len(shape) + 1)
        rights[-1] = np.ones((len(sample_indices), 1))
        for site in range(len(shape) - 1, -1, -1):
            rights[site] = np.einsum(
                "nab,nb->na", selected[site], rights[site + 1], optimize=True
            )
        return lefts, rights

    validation_indices = validation_values = None
    if validation is not None:
        validation_indices = np.asarray(validation[0], dtype=int)
        validation_values = np.asarray(validation[1], dtype=float)
    history = []
    best = None
    for sweep in range(1, int(sweeps) + 1):
        for direction in (range(len(shape)), range(len(shape) - 1, -1, -1)):
            for site in direction:
                lefts, rights = environments(indices)
                left = lefts[site]
                right = rights[site + 1]
                updated = cores[site].copy()
                feature_size = left.shape[1] * right.shape[1]
                identity = np.eye(feature_size)
                for physical in range(shape[site]):
                    selected = indices[:, site] == physical
                    if not np.any(selected):
                        continue
                    design = np.einsum(
                        "na,nb->nab",
                        left[selected],
                        right[selected],
                        optimize=True,
                    ).reshape(np.count_nonzero(selected), feature_size)
                    gram = design.T @ design
                    ridge = float(regularization) * max(
                        float(np.trace(gram)) / max(feature_size, 1), 1.0e-14
                    )
                    solution = np.linalg.solve(
                        gram + ridge * identity,
                        design.T @ values[selected],
                    )
                    updated[:, physical, :] = solution.reshape(
                        left.shape[1], right.shape[1]
                    )
                cores[site] = updated
        train_error = discrete_tt_values(cores, indices) - values
        train_rms = float(np.sqrt(np.mean(train_error**2)))
        if validation_indices is None:
            validation_rms = train_rms
        else:
            validation_error = (
                discrete_tt_values(cores, validation_indices) - validation_values
            )
            validation_rms = float(np.sqrt(np.mean(validation_error**2)))
        history.append(
            {
                "sweep": sweep,
                "train_rms_hartree": train_rms,
                "validation_rms_hartree": validation_rms,
            }
        )
        if best is None or validation_rms < best[0]:
            best = (validation_rms, [core.copy() for core in cores])
    return best[1], {
        "method": "state-weighted-discrete-tt-als",
        "rank": int(rank),
        "ranks": ranks,
        "sweeps": int(sweeps),
        "regularization": float(regularization),
        "history": history,
        "best_validation_rms_hartree": float(best[0]),
    }


def build_residual_potential(
    args,
    axes,
    base_components,
    support_state,
    *,
    reflection_maps=None,
    periodic_sites=(),
):
    """Fit successive low-rank residual MPOs and qualify their summed field."""
    scalar = ReflectionScalarMLP.load(args.scalar_potential)
    shape = tuple(len(axis) for axis in axes)
    rng = np.random.default_rng(args.seed + 191)
    guard_indices = np.column_stack(
        [
            rng.integers(0, size, size=args.potential_residual_guard_validation)
            for size in shape
        ]
    )
    support_indices = sample_mps_indices(
        support_state,
        args.potential_residual_support_validation,
        args.seed + 193,
    )
    weighted_windows, weighted_plateaus = support_plateau_windows(
        support_indices,
        shape,
        quantile=args.potential_residual_weighted_window_quantile,
        periodic_sites=periodic_sites,
    )

    def window_values(indices):
        indices = np.asarray(indices, dtype=int)
        values = np.ones(len(indices), dtype=float)
        for site, window in enumerate(weighted_windows):
            values *= window[indices[:, site]]
        return values

    def coordinates(indices):
        indices = np.asarray(indices, dtype=int)
        return np.column_stack(
            [axis[indices[:, site]] for site, axis in enumerate(axes)]
        )

    def oracle(indices):
        return scalar.predict(coordinates(indices))

    components = list(base_components)
    corrections = []
    levels = []

    def current_residual(indices):
        indices = np.asarray(indices, dtype=int)
        return oracle(indices) - mpo_diagonal_values(components, indices)

    def qualification(level, cross=None, queries=0):
        support_error = mpo_diagonal_values(components, support_indices) - oracle(
            support_indices
        )
        guard_error = mpo_diagonal_values(components, guard_indices) - oracle(
            guard_indices
        )
        reflected = guard_indices.copy()
        reflected[:, 1] = shape[1] - 1 - reflected[:, 1]
        reflected[:, 3] = shape[3] - 1 - reflected[:, 3]
        reflection_defect = np.max(
            np.abs(
                mpo_diagonal_values(components, guard_indices)
                - mpo_diagonal_values(components, reflected)
            )
        )
        result = {
            "level": int(level),
            "support": _potential_error_statistics(support_error),
            "guard": _potential_error_statistics(guard_error),
            "maximum_reflection_defect_hartree": float(reflection_defect),
            "geometry_queries": int(queries),
        }
        if cross is not None:
            result["cross"] = cross
        result["passed"] = bool(
            result["support"]["rms_ev"]
            <= args.potential_residual_target_ev
            and result["guard"]["rms_ev"]
            <= args.potential_residual_guard_target_ev
            and reflection_defect < 1.0e-12
        )
        return result

    levels.append(qualification(0))
    print(
        "[potential residual] level 0: "
        f"support={levels[-1]['support']['rms_ev']:.6f} eV "
        f"guard={levels[-1]['guard']['rms_ev']:.6f} eV",
        flush=True,
    )
    reference = np.asarray(
        [int(np.argmin(np.abs(axis - value))) for axis, value in zip(axes, PhenolReactiveChart().equilibrium)]
    )[None, :]
    for level in range(1, args.potential_residual_levels + 1):
        value_cache = {}

        def raw_residual(indices):
            indices = np.asarray(indices, dtype=int)
            keys = [tuple(row) for row in indices]
            missing = list(dict.fromkeys(key for key in keys if key not in value_cache))
            if missing:
                missing_indices = np.asarray(missing, dtype=int)
                values = oracle(missing_indices) - mpo_diagonal_values(
                    components, missing_indices
                )
                value_cache.update(zip(missing, values))
            return np.asarray([value_cache[key] for key in keys])

        offset = float(raw_residual(reference)[0])

        def centered_residual(indices):
            return raw_residual(indices) - offset

        cores, cross_info = tt_cross(
            shape,
            lambda _index: 0.0,
            batch_evaluator=centered_residual,
            max_rank=args.potential_residual_rank,
            sweeps=args.potential_residual_sweeps,
            rtol=0.0,
            validation=args.potential_residual_cross_validation,
            seed=args.seed + 197 * level,
            start_rank=args.potential_residual_start_rank,
            kick_rank=args.potential_residual_kick_rank,
        )
        cores = add_constant_to_tt(cores, offset)
        correction = tt_cores_to_diagonal_mpo(cores)
        corrections.append(correction)
        components.extend(
            reflection_pair(correction, reflection_maps=reflection_maps)
        )
        levels.append(
            qualification(level, cross=cross_info, queries=len(value_cache))
        )
        print(
            f"[potential residual] level {level}: "
            f"support={levels[-1]['support']['rms_ev']:.6f} eV "
            f"guard={levels[-1]['guard']['rms_ev']:.6f} eV",
            flush=True,
        )
        if levels[-1]["passed"]:
            break
    for discrete_level in range(1, args.potential_residual_discrete_levels + 1):
        if levels[-1]["passed"]:
            break
        training_support = sample_mps_indices(
            support_state,
            args.potential_residual_discrete_samples,
            args.seed + 401 * discrete_level,
        )
        training_guard = np.column_stack(
            [
                rng.integers(
                    0,
                    size,
                    size=args.potential_residual_discrete_guard_samples,
                )
                for size in shape
            ]
        )
        training_indices = np.vstack((training_support, training_guard))
        training_reflected = training_indices.copy()
        for site in (1, 3):
            mapping = (
                np.arange(shape[site] - 1, -1, -1)
                if reflection_maps is None
                else np.asarray(reflection_maps[site], dtype=int)
            )
            training_reflected[:, site] = mapping[training_reflected[:, site]]
        training_indices = np.vstack((training_indices, training_reflected))
        training_window = window_values(training_indices)
        retained = training_window >= 0.5
        training_indices = training_indices[retained]
        training_window = training_window[retained]
        training_target = current_residual(training_indices) / training_window
        validation_target = current_residual(support_indices)
        discrete_cores, discrete_info = fit_discrete_tt(
            training_indices,
            training_target,
            shape,
            rank=args.potential_residual_discrete_rank,
            sweeps=args.potential_residual_discrete_sweeps,
            regularization=args.potential_residual_discrete_regularization,
            seed=args.seed + 409 * discrete_level,
            validation=(support_indices, validation_target),
        )
        for site, window in enumerate(weighted_windows):
            discrete_cores[site] = (
                discrete_cores[site] * window[None, :, None]
            )
        correction = tt_cores_to_diagonal_mpo(discrete_cores)
        corrections.append(correction)
        components.extend(
            reflection_pair(correction, reflection_maps=reflection_maps)
        )
        discrete_info.update(
            {
                "discrete_level": int(discrete_level),
                "training_support_samples": int(len(training_support)),
                "training_guard_samples": int(len(training_guard)),
                "reflection_augmented_samples": int(len(training_indices)),
                "plateau_indices": [list(value) for value in weighted_plateaus],
                "window_quantile": float(
                    args.potential_residual_weighted_window_quantile
                ),
            }
        )
        levels.append(
            qualification(
                len(levels),
                cross=discrete_info,
                queries=len(training_indices),
            )
        )
        print(
            f"[potential residual] discrete {discrete_level}: "
            f"support={levels[-1]['support']['rms_ev']:.6f} eV "
            f"guard={levels[-1]['guard']['rms_ev']:.6f} eV",
            flush=True,
        )
    for weighted_level in range(1, args.potential_residual_weighted_levels + 1):
        if levels[-1]["passed"]:
            break
        training_support = sample_mps_indices(
            support_state,
            args.potential_residual_weighted_samples,
            args.seed + 307 * weighted_level,
        )
        training_guard = np.column_stack(
            [
                rng.integers(
                    0,
                    size,
                    size=args.potential_residual_weighted_guard_samples,
                )
                for size in shape
            ]
        )
        training_indices = np.vstack((training_support, training_guard))
        training_coordinates = coordinates(training_indices)
        training_target = current_residual(training_indices)
        validation_target = current_residual(support_indices)
        model = FunctionalTT(
            degrees=(args.potential_residual_weighted_degree,) * len(shape),
            rank=args.potential_residual_weighted_rank,
            bounds=np.asarray([(axis[0], axis[-1]) for axis in axes]),
            normalization="frobenius",
            hermitian=False,
            regularization=args.potential_residual_weighted_regularization,
            sweeps=args.potential_residual_weighted_sweeps,
            rtol=1.0e-8,
            local_rtol=1.0e-6,
            local_maxiter=30,
            patience=4,
            random_state=args.seed + 311 * weighted_level,
        ).fit(
            training_coordinates,
            training_target,
            validation=(coordinates(support_indices), validation_target),
        )
        weighted_cores = model.grid_cores(axes)
        for site, window in enumerate(weighted_windows):
            weighted_cores[site] = (
                weighted_cores[site] * window[None, :, None]
            )
        correction = tt_cores_to_diagonal_mpo(weighted_cores)
        corrections.append(correction)
        components.extend(
            reflection_pair(correction, reflection_maps=reflection_maps)
        )
        levels.append(
            qualification(
                len(levels),
                cross={
                    "method": "state-weighted-functional-tt",
                    "weighted_level": int(weighted_level),
                    "rank": int(args.potential_residual_weighted_rank),
                    "degree": int(args.potential_residual_weighted_degree),
                    "training_support_samples": int(len(training_support)),
                    "training_guard_samples": int(len(training_guard)),
                    "sweeps": int(model.n_sweeps),
                    "train_relative_error": float(model.error),
                    "validation_relative_error": float(model.validation_error),
                    "plateau_indices": [list(value) for value in weighted_plateaus],
                    "window_quantile": float(
                        args.potential_residual_weighted_window_quantile
                    ),
                },
                queries=len(training_indices),
            )
        )
        print(
            f"[potential residual] weighted {weighted_level}: "
            f"support={levels[-1]['support']['rms_ev']:.6f} eV "
            f"guard={levels[-1]['guard']['rms_ev']:.6f} eV",
            flush=True,
        )
    return tuple(corrections), {
        "method": "adaptive reflection-symmetric residual sum-of-MPO",
        "levels": levels,
        "support_samples": int(len(support_indices)),
        "guard_samples": int(len(guard_indices)),
        "support_target_ev": float(args.potential_residual_target_ev),
        "guard_target_ev": float(args.potential_residual_guard_target_ev),
        "passed": bool(levels[-1]["passed"]),
    }


def build_parent_potential(args, chart, axes):
    if args.scalar_potential is None:
        fit = MACE.load(args.checkpoint, chart.geometry, device="cpu", distill=False)
        fit.neural_energy = CorrectedMatrixField(
            fit.neural_energy,
            RadialMatrixCorrection.load(args.radial_correction),
        )

        def evaluate(coordinates):
            return fit.neural_energy.predict(coordinates)[
                :, args.electronic_state, args.electronic_state
            ].real

        representation = "local P-gauge electronic diagonal"
    else:
        scalar = ReflectionScalarMLP.load(args.scalar_potential)

        def evaluate(coordinates):
            return scalar.predict(coordinates)

        representation = "reflection-symmetrized scalar P-gauge parent fit"
    reference_coordinate = np.asarray(chart.equilibrium, dtype=float)[None, :]
    energy_offset = float(evaluate(reference_coordinate)[0])
    cache = {}

    def batch(indices):
        indices = np.asarray(indices, dtype=int)
        keys = [tuple(row) for row in indices]
        missing = list(dict.fromkeys(key for key in keys if key not in cache))
        if missing:
            coordinates = np.asarray(
                [
                    [axes[axis][index] for axis, index in enumerate(key)]
                    for key in missing
                ]
            )
            values = evaluate(coordinates) - energy_offset
            cache.update(zip(missing, values))
        return np.asarray([cache[key] for key in keys])

    cores, cross_info = tt_cross(
        tuple(len(axis) for axis in axes),
        lambda _index: 0.0,
        batch_evaluator=batch,
        max_rank=args.potential_tt_rank,
        sweeps=args.potential_cross_sweeps,
        rtol=args.potential_cross_rtol,
        validation=args.potential_cross_validation,
        seed=args.seed,
        start_rank=args.potential_cross_start_rank,
        kick_rank=args.potential_cross_kick_rank,
    )
    cores = add_constant_to_tt(cores, energy_offset)
    factors = []
    for core in cores:
        left, physical, right = core.shape
        factor = np.zeros((left, right, physical, physical), dtype=core.dtype)
        diagonal = np.arange(physical)
        factor[:, :, diagonal, diagonal] = core.transpose(0, 2, 1)
        factors.append(factor)
    potential = MPO(factors)

    rng = np.random.default_rng(args.seed + 17)
    validation_indices = np.column_stack(
        [
            rng.integers(0, len(axis), size=args.potential_validation_points)
            for axis in axes
        ]
    )
    validation = np.column_stack(
        [axis[validation_indices[:, site]] for site, axis in enumerate(axes)]
    )
    reference = evaluate(validation)
    predicted = []
    for indices in validation_indices:
        environment = np.ones((1, 1), dtype=complex)
        for factor, index in zip(potential.factors, indices):
            environment = environment @ factor[:, :, index, index]
        predicted.append(environment[0, 0].real)
    predicted = np.asarray(predicted)
    error = predicted - reference
    info = {
        "method": "scalar-grid-cross",
        "representation": representation,
        "rank": int(args.potential_tt_rank),
        "energy_offset_hartree": energy_offset,
        "centered_cross": True,
        "geometry_queries": len(cache),
        "validation_points": len(validation),
        "validation_rms_hartree": float(np.sqrt(np.mean(error**2))),
        "validation_max_hartree": float(np.max(np.abs(error))),
        "validation_rms_ev": float(np.sqrt(np.mean(error**2)) * HARTREE_TO_EV),
        "validation_max_ev": float(np.max(np.abs(error)) * HARTREE_TO_EV),
        "cross": cross_info,
    }
    return potential, info


def load_scalar_keo_cache(directory, axes, spec):
    directory = Path(directory).expanduser()
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("spec") != spec:
        return None
    cached_axes = tuple(np.load(directory / name) for name in metadata["axes"])
    if len(cached_axes) != len(axes) or any(
        not np.array_equal(left, right) for left, right in zip(cached_axes, axes)
    ):
        return None
    components = []
    for entries in metadata["components"]:
        components.append(
            MPO(
                [
                    np.load(directory / entry["path"], mmap_mode="r")
                    for entry in entries
                ]
            )
        )
    return tuple(components), metadata["keo_info"]


def save_scalar_keo_cache(directory, axes, components, spec, keo_info):
    directory = Path(directory).expanduser()
    stage = directory.with_name(directory.name + ".tmp")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    axis_names = []
    for index, axis in enumerate(axes):
        name = f"axis_{index:02d}.npy"
        np.save(stage / name, axis)
        axis_names.append(name)
    manifests = []
    for component_index, component in enumerate(components):
        entries = []
        for site, factor in enumerate(component.factors):
            name = f"component_{component_index:02d}_site_{site:02d}.npy"
            values = np.asarray(factor)
            np.save(stage / name, values)
            entries.append({"path": name, "shape": list(values.shape)})
        manifests.append(entries)
    metadata = {
        "version": 1,
        "spec": spec,
        "axes": axis_names,
        "components": manifests,
        "keo_info": keo_info,
    }
    (stage / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if directory.exists():
        shutil.rmtree(directory)
    stage.replace(directory)


def product_gaussian(axes, center, widths, sites=None):
    factors = []
    for axis, origin, width in zip(axes, center, widths):
        values = np.exp(-0.25 * ((axis - origin) / width) ** 2).astype(complex)
        values /= np.linalg.norm(values)
        factors.append(values[None, :, None])
    return MPS(factors, sites=sites)


def seeded_state(product, max_bond, amplitude=1.0e-3, seed=731):
    """Add a normalized low-amplitude full-rank direction to a product state."""
    dims = product.dims
    total = int(np.prod(dims))
    ranks = []
    left = 1
    right = total
    for dimension in dims[:-1]:
        left *= dimension
        right //= dimension
        ranks.append(min(int(max_bond), left, right))
    rng = np.random.default_rng(seed)
    random_factors = []
    left_rank = 1
    for site, dimension in enumerate(dims):
        right_rank = ranks[site] if site < len(ranks) else 1
        shape = (left_rank, dimension, right_rank)
        factor = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        factor /= np.sqrt(factor.size)
        random_factors.append(factor)
        left_rank = right_rank
    random_state = MPS(random_factors, sites=product.sites).right_canonicalize()

    base = product.copy()
    base.factors[0] *= np.sqrt(max(0.0, 1.0 - amplitude**2))
    random_state.factors[0] *= amplitude
    state = (base + random_state).compress(max_bond).right_canonicalize()
    state.normalize()
    return state


def mps_to_dense(state):
    factors = [state._get_std_B(index) for index in range(state.L)]
    values = factors[0][0]
    for factor in factors[1:]:
        values = np.tensordot(values, factor, axes=([-1], [0]))
    return np.asarray(values[..., 0])


def mps_overlap(bra, ket):
    if bra.L != ket.L:
        raise ValueError("MPS lengths differ")
    environment = np.ones((1, 1), dtype=complex)
    for index in range(bra.L):
        left = bra._get_std_B(index)
        right = ket._get_std_B(index)
        environment = np.einsum(
            "ab,api,bpj->ij", environment, left.conj(), right, optimize=True
        )
    return complex(environment[0, 0])


def mps_marginals(state):
    factors = [state._get_std_B(index) for index in range(state.L)]
    lefts = [np.ones((1, 1), dtype=complex)]
    for factor in factors:
        lefts.append(
            np.einsum(
                "ab,api,bpj->ij",
                lefts[-1],
                factor.conj(),
                factor,
                optimize=True,
            )
        )
    rights = [None] * (state.L + 1)
    rights[-1] = np.ones((1, 1), dtype=complex)
    for site in range(state.L - 1, -1, -1):
        factor = factors[site]
        rights[site] = np.einsum(
            "api,bpj,ij->ab",
            factor.conj(),
            factor,
            rights[site + 1],
            optimize=True,
        )
    output = []
    for site, factor in enumerate(factors):
        values = np.einsum(
            "ab,api,bpj,ij->p",
            lefts[site],
            factor.conj(),
            factor,
            rights[site + 1],
            optimize=True,
        ).real
        values = np.maximum(values, 0.0)
        output.append(values / values.sum())
    return tuple(output)


def exact_energy(state, components):
    return float(np.real(sum(state.expectation(term) for term in components)))


def shifted_identity(axes, energy):
    factors = [np.eye(len(axis), dtype=complex)[None, None, :, :] for axis in axes]
    factors[0] = -float(energy) * factors[0]
    return MPO(factors)


def potential_radial_slice(potential, axes, equilibrium):
    components = (potential,) if isinstance(potential, MPO) else tuple(potential)
    indices = [
        int(np.argmin(np.abs(axis - value)))
        for axis, value in zip(axes, equilibrium)
    ]
    grid_indices = np.asarray(
        [[radial, *indices[1:]] for radial in range(len(axes[0]))]
    )
    values = mpo_diagonal_values(components, grid_indices)
    return values - values.min()


def save_checkpoint(path, state, history, axes=None):
    payload = {
        "history_json": np.asarray(json.dumps(history)),
        "factor_count": np.asarray(state.L),
    }
    payload.update(
        {f"factor_{index}": state._get_std_B(index) for index in range(state.L)}
    )
    if axes is not None:
        payload.update(
            {f"axis_{index}": np.asarray(axis) for index, axis in enumerate(axes)}
        )
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **payload)
    temporary.replace(path)


def load_checkpoint(path, sites):
    with np.load(path, allow_pickle=False) as saved:
        count = int(saved["factor_count"])
        factors = [np.asarray(saved[f"factor_{index}"]) for index in range(count)]
        history = json.loads(str(saved["history_json"]))
    state = MPS(factors, sites=sites).right_canonicalize()
    state.normalize()
    return state, history


def load_interpolated_state(path, axes, sites):
    """Interpolate a saved grid MPS onto new one-dimensional DVR axes."""
    with np.load(path, allow_pickle=False) as saved:
        old_axes = tuple(
            np.asarray(saved[f"axis_{site}"], dtype=float)
            for site in range(len(axes))
        ) if all(f"axis_{site}" in saved.files for site in range(len(axes))) else None
        prefix = (
            "mps_factor_"
            if all(f"mps_factor_{site}" in saved.files for site in range(len(axes)))
            else "factor_"
        )
        old_factors = tuple(
            np.asarray(saved[f"{prefix}{site}"])
            for site in range(len(axes))
        )
    if old_axes is None:
        if any(
            factor.shape[1] != len(axis)
            for factor, axis in zip(old_factors, axes)
        ):
            raise ValueError(
                "axis-free MPS checkpoint can only be loaded on its original grid"
            )
        old_axes = tuple(np.asarray(axis, dtype=float) for axis in axes)
    factors = []
    for old_axis, new_axis, factor in zip(old_axes, axes, old_factors):
        left, _physical, right = factor.shape
        interpolated = np.empty((left, len(new_axis), right), dtype=complex)
        for left_index in range(left):
            for right_index in range(right):
                values = factor[left_index, :, right_index]
                interpolated[left_index, :, right_index] = np.interp(
                    new_axis, old_axis, values.real, left=0.0, right=0.0
                ) + 1j * np.interp(
                    new_axis, old_axis, values.imag, left=0.0, right=0.0
                )
        factors.append(interpolated)
    state = MPS(factors, sites=sites).right_canonicalize()
    state.normalize()
    return state


def _plot(
    output,
    axes,
    initial_marginals,
    final_marginals,
    potential,
    history,
    periodic_axes=(),
):
    figure, panels = plt.subplots(2, 3, figsize=(13.0, 7.6), constrained_layout=True)
    energies = np.asarray([row["energy_hartree"] for row in history], dtype=float)
    imaginary_time = np.asarray([row["imaginary_time_au"] for row in history])
    panels[0, 0].semilogy(
        imaginary_time,
        np.maximum(
            np.abs(energies - energies[-1]) * HARTREE_TO_WAVENUMBER,
            1.0e-8,
        ),
        "o-",
    )
    panels[0, 0].set(
        xlabel=r"imaginary time $\tau$ (a.u.)",
        ylabel=r"$|E-E_f|$ (cm$^{-1}$)",
        title="Exact sum-of-MPO convergence",
    )

    panels[0, 1].plot(
        axes[0], initial_marginals[0], "--", color="0.4", label="Condon Gaussian"
    )
    panels[0, 1].plot(axes[0], final_marginals[0], "o-", label=r"$S_1$ parent")
    panels[0, 1].set(
        xlabel=r"$R_{OH}$ (angstrom)", ylabel="probability", title="Radial localization"
    )
    panels[0, 1].legend(frameon=False)
    potential_axis = panels[0, 1].twinx()
    potential_axis.plot(axes[0], potential * HARTREE_TO_EV, color="C3", alpha=0.55)
    potential_axis.set_ylabel(r"$V_{S_1}-V_{min}$ (eV)", color="C3")

    plot_axes = (axes[1], np.rad2deg(axes[2]), axes[3], axes[4])
    labels = (
        r"$q_1=\phi$ (rad)",
        r"$q_2=\theta$ (deg)",
        r"$q_3=Q_{16a}$",
        r"$q_4=Q_{8a}$",
    )
    for site, (panel, axis, initial, final, label) in enumerate(zip(
        (panels[0, 2], panels[1, 0], panels[1, 1], panels[1, 2]),
        plot_axes,
        initial_marginals[1:],
        final_marginals[1:],
        labels,
    ), start=1):
        if site in periodic_axes:
            period = float((axis[1] - axis[0]) * len(axis))
            axis = np.append(axis, axis[0] + period)
            initial = np.append(initial, initial[0])
            final = np.append(final, final[0])
        panel.plot(axis, initial, "--", color="0.4", label="Condon Gaussian")
        panel.plot(axis, final, "o-", label=r"$S_1$ parent")
        panel.set(xlabel=label, ylabel="probability", title=f"{label} marginal")
    png = output / "phenol_sa_casscf_5d_s1_quasibound.png"
    pdf = output / "phenol_sa_casscf_5d_s1_quasibound.pdf"
    figure.savefig(png, dpi=220)
    figure.savefig(pdf)
    plt.close(figure)
    return png, pdf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator-cache", type=Path, default=DEFAULT_PHENOL_5D_OPERATOR_CACHE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_PHENOL_5D_CHECKPOINT)
    parser.add_argument(
        "--radial-correction", type=Path, default=DEFAULT_PHENOL_5D_RADIAL_CORRECTION
    )
    parser.add_argument(
        "--scalar-potential", type=Path,
        help="saved reflection-symmetrized scalar parent potential",
    )
    parser.add_argument(
        "--potential-source", choices=("mace", "operator-cache"), default="mace"
    )
    parser.add_argument(
        "--parent-potential-cache", type=Path, default=DEFAULT_PARENT_POTENTIAL_CACHE
    )
    parser.add_argument(
        "--potential-residual-cache",
        type=Path,
        default=DEFAULT_RESIDUAL_POTENTIAL_CACHE,
    )
    parser.add_argument(
        "--reuse-potential-residual-cache",
        action="store_true",
        help="reuse a matching residual cache after its support checkpoint advances",
    )
    parser.add_argument("--potential-residual-levels", type=int, default=0)
    parser.add_argument("--potential-residual-rank", type=int, default=24)
    parser.add_argument("--potential-residual-sweeps", type=int, default=8)
    parser.add_argument("--potential-residual-start-rank", type=int, default=8)
    parser.add_argument("--potential-residual-kick-rank", type=int, default=8)
    parser.add_argument("--potential-residual-cross-validation", type=int, default=256)
    parser.add_argument("--potential-residual-support-validation", type=int, default=2048)
    parser.add_argument("--potential-residual-guard-validation", type=int, default=512)
    parser.add_argument("--potential-residual-target-ev", type=float, default=0.005)
    parser.add_argument(
        "--potential-residual-guard-target-ev", type=float, default=0.03
    )
    parser.add_argument(
        "--potential-residual-state",
        type=Path,
        help="saved MPS whose Born distribution qualifies residual-potential error",
    )
    parser.add_argument("--potential-residual-weighted-levels", type=int, default=0)
    parser.add_argument("--potential-residual-weighted-rank", type=int, default=8)
    parser.add_argument("--potential-residual-weighted-degree", type=int, default=8)
    parser.add_argument(
        "--potential-residual-weighted-samples", type=int, default=8192
    )
    parser.add_argument(
        "--potential-residual-weighted-guard-samples", type=int, default=2048
    )
    parser.add_argument("--potential-residual-weighted-sweeps", type=int, default=12)
    parser.add_argument(
        "--potential-residual-weighted-regularization", type=float, default=1.0e-7
    )
    parser.add_argument(
        "--potential-residual-weighted-window-quantile", type=float, default=0.999
    )
    parser.add_argument("--potential-residual-discrete-levels", type=int, default=0)
    parser.add_argument("--potential-residual-discrete-rank", type=int, default=8)
    parser.add_argument(
        "--potential-residual-discrete-samples", type=int, default=8192
    )
    parser.add_argument(
        "--potential-residual-discrete-guard-samples", type=int, default=8192
    )
    parser.add_argument("--potential-residual-discrete-sweeps", type=int, default=12)
    parser.add_argument(
        "--potential-residual-discrete-regularization", type=float, default=1.0e-6
    )
    parser.add_argument("--scalar-keo-cache", type=Path, default=DEFAULT_SCALAR_KEO_CACHE)
    parser.add_argument("--data", type=Path, default=DEFAULT_PHENOL_5D_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--grid-shape", type=_shape, default=DEFAULT_GRID_SHAPE)
    parser.add_argument("--torsion-limit", type=float, default=1.00)
    parser.add_argument(
        "--periodic-torsion",
        action="store_true",
        help="use a full -pi <= phi < pi periodic exponential DVR",
    )
    parser.add_argument("--bend-min-deg", type=float, default=90.0)
    parser.add_argument("--bend-max-deg", type=float, default=134.0)
    parser.add_argument("--q16a-limit", type=float, default=1.00)
    parser.add_argument("--q8a-limit", type=float, default=0.40)
    parser.add_argument("--electronic-state", type=int, default=1)
    parser.add_argument("--potential-tt-rank", type=int, default=24)
    parser.add_argument("--potential-cross-sweeps", type=int, default=12)
    parser.add_argument("--potential-cross-rtol", type=float, default=1.0e-7)
    parser.add_argument("--potential-cross-validation", type=int, default=128)
    parser.add_argument("--potential-cross-start-rank", type=int, default=8)
    parser.add_argument("--potential-cross-kick-rank", type=int, default=8)
    parser.add_argument("--potential-validation-points", type=int, default=128)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--bond-dimension", type=int, default=24)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--initial-tau", type=float, default=0.10)
    parser.add_argument("--max-tau", type=float, default=8.0)
    parser.add_argument("--tau-growth", type=float, default=1.35)
    parser.add_argument("--energy-tol", type=float, default=1.0e-9)
    parser.add_argument("--state-tol", type=float, default=1.0e-8)
    parser.add_argument("--krylov-dimension", type=int, default=16)
    parser.add_argument("--krylov-tol", type=float, default=1.0e-11)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed-amplitude", type=float, default=1.0e-3)
    parser.add_argument(
        "--initial-state",
        type=Path,
        help="saved quasibound NPZ to interpolate onto the requested grid",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    data = np.load(args.data, allow_pickle=True)
    chart = PhenolReactiveChart(modes=data["modes"])
    equilibrium = chart.equilibrium
    training_bounds = np.column_stack(
        (np.min(data["coordinates"], axis=0), np.max(data["coordinates"], axis=0))
    )
    source_metadata = json.loads((args.operator_cache / "metadata.json").read_text())
    periodic_axes = (1,) if args.periodic_torsion else ()
    if args.potential_source == "operator-cache":
        axes, potential_operator, source_metadata = load_parent_potential(
            args.operator_cache, args.electronic_state
        )
        grid_bounds = np.column_stack(
            ([axis[0] for axis in axes], [axis[-1] for axis in axes])
        )
        rebuilt_axes, dvrs = build_dvrs(
            tuple(len(axis) for axis in axes),
            grid_bounds,
            chart,
            periodic_axes=periodic_axes,
        )
        if any(
            not np.allclose(left, right, atol=1.0e-12)
            for left, right in zip(axes, rebuilt_axes)
        ):
            raise RuntimeError("the scalar KEO grid does not match the potential grid")
        potential_cache_hit = True
        potential_distillation = source_metadata["payload"]["potential_info"]
        potential_components = (potential_operator,)
        residual_cache_hit = None
        residual_qualification = None
        potential_reflection_enforced = False
    else:
        grid_bounds = np.array(DEFAULT_WIDE_BOUNDS, copy=True)
        grid_bounds[0] = training_bounds[0]
        grid_bounds[1] = (
            (-np.pi, np.pi)
            if args.periodic_torsion
            else (-args.torsion_limit, args.torsion_limit)
        )
        grid_bounds[2] = np.deg2rad((args.bend_min_deg, args.bend_max_deg))
        grid_bounds[3] = (-args.q16a_limit, args.q16a_limit)
        grid_bounds[4] = (-args.q8a_limit, args.q8a_limit)
        axes, dvrs = build_dvrs(
            args.grid_shape,
            grid_bounds,
            chart,
            periodic_axes=periodic_axes,
        )
        potential_spec = _parent_potential_spec(args, axes)
        cached_potential = load_parent_potential_cache(
            args.parent_potential_cache, axes, potential_spec
        )
        if cached_potential is None:
            potential_operator, potential_distillation = build_parent_potential(
                args, chart, axes
            )
            save_parent_potential_cache(
                args.parent_potential_cache,
                axes,
                potential_operator,
                potential_spec,
                potential_distillation,
            )
            potential_cache_hit = False
            print(
                f"[parent potential cache] saved {args.parent_potential_cache}",
                flush=True,
            )
        else:
            potential_operator, potential_distillation = cached_potential
            potential_cache_hit = True
            print(
                f"[parent potential cache] loaded {args.parent_potential_cache}",
                flush=True,
            )
        reflection_maps = {
            site: _reflection_indices(dvrs[site]) for site in (1, 3)
        }
        potential_components = reflection_pair(
            potential_operator, reflection_maps=reflection_maps
        )
        potential_reflection_enforced = True
        residual_cache_hit = None
        residual_qualification = None
        if (
            args.potential_residual_levels
            or args.potential_residual_weighted_levels
            or args.potential_residual_discrete_levels
        ):
            if args.scalar_potential is None:
                raise ValueError("residual MPO fitting requires --scalar-potential")
            residual_state_path = (
                args.potential_residual_state
                if args.potential_residual_state is not None
                else args.initial_state
            )
            if residual_state_path is None:
                raise ValueError(
                    "residual MPO fitting requires --potential-residual-state or --initial-state"
                )
            support_state = load_interpolated_state(
                residual_state_path, axes, None
            )
            support_state = symmetrize_state(
                support_state,
                reflection_maps=reflection_maps,
                max_bond=args.bond_dimension,
            )
            residual_spec = _residual_potential_spec(
                args, axes, potential_spec, residual_state_path
            )
            cached_residual = load_residual_potential_cache(
                args.potential_residual_cache,
                axes,
                residual_spec,
                allow_state_signature_change=args.reuse_potential_residual_cache,
            )
            if cached_residual is None:
                corrections, residual_qualification = build_residual_potential(
                    args,
                    axes,
                    potential_components,
                    support_state,
                    reflection_maps=reflection_maps,
                    periodic_sites=periodic_axes,
                )
                save_residual_potential_cache(
                    args.potential_residual_cache,
                    axes,
                    corrections,
                    residual_spec,
                    residual_qualification,
                )
                residual_cache_hit = False
                print(
                    f"[potential residual cache] saved {args.potential_residual_cache}",
                    flush=True,
                )
            else:
                corrections, residual_qualification = cached_residual
                residual_cache_hit = True
                print(
                    f"[potential residual cache] loaded {args.potential_residual_cache}",
                    flush=True,
                )
            for correction in corrections:
                potential_components += reflection_pair(
                    correction, reflection_maps=reflection_maps
                )
        potential_distillation = {
            "base": potential_distillation,
            "exact_reflection_symmetrization": True,
            "residual": residual_qualification,
        }
    extrapolated_axes = [
        bool(lower < train_lower - 1.0e-12 or upper > train_upper + 1.0e-12)
        for (lower, upper), (train_lower, train_upper) in zip(
            grid_bounds, training_bounds
        )
    ]
    keo_spec = _scalar_keo_spec(source_metadata, axes, periodic_axes)
    keo_started = time.perf_counter()
    cached_keo = load_scalar_keo_cache(args.scalar_keo_cache, axes, keo_spec)
    if cached_keo is None:
        settings = source_metadata["spec"]
        split_keo, keo_info = build_phenol_5d_keo_mpo(
            dvrs,
            chart,
            cross_max_rank=settings["keo_cross_rank"],
            cross_sweeps=settings["keo_cross_sweeps"],
            cross_rtol=settings["keo_cross_rtol"],
            cross_validation=settings["keo_cross_validation"],
            mpo_max_rank=settings["keo_mpo_rank"],
            seed=settings["seed"],
            split=True,
            return_info=True,
        )
        kinetic_components = tuple(component for _active, component in split_keo)
        save_scalar_keo_cache(
            args.scalar_keo_cache,
            axes,
            kinetic_components,
            keo_spec,
            keo_info,
        )
        keo_cache_hit = False
        print(f"[scalar KEO cache] saved {args.scalar_keo_cache}", flush=True)
    else:
        kinetic_components, keo_info = cached_keo
        keo_cache_hit = True
        print(f"[scalar KEO cache] loaded {args.scalar_keo_cache}", flush=True)
    keo_seconds = time.perf_counter() - keo_started
    components = (*kinetic_components, *potential_components)
    initial = product_gaussian(
        axes,
        equilibrium,
        PACKET_WIDTHS,
        sites=components[0].input_sites,
    )
    initial_marginals = mps_marginals(initial)
    initial_energy = exact_energy(initial, components)
    evolution_components = (*components, shifted_identity(axes, initial_energy))
    checkpoint = args.output / "scalar_parent_imaginary_time_checkpoint.npz"

    if args.resume and checkpoint.is_file():
        state, history = load_checkpoint(checkpoint, components[0].input_sites)
        energy = exact_energy(state, components)
        cumulative_tau = float(history[-1]["imaginary_time_au"])
        print(f"[resume] step={len(history) - 1} E={energy:.12f} Eh", flush=True)
    else:
        if args.initial_state is None:
            state = seeded_state(
                initial,
                args.bond_dimension,
                amplitude=args.seed_amplitude,
            )
        else:
            state = load_interpolated_state(
                args.initial_state, axes, components[0].input_sites
            ).compress(args.bond_dimension)
            state.normalize()
            print(f"[initial state] interpolated {args.initial_state}", flush=True)
        if potential_reflection_enforced:
            state = symmetrize_state(
                state,
                reflection_maps=reflection_maps,
                max_bond=args.bond_dimension,
            )
            print("[initial state] projected into even reflection sector", flush=True)
        energy = exact_energy(state, components)
        cumulative_tau = 0.0
        history = [
            {
                "step": 0,
                "imaginary_time_au": 0.0,
                "step_tau_au": 0.0,
                "energy_hartree": energy,
                "energy_change_hartree": None,
                "fidelity": None,
                "seconds": 0.0,
            }
        ]
        save_checkpoint(checkpoint, state, history, axes)

    tau = args.initial_tau
    stable_steps = 0
    converged = False
    solve_started = time.perf_counter()
    executor = ThreadPoolExecutor(max_workers=args.workers)
    for step in range(len(history), args.max_steps + 1):
        accepted = False
        for _attempt in range(12):
            started = time.perf_counter()
            trial, info = one_site_tdvp_sum_step(
                state,
                evolution_components,
                tau,
                krylov_dim=args.krylov_dimension,
                krylov_tol=args.krylov_tol,
                canonicalize=False,
                normalize=True,
                imaginary_time=True,
                return_info=True,
                _executor=executor,
            )
            trial_energy = exact_energy(trial, components)
            if np.isfinite(trial_energy) and trial_energy <= energy + 2.0e-10:
                accepted = True
                break
            tau *= 0.5
            if tau < 1.0e-5:
                raise RuntimeError("imaginary-time sweep could not find a stable step")
            print(
                f"[imaginary time] rejected energy increase; retry tau={tau:.6g} au",
                flush=True,
            )
        if not accepted:
            raise RuntimeError("imaginary-time sweep failed after 12 retries")

        fidelity = min(1.0, abs(mps_overlap(state, trial)) ** 2)
        energy_change = trial_energy - energy
        cumulative_tau += tau
        elapsed = time.perf_counter() - started
        state = trial
        energy = trial_energy
        history.append(
            {
                "step": step,
                "imaginary_time_au": cumulative_tau,
                "step_tau_au": tau,
                "energy_hartree": energy,
                "energy_change_hartree": energy_change,
                "fidelity": fidelity,
                "seconds": elapsed,
                "backend": info["backend"],
            }
        )
        save_checkpoint(checkpoint, state, history, axes)
        print(
            f"[imaginary time] step {step:03d}: tau={tau:.5g} au "
            f"E={energy:.12f} Eh dE={energy_change:+.3e} "
            f"1-F={1.0 - fidelity:.3e} ({elapsed:.1f} s)",
            flush=True,
        )
        if (
            step >= 8
            and abs(energy_change) < args.energy_tol
            and 1.0 - fidelity < args.state_tol
        ):
            stable_steps += 1
        else:
            stable_steps = 0
        if stable_steps >= 3:
            converged = True
            break
        tau = min(args.max_tau, tau * args.tau_growth)

    executor.shutdown(wait=True)
    solve_seconds = time.perf_counter() - solve_started
    state.normalize()
    final_marginals = mps_marginals(state)

    propagated = one_site_tdvp_sum_step(
        state,
        components,
        0.5,
        krylov_dim=args.krylov_dimension,
        krylov_tol=args.krylov_tol,
        canonicalize=False,
        normalize=True,
    )
    survival = min(1.0, abs(mps_overlap(state, propagated)) ** 2)
    stationarity_scale = np.sqrt(max(0.0, 1.0 - survival)) / 0.5

    radial = axes[0]
    radial_tail = {
        str(cutoff): float(final_marginals[0][radial >= cutoff].sum())
        for cutoff in (1.3, 1.5, 1.8, 2.45)
    }
    coordinate_statistics = []
    boundary_probabilities = []
    for site, (axis, marginal) in enumerate(zip(axes, final_marginals)):
        if site in periodic_axes:
            step = float(axis[1] - axis[0])
            period = step * len(axis)
            center = float(axis[0] + 0.5 * period)
            phase = np.dot(
                np.exp(2.0j * np.pi * (axis - center) / period), marginal
            )
            mean = float(center + period * np.angle(phase) / (2.0 * np.pi))
            variance = float(
                (-2.0 * np.log(max(abs(phase), np.finfo(float).tiny)))
                * (period / (2.0 * np.pi)) ** 2
            )
        else:
            mean = float(np.dot(axis, marginal))
            variance = float(np.dot((axis - mean) ** 2, marginal))
        coordinate_statistics.append(
            {"mean": mean, "standard_deviation": np.sqrt(variance)}
        )
        boundary_probabilities.append(
            None if site in periodic_axes else float(marginal[0] + marginal[-1])
        )

    potential = potential_radial_slice(potential_components, axes, equilibrium)
    figure_png, figure_pdf = _plot(
        args.output,
        axes,
        initial_marginals,
        final_marginals,
        potential,
        history,
        periodic_axes,
    )

    result = args.output / "phenol_sa_casscf_5d_s1_quasibound.npz"
    payload = {
        "electronic_state": np.asarray(args.electronic_state),
        "electronic_factor": np.eye(3, dtype=complex)[args.electronic_state][
            None, :, None
        ],
    }
    payload.update({f"axis_{index}": axis for index, axis in enumerate(axes)})
    payload.update(
        {f"marginal_{index}": values for index, values in enumerate(final_marginals)}
    )
    payload.update(
        {f"mps_factor_{index}": state._get_std_B(index) for index in range(state.L)}
    )
    np.savez_compressed(result, **payload)

    summary = {
        "definition": (
            "lowest vibrational eigenstate of the bare scalar 5D KEO plus the "
            "local P-gauge S1 potential"
        ),
        "method": "normalized imaginary-time one-site TDVP on a scalar sum of MPO components",
        "electronic_state": args.electronic_state,
        "grid_shape": [len(axis) for axis in axes],
        "grid_bounds": grid_bounds.tolist(),
        "periodic_axes": list(periodic_axes),
        "training_bounds": training_bounds.tolist(),
        "extrapolated_axes": extrapolated_axes,
        "operator_components": len(components),
        "potential_source": (
            "scalar-reflection-mlp"
            if args.scalar_potential is not None
            else args.potential_source
        ),
        "parent_potential_cache": {
            "path": str(args.parent_potential_cache.resolve()),
            "hit": potential_cache_hit,
        },
        "potential_reflection_enforced": potential_reflection_enforced,
        "potential_residual_cache": {
            "path": str(args.potential_residual_cache.resolve()),
            "hit": residual_cache_hit,
        },
        "potential_distillation": potential_distillation,
        "potential_cache_version": source_metadata.get("version"),
        "scalar_keo_cache": {
            "path": str(args.scalar_keo_cache.resolve()),
            "hit": keo_cache_hit,
        },
        "state_bond_dimension": args.bond_dimension,
        "initial_state_source": (
            None if args.initial_state is None else str(args.initial_state.resolve())
        ),
        "state_bond_orders": state.bond_orders(),
        "converged": converged,
        "energy_hartree": energy,
        "energy_ev": energy * HARTREE_TO_EV,
        "initial_energy_hartree": initial_energy,
        "stationarity_fidelity_dt_0p5au": survival,
        "stationarity_energy_scale_hartree": float(stationarity_scale),
        "stationarity_energy_scale_wavenumber": float(
            stationarity_scale * HARTREE_TO_WAVENUMBER
        ),
        "radial_tail_probability": radial_tail,
        "coordinate_statistics": coordinate_statistics,
        "boundary_probabilities": boundary_probabilities,
        "maximum_boundary_probability": max(
            value for value in boundary_probabilities if value is not None
        ),
        "range_converged": bool(
            max(
                value
                for site, value in enumerate(boundary_probabilities)
                if site != 0 and value is not None
            )
            < 1.0e-3
        ),
        "ab_initio_supported": not any(extrapolated_axes),
        "imaginary_time_history": history,
        "timings_seconds": {"scalar_keo": keo_seconds, "solve": solve_seconds},
        "checkpoint": str(checkpoint.resolve()),
        "result": str(result.resolve()),
        "figure_png": str(figure_png.resolve()),
        "figure_pdf": str(figure_pdf.resolve()),
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
