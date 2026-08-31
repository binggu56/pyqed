#!/usr/bin/env python3
"""Distill the phenol SA(6)-CASSCF 5D field and run numerical-g TTLDR."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import tempfile
import time

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

if os.environ.get("PYQED_NO_MATPLOTLIB") == "1":
    plt = None
else:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
import numpy as np

from pyqed.cache import file_signature as _file_signature, specs_equivalent
from examples.namd.phenol_sa_casscf_paths import (
    DEFAULT_PHENOL_5D_CHECKPOINT,
    DEFAULT_PHENOL_5D_DATA,
    DEFAULT_PHENOL_5D_RADIAL_CORRECTION,
)
from pyqed.dvr import ExponentialDVR, SineDVR
from pyqed.ml import CorrectedMatrixField, MACE, RadialMatrixCorrection
from pyqed.models.phenol_coordinates import PhenolReactiveChart
from pyqed.mps.functional import FunctionalTT
from pyqed.mps.mps import MPO, _release_free_numeric_pages
from pyqed.namd.phenol import build_phenol_5d_keo_mpo
from pyqed.namd.ttldr import TTLDR, polynomial_cap
from pyqed.units import au2fs, au2mev


HARTREE_TO_MEV = au2mev
COLORS = ("#0072B2", "#D55E00", "#009E73")


def peak_rss_gib():
    """Return this process's peak resident memory in GiB."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1.0 if sys.platform == "darwin" else 1024.0
    return float(value * scale / 2**30)


def _shape(value):
    shape = tuple(int(item) for item in str(value).split(","))
    if len(shape) != 5 or any(item < 2 for item in shape):
        raise argparse.ArgumentTypeError("grid shape must contain five integers >= 2")
    return shape


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _distillation_cache_spec(args, axes, shape):
    spec = {
        "checkpoint": _file_signature(args.checkpoint),
        "radial_correction": _file_signature(args.radial_correction),
        "grid_bounds": [[float(axis[0]), float(axis[-1])] for axis in axes],
        "method": args.distill_method,
        "rank": int(args.tt_rank),
        "degree": int(args.tt_degree),
        "cross_points": int(args.cross_points),
        "cross_sweeps": int(args.cross_sweeps),
        "cross_rtol": float(args.cross_rtol),
        "cross_validation": int(args.cross_validation),
        "validation_points": int(args.validation_points),
        "seed": int(args.seed),
    }
    if args.distill_method == "grid":
        spec["grid_shape"] = list(shape)
    return spec


def _operator_cache_spec(args):
    return {
        "data": _file_signature(args.data),
        "checkpoint": _file_signature(args.checkpoint),
        "radial_correction": _file_signature(args.radial_correction),
        "grid_shape": list(args.grid_shape),
        "distill_grid_shape": (
            None
            if args.distill_grid_shape is None
            else list(args.distill_grid_shape)
        ),
        "distill_method": args.distill_method,
        "tt_rank": int(args.tt_rank),
        "tt_degree": int(args.tt_degree),
        "keo_cross_rank": int(args.keo_cross_rank),
        "keo_cross_sweeps": int(args.keo_cross_sweeps),
        "keo_cross_rtol": float(args.keo_cross_rtol),
        "keo_cross_validation": int(args.keo_cross_validation),
        "keo_mpo_rank": int(args.keo_mpo_rank),
        "keo_metric_derivative": "dirichlet-complete-procrustes-v1",
        "overlap_rank": int(args.overlap_rank),
        "potential_rank": int(args.potential_rank),
        "operator_rank": int(args.operator_rank),
        "seed": int(args.seed),
    }


def _load_operator_cache(directory, expected_spec, candidate_axes):
    directory = Path(directory)
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("version") != 1 or not specs_equivalent(
        metadata.get("spec"), expected_spec
    ):
        raise RuntimeError(f"operator cache {directory} does not match this calculation")
    axes = tuple(
        np.asarray(np.load(directory / name), dtype=float)
        for name in metadata["axes"]
    )
    if len(axes) != len(candidate_axes) or any(
        left.shape != right.shape or not np.allclose(left, right, atol=1.0e-12)
        for left, right in zip(axes, candidate_axes)
    ):
        raise RuntimeError("operator cache axes do not match the requested grid")
    components = []
    for component_info in metadata["components"]:
        factors = []
        for factor_info in component_info:
            factor = np.load(directory / factor_info["path"], mmap_mode="r")
            if list(factor.shape) != factor_info["shape"] or str(factor.dtype) != factor_info["dtype"]:
                raise IOError(f"operator cache factor {factor_info['path']} is inconsistent")
            factors.append(factor)
        components.append(MPO(factors))
    payload = metadata["payload"]
    driver = TTLDR.from_components(
        components,
        grids=axes,
        overlap_info=payload["overlap_info"],
        potential_info=payload["potential_info"],
    )
    return driver, axes, payload


def _save_operator_cache(directory, spec, axes, driver, payload):
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"operator cache already exists: {directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{directory.name}.", dir=directory.parent)
    )
    try:
        axis_names = []
        for index, axis in enumerate(axes):
            name = f"axis_{index:02d}.npy"
            np.save(temporary / name, np.asarray(axis))
            axis_names.append(name)
        component_manifest = []
        for component_index, component in enumerate(driver.components):
            factor_manifest = []
            for site, factor in enumerate(component.factors):
                name = f"component_{component_index:02d}_site_{site:02d}.npy"
                array = np.asarray(factor)
                np.save(temporary / name, array)
                factor_manifest.append(
                    {"path": name, "shape": list(array.shape), "dtype": str(array.dtype)}
                )
            component_manifest.append(factor_manifest)
        metadata = {
            "version": 1,
            "spec": spec,
            "axes": axis_names,
            "components": component_manifest,
            "payload": _jsonable(payload),
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        os.replace(temporary, directory)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _load_distillation_cache(fit, directory, expected):
    directory = Path(directory)
    metadata_path = directory / "metadata.json"
    energy_path = directory / "energy.npz"
    feature_path = directory / "feature.npz"
    if not all(path.is_file() for path in (metadata_path, energy_path, feature_path)):
        return False
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("spec") != expected:
        return False
    fit.energy = FunctionalTT.load(energy_path)
    fit.feature = FunctionalTT.load(feature_path)
    fit.links = None
    fit.info["distillation"] = metadata["distillation"]
    return True


def _save_distillation_cache(fit, directory, spec):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fit.energy.save(directory / "energy.npz")
    fit.feature.save(directory / "feature.npz")
    metadata = {
        "spec": spec,
        "distillation": _jsonable(fit.info["distillation"]),
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def _dvr_walls(first, last, points):
    """Place the first and last sine-DVR nodes at the requested limits."""
    step = (float(last) - float(first)) / (int(points) - 1)
    return float(first) - step, float(last) + step


def build_dvrs(shape, bounds, chart, *, periodic_axes=()):
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (5, 2):
        raise ValueError("bounds must have shape (5, 2)")
    atomic_bounds = np.asarray(
        [chart.coordinate_to_atomic(bound) for bound in bounds.T]
    ).T
    periodic_axes = {int(axis) for axis in periodic_axes}
    if any(axis < 0 or axis >= 5 for axis in periodic_axes):
        raise ValueError("periodic DVR axis is out of range")
    dvrs = []
    for axis in range(5):
        if axis in periodic_axes:
            lower, upper = atomic_bounds[axis]
            dvrs.append(
                ExponentialDVR(
                    npts=int(shape[axis]),
                    L=float(upper - lower),
                    x0=float(0.5 * (lower + upper)),
                    mass=1.0,
                )
            )
        else:
            dvrs.append(
                SineDVR(
                    *_dvr_walls(*atomic_bounds[axis], shape[axis]),
                    int(shape[axis]),
                    mass=1.0,
                )
            )
    dvrs = tuple(dvrs)
    public_scale = chart.coordinate_from_atomic(np.ones(5))
    axes = tuple(
        dvr.x * public_scale[axis] for axis, dvr in enumerate(dvrs)
    )
    return axes, dvrs


def grid_coordinates(axes):
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([value.reshape(-1) for value in mesh], axis=1)


def _parallel_transport_phase(vectors, anchor):
    vectors = np.asarray(vectors, dtype=complex)
    grid_shape = vectors.shape[:-1]
    anchor = tuple(int(item) for item in anchor)
    result = np.zeros_like(vectors)
    visited = np.zeros(grid_shape, dtype=bool)
    pending = [anchor]
    reference = vectors[anchor]
    pivot = reference[np.argmax(np.abs(reference))]
    result[anchor] = reference * pivot.conjugate() / abs(pivot)
    visited[anchor] = True
    while pending:
        parent = pending.pop(0)
        for axis in range(len(grid_shape)):
            for step in (-1, 1):
                child = list(parent)
                child[axis] += step
                child = tuple(child)
                if not 0 <= child[axis] < grid_shape[axis] or visited[child]:
                    continue
                raw = vectors[child]
                overlap = np.vdot(result[parent], raw)
                if abs(overlap) > 1.0e-12:
                    raw = raw * overlap.conjugate() / abs(overlap)
                result[child] = raw
                visited[child] = True
                pending.append(child)
    return result


def initial_packet(axes, potential, equilibrium, state=1):
    """Return a localized Condon packet in one local adiabatic channel."""
    widths = np.asarray((0.060, 0.13, np.deg2rad(2.8), 0.10, 0.055))
    factors = [
        np.exp(-0.25 * ((axis - center) / width) ** 2)
        for axis, center, width in zip(axes, equilibrium, widths)
    ]
    nuclear = factors[0]
    for factor in factors[1:]:
        nuclear = np.multiply.outer(nuclear, factor)
    _energies, vectors = np.linalg.eigh(potential)
    anchor = tuple(
        int(np.argmin(np.abs(axis - center)))
        for axis, center in zip(axes, equilibrium)
    )
    electronic = _parallel_transport_phase(vectors[..., :, int(state)], anchor)
    packet = nuclear[..., None] * electronic
    return packet / np.linalg.norm(packet)


def _load_resume(path):
    required = (
        "times_fs",
        "populations",
        "norms",
        "final_wavefunction",
        "initial_radial",
        "cap_expectations",
        "cap_yields",
        "absorbed_probabilities",
    )
    with np.load(path, allow_pickle=True) as archive:
        missing = [name for name in required if name not in archive]
        if missing:
            raise ValueError(f"resume archive is missing {missing}")
        result = {name: np.array(archive[name], copy=True) for name in required}
    times = result["times_fs"]
    if times.ndim != 1 or not len(times) or np.any(np.diff(times) <= 0.0):
        raise ValueError("resume times must be a nonempty increasing vector")
    if result["populations"].shape[0] != len(times):
        raise ValueError("resume populations do not match the saved times")
    if result["norms"].shape != times.shape:
        raise ValueError("resume norms do not match the saved times")
    return result


def _append_resume_history(previous, current, time_offset_fs):
    times_fs, populations, norms, cap_expectations, cap_yields, absorbed = current
    population_jump = float(
        np.max(np.abs(previous["populations"][-1] - populations[0]))
    )
    norm_jump = float(abs(previous["norms"][-1] - norms[0]))
    previous_yield = previous["cap_yields"][-1]
    previous_absorbed = float(previous["absorbed_probabilities"][-1])
    combined = {
        "times_fs": np.concatenate(
            (previous["times_fs"], time_offset_fs + times_fs[1:])
        ),
        "populations": np.concatenate(
            (previous["populations"], populations[1:]), axis=0
        ),
        "norms": np.concatenate((previous["norms"], norms[1:])),
        "cap_expectations": np.concatenate(
            (previous["cap_expectations"], cap_expectations[1:]), axis=0
        ),
        "cap_yields": np.concatenate(
            (previous["cap_yields"], previous_yield + cap_yields[1:]), axis=0
        ),
        "absorbed": np.concatenate(
            (
                previous["absorbed_probabilities"],
                previous_absorbed + absorbed[1:],
            )
        ),
        "population_jump": population_jump,
        "norm_jump": norm_jump,
    }
    return combined


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    observation_interval = (
        max(1, int(args.steps) // 100)
        if args.interval is None
        else int(args.interval)
    )
    if observation_interval < 1:
        raise ValueError("observation interval must be positive")
    memory_peaks = {}
    data = np.load(args.data, allow_pickle=True)
    chart = PhenolReactiveChart(modes=data["modes"])
    sampled_bounds = np.column_stack(
        (np.min(data["coordinates"], axis=0), np.max(data["coordinates"], axis=0))
    )
    axes, dvrs = build_dvrs(args.grid_shape, sampled_bounds, chart)
    operator_spec = _operator_cache_spec(args)
    cached_operator = None
    operator_cache_seconds = 0.0
    if args.operator_cache is not None:
        cache_started = time.perf_counter()
        cached_operator = _load_operator_cache(
            args.operator_cache, operator_spec, axes
        )
        operator_cache_seconds = time.perf_counter() - cache_started
        if cached_operator is not None:
            _cached_driver, axes, _cached_payload = cached_operator
            print(f"[operator cache] loaded {args.operator_cache}", flush=True)
    distill_shape = (
        args.grid_shape
        if args.distill_grid_shape is None
        else args.distill_grid_shape
    )
    distill_axes = (
        axes
        if distill_shape == args.grid_shape
        else build_dvrs(distill_shape, sampled_bounds, chart)[0]
    )

    loaded = time.perf_counter()
    fit = MACE.load(args.checkpoint, chart.geometry, device="cpu", distill=False)
    load_seconds = time.perf_counter() - loaded
    memory_peaks["checkpoint_load"] = peak_rss_gib()
    if fit._fit_mode != "features":
        raise RuntimeError("the 5D TTLDR route requires a MACE-Y feature checkpoint")
    fit.grids = tuple(np.asarray(axis) for axis in distill_axes)
    fit.shape = tuple(len(axis) for axis in distill_axes)
    correction = None
    if args.radial_correction is not None:
        correction = RadialMatrixCorrection.load(args.radial_correction)
        fit.neural_energy = CorrectedMatrixField(fit.neural_energy, correction)

    distilled = time.perf_counter()
    cache_spec = _distillation_cache_spec(args, distill_axes, distill_shape)
    cache_hit = (
        args.distilled_cache is not None
        and _load_distillation_cache(fit, args.distilled_cache, cache_spec)
    )
    if cache_hit:
        print(f"[distillation cache] loaded {args.distilled_cache}", flush=True)
    else:
        if args.distill_method == "grid":
            fit.distill_y(
                rank=args.tt_rank,
                degree=args.tt_degree,
                method="grid",
                prediction_batch_size=args.prediction_batch_size,
                validation_points=args.validation_points,
                seed=args.seed,
            )
        else:
            fit.distill_y(
                rank=args.tt_rank,
                degree=args.tt_degree,
                method="cross",
                prediction_batch_size=args.prediction_batch_size,
                cross_points=args.cross_points,
                cross_sweeps=args.cross_sweeps,
                cross_rtol=args.cross_rtol,
                cross_validation=args.cross_validation,
                validation_points=args.validation_points,
                seed=args.seed,
            )
        if args.distilled_cache is not None:
            _save_distillation_cache(fit, args.distilled_cache, cache_spec)
            print(f"[distillation cache] saved {args.distilled_cache}", flush=True)
    distill_seconds = time.perf_counter() - distilled
    memory_peaks["ftt_distillation"] = peak_rss_gib()
    fit.grids = tuple(np.asarray(axis) for axis in axes)
    fit.shape = tuple(len(axis) for axis in axes)

    rng = np.random.default_rng(args.seed + 101)
    validation = rng.uniform(
        sampled_bounds[:, 0], sampled_bounds[:, 1],
        size=(args.validation_points, 5),
    )
    neural_energy = fit.neural_energy.predict(validation)
    fitted_energy = fit.energy.predict(validation)
    spectral_error = np.abs(
        np.linalg.eigvalsh(fitted_energy) - np.linalg.eigvalsh(neural_energy)
    ) * HARTREE_TO_MEV

    if cached_operator is None:
        keo_started = time.perf_counter()
        keo, keo_info = build_phenol_5d_keo_mpo(
            dvrs,
            chart,
            cross_max_rank=args.keo_cross_rank,
            cross_sweeps=args.keo_cross_sweeps,
            cross_rtol=args.keo_cross_rtol,
            cross_validation=args.keo_cross_validation,
            mpo_max_rank=args.keo_mpo_rank,
            seed=args.seed,
            split=True,
            return_info=True,
        )
        keo_seconds = time.perf_counter() - keo_started
        memory_peaks["numerical_g_keo"] = peak_rss_gib()
        kinetic_component_active_axes = [active for active, _operator in keo]
        kinetic_component_ranks = [
            operator.bond_orders() for _active, operator in keo
        ]

        setup_started = time.perf_counter()
        driver = TTLDR.from_fit(
            fit,
            grids=axes,
            keo=keo,
            overlap_rank=args.overlap_rank,
            potential_rank=args.potential_rank,
            operator_rank=args.operator_rank,
            fitted_kinetic_backend="link-mpo",
        )
        setup_seconds = time.perf_counter() - setup_started
        memory_peaks["dressed_operator_setup"] = peak_rss_gib()
        if args.operator_cache is not None:
            overlap_info = {
                name: driver.overlap_info[name]
                for name in ("backend", "fields", "memory_profile")
                if name in driver.overlap_info
            }
            cache_payload = {
                "keo_info": keo_info,
                "kinetic_component_active_axes": kinetic_component_active_axes,
                "kinetic_component_ranks": kinetic_component_ranks,
                "overlap_info": overlap_info,
                "potential_info": driver.potential_info,
            }
            _save_operator_cache(
                args.operator_cache, operator_spec, axes, driver, cache_payload
            )
            print(f"[operator cache] saved {args.operator_cache}", flush=True)
    else:
        driver, axes, cache_payload = cached_operator
        keo = None
        keo_info = cache_payload["keo_info"]
        kinetic_component_active_axes = cache_payload[
            "kinetic_component_active_axes"
        ]
        kinetic_component_ranks = cache_payload["kinetic_component_ranks"]
        keo_seconds = 0.0
        setup_seconds = operator_cache_seconds
        memory_peaks["numerical_g_keo"] = peak_rss_gib()
        memory_peaks["dressed_operator_setup"] = peak_rss_gib()

    previous = None
    restart_info = None
    if args.resume_from is None:
        coordinates = grid_coordinates(axes)
        potential = fit.energy.predict(coordinates).reshape(
            *args.grid_shape, 3, 3
        )
        memory_peaks["dense_potential"] = peak_rss_gib()
        initial = initial_packet(
            axes, potential, chart.equilibrium, state=args.bright_state
        )
        state = driver.state(initial, max_rank=args.state_rank)
        initial_radial = np.abs(initial) ** 2
        initial_radial = initial_radial.sum(axis=(1, 2, 3, 4, 5))
        initial_radial /= initial_radial.sum()
        del coordinates, potential, initial
    else:
        previous = _load_resume(args.resume_from)
        initial = previous["final_wavefunction"]
        expected_shape = (*args.grid_shape, 3)
        if initial.shape != expected_shape:
            raise ValueError(
                f"resume wavefunction shape {initial.shape} != {expected_shape}"
            )
        state = driver.state(initial, max_rank=args.state_rank, normalize=False)
        reconstructed = driver.dense(state)
        reconstruction_error = float(
            np.linalg.norm((reconstructed - initial).ravel())
            / max(np.linalg.norm(initial.ravel()), np.finfo(float).tiny)
        )
        if reconstruction_error > args.restart_tol:
            raise RuntimeError(
                "resume state exceeds the reconstruction tolerance: "
                f"{reconstruction_error:.3e} > {args.restart_tol:.3e}"
            )
        initial_radial = np.asarray(previous["initial_radial"], dtype=float)
        restart_info = {
            "from": str(Path(args.resume_from).resolve()),
            "time_fs": float(previous["times_fs"][-1]),
            "state_reconstruction_relative_l2": reconstruction_error,
        }
        previous.pop("final_wavefunction")
        del initial, reconstructed
    memory_peaks["dense_potential"] = peak_rss_gib()
    distillation_info = fit.info["distillation"]
    driver.keo = None
    del fit, keo
    gc.collect()
    _release_free_numeric_pages()
    memory_peaks["initial_state_preparation"] = peak_rss_gib()
    use_cap = args.cap_strength > 0.0
    cap = (
        polynomial_cap(axes[0], args.cap_start, args.cap_strength, args.cap_order)
        if use_cap else None
    )
    if use_cap:
        print(
            f"[split CAP] start={args.cap_start:.3f} A, "
            f"Wmax={np.max(cap):.4f} Eh, order={args.cap_order}",
            flush=True,
        )
    times_fs = np.linspace(0.0, args.tmax_fs, args.steps + 1)
    propagated = time.perf_counter()
    driver.run(
        state,
        dt=float((times_fs[1] - times_fs[0]) / au2fs),
        steps=args.steps,
        interval=observation_interval,
        max_bond=args.state_rank,
        integrator=args.integrator,
        cutoff=args.cutoff,
        krylov_dim=args.krylov_dim,
        krylov_tol=args.krylov_tol,
        workers=args.workers,
        progress=args.progress,
        e_ops=driver.projectors(),
        absorber=cap,
        absorber_site=0,
    )
    propagation_seconds = time.perf_counter() - propagated
    memory_peaks["propagation"] = peak_rss_gib()

    final = driver.dense(driver.final_state)
    probability = np.abs(final) ** 2
    radial_absolute = probability.sum(axis=(1, 2, 3, 4, 5))
    final_norm = float(np.sum(radial_absolute))
    radial = radial_absolute / max(final_norm, np.finfo(float).tiny)
    populations = np.asarray(driver.populations)
    norms = np.asarray(driver.norms)
    recorded_times_fs = np.asarray(driver.times) * au2fs
    cap_expectations = (
        np.asarray(driver.absorber_expectations)
        if use_cap else np.zeros_like(populations)
    )
    cap_yields = (
        np.asarray(driver.absorber_yields)
        if use_cap else np.zeros_like(populations)
    )
    absorbed = (
        np.asarray(driver.absorbed_probabilities)
        if use_cap else 1.0 - np.asarray(driver.norms)
    )
    if previous is not None:
        combined = _append_resume_history(
            previous,
            (
                recorded_times_fs,
                populations,
                norms,
                cap_expectations,
                cap_yields,
                absorbed,
            ),
            restart_info["time_fs"],
        )
        recorded_times_fs = combined["times_fs"]
        populations = combined["populations"]
        norms = combined["norms"]
        cap_expectations = combined["cap_expectations"]
        cap_yields = combined["cap_yields"]
        absorbed = combined["absorbed"]
        restart_info["population_jump"] = combined["population_jump"]
        restart_info["norm_jump"] = combined["norm_jump"]
        if max(combined["population_jump"], combined["norm_jump"]) > args.restart_tol:
            raise RuntimeError(
                "resume observables exceed the restart tolerance: "
                f"population={combined['population_jump']:.3e}, "
                f"norm={combined['norm_jump']:.3e}"
            )
    absorption_closure = np.sum(cap_yields, axis=1) - absorbed
    fc_radial_index = int(np.argmin(np.abs(axes[0] - chart.equilibrium[0])))

    summary = {
        "electronic_model": "SA(6)-CASSCF three-state diagnostic-root P gauge MACE-Y",
        "checkpoint": str(args.checkpoint),
        "radial_correction": (
            None if args.radial_correction is None else str(args.radial_correction)
        ),
        "grid_shape": args.grid_shape,
        "sampled_bounds": sampled_bounds,
        "radial_grid": {
            "domain_angstrom": [float(axes[0][0]), float(axes[0][-1])],
            "nearest_equilibrium_node_angstrom": float(axes[0][fc_radial_index]),
            "equilibrium_node_offset_angstrom": float(
                axes[0][fc_radial_index] - chart.equilibrium[0]
            ),
            "initial_endpoint_probability": float(
                initial_radial[0] + initial_radial[-1]
            ),
        },
        "kinetic_model": "AD G-matrix J=0 Podolsky KEO",
        "kinetic_components": len(kinetic_component_active_axes),
        "kinetic_component_active_axes": kinetic_component_active_axes,
        "kinetic_component_ranks": kinetic_component_ranks,
        "keo_cross": keo_info,
        "dressed_keo_backend": driver.overlap_info["backend"],
        "distillation": distillation_info,
        "distillation_grid_shape": distill_shape,
        "distillation_cache": {
            "path": None if args.distilled_cache is None else str(args.distilled_cache),
            "hit": cache_hit,
        },
        "operator_cache": {
            "path": None if args.operator_cache is None else str(args.operator_cache),
            "hit": cached_operator is not None,
        },
        "validation_spectral_rms_mev": float(np.sqrt(np.mean(spectral_error**2))),
        "validation_spectral_max_mev": float(np.max(spectral_error)),
        "operator_ranks": driver.operator_ranks,
        "operator_compression": [
            field.get("compression") for field in driver.overlap_info["fields"]
        ],
        "operator_memory_profile": driver.overlap_info.get("memory_profile"),
        "cap": {
            "enabled": use_cap,
            "algorithm": "Strang-split exact local damping around Hermitian TDVP",
            "start_angstrom": args.cap_start if use_cap else None,
            "strength_hartree": args.cap_strength if use_cap else None,
            "order": args.cap_order if use_cap else None,
            "final_norm": final_norm,
            "final_absorbed_probability": float(absorbed[-1]),
            "final_channel_yields": cap_yields[-1],
            "maximum_absorption_closure_defect": float(
                np.max(np.abs(absorption_closure))
            ),
        },
        "restart": restart_info,
        "maximum_population_sum_defect": float(
            np.max(np.abs(np.sum(populations, axis=1) - norms))
        ),
        "peak_rss_gib": peak_rss_gib(),
        "cumulative_peak_rss_gib": memory_peaks,
        "timings_seconds": {
            "checkpoint_load": load_seconds,
            "ftt_distillation": distill_seconds,
            "numerical_g_keo": keo_seconds,
            "dressed_operator_setup": setup_seconds,
            "propagation": propagation_seconds,
        },
        "dynamics": {
            "initial": "localized Gaussian Condon packet in the local adiabatic channel",
            "bright_adiabatic_state": args.bright_state,
            "time_fs": float(recorded_times_fs[-1]),
            "segment_time_fs": args.tmax_fs,
            "steps": int(round(recorded_times_fs[-1] / (args.tmax_fs / args.steps))),
            "segment_steps": args.steps,
            "observation_interval_steps": observation_interval,
            "state_rank": args.state_rank,
            "workers": args.workers,
            "integrator": args.integrator,
            "krylov_dim": args.krylov_dim,
            "krylov_tolerance": args.krylov_tol,
        },
    }

    figure, panels = plt.subplots(2, 2, figsize=(9.2, 6.6), constrained_layout=True)
    for root, color in enumerate(COLORS):
        panels[0, 0].hist(
            spectral_error[:, root], bins=18, histtype="step", lw=1.5,
            color=color, label=f"state {root}",
        )
    panels[0, 0].set(
        xlabel="MACE-to-FTT spectral error (meV)", ylabel="count",
        title="Electronic-field distillation",
    )
    panels[0, 0].legend(frameon=False, fontsize=8)
    panels[0, 1].plot(
        recorded_times_fs, absorbed, color="black", lw=2.0, label="total absorbed"
    )
    for channel, color in enumerate(COLORS):
        panels[0, 1].plot(
            recorded_times_fs,
            cap_yields[:, channel],
            color=color,
            lw=1.25,
            label=fr"$Y_{channel}$",
        )
    panels[0, 1].set(
        xlabel="time (fs)", ylabel="dissociation probability",
        title="CAP flux",
    )
    panels[0, 1].legend(frameon=False, fontsize=7)
    for channel, color in enumerate(COLORS):
        panels[1, 0].plot(
            recorded_times_fs,
            populations[:, channel],
            color=color,
            label=f"P{channel}",
        )
    panels[1, 0].set(
        xlabel="time (fs)", ylabel="P-gauge population", ylim=(-0.02, 1.02),
        title="Five-dimensional TTLDR",
    )
    panels[1, 0].legend(frameon=False, fontsize=8)
    panels[1, 1].plot(axes[0], initial_radial, "--", color="0.35", label="initial")
    panels[1, 1].plot(axes[0], radial, "o-", color=COLORS[0], label="final")
    if use_cap:
        panels[1, 1].axvspan(
            args.cap_start, axes[0][-1], color=COLORS[1], alpha=0.10, label="CAP"
        )
    panels[1, 1].set(
        xlabel=r"$R_{OH}$ (angstrom)", ylabel="radial probability",
        title="Radial marginal",
    )
    panels[1, 1].legend(frameon=False, fontsize=8)
    for panel in panels.flat:
        panel.spines[["top", "right"]].set_visible(False)
    figure_path = args.output / "phenol_sa_casscf_5d_ftt_ttldr.png"
    figure.savefig(figure_path, dpi=260)
    figure.savefig(args.output / "phenol_sa_casscf_5d_ftt_ttldr.pdf")
    plt.close(figure)

    np.savez(
        args.output / "phenol_sa_casscf_5d_ftt_ttldr.npz",
        times_fs=recorded_times_fs,
        populations=populations,
        norms=norms,
        axes=np.asarray(axes, dtype=object),
        final_wavefunction=final,
        initial_radial=initial_radial,
        final_radial=radial,
        final_radial_absolute=radial_absolute,
        cap_profile=np.asarray([] if cap is None else cap),
        cap_expectations=cap_expectations,
        cap_yields=cap_yields,
        absorbed_probabilities=absorbed,
        absorption_closure=absorption_closure,
    )
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2) + "\n")
    print(json.dumps(_jsonable(summary), indent=2))
    print(f"figure: {figure_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path,
        default=DEFAULT_PHENOL_5D_DATA,
    )
    parser.add_argument(
        "--radial-correction", type=Path,
        default=DEFAULT_PHENOL_5D_RADIAL_CORRECTION,
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=DEFAULT_PHENOL_5D_CHECKPOINT,
    )
    parser.add_argument("--grid-shape", type=_shape, default=(25, 5, 3, 5, 5))
    parser.add_argument(
        "--distill-grid-shape", type=_shape,
        help="separate grid used to distill the continuous electronic field",
    )
    parser.add_argument("--distill-method", choices=("grid", "cross"), default="grid")
    parser.add_argument("--tt-rank", type=int, default=24)
    parser.add_argument("--tt-degree", type=int, default=6)
    parser.add_argument("--cross-points", type=int, default=8)
    parser.add_argument("--cross-sweeps", type=int, default=8)
    parser.add_argument("--cross-rtol", type=float, default=1.0e-7)
    parser.add_argument("--cross-validation", type=int, default=128)
    parser.add_argument("--validation-points", type=int, default=128)
    parser.add_argument("--prediction-batch-size", type=int, default=1024)
    parser.add_argument(
        "--distilled-cache", type=Path,
        help="directory for metadata-checked FunctionalTT energy/feature caching",
    )
    parser.add_argument(
        "--operator-cache",
        type=Path,
        help="directory for the exact dressed sum-of-MPO Hamiltonian cache",
    )
    parser.add_argument("--keo-cross-rank", type=int, default=12)
    parser.add_argument("--keo-cross-sweeps", type=int, default=8)
    parser.add_argument("--keo-cross-rtol", type=float, default=1.0e-7)
    parser.add_argument("--keo-cross-validation", type=int, default=128)
    parser.add_argument("--keo-mpo-rank", type=int, default=96)
    parser.add_argument("--overlap-rank", type=int, default=16)
    parser.add_argument("--potential-rank", type=int, default=32)
    parser.add_argument("--operator-rank", type=int, default=64)
    parser.add_argument("--state-rank", type=int, default=24)
    parser.add_argument("--bright-state", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="continue from a prior cumulative phenol 5D dynamics NPZ",
    )
    parser.add_argument("--restart-tol", type=float, default=1.0e-8)
    parser.add_argument("--cap-start", type=float, default=2.45)
    parser.add_argument("--cap-strength", type=float, default=0.02)
    parser.add_argument("--cap-order", type=int, default=4)
    parser.add_argument("--tmax-fs", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument(
        "--interval",
        type=int,
        help="observation interval in steps (default: about 100 checkpoints)",
    )
    parser.add_argument("--cutoff", type=float, default=1.0e-10)
    parser.add_argument("--krylov-dim", type=int, default=8)
    parser.add_argument("--krylov-tol", type=float, default=1.0e-10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--integrator", choices=("tdvp", "tdvp2"), default="tdvp")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument(
        "--output", type=Path,
        default=Path("/private/tmp/phenol_sa_casscf_5d_ftt_ttldr"),
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
