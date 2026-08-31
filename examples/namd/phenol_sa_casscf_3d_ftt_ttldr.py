#!/usr/bin/env python3
"""Distill the phenol SA-CASSCF MACE-Y field and run a short 3D TTLDR pilot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
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
from scipy.sparse.linalg import LinearOperator, eigsh

from pyqed.dvr import SineDVR
from pyqed.ml import MACE
from pyqed.models.phenol import Phenol3D
from pyqed.models.phenol_coordinates import PhenolReactiveChart
from pyqed.mps.functional import FunctionalTT
from pyqed.mps.mps import MPO
from pyqed.namd.ttldr import TTLDR, polynomial_cap
from pyqed.units import au2angstrom, au2ev, au2fs


HARTREE_TO_EV = au2ev
COLORS = ("#0072B2", "#D55E00", "#009E73")
CHART = PhenolReactiveChart()


def geometry(coordinate):
    value = np.array(CHART.equilibrium, copy=True)
    value[:3] = coordinate
    return CHART.geometry(value)


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


def _integers(value):
    result = tuple(int(item) for item in str(value).split(",") if item.strip())
    if not result or any(item < 1 for item in result):
        raise argparse.ArgumentTypeError("provide a comma-separated list of positive integers")
    return result


def _predict(field, coordinates, batch=384):
    coordinates = np.asarray(coordinates, dtype=float)
    return np.concatenate(
        [field.predict(coordinates[start : start + batch]) for start in range(0, len(coordinates), batch)]
    )


def build_dvrs(nr, nphi, ntheta, bounds):
    probe = Phenol3D([Phenol3D.r_eq], [Phenol3D.theta_eq], [0.0])
    radial = SineDVR(
        bounds[0, 0] / au2angstrom,
        bounds[0, 1] / au2angstrom,
        int(nr),
        mass=probe.radial_mass,
    )
    torsion = SineDVR(
        bounds[1, 0], bounds[1, 1], int(nphi), mass=probe.torsional_inertia
    )
    bend = SineDVR(
        bounds[2, 0], bounds[2, 1], int(ntheta), mass=probe.bend_inertia
    )
    axes = (radial.x * au2angstrom, torsion.x.copy(), bend.x.copy())
    return axes, (radial, torsion, bend)


def kinetic_terms(dvrs):
    dimensions = tuple(dvr.npts for dvr in dvrs)
    identities = tuple(np.eye(size) for size in dimensions)
    terms = []
    for active, dvr in enumerate(dvrs):
        factors = list(identities)
        factors[active] = dvr.t()
        terms.append((1.0, tuple(factors)))
    return tuple(terms)


def cap_profile(radial, start, strength, order=4):
    """Return a polynomial complex-absorbing-potential strength on a DVR."""
    return polynomial_cap(radial, start, strength, order)


def cap_operators(axes, profile, nstates=3):
    """Build the total and channel-resolved radial CAP MPOs."""
    profile = np.asarray(profile, dtype=float)
    dimensions = tuple(len(axis) for axis in axes)
    if profile.shape != (dimensions[0],):
        raise ValueError("CAP profile must match the radial DVR")
    nuclear = [np.diag(profile)]
    nuclear.extend(np.eye(size) for size in dimensions[1:])

    def product(electronic):
        local = (*nuclear, np.asarray(electronic, dtype=complex))
        return MPO([matrix.reshape(1, 1, *matrix.shape) for matrix in local])

    total = product(np.eye(int(nstates)))
    channels = []
    for state in range(int(nstates)):
        projector = np.zeros((int(nstates), int(nstates)))
        projector[state, state] = 1.0
        channels.append(product(projector))
    return total, tuple(channels)


def cumulative_cap_yield(times, expectations):
    r"""Integrate $2\langle W_s\rangle$ to obtain state-resolved CAP yields."""
    times = np.asarray(times, dtype=float)
    expectations = np.asarray(expectations, dtype=float)
    if expectations.shape[0] != len(times):
        raise ValueError("CAP expectations and times must have the same length")
    result = np.zeros_like(expectations)
    if len(times) > 1:
        increments = (
            (times[1:] - times[:-1])[:, None]
            * (expectations[1:] + expectations[:-1])
        )
        result[1:] = np.cumsum(increments, axis=0)
    return result


def gaussian_nuclear_packet(axes):
    centers = CHART.equilibrium[:3]
    widths = np.asarray((0.040, 0.085, np.deg2rad(1.8)))
    factors = [
        np.exp(-0.25 * ((axis - center) / width) ** 2)
        for axis, center, width in zip(axes, centers, widths)
    ]
    nuclear = factors[0]
    for factor in factors[1:]:
        nuclear = np.multiply.outer(nuclear, factor)
    return nuclear / np.linalg.norm(nuclear)


def condon_packet(nuclear, state=1, nstates=3):
    """Apply a state-selective constant transition dipole to a nuclear state."""
    nuclear = np.asarray(nuclear, dtype=complex)
    values = np.zeros((*nuclear.shape, int(nstates)), dtype=complex)
    values[..., int(state)] = nuclear
    return values / np.linalg.norm(values)


def initial_packet(axes, state=1):
    """Return the earlier hand-built Gaussian packet for comparison runs."""
    return condon_packet(gaussian_nuclear_packet(axes), state=state)


def vibrational_ground_state(kinetic, potential, *, guess=None, tolerance=1.0e-11):
    """Solve the scalar multidimensional DVR ground state without forming dense H."""
    potential = np.asarray(potential)
    kinetic = tuple(np.asarray(term) for term in kinetic)
    shape = potential.shape
    if len(kinetic) != potential.ndim:
        raise ValueError("one kinetic matrix is required per potential dimension")
    if any(term.shape != (shape[axis], shape[axis]) for axis, term in enumerate(kinetic)):
        raise ValueError("kinetic matrix dimensions must match the potential grid")
    dtype = np.result_type(potential, *kinetic)

    def apply(vector):
        state = np.asarray(vector).reshape(shape)
        result = potential * state
        for axis, term in enumerate(kinetic):
            moved = np.moveaxis(state, axis, 0)
            transformed = np.tensordot(term, moved, axes=(1, 0))
            result = result + np.moveaxis(transformed, 0, axis)
        return np.asarray(result, dtype=dtype).reshape(-1)

    operator = LinearOperator(
        (potential.size, potential.size), matvec=apply, dtype=dtype
    )
    v0 = None if guess is None else np.asarray(guess, dtype=dtype).reshape(-1)
    energies, vectors = eigsh(
        operator, k=1, which="SA", v0=v0, tol=float(tolerance)
    )
    state = vectors[:, 0].reshape(shape)
    largest = np.unravel_index(np.argmax(np.abs(state)), shape)
    phase = state[largest] / max(abs(state[largest]), np.finfo(float).tiny)
    state = state / phase
    state /= np.linalg.norm(state)
    residual = np.linalg.norm(apply(state.reshape(-1)) - energies[0] * state.reshape(-1))
    return float(np.real(energies[0])), state, float(residual)


def _parallel_transport_phase(vectors, anchor):
    """Choose a continuous phase for one adiabatic vector on a product grid."""
    vectors = np.asarray(vectors, dtype=complex)
    grid_shape = vectors.shape[:-1]
    anchor = tuple(map(int, anchor))
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
                if abs(overlap) > 1.0e-10:
                    raw = raw * overlap.conjugate() / abs(overlap)
                else:
                    pivot = raw[np.argmax(np.abs(raw))]
                    raw = raw * pivot.conjugate() / abs(pivot)
                result[child] = raw
                visited[child] = True
                pending.append(child)
    return result


def ground_condon_packet(
    axes, dvrs, p_hamiltonian, state=1, *, electronic="adiabatic"
):
    """Build a vertical Condon packet from the fitted S0 vibrational ground state."""
    p_hamiltonian = np.asarray(p_hamiltonian)
    expected = (*tuple(len(axis) for axis in axes), 3, 3)
    if p_hamiltonian.shape != expected:
        raise ValueError(f"P-gauge Hamiltonian must have shape {expected}")
    adiabatic_energies, adiabatic_vectors = np.linalg.eigh(p_hamiltonian)
    ground_surface = adiabatic_energies[..., 0]
    energy, nuclear, residual = vibrational_ground_state(
        tuple(dvr.t() for dvr in dvrs),
        ground_surface,
        guess=gaussian_nuclear_packet(axes),
    )
    if electronic == "adiabatic":
        anchor = tuple(
            int(np.argmin(np.abs(axis - center)))
            for axis, center in zip(axes, CHART.equilibrium[:3])
        )
        electronic_vector = _parallel_transport_phase(
            adiabatic_vectors[..., :, int(state)], anchor
        )
        packet = nuclear[..., None] * electronic_vector
        packet /= np.linalg.norm(packet)
    elif electronic == "p-channel":
        packet = condon_packet(nuclear, state=state)
    else:
        raise ValueError("electronic must be 'adiabatic' or 'p-channel'")
    adiabatic_amplitudes = np.einsum(
        "...pa,...p->...a", adiabatic_vectors.conj(), packet, optimize=True
    )
    adiabatic_population = np.sum(
        np.abs(adiabatic_amplitudes) ** 2,
        axis=tuple(range(packet.ndim - 1)),
    )
    weight = np.sum(np.abs(packet) ** 2, axis=-1)
    moments = []
    widths = []
    edge_probabilities = []
    for axis, values in enumerate(axes):
        inactive = tuple(index for index in range(weight.ndim) if index != axis)
        marginal = np.sum(weight, axis=inactive)
        mean = float(np.dot(values, marginal))
        moments.append(mean)
        widths.append(float(np.sqrt(np.dot((values - mean) ** 2, marginal))))
        edge_probabilities.append(float(marginal[0] + marginal[-1]))
    info = {
        "kind": "state-selective-condon",
        "nuclear_state": "fitted-S0-vibrational-ground-state",
        "bright_adiabatic_state": int(state),
        "electronic_preparation": str(electronic),
        "ground_vibrational_energy_hartree": energy,
        "ground_eigenpair_residual": residual,
        "coordinate_means": moments,
        "coordinate_standard_deviations": widths,
        "edge_node_probabilities": edge_probabilities,
        "initial_local_adiabatic_populations": adiabatic_population,
        "transition_dipole_model": "state-selective Condon amplitude in the local adiabatic basis",
    }
    return packet, info


def validation_design(bounds, axes, count, seed):
    rng = np.random.default_rng(int(seed))
    coordinates = rng.uniform(bounds[:, 0], bounds[:, 1], size=(int(count), 3))
    edge_coordinates = []
    for axis, grid in enumerate(axes):
        step = float(np.median(np.diff(grid)))
        left = rng.uniform(bounds[:, 0], bounds[:, 1], size=(int(count) // 3, 3))
        left[:, axis] = rng.uniform(
            bounds[axis, 0], bounds[axis, 1] - step, size=len(left)
        )
        right = left.copy()
        right[:, axis] += step
        edge_coordinates.append((left, right))
    return coordinates, tuple(edge_coordinates)


def dvr_validation_design(axes):
    mesh = np.meshgrid(*axes, indexing="ij")
    coordinates = np.stack([value.reshape(-1) for value in mesh], axis=1)
    edges = []
    for active in range(len(axes)):
        left_axes = list(axes)
        right_axes = list(axes)
        left_axes[active] = left_axes[active][:-1]
        right_axes[active] = right_axes[active][1:]
        left_mesh = np.meshgrid(*left_axes, indexing="ij")
        right_mesh = np.meshgrid(*right_axes, indexing="ij")
        edges.append(
            (
                np.stack([value.reshape(-1) for value in left_mesh], axis=1),
                np.stack([value.reshape(-1) for value in right_mesh], axis=1),
            )
        )
    return coordinates, tuple(edges)


def distillation_metrics(fit, coordinates, edges):
    neural_h = _predict(fit.neural_energy, coordinates)
    neural_y = _predict(fit.neural_feature, coordinates)
    ftt_h = fit.energy.predict(coordinates)
    ftt_y = fit.feature.predict(coordinates)
    h_error = ftt_h - neural_h
    spectral = np.abs(np.linalg.eigvalsh(ftt_h) - np.linalg.eigvalsh(neural_h))
    gram = ftt_y.conj().swapaxes(-1, -2) @ ftt_y
    isometry = np.linalg.norm(gram - np.eye(3), axis=(1, 2))
    link_relative = []
    for left, right in edges:
        neural_left = _predict(fit.neural_feature, left)
        neural_right = _predict(fit.neural_feature, right)
        reference = neural_left.conj().swapaxes(-1, -2) @ neural_right
        ftt_left = fit.feature.predict(left)
        ftt_right = fit.feature.predict(right)
        predicted = ftt_left.conj().swapaxes(-1, -2) @ ftt_right
        link_relative.extend(
            np.linalg.norm(predicted - reference, axis=(1, 2))
            / np.maximum(np.linalg.norm(reference, axis=(1, 2)), 1.0e-15)
        )
    link_relative = np.asarray(link_relative)
    reflected = coordinates.copy()
    reflected[:, 1] *= -1.0
    reflection = np.diag((1.0, 1.0, -1.0))
    reflected_h = fit.energy.predict(reflected)
    transformed_h = np.einsum(
        "ab,nbc,cd->nad", reflection, ftt_h, reflection, optimize=True
    )
    hermiticity = np.linalg.norm(
        ftt_h - ftt_h.conj().swapaxes(-1, -2), axis=(1, 2)
    )
    return {
        "energy_matrix_rmse_mev": float(
            np.sqrt(np.mean(np.abs(h_error) ** 2)) * HARTREE_TO_EV * 1000.0
        ),
        "energy_matrix_max_mev": float(
            np.max(np.linalg.norm(h_error, axis=(1, 2))) * HARTREE_TO_EV * 1000.0
        ),
        "energy_spectral_rmse_mev": float(
            np.sqrt(np.mean(spectral**2)) * HARTREE_TO_EV * 1000.0
        ),
        "energy_spectral_max_mev": float(np.max(spectral) * HARTREE_TO_EV * 1000.0),
        "feature_relative_error": float(
            np.linalg.norm(ftt_y - neural_y) / np.linalg.norm(neural_y)
        ),
        "feature_isometry_rms": float(np.sqrt(np.mean(isometry**2))),
        "feature_isometry_max": float(np.max(isometry)),
        "link_relative_rms": float(np.sqrt(np.mean(link_relative**2))),
        "link_relative_max": float(np.max(link_relative)),
        "maximum_hermiticity_defect": float(np.max(hermiticity)),
        "maximum_reflection_covariance_defect": float(
            np.max(np.linalg.norm(reflected_h - transformed_h, axis=(1, 2)))
        ),
    }


def distill_rank_scan(fit, args, axes, bounds):
    if args.ftt_method == "dvr-grid":
        validation, edges = dvr_validation_design(axes)
        shape = tuple(len(axis) for axis in axes)
        energy_values = _predict(fit.neural_energy, validation).reshape(
            *shape, 3, 3
        )
        feature_values = _predict(fit.neural_feature, validation).reshape(
            *shape, fit.feature_rank, 3
        )
    else:
        validation, edges = validation_design(
            bounds, axes, args.validation_points, args.seed + 101
        )
    results = []
    for rank in args.ftt_ranks:
        started = time.perf_counter()
        if args.ftt_method == "dvr-grid":
            common = {
                "degrees": tuple(len(axis) - 1 for axis in axes),
                "rank": rank,
                "bounds": tuple((float(axis[0]), float(axis[-1])) for axis in axes),
                "normalization": "frobenius",
            }
            fit.energy = FunctionalTT(
                **common, hermitian=True
            ).fit_grid(axes, energy_values)
            fit.feature = FunctionalTT(
                **common, hermitian=False
            ).fit_grid(axes, feature_values)
            fit.links = None
        else:
            fit.distill_y(
                rank=rank,
                degree=args.ftt_degree,
                method="cross",
                cross_points=args.cross_points,
                cross_sweeps=args.cross_sweeps,
                cross_rtol=args.cross_rtol,
                cross_validation=args.cross_validation,
                validation_points=args.validation_points,
                seed=args.seed + rank,
            )
        metrics = distillation_metrics(fit, validation, edges)
        metrics.update(
            rank=int(rank),
            seconds=time.perf_counter() - started,
            energy_ranks=list(map(int, fit.energy.ranks_)),
            feature_ranks=list(map(int, fit.feature.ranks_)),
            method=args.ftt_method,
            cross=(
                fit.info["distillation"].get("cross")
                if args.ftt_method == "cross" else None
            ),
        )
        energy_path = args.output / f"energy_rank{rank}.npz"
        feature_path = args.output / f"feature_rank{rank}.npz"
        fit.energy.save(energy_path)
        fit.feature.save(feature_path)
        metrics["energy_model"] = str(energy_path)
        metrics["feature_model"] = str(feature_path)
        results.append(metrics)
        print(
            f"[FTT rank {rank}] energy={metrics['energy_spectral_rmse_mev']:.3f} meV, "
            f"link={metrics['link_relative_rms']:.3e}, "
            f"isometry={metrics['feature_isometry_max']:.3e}",
            flush=True,
        )
    candidates = [
        result for result in results
        if result["rank"] >= args.minimum_dynamics_ftt_rank
        and result["energy_spectral_rmse_mev"] <= 1.0
        and result["energy_spectral_max_mev"] <= 5.0
        and result["link_relative_rms"] <= 1.0e-3
        and result["link_relative_max"] <= 5.0e-3
        and result["feature_isometry_max"] <= 5.0e-3
    ]
    chosen = min(candidates, key=lambda item: item["rank"]) if candidates else results[-1]
    fit.energy = FunctionalTT.load(chosen["energy_model"])
    fit.feature = FunctionalTT.load(chosen["feature_model"])
    fit.links = None
    return results, chosen


def load_rank_scan(fit, output, rank):
    summary_path = output / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"cannot reuse FTT rank {rank}: missing {summary_path}"
        )
    distillation = json.loads(summary_path.read_text())["distillation"]
    matches = [item for item in distillation if int(item["rank"]) == int(rank)]
    if not matches:
        available = [int(item["rank"]) for item in distillation]
        raise ValueError(f"FTT rank {rank} is unavailable; found {available}")
    chosen = matches[0]
    fit.energy = FunctionalTT.load(chosen["energy_model"])
    fit.feature = FunctionalTT.load(chosen["feature_model"])
    fit.links = None
    return distillation, chosen


def directional_link_ftts(feature, axes, rank, output):
    shape = tuple(len(axis) for axis in axes)
    coordinates, _ = dvr_validation_design(axes)
    feature_shape = tuple(map(int, feature.output_shape_))
    values = feature.predict(coordinates).reshape(*shape, *feature_shape)
    models = []
    records = []
    for active in range(len(axes)):
        left_slice = [slice(None)] * len(axes)
        right_slice = [slice(None)] * len(axes)
        left_slice[active] = slice(None, -1)
        right_slice[active] = slice(1, None)
        left = values[tuple(left_slice)]
        right = values[tuple(right_slice)]
        links = np.einsum(
            "...ra,...rb->...ab", left.conj(), right, optimize=True
        )
        edge_axes = list(axes)
        edge_axes[active] = 0.5 * (
            edge_axes[active][:-1] + edge_axes[active][1:]
        )
        model = FunctionalTT(
            degrees=tuple(len(axis) - 1 for axis in edge_axes),
            rank=int(rank),
            bounds=tuple(
                (float(axis[0]), float(axis[-1])) for axis in edge_axes
            ),
            normalization="frobenius",
            hermitian=False,
        ).fit_grid(edge_axes, links)
        mesh = np.meshgrid(*edge_axes, indexing="ij")
        edge_coordinates = np.stack(
            [value.reshape(-1) for value in mesh], axis=1
        )
        predicted = model.predict(edge_coordinates).reshape(links.shape)
        errors = np.linalg.norm(predicted - links, axis=(-2, -1))
        scales = np.maximum(np.linalg.norm(links, axis=(-2, -1)), 1.0e-15)
        relative = errors / scales
        path = output / f"link_axis{active}_rank{rank}.npz"
        model.save(path)
        records.append(
            {
                "axis": active,
                "rank": int(rank),
                "ranks": list(map(int, model.ranks_)),
                "relative_rms": float(np.sqrt(np.mean(relative**2))),
                "relative_max": float(np.max(relative)),
                "model": str(path),
            }
        )
        models.append(model)
    return tuple(models), records


def run_dynamics(fit, args, axes, dvrs):
    started = time.perf_counter()
    links, link_info = directional_link_ftts(
        fit.feature, axes, args.directional_link_rank, args.output
    )
    fit.links = links
    fit.feature = None
    print(
        "[link FTT] "
        + ", ".join(
            f"q{item['axis']}={item['relative_rms']:.2e}"
            for item in link_info
        ),
        flush=True,
    )
    driver = TTLDR.from_fit(
        fit,
        grids=axes,
        keo=kinetic_terms(dvrs),
        overlap_rank=args.overlap_rank,
        potential_rank=args.potential_rank,
        operator_rank=args.operator_rank,
        fitted_kinetic_backend="link-mpo",
    )
    print(
        f"[LDR setup] {time.perf_counter() - started:.1f} s, "
        f"operator ranks={driver.operator_ranks}",
        flush=True,
    )
    if args.initial_condition == "ground-condon":
        coordinates, _ = dvr_validation_design(axes)
        p_hamiltonian = fit.energy.predict(coordinates).reshape(
            *tuple(len(axis) for axis in axes), driver.nstates, driver.nstates
        )
        initial, initial_info = ground_condon_packet(
            axes,
            dvrs,
            p_hamiltonian,
            args.bright_state,
            electronic=args.initial_electronic_state,
        )
        if initial_info["edge_node_probabilities"][0] > args.maximum_initial_edge_probability:
            raise RuntimeError(
                "the S0 vibrational ground state reaches the inner radial DVR wall: "
                f"edge-node probability={initial_info['edge_node_probabilities'][0]:.3e}, "
                f"limit={args.maximum_initial_edge_probability:.3e}; extend the fitted "
                "R_OH domain inward before propagating this initial state"
            )
        print(
            f"[initial] S0 vibrational ground state, "
            f"E={initial_info['ground_vibrational_energy_hartree']:.8f} Eh, "
            f"residual={initial_info['ground_eigenpair_residual']:.2e}, "
            f"adiabatic P={np.round(initial_info['initial_local_adiabatic_populations'], 6)}",
            flush=True,
        )
    else:
        initial = initial_packet(axes, args.bright_state)
        initial_info = {
            "kind": "hand-built-gaussian",
            "nuclear_state": "product Gaussian",
            "bright_p_channel": int(args.bright_state),
        }
    projectors = driver.projectors()
    use_cap = args.cap_strength > 0.0
    profile = None
    if use_cap:
        profile = cap_profile(
            axes[0], args.cap_start, args.cap_strength, args.cap_order
        )
        print(
            f"[split CAP] start={args.cap_start:.3f} A, "
            f"Wmax={np.max(profile):.4f} Eh, order={args.cap_order}",
            flush=True,
        )
    results = []
    for rank in args.state_ranks:
        state = driver.state(initial, max_rank=rank)
        driver.run(
            state,
            dt=float(args.tmax_fs / args.steps / au2fs),
            steps=args.steps,
            interval=1,
            max_bond=rank,
            integrator="tdvp2",
            cutoff=args.cutoff,
            krylov_dim=args.krylov_dim,
            krylov_tol=args.krylov_tol,
            normalize=not use_cap,
            progress=args.progress,
            workers=args.workers,
            e_ops=projectors,
            absorber=profile,
            absorber_site=0,
        )
        final = driver.dense(driver.final_state)
        probability = np.abs(final) ** 2
        radial_absolute = probability.sum(axis=(1, 2, 3))
        final_norm = float(np.sum(radial_absolute))
        radial = radial_absolute / max(final_norm, np.finfo(float).tiny)
        populations = np.asarray(driver.populations)
        cap_expectations = (
            np.asarray(driver.absorber_expectations)
            if use_cap else np.zeros((len(populations), driver.nstates))
        )
        cap_yields = (
            np.asarray(driver.absorber_yields)
            if use_cap else np.zeros((len(populations), driver.nstates))
        )
        absorbed = (
            np.asarray(driver.absorbed_probabilities)
            if use_cap else 1.0 - np.asarray(driver.norms)
        )
        absorption_closure = (
            np.asarray(driver.absorption_closure)
            if use_cap else np.sum(cap_yields, axis=1) - absorbed
        )
        result = {
            "rank": int(rank),
            "times_fs": np.asarray(driver.times) * au2fs,
            "populations": populations,
            "norms": np.asarray(driver.norms),
            "radial": radial,
            "radial_absolute": radial_absolute,
            "cap_expectations": cap_expectations,
            "cap_yields": cap_yields,
            "final_norm": final_norm,
            "final_absorbed_probability": float(absorbed[-1]),
            "maximum_absorption_closure_defect": float(
                np.max(np.abs(absorption_closure))
            ),
            "maximum_norm_drift": float(np.max(np.abs(driver.norms - 1.0))),
            "maximum_population_sum_defect": float(
                np.max(np.abs(np.sum(populations, axis=1) - driver.norms))
            ),
            "final_mean_r_angstrom": float(np.dot(axes[0], radial)),
            "outer_two_point_probability": float(np.sum(radial_absolute[-2:])),
            "final_state_ranks": list(map(int, driver.final_state.bond_orders())),
        }
        results.append(result)
        print(
            f"[TTLDR rank {rank}] final norm={result['final_norm']:.6f}, "
            f"absorbed={result['final_absorbed_probability']:.3e}, "
            f"<R>={result['final_mean_r_angstrom']:.4f} A, "
            f"P={np.round(result['populations'][-1], 5)}, "
            f"Y={np.round(result['cap_yields'][-1], 5)}",
            flush=True,
        )
    _update_rank_convergence(results)
    cap_info = {
        "enabled": use_cap,
        "start_angstrom": args.cap_start if use_cap else None,
        "strength_hartree": args.cap_strength if use_cap else None,
        "order": args.cap_order if use_cap else None,
        "profile": profile,
    }
    return driver, initial, initial_info, results, link_info, cap_info


def _update_rank_convergence(results):
    results.sort(key=lambda item: int(item["rank"]))
    highest = results[-1]
    for result in results[:-1]:
        result["population_difference_from_highest"] = float(
            np.max(np.abs(result["populations"] - highest["populations"]))
        )
        result["radial_l1_difference_from_highest"] = float(
            np.sum(np.abs(result["radial"] - highest["radial"]))
        )
    highest["population_difference_from_highest"] = 0.0
    highest["radial_l1_difference_from_highest"] = 0.0


def load_prior_dynamics(path, args, axes):
    """Load compatible lower-rank trajectories for an appended rank scan."""
    summary = json.loads(Path(path).read_text())
    domain = summary["domain"]
    expected_shape = [len(axis) for axis in axes]
    if domain["dvr_shape"] != expected_shape:
        raise ValueError("prior dynamics used a different DVR shape")
    if not np.isclose(domain["time_fs"], args.tmax_fs) or int(domain["steps"]) != args.steps:
        raise ValueError("prior dynamics used a different time grid")
    prior_initial = domain.get("initial_condition", "gaussian")
    if prior_initial != args.initial_condition:
        raise ValueError("prior dynamics used a different initial condition")
    prior_electronic = domain.get("initial_electronic_state", "p-channel")
    if prior_electronic != args.initial_electronic_state:
        raise ValueError("prior dynamics used a different initial electronic state")
    if domain.get("bright_initial_channel") != f"P{args.bright_state}":
        raise ValueError("prior dynamics used a different bright channel")
    prior_cap = summary.get("cap", {})
    expected_cap = args.cap_strength > 0.0
    if bool(prior_cap.get("enabled", False)) != expected_cap:
        raise ValueError("prior dynamics used a different CAP setting")
    if expected_cap and not (
        np.isclose(prior_cap["start_angstrom"], args.cap_start)
        and np.isclose(prior_cap["strength_hartree"], args.cap_strength)
        and int(prior_cap["order"]) == args.cap_order
    ):
        raise ValueError("prior dynamics used different CAP parameters")
    array_fields = (
        "times_fs", "populations", "norms", "radial", "radial_absolute",
        "cap_expectations", "cap_yields",
    )
    results = []
    for record in summary["dynamics"]:
        record = dict(record)
        for field in array_fields:
            record[field] = np.asarray(record[field])
        results.append(record)
    return results


def plot_results(output, distillation, chosen, dynamics, axes, initial, cap_info):
    figure, panels = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    ranks = [item["rank"] for item in distillation]
    panels[0, 0].semilogy(
        ranks, [item["energy_spectral_rmse_mev"] for item in distillation],
        "o-", color=COLORS[0], label="energy RMS (meV)",
    )
    panels[0, 0].semilogy(
        ranks, [1.0e3 * item["link_relative_rms"] for item in distillation],
        "s--", color=COLORS[1], label=r"$10^3\times$ link RMS",
    )
    panels[0, 0].axvline(chosen["rank"], color="0.4", ls=":", lw=1.0)
    panels[0, 0].set(
        xlabel="maximum FTT rank", ylabel="distillation error",
        title=r"MACE $\rightarrow$ FTT",
    )
    panels[0, 0].legend(frameon=False, fontsize=8)

    highest = dynamics[-1]
    yield_axis = panels[0, 1].twinx() if cap_info["enabled"] else None
    for state, color in enumerate(COLORS):
        panels[0, 1].plot(
            highest["times_fs"], highest["populations"][:, state],
            color=color, lw=1.5, label=f"P{state}",
        )
        if yield_axis is not None:
            yield_axis.plot(
                highest["times_fs"], highest["cap_yields"][:, state],
                color=color, lw=1.1, ls="--", label=f"Y{state}",
            )
    panels[0, 1].set(
        xlabel="time (fs)", ylabel="P-gauge population", ylim=(-0.02, 1.02),
        title=f"TTLDR dynamics (rank {highest['rank']})",
    )
    if yield_axis is None:
        panels[0, 1].legend(frameon=False, ncol=3, fontsize=8)
    else:
        yield_axis.set(
            ylabel="cumulative CAP yield",
            ylim=(0.0, 1.12 * np.max(highest["cap_yields"])),
        )
        handles, labels = panels[0, 1].get_legend_handles_labels()
        other_handles, other_labels = yield_axis.get_legend_handles_labels()
        panels[0, 1].legend(
            handles + other_handles, labels + other_labels,
            frameon=False, ncol=3, fontsize=8,
        )

    state_ranks = [item["rank"] for item in dynamics]
    panels[1, 0].semilogy(
        state_ranks,
        [max(item["population_difference_from_highest"], 1.0e-16) for item in dynamics],
        "o-", color=COLORS[0], label="population max error",
    )
    panels[1, 0].semilogy(
        state_ranks,
        [max(item["radial_l1_difference_from_highest"], 1.0e-16) for item in dynamics],
        "s--", color=COLORS[1], label="radial $L^1$ error",
    )
    panels[1, 0].semilogy(
        state_ranks,
        [max(item["maximum_absorption_closure_defect"], 1.0e-16) for item in dynamics],
        "^:", color=COLORS[2], label="CAP closure",
    )
    panels[1, 0].set(xlabel="maximum MPS rank", ylabel="convergence metric", title="Tensor-dynamics convergence")
    panels[1, 0].legend(frameon=False, fontsize=8)

    initial_radial = np.abs(initial) ** 2
    initial_radial = initial_radial.sum(axis=(1, 2, 3))
    initial_radial /= initial_radial.sum()
    panels[1, 1].plot(axes[0], initial_radial, "--", color="0.35", label="initial")
    panels[1, 1].plot(axes[0], highest["radial"], "o-", ms=3.2, color=COLORS[0], label="final")
    if cap_info["enabled"]:
        cap_scaled = np.asarray(cap_info["profile"]).copy()
        cap_scaled /= max(float(np.max(cap_scaled)), np.finfo(float).tiny)
        panels[1, 1].fill_between(
            axes[0], 0.0, cap_scaled * max(float(np.max(highest["radial"])), 1.0e-12),
            color=COLORS[1], alpha=0.13, label="CAP region",
        )
    panels[1, 1].set(xlabel=r"$R_{OH}$ ($\AA$)", ylabel="radial probability", title="Nuclear wavepacket")
    panels[1, 1].legend(frameon=False)
    for label, panel in zip("abcd", panels.flat):
        panel.text(0.02, 0.96, label, transform=panel.transAxes, va="top", fontweight="bold")
        panel.grid(alpha=0.18)
    png = output / "phenol_sa6_3d_ftt_ttldr.png"
    pdf = output / "phenol_sa6_3d_ftt_ttldr.pdf"
    figure.savefig(png, dpi=350)
    figure.savefig(pdf)
    plt.close(figure)
    return png, pdf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("/private/tmp/phenol_sa6_3d_mace_y_dynamics_20260821/phenol_sa6_3d_mace_y.pt"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("/private/tmp/phenol_sa6_3d_ftt_ttldr_20260821"),
    )
    parser.add_argument("--nr", type=int, default=25)
    parser.add_argument("--nphi", type=int, default=15)
    parser.add_argument("--ntheta", type=int, default=11)
    parser.add_argument("--ftt-ranks", type=_integers, default=(12, 24, 32))
    parser.add_argument(
        "--ftt-method", choices=("dvr-grid", "cross"), default="dvr-grid",
    )
    parser.add_argument("--minimum-dynamics-ftt-rank", type=int, default=24)
    parser.add_argument("--ftt-degree", type=int, default=10)
    parser.add_argument("--cross-points", type=int, default=13)
    parser.add_argument("--cross-sweeps", type=int, default=8)
    parser.add_argument("--cross-rtol", type=float, default=1.0e-8)
    parser.add_argument("--cross-validation", type=int, default=128)
    parser.add_argument("--validation-points", type=int, default=384)
    parser.add_argument("--state-ranks", type=_integers, default=(16, 24, 32))
    parser.add_argument("--bright-state", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument(
        "--initial-condition",
        choices=("ground-condon", "gaussian"),
        default="ground-condon",
        help="use the fitted S0 vibrational ground state or the earlier Gaussian pilot",
    )
    parser.add_argument(
        "--initial-electronic-state",
        choices=("adiabatic", "p-channel"),
        default="adiabatic",
        help="prepare local S1 or a constant P-gauge channel in the Condon packet",
    )
    parser.add_argument(
        "--maximum-initial-edge-probability",
        type=float,
        default=5.0e-3,
        help="reject a physical packet that is not negligible at the inner radial DVR edge",
    )
    parser.add_argument("--tmax-fs", type=float, default=10.0)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--overlap-rank", type=int, default=16)
    parser.add_argument("--directional-link-rank", type=int, default=16)
    parser.add_argument("--potential-rank", type=int, default=32)
    parser.add_argument("--operator-rank", type=int, default=32)
    parser.add_argument("--cutoff", type=float, default=1.0e-10)
    parser.add_argument("--krylov-dim", type=int, default=16)
    parser.add_argument("--krylov-tol", type=float, default=1.0e-11)
    parser.add_argument(
        "--workers", type=int, default=4,
        help="parallelize compact MPO components in TDVPEngine",
    )
    parser.add_argument("--cap-start", type=float, default=2.45)
    parser.add_argument("--cap-strength", type=float, default=0.0)
    parser.add_argument("--cap-order", type=int, default=4)
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--distill-only", action="store_true")
    parser.add_argument(
        "--reuse-ftt-rank", type=int,
        help="reuse this rank from OUTPUT/summary.json instead of redistilling",
    )
    parser.add_argument(
        "--reuse-dynamics-summary", type=Path,
        help="append new state ranks to compatible trajectories in this summary",
    )
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    fit = MACE.load(args.checkpoint, geometry, device="cpu", distill=False)
    bounds = np.asarray([(grid[0], grid[-1]) for grid in fit.grids], dtype=float)
    axes, dvrs = build_dvrs(args.nr, args.nphi, args.ntheta, bounds)
    if args.reuse_ftt_rank is None:
        distillation, chosen = distill_rank_scan(fit, args, axes, bounds)
    else:
        distillation, chosen = load_rank_scan(
            fit, args.output, args.reuse_ftt_rank
        )
    if args.distill_only:
        dynamics = []
        driver = initial = initial_info = None
        link_info = []
        figures = {}
    else:
        driver, initial, initial_info, dynamics, link_info, cap_info = run_dynamics(
            fit, args, axes, dvrs
        )
        if args.reuse_dynamics_summary is not None:
            prior = load_prior_dynamics(args.reuse_dynamics_summary, args, axes)
            new_ranks = {int(item["rank"]) for item in dynamics}
            dynamics = [
                item for item in prior if int(item["rank"]) not in new_ranks
            ] + dynamics
            _update_rank_convergence(dynamics)
        png, pdf = plot_results(
            args.output, distillation, chosen, dynamics, axes, initial, cap_info
        )
        figures = {"diagnostics": str(png), "diagnostics_pdf": str(pdf)}
        np.savez_compressed(
            args.output / "phenol_sa6_3d_ftt_ttldr.npz",
            r_oh=axes[0], phi=axes[1], theta=axes[2],
            initial=initial,
            **{
                f"times_rank{item['rank']}": item["times_fs"]
                for item in dynamics
            },
            **{
                f"populations_rank{item['rank']}": item["populations"]
                for item in dynamics
            },
            **{f"radial_rank{item['rank']}": item["radial"] for item in dynamics},
            **{
                f"cap_yields_rank{item['rank']}": item["cap_yields"]
                for item in dynamics
            },
        )
    distill_passed = bool(
        chosen["energy_spectral_rmse_mev"] <= 1.0
        and chosen["energy_spectral_max_mev"] <= 5.0
        and chosen["link_relative_rms"] <= 1.0e-3
        and chosen["link_relative_max"] <= 5.0e-3
        and chosen["feature_isometry_max"] <= 5.0e-3
    )
    dynamics_passed = bool(
        not dynamics
        or (
            max(item["maximum_population_sum_defect"] for item in dynamics) <= 1.0e-5
            and max(item["maximum_absorption_closure_defect"] for item in dynamics) <= 5.0e-3
            and (
                len(dynamics) == 1
                or dynamics[-2]["population_difference_from_highest"] <= 5.0e-3
            )
            and (
                len(dynamics) == 1
                or dynamics[-2]["radial_l1_difference_from_highest"] <= 2.0e-2
            )
            and dynamics[-1]["outer_two_point_probability"] <= 1.0e-2
        )
    )
    summary = {
        "passed": bool(distill_passed and dynamics_passed),
        "gates": {
            "ftt_distillation": distill_passed,
            "ttl_dr_dynamics": dynamics_passed,
        },
        "domain": {
            "bounds": bounds,
            "dvr_shape": [len(axis) for axis in axes],
            "bright_initial_channel": f"P{args.bright_state}",
            "initial_condition": args.initial_condition,
            "initial_electronic_state": args.initial_electronic_state,
            "time_fs": args.tmax_fs,
            "steps": args.steps,
        },
        "initial_condition": initial_info,
        "distillation": distillation,
        "chosen_ftt_rank": chosen["rank"],
        "operator_ranks": None if driver is None else driver.operator_ranks,
        "directional_link_distillation": link_info,
        "cap": {} if not dynamics else cap_info,
        "dynamics": dynamics,
        "figures": figures,
        "checkpoint": str(args.checkpoint),
        "seconds": time.perf_counter() - started,
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2) + "\n")
    print(json.dumps(_jsonable(summary), indent=2))


if __name__ == "__main__":
    main()
