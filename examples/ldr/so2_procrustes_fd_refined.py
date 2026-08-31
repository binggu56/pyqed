#!/usr/bin/env python3
"""Run interpolated, sparse covariant-FD SO2 dynamics on a refined grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import expm, logm
from scipy.sparse import coo_matrix, csr_matrix, kron

from examples.ldr.so2_casci_cgldr import (
    DEFAULT_SCAN_DIR,
    REFERENCE_BOND,
    REFERENCE_BOND_WIDTH,
    REFERENCE_THETA_DEG,
    REFERENCE_THETA_WIDTH_DEG,
    SQRT2,
    load_so2_linked_scan,
)
from examples.ldr.so2_casci_cgldr_dense import dense_kinetic, nuclear_packet, observables
from examples.ldr.so2_casci_full_ldr import full_hamiltonian, path_overlap
from examples.ldr.so2_procrustes_dynamics import DEFAULT_GAUGE, DEFAULT_REFERENCE, propagate
from examples.ldr.so2_procrustes_fd import LocalAxis, aligned_links
from examples.ldr.so2_procrustes_gauge import local_hamiltonian
from pyqed.ldr.overlap import between, procrustes, unpack
from pyqed.units import au2fs


def refined_grid(points, factor):
    points = np.asarray(points, dtype=float)
    factor = int(factor)
    if factor < 1:
        raise ValueError("refinement factor must be positive")
    pieces = [
        np.linspace(points[index], points[index + 1], factor + 1)[:-1]
        for index in range(len(points) - 1)
    ]
    return np.concatenate((*pieces, points[-1:]))


def cell_weights(points, lower, upper):
    points = np.asarray(points, dtype=float)
    edges = np.empty(len(points) + 1)
    edges[0] = float(lower)
    edges[-1] = float(upper)
    edges[1:-1] = 0.5 * (points[:-1] + points[1:])
    weights = np.diff(edges)
    if np.any(weights <= 0.0):
        raise ValueError("grid points lie outside their quadrature domain")
    return weights


def interpolate_field(grids, values, targets, *, method="linear"):
    mesh = np.meshgrid(*targets, indexing="ij")
    points = np.stack([axis.reshape(-1) for axis in mesh], axis=-1)
    interpolator = RegularGridInterpolator(
        tuple(np.asarray(grid) for grid in grids),
        np.asarray(values),
        method=method,
        bounds_error=False,
        fill_value=None,
    )
    trailing = np.asarray(values).shape[len(grids) :]
    return interpolator(points).reshape(*(len(grid) for grid in targets), *trailing)


def polar_unitary_field(values):
    values = np.asarray(values, dtype=complex)
    flat = values.reshape(-1, values.shape[-2], values.shape[-1])
    output = np.empty_like(flat)
    for index, value in enumerate(flat):
        output[index] = procrustes(value)[0]
    return output.reshape(values.shape)


def regularized_generator(link, spacing, floor):
    rotation, positive, _singular = procrustes(link)
    values, vectors = np.linalg.eigh(positive)
    clipped = np.maximum(values, float(floor))
    regularized = rotation @ ((vectors * clipped[None, :]) @ vectors.conj().T)
    return logm(regularized) / float(spacing)


def refined_links(coarse_grids, fine_grids, links, *, floor=1.0e-6):
    shape = tuple(len(grid) for grid in coarse_grids)
    nstates = np.asarray(next(iter(links.values()))).shape[0]
    output = {}
    for axis in range(len(shape)):
        link_shape = list(shape)
        link_shape[axis] -= 1
        generators = np.empty((*link_shape, nstates, nstates), dtype=complex)
        for index in np.ndindex(*link_shape):
            spacing = coarse_grids[axis][index[axis] + 1] - coarse_grids[axis][index[axis]]
            generators[index] = regularized_generator(
                links[(axis, index)],
                spacing,
                floor,
            )
        source = list(coarse_grids)
        source[axis] = 0.5 * (
            coarse_grids[axis][:-1] + coarse_grids[axis][1:]
        )
        target = list(fine_grids)
        target[axis] = 0.5 * (fine_grids[axis][:-1] + fine_grids[axis][1:])
        sampled = interpolate_field(source, generators, target)
        for index in np.ndindex(*sampled.shape[:-2]):
            spacing = fine_grids[axis][index[axis] + 1] - fine_grids[axis][index[axis]]
            raw = expm(spacing * sampled[index])
            rotation, positive, _singular = procrustes(raw)
            values, vectors = np.linalg.eigh(positive)
            values = np.clip(values, float(floor), 1.0)
            output[(axis, index)] = rotation @ (
                (vectors * values[None, :]) @ vectors.conj().T
            )
    return output


def hermitian_krylov_step(matrix, state, interval, dimension):
    """Apply one projected exponential step for a sparse Hermitian matrix."""
    state = np.asarray(state, dtype=complex)
    norm = np.linalg.norm(state)
    dimension = min(int(dimension), len(state))
    basis = np.zeros((dimension + 1, len(state)), dtype=complex)
    hessenberg = np.zeros((dimension + 1, dimension), dtype=complex)
    basis[0] = state / norm
    used = dimension
    for column in range(dimension):
        work = matrix @ basis[column]
        for row in range(column + 1):
            value = np.vdot(basis[row], work)
            hessenberg[row, column] += value
            work -= value * basis[row]
        for row in range(column + 1):
            correction = np.vdot(basis[row], work)
            hessenberg[row, column] += correction
            work -= correction * basis[row]
        next_norm = np.linalg.norm(work)
        if next_norm < 1.0e-13 or column + 1 == dimension:
            used = column + 1
            break
        hessenberg[column + 1, column] = next_norm
        basis[column + 1] = work / next_norm
    projected = hessenberg[:used, :used]
    projected = 0.5 * (projected + projected.conj().T)
    values, vectors = np.linalg.eigh(projected)
    coefficients = vectors @ (
        np.exp(-1j * float(interval) * values) * vectors[0].conj()
    )
    return norm * (coefficients @ basis[:used])


def propagate_sparse(matrix, initial, times, dimension, *, progress=False, label="CFD"):
    states = [np.asarray(initial, dtype=complex)]
    current = states[0]
    for step, interval in enumerate(np.diff(np.asarray(times) / au2fs), start=1):
        current = hermitian_krylov_step(matrix, current, interval, dimension)
        current /= np.linalg.norm(current)
        states.append(current)
        if progress:
            print(f"[{label}] step {step}/{len(times) - 1}", flush=True)
    return np.asarray(states)


def product_terms_sparse(terms):
    matrix = None
    for _label, coefficient, *operators in terms:
        block = csr_matrix(operators[0])
        for operator in operators[1:]:
            block = kron(block, csr_matrix(operator), format="csr")
        block = coefficient * block
        matrix = block if matrix is None else matrix + block
    matrix = 0.5 * (matrix + matrix.getH())
    matrix.eliminate_zeros()
    return matrix.tocsr()


def sparse_ldr(kinetic, shape, nstates, links, local, *, average_paths=True):
    kinetic = coo_matrix(kinetic)
    rows = []
    columns = []
    data = []
    states = np.arange(nstates)
    block_rows = np.repeat(states, nstates)
    block_columns = np.tile(states, nstates)
    for left, right, coefficient in zip(kinetic.row, kinetic.col, kinetic.data):
        if left > right or abs(coefficient) < 1.0e-15:
            continue
        bra = np.unravel_index(int(left), shape)
        ket = np.unravel_index(int(right), shape)
        overlap = between(
            bra,
            ket,
            links,
            nstates=nstates,
            average_paths=average_paths,
        )
        block = coefficient * overlap
        rows.append(left * nstates + block_rows)
        columns.append(right * nstates + block_columns)
        data.append(block.reshape(-1))
        if left != right:
            rows.append(right * nstates + block_rows)
            columns.append(left * nstates + block_columns)
            data.append(block.conj().T.reshape(-1))
    for point, block in enumerate(np.asarray(local).reshape(-1, nstates, nstates)):
        rows.append(point * nstates + block_rows)
        columns.append(point * nstates + block_columns)
        data.append(block.reshape(-1))
    size = int(np.prod(shape)) * nstates
    matrix = coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(columns))),
        shape=(size, size),
    ).tocsr()
    matrix = 0.5 * (matrix + matrix.getH())
    matrix.eliminate_zeros()
    return matrix


def packet(grids, weights):
    mesh = np.meshgrid(*grids, indexing="ij")
    center = (SQRT2 * REFERENCE_BOND, np.deg2rad(REFERENCE_THETA_DEG), 0.0)
    width = (
        REFERENCE_BOND_WIDTH,
        np.deg2rad(REFERENCE_THETA_WIDTH_DEG),
        REFERENCE_BOND_WIDTH,
    )
    amplitude = np.exp(
        -0.5 * sum(((axis - origin) / sigma) ** 2 for axis, origin, sigma in zip(mesh, center, width))
    )
    quadrature = (
        weights[0][:, None, None]
        * weights[1][None, :, None]
        * weights[2][None, None, :]
    )
    values = np.sqrt(quadrature) * amplitude
    return values / np.linalg.norm(values)


def aligned_observables(states, grids, primary_transform):
    states = np.asarray(states).reshape(len(states), -1, states.shape[-1])
    transform = np.asarray(primary_transform).reshape(-1, states.shape[-1], states.shape[-1])
    primary = np.einsum("gia,tga->tgi", transform.conj(), states, optimize=True)
    probability = abs(primary) ** 2
    norm = probability.sum(axis=(1, 2))
    populations = probability.sum(axis=1) / norm[:, None]
    nuclear = probability.sum(axis=2)
    mesh = np.meshgrid(*grids, indexing="ij")
    means = np.stack(
        [(nuclear @ coordinate.reshape(-1)) / norm for coordinate in mesh],
        axis=1,
    )
    return populations, means, norm


def plot(path, times, reference, refined):
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)
    for state, color in zip((1, 2), ("#0072B2", "#D55E00")):
        axes[0].plot(times, reference[:, state], color=color, label=rf"DVR $P_{state}$")
        axes[0].plot(times, refined[:, state], "--", color=color, label=rf"Refined CFD $P_{state}$")
    axes[1].semilogy(
        times,
        np.maximum(np.max(abs(refined - reference), axis=1), 1.0e-16),
        color="#009E73",
    )
    axes[0].set(xlabel="Time (fs)", ylabel="Population", ylim=(-0.03, 1.03))
    axes[1].set(xlabel="Time (fs)", ylabel="Maximum population error")
    axes[0].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(False)
    figure.savefig(path.with_suffix(".png"), dpi=350)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--gauge", type=Path, default=DEFAULT_GAUGE)
    parser.add_argument("--scan-dir", type=Path, default=DEFAULT_SCAN_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("/private/tmp/so2_procrustes_cfd_refined"))
    parser.add_argument("--refine", type=int, default=2)
    parser.add_argument("--link-floor", type=float, default=1.0e-6)
    parser.add_argument("--initial-state", type=int, default=2)
    parser.add_argument("--time-fs", type=float, default=20.0)
    parser.add_argument("--dt-fs", type=float, default=0.5)
    parser.add_argument("--krylov-dim", type=int, default=36)
    parser.add_argument("--krylov-check-dim", type=int, default=0)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.reference) as archive:
        energies = np.asarray(archive["energies"], dtype=float)
        coarse_grids = tuple(np.asarray(archive[name], dtype=float) for name in ("qs", "theta", "qa"))
        links = unpack(archive["link_axes"], archive["link_indices"], archive["link_data"])
    with np.load(args.gauge) as archive:
        gauge = np.asarray(archive["gauge"], dtype=complex)
        primary_gauge = np.asarray(archive["primary_gauge"], dtype=complex)

    coarse_shape = energies.shape[:-1]
    nstates = energies.shape[-1]
    scan = load_so2_linked_scan(args.scan_dir)
    dense_k, reference_axes = dense_kinetic(scan, *coarse_grids)
    fine_grids = tuple(refined_grid(grid, args.refine) for grid in coarse_grids)
    bounds = tuple((axis.xmin, axis.xmax) for axis in reference_axes)
    weights = tuple(
        cell_weights(grid, lower, upper)
        for grid, (lower, upper) in zip(fine_grids, bounds)
    )
    fine_axes = tuple(
        LocalAxis(axis, weight, 3)
        for axis, weight in zip(
            (
                type("Axis", (), {"x": fine_grids[0]})(),
                type("Axis", (), {"x": fine_grids[1]})(),
                type("Axis", (), {"x": fine_grids[2]})(),
            ),
            weights,
        )
    )
    terms = scan.solver.buildK_qsqa_terms(
        fine_axes,
        sparse=True,
        symmetrize=True,
        svd_tol=0.0,
    )
    kinetic = product_terms_sparse(terms)

    flat_gauge = gauge.reshape(-1, nstates, nstates)
    coarse_local = local_hamiltonian(energies, flat_gauge).reshape(*coarse_shape, nstates, nstates)
    fine_local = interpolate_field(coarse_grids, coarse_local, fine_grids, method="cubic")
    fine_local = 0.5 * (fine_local + fine_local.swapaxes(-1, -2).conj())
    primary_transform = np.einsum(
        "...ia,...ib->...ab",
        gauge.conj(),
        primary_gauge,
        optimize=True,
    )
    fine_transform = polar_unitary_field(
        interpolate_field(coarse_grids, primary_transform, fine_grids, method="linear")
    )
    coarse_aligned_links = aligned_links(coarse_shape, links, gauge)
    fine_links = refined_links(
        coarse_grids,
        fine_grids,
        coarse_aligned_links,
        floor=args.link_floor,
    )
    started = time.perf_counter()
    hamiltonian = sparse_ldr(
        kinetic,
        tuple(len(grid) for grid in fine_grids),
        nstates,
        fine_links,
        fine_local,
    )
    build_seconds = time.perf_counter() - started
    if args.progress:
        print(
            f"[CFD] H={hamiltonian.shape}, nnz={hamiltonian.nnz:,}, "
            f"build={build_seconds:.2f} s",
            flush=True,
        )

    nuclear = packet(fine_grids, weights)
    initial = (nuclear[..., None] * fine_transform[..., args.initial_state]).reshape(-1)
    initial /= np.linalg.norm(initial)
    times = np.arange(0.0, args.time_fs + 0.5 * args.dt_fs, args.dt_fs)
    started = time.perf_counter()
    states = propagate_sparse(
        hamiltonian,
        initial,
        times,
        args.krylov_dim,
        progress=args.progress,
    )
    krylov_error = None
    if args.krylov_check_dim > args.krylov_dim:
        checked = propagate_sparse(
            hamiltonian,
            initial,
            times,
            args.krylov_check_dim,
            progress=False,
        )
        overlaps = np.abs(np.einsum("ti,ti->t", checked.conj(), states))
        krylov_error = float(np.max(np.sqrt(np.maximum(2.0 - 2.0 * overlaps, 0.0))))
        states = checked
    propagation_seconds = time.perf_counter() - started
    refined_population, refined_means, refined_norm = aligned_observables(
        states.reshape(len(times), *tuple(len(grid) for grid in fine_grids), nstates),
        fine_grids,
        fine_transform,
    )

    coarse_overlap = path_overlap(coarse_shape, links)
    exact_h = full_hamiltonian(dense_k, coarse_overlap, energies)
    coarse_packet = nuclear_packet(*coarse_grids, reference_axes)
    exact_initial = (coarse_packet[..., None] * primary_gauge[..., args.initial_state]).reshape(-1)
    exact_initial /= np.linalg.norm(exact_initial)
    exact_states = propagate(exact_h, exact_initial, times)
    exact_obs = observables(exact_states, coarse_grids, primary_gauge)
    spans = np.asarray([np.ptp(grid) for grid in coarse_grids])
    summary = {
        "coarse_grid": list(coarse_shape),
        "refined_grid": [len(grid) for grid in fine_grids],
        "electronic_structure_points": int(np.prod(coarse_shape)),
        "interpolated_nuclear_points": int(np.prod([len(grid) for grid in fine_grids])),
        "new_electronic_structure_calculations": 0,
        "link_model": "interpolated regularized matrix-log connection",
        "link_floor": args.link_floor,
        "hamiltonian_dimension": int(hamiltonian.shape[0]),
        "hamiltonian_nnz": int(hamiltonian.nnz),
        "build_seconds": build_seconds,
        "propagation_seconds": propagation_seconds,
        "krylov_dimension": (
            args.krylov_check_dim
            if args.krylov_check_dim > args.krylov_dim
            else args.krylov_dim
        ),
        "krylov_crosscheck_dimension": (
            args.krylov_dim if args.krylov_check_dim > args.krylov_dim else None
        ),
        "max_krylov_state_difference": krylov_error,
        "max_population_error": float(np.max(abs(refined_population - exact_obs[1]))),
        "max_scaled_coordinate_error": float(
            np.max(abs(refined_means - exact_obs[2]) / spans[None, :])
        ),
        "max_norm_error": float(np.max(abs(refined_norm - 1.0))),
    }
    with (args.output_dir / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    np.savez(
        args.output_dir / "dynamics.npz",
        times_fs=times,
        reference_populations=exact_obs[1],
        refined_populations=refined_population,
        reference_means=exact_obs[2],
        refined_means=refined_means,
        refined_norms=refined_norm,
    )
    plot(
        args.output_dir / "so2_procrustes_cfd_refined",
        times,
        exact_obs[1],
        refined_population,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
