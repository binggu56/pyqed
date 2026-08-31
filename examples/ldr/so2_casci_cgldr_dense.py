#!/usr/bin/env python3
"""Propagate a cached 9^3 SO2 CGLDR model by dense diagonalization."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

from examples.ldr.so2_casci_cgldr import (
    DEFAULT_SCAN_DIR,
    REFERENCE_BOND,
    REFERENCE_BOND_WIDTH,
    REFERENCE_THETA_DEG,
    REFERENCE_THETA_WIDTH_DEG,
    SQRT2,
    casci_overlap_active,
    infer_legendre_domain,
    infer_sine_domain,
    load_so2_linked_scan,
)
from pyqed.dvr import DVR, LegendreDVR, SineDVR
from pyqed.ldr import CGLDRElectronicData, SeparableHamiltonian
from pyqed.mps.mps import _mpo_to_dense_operator
from pyqed.units import au2fs


def nuclear_grid(data):
    names = tuple(data.metadata["sampled_coordinates"])
    names += tuple(data.metadata["expanded_coordinates"])
    grids = tuple(np.asarray(grid, dtype=float) for grid in data.reactive_grids)
    grids += tuple(np.asarray(grid, dtype=float) for grid in data.expanded_grids)
    if names != ("qs", "theta", "qa"):
        raise ValueError(
            "Dense SO2 propagation requires coordinate order (qs, theta, qa); "
            f"got {names}."
        )
    return grids


def single_anchor_quadratic(data):
    """Derive the center-anchor quadratic model from cached analytical F/G."""
    if data.expanded_grids is None or len(data.expanded_grids) != 1:
        raise ValueError("Single-anchor conversion requires one expanded coordinate")
    if data.hamiltonian_gradients is None or data.hamiltonian_hessians is None:
        raise ValueError("Single-anchor conversion requires cached analytical F/G")
    energies = np.asarray(data.energies)
    gradients = np.asarray(data.hamiltonian_gradients)
    hessians = np.asarray(data.hamiltonian_hessians)
    nstates = energies.shape[-1]
    expected_gradient = (*energies.shape[:-1], 1, nstates, nstates)
    expected_hessian = (*energies.shape[:-1], 1, 1, nstates, nstates)
    if gradients.shape != expected_gradient or hessians.shape != expected_hessian:
        raise ValueError("Cached F/G tensors do not describe one expanded coordinate")

    hamiltonians = np.zeros(
        (*energies.shape[:-1], nstates, nstates),
        dtype=np.result_type(energies, gradients, hessians),
    )
    states = np.arange(nstates)
    hamiltonians[..., states, states] = energies
    coefficients = np.stack(
        (hamiltonians, gradients[..., 0, :, :], 0.5 * hessians[..., 0, 0, :, :]),
        axis=-3,
    )
    coefficients = 0.5 * (
        coefficients + coefficients.swapaxes(-1, -2).conj()
    )
    qa = np.asarray(data.expanded_grids[0], dtype=float)
    center_index = int(np.argmin(np.abs(qa)))
    center = float(qa[center_index])
    separable = SeparableHamiltonian.polynomial(
        qa,
        coefficients,
        center=center,
    )
    metadata = dict(data.metadata)
    metadata.update(
        {
            "derived_from_qa_model": metadata.get("qa_model"),
            "qa_model": "single-reference-quadratic",
            "qa_anchor_count": 1,
            "qa_anchor_indices": [center_index],
            "qa_anchor_values": [center],
            "qa_extrapolation": "quadratic",
            "electronic_structure_recomputed": False,
        }
    )
    return CGLDRElectronicData(
        energies=energies,
        overlaps=data.overlaps,
        hamiltonian_gradients=gradients,
        hamiltonian_hessians=hessians,
        separable_hamiltonian=separable,
        reactive_grids=data.reactive_grids,
        expanded_grids=data.expanded_grids,
        basis_transforms=data.basis_transforms,
        metric_eigenvalues=data.metric_eigenvalues,
        metadata=metadata,
    )


def dense_kinetic(scan, qs, theta, qa):
    axes = (
        SineDVR(*infer_sine_domain(qs), len(qs)),
        LegendreDVR(*infer_legendre_domain(theta), len(theta)),
        SineDVR(*infer_sine_domain(qa), len(qa)),
    )
    dvr = DVR.from_axes(axes, names=("qs", "theta", "qa"))
    mpo = scan.solver.buildK_qsqa_mpo(
        dvr.axes,
        max_rank=None,
        symmetrize=True,
        svd_tol=0.0,
    )
    kinetic = np.asarray(_mpo_to_dense_operator(mpo), dtype=complex)
    return 0.5 * (kinetic + kinetic.conj().T), axes


def polar_positive_factor(overlap):
    """Return the positive left polar factor of an overlap matrix."""
    left, singular_values, _right = np.linalg.svd(
        np.asarray(overlap, dtype=complex),
        full_matrices=False,
    )
    factor = (left * singular_values[None, :]) @ left.conj().T
    return 0.5 * (factor + factor.conj().T), singular_values


def overlap_quantum_metric(
    overlap,
    displacement,
    *,
    eigenvalue_floor=1.0e-12,
):
    r"""Fit ``P = exp(-g * displacement**2 / 2)`` from one overlap."""
    displacement = float(displacement)
    if not np.isfinite(displacement) or displacement == 0.0:
        raise ValueError("displacement must be finite and nonzero")
    if not np.isfinite(eigenvalue_floor) or not 0.0 < eigenvalue_floor < 1.0:
        raise ValueError("eigenvalue_floor must lie strictly between zero and one")
    factor, singular_values = polar_positive_factor(overlap)
    values, vectors = np.linalg.eigh(factor)
    scale = max(float(np.max(values)), np.finfo(float).tiny)
    floor = max(float(eigenvalue_floor) * scale, np.finfo(float).tiny)
    clipped = np.clip(values, floor, 1.0)
    metric_values = -2.0 * np.log(clipped) / displacement**2
    metric = (vectors * metric_values[None, :]) @ vectors.conj().T
    metric = 0.5 * (metric + metric.conj().T)
    ratio = float(np.min(singular_values) / max(np.max(singular_values), np.finfo(float).tiny))
    return metric, np.sort(singular_values), ratio


def harmonic_matrix_extension(values, valid, *, coordinates=None):
    """Harmonically continue a matrix field from reliable grid points."""
    values = np.asarray(values, dtype=complex)
    valid = np.asarray(valid, dtype=bool)
    grid_shape = valid.shape
    if values.shape[: valid.ndim] != grid_shape:
        raise ValueError("values and valid must have matching grid dimensions")
    if not np.any(valid):
        raise ValueError("harmonic continuation requires at least one valid point")
    if np.all(valid):
        return np.array(values, copy=True)
    if coordinates is None:
        coordinates = tuple(np.arange(size, dtype=float) for size in grid_shape)
    if len(coordinates) != valid.ndim:
        raise ValueError("coordinates must contain one grid for each field axis")
    scaled = []
    for axis, (grid, size) in enumerate(zip(coordinates, grid_shape)):
        grid = np.asarray(grid, dtype=float)
        if grid.shape != (size,) or np.any(np.diff(grid) <= 0.0):
            raise ValueError(f"coordinate axis {axis} must be finite and increasing")
        span = float(grid[-1] - grid[0])
        scaled.append((grid - grid[0]) / span if span > 0.0 else grid)

    missing = [index for index in np.ndindex(grid_shape) if not valid[index]]
    missing_position = {index: row for row, index in enumerate(missing)}
    matrix = np.zeros((len(missing), len(missing)), dtype=float)
    trailing_shape = values.shape[valid.ndim :]
    right_hand_side = np.zeros(
        (len(missing), int(np.prod(trailing_shape))),
        dtype=complex,
    )
    for row, index in enumerate(missing):
        for axis, size in enumerate(grid_shape):
            for step in (-1, 1):
                neighbor = list(index)
                neighbor[axis] += step
                if not 0 <= neighbor[axis] < size:
                    continue
                neighbor = tuple(neighbor)
                spacing = abs(scaled[axis][neighbor[axis]] - scaled[axis][index[axis]])
                weight = 1.0 / max(spacing**2, np.finfo(float).eps)
                matrix[row, row] += weight
                if valid[neighbor]:
                    right_hand_side[row] += weight * values[neighbor].reshape(-1)
                else:
                    matrix[row, missing_position[neighbor]] -= weight
    continued = np.array(values, copy=True)
    solution = np.linalg.solve(matrix, right_hand_side)
    for row, index in enumerate(missing):
        continued[index] = solution[row].reshape(trailing_shape)
    return continued


def _project_continued_metric(metrics, valid):
    """Hermitize and PSD-project only the harmonically filled matrices."""
    metrics = np.asarray(metrics, dtype=complex)
    valid = np.asarray(valid, dtype=bool)
    reliable_max = max(
        float(np.max(np.linalg.eigvalsh(0.5 * (matrix + matrix.conj().T))))
        for matrix in metrics[valid]
    )
    projected = np.array(metrics, copy=True)
    for index in np.ndindex(valid.shape):
        matrix = 0.5 * (projected[index] + projected[index].conj().T)
        if not valid[index]:
            values, vectors = np.linalg.eigh(matrix)
            values = np.clip(values, 0.0, reliable_max)
            matrix = (vectors * values[None, :]) @ vectors.conj().T
        projected[index] = 0.5 * (matrix + matrix.conj().T)
    return projected


def qa_p_metric_kernel(
    data,
    anchor_cache_dir,
    *,
    regularization="harmonic",
    reliability_ratio=1.0e-6,
    eigenvalue_floor=1.0e-12,
):
    """Build a three-anchor fitted positive-overlap kernel along ``q_a``."""
    if regularization not in {"floor", "harmonic"}:
        raise ValueError("regularization must be 'floor' or 'harmonic'")
    if not np.isfinite(reliability_ratio) or not 0.0 < reliability_ratio < 1.0:
        raise ValueError("reliability_ratio must lie strictly between zero and one")
    metadata = data.metadata
    anchor_indices = np.asarray(metadata.get("qa_anchor_indices"), dtype=int)
    anchor_values = np.asarray(metadata.get("qa_anchor_values"), dtype=float)
    if anchor_indices.shape != (3,) or anchor_values.shape != (3,):
        raise ValueError("q_a positive-metric fitting requires exactly three anchors")
    if not anchor_values[0] < anchor_values[1] < anchor_values[2]:
        raise ValueError("q_a anchors must be strictly increasing")
    qa = np.asarray(data.expanded_grids[0], dtype=float)
    sampled_shape = tuple(len(grid) for grid in data.reactive_grids)
    nstates = data.energies.shape[-1]
    state_ids = tuple(int(value) for value in metadata["state_ids"])
    cache_dir = Path(anchor_cache_dir)

    metrics = np.empty((2, *sampled_shape, nstates, nstates), dtype=complex)
    anchor_singular_values = np.empty((2, *sampled_shape, nstates), dtype=float)
    reliability = np.empty((2, *sampled_shape), dtype=float)
    side_anchors = (0, 2)
    overlap_evaluations = 0
    for index in np.ndindex(sampled_shape):
        points = []
        for local_anchor, qa_index in enumerate(anchor_indices):
            path = cache_dir / (
                f"anchor_{index[0]}_{index[1]}_qa{int(qa_index)}.pkl"
            )
            if not path.is_file():
                local_path = cache_dir / (
                    f"anchor_{index[0]}_{index[1]}_{local_anchor}.pkl"
                )
                if not local_path.is_file():
                    raise FileNotFoundError(f"Missing cached q_a anchor: {path}")
                path = local_path
            with path.open("rb") as stream:
                result = pickle.load(stream)
            if tuple(result[:2]) != index:
                raise ValueError(f"Cached q_a anchor indices do not match {path}")
            points.append(result[3])
        center = points[1]
        for side, local_anchor in enumerate(side_anchors):
            overlap = casci_overlap_active(
                center,
                points[local_anchor],
                state_ids,
                polar=False,
            )
            metric, singular_values, ratio = overlap_quantum_metric(
                overlap,
                anchor_values[local_anchor] - anchor_values[1],
                eigenvalue_floor=eigenvalue_floor,
            )
            metrics[(side, *index)] = metric
            anchor_singular_values[(side, *index)] = singular_values
            reliability[(side, *index)] = ratio
            overlap_evaluations += 1

    valid = reliability >= float(reliability_ratio)
    fitted_metrics = np.array(metrics, copy=True)
    if regularization == "harmonic":
        for side in range(2):
            fitted_metrics[side] = harmonic_matrix_extension(
                fitted_metrics[side],
                valid[side],
                coordinates=data.reactive_grids,
            )
            fitted_metrics[side] = _project_continued_metric(
                fitted_metrics[side],
                valid[side],
            )

    center = float(anchor_values[1])
    links = np.empty((*sampled_shape, len(qa) - 1, nstates, nstates), dtype=complex)
    for index in np.ndindex(sampled_shape):
        for link, midpoint in enumerate(0.5 * (qa[:-1] + qa[1:])):
            side = 0 if midpoint < center else 1
            values, vectors = np.linalg.eigh(fitted_metrics[(side, *index)])
            values = np.maximum(values, 0.0)
            spacing = float(qa[link + 1] - qa[link])
            link_values = np.exp(-0.5 * values * spacing**2)
            links[index + (link,)] = (
                vectors * link_values[None, :]
            ) @ vectors.conj().T

    kernel = np.empty((*sampled_shape, len(qa), len(qa), nstates, nstates), dtype=complex)
    identity = np.eye(nstates, dtype=complex)
    for index in np.ndindex(sampled_shape):
        for bra in range(len(qa)):
            kernel[index + (bra, bra)] = identity
            product = identity
            for ket in range(bra + 1, len(qa)):
                product = product @ links[index + (ket - 1,)]
                kernel[index + (bra, ket)] = product
                kernel[index + (ket, bra)] = product.conj().T
    diagnostics = {
        "qa_p_anchor_singular_values": anchor_singular_values,
        "qa_p_reliability_ratios": reliability,
        "qa_p_valid_mask": valid,
        "qa_p_raw_metrics": metrics,
        "qa_p_fitted_metrics": fitted_metrics,
        "qa_p_nearest_links": links,
        "qa_p_overlap_evaluations": overlap_evaluations,
    }
    return kernel, diagnostics


def dense_hamiltonian(data, kinetic, *, qa_overlap_kernel=None):
    shape = tuple(len(grid) for grid in nuclear_grid(data))
    sampled_shape = tuple(len(grid) for grid in data.reactive_grids)
    nstates = data.energies.shape[-1]
    nuclear_indices = np.indices(shape).reshape(len(shape), -1).T
    sampled_indices = np.ravel_multi_index(
        nuclear_indices[:, : len(sampled_shape)].T,
        sampled_shape,
    )
    overlaps = np.asarray(data.overlaps, dtype=complex).reshape(
        np.prod(sampled_shape), nstates, np.prod(sampled_shape), nstates
    )
    full_overlaps = overlaps[
        sampled_indices[:, None], :, sampled_indices[None, :], :
    ].transpose(0, 2, 1, 3)
    if qa_overlap_kernel is not None:
        qa_overlap_kernel = np.asarray(qa_overlap_kernel, dtype=complex).reshape(
            np.prod(sampled_shape), shape[-1], shape[-1], nstates, nstates
        )
        qa_indices = nuclear_indices[:, -1]
        qa_blocks = qa_overlap_kernel[
            sampled_indices[:, None],
            qa_indices[:, None],
            qa_indices[None, :],
        ].transpose(0, 2, 1, 3)
        same_sampled_point = sampled_indices[:, None] == sampled_indices[None, :]
        full_overlaps = np.where(
            same_sampled_point[:, None, :, None],
            qa_blocks,
            full_overlaps,
        )
    matrix = (
        kinetic[:, None, :, None] * full_overlaps
    ).reshape(kinetic.shape[0] * nstates, -1)

    if data.separable_hamiltonian is None:
        raise ValueError("Dense propagation requires a separable Hamiltonian")
    local = np.asarray(data.separable_hamiltonian.evaluate(), dtype=complex)
    local = 0.5 * (local + local.swapaxes(-1, -2).conj())
    for point, block in enumerate(local.reshape(-1, nstates, nstates)):
        begin = point * nstates
        matrix[begin : begin + nstates, begin : begin + nstates] += block
    return 0.5 * (matrix + matrix.conj().T)


def nuclear_packet(qs, theta, qa, axes):
    grids = np.meshgrid(qs, theta, qa, indexing="ij")
    center = (SQRT2 * REFERENCE_BOND, np.deg2rad(REFERENCE_THETA_DEG), 0.0)
    width = (
        REFERENCE_BOND_WIDTH,
        np.deg2rad(REFERENCE_THETA_WIDTH_DEG),
        REFERENCE_BOND_WIDTH,
    )
    amplitude = np.exp(
        -0.5 * sum(((grid - c) / w) ** 2 for grid, c, w in zip(grids, center, width))
    )
    weights = (
        np.full(len(qs), axes[0].dx)[:, None, None]
        * axes[1].w[None, :, None]
        * np.full(len(qa), axes[2].dx)[None, None, :]
    )
    packet = np.sqrt(weights) * amplitude
    return packet / np.linalg.norm(packet)


def electronic_transport(data, shape, initial_state):
    sampled_shape = tuple(len(grid) for grid in data.reactive_grids)
    sampled_center = tuple(size // 2 for size in sampled_shape)
    nstates = data.energies.shape[-1]
    selection = (
        (slice(None),) * len(sampled_shape)
        + (slice(None),)
        + sampled_center
        + (slice(None),)
    )
    blocks = np.asarray(data.overlaps, dtype=complex)[selection]
    left, _singular, right = np.linalg.svd(blocks, full_matrices=False)
    sampled_transport = left @ right
    expanded_shape = shape[len(sampled_shape) :]
    transport = sampled_transport.reshape(
        *sampled_shape,
        *(1 for _ in expanded_shape),
        nstates,
        nstates,
    )
    transport = np.broadcast_to(transport, (*shape, nstates, nstates))
    projector = transport[..., initial_state]
    return transport, projector


def observables(states, grids, transport):
    shape = tuple(len(grid) for grid in grids)
    nstates = transport.shape[-1]
    wavefunctions = states.reshape(len(states), *shape, nstates)
    probability = np.abs(wavefunctions) ** 2
    sum_axes = tuple(range(1, 1 + len(shape)))
    norms = probability.sum(axis=(*sum_axes, len(shape) + 1))
    populations = probability.sum(axis=sum_axes) / norms[:, None]
    reference = np.einsum(
        "...ik,t...i->t...k", transport.conj(), wavefunctions, optimize=True
    )
    reference_probability = np.abs(reference) ** 2
    reference_populations = reference_probability.sum(axis=sum_axes)
    reference_populations /= reference_populations.sum(axis=1, keepdims=True)
    nuclear = probability.sum(axis=-1)
    nuclear_flat = nuclear.reshape(len(states), -1)
    meshes = np.meshgrid(*grids, indexing="ij")
    means = np.empty((len(states), len(shape)))
    variances = np.empty_like(means)
    for axis, mesh in enumerate(meshes):
        values = mesh.reshape(-1)
        means[:, axis] = (nuclear_flat @ values) / norms
        second = (nuclear_flat @ values**2) / norms
        variances[:, axis] = np.maximum(second - means[:, axis] ** 2, 0.0)
    return populations, reference_populations, means, variances, norms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("electronic_data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scan-dir", type=Path, default=DEFAULT_SCAN_DIR)
    parser.add_argument("--initial-state", type=int, default=2)
    parser.add_argument("--time-fs", type=float, default=20.0)
    parser.add_argument("--dt-fs", type=float, default=0.5)
    parser.add_argument(
        "--single-anchor-quadratic",
        action="store_true",
        help="Use only cached center-anchor analytical F/G; performs no qchem.",
    )
    parser.add_argument(
        "--save-electronic-data",
        type=Path,
        default=None,
        help="Optionally save the derived electronic model.",
    )
    parser.add_argument(
        "--qa-p-metric-cache",
        type=Path,
        default=None,
        help="Fit the omitted q_a positive overlap from a three-anchor cache.",
    )
    parser.add_argument(
        "--qa-p-regularization",
        choices=("floor", "harmonic"),
        default="harmonic",
        help="Regularization used for ill-conditioned anchor logarithms.",
    )
    parser.add_argument("--qa-p-reliability-ratio", type=float, default=1.0e-6)
    parser.add_argument("--qa-p-eigenvalue-floor", type=float, default=1.0e-12)
    args = parser.parse_args()

    started = time.perf_counter()
    data = CGLDRElectronicData.from_npz(args.electronic_data)
    if args.single_anchor_quadratic:
        data = single_anchor_quadratic(data)
    if args.save_electronic_data is not None:
        args.save_electronic_data.parent.mkdir(parents=True, exist_ok=True)
        data.to_npz(args.save_electronic_data)
    grids = nuclear_grid(data)
    scan = load_so2_linked_scan(args.scan_dir)
    kinetic, axes = dense_kinetic(scan, *grids)
    qa_overlap_kernel = None
    qa_p_diagnostics = {}
    if args.qa_p_metric_cache is not None:
        qa_overlap_kernel, qa_p_diagnostics = qa_p_metric_kernel(
            data,
            args.qa_p_metric_cache,
            regularization=args.qa_p_regularization,
            reliability_ratio=args.qa_p_reliability_ratio,
            eigenvalue_floor=args.qa_p_eigenvalue_floor,
        )
        valid = qa_p_diagnostics["qa_p_valid_mask"]
        print(
            f"[dense] q_a P metric: {np.count_nonzero(valid)}/{valid.size} "
            f"reliable anchor fits ({args.qa_p_regularization})",
            flush=True,
        )
    hamiltonian = dense_hamiltonian(
        data,
        kinetic,
        qa_overlap_kernel=qa_overlap_kernel,
    )
    print(
        f"[dense] H={hamiltonian.shape}, Hermitian error="
        f"{np.max(np.abs(hamiltonian - hamiltonian.conj().T)):.3e}",
        flush=True,
    )

    packet = nuclear_packet(*grids, axes)
    transport, projector = electronic_transport(
        data, packet.shape, args.initial_state
    )
    psi0 = (packet[..., None] * projector).reshape(-1)
    psi0 /= np.linalg.norm(psi0)
    energies, vectors = eigh(hamiltonian, overwrite_a=True, check_finite=False)
    times_fs = np.arange(0.0, args.time_fs + 0.5 * args.dt_fs, args.dt_fs)
    coefficients = vectors.conj().T @ psi0
    phases = np.exp(-1j * np.outer(times_fs / au2fs, energies))
    states = (phases * coefficients[None, :]) @ vectors.conj().T
    results = observables(states, grids, transport)
    elapsed = time.perf_counter() - started

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_metadata = dict(data.metadata)
    if args.qa_p_metric_cache is not None:
        output_metadata.update(
            {
                "qa_p_metric": "three-anchor-quantum-metric",
                "qa_p_regularization": args.qa_p_regularization,
                "qa_p_reliability_ratio": args.qa_p_reliability_ratio,
                "qa_p_eigenvalue_floor": args.qa_p_eigenvalue_floor,
                "qa_p_electronic_structure_recomputed": False,
            }
        )
    output = dict(
        times_fs=times_fs,
        coordinate_names=np.asarray(("qs", "theta", "qa")),
        populations=results[0],
        reference_populations=results[1],
        means=results[2],
        variances=results[3],
        norms=results[4],
        qs=grids[0],
        theta=grids[1],
        qa=grids[2],
        elapsed_seconds=elapsed,
        metadata_json=np.array(json.dumps(output_metadata, sort_keys=True)),
    )
    output.update(qa_p_diagnostics)
    np.savez(args.output, **output)
    print(
        f"[dense] final reference populations={results[1][-1].tolist()}",
        flush=True,
    )
    print(f"[dense] wrote {args.output} in {elapsed:.2f} s", flush=True)


if __name__ == "__main__":
    main()
