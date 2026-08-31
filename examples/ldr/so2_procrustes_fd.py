#!/usr/bin/env python3
"""Compare dense-DVR and local finite-difference SO2 LDR dynamics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from examples.ldr.so2_casci_cgldr import DEFAULT_SCAN_DIR, load_so2_linked_scan
from examples.ldr.so2_casci_cgldr_dense import dense_kinetic, nuclear_packet, observables
from examples.ldr.so2_casci_full_ldr import full_hamiltonian, path_overlap
from examples.ldr.so2_procrustes_dynamics import (
    DEFAULT_GAUGE,
    DEFAULT_REFERENCE,
    propagate,
    transform_states,
)
from examples.ldr.so2_procrustes_gauge import local_hamiltonian
from pyqed.ldr.overlap import unpack
from pyqed.units import au2fs


def local_derivative(points, weights, stencil=3):
    """Return a local polynomial derivative in a quadrature-normalized basis."""
    points = np.asarray(points, dtype=float)
    weights = np.asarray(weights, dtype=float)
    stencil = int(stencil)
    if stencil < 3 or stencil > len(points) or stencil % 2 == 0:
        raise ValueError("finite-difference stencil must be odd and between 3 and the grid size")
    if points.ndim != 1 or len(points) < stencil or np.any(np.diff(points) <= 0.0):
        raise ValueError("finite-difference points must be an increasing vector")
    if weights.shape != points.shape or np.any(weights <= 0.0):
        raise ValueError("quadrature weights must be positive and match the grid")
    derivative = np.zeros((len(points), len(points)), dtype=float)
    radius = stencil // 2
    for row in range(len(points)):
        start = min(max(row - radius, 0), len(points) - stencil)
        columns = np.arange(start, start + stencil)
        offsets = points[columns] - points[row]
        moments = np.vstack([offsets**order for order in range(stencil)])
        target = np.zeros(stencil)
        target[1] = 1.0
        derivative[row, columns] = np.linalg.solve(
            moments,
            target,
        )
    roots = np.sqrt(weights)
    return roots[:, None] * derivative / roots[None, :]


class LocalAxis:
    """Minimal DVR-compatible axis carrying a local momentum matrix."""

    def __init__(self, source, weights, stencil):
        self.x = np.asarray(source.x, dtype=float)
        self.w = np.asarray(weights, dtype=float)
        self.npts = len(self.x)
        self.dx = float(np.mean(self.w))
        self._momentum = -1j * local_derivative(self.x, self.w, stencil)

    def momentum(self, sparse=False):
        if not sparse:
            return self._momentum
        from scipy.sparse import csr_matrix

        return csr_matrix(self._momentum)


def local_axes(reference_axes, stencil):
    return (
        LocalAxis(reference_axes[0], np.full(reference_axes[0].npts, reference_axes[0].dx), stencil),
        LocalAxis(reference_axes[1], reference_axes[1].w, stencil),
        LocalAxis(reference_axes[2], np.full(reference_axes[2].npts, reference_axes[2].dx), stencil),
    )


def product_terms_dense(terms):
    matrix = None
    for _label, coefficient, *operators in terms:
        block = np.asarray(operators[0])
        for operator in operators[1:]:
            block = np.kron(block, np.asarray(operator))
        block = coefficient * block
        matrix = block if matrix is None else matrix + block
    return 0.5 * (matrix + matrix.conj().T)


def aligned_links(shape, links, gauge):
    gauge = np.asarray(gauge, dtype=complex).reshape(*shape, gauge.shape[-2], gauge.shape[-1])
    output = {}
    for (axis, index), block in links.items():
        neighbor = list(index)
        neighbor[int(axis)] += 1
        output[(int(axis), tuple(index))] = (
            gauge[index].conj().T @ np.asarray(block) @ gauge[tuple(neighbor)]
        )
    return output


def aligned_hamiltonian(kinetic, overlap, local):
    ngrid, nstates = overlap.shape[:2]
    matrix = np.asarray(kinetic)[:, None, :, None] * overlap
    for point in range(ngrid):
        matrix[point, :, point, :] += local[point]
    matrix = matrix.reshape(ngrid * nstates, ngrid * nstates)
    return 0.5 * (matrix + matrix.conj().T)


def fidelity(reference, trial):
    numerator = np.abs(np.einsum("ti,ti->t", reference.conj(), trial)) ** 2
    denominator = np.sum(abs(reference) ** 2, axis=1) * np.sum(abs(trial) ** 2, axis=1)
    return np.clip(numerator / denominator, 0.0, 1.0)


def plot(path, times, reference, covariant, identity):
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    colors = ("#0072B2", "#D55E00")
    for state, color in zip((1, 2), colors):
        axes[0].plot(times, reference[:, state], color=color, label=rf"DVR $P_{state}$")
        axes[0].plot(times, covariant[:, state], "--", color=color, label=rf"Covariant FD $P_{state}$")
        axes[0].plot(times, identity[:, state], ":", color=color, label=rf"FD, $S=I$, $P_{state}$")
    axes[1].semilogy(
        times,
        np.maximum(np.max(abs(covariant - reference), axis=1), 1.0e-16),
        color="#009E73",
        label="Covariant FD",
    )
    axes[1].semilogy(
        times,
        np.maximum(np.max(abs(identity - reference), axis=1), 1.0e-16),
        color="#CC79A7",
        linestyle=":",
        label=r"FD, $S=I$",
    )
    axes[0].set(xlabel="Time (fs)", ylabel="Population", ylim=(-0.03, 1.03))
    axes[1].set(xlabel="Time (fs)", ylabel="Maximum population error")
    for axis in axes:
        axis.legend(frameon=False, fontsize=7)
        axis.grid(False)
    figure.savefig(path.with_suffix(".png"), dpi=350)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--gauge", type=Path, default=DEFAULT_GAUGE)
    parser.add_argument("--scan-dir", type=Path, default=DEFAULT_SCAN_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("/private/tmp/so2_procrustes_fd"))
    parser.add_argument("--initial-state", type=int, default=2)
    parser.add_argument("--time-fs", type=float, default=20.0)
    parser.add_argument("--dt-fs", type=float, default=0.5)
    parser.add_argument("--stencil", type=int, choices=(3, 5, 7, 9), default=3)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.reference) as archive:
        energies = np.asarray(archive["energies"], dtype=float)
        grids = tuple(np.asarray(archive[name], dtype=float) for name in ("qs", "theta", "qa"))
        links = unpack(archive["link_axes"], archive["link_indices"], archive["link_data"])
    with np.load(args.gauge) as archive:
        gauge = np.asarray(archive["gauge"], dtype=complex)
        primary_gauge = np.asarray(archive["primary_gauge"], dtype=complex)

    shape = energies.shape[:-1]
    nstates = energies.shape[-1]
    ngrid = int(np.prod(shape))
    scan = load_so2_linked_scan(args.scan_dir)
    dense_k, reference_axes = dense_kinetic(scan, *grids)
    fd_axes = local_axes(reference_axes, args.stencil)
    fd_terms = scan.solver.buildK_qsqa_terms(
        fd_axes,
        symmetrize=True,
        svd_tol=0.0,
    )
    fd_k = product_terms_dense(fd_terms)

    original_overlap = path_overlap(shape, links).reshape(ngrid, nstates, ngrid, nstates)
    exact_h = full_hamiltonian(dense_k, original_overlap, energies)
    link_overlap = path_overlap(shape, aligned_links(shape, links, gauge)).reshape(
        ngrid, nstates, ngrid, nstates
    )
    local = local_hamiltonian(energies, gauge.reshape(ngrid, nstates, nstates))
    covariant_h = aligned_hamiltonian(fd_k, link_overlap, local)
    identity_overlap = np.broadcast_to(
        np.eye(nstates)[None, :, None, :],
        (ngrid, nstates, ngrid, nstates),
    )
    identity_h = aligned_hamiltonian(fd_k, identity_overlap, local)

    packet = nuclear_packet(*grids, reference_axes)
    original_initial = (packet[..., None] * primary_gauge[..., args.initial_state]).reshape(-1)
    original_initial /= np.linalg.norm(original_initial)
    aligned_initial = np.einsum(
        "...ia,...i->...a",
        gauge.conj(),
        original_initial.reshape(*shape, nstates),
    ).reshape(-1)
    delta_hamiltonian = covariant_h - identity_h
    covariant_action = covariant_h @ aligned_initial
    delta_action = delta_hamiltonian @ aligned_initial
    hamiltonian_diagnostics = {
        "relative_frobenius_difference": float(
            np.linalg.norm(delta_hamiltonian) / np.linalg.norm(covariant_h)
        ),
        "relative_initial_action_difference": float(
            np.linalg.norm(delta_action) / np.linalg.norm(covariant_action)
        ),
        "initial_energy_difference_hartree": float(
            np.vdot(aligned_initial, delta_action).real
        ),
    }
    times = np.arange(0.0, args.time_fs + 0.5 * args.dt_fs, args.dt_fs)

    started = time.perf_counter()
    exact_states = propagate(exact_h, original_initial, times)
    covariant_states = transform_states(propagate(covariant_h, aligned_initial, times), gauge)
    identity_states = transform_states(propagate(identity_h, aligned_initial, times), gauge)
    propagation_seconds = time.perf_counter() - started

    exact_obs = observables(exact_states, grids, primary_gauge)
    covariant_obs = observables(covariant_states, grids, primary_gauge)
    identity_obs = observables(identity_states, grids, primary_gauge)
    covariant_fidelity = fidelity(exact_states, covariant_states)
    identity_fidelity = fidelity(exact_states, identity_states)
    spans = np.asarray([np.ptp(grid) for grid in grids])
    summary = {
        "grid": list(shape),
        "time_fs": args.time_fs,
        "dt_fs": args.dt_fs,
        "fd_stencil": f"{args.stencil}-point local weak-form derivative",
        "propagation_seconds": propagation_seconds,
        "covariant_vs_identity_hamiltonian": hamiltonian_diagnostics,
        "covariant_fd": {
            "max_population_error": float(np.max(abs(covariant_obs[1] - exact_obs[1]))),
            "max_scaled_coordinate_error": float(
                np.max(abs(covariant_obs[2] - exact_obs[2]) / spans[None, :])
            ),
            "minimum_fidelity": float(np.min(covariant_fidelity)),
            "final_fidelity": float(covariant_fidelity[-1]),
        },
        "identity_link_fd": {
            "max_population_error": float(np.max(abs(identity_obs[1] - exact_obs[1]))),
            "max_scaled_coordinate_error": float(
                np.max(abs(identity_obs[2] - exact_obs[2]) / spans[None, :])
            ),
            "minimum_fidelity": float(np.min(identity_fidelity)),
            "final_fidelity": float(identity_fidelity[-1]),
        },
    }
    with (args.output_dir / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    np.savez(
        args.output_dir / "dynamics.npz",
        times_fs=times,
        reference_populations=exact_obs[1],
        covariant_fd_populations=covariant_obs[1],
        identity_link_fd_populations=identity_obs[1],
        covariant_fidelity=covariant_fidelity,
        identity_link_fidelity=identity_fidelity,
    )
    plot(
        args.output_dir / "so2_procrustes_fd",
        times,
        exact_obs[1],
        covariant_obs[1],
        identity_obs[1],
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
