#!/usr/bin/env python3
"""LETTA-inspired autoregressive NN on a 4x4 J1-J2 Heisenberg model."""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

from examples.mps.letta_inspired_autoregressive_crossing import (
    initialize_letta,
    initialize_standard,
    letta_log_amplitude,
    parameter_count,
    standard_log_amplitude,
    state_vector,
    train,
)


jax.config.update("jax_enable_x64", True)


def snake_order(rows, cols):
    return tuple(
        row * cols + col
        for row in range(rows)
        for col in (range(cols) if row % 2 == 0 else reversed(range(cols)))
    )


def square_edges(rows, cols, j2):
    """Return coupling edges in snake-scan coordinates."""

    order = snake_order(rows, cols)
    physical_to_scan = np.empty(rows * cols, dtype=int)
    physical_to_scan[np.asarray(order)] = np.arange(rows * cols)
    physical_edges = []
    for row in range(rows):
        for col in range(cols):
            site = row * cols + col
            if col + 1 < cols:
                physical_edges.append((site, site + 1, 1.0))
            if row + 1 < rows:
                physical_edges.append((site, site + cols, 1.0))
            if row + 1 < rows and col + 1 < cols and j2:
                physical_edges.append((site, site + cols + 1, j2))
                physical_edges.append((site + 1, site + cols, j2))
    return tuple(
        (int(physical_to_scan[left]), int(physical_to_scan[right]), coupling)
        for left, right, coupling in physical_edges
    )


def configurations(n_sites):
    indices = np.arange(2**n_sites, dtype=np.uint32)
    shifts = np.arange(n_sites - 1, -1, -1, dtype=np.uint32)
    return jnp.asarray(((indices[:, None] >> shifts) & 1).astype(np.int8))


def hamiltonian_data(configs, edges):
    """Return diagonal and flip indices for a matrix-free H action."""

    configs = np.asarray(configs)
    n_sites = configs.shape[1]
    indices = np.arange(len(configs), dtype=np.uint32)
    spins = 1 - 2 * configs
    diagonal = np.zeros(len(configs), dtype=float)
    flipped = []
    active_flips = []
    couplings = []
    for left, right, coupling in edges:
        diagonal += 0.25 * coupling * spins[:, left] * spins[:, right]
        mask = (1 << (n_sites - 1 - left)) | (1 << (n_sites - 1 - right))
        flipped.append(indices ^ mask)
        active_flips.append(configs[:, left] != configs[:, right])
        couplings.append(coupling)
    return (
        jnp.asarray(diagonal),
        jnp.asarray(np.stack(flipped)),
        jnp.asarray(np.stack(active_flips)),
        jnp.asarray(couplings),
    )


def apply_hamiltonian(state, diagonal, flipped, active_flips, couplings):
    off_diagonal = jnp.sum(
        0.5 * couplings[:, None] * active_flips * state[flipped], axis=0
    )
    return diagonal * state + off_diagonal


def make_energy_function(
    configs, diagonal, flipped, active_flips, couplings, log_amplitude
):
    def energy(parameters):
        state = state_vector(parameters, log_amplitude, configs)
        applied = apply_hamiltonian(
            state, diagonal, flipped, active_flips, couplings
        )
        return jnp.vdot(state, applied).real / jnp.vdot(state, state).real

    return energy


def exact_sector_energy(n_sites, edges):
    """Diagonalize H in the half-filled magnetization sector."""

    full_dimension = 2**n_sites
    basis = np.asarray(
        [value for value in range(full_dimension) if value.bit_count() == n_sites // 2],
        dtype=np.int64,
    )
    lookup = np.full(full_dimension, -1, dtype=np.int64)
    lookup[basis] = np.arange(len(basis))
    shifts = np.arange(n_sites - 1, -1, -1)
    bits = ((basis[:, None] >> shifts) & 1).astype(np.int8)
    spins = 1 - 2 * bits
    diagonal = np.zeros(len(basis))
    rows = [np.arange(len(basis))]
    columns = [np.arange(len(basis))]
    values = []
    for left, right, coupling in edges:
        diagonal += 0.25 * coupling * spins[:, left] * spins[:, right]
        active = bits[:, left] != bits[:, right]
        mask = (1 << (n_sites - 1 - left)) | (1 << (n_sites - 1 - right))
        rows.append(np.flatnonzero(active))
        columns.append(lookup[basis[active] ^ mask])
        values.append(np.full(np.count_nonzero(active), 0.5 * coupling))
    matrix = coo_matrix(
        (
            np.concatenate([diagonal, *values]),
            (np.concatenate(rows), np.concatenate(columns)),
        ),
        shape=(len(basis), len(basis)),
    ).tocsr()
    energy = eigsh(matrix, k=1, which="SA", tol=1.0e-11, return_eigenvectors=False)[0]
    return float(energy), len(basis)


def diagnostics(parameters, energy_function, configs, data, log_amplitude):
    diagonal, flipped, active_flips, couplings = data
    state = state_vector(parameters, log_amplitude, configs)
    energy = energy_function(parameters)
    residual = (
        apply_hamiltonian(state, diagonal, flipped, active_flips, couplings)
        - energy * state
    )
    norm = jnp.vdot(state, state).real
    variance = jnp.vdot(residual, residual).real / norm
    probabilities = jnp.abs(state) ** 2
    magnetization = jnp.sum(configs, axis=1)
    sector_weight = jnp.sum(probabilities[magnetization == configs.shape[1] // 2])
    return float(energy), float(variance), float(norm), float(sector_weight)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--j2", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--standard-hidden-dim", type=int, default=24)
    parser.add_argument("--letta-hidden-dim", type=int, default=16)
    parser.add_argument("--virtual-dim", type=int, default=8)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--report-every", type=int, default=100)
    parser.add_argument(
        "--model",
        choices=("both", "standard", "letta"),
        default="both",
    )
    args = parser.parse_args(argv)

    n_sites = args.rows * args.cols
    if n_sites > 20:
        raise ValueError("exact-enumeration benchmark is limited to at most 20 sites")
    configs = configurations(n_sites)
    edges = square_edges(args.rows, args.cols, args.j2)
    data = hamiltonian_data(configs, edges)
    exact_energy, sector_dimension = exact_sector_energy(n_sites, edges)
    models = (
        (
            "standard-autoregressive",
            initialize_standard(
                jax.random.PRNGKey(args.seed),
                args.standard_hidden_dim,
                n_sites=n_sites,
            ),
            standard_log_amplitude,
        ),
        (
            "LETTA-inspired",
            initialize_letta(
                jax.random.PRNGKey(args.seed + 1),
                args.letta_hidden_dim,
                args.virtual_dim,
                args.rank,
                n_sites=n_sites,
            ),
            letta_log_amplitude,
        ),
    )
    if args.model != "both":
        selected = 0 if args.model == "standard" else 1
        models = (models[selected],)

    print(
        f"lattice: {args.rows}x{args.cols}, J2={args.j2}, "
        f"snake order={snake_order(args.rows, args.cols)}"
    )
    print(f"edges: {len(edges)}, full dimension: {len(configs)}")
    print(f"Sz=0 dimension: {sector_dimension}")
    print(f"exact energy: {exact_energy:.12f}\n")
    for name, parameters, log_amplitude in models:
        energy_function = make_energy_function(configs, *data, log_amplitude)
        initial_energy = float(energy_function(parameters))
        parameters, history = train(
            parameters,
            energy_function,
            steps=args.steps,
            learning_rate=args.learning_rate,
            report_every=args.report_every,
        )
        energy, variance, norm, sector_weight = diagnostics(
            parameters, energy_function, configs, data, log_amplitude
        )
        trace = ", ".join(f"{step}:{value:.6f}" for step, value in history)
        print(name)
        print(f"  parameters      : {parameter_count(parameters)}")
        print(f"  initial energy  : {initial_energy:.12f}")
        print(f"  final energy    : {energy:.12f}")
        print(f"  error           : {energy - exact_energy:.3e}")
        print(f"  energy per site : {energy / n_sites:.12f}")
        print(f"  variance        : {variance:.3e}")
        print(f"  norm check      : {norm:.12f}")
        print(f"  Sz=0 weight     : {sector_weight:.8f}")
        print(f"  trace           : {trace}\n")


if __name__ == "__main__":
    main()
