"""Scalable variational TTN calculation for the critical Ising chain."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyqed.narg import transverse_field_ising_hamiltonian
from pyqed.tn import TTN


def balanced_tree_parents(nsites):
    """Return a balanced binary tree over sites in their chain order."""
    if nsites < 1:
        raise ValueError("nsites must be positive.")
    parents = [None] * nsites

    def add_subtree(left, right, parent):
        if left >= right:
            return
        site = (left + right) // 2
        parents[site] = parent
        add_subtree(left, site, site)
        add_subtree(site + 1, right, site)

    add_subtree(0, nsites, None)
    return tuple(parents)


def ising_terms(nsites, *, coupling=1.0, field=1.0, periodic=False):
    """Return product-operator terms for ``-J ZZ - h X``."""
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.diag([1.0, -1.0])
    terms = [(-field, {site: x}) for site in range(nsites)]
    bonds = [(site, site + 1) for site in range(nsites - 1)]
    if periodic and nsites > 2:
        bonds.append((nsites - 1, 0))
    terms.extend((-coupling, {left: z, right: z}) for left, right in bonds)
    return terms, z


def effective_hamiltonian(state, terms, center):
    """Build the center effective Hamiltonian using only tree contractions."""
    effective = state.effective_operator_sum(terms, center=center)
    return 0.5 * (effective + effective.T.conj())


def optimize_site(state, site, terms):
    """Optimize one tensor in its canonical orthonormal branch frame."""
    state.canonicalize(site)
    shape = state.tensors[site].shape
    effective = effective_hamiltonian(state, terms, site)
    energies, vectors = eigh(
        effective,
        subset_by_index=(0, 0),
        check_finite=False,
    )
    state.tensors[site] = vectors[:, 0].reshape(shape)
    state.center = site
    return float(energies[0])


def hamiltonian_expectation(state, terms):
    """Contract the total energy without materializing the state vector."""
    return float(
        np.real(
            sum(
                coefficient * state.expectation_value(operators)
                for coefficient, operators in terms
            )
        )
    )


def connected_zz(state, z, left, right):
    """Contract a connected spin correlation using TTN messages."""
    left_mean = state.expectation_value({left: z})
    right_mean = state.expectation_value({right: z})
    zz = state.expectation_value({left: z, right: z})
    return float(np.real(zz - left_mean * right_mean))


def exact_reference(nsites, *, coupling, field, periodic):
    """Return a sparse exact ground state for small-system validation."""
    hamiltonian = transverse_field_ising_hamiltonian(
        nsites,
        j=coupling,
        field=field,
        periodic=periodic,
        sparse=True,
    )
    energies, vectors = eigsh(hamiltonian, k=1, which="SA")
    return float(energies[0]), vectors[:, 0]


def exact_open_energy(nsites, *, coupling, field):
    """Return the open-chain free-fermion ground-state energy."""
    coupling_matrix = field * np.eye(nsites)
    coupling_matrix[np.arange(1, nsites), np.arange(nsites - 1)] = coupling
    return -float(np.linalg.svd(coupling_matrix, compute_uv=False).sum())


def dense_connected_zz(vector, left, right, nsites):
    """Evaluate a small-system reference correlation from a dense vector."""
    probabilities = np.abs(vector.reshape((2,) * nsites)) ** 2
    spins = np.array([1.0, -1.0])
    left_mean = np.tensordot(probabilities, spins, axes=([left], [0])).sum()
    right_mean = np.tensordot(probabilities, spins, axes=([right], [0])).sum()
    zz = np.tensordot(
        probabilities,
        np.outer(spins, spins),
        axes=([left, right], [0, 1]),
    ).sum()
    return float(zz - left_mean * right_mean)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nsites", type=int, default=16)
    parser.add_argument("--bond-dim", type=int, default=4)
    parser.add_argument("--sweeps", type=int, default=12)
    parser.add_argument("--tolerance", type=float, default=1.0e-10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--periodic", action="store_true")
    parser.add_argument(
        "--exact-max-sites",
        type=int,
        default=12,
        help="run sparse exact validation at or below this size",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    coupling = field = 1.0
    parents = balanced_tree_parents(args.nsites)
    terms, z = ising_terms(
        args.nsites,
        coupling=coupling,
        field=field,
        periodic=args.periodic,
    )
    state = TTN(
        (2,) * args.nsites,
        parents,
        bond_dim=args.bond_dim,
        seed=args.seed,
    )

    print(
        f"critical Ising TTN: N={args.nsites}, chi={args.bond_dim}, "
        f"boundary={'periodic' if args.periodic else 'open'}"
    )
    previous_energy = np.inf
    for sweep in range(args.sweeps):
        path = state.preorder + tuple(reversed(state.preorder))
        for site in path:
            energy = optimize_site(state, site, terms)
        change = previous_energy - energy
        print(f"sweep {sweep + 1:2d}: E = {energy:.12f}, delta = {change:.3e}")
        if abs(change) < args.tolerance:
            break
        previous_energy = energy

    energy = hamiltonian_expectation(state, terms)
    print(f"contracted TTN energy: {energy:.12f}")
    print(f"energy per site:       {energy / args.nsites:.12f}")
    benchmark_energy = None
    if not args.periodic:
        benchmark_energy = exact_open_energy(
            args.nsites,
            coupling=coupling,
            field=field,
        )
        print(f"free-fermion energy:   {benchmark_energy:.12f}")
        print(f"TTN energy error:      {energy - benchmark_energy:.3e}")

    reference = args.nsites // 4
    max_distance = min(args.nsites // 2, args.nsites - reference - 1)
    distances = range(1, max_distance + 1)
    correlations = [
        connected_zz(state, z, reference, reference + distance)
        for distance in distances
    ]

    exact_energy = exact_vector = None
    if args.nsites <= args.exact_max_sites:
        exact_energy, exact_vector = exact_reference(
            args.nsites,
            coupling=coupling,
            field=field,
            periodic=args.periodic,
        )
        vector = state.state_vector(normalize=True)
        fidelity = abs(np.vdot(exact_vector, vector)) ** 2
        if benchmark_energy is None:
            print(f"exact sparse energy:    {exact_energy:.12f}")
            print(f"TTN energy error:       {energy - exact_energy:.3e}")
        print(f"ground-state fidelity:  {fidelity:.12f}")

    print(f"connected Czz from reference site {reference}")
    print("r   TTN Czz       exact Czz")
    for distance, value in zip(distances, correlations):
        if exact_vector is None:
            print(f"{distance}   {value: .8f}       --")
        else:
            exact = dense_connected_zz(
                exact_vector,
                reference,
                reference + distance,
                args.nsites,
            )
            print(f"{distance}   {value: .8f}   {exact: .8f}")


if __name__ == "__main__":
    main()
