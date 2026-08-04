"""Optimize a balanced TTN at the critical transverse-field Ising point."""

from __future__ import annotations

import argparse

import numpy as np

from pyqed.letta import LocalHamiltonian, LocalTerm
from pyqed.tn import balanced_ttn


def critical_ising_hamiltonian(dims, nspins, *, coupling=1.0, field=1.0):
    """Return the open critical Ising Hamiltonian on the physical leaves."""
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.diag([1.0, -1.0])
    terms = [
        LocalTerm((site,), -field * z)
        for site in range(nspins)
    ]
    terms.extend(
        LocalTerm((site, site + 1), -coupling * np.kron(x, x))
        for site in range(nspins - 1)
    )
    return LocalHamiltonian(dims, terms)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nspins", type=int, default=4)
    parser.add_argument("--bond-dim", type=int, default=4)
    parser.add_argument("--sweeps", type=int, default=6)
    parser.add_argument("--tol", type=float, default=1.0e-10)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    state = balanced_ttn(
        args.nspins,
        physical_dim=2,
        bond_dim=args.bond_dim,
        seed=args.seed,
    )
    hamiltonian = critical_ising_hamiltonian(state.dims, args.nspins)
    initial = state.expectation(hamiltonian)
    state.run(hamiltonian, nsweeps=args.sweeps, tol=args.tol)

    print(f"initial energy: {initial:.12f}")
    print(f"TTN energy:     {state.energy:.12f}")
    print(f"sweeps:         {state.ncompleted}")
    print(f"status:         {state.message}")
    if args.nspins <= 12:
        exact = np.linalg.eigvalsh(hamiltonian.to_dense())[0]
        print(f"exact energy:   {exact:.12f}")
        print(f"error:          {state.energy - exact:.3e}")


if __name__ == "__main__":
    main()
