#!/usr/bin/env python3
"""
Sparse exact diagonalization for a finite 2D spinful Hubbard lattice.

The Hamiltonian is

    H = -t sum_<ij>,sigma (c^dagger_i,sigma c_j,sigma + h.c.)
        + U sum_i n_i,up n_i,down
        - mu sum_i (n_i,up + n_i,down)

Sites are ordered row-major, i = x + Lx * y.  The calculation works in a
fixed (N_up, N_down) sector, so it is useful for small exact 2D references.
"""

from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np
from scipy.sparse import coo_matrix, diags, eye, kron
from scipy.sparse.linalg import eigsh


def square_lattice_bonds(lx: int, ly: int, *, periodic_x: bool = False, periodic_y: bool = False):
    """Return nearest-neighbor bonds for an Lx by Ly square lattice."""
    bonds = []
    for y in range(ly):
        for x in range(lx):
            site = x + lx * y
            if x + 1 < lx:
                bonds.append((site, site + 1))
            elif periodic_x and lx > 2:
                bonds.append((site, lx * y))
            if y + 1 < ly:
                bonds.append((site, site + lx))
            elif periodic_y and ly > 2:
                bonds.append((site, x))
    return bonds


def fixed_particle_basis(nsites: int, nelec: int):
    """Return integer bitstrings with exactly nelec occupied sites."""
    if nelec < 0 or nelec > nsites:
        raise ValueError("nelec must lie between 0 and nsites.")
    basis = []
    for occ in combinations(range(nsites), nelec):
        bits = 0
        for site in occ:
            bits |= 1 << site
        basis.append(bits)
    return np.asarray(basis, dtype=np.int64)


def fermion_hop_sign(bits: int, dst: int, src: int) -> int:
    """Sign of c^dagger_dst c_src acting on a one-spin bitstring."""
    lo, hi = sorted((dst, src))
    between_mask = ((1 << hi) - (1 << (lo + 1))) if hi > lo + 1 else 0
    return -1 if ((int(bits) & between_mask).bit_count() % 2) else 1


def one_spin_hopping(nsites: int, nelec: int, bonds, t: float):
    """Build -t sum_<ij> (c^dag_i c_j + c^dag_j c_i) in a fixed-N sector."""
    basis = fixed_particle_basis(nsites, nelec)
    index = {int(bits): row for row, bits in enumerate(basis)}
    rows = []
    cols = []
    data = []

    for col, bits in enumerate(basis):
        bits = int(bits)
        for i, j in bonds:
            for dst, src in ((i, j), (j, i)):
                if (bits >> src) & 1 and not ((bits >> dst) & 1):
                    hopped = bits ^ (1 << src) ^ (1 << dst)
                    rows.append(index[hopped])
                    cols.append(col)
                    data.append(-float(t) * fermion_hop_sign(bits, dst, src))

    dim = len(basis)
    return coo_matrix((data, (rows, cols)), shape=(dim, dim)).tocsr(), basis


def hubbard_2d_hamiltonian(
    lx: int,
    ly: int,
    *,
    nup: int,
    ndown: int,
    t: float,
    u: float,
    mu: float = 0.0,
    periodic_x: bool = False,
    periodic_y: bool = False,
):
    """Build the sparse 2D Hubbard Hamiltonian in a fixed (N_up, N_down) sector."""
    nsites = int(lx) * int(ly)
    bonds = square_lattice_bonds(lx, ly, periodic_x=periodic_x, periodic_y=periodic_y)
    hup, up_basis = one_spin_hopping(nsites, nup, bonds, t)
    hdown, down_basis = one_spin_hopping(nsites, ndown, bonds, t)

    iup = eye(len(up_basis), format="csr")
    idown = eye(len(down_basis), format="csr")
    hamiltonian = kron(hup, idown, format="csr") + kron(iup, hdown, format="csr")

    diag = np.empty(len(up_basis) * len(down_basis), dtype=float)
    offset = -float(mu) * (int(nup) + int(ndown))
    k = 0
    for up_bits in up_basis:
        up_bits = int(up_bits)
        for down_bits in down_basis:
            doublons = (up_bits & int(down_bits)).bit_count()
            diag[k] = float(u) * doublons + offset
            k += 1
    hamiltonian = hamiltonian + diags(diag, format="csr")

    return hamiltonian, {
        "nsites": nsites,
        "bonds": bonds,
        "nup": int(nup),
        "ndown": int(ndown),
        "dimension": int(hamiltonian.shape[0]),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lx", type=int, default=2, help="Number of sites along x.")
    parser.add_argument("--ly", type=int, default=2, help="Number of sites along y.")
    parser.add_argument("--nup", type=int, default=None, help="Number of up-spin electrons.")
    parser.add_argument("--ndown", type=int, default=None, help="Number of down-spin electrons.")
    parser.add_argument("-t", "--hopping", type=float, default=1.0, help="Nearest-neighbor hopping t.")
    parser.add_argument("-U", "--onsite-u", type=float, default=4.0, help="On-site repulsion U.")
    parser.add_argument("--mu", type=float, default=0.0, help="Chemical potential.")
    parser.add_argument("--periodic-x", action="store_true", help="Use periodic boundary conditions in x.")
    parser.add_argument("--periodic-y", action="store_true", help="Use periodic boundary conditions in y.")
    parser.add_argument("--nroots", type=int, default=4, help="Number of lowest eigenvalues.")
    return parser.parse_args()


def main():
    args = parse_args()
    nsites = args.lx * args.ly
    nup = nsites // 2 if args.nup is None else args.nup
    ndown = nsites - nup if args.ndown is None else args.ndown

    hamiltonian, info = hubbard_2d_hamiltonian(
        args.lx,
        args.ly,
        nup=nup,
        ndown=ndown,
        t=args.hopping,
        u=args.onsite_u,
        mu=args.mu,
        periodic_x=args.periodic_x,
        periodic_y=args.periodic_y,
    )

    nroots = min(max(1, int(args.nroots)), hamiltonian.shape[0])
    if nroots >= hamiltonian.shape[0]:
        evals = np.linalg.eigvalsh(hamiltonian.toarray())[:nroots]
    else:
        evals = eigsh(hamiltonian, k=nroots, which="SA", return_eigenvectors=False)
        evals.sort()

    print(
        f"2D Hubbard {args.lx}x{args.ly}, "
        f"N_up={info['nup']}, N_down={info['ndown']}, "
        f"t={args.hopping:g}, U={args.onsite_u:g}, mu={args.mu:g}"
    )
    print(f"bonds={len(info['bonds'])}, Hilbert dimension={info['dimension']}")
    for root, energy in enumerate(evals):
        print(f"Root {root}: E = {energy:.12f}")


if __name__ == "__main__":
    main()
