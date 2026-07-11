#!/usr/bin/env python3
"""
Abelian NARG calculation for a finite 2D spinful Hubbard lattice.

This uses the quantum-chemistry NARG backend with Hubbard-model integrals:

    h_ij = -t for nearest-neighbor square-lattice bonds
    (ii|ii) = U

The 2D lattice is flattened in row-major order, i = x + Lx * y.  For small
lattices the script also prints a fixed-sector sparse ED reference from
``hubbard_2d_ed.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hubbard_2d_ed import hubbard_2d_hamiltonian, square_lattice_bonds
from pyqed.narg.qchem import abelian as abelian_narg


class HubbardMol:
    """Minimal molecule-like object expected by the qchem NARG kernel."""

    def __init__(self, nup: int, ndown: int):
        self.nelec = (int(nup), int(ndown))
        self.spin = int(nup) - int(ndown)

    def energy_nuc(self):
        return 0.0


def hubbard_2d_integrals(
    lx: int,
    ly: int,
    *,
    t: float,
    u: float,
    mu: float = 0.0,
    periodic_x: bool = False,
    periodic_y: bool = False,
    ordering: str = "row-major",
):
    """Return spatial-orbital one- and two-electron integrals for 2D Hubbard."""
    nsites = int(lx) * int(ly)
    orbital_order = lattice_site_order(lx, ly, ordering=ordering)
    lattice_to_orbital = {site: orbital for orbital, site in enumerate(orbital_order)}
    h1e = np.zeros((nsites, nsites), dtype=float)
    for i, j in square_lattice_bonds(lx, ly, periodic_x=periodic_x, periodic_y=periodic_y):
        p = lattice_to_orbital[i]
        q = lattice_to_orbital[j]
        h1e[p, q] += -float(t)
        h1e[q, p] += -float(t)
    if mu:
        h1e[np.diag_indices(nsites)] -= float(mu)

    eri = np.zeros((nsites, nsites, nsites, nsites), dtype=float)
    for site in range(nsites):
        eri[site, site, site, site] = float(u)
    return h1e, eri


def closed_shell_density_from_orbitals(orbital_energy, coeff, n_pairs: int, *, degeneracy_tol=1.0e-9):
    """Return a spin-summed closed-shell density, fractionally filling degeneracies."""
    n_pairs = int(n_pairs)
    occ = np.zeros_like(orbital_energy, dtype=float)
    start = 0
    remaining = n_pairs
    while start < len(orbital_energy) and remaining > 0:
        stop = start + 1
        while (
            stop < len(orbital_energy)
            and abs(float(orbital_energy[stop] - orbital_energy[start])) <= degeneracy_tol
        ):
            stop += 1
        group_size = stop - start
        if remaining >= group_size:
            occ[start:stop] = 2.0
            remaining -= group_size
        else:
            occ[start:stop] = 2.0 * remaining / group_size
            remaining = 0
        start = stop
    return coeff @ np.diag(occ) @ coeff.T


def restricted_hubbard_hf(
    h1e,
    eri,
    *,
    nup: int,
    ndown: int,
    max_cycle=100,
    conv_tol=1.0e-10,
    degeneracy_tol=1.0e-9,
):
    """Solve a restricted Hubbard HF problem and return canonical orbitals."""
    if int(nup) != int(ndown):
        raise ValueError("Restricted Hubbard HF requires nup == ndown.")
    nocc = int(nup)
    onsite_u = np.asarray([eri[i, i, i, i] for i in range(h1e.shape[0])], dtype=float)
    eps, coeff = np.linalg.eigh(h1e)
    density = closed_shell_density_from_orbitals(
        eps,
        coeff,
        nocc,
        degeneracy_tol=degeneracy_tol,
    )
    energy = float(
        np.einsum("ij,ji->", density, h1e)
        + 0.25 * np.dot(onsite_u, np.diag(density) ** 2)
    )

    for _cycle in range(int(max_cycle)):
        fock = h1e + np.diag(0.5 * onsite_u * np.diag(density))
        new_eps, new_coeff = np.linalg.eigh(fock)
        new_density = closed_shell_density_from_orbitals(
            new_eps,
            new_coeff,
            nocc,
            degeneracy_tol=degeneracy_tol,
        )
        new_energy = float(
            np.einsum("ij,ji->", new_density, h1e)
            + 0.25 * np.dot(onsite_u, np.diag(new_density) ** 2)
        )

        d_energy = abs(new_energy - energy)
        d_density = np.linalg.norm(new_density - density)
        if d_energy < conv_tol and d_density < np.sqrt(conv_tol):
            return new_energy, new_eps, new_coeff, new_density

        energy = new_energy
        density = new_density
        eps = new_eps
        coeff = new_coeff

    return energy, eps, coeff, density


def unrestricted_hubbard_hf(
    h1e,
    eri,
    *,
    nup: int,
    ndown: int,
    lx: int,
    ly: int,
    ordering: str,
    max_cycle=200,
    conv_tol=1.0e-10,
    damping=0.35,
):
    """Solve a simple UHF Hubbard mean-field problem and return densities."""
    onsite_u = np.asarray([eri[i, i, i, i] for i in range(h1e.shape[0])], dtype=float)
    order = lattice_site_order(lx, ly, ordering=ordering)
    lattice_to_orbital = {site: orbital for orbital, site in enumerate(order)}
    nalpha = np.full(h1e.shape[0], 0.5)
    nbeta = np.full(h1e.shape[0], 0.5)
    for y in range(ly):
        for x in range(lx):
            orbital = lattice_to_orbital[x + lx * y]
            stagger = 1.0 if (x + y) % 2 == 0 else -1.0
            nalpha[orbital] += 0.2 * stagger
            nbeta[orbital] -= 0.2 * stagger

    pa = np.diag(nalpha)
    pb = np.diag(nbeta)
    energy = None
    eps_a = eps_b = None
    ca = cb = None

    for _cycle in range(int(max_cycle)):
        fa = h1e + np.diag(onsite_u * np.diag(pb))
        fb = h1e + np.diag(onsite_u * np.diag(pa))
        eps_a, ca = np.linalg.eigh(fa)
        eps_b, cb = np.linalg.eigh(fb)
        new_pa = ca[:, : int(nup)] @ ca[:, : int(nup)].T
        new_pb = cb[:, : int(ndown)] @ cb[:, : int(ndown)].T
        mixed_pa = (1.0 - damping) * new_pa + damping * pa
        mixed_pb = (1.0 - damping) * new_pb + damping * pb
        new_energy = float(
            np.einsum("ij,ji->", mixed_pa + mixed_pb, h1e)
            + np.dot(onsite_u, np.diag(mixed_pa) * np.diag(mixed_pb))
        )
        if energy is not None:
            d_energy = abs(new_energy - energy)
            d_density = np.linalg.norm(mixed_pa - pa) + np.linalg.norm(mixed_pb - pb)
            if d_energy < conv_tol and d_density < np.sqrt(conv_tol):
                return new_energy, eps_a, eps_b, ca, cb, mixed_pa, mixed_pb
        energy = new_energy
        pa = mixed_pa
        pb = mixed_pb

    return energy, eps_a, eps_b, ca, cb, pa, pb


def natural_orbitals_from_density(density):
    """Return natural occupations and orbitals, ordered around half filling."""
    occ, coeff = np.linalg.eigh(0.5 * (density + density.T))
    order = np.lexsort((np.arange(len(occ)), np.abs(occ - 1.0)))
    return occ[order], coeff[:, order]


def _unique_momentum_group(keys):
    seen = set()
    group = []
    for key in keys:
        if key not in seen:
            seen.add(key)
            group.append(key)
    return group


def _paired_momentum_order(records, lx: int, ly: int, *, quartet: bool = False, nesting: bool = False):
    by_index = {(item[3], item[4]): item for item in records}
    visited = set()
    ordered = []
    qx = lx // 2
    qy = ly // 2
    for ny in range(ly):
        for nx in range(lx):
            key = (nx, ny)
            if key in visited:
                continue
            if nesting:
                keys = _unique_momentum_group(
                    [
                        key,
                        ((nx + qx) % lx, (ny + qy) % ly),
                        ((lx - nx) % lx, (ly - ny) % ly),
                        ((qx - nx) % lx, (qy - ny) % ly),
                    ]
                )
            elif quartet:
                keys = _unique_momentum_group(
                    [
                        key,
                        ((lx - nx) % lx, (ly - ny) % ly),
                        (nx, (ly - ny) % ly),
                        ((lx - nx) % lx, ny),
                    ]
                )
            else:
                keys = _unique_momentum_group([key, ((lx - nx) % lx, (ly - ny) % ly)])
            for item_key in keys:
                if item_key not in visited:
                    ordered.append(by_index[item_key])
                    visited.add(item_key)
    return ordered


def momentum_orbitals(lx: int, ly: int, *, ordering: str = "row-major", order_by: str = "energy", t: float = 1.0):
    """Return 2D Fourier orbitals in the requested one-dimensional site order."""
    site_order = lattice_site_order(lx, ly, ordering=ordering)
    momenta = []
    for ny in range(ly):
        ky = 2.0 * np.pi * ny / ly
        for nx in range(lx):
            kx = 2.0 * np.pi * nx / lx
            energy = -2.0 * float(t) * (np.cos(kx) + np.cos(ky))
            momenta.append((kx, ky, energy, nx, ny))

    key = order_by.lower().replace("_", "-")
    if key == "energy":
        momenta.sort(key=lambda item: (item[2], item[4], item[3]))
    elif key == "fermi":
        energies = np.sort(np.asarray([item[2] for item in momenta], dtype=float))
        nocc = len(momenta) // 2
        fermi = 0.5 * (energies[nocc - 1] + energies[nocc])
        momenta.sort(key=lambda item: (abs(item[2] - fermi), item[2], item[4], item[3]))
    elif key in {"inversion-pair", "inversion-pairs", "pair", "pm-pair"}:
        momenta = _paired_momentum_order(momenta, lx, ly)
    elif key in {"sign-quartet", "sign-quartets", "quartet"}:
        momenta = _paired_momentum_order(momenta, lx, ly, quartet=True)
    elif key in {"nesting-quartet", "nesting-quartets", "nested-quartet"}:
        if lx % 2 or ly % 2:
            raise ValueError("nesting-quartet momentum order requires even lx and ly.")
        momenta = _paired_momentum_order(momenta, lx, ly, nesting=True)
    elif key not in {"grid", "natural"}:
        raise ValueError(
            "momentum order must be 'energy', 'fermi', 'grid', "
            "'inversion-pair', 'sign-quartet', or 'nesting-quartet'."
        )

    coeff = np.empty((lx * ly, lx * ly), dtype=complex)
    for col, (kx, ky, _energy, _nx, _ny) in enumerate(momenta):
        for row, lattice_site in enumerate(site_order):
            x = lattice_site % lx
            y = lattice_site // lx
            coeff[row, col] = np.exp(1j * (kx * x + ky * y)) / np.sqrt(lx * ly)
    return coeff, momenta


def transform_spatial_integrals(h1e, eri, coeff):
    """Transform spatial integrals by C_mu,p canonical orbital coefficients."""
    h_mo = coeff.conj().T @ h1e @ coeff
    eri_mo = np.einsum(
        "ijkl,ip,jq,kr,ls->pqrs",
        eri,
        coeff.conj(),
        coeff,
        coeff.conj(),
        coeff,
        optimize=True,
    )
    return np.real_if_close(h_mo, tol=1000), np.real_if_close(eri_mo, tol=1000)


def lattice_site_order(lx: int, ly: int, *, ordering: str = "row-major"):
    """Return lattice-site labels in the requested 1D orbital order."""
    key = ordering.lower().replace("_", "-")
    if key in {"row", "row-major", "rowmajor"}:
        return [x + lx * y for y in range(ly) for x in range(lx)]
    if key == "snake":
        order = []
        for y in range(ly):
            xs = range(lx) if y % 2 == 0 else range(lx - 1, -1, -1)
            order.extend(x + lx * y for x in xs)
        return order
    raise ValueError("ordering must be 'row-major' or 'snake'.")


def lowest_ed_energies(hamiltonian, nroots: int):
    """Dense fallback for tiny fixed-sector ED references."""
    nroots = min(max(1, int(nroots)), hamiltonian.shape[0])
    if nroots >= hamiltonian.shape[0]:
        return np.linalg.eigvalsh(hamiltonian.toarray())[:nroots]

    from scipy.sparse.linalg import eigsh

    evals = eigsh(hamiltonian, k=nroots, which="SA", return_eigenvectors=False)
    evals.sort()
    return evals


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
    parser.add_argument("--ordering", choices=("row-major", "snake"), default="row-major")
    parser.add_argument("--basis", choices=("site", "hcore", "momentum", "rhf", "uhf-natural"), default="site")
    parser.add_argument(
        "--momentum-order",
        choices=("energy", "fermi", "grid", "inversion-pair", "sign-quartet", "nesting-quartet"),
        default="energy",
    )
    parser.add_argument("--hf-max-cycle", type=int, default=100)
    parser.add_argument("--hf-conv-tol", type=float, default=1.0e-10)
    parser.add_argument("--hf-damping", type=float, default=0.35)
    parser.add_argument("--D", type=int, default=32, help="Retained NARG bond dimension.")
    parser.add_argument("--n0", type=int, default=1, help="Initial block size in spatial orbitals.")
    parser.add_argument("--growth-sites", default="2", help="Use 1, 2, 3, 4, or auto site growth.")
    parser.add_argument("--nroots", type=int, default=1, help="Number of lowest roots.")
    parser.add_argument("--no-ed", action="store_true", help="Skip the ED reference calculation.")
    return parser.parse_args()


def main():
    args = parse_args()
    nsites = args.lx * args.ly
    nup = nsites // 2 if args.nup is None else int(args.nup)
    ndown = nsites - nup if args.ndown is None else int(args.ndown)
    growth_sites = args.growth_sites if args.growth_sites == "auto" else int(args.growth_sites)
    n0 = min(max(1, int(args.n0)), nsites - 1)

    h1e, eri = hubbard_2d_integrals(
        args.lx,
        args.ly,
        t=args.hopping,
        u=args.onsite_u,
        mu=args.mu,
        periodic_x=args.periodic_x,
        periodic_y=args.periodic_y,
        ordering=args.ordering,
    )

    mf_label = None
    if args.basis == "hcore":
        hcore_eps, hcore_coeff = np.linalg.eigh(h1e)
        h1e, eri = transform_spatial_integrals(h1e, eri, hcore_coeff)
        mf_label = f"Hcore orbital energies [{hcore_eps[0]:.6f}, ..., {hcore_eps[-1]:.6f}]"
    elif args.basis == "momentum":
        if not (args.periodic_x and args.periodic_y):
            raise ValueError("--basis momentum requires --periodic-x --periodic-y.")
        momentum_coeff, momenta = momentum_orbitals(
            args.lx,
            args.ly,
            ordering=args.ordering,
            order_by=args.momentum_order,
            t=args.hopping,
        )
        h1e, eri = transform_spatial_integrals(h1e, eri, momentum_coeff)
        momentum_energy = np.asarray([item[2] for item in momenta], dtype=float)
        mf_label = (
            f"Momentum basis order={args.momentum_order}; "
            f"epsilon range = [{momentum_energy.min():.6f}, {momentum_energy.max():.6f}]"
        )
    elif args.basis == "rhf":
        hf_energy, hf_eps, hf_coeff, _hf_density = restricted_hubbard_hf(
            h1e,
            eri,
            nup=nup,
            ndown=ndown,
            max_cycle=args.hf_max_cycle,
            conv_tol=args.hf_conv_tol,
        )
        h1e, eri = transform_spatial_integrals(h1e, eri, hf_coeff)
        mf_label = f"RHF energy = {hf_energy:.12f}"
    elif args.basis == "uhf-natural":
        uhf = unrestricted_hubbard_hf(
            h1e,
            eri,
            nup=nup,
            ndown=ndown,
            lx=args.lx,
            ly=args.ly,
            ordering=args.ordering,
            max_cycle=args.hf_max_cycle,
            conv_tol=args.hf_conv_tol,
            damping=args.hf_damping,
        )
        hf_energy, _eps_a, _eps_b, _ca, _cb, pa, pb = uhf
        natural_occ, natural_coeff = natural_orbitals_from_density(pa + pb)
        h1e, eri = transform_spatial_integrals(h1e, eri, natural_coeff)
        mf_label = (
            f"UHF energy = {hf_energy:.12f}; "
            f"natural occ range = [{natural_occ.min():.6f}, {natural_occ.max():.6f}]"
        )

    abelian_narg.mol = HubbardMol(nup, ndown)
    energies, _vectors, tensors, tensor_qns = abelian_narg.kernel(
        h1e,
        eri,
        D=int(args.D),
        n0=n0,
        nstates=int(args.nroots),
        growth_sites=growth_sites,
        return_tensors=True,
        return_tensor_qns=True,
    )

    print(
        f"2D Hubbard NARG {args.lx}x{args.ly}, "
        f"N_up={nup}, N_down={ndown}, t={args.hopping:g}, "
        f"U={args.onsite_u:g}, mu={args.mu:g}, "
        f"ordering={args.ordering}, basis={args.basis}"
    )
    if mf_label is not None:
        print(mf_label)
    print(f"D={args.D}, n0={n0}, growth_sites={growth_sites}")
    print(f"growth pattern={[factor.get('growth_sites') for factor in tensor_qns['factors']]}")
    for root, energy in enumerate(np.ravel(energies)):
        print(f"NARG root {root}: E = {energy:.12f}")

    if not args.no_ed:
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
        ed = lowest_ed_energies(hamiltonian, args.nroots)
        print(f"ED Hilbert dimension={info['dimension']}")
        for root, energy in enumerate(ed):
            delta = np.ravel(energies)[root] - energy if root < len(energies) else np.nan
            print(f"ED root {root}:   E = {energy:.12f}   NARG-ED = {delta:.3e}")


if __name__ == "__main__":
    main()
