"""Exact spinful-fermion chains in canonical physical and symmetry spaces."""

from __future__ import annotations

from numbers import Integral

import numpy as np
from scipy.sparse import eye

from pyqed import dag
from pyqed.lattice import Block, SpinHalfFermionSite
from pyqed.phys import eigh
from pyqed.qchem.jordan_wigner.spinful import annihilate, create
from pyqed.symmetry import (
    Irrep,
    Leg,
    ProductSymmetry,
    U1Symmetry,
)


def _product_irrep_layout(site, nsites, *, charge_components):
    """Return an ``Leg`` and primitive indices for a product basis."""
    nsites = int(nsites)
    components = tuple(int(component) for component in charge_components)
    if nsites < 1:
        raise ValueError("nsites must be positive.")
    if site.charges is None:
        raise ValueError("product-sector construction requires site charges.")
    if not components:
        raise ValueError("at least one charge component is required.")
    charge_rank = len(site.charges[0])
    if any(component < 0 or component >= charge_rank for component in components):
        raise IndexError("charge component is out of range.")

    total_dim = site.dim**nsites
    configurations = np.column_stack(
        np.unravel_index(np.arange(total_dim), (site.dim,) * nsites)
    )
    local_charges = np.asarray(site.charges, dtype=int)
    charges = local_charges[configurations].sum(axis=1)[:, components]
    if len(components) == 1:
        symmetry = U1Symmetry(site.charge_labels[components[0]])
        state_irreps = tuple(Irrep(int(charge[0])) for charge in charges)
    else:
        factors = tuple(
            U1Symmetry(site.charge_labels[component])
            for component in components
        )
        symmetry = ProductSymmetry(
            factors,
            name="x".join(factor.name for factor in factors),
        )
        state_irreps = tuple(
            Irrep(tuple(int(value) for value in charge))
            for charge in charges
        )

    grouped_indices = {}
    for index, irrep in enumerate(state_irreps):
        grouped_indices.setdefault(irrep, []).append(index)
    indices = {
        irrep: np.asarray(values, dtype=int)
        for irrep, values in grouped_indices.items()
    }
    space = Leg(
        {irrep: int(values.size) for irrep, values in indices.items()},
        symmetry=symmetry,
    )
    return space, indices


def _restricted_matrix(matrix, indices):
    indices = np.asarray(indices, dtype=int)
    return matrix[np.ix_(indices, indices)]


class SpinHalfFermionChain:
    """Exact diagonalization of a spin-$1/2$ fermion orbital chain.

    The canonical :class:`SpinHalfFermionSite` describes every local physical
    space.  ``sector_space`` and ``sector_indices`` describe the selected
    many-body symmetry decomposition; energies and vectors remain solver
    state owned by this chain.
    """

    def __init__(self, h1e, eri, nelec=None):
        h1e = np.asarray(h1e)
        eri = np.asarray(eri)
        if h1e.ndim != 2 or h1e.shape[0] != h1e.shape[1]:
            raise ValueError("h1e must be a square matrix.")
        nsites = int(h1e.shape[0])
        if eri.shape != (nsites,) * 4:
            raise ValueError(f"eri must have shape {(nsites,) * 4}.")

        self.L = self.nsites = nsites
        self.h1e = h1e
        self.eri = eri
        self.nelec = nelec
        self.site = SpinHalfFermionSite()
        self.sites = (self.site,) * nsites
        self.d = self.site.dim

        self.H = None
        self.e_tot = None
        self.X = None
        self.operators = None
        self.sector_space = None
        self.sector_indices = {}
        self.sector_energies = {}
        self.sector_vectors = {}
        self.target_irrep = None
        self.block = None

        self.Cu = None
        self.Cd = None
        self.Cdd = None
        self.Cdu = None
        self.Nu_tot = None
        self.Nd_tot = None
        self.Ntot = None
        self.Sz = None
        self.Sx = None
        self.Sy = None
        self.Sp = None
        self.S2 = None

    def full_diagonalization(self, nstates=1):
        return self.brute_force(nstates)

    def brute_force(self, nstates=1):
        """Diagonalize the full Hilbert space without symmetry restriction."""
        if self.H is None:
            self.jordan_wigner()
        energies, vectors = eigh(self.H, k=int(nstates), which="SA")
        self.e_tot = energies
        self.X = vectors

        nu = np.diag(dag(vectors) @ self.Nu_tot @ vectors)
        nd = np.diag(dag(vectors) @ self.Nd_tot @ vectors)
        spin = np.real(np.diag(dag(vectors) @ self.S2 @ vectors))
        print("\n   Energy     Nu     Nd     SS")
        for energy, n_up, n_down, spin_value in zip(energies, nu, nd, spin):
            print(f"{energy:12.6f}  {n_up:4.2f}   {n_down:4.2f}  {spin_value:4.2f}")
        return energies, vectors

    def _set_sector_layout(self, *, spin_resolved):
        components = (0, 1) if spin_resolved else (0,)
        self.sector_space, self.sector_indices = _product_irrep_layout(
            self.site,
            self.nsites,
            charge_components=components,
        )
        self.block = Block(
            h=self.H,
            qn=self.sector_space,
            data={
                "indices": self.sector_indices,
                "energies": self.sector_energies,
                "vectors": self.sector_vectors,
            },
        )

    def _solve_sector(self, irrep, nstates):
        try:
            indices = self.sector_indices[irrep]
        except KeyError as error:
            raise ValueError(f"requested symmetry sector {irrep.charge!r} is empty.") from error
        block = _restricted_matrix(self.H, indices)
        energies, vectors = eigh(block, k=int(nstates), which="SA")
        self.sector_energies[irrep] = energies
        self.sector_vectors[irrep] = vectors
        return energies, vectors

    def run(self, nstates=1):
        """Diagonalize in total-number or fixed-$(N_\\uparrow,N_\\downarrow)$ sectors."""
        if self.H is None:
            self.jordan_wigner()
        self.sector_energies = {}
        self.sector_vectors = {}

        if self.nelec is None:
            self._set_sector_layout(spin_resolved=False)
            energies = []
            vectors = []
            for irrep in self.sector_space.irreps:
                sector_energies, sector_vectors = self._solve_sector(irrep, nstates)
                energies.append(sector_energies.copy())
                vectors.append(sector_vectors.copy())
                print(f"# electrons = {irrep.charge}, e = {sector_energies}")
            self.e_tot = energies
            self.X = vectors
            self.target_irrep = None
        elif isinstance(self.nelec, Integral):
            self._set_sector_layout(spin_resolved=False)
            self.target_irrep = Irrep(int(self.nelec))
            self.e_tot, self.X = self._solve_sector(self.target_irrep, nstates)
            for root, energy in enumerate(self.e_tot):
                print(f"Root {root} = {energy}")
        elif isinstance(self.nelec, (list, tuple)) and len(self.nelec) == 2:
            n_up, n_down = (int(value) for value in self.nelec)
            self._set_sector_layout(spin_resolved=True)
            self.target_irrep = Irrep((n_up + n_down, n_up - n_down))
            self.e_tot, self.X = self._solve_sector(self.target_irrep, nstates)
            print("\nExact diagonalization with N x 2Sz U(1) symmetry")
            print("number of states = ", self.sector_indices[self.target_irrep].size)
        else:
            raise ValueError("nelec must be an integer or an (N_up, N_down) pair.")
        return self

    def jordan_wigner(self, forward=True, aosym="8"):
        """Construct the second-quantized Hamiltonian by Jordan–Wigner mapping."""
        del aosym
        h1e = self.h1e
        eri = self.eri
        norb = self.nsites

        Cu = annihilate(norb, spin="up", forward=forward)
        Cd = annihilate(norb, spin="down", forward=forward)
        Cdu = create(norb, spin="up", forward=forward)
        Cdd = create(norb, spin="down", forward=forward)
        self.Cu, self.Cd, self.Cdu, self.Cdd = Cu, Cd, Cdu, Cdd

        sx = sy = sz = sp = 0
        for orbital in range(norb):
            sz += 0.5 * (Cdu[orbital] @ Cu[orbital] - Cdd[orbital] @ Cd[orbital])
            sx += 0.5 * (Cdu[orbital] @ Cd[orbital] + Cdd[orbital] @ Cu[orbital])
            sy += -0.5j * (Cdu[orbital] @ Cd[orbital] - Cdd[orbital] @ Cu[orbital])
            sp += Cdu[orbital] @ Cd[orbital]
        self.Sx, self.Sy, self.Sz, self.Sp = sx, sy, sz, sp
        self.S2 = sx @ sx + sy @ sy + sz @ sz

        hamiltonian = 0
        for p in range(norb):
            for q in range(norb):
                hamiltonian += h1e[p, q] * (
                    Cdu[p] @ Cu[q] + Cdd[p] @ Cd[q]
                )

        n_up = n_down = 0
        for orbital in range(norb):
            n_up += Cdu[orbital] @ Cu[orbital]
            n_down += Cdd[orbital] @ Cd[orbital]
        self.Nu_tot = n_up
        self.Nd_tot = n_down
        self.Ntot = n_up + n_down

        for p in range(norb):
            for q in range(norb):
                for r in range(norb):
                    for s in range(norb):
                        hamiltonian += 0.5 * eri[p, q, r, s] * (
                            Cdu[p] @ Cdu[r] @ Cu[s] @ Cu[q]
                            + Cdu[p] @ Cdd[r] @ Cd[s] @ Cu[q]
                            + Cdd[p] @ Cdu[r] @ Cu[s] @ Cd[q]
                            + Cdd[p] @ Cdd[r] @ Cd[s] @ Cd[q]
                        )

        self.H = hamiltonian
        self.operators = {
            "H": hamiltonian,
            "Cu": Cu,
            "Cd": Cd,
            "Cdu": Cdu,
            "Cdd": Cdd,
            "Nu": n_up,
            "Nd": n_down,
            "Ntot": self.Ntot,
        }
        return hamiltonian

    def fix_nelec(self, nelec=None, strength=1.0):
        """Add a quadratic particle-number penalty to the Hamiltonian."""
        if self.H is None:
            self.jordan_wigner()
        target = self.nelec if nelec is None else nelec
        if target is None:
            raise ValueError("a target electron number is required.")
        total = int(sum(target)) if isinstance(target, (tuple, list)) else int(target)
        identity = eye(self.H.shape[0], format="csr")
        delta = self.Ntot - total * identity
        self.H = self.H + float(strength) * (delta @ delta)
        return self


__all__ = ["SpinHalfFermionChain", "annihilate", "create"]
