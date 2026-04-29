#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.linalg import eigh

from pyqed.qchem.basis import S, T
from pyqed.qchem.pbc.ewald import (
    ewald_nuclear_repulsion_1d_inf_vacuum,
    gaussian_pair_ft_s,
    inf_vacuum_1d_gv_weights,
    reciprocal_vectors,
    short_range_eri_s,
    short_range_point_charge_s,
)


def _shifted_gaussian(fn, shift):
    shifted = object.__new__(fn.__class__)
    shifted.__dict__ = dict(fn.__dict__)
    shifted.origin = np.asarray(fn.origin, dtype=float) + np.asarray(shift, dtype=float)
    return shifted


def _symmetrize(mat):
    return 0.5 * (mat + mat.conj().T)


class EwaldRHF:
    """
    Experimental native Ewald/AFT RHF for Cartesian Gaussian PBC cells.

    This is a gamma-point 3D-supercell Ewald path for the current native PBC
    milestone. It is intentionally separate from the finite-image RHF class so
    unsupported angular momentum and k-point cases fail explicitly.
    """

    def __init__(
        self,
        cell,
        eta=0.5,
        real_cut=4,
        recip_cut=8,
        mesh=None,
        damping=0.5,
        nuclear_background=False,
    ):
        self.cell = cell
        self.eta = float(eta)
        self.real_cut = int(real_cut)
        self.recip_cut = int(recip_cut)
        self.mesh = None if mesh is None else tuple(int(x) for x in mesh)
        self.damping = float(damping)
        self.nuclear_background = bool(nuclear_background)

        self.e_tot = None
        self.e_elec = None
        self.e_nuc = None
        self.mo_energy = None
        self.mo_coeff = None
        self.mo_occ = None
        self.dm = None
        self.fock = None
        self.converged = False
        self.nkpts = 1

        self.overlap = None
        self.hcore = None
        self.eri = None
        self.madelung = None
        self._shift_cache = None
        self._shifted_basis_cache = None

    def _build_integrals(self):
        if not self.cell.built:
            self.cell.build()
        if self.cell.dimension != 1:
            raise NotImplementedError("EwaldRHF currently supports only dimension=1 cells.")
        if self.eta <= 0.0:
            raise ValueError("eta must be positive for method='ewald'.")

        basis = tuple(self.cell.unit_molecule._bas)
        if len(basis) != int(self.cell.nao):
            raise NotImplementedError(
                "EwaldRHF currently supports Cartesian AO builds only. "
                "Use integral_options={'coord_type': 'cartesian'} for p/d basis tests."
            )

        overlap, kinetic = self._periodic_overlap_kinetic(basis)
        vne = self._periodic_short_range_nuclear_attraction(basis)
        vne += self._periodic_reciprocal_nuclear_attraction(basis).real

        eri = self._periodic_short_range_eri(basis)
        eri += self._periodic_reciprocal_eri(basis)

        self.overlap = overlap
        self.hcore = _symmetrize(kinetic + vne).real
        self.eri = np.asarray(eri, dtype=float)
        if self._use_inf_vacuum_1d():
            self.madelung = self._madelung()
            self.e_nuc = float(
                ewald_nuclear_repulsion_1d_inf_vacuum(
                    self.cell.unit_molecule.atom_charges(),
                    self.cell._atom_coords,
                    self.cell.lattice_vectors,
                    eta=self.eta,
                    real_cut=self.real_cut,
                    mesh=self._reciprocal_mesh(),
                )
            )
        else:
            self.madelung = None
            self.e_nuc = float(
                self.cell.ewald_nuclear_repulsion(
                    eta=self.eta,
                    real_cut=self.real_cut,
                    recip_cut=self.recip_cut,
                    neutralizing_background=self.nuclear_background,
                )
            )

    def _use_inf_vacuum_1d(self):
        return (
            int(self.cell.dimension) == 1
            and str(getattr(self.cell, "low_dim_ft_type", "")).lower() == "inf_vacuum"
        )

    def _reciprocal_mesh(self):
        if self.mesh is not None:
            return self.mesh
        # Matches the PySCF mesh for the small H/STO-3G chain benchmark and is
        # intentionally explicit until a native cutoff-to-mesh estimator exists.
        return (31, 38, 38)

    def _periodic_shifts(self):
        if self._shift_cache is not None:
            return self._shift_cache
        a1 = np.asarray(self.cell.lattice_vectors[0], dtype=float)
        self._shift_cache = [
            float(n) * a1 for n in range(-self.real_cut, self.real_cut + 1)
        ]
        return self._shift_cache

    def _periodic_shifted_basis_sets(self, basis):
        if self._shifted_basis_cache is None:
            self._shifted_basis_cache = [
                [_shifted_gaussian(fn, shift) for fn in basis]
                for shift in self._periodic_shifts()
            ]
        return self._shifted_basis_cache

    def _periodic_overlap_kinetic(self, basis):
        nao = len(basis)
        overlap = np.zeros((nao, nao), dtype=float)
        kinetic = np.zeros((nao, nao), dtype=float)
        for shifted_basis in self._periodic_shifted_basis_sets(basis):
            for p, bp in enumerate(basis):
                for q, bq in enumerate(shifted_basis):
                    overlap[p, q] += S(bp, bq)
                    kinetic[p, q] += T(bp, bq)
        return 0.5 * (overlap + overlap.T), 0.5 * (kinetic + kinetic.T)

    def _periodic_pair_ft(self, basis, gvec):
        nao = len(basis)
        pair = np.zeros((nao, nao), dtype=np.complex128)
        for shifted_basis in self._periodic_shifted_basis_sets(basis):
            for p, bp in enumerate(basis):
                for q, bq in enumerate(shifted_basis):
                    pair[p, q] += gaussian_pair_ft_s(bp, bq, gvec)
        return pair

    def _periodic_short_range_nuclear_attraction(self, basis):
        charges = np.asarray(self.cell.unit_molecule.atom_charges(), dtype=float)
        coords = np.asarray(self.cell._atom_coords, dtype=float)
        nao = len(basis)
        mat = np.zeros((nao, nao), dtype=float)
        shifts = self._periodic_shifts()
        for shifted_basis in self._periodic_shifted_basis_sets(basis):
            for nuc_shift in shifts:
                for charge, coord in zip(charges, coords):
                    center = coord + nuc_shift
                    for p, bp in enumerate(basis):
                        for q, bq in enumerate(shifted_basis):
                            mat[p, q] -= charge * short_range_point_charge_s(
                                bp,
                                bq,
                                center,
                                self.eta,
                            )
        return 0.5 * (mat + mat.T)

    def _periodic_short_range_eri(self, basis):
        nao = len(basis)
        eri = np.zeros((nao, nao, nao, nao), dtype=float)
        shifted_sets = self._periodic_shifted_basis_sets(basis)
        for q_basis in shifted_sets:
            for r_basis in shifted_sets:
                for s_basis in shifted_sets:
                    for p, bp in enumerate(basis):
                        for q, bq in enumerate(q_basis):
                            for r, br in enumerate(r_basis):
                                for s, bs in enumerate(s_basis):
                                    eri[p, q, r, s] += short_range_eri_s(
                                        bp,
                                        bq,
                                        br,
                                        bs,
                                        self.eta,
                                    )
        return eri

    def _periodic_reciprocal_nuclear_attraction(self, basis):
        charges = np.asarray(self.cell.unit_molecule.atom_charges(), dtype=float)
        coords = np.asarray(self.cell._atom_coords, dtype=float)
        lattice = np.asarray(self.cell.lattice_vectors, dtype=float)
        mat = np.zeros((len(basis), len(basis)), dtype=np.complex128)
        for gvec, weight in self._reciprocal_g_weights(lattice):
            g2 = float(np.dot(gvec, gvec))
            if g2 <= 0.0:
                continue
            damping = np.exp(-g2 / (4.0 * self.eta * self.eta))
            rho_nuc = np.sum(charges * np.exp(-1j * coords @ gvec))
            pair_plus_g = self._periodic_pair_ft(basis, -gvec)
            mat += -(4.0 * np.pi) * weight * damping * rho_nuc * pair_plus_g / g2
        return _symmetrize(mat)

    def _periodic_reciprocal_eri(self, basis):
        lattice = np.asarray(self.cell.lattice_vectors, dtype=float)
        nao = len(basis)
        eri = np.zeros((nao, nao, nao, nao), dtype=np.complex128)
        for gvec, weight in self._reciprocal_g_weights(lattice):
            g2 = float(np.dot(gvec, gvec))
            if g2 <= 0.0:
                continue
            damping = np.exp(-g2 / (4.0 * self.eta * self.eta))
            pair_g = self._periodic_pair_ft(basis, gvec)
            pair_minus_g = self._periodic_pair_ft(basis, -gvec)
            eri += (
                (4.0 * np.pi)
                * weight
                * damping
                / g2
                * np.einsum("pq,rs->pqrs", pair_g, pair_minus_g, optimize=True)
            )
        eri = 0.5 * (eri + eri.transpose(1, 0, 3, 2).conj())
        return np.asarray(eri.real, dtype=float)

    def _reciprocal_g_weights(self, lattice):
        if self._use_inf_vacuum_1d():
            gvecs, weights = inf_vacuum_1d_gv_weights(lattice, self._reciprocal_mesh())
            mask = np.einsum("gi,gi->g", gvecs, gvecs) > 1e-16
            return zip(gvecs[mask], weights[mask])

        volume = abs(float(np.linalg.det(lattice)))
        return (
            (gvec, 1.0 / volume)
            for _h, _k, _l, gvec in reciprocal_vectors(lattice, self.recip_cut, include_zero=False)
        )

    def _madelung(self):
        energy = ewald_nuclear_repulsion_1d_inf_vacuum(
            np.asarray([1.0]),
            np.zeros((1, 3), dtype=float),
            self.cell.lattice_vectors,
            eta=self.eta,
            real_cut=self.real_cut,
            mesh=self._reciprocal_mesh(),
        )
        return -2.0 * energy

    def _solve_fock(self, fock):
        nelec = int(self.cell.nelectron)
        if nelec % 2 != 0:
            raise NotImplementedError("EwaldRHF currently supports only closed-shell even-electron cells.")
        nocc = nelec // 2
        evals, evecs = eigh(_symmetrize(fock), _symmetrize(self.overlap))
        occ = np.zeros_like(evals)
        occ[:nocc] = 2.0
        cocc = evecs[:, :nocc]
        dm = 2.0 * cocc @ cocc.conj().T
        return evals, evecs, occ, _symmetrize(dm)

    def get_veff(self, dm):
        dm = np.asarray(dm, dtype=np.complex128)
        vj = np.einsum("pqrs,rs->pq", self.eri, dm, optimize=True)
        vk = np.einsum("prqs,rs->pq", self.eri, dm, optimize=True)
        if self.madelung is not None:
            vk = vk + self.madelung * (self.overlap @ dm @ self.overlap)
        return _symmetrize(vj - 0.5 * vk)

    def get_fock(self, dm=None):
        if dm is None:
            if self.fock is None:
                return self.hcore
            return self.fock
        return _symmetrize(self.hcore + self.get_veff(dm))

    def _electronic_energy(self, dm, fock):
        return float(0.5 * np.trace(dm @ (self.hcore + fock)).real)

    def run(self, max_cycle=50, conv_tol=1e-8, conv_tol_dm=1e-6):
        self._build_integrals()
        mo_energy, mo_coeff, mo_occ, dm = self._solve_fock(self.hcore)

        e_last = None
        converged = False
        fock = self.hcore
        for cycle in range(int(max_cycle)):
            fock = self.get_fock(dm)
            mo_energy_new, mo_coeff_new, mo_occ_new, dm_new = self._solve_fock(fock)
            if cycle > 0 and self.damping > 0.0:
                dm_new = (1.0 - self.damping) * dm + self.damping * dm_new

            e_elec = self._electronic_energy(dm_new, fock)
            if e_last is not None:
                de = abs(e_elec - e_last)
                ddm = np.linalg.norm(dm_new - dm)
                if de < conv_tol and ddm < conv_tol_dm:
                    converged = True
                    dm = dm_new
                    mo_energy = mo_energy_new
                    mo_coeff = mo_coeff_new
                    mo_occ = mo_occ_new
                    break

            e_last = e_elec
            dm = dm_new
            mo_energy = mo_energy_new
            mo_coeff = mo_coeff_new
            mo_occ = mo_occ_new

        fock = self.get_fock(dm)
        e_elec = self._electronic_energy(dm, fock)
        self.e_elec = float(e_elec)
        self.e_tot = float(e_elec + self.e_nuc)
        self.mo_energy = mo_energy
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.dm = dm
        self.fock = fock
        self.converged = bool(converged)
        return self

    def kernel(self, **kwargs):
        return self.run(**kwargs).e_tot

    def get_hcore(self):
        return self.hcore

    def get_ovlp(self):
        return self.overlap

    def make_rdm1(self, mo_coeff=None, mo_occ=None):
        if mo_coeff is None:
            return self.dm
        nocc = int(np.count_nonzero(np.asarray(mo_occ) > 1e-12))
        cocc = mo_coeff[:, :nocc]
        return 2.0 * cocc @ cocc.conj().T
