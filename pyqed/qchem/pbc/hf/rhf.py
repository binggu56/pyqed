#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.linalg import eigh


def _normalize_kpts(kpts):
    if kpts is None:
        return np.zeros((1, 3), dtype=float)
    arr = np.asarray(kpts, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, 3)
    if arr.shape[-1] != 3:
        raise ValueError("kpts must have shape (nk, 3) or (3,).")
    return arr


def _symmetrize(mat):
    return 0.5 * (mat + mat.conj().T)


def _matrix_norm(mats_a, mats_b):
    total = 0.0
    for a, b in zip(mats_a, mats_b):
        total += np.linalg.norm(a - b)
    return total


class RHF:
    """
    Native 1D periodic RHF reference implementation.

    This uses image-summed molecular integrals from the builtin molecular stack.
    It is intended as a clean, dependency-free first milestone, not a production
    fast periodic HF engine.
    """

    def __init__(self, cell, kpts=None, nimages=1, damping=0.5):
        self.cell = cell
        self.kpts = _normalize_kpts(kpts)
        self.nimages = int(nimages)
        self.damping = float(damping)

        self.e_tot = None
        self.e_elec = None
        self.e_nuc = None
        self.mo_energy = None
        self.mo_coeff = None
        self.mo_occ = None
        self.dm = None
        self.fock = None
        self.converged = False
        self.nkpts = int(len(self.kpts))

        self._cluster_mol = None
        self._rels = None
        self._rel_to_idx = None
        self._rel_vectors = None
        self._s_r = None
        self._h_r = None
        self._eri_rtu = None

    def _build_real_space_data(self):
        if not self.cell.built:
            self.cell.build()
        if self.cell.dimension != 1:
            raise NotImplementedError("Native periodic RHF currently supports only dimension=1.")

        cluster = self.cell.build_image_molecule(self.nimages)
        self._cluster_mol = cluster

        nao0 = int(self.cell.nao)
        rels = list(range(-self.nimages, self.nimages + 1))
        center = self.nimages
        ao_slices = {
            rel: slice((center + rel) * nao0, (center + rel + 1) * nao0)
            for rel in rels
        }

        s_r = {}
        h_r = {}
        eri_rtu = {}

        csl = ao_slices[0]
        for r in rels:
            rsl = ao_slices[r]
            s_r[r] = np.asarray(cluster.overlap[csl, rsl], dtype=np.complex128)
            h_r[r] = np.asarray(cluster.hcore[csl, rsl], dtype=np.complex128)

        eri = np.asarray(cluster.eri, dtype=float)
        for r in rels:
            rsl = ao_slices[r]
            for t in rels:
                tsl = ao_slices[t]
                for u in rels:
                    usl = ao_slices[u]
                    eri_rtu[(r, t, u)] = np.asarray(
                        eri[csl, rsl, tsl, usl], dtype=np.complex128
                    )

        self._rels = rels
        self._rel_to_idx = {rel: idx for idx, rel in enumerate(rels)}
        self._rel_vectors = {
            rel: self.cell.translation_vector(rel)
            for rel in rels
        }
        self._s_r = s_r
        self._h_r = h_r
        self._eri_rtu = eri_rtu

    def _fourier_sum(self, mats_r):
        nk = len(self.kpts)
        out = []
        for ik in range(nk):
            acc = np.zeros_like(next(iter(mats_r.values())), dtype=np.complex128)
            kvec = self.kpts[ik]
            for rel in self._rels:
                phase = np.exp(1j * np.dot(kvec, self._rel_vectors[rel]))
                acc += phase * mats_r[rel]
            out.append(_symmetrize(acc))
        return out

    def _density_k_to_r(self, dm_k):
        nk = len(dm_k)
        dm_r = {}
        for rel in self._rels:
            acc = np.zeros_like(dm_k[0], dtype=np.complex128)
            for kvec, dmk in zip(self.kpts, dm_k):
                phase = np.exp(-1j * np.dot(kvec, self._rel_vectors[rel]))
                acc += phase * dmk
            dm_r[rel] = acc / nk
        return dm_r

    def _fock_r_from_density(self, dm_r):
        f_r = {}
        zero = np.zeros((self.cell.nao, self.cell.nao), dtype=np.complex128)
        for r in self._rels:
            vj = np.zeros_like(self._h_r[r], dtype=np.complex128)
            vk = np.zeros_like(self._h_r[r], dtype=np.complex128)
            for t in self._rels:
                for u in self._rels:
                    diff = u - t
                    dtu = dm_r.get(diff, zero)
                    g_coul = self._eri_rtu[(r, t, u)]
                    g_ex = self._eri_rtu[(t, r, u)]
                    vj += np.einsum('pqrs,rs->pq', g_coul, dtu, optimize=True)
                    vk += np.einsum('prqs,rs->pq', g_ex, dtu, optimize=True)
            f_r[r] = self._h_r[r] + vj - 0.5 * vk
        return f_r

    def _solve_fock(self, f_k, s_k):
        nelec = int(self.cell.nelectron)
        if nelec % 2 != 0:
            raise NotImplementedError("Native periodic RHF currently supports only closed-shell even-electron cells.")
        nocc = nelec // 2

        mo_energy = []
        mo_coeff = []
        mo_occ = []
        dm_k = []
        for fk, sk in zip(f_k, s_k):
            evals, evecs = eigh(_symmetrize(fk), _symmetrize(sk))
            occ = np.zeros_like(evals)
            occ[:nocc] = 2.0
            cocc = evecs[:, :nocc]
            dmk = 2.0 * cocc @ cocc.conj().T
            mo_energy.append(evals)
            mo_coeff.append(evecs)
            mo_occ.append(occ)
            dm_k.append(_symmetrize(dmk))
        return mo_energy, mo_coeff, mo_occ, dm_k

    def _electronic_energy(self, dm_k, h_k, f_k):
        e = 0.0
        nk = len(dm_k)
        for dmk, hk, fk in zip(dm_k, h_k, f_k):
            e += 0.5 * np.trace(dmk @ (hk + fk)).real
        return e / nk

    def run(self, max_cycle=50, conv_tol=1e-8, conv_tol_dm=1e-6):
        self._build_real_space_data()
        s_k = self._fourier_sum(self._s_r)
        h_k = self._fourier_sum(self._h_r)
        mo_energy, mo_coeff, mo_occ, dm_k = self._solve_fock(h_k, s_k)

        e_last = None
        converged = False
        f_k = h_k
        for cycle in range(int(max_cycle)):
            dm_r = self._density_k_to_r(dm_k)
            f_r = self._fock_r_from_density(dm_r)
            f_k = self._fourier_sum(f_r)
            mo_energy_new, mo_coeff_new, mo_occ_new, dm_k_new = self._solve_fock(f_k, s_k)

            if cycle > 0 and self.damping > 0.0:
                dm_k_new = [
                    (1.0 - self.damping) * d_old + self.damping * d_new
                    for d_old, d_new in zip(dm_k, dm_k_new)
                ]

            e_elec = self._electronic_energy(dm_k_new, h_k, f_k)
            if e_last is not None:
                de = abs(e_elec - e_last)
                ddm = _matrix_norm(dm_k_new, dm_k)
                if de < conv_tol and ddm < conv_tol_dm:
                    converged = True
                    dm_k = dm_k_new
                    mo_energy = mo_energy_new
                    mo_coeff = mo_coeff_new
                    mo_occ = mo_occ_new
                    break

            e_last = e_elec
            dm_k = dm_k_new
            mo_energy = mo_energy_new
            mo_coeff = mo_coeff_new
            mo_occ = mo_occ_new

        dm_r = self._density_k_to_r(dm_k)
        f_r = self._fock_r_from_density(dm_r)
        f_k = self._fourier_sum(f_r)
        e_elec = self._electronic_energy(dm_k, h_k, f_k)
        e_nuc = float(self.cell.nuclear_repulsion(self.nimages))

        self.e_elec = float(e_elec)
        self.e_nuc = float(e_nuc)
        self.e_tot = float(e_elec + e_nuc)
        self.mo_energy = mo_energy[0] if self.nkpts == 1 else mo_energy
        self.mo_coeff = mo_coeff[0] if self.nkpts == 1 else mo_coeff
        self.mo_occ = mo_occ[0] if self.nkpts == 1 else mo_occ
        self.dm = dm_k[0] if self.nkpts == 1 else dm_k
        self.fock = f_k[0] if self.nkpts == 1 else f_k
        self.converged = bool(converged or self.nkpts == 1)
        return self

    def kernel(self, **kwargs):
        return self.run(**kwargs).e_tot

    def get_hcore(self):
        h_k = self._fourier_sum(self._h_r)
        return h_k[0] if self.nkpts == 1 else h_k

    def get_ovlp(self):
        s_k = self._fourier_sum(self._s_r)
        return s_k[0] if self.nkpts == 1 else s_k

    def get_fock(self, dm=None):
        if dm is None:
            return self.fock
        dm_k = [dm] if self.nkpts == 1 else dm
        dm_r = self._density_k_to_r(dm_k)
        f_r = self._fock_r_from_density(dm_r)
        f_k = self._fourier_sum(f_r)
        return f_k[0] if self.nkpts == 1 else f_k

    def make_rdm1(self, mo_coeff=None, mo_occ=None):
        if mo_coeff is None:
            return self.dm

        if self.nkpts == 1:
            nocc = int(np.count_nonzero(np.asarray(mo_occ) > 1e-12))
            cocc = mo_coeff[:, :nocc]
            return 2.0 * cocc @ cocc.conj().T

        dm = []
        for coeff_k, occ_k in zip(mo_coeff, mo_occ):
            nocc = int(np.count_nonzero(np.asarray(occ_k) > 1e-12))
            cocc = coeff_k[:, :nocc]
            dm.append(2.0 * cocc @ cocc.conj().T)
        return dm
