#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Native AO-based unrestricted Hartree-Fock.
"""

import logging
import math
from functools import reduce

import numpy as np
from opt_einsum import contract
from scipy.linalg import eigh

from pyqed import dag, dagger


_AUFBAU_ORDER = (
    (1, 0, 2),
    (2, 0, 2),
    (2, 1, 6),
    (3, 0, 2),
    (3, 1, 6),
    (4, 0, 2),
    (3, 2, 10),
    (4, 1, 6),
    (5, 0, 2),
    (4, 2, 10),
    (5, 1, 6),
    (6, 0, 2),
    (4, 3, 14),
    (5, 2, 10),
    (6, 1, 6),
    (7, 0, 2),
)

_AUFBAU_EXCEPTIONS = {
    24: {(4, 0): 1, (3, 2): 5},
    29: {(4, 0): 1, (3, 2): 10},
}


def _split_nelec(nelec, spin):
    spin = int(round(spin))
    nelec = int(round(nelec))
    if (nelec + spin) % 2 != 0:
        raise ValueError(
            f"Incompatible electron count/spin: nelec={nelec}, spin={spin}."
        )

    nalpha = (nelec + spin) // 2
    nbeta = nelec - nalpha
    if nalpha < 0 or nbeta < 0:
        raise ValueError(
            f"Invalid unrestricted occupation: nalpha={nalpha}, nbeta={nbeta}."
        )
    return nalpha, nbeta


class UHF:
    def __init__(self, mol, init_guess='h1e'):
        self.mol = mol
        self.max_cycle = 100
        self.init_guess = init_guess
        self.conv_tol = 1e-8
        self.conv_tol_dm = 1e-6
        self.damping = 0.0
        self.mom = False

        self.na = None
        self.nb = None
        self.nocc = None
        self.nvir = None

        self.nao = self.nmo = mol.nao
        self.nso = 2 * mol.nao
        self.nelec = mol.nelec

        self.mo_occ = None
        self.mo_coeff = None
        self.e_tot = None
        self.e_nuc = None
        self.hcore = None
        self.vhf = None
        self.dm = None
        self.mo_energy = None
        self.converged = False

        self.eri = mol.eri

    def run(self, dm0=None, **kwargs):
        out = unrestricted_hartree_fock(
            self.mol,
            dm0=dm0,
            init_guess=kwargs.get('init_guess', self.init_guess),
            mo_occ0=kwargs.get('mo_occ0', None),
            mom=kwargs.get('mom', self.mom),
            max_cycle=kwargs.get('max_cycle', self.max_cycle),
            tol=kwargs.get('tol', kwargs.get('conv_tol', self.conv_tol)),
            d_tol=kwargs.get('d_tol', kwargs.get('conv_tol_dm', self.conv_tol_dm)),
            damping=kwargs.get('damping', self.damping),
            verbose=kwargs.get('verbose', 0),
        )

        self.e_tot = out['e_tot']
        self.e_nuc = out['e_nuc']
        self.mo_energy = out['mo_energy']
        self.mo_coeff = out['mo_coeff']
        self.mo_occ = out['mo_occ']
        self.hcore = out['hcore']
        self.vhf = out['vhf']
        self.dm = out['dm']
        self.converged = out['converged']
        self.na = out['na']
        self.nb = out['nb']
        self.nocc = (self.na, self.nb)
        self.nvir = (self.nao - self.na, self.nao - self.nb)
        return self

    def make_rdm1(self):
        if self.mo_coeff is None or self.mo_occ is None:
            raise ValueError("No converged orbitals are available yet.")
        return np.array((
            make_rdm1(self.mo_coeff[0], self.mo_occ[0]),
            make_rdm1(self.mo_coeff[1], self.mo_occ[1]),
        ))

    def cluster(
        self,
        method='spectral',
        n_clusters=None,
        max_size=4,
        weights='integral+rdm',
        orbitals='canonical',
        localization='pm',
        mo_coeff=None,
        dm=None,
        active=None,
        space=None,
        localize_kwargs=None,
        return_info=False,
        return_orbitals=False,
        **kwargs,
    ):
        """Cluster MOs for active-space/NARG workflows."""
        from pyqed.qchem.orbital_clustering import cluster_mf_orbitals

        return cluster_mf_orbitals(
            self,
            method=method,
            n_clusters=n_clusters,
            max_size=max_size,
            weights=weights,
            orbitals=orbitals,
            localization=localization,
            mo_coeff=mo_coeff,
            dm=dm,
            active=active,
            space=space,
            localize_kwargs=localize_kwargs,
            return_info=return_info,
            return_orbitals=return_orbitals,
            **kwargs,
        )

    def NARG(self, *args, **kwargs):
        """Build a qchem NARG solver from this mean-field reference."""
        from pyqed.narg import NARG

        return NARG(self, *args, **kwargs)

    def get_ovlp(self):
        return self.mol.overlap

    def get_hcore(self):
        hcore = self.mol.hcore
        return (hcore, hcore)

    def get_hcore_mo(self, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        hcore = self.mol.hcore
        return (
            dag(mo_coeff[0]) @ hcore @ mo_coeff[0],
            dag(mo_coeff[1]) @ hcore @ mo_coeff[1],
        )

    def get_veff(self, dm=None):
        if dm is None:
            if self.dm is None:
                raise ValueError("No density matrix is available yet.")
            dm = self.dm
        return get_veff(self.mol, dm)

    def get_fock(self, dm=None):
        if dm is None:
            if self.dm is None:
                raise ValueError("No density matrix is available yet.")
            dm = self.dm
        veff = self.get_veff(dm)
        hcore = self.mol.hcore
        return np.array((hcore + veff[0], hcore + veff[1]))

    def get_j(self, dm=None):
        if dm is None:
            if self.dm is None:
                raise ValueError("No density matrix is available yet.")
            dm = self.dm
        dm = np.asarray(dm)
        return get_j(self.mol, dm[0] + dm[1])

    def get_k(self, dm=None):
        if dm is None:
            if self.dm is None:
                raise ValueError("No density matrix is available yet.")
            dm = self.dm
        dm = np.asarray(dm)
        return np.array((get_k(self.mol, dm[0]), get_k(self.mol, dm[1])))

    def get_jk(self, dm=None):
        if dm is None:
            if self.dm is None:
                raise ValueError("No density matrix is available yet.")
            dm = self.dm
        dm = np.asarray(dm)
        return self.get_j(dm), self.get_k(dm)

    def energy_elec(self, dm=None):
        if dm is None:
            if self.dm is None:
                raise ValueError("No density matrix is available yet.")
            dm = self.dm
        vhf = self.get_veff(dm)
        return energy_elec(dm, self.mol.hcore, vhf)

    def energy_nuc(self):
        if self.e_nuc is not None:
            return self.e_nuc
        return self.mol.energy_nuc()

    def get_eri(self, representation='ao'):
        if representation == 'ao':
            return _dense_ao_eri(self.mol)
        if representation == 'mo':
            if self.mo_coeff is None:
                raise ValueError("Run UHF before requesting MO integrals.")
            return self.get_eri_mo()
        raise ValueError("representation must be 'ao' or 'mo'.")

    def get_eri_mo(self, mo_coeff=None, notation='chem'):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_coeff is None:
            raise ValueError("Run UHF before requesting MO integrals.")

        ca, cb = mo_coeff
        factors = getattr(self.mol, 'eri_factors', None)
        if factors is not None:
            from pyqed.qchem.basis import mo_pair_factors

            factors_aa = mo_pair_factors(factors, ca, ca)
            factors_bb = mo_pair_factors(factors, cb, cb)
            aa = contract('Pij,Pkl->ijkl', factors_aa, factors_aa)
            bb = contract('Pij,Pkl->ijkl', factors_bb, factors_bb)
            ab = contract('Pij,Pkl->ijkl', factors_aa, factors_bb)
            ba = contract('Pij,Pkl->ijkl', factors_bb, factors_aa)
        else:
            eri = _dense_ao_eri(self.mol)
            aa = contract('pqrs,pi,qj,rk,sl->ijkl', eri, ca.conj(), ca, ca.conj(), ca)
            bb = contract('pqrs,pi,qj,rk,sl->ijkl', eri, cb.conj(), cb, cb.conj(), cb)
            ab = contract('pqrs,pi,qj,rk,sl->ijkl', eri, ca.conj(), ca, cb.conj(), cb)
            ba = contract('pqrs,pi,qj,rk,sl->ijkl', eri, cb.conj(), cb, ca.conj(), ca)

        if notation == 'phys':
            aa = np.transpose(aa, (0, 2, 1, 3))
            bb = np.transpose(bb, (0, 2, 1, 3))
            ab = np.transpose(ab, (0, 2, 1, 3))
            ba = np.transpose(ba, (0, 2, 1, 3))
        elif notation != 'chem':
            raise ValueError("notation must be 'chem' or 'phys'.")

        return np.array(((aa, ab), (ba, bb)), dtype=complex)


def _dense_ao_eri(mol):
    eri = getattr(mol, 'eri', None)
    if eri is not None:
        return np.asarray(eri)
    eri_s4 = getattr(mol, 'eri_s4', None)
    if eri_s4 is not None:
        from pyqed.qchem.basis import unpack_eri_s4
        return unpack_eri_s4(eri_s4, mol.nao)
    eri_s8 = getattr(mol, 'eri_s8', None)
    if eri_s8 is not None:
        from pyqed.qchem.basis import unpack_eri_s8
        return unpack_eri_s8(eri_s8, mol.nao)
    factors = getattr(mol, 'eri_factors', None)
    if factors is not None:
        return contract('Ppq,Prs->pqrs', factors, factors)
    raise ValueError("No AO ERI representation is available.")


def get_j(mol, dm):
    from pyqed.qchem.hf.rhf import get_jk
    return get_jk(mol, dm)[0]


def get_k(mol, dm):
    from pyqed.qchem.hf.rhf import get_jk
    return get_jk(mol, dm)[1]


def get_veff(mol, dm, dm_last=None, vhf_last=None):
    dm = np.asarray(dm)
    if dm.shape[0] != 2:
        raise ValueError("UHF density matrix must have shape (2, nao, nao).")

    if dm_last is not None:
        ddm = dm - np.asarray(dm_last)
        dvhf = get_veff(mol, ddm)
        return dvhf + np.asarray(vhf_last)

    j_tot = get_j(mol, dm[0] + dm[1])
    k_a = get_k(mol, dm[0])
    k_b = get_k(mol, dm[1])
    return np.array((j_tot - k_a, j_tot - k_b))


def energy_elec(dm, h1e=None, vhf=None):
    dm = np.asarray(dm)
    if dm.shape[0] != 2:
        raise ValueError("UHF density matrix must have shape (2, nao, nao).")

    if h1e is None:
        raise ValueError("h1e is required.")
    if vhf is None:
        raise ValueError("vhf is required.")

    if np.asarray(h1e).ndim == 2:
        h1e = np.array((h1e, h1e))
    vhf = np.asarray(vhf)

    e1 = (
        np.einsum('ij,ji->', h1e[0], dm[0]).real
        + np.einsum('ij,ji->', h1e[1], dm[1]).real
    )
    e2 = 0.5 * (
        np.einsum('ij,ji->', vhf[0], dm[0]).real
        + np.einsum('ij,ji->', vhf[1], dm[1]).real
    )
    return e1 + e2


def make_rdm1(mo_coeff, mo_occ, **kwargs):
    mocc = mo_coeff[:, mo_occ > 0]
    return np.dot(mocc * mo_occ[mo_occ > 0], mocc.conj().T)


def _orthogonalizer(overlap):
    s, u = eigh(overlap)
    if np.min(s) <= 0.0:
        raise ValueError("Overlap matrix is not positive definite.")
    return u @ np.diagflat(s ** (-0.5)) @ dagger(u)


def _diagonalize(fock, orth):
    fock_orth = reduce(np.dot, (dagger(orth), fock, orth))
    mo_energy, c_orth = eigh(fock_orth)
    mo_coeff = np.real_if_close(orth @ c_orth)
    return mo_energy, mo_coeff


def _validate_occ_vector(name, occ, nmo, nelec):
    occ = np.asarray(occ, dtype=float)
    if occ.shape != (nmo,):
        raise ValueError(f"{name} must have shape ({nmo},), got {occ.shape}.")
    if not np.allclose(occ, np.round(occ), atol=1e-12):
        raise ValueError(f"{name} must contain integer occupations.")
    if np.any(occ < 0.0) or np.any(occ > 1.0):
        raise ValueError(f"{name} occupations must lie in [0, 1].")
    if int(round(np.sum(occ))) != int(nelec):
        raise ValueError(
            f"{name} must sum to {nelec}, got {np.sum(occ)!r}."
        )
    return occ


def _resolve_initial_occ(hcore, na, nb, mo_occ0=None):
    if mo_occ0 is None:
        mo_occ_a = np.zeros(hcore.shape[0])
        mo_occ_b = np.zeros(hcore.shape[0])
        mo_occ_a[:na] = 1.0
        mo_occ_b[:nb] = 1.0
        return mo_occ_a, mo_occ_b

    try:
        occ_a, occ_b = mo_occ0
    except (TypeError, ValueError):
        raise ValueError("mo_occ0 must be a 2-tuple of (occ_a, occ_b).") from None

    mo_occ_a = _validate_occ_vector('mo_occ0[0]', occ_a, hcore.shape[0], na)
    mo_occ_b = _validate_occ_vector('mo_occ0[1]', occ_b, hcore.shape[0], nb)
    return mo_occ_a, mo_occ_b


def _subshell_total_occupations(nelec):
    remaining = int(nelec)
    occ = {}
    for n, l, capacity in _AUFBAU_ORDER:
        if remaining <= 0:
            break
        fill = min(remaining, capacity)
        occ[(n, l)] = fill
        remaining -= fill
    if remaining != 0:
        raise ValueError(f"atom_config does not support nelec={nelec}.")
    if int(nelec) in _AUFBAU_EXCEPTIONS:
        for key, value in _AUFBAU_EXCEPTIONS[int(nelec)].items():
            occ[key] = value
    return occ


def _subshell_spin_occupations(total_occ):
    spin_occ = {}
    for (n, l), occ in total_occ.items():
        degeneracy = 2 * l + 1
        na = min(occ, degeneracy)
        nb = max(0, occ - degeneracy)
        spin_occ[(n, l)] = (na, nb)
    return spin_occ


def _adjust_spin_occupations(spin_occ, na_target, nb_target):
    na_current = sum(na for na, _ in spin_occ.values())
    nb_current = sum(nb for _, nb in spin_occ.values())
    delta_a = int(na_target - na_current)
    delta_b = int(nb_target - nb_current)
    if delta_a != -delta_b:
        raise ValueError("Inconsistent target spin occupations.")
    if delta_a == 0:
        return spin_occ

    # Work from outer shells inward when changing spin pairing.
    keys = sorted(spin_occ, reverse=True)
    updated = dict(spin_occ)
    if delta_a > 0:
        for key in keys:
            na, nb = updated[key]
            move = min(delta_a, nb)
            if move:
                updated[key] = (na + move, nb - move)
                delta_a -= move
            if delta_a == 0:
                break
    else:
        delta_a = -delta_a
        for key in keys:
            na, nb = updated[key]
            singly_occupied_alpha = max(0, na - nb)
            move = min(delta_a, singly_occupied_alpha)
            if move:
                updated[key] = (na - move, nb + move)
                delta_a -= move
            if delta_a == 0:
                break
    if delta_a != 0:
        raise ValueError("Unable to match requested spin with atom_config guess.")
    return updated


def _builtin_orientation_groups(mol):
    if getattr(mol, '_bas', None) is None:
        raise ValueError("Molecule basis is not built.")
    if getattr(mol, 'natom', 0) != 1:
        raise ValueError("init_guess='atom_config' currently supports single atoms only.")

    groups = {}
    for idx, bf in enumerate(mol._bas):
        l = int(sum(bf.shell))
        key = (
            l,
            tuple(int(x) for x in bf.shell),
        )
        if key not in groups:
            groups[key] = {
                'indices': [],
                'l': l,
            }
        groups[key]['indices'].append(idx)

    by_l = {}
    for info in groups.values():
        by_l.setdefault(info['l'], []).append(info)
    for l in by_l:
        by_l[l].sort(key=lambda item: item['indices'][0])
    return by_l


def _atom_config_density(mol, overlap, na, nb):
    by_l = _builtin_orientation_groups(mol)
    spin_occ = _subshell_spin_occupations(_subshell_total_occupations(mol.nelec))
    spin_occ = _adjust_spin_occupations(spin_occ, na, nb)

    subshells_by_l = {}
    for (n, l), occ in spin_occ.items():
        subshells_by_l.setdefault(l, []).append((n - l - 1, occ))
    for l in subshells_by_l:
        subshells_by_l[l].sort()

    def build_dm_from_fock(fock_a, fock_b):
        dm_a = np.zeros_like(overlap)
        dm_b = np.zeros_like(overlap)
        for l, shell_entries in subshells_by_l.items():
            orientation_groups = by_l.get(l, [])
            if not orientation_groups:
                raise ValueError(
                    f"Basis does not provide any l={l} functions for atom_config."
                )

            nradial = len(orientation_groups[0]['indices'])
            norient = len(orientation_groups)
            s_avg = np.zeros((nradial, nradial), dtype=float)
            f_avg_a = np.zeros((nradial, nradial), dtype=float)
            f_avg_b = np.zeros((nradial, nradial), dtype=float)
            for orient in orientation_groups:
                idx = orient['indices']
                block = np.ix_(idx, idx)
                s_avg += overlap[block]
                f_avg_a += fock_a[block]
                f_avg_b += fock_b[block]
            s_avg /= norient
            f_avg_a /= norient
            f_avg_b /= norient

            _, coeff_a = eigh(f_avg_a, s_avg)
            _, coeff_b = eigh(f_avg_b, s_avg)

            for shell_index, (nalpha_shell, nbeta_shell) in shell_entries:
                if shell_index < 0 or shell_index >= nradial:
                    raise ValueError(
                        f"Basis does not provide enough l={l} shells for atom_config."
                    )
                occ_a_per_orient = nalpha_shell / norient
                occ_b_per_orient = nbeta_shell / norient
                vec_a = coeff_a[:, shell_index:shell_index + 1]
                vec_b = coeff_b[:, shell_index:shell_index + 1]
                proj_a = vec_a @ vec_a.conj().T
                proj_b = vec_b @ vec_b.conj().T
                for orient in orientation_groups:
                    idx = orient['indices']
                    block = np.ix_(idx, idx)
                    if occ_a_per_orient:
                        dm_a[block] += occ_a_per_orient * proj_a
                    if occ_b_per_orient:
                        dm_b[block] += occ_b_per_orient * proj_b
        return np.array((dm_a, dm_b))

    dm = build_dm_from_fock(mol.hcore, mol.hcore)
    for _ in range(12):
        vhf = get_veff(mol, dm)
        fock_a = mol.hcore + vhf[0]
        fock_b = mol.hcore + vhf[1]
        dm_new = build_dm_from_fock(fock_a, fock_b)
        if np.linalg.norm(dm_new - dm) < 1e-10:
            dm = dm_new
            break
        dm = 0.7 * dm_new + 0.3 * dm
    return dm


def _initial_guess(mol, hcore, overlap, orth, na, nb, dm0=None, init_guess='h1e', mo_occ0=None):
    mo_occ_a, mo_occ_b = _resolve_initial_occ(hcore, na, nb, mo_occ0=mo_occ0)

    if dm0 is not None:
        dm = np.asarray(dm0, dtype=float)
        if dm.shape != (2, hcore.shape[0], hcore.shape[0]):
            raise ValueError("dm0 must have shape (2, nao, nao).")
        return dm, (mo_occ_a, mo_occ_b)

    if init_guess not in ('h1e', 'hcore', 'atom_config'):
        raise ValueError(
            "Only init_guess='h1e'/'hcore'/'atom_config' is currently supported."
        )

    if init_guess == 'atom_config':
        dm = _atom_config_density(mol=mol, overlap=overlap, na=na, nb=nb)
        return dm, (mo_occ_a, mo_occ_b)

    mo_energy, mo_coeff = _diagonalize(hcore, orth)
    dm_a = make_rdm1(mo_coeff, mo_occ_a)
    dm_b = make_rdm1(mo_coeff, mo_occ_b)
    return np.array((dm_a, dm_b)), (mo_occ_a, mo_occ_b)


def _occupied_subspace_from_density(dm_spin, overlap, nocc):
    occ_vals, coeff = eigh(dm_spin, overlap)
    order = np.argsort(occ_vals)[::-1]
    if nocc == 0:
        return coeff[:, :0]
    return coeff[:, order[:nocc]]


def _mom_select_occupations(mo_coeff, prev_occ_coeff, overlap, nocc):
    if nocc == 0:
        return np.zeros(mo_coeff.shape[1], dtype=float), mo_coeff[:, :0]
    proj = prev_occ_coeff.conj().T @ overlap @ mo_coeff
    scores = np.sum(np.abs(proj) ** 2, axis=0)
    occ_idx = np.argsort(scores)[::-1][:nocc]
    occ = np.zeros(mo_coeff.shape[1], dtype=float)
    occ[occ_idx] = 1.0
    return occ, mo_coeff[:, occ_idx]


def _reorder_fixed_occupations(mo_energy, mo_coeff, mo_occ, prev_occ_coeff, overlap):
    """Keep explicitly occupied orbitals on their requested column indices."""
    targets = np.flatnonzero(np.asarray(mo_occ) > 0)
    if targets.size == 0:
        return mo_energy, mo_coeff, mo_coeff[:, :0]
    projection = prev_occ_coeff.conj().T @ overlap @ mo_coeff
    scores = np.sum(np.abs(projection) ** 2, axis=0)
    selected = list(np.argsort(scores)[::-1][:targets.size])
    remaining = [idx for idx in range(mo_coeff.shape[1]) if idx not in selected]
    order = np.empty(mo_coeff.shape[1], dtype=int)
    order[targets] = selected
    order[np.flatnonzero(np.asarray(mo_occ) <= 0)] = remaining
    aligned_coeff = mo_coeff[:, order]
    aligned_energy = np.asarray(mo_energy)[order]
    return aligned_energy, aligned_coeff, aligned_coeff[:, targets]


def _make_diis_extrapolator(max_diis=6):
    error_a = np.zeros((max_diis, 1, 1))
    error_b = np.zeros((max_diis, 1, 1))
    fock_a = np.zeros((max_diis, 1, 1))
    fock_b = np.zeros((max_diis, 1, 1))

    def diis(fock_pair, dens_pair, overlap, orth, iteration):
        fa, fb = fock_pair
        da, db = dens_pair

        if iteration <= 1:
            return np.array((fa, fb)), 0.0

        nonlocal error_a, error_b, fock_a, fock_b
        if error_a.shape[1:] != fa.shape:
            error_a = np.zeros((max_diis,) + fa.shape)
            error_b = np.zeros((max_diis,) + fb.shape)
            fock_a = np.zeros((max_diis,) + fa.shape)
            fock_b = np.zeros((max_diis,) + fb.shape)

        for k in reversed(range(1, min(iteration, max_diis))):
            error_a[k] = error_a[k - 1]
            error_b[k] = error_b[k - 1]
            fock_a[k] = fock_a[k - 1]
            fock_b[k] = fock_b[k - 1]

        ea = fa @ da @ overlap - overlap @ da @ fa
        eb = fb @ db @ overlap - overlap @ db @ fb
        error_a[0] = orth.T @ ea @ orth
        error_b[0] = orth.T @ eb @ orth
        fock_a[0] = fa
        fock_b[0] = fb

        diis_error = max(
            np.max(np.abs(error_a[0])),
            np.max(np.abs(error_b[0])),
        )

        bsize = min(iteration, max_diis) - 1
        bmat = -1.0 * np.ones((bsize + 1, bsize + 1))
        rhs = np.zeros(bsize + 1)
        bmat[bsize, bsize] = 0.0
        rhs[bsize] = -1.0
        for i in range(bsize):
            for j in range(bsize):
                bmat[i, j] = (
                    np.trace(error_a[i] @ error_a[j])
                    + np.trace(error_b[i] @ error_b[j])
                )

        try:
            coeff = np.linalg.solve(bmat, rhs)
        except np.linalg.LinAlgError:
            return np.array((fa, fb)), diis_error

        fa_diis = np.zeros_like(fa)
        fb_diis = np.zeros_like(fb)
        for c, fa_hist, fb_hist in zip(coeff[:-1], fock_a, fock_b):
            fa_diis += c * fa_hist
            fb_diis += c * fb_hist

        return np.array((fa_diis, fb_diis)), diis_error

    return diis


def unrestricted_hartree_fock(
    mol,
    dm0=None,
    init_guess='h1e',
    mo_occ0=None,
    mom=False,
    max_cycle=50,
    tol=1e-8,
    d_tol=1e-6,
    damping=0.0,
    verbose=0,
):
    overlap = mol.overlap
    hcore = mol.hcore
    e_nuc = mol.energy_nuc()
    orth = _orthogonalizer(overlap)
    na, nb = _split_nelec(mol.nelec, getattr(mol, 'spin', 0))

    dm, mo_occ = _initial_guess(
        mol,
        hcore,
        overlap,
        orth,
        na,
        nb,
        dm0=dm0,
        init_guess=init_guess,
        mo_occ0=mo_occ0,
    )
    prev_occ_coeff = (
        _occupied_subspace_from_density(dm[0], overlap, na),
        _occupied_subspace_from_density(dm[1], overlap, nb),
    )
    track_fixed_occupations = mo_occ0 is not None and not mom
    diis = _make_diis_extrapolator()

    e_last = None
    converged = False

    if verbose:
        logging.info("\n {:4s} {:16s} {:12s} {:12s}".format(
            'iter', 'total energy', 'de', 'ddm'))

    for scf_iter in range(max_cycle):
        vhf = get_veff(mol, dm)
        fock = np.array((hcore + vhf[0], hcore + vhf[1]))
        fock, diis_error = diis(fock, dm, overlap, orth, scf_iter)

        if damping:
            fock = damping * fock + (1.0 - damping) * np.array((hcore + vhf[0], hcore + vhf[1]))

        mo_energy_a, mo_coeff_a = _diagonalize(fock[0], orth)
        mo_energy_b, mo_coeff_b = _diagonalize(fock[1], orth)

        if mom:
            mo_occ_a, occ_coeff_a = _mom_select_occupations(
                mo_coeff_a, prev_occ_coeff[0], overlap, na
            )
            mo_occ_b, occ_coeff_b = _mom_select_occupations(
                mo_coeff_b, prev_occ_coeff[1], overlap, nb
            )
            mo_occ = (mo_occ_a, mo_occ_b)
            prev_occ_coeff = (occ_coeff_a, occ_coeff_b)
        elif track_fixed_occupations:
            mo_energy_a, mo_coeff_a, occ_coeff_a = _reorder_fixed_occupations(
                mo_energy_a, mo_coeff_a, mo_occ[0], prev_occ_coeff[0], overlap
            )
            mo_energy_b, mo_coeff_b, occ_coeff_b = _reorder_fixed_occupations(
                mo_energy_b, mo_coeff_b, mo_occ[1], prev_occ_coeff[1], overlap
            )
            prev_occ_coeff = (occ_coeff_a, occ_coeff_b)
        else:
            prev_occ_coeff = (
                mo_coeff_a[:, mo_occ[0] > 0],
                mo_coeff_b[:, mo_occ[1] > 0],
            )

        dm_new = np.array((
            make_rdm1(mo_coeff_a, mo_occ[0]),
            make_rdm1(mo_coeff_b, mo_occ[1]),
        ))

        vhf_new = get_veff(mol, dm_new)
        e_elec = energy_elec(dm_new, hcore, vhf_new)
        e_tot = e_elec + e_nuc
        de = None if e_last is None else e_tot - e_last
        ddm = np.linalg.norm(dm_new - dm)

        if verbose:
            logging.info("{:3d} {:16.10f} {:12.4e} {:12.4e}".format(
                scf_iter,
                e_tot,
                0.0 if de is None else de,
                ddm,
            ))

        dm = dm_new
        vhf = vhf_new

        if e_last is not None and abs(de) < tol and ddm < d_tol:
            converged = True
            break

        e_last = e_tot

        if diis_error < max(math.sqrt(tol), 1e-10):
            damping = 0.0

    return {
        'converged': converged,
        'e_tot': e_tot,
        'e_nuc': e_nuc,
        'mo_energy': (mo_energy_a, mo_energy_b),
        'mo_coeff': (mo_coeff_a, mo_coeff_b),
        'mo_occ': mo_occ,
        'hcore': (hcore, hcore),
        'vhf': vhf,
        'dm': dm,
        'na': na,
        'nb': nb,
    }
