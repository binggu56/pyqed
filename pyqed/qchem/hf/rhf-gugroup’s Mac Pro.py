#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 17:05:29 2022

HF with matrix elements extracted from PySCF

@author: Bing Gu (gubing@westlake.edu.cn)
"""


import logging
import numpy as np
from scipy.linalg import eigh
# from pyscf.scf import _vhf
# from pyscf import ao2mo
import sys
import re

from pyqed import dagger, dag
from opt_einsum import contract

from functools import reduce
import math


_AO_LABEL_RE = re.compile(r"^(?P<n>\d+)(?P<l>[a-z])(?P<comp>.*)$", re.IGNORECASE)


def _parse_ao_label(label):
    parts = str(label).split()
    if len(parts) < 3:
        raise ValueError(f"Unable to parse AO label '{label}'.")
    atom_index = int(parts[0])
    symbol = parts[1]
    orbital = parts[2]
    match = _AO_LABEL_RE.match(orbital)
    if match is None:
        raise ValueError(f"Unable to parse shell information from AO label '{label}'.")
    shell = f"{match.group('n')}{match.group('l')}"
    component = match.group('comp')
    return {
        'atom_index': atom_index,
        'symbol': symbol,
        'shell': shell,
        'component': component,
        'label': str(label),
    }


def _group_ao_indices_by_atom(ao_labels, natom):
    groups = [[] for _ in range(natom)]
    for ao_idx, label in enumerate(ao_labels):
        atom_idx = _parse_ao_label(label)['atom_index']
        groups[atom_idx].append(ao_idx)
    return [np.asarray(group, dtype=int) for group in groups]


def _cross_ao_overlap_matrix(mol_bra, mol_ket):
    basis_bra = mol_bra._bas_cart if getattr(mol_bra, "_bas_cart", None) is not None else mol_bra._bas
    basis_ket = mol_ket._bas_cart if getattr(mol_ket, "_bas_cart", None) is not None else mol_ket._bas
    if basis_bra is None or basis_ket is None:
        raise ValueError("Build both molecules before requesting cross AO overlaps.")

    try:
        from pyqed.qchem.basis import S as native_overlap

        s12 = np.empty((len(basis_bra), len(basis_ket)), dtype=float)
        for i, bra_ao in enumerate(basis_bra):
            for j, ket_ao in enumerate(basis_ket):
                s12[i, j] = float(native_overlap(bra_ao, ket_ao))
    except Exception:
        s12 = None

    if s12 is None:
        try:
            from gbasis.integrals.overlap_asymm import overlap_integral_asymmetric

            s12 = np.asarray(overlap_integral_asymmetric(basis_bra, basis_ket), dtype=float)
        except Exception as err:
            raise ValueError(
                "Cross-AO overlap requires PyQED builtin AO objects or gbasis shells."
            ) from err

    transform_bra = getattr(mol_bra, "_ao_cart2sph", None)
    transform_ket = getattr(mol_ket, "_ao_cart2sph", None)
    if transform_bra is not None:
        s12 = np.einsum('pi,pq->iq', transform_bra, s12, optimize=True)
    if transform_ket is not None:
        s12 = np.einsum('pq,qj->pj', s12, transform_ket, optimize=True)
    return s12


def _lowdin_sqrt_overlap(overlap, thresh=1e-12):
    eigvals, eigvecs = eigh(np.asarray(overlap, dtype=float))
    if np.any(eigvals < -thresh):
        raise ValueError("AO overlap matrix is not positive semidefinite.")
    eigvals = np.clip(eigvals, 0.0, None)
    return (eigvecs * np.sqrt(eigvals)) @ eigvecs.T


def _as_xyz_operator(op, name):
    op = np.asarray(op, dtype=float)
    if op.ndim != 3:
        raise ValueError(f"{name} must be a rank-3 array.")
    if op.shape[0] == 3:
        return op
    if op.shape[-1] == 3:
        return np.moveaxis(op, -1, 0)
    raise ValueError(f"{name} expects shape (3, nao, nao) or (nao, nao, 3).")


def _boys_objective_from_position(r_mo):
    centers = np.diagonal(r_mo, axis1=1, axis2=2).T
    return float(np.sum(centers * centers))


def _boys_localize(
    mo_coeff,
    r_ao,
    max_cycle=100,
    conv_tol=1e-10,
    rotation_tol=1e-12,
):
    coeff = np.array(mo_coeff, dtype=float, copy=True)
    if coeff.ndim != 2:
        raise ValueError("mo_coeff must be a 2D array.")

    nloc = coeff.shape[1]
    info = {
        'method': 'boys',
        'converged': True,
        'ncycle': 0,
        'initial_objective': 0.0,
        'final_objective': 0.0,
    }
    if nloc <= 1:
        r_mo = contract('xij,ip,jq->xpq', r_ao, coeff, coeff, optimize=True)
        obj = _boys_objective_from_position(r_mo)
        info['initial_objective'] = obj
        info['final_objective'] = obj
        return coeff, info

    r_mo = contract('xij,ip,jq->xpq', r_ao, coeff, coeff, optimize=True)
    objective = _boys_objective_from_position(r_mo)
    info['initial_objective'] = objective

    for cycle in range(1, int(max_cycle) + 1):
        start_objective = objective
        max_rotation = 0.0

        for p in range(nloc - 1):
            for q in range(p + 1, nloc):
                center_diff = 0.5 * (r_mo[:, p, p] - r_mo[:, q, q])
                coupling = r_mo[:, p, q]
                a2 = float(np.dot(center_diff, center_diff))
                b2 = float(np.dot(coupling, coupling))
                ab = float(np.dot(center_diff, coupling))

                theta = 0.25 * math.atan2(2.0 * ab, a2 - b2)
                candidates = (theta, theta + 0.25 * math.pi)

                best_theta = 0.0
                best_value = a2
                for trial in candidates:
                    phi = 2.0 * trial
                    rotated = center_diff * math.cos(phi) + coupling * math.sin(phi)
                    value = float(np.dot(rotated, rotated))
                    if value > best_value:
                        best_value = value
                        best_theta = trial

                if abs(best_theta) < rotation_tol or best_value - a2 < conv_tol * 0.1:
                    continue

                c = math.cos(best_theta)
                s = math.sin(best_theta)

                cp = coeff[:, p].copy()
                cq = coeff[:, q].copy()
                coeff[:, p] = c * cp + s * cq
                coeff[:, q] = -s * cp + c * cq

                rp = r_mo[:, :, p].copy()
                rq = r_mo[:, :, q].copy()
                r_mo[:, :, p] = c * rp + s * rq
                r_mo[:, :, q] = -s * rp + c * rq

                rp = r_mo[:, p, :].copy()
                rq = r_mo[:, q, :].copy()
                r_mo[:, p, :] = c * rp + s * rq
                r_mo[:, q, :] = -s * rp + c * rq

                objective = _boys_objective_from_position(r_mo)
                max_rotation = max(max_rotation, abs(best_theta))

        info['ncycle'] = cycle
        if objective - start_objective < conv_tol and max_rotation < math.sqrt(rotation_tol):
            break
    else:
        info['converged'] = False

    info['final_objective'] = objective
    return coeff, info


def _population_objective(y_coeff, atom_groups):
    objective = 0.0
    for group in atom_groups:
        populations = np.sum(y_coeff[group, :] ** 2, axis=0)
        objective += float(np.dot(populations, populations))
    return objective


def _mulliken_population_objective(coeff, scoeff, atom_groups):
    objective = 0.0
    for group in atom_groups:
        populations = np.sum(coeff[group, :] * scoeff[group, :], axis=0)
        objective += float(np.dot(populations, populations))
    return objective


def _pipek_mezey_localize(
    mo_coeff,
    overlap,
    atom_groups,
    method='pm',
    max_cycle=100,
    conv_tol=1e-10,
    rotation_tol=1e-12,
):
    coeff = np.array(mo_coeff, dtype=float, copy=True)
    if coeff.ndim != 2:
        raise ValueError("mo_coeff must be a 2D array.")

    nloc = coeff.shape[1]
    overlap = np.asarray(overlap, dtype=float)
    scoeff = overlap @ coeff
    objective = _mulliken_population_objective(coeff, scoeff, atom_groups)
    info = {
        'method': str(method),
        'backend': 'native',
        'population_metric': 'mulliken',
        'converged': True,
        'ncycle': 0,
        'initial_objective': objective,
        'final_objective': objective,
    }
    if nloc <= 1:
        return coeff, info

    for cycle in range(1, int(max_cycle) + 1):
        start_objective = objective
        max_rotation = 0.0

        for p in range(nloc - 1):
            for q in range(p + 1, nloc):
                diff = np.empty(len(atom_groups), dtype=float)
                coupling = np.empty(len(atom_groups), dtype=float)
                for atom_idx, group in enumerate(atom_groups):
                    cp = coeff[group, p]
                    cq = coeff[group, q]
                    sp = scoeff[group, p]
                    sq = scoeff[group, q]
                    pop_p = float(np.dot(cp, sp))
                    pop_q = float(np.dot(cq, sq))
                    cross = 0.5 * (float(np.dot(cp, sq)) + float(np.dot(cq, sp)))
                    diff[atom_idx] = 0.5 * (pop_p - pop_q)
                    coupling[atom_idx] = cross

                a2 = float(np.dot(diff, diff))
                b2 = float(np.dot(coupling, coupling))
                ab = float(np.dot(diff, coupling))

                theta = 0.25 * math.atan2(2.0 * ab, a2 - b2)
                candidates = (theta, theta + 0.25 * math.pi)

                best_theta = 0.0
                best_value = a2
                for trial in candidates:
                    phi = 2.0 * trial
                    rotated = diff * math.cos(phi) + coupling * math.sin(phi)
                    value = float(np.dot(rotated, rotated))
                    if value > best_value:
                        best_value = value
                        best_theta = trial

                if abs(best_theta) < rotation_tol or best_value - a2 < conv_tol * 0.1:
                    continue

                c = math.cos(best_theta)
                s = math.sin(best_theta)

                cp = coeff[:, p].copy()
                cq = coeff[:, q].copy()
                coeff[:, p] = c * cp + s * cq
                coeff[:, q] = -s * cp + c * cq

                sp = scoeff[:, p].copy()
                sq = scoeff[:, q].copy()
                scoeff[:, p] = c * sp + s * sq
                scoeff[:, q] = -s * sp + c * sq

                objective = _mulliken_population_objective(coeff, scoeff, atom_groups)
                max_rotation = max(max_rotation, abs(best_theta))

        info['ncycle'] = cycle
        if objective - start_objective < conv_tol and max_rotation < math.sqrt(rotation_tol):
            break
    else:
        info['converged'] = False

    info['final_objective'] = objective
    return coeff, info


def _ibo_localize(
    mo_coeff,
    overlap,
    atom_groups,
    method='ibo',
    max_cycle=100,
    conv_tol=1e-10,
    rotation_tol=1e-12,
):
    coeff = np.array(mo_coeff, dtype=float, copy=True)
    if coeff.ndim != 2:
        raise ValueError("mo_coeff must be a 2D array.")

    nloc = coeff.shape[1]
    sqrt_overlap = _lowdin_sqrt_overlap(overlap)
    y_coeff = sqrt_overlap @ coeff
    objective = _population_objective(y_coeff, atom_groups)
    info = {
        'method': str(method),
        'backend': 'native',
        'population_metric': 'lowdin',
        'converged': True,
        'ncycle': 0,
        'initial_objective': objective,
        'final_objective': objective,
    }
    if nloc <= 1:
        return coeff, info

    for cycle in range(1, int(max_cycle) + 1):
        start_objective = objective
        max_rotation = 0.0

        for p in range(nloc - 1):
            for q in range(p + 1, nloc):
                diff = np.empty(len(atom_groups), dtype=float)
                coupling = np.empty(len(atom_groups), dtype=float)
                for atom_idx, group in enumerate(atom_groups):
                    yp = y_coeff[group, p]
                    yq = y_coeff[group, q]
                    diff[atom_idx] = 0.5 * (float(np.dot(yp, yp)) - float(np.dot(yq, yq)))
                    coupling[atom_idx] = float(np.dot(yp, yq))

                a2 = float(np.dot(diff, diff))
                b2 = float(np.dot(coupling, coupling))
                ab = float(np.dot(diff, coupling))

                theta = 0.25 * math.atan2(2.0 * ab, a2 - b2)
                candidates = (theta, theta + 0.25 * math.pi)

                best_theta = 0.0
                best_value = a2
                for trial in candidates:
                    phi = 2.0 * trial
                    rotated = diff * math.cos(phi) + coupling * math.sin(phi)
                    value = float(np.dot(rotated, rotated))
                    if value > best_value:
                        best_value = value
                        best_theta = trial

                if abs(best_theta) < rotation_tol or best_value - a2 < conv_tol * 0.1:
                    continue

                c = math.cos(best_theta)
                s = math.sin(best_theta)

                cp = coeff[:, p].copy()
                cq = coeff[:, q].copy()
                coeff[:, p] = c * cp + s * cq
                coeff[:, q] = -s * cp + c * cq

                yp = y_coeff[:, p].copy()
                yq = y_coeff[:, q].copy()
                y_coeff[:, p] = c * yp + s * yq
                y_coeff[:, q] = -s * yp + c * yq

                objective = _population_objective(y_coeff, atom_groups)
                max_rotation = max(max_rotation, abs(best_theta))

        info['ncycle'] = cycle
        if objective - start_objective < conv_tol and max_rotation < math.sqrt(rotation_tol):
            break
    else:
        info['converged'] = False

    info['final_objective'] = objective
    return coeff, info


class RHF:
    def __init__(self, mol, init_guess='h1e', verbose=0):
        self.mol = mol
        self.max_cycle = 100
        self.init_guess = init_guess
        self.verbose = int(verbose)

        self.nocc = self.mol.nelec//2

        self.nao = self.nmo = mol.nao # number of MOs
        self.nvir = self.nmo - self.nocc

        self.nso = None
        self.nelec = mol.nelec

        self.mo_occ = None
        self.mo_coeff = None
        self.e_tot = None
        self.e_nuc = None
        self.e_kin = None
        self.e_ne = None
        self.e_j = None
        self.e_k = None




        ####
        self.hcore = None
        self.vhf = None
        self.dm = None

        self.eri = mol.eri
        self.eri_s4 = getattr(mol, 'eri_s4', None)
        self.eri_s8 = getattr(mol, 'eri_s8', None)
        # self.eri_so = None

        self.mo_energy = None
        self._pyscf_mf = None
        self.density_fit = False
        self.auxbasis = None
        self.cholesky_jk = False
        self.cholesky_tol = None
        self.cholesky_max_rank = None
        self.low_rank_jk = False
        self.low_rank_tol = None
        self.low_rank_max_rank = None
        self.eri_factors = None

    def run(self, **kwargs):
        verbose = int(kwargs.pop('verbose', self.verbose))
        self.verbose = verbose
        density_fit = bool(kwargs.pop('density_fit', False))
        auxbasis = kwargs.pop('auxbasis', None)
        cholesky_jk_kw = kwargs.pop('cholesky_jk', None)
        low_rank_jk_kw = kwargs.pop('low_rank_jk', None)
        if cholesky_jk_kw is not None and low_rank_jk_kw is not None \
                and bool(cholesky_jk_kw) != bool(low_rank_jk_kw):
            raise ValueError("cholesky_jk and low_rank_jk aliases disagree.")

        cholesky_tol_kw = kwargs.pop('cholesky_tol', None)
        low_rank_tol_kw = kwargs.pop('low_rank_tol', None)
        if cholesky_tol_kw is not None and low_rank_tol_kw is not None \
                and float(cholesky_tol_kw) != float(low_rank_tol_kw):
            raise ValueError("cholesky_tol and low_rank_tol aliases disagree.")

        cholesky_max_rank_kw = kwargs.pop('cholesky_max_rank', None)
        low_rank_max_rank_kw = kwargs.pop('low_rank_max_rank', None)
        if cholesky_max_rank_kw is not None and low_rank_max_rank_kw is not None \
                and cholesky_max_rank_kw != low_rank_max_rank_kw:
            raise ValueError("cholesky_max_rank and low_rank_max_rank aliases disagree.")

        cholesky_jk = bool(
            False if cholesky_jk_kw is None and low_rank_jk_kw is None
            else (cholesky_jk_kw if cholesky_jk_kw is not None else low_rank_jk_kw)
        )
        cholesky_tol = (
            1e-8 if cholesky_tol_kw is None and low_rank_tol_kw is None
            else (cholesky_tol_kw if cholesky_tol_kw is not None else low_rank_tol_kw)
        )
        cholesky_max_rank = (
            None if cholesky_max_rank_kw is None and low_rank_max_rank_kw is None
            else (cholesky_max_rank_kw if cholesky_max_rank_kw is not None else low_rank_max_rank_kw)
        )

        if (
            cholesky_jk_kw is None
            and low_rank_jk_kw is None
            and getattr(self.mol, 'eri_factors', None) is not None
        ):
            cholesky_jk = True
            cholesky_tol = getattr(
                self.mol, 'builtin_low_rank_tol',
                getattr(self.mol, 'native_low_rank_tol', cholesky_tol),
            )
            cholesky_max_rank = getattr(
                self.mol, 'builtin_low_rank_max_rank',
                getattr(self.mol, 'native_low_rank_max_rank', cholesky_max_rank),
            )

        if density_fit and cholesky_jk:
            raise ValueError("density_fit and cholesky_jk are mutually exclusive RHF accelerators.")

        if density_fit:
            self.e_tot, self.e_nuc, self.mo_energy, self.mo_coeff, self.mo_occ, self.hcore, \
                self.vhf, self.dm, self._pyscf_mf = pyscf_density_fit_rhf(
                    self.mol,
                    dm0=kwargs.pop('dm0', None),
                    init_guess=kwargs.pop('init_guess', self.init_guess),
                    max_cycle=kwargs.pop('max_cycle', 50),
                    tol=kwargs.pop('tol', 1e-8),
                    auxbasis=auxbasis,
                    verbose=verbose,
                )
            self.density_fit = True
            self.auxbasis = auxbasis
            self.cholesky_jk = False
            self.cholesky_tol = None
            self.cholesky_max_rank = None
            self.low_rank_jk = False
            self.low_rank_tol = None
            self.low_rank_max_rank = None
            self.eri_factors = None
        else:
            self.e_tot, self.e_nuc, self.mo_energy, self.mo_coeff, self.mo_occ, self.hcore, \
                self.vhf, self.dm = hartree_fock(
                    self.mol,
                    low_rank_jk=cholesky_jk,
                    low_rank_tol=cholesky_tol,
                    low_rank_max_rank=cholesky_max_rank,
                    verbose=verbose,
                    **kwargs,
                )
            self._pyscf_mf = None
            self.density_fit = False
            self.auxbasis = None
            self.cholesky_jk = cholesky_jk
            self.cholesky_tol = cholesky_tol if cholesky_jk else None
            self.cholesky_max_rank = cholesky_max_rank if cholesky_jk else None
            self.low_rank_jk = self.cholesky_jk
            self.low_rank_tol = self.cholesky_tol
            self.low_rank_max_rank = self.cholesky_max_rank
            self.eri_factors = (
                get_or_build_low_rank_eri_factors(
                    self.mol,
                    tol=cholesky_tol,
                    max_rank=cholesky_max_rank,
                )
                if cholesky_jk else None
            )
        return self

    def as_scanner(self, build_driver=None):
        """
        Return a lightweight scanner that reuses the previous density matrix and,
        when enabled, low-rank ERI factors across nearby geometries.
        """
        return RHFScanner(self, build_driver=build_driver)
    

    def get_eri(self, representation='mo'):
        """
        electron repulsion integral in MO

        Returns
        -------
        eri : TYPE
            DESCRIPTION.

        """

        # nmo = len(self.mo_energy)

        # eri = ao2mo.general(self.mol, (C,C,C,C),
        #               compact=False).reshape(nmo,nmo,nmo,nmo, order='C')

        if representation == 'ao':
            return self.eri 
        
        elif representation == 'mo':
            
            C = self.mo_coeff            
            eri = contract('abcd, ap, br, cs, dq -> pqrs', self.eri, C.conj(), C, C.conj(), C)
                                    
            return eri

    def dipole(self, center=None, basis='ao'):
        """
        Electric-dipole operator integrals from the current RHF reference.

        Parameters
        ----------
        center : array_like, optional
            Dipole origin. Defaults to the nuclear center of mass.
        basis : {'ao', 'mo'}, optional
            Basis for the returned operator blocks.

        Returns
        -------
        np.ndarray
            Shape ``(3, nao, nao)`` for ``basis='ao'`` or ``(3, nmo, nmo)`` for
            ``basis='mo'``. The returned operator is the electronic dipole
            operator ``mu = -r``, not the position operator ``r``.
        """
        if center is None:
            center = self.mol.center_of_mass()
        op = -np.asarray(self.mol.moment_integral(center=center), dtype=float)
        if op.ndim != 3:
            raise ValueError("moment_integral() must return a rank-3 array.")
        if op.shape[0] != 3:
            if op.shape[-1] == 3:
                op = np.moveaxis(op, -1, 0)
            else:
                raise ValueError("dipole() expects shape (3, nao, nao) or (nao, nao, 3).")

        key = basis.lower()
        if key == 'ao':
            return op
        if key == 'mo':
            if self.mo_coeff is None:
                raise ValueError("Run RHF before requesting dipole operators in the MO basis.")
            coeff = np.asarray(self.mo_coeff)
            return contract('xij,ip,jq->xpq', op, coeff.conj(), coeff, optimize=True)
        raise ValueError("basis must be 'ao' or 'mo'.")

    def get_eri_so(self):
        """
        get electron repulsion integral in spin orbitals

        Integrals are in "chemist's notation"
        .. math:: 
            
            eri[i,j,k,l] = (ij|kl) = \int i(1) j(1) 1/r12 k(r2) l(r2)

        Returns
        -------
        eri : TYPE
            DESCRIPTION.

        """
        # RHF, convert to spin-orbitals
        nso = 2 * len(self.mo_energy)
        self.nso = nso
        # self.e_mf = np.zeros(nso)
        # self.e_mf[0::2] = self.e_mf[1::2] = self.mo_energy

        # b = np.zeros((nso//2,nso))
        # b[:,0::2] = b[:,1::2] = self.mo_coeff

        # self.v_mf = 0.5 * reduce(np.dot, (b.T, v_mf, b))
        # self.v_mf[::2,1::2] = self.v_mf[1::2,::2] = 0

        # # electron repulsion integral
        # eri = ao2mofn(mf.mol, (b,b,b,b),
        #               compact=False).reshape(nso,nso,nso,nso)
        I = np.eye(2)
        eri = contract('ijkl, ab, cd -> iajbkcld', self.eri, I, I).reshape(nso, nso, nso, nso)

        # eri[::2,1::2] = eri[1::2,::2] = eri[:,:,::2,1::2] = eri[:,:,1::2,::2] = 0

        # print("Imag part of ERIs =", np.linalg.norm(eri.imag))
        return eri

    def get_soc_pvxp_ao(self, one_center=True):
        """
        Raw three-component p V x p operator in the AO basis.
        """
        from pyqed.qchem.soc import get_pvxp_ao
        return get_pvxp_ao(self.mol, one_center=one_center)

    def get_soc_1e_ao(self, one_center=True, with_prefactor=True, light_speed=None):
        """
        One-electron Breit-Pauli SOC operator in the AO basis.
        """
        from pyqed.qchem.soc import get_soc_1e_ao
        return get_soc_1e_ao(
            self.mol,
            one_center=one_center,
            with_prefactor=with_prefactor,
            light_speed=light_speed,
        )

    def get_soc_1e_mo(self, mo_coeff=None, one_center=True, with_prefactor=True,
                      light_speed=None):
        """
        One-electron Breit-Pauli SOC operator in the MO basis.
        """
        from pyqed.qchem.soc import get_soc_1e_mo
        return get_soc_1e_mo(
            self,
            mo_coeff=mo_coeff,
            one_center=one_center,
            with_prefactor=with_prefactor,
            light_speed=light_speed,
        )

    def get_soc_1e_so(self, representation='mo', mo_coeff=None, one_center=True,
                      with_prefactor=True, light_speed=None):
        """
        One-electron Breit-Pauli SOC Hamiltonian in a spin-orbital basis.
        """
        from pyqed.qchem.soc import get_soc_1e_spin_orbital
        return get_soc_1e_spin_orbital(
            self,
            representation=representation,
            mo_coeff=mo_coeff,
            one_center=one_center,
            with_prefactor=with_prefactor,
            light_speed=light_speed,
        )

    def get_soc_2e_somf_ao(self, dm=None, states=None, with_prefactor=True,
                           light_speed=None):
        """
        Two-electron SOMF SOC operator in the AO basis.
        """
        from pyqed.qchem.soc import get_soc_2e_somf_ao
        return get_soc_2e_somf_ao(
            self,
            dm=dm,
            states=states,
            with_prefactor=with_prefactor,
            light_speed=light_speed,
        )

    def get_soc_somf_mo(self, mo_coeff=None, dm=None, states=None, include_1e=True,
                        one_center=True, with_prefactor=True, light_speed=None):
        """
        Full SOMF SOC operator in the MO basis.
        """
        from pyqed.qchem.soc import get_soc_somf_mo
        return get_soc_somf_mo(
            self,
            mo_coeff=mo_coeff,
            dm=dm,
            states=states,
            include_1e=include_1e,
            one_center=one_center,
            with_prefactor=with_prefactor,
            light_speed=light_speed,
        )

    def get_soc_somf_so(self, representation='mo', mo_coeff=None, dm=None,
                        states=None, include_1e=True, one_center=True,
                        with_prefactor=True, light_speed=None):
        """
        Full SOMF SOC Hamiltonian in a spin-orbital basis.
        """
        from pyqed.qchem.soc import get_soc_somf_spin_orbital
        return get_soc_somf_spin_orbital(
            self,
            representation=representation,
            mo_coeff=mo_coeff,
            dm=dm,
            states=states,
            include_1e=include_1e,
            one_center=one_center,
            with_prefactor=with_prefactor,
            light_speed=light_speed,
        )

    def make_rdm1(self):
        return make_rdm1(self.mo_coeff, self.mo_occ)

    def get_ovlp(self):
        if self._pyscf_mf is not None:
            return self._pyscf_mf.get_ovlp()
        return self.mol.overlap

    def localize_orbitals(
        self,
        method='ibo',
        space='occ',
        mo_coeff=None,
        occ_threshold=0.5,
        return_indices=False,
        return_info=False,
        **kwargs,
    ):
        """
        Localize molecular orbitals from the converged RHF solution.

        Examples
        --------
        >>> mf = mol.RHF().run()
        >>> c_ibo = mf.localize_orbitals(method="ibo", space="occ")

        Parameters
        ----------
        method : str
            Localization method. Supports native ``"boys"``, ``"pm"``
            (Pipek-Mezey), ``"ibo"``, and ``"lm"`` localization.
        space : str
            Orbital subspace to localize when ``mo_coeff`` is not supplied.
            Currently ``"occ"`` is supported.
        mo_coeff : ndarray, optional
            MO coefficient block to localize directly. If omitted, orbitals are
            selected from ``self.mo_coeff`` using ``space``.
        occ_threshold : float
            Orbitals with occupation larger than this value are treated as occupied.
        return_indices : bool
            If True, return ``(localized_coeff, mo_indices)``.
        return_info : bool
            If True, return convergence metadata. With ``return_indices=True``,
            the return value is ``(localized_coeff, mo_indices, info)``.
        **kwargs
            Forwarded to the backend localization routine.
        """
        if self.mo_coeff is None or self.mo_occ is None:
            raise RuntimeError("Run RHF before localizing orbitals.")

        method_key = str(method).lower().replace('-', '_')
        if method_key in {'intrinsic_bond_orbital', 'intrinsic_bond_orbitals'}:
            method_key = 'ibo'
        if method_key in {'lowdin_mulliken', 'lowdin_mulliken_population'}:
            method_key = 'lm'
        if method_key in {'pipek_mezey', 'pipek_mezey_population'}:
            method_key = 'pm'
        if method_key not in {'ibo', 'boys', 'lm', 'pm'}:
            raise ValueError("Supported localization methods are 'boys', 'pm', 'ibo', and 'lm'.")

        if mo_coeff is None:
            space_key = str(space).lower().replace('-', '_')
            if space_key == 'occupied':
                space_key = 'occ'
            if space_key != 'occ':
                raise ValueError("Only space='occ' is currently supported when mo_coeff is not supplied.")

            coeff = self.mo_coeff
            occ = np.asarray(self.mo_occ)
            orb_indices = np.flatnonzero(occ > occ_threshold)
            if orb_indices.size == 0:
                raise ValueError("No occupied orbitals found for localization.")
            orb = coeff[:, orb_indices]
        else:
            orb_indices = None
            orb = mo_coeff

        coeff = np.asarray(orb)
        if coeff.ndim != 2:
            raise ValueError("mo_coeff must be a 2D array.")
        if coeff.shape[0] != self.nao:
            raise ValueError(f"mo_coeff has {coeff.shape[0]} AO rows, expected {self.nao}.")

        if method_key == 'boys':
            r_ao = _as_xyz_operator(
                self.mol.moment_integral(center=np.zeros(3)),
                'moment_integral()',
            )
            max_cycle = kwargs.pop('max_cycle', 100)
            conv_tol = kwargs.pop('conv_tol', kwargs.pop('tol', 1e-10))
            rotation_tol = kwargs.pop('rotation_tol', 1e-12)
            if kwargs:
                unknown = ', '.join(sorted(kwargs))
                raise TypeError(f"Unsupported Boys localization keyword(s): {unknown}.")
            localized, info = _boys_localize(
                coeff,
                r_ao,
                max_cycle=max_cycle,
                conv_tol=conv_tol,
                rotation_tol=rotation_tol,
            )
            if return_indices and return_info:
                return localized, orb_indices, info
            if return_indices:
                return localized, orb_indices
            if return_info:
                return localized, info
            return localized

        max_cycle = kwargs.pop('max_cycle', 100)
        conv_tol = kwargs.pop('conv_tol', kwargs.pop('tol', 1e-10))
        rotation_tol = kwargs.pop('rotation_tol', 1e-12)
        if kwargs:
            unknown = ', '.join(sorted(kwargs))
            raise TypeError(f"Unsupported {method_key.upper()} localization keyword(s): {unknown}.")

        ao_labels = self.mol.ao_labels()
        atom_groups = _group_ao_indices_by_atom(ao_labels, self.mol.natom)
        if method_key == 'pm':
            localized, info = _pipek_mezey_localize(
                coeff,
                self.get_ovlp(),
                atom_groups,
                method=method_key,
                max_cycle=max_cycle,
                conv_tol=conv_tol,
                rotation_tol=rotation_tol,
            )
        else:
            localized, info = _ibo_localize(
                coeff,
                self.get_ovlp(),
                atom_groups,
                method=method_key,
                max_cycle=max_cycle,
                conv_tol=conv_tol,
                rotation_tol=rotation_tol,
            )

        if return_indices and return_info:
            return localized, orb_indices, info
        if return_indices:
            return localized, orb_indices
        if return_info:
            return localized, info
        return localized

    def analyze(self):
        from .analysis import RHFAnalysis
        return RHFAnalysis(self)

    def mo_components(
        self,
        mo_indices=None,
        metric='mulliken',
        min_contribution=0.0,
        sort=True,
    ):
        return self.analyze().mo_components(
            mo_indices=mo_indices,
            metric=metric,
            min_contribution=min_contribution,
            sort=sort,
        )

    def print_mo_components(
        self,
        mo_indices=None,
        metric='mulliken',
        min_contribution=0.0,
        sort=True,
    ):
        return self.analyze().print_mo_components(
            mo_indices=mo_indices,
            metric=metric,
            min_contribution=min_contribution,
            sort=sort,
        )

    def mulliken_charges(self, dm=None):
        return self.analyze().mulliken_charges(dm=dm)

    def print_mulliken_charges(self, dm=None):
        return self.analyze().print_mulliken_charges(dm=dm)

    def lowdin_charges(self, dm=None):
        return self.analyze().lowdin_charges(dm=dm)

    def print_lowdin_charges(self, dm=None):
        return self.analyze().print_lowdin_charges(dm=dm)

    def mayer_bond_orders(self, dm=None):
        return self.analyze().mayer_bond_orders(dm=dm)

    def print_mayer_bond_orders(self, dm=None, min_bond_order=0.0):
        return self.analyze().print_mayer_bond_orders(dm=dm, min_bond_order=min_bond_order)

    def wiberg_bond_orders(self, dm=None):
        return self.analyze().wiberg_bond_orders(dm=dm)

    def print_wiberg_bond_orders(self, dm=None, min_bond_order=0.0):
        return self.analyze().print_wiberg_bond_orders(dm=dm, min_bond_order=min_bond_order)

    def mo_composition(
        self,
        mo_indices=None,
        metric='mulliken',
        group_by='atom+shell',
        min_contribution=0.0,
        sort=True,
    ):
        return self.analyze().mo_composition(
            mo_indices=mo_indices,
            metric=metric,
            group_by=group_by,
            min_contribution=min_contribution,
            sort=sort,
        )

    def print_mo_composition(
        self,
        mo_indices=None,
        metric='mulliken',
        group_by='atom+shell',
        min_contribution=0.0,
        sort=True,
    ):
        return self.analyze().print_mo_composition(
            mo_indices=mo_indices,
            metric=metric,
            group_by=group_by,
            min_contribution=min_contribution,
            sort=sort,
        )

    def mo_overlap(self, other, mo_indices=None, other_mo_indices=None):
        return self.analyze().mo_overlap(
            other,
            mo_indices=mo_indices,
            other_mo_indices=other_mo_indices,
        )

    def sample_mo(self, mo_index, coords, screen_basis=True, tol_screen=1e-8):
        return self.analyze().sample_mo(
            mo_index,
            coords,
            screen_basis=screen_basis,
            tol_screen=tol_screen,
        )

    def sample_mo_grid(
        self,
        mo_index,
        nx=40,
        ny=None,
        nz=None,
        margin=3.0,
        bounds=None,
        screen_basis=True,
        tol_screen=1e-8,
    ):
        return self.analyze().sample_mo_grid(
            mo_index,
            nx=nx,
            ny=ny,
            nz=nz,
            margin=margin,
            bounds=bounds,
            screen_basis=screen_basis,
            tol_screen=tol_screen,
        )

    def sample_density_grid(
        self,
        nx=40,
        ny=None,
        nz=None,
        margin=3.0,
        bounds=None,
        dm=None,
        screen_basis=True,
        tol_screen=1e-8,
    ):
        return self.analyze().sample_density_grid(
            nx=nx,
            ny=ny,
            nz=nz,
            margin=margin,
            bounds=bounds,
            dm=dm,
            screen_basis=screen_basis,
            tol_screen=tol_screen,
        )

    def orbital_cube(
        self,
        orbital_index,
        filename,
        coeff=None,
        nx=40,
        ny=None,
        nz=None,
        margin=3.0,
        bounds=None,
        screen_basis=True,
        tol_screen=1e-8,
        comment=None,
    ):
        return self.analyze().orbital_cube(
            orbital_index,
            filename,
            coeff=coeff,
            nx=nx,
            ny=ny,
            nz=nz,
            margin=margin,
            bounds=bounds,
            screen_basis=screen_basis,
            tol_screen=tol_screen,
            comment=comment,
        )

    def density_cube(
        self,
        filename,
        dm=None,
        nx=40,
        ny=None,
        nz=None,
        margin=3.0,
        bounds=None,
        screen_basis=True,
        tol_screen=1e-8,
        comment=None,
    ):
        from pyqed.qchem.tools import cubegen

        return cubegen.density(
            self,
            filename,
            dm=dm,
            nx=nx,
            ny=ny,
            nz=nz,
            margin=margin,
            bounds=bounds,
            screen_basis=screen_basis,
            tol_screen=tol_screen,
            comment=comment,
        )

    def plot_mo(self, mo_index='homo', save=None, **kwargs):
        return self.analyze().plot_mo(mo_index, save=save, **kwargs)

    def plot_mo_3d(
        self,
        mo_index,
        nx=40,
        ny=None,
        nz=None,
        margin=3.0,
        bounds=None,
        isovalue=None,
        isovalue_fraction=0.2,
        positive_color='#1f77b4',
        negative_color='#d62728',
        alpha=0.45,
        atom_size=60.0,
        show_atoms=True,
        show_bonds=None,
        bond_scale=1.25,
        bond_color='#555555',
        bond_linewidth=1.6,
        atom_colors=None,
        atom_render='sphere',
        atom_alpha=None,
        sphere_quality=20,
        label_atoms=None,
        screen_basis=True,
        tol_screen=1e-8,
        ax=None,
        figsize=(7.0, 6.0),
        elev=20.0,
        azim=-60.0,
        clean_axes=None,
        axis_off=None,
        style='default',
        title=None,
        title_fontsize=None,
        title_pad=None,
        backend='matplotlib',
        save=None,
    ):
        return self.analyze().plot_mo_3d(
            mo_index,
            nx=nx,
            ny=ny,
            nz=nz,
            margin=margin,
            bounds=bounds,
            isovalue=isovalue,
            isovalue_fraction=isovalue_fraction,
            positive_color=positive_color,
            negative_color=negative_color,
            alpha=alpha,
            atom_size=atom_size,
            show_atoms=show_atoms,
            show_bonds=show_bonds,
            bond_scale=bond_scale,
            bond_color=bond_color,
            bond_linewidth=bond_linewidth,
            atom_colors=atom_colors,
            atom_render=atom_render,
            atom_alpha=atom_alpha,
            sphere_quality=sphere_quality,
            label_atoms=label_atoms,
            screen_basis=screen_basis,
            tol_screen=tol_screen,
            ax=ax,
            figsize=figsize,
            elev=elev,
            azim=azim,
            clean_axes=clean_axes,
            axis_off=axis_off,
            style=style,
            title=title,
            title_fontsize=title_fontsize,
            title_pad=title_pad,
            backend=backend,
            save=save,
        )

    def plot_frontier_mos_3d(self, mo_indices=None, figsize=(12.0, 5.5), save=None, **kwargs):
        return self.analyze().plot_frontier_mos_3d(
            mo_indices=mo_indices,
            figsize=figsize,
            save=save,
            **kwargs,
        )

    def plot_density_3d(
        self,
        nx=40,
        ny=None,
        nz=None,
        margin=3.0,
        bounds=None,
        dm=None,
        isovalue=None,
        isovalues=None,
        isovalue_fraction=0.2,
        isovalue_fractions=(0.01, 0.03, 0.08),
        color='#4c78a8',
        colors=None,
        alpha=0.40,
        alphas=None,
        atom_size=60.0,
        show_atoms=True,
        show_bonds=None,
        bond_scale=1.25,
        bond_color='#555555',
        bond_linewidth=1.6,
        atom_colors=None,
        atom_render='sphere',
        atom_alpha=None,
        sphere_quality=20,
        label_atoms=None,
        screen_basis=True,
        tol_screen=1e-8,
        ax=None,
        figsize=(7.0, 6.0),
        elev=20.0,
        azim=-60.0,
        clean_axes=None,
        axis_off=None,
        style='default',
        title='Electron Density',
        title_fontsize=None,
        title_pad=None,
        smooth_sigma=None,
        backend='matplotlib',
        save=None,
    ):
        return self.analyze().plot_density_3d(
            nx=nx,
            ny=ny,
            nz=nz,
            margin=margin,
            bounds=bounds,
            dm=dm,
            isovalue=isovalue,
            isovalues=isovalues,
            isovalue_fraction=isovalue_fraction,
            isovalue_fractions=isovalue_fractions,
            color=color,
            colors=colors,
            alpha=alpha,
            alphas=alphas,
            atom_size=atom_size,
            show_atoms=show_atoms,
            show_bonds=show_bonds,
            bond_scale=bond_scale,
            bond_color=bond_color,
            bond_linewidth=bond_linewidth,
            atom_colors=atom_colors,
            atom_render=atom_render,
            atom_alpha=atom_alpha,
            sphere_quality=sphere_quality,
            label_atoms=label_atoms,
            screen_basis=screen_basis,
            tol_screen=tol_screen,
            ax=ax,
            figsize=figsize,
            elev=elev,
            azim=azim,
            clean_axes=clean_axes,
            axis_off=axis_off,
            style=style,
            title=title,
            title_fontsize=title_fontsize,
            title_pad=title_pad,
            smooth_sigma=smooth_sigma,
            backend=backend,
            save=save,
        )

    def get_fock(self, dm=None):
        
        if dm is None:
            dm = self.dm 
            
        hcore = self.get_hcore()
        veff = self.get_veff(dm)
        
        return hcore + veff

    def get_veff(self, dm):
        if self._pyscf_mf is not None:
            return self._pyscf_mf.get_veff(dm=dm)
        if self.eri_factors is not None:
            return get_veff(self.mol, dm, eri_factors=self.eri_factors)
        return get_veff(self.mol, dm)

    def get_j(self):
        if self._pyscf_mf is not None:
            return self._pyscf_mf.get_j(dm=self.dm)
        if self.eri_factors is not None:
            return get_jk(self.mol, self.dm, eri_factors=self.eri_factors)[0]
        return get_jk(self.mol, self.dm)[0]
    
    def get_jk(self):
        if self._pyscf_mf is not None:
            return self._pyscf_mf.get_jk(dm=self.dm)
        if self.eri_factors is not None:
            return get_jk(self.mol, self.dm, eri_factors=self.eri_factors)
        return get_jk(self.mol, self.dm)

    def get_hcore(self):
        """
        get core Hamiltonian in AO

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # return get_hcore(self.mol)
        if self._pyscf_mf is not None:
            return self._pyscf_mf.get_hcore()
        return self.mol.hcore

    def get_hcore_mo(self, mo_coeff=None):
        """
        get core Hamiltonian in canonical molecular orbitals

        Returns
        -------
        ndarray.

        """
        if mo_coeff is None:
            C = self.mo_coeff
        else:
            C = mo_coeff
            
        return dag(C) @ self.mol.hcore @ C

    def get_eri_mo(self, mo_coeff=None, notation='chem'):
        """
        get electron repulsion integrals in MOs

        Parameters
        ----------
        mo_coeff: MO coefficients. If None, use canonical MOs.

        notation: Default: chem.

        Returns
        -------
        eri_mo : TYPE
            DESCRIPTION.

        """
        if mo_coeff is None:
            C = self.mo_coeff
        else:
            C = mo_coeff

        eri_source = self.eri
        if eri_source is None and getattr(self, 'eri_s4', None) is not None:
            from pyqed.qchem.basis import unpack_eri_s4
            eri_source = unpack_eri_s4(self.eri_s4, self.mol.nao)
        if eri_source is None and getattr(self, 'eri_s8', None) is not None:
            from pyqed.qchem.basis import unpack_eri_s8
            eri_source = unpack_eri_s8(self.eri_s8, self.mol.nao)
        if eri_source is None:
            eri_source = getattr(self.mol, 'eri', None)
        if eri_source is None and getattr(self.mol, 'eri_s4', None) is not None:
            from pyqed.qchem.basis import unpack_eri_s4
            eri_source = unpack_eri_s4(self.mol.eri_s4, self.mol.nao)
        if eri_source is None and getattr(self.mol, 'eri_s8', None) is not None:
            from pyqed.qchem.basis import unpack_eri_s8
            eri_source = unpack_eri_s8(self.mol.eri_s8, self.mol.nao)

        eri_ndim = None if eri_source is None else np.asarray(eri_source).ndim
        if eri_source is not None and eri_ndim == 4:
            if notation == 'chem':
                eri_mo = contract('ijkl, ip, jq, kr, ls -> pqrs', eri_source, C.conj(), C, C.conj(), C)
            elif notation == 'phys':
                eri_mo = contract('ijkl, ip, jq, kr, ls -> prqs', eri_source, C.conj(), C, C.conj(), C)
            else:
                raise ValueError("notation must be 'chem' or 'phys'.")
            return eri_mo

        eri_factors = getattr(self, 'eri_factors', None)
        if eri_factors is None:
            eri_factors = getattr(self.mol, 'eri_factors', None)
        if eri_factors is None and eri_source is not None and eri_ndim in (2, 3):
            eri_factors = eri_source
        if eri_factors is None:
            raise ValueError("RHF.get_eri_mo() requires either dense eri or eri_factors.")

        from pyqed.qchem.basis import transform_ri_factors_to_mo_pair
        pair_factors = transform_ri_factors_to_mo_pair(eri_factors, C)
        eri_mo = contract('Ppq,Prs->pqrs', pair_factors, pair_factors)

        if notation == 'chem':
            return eri_mo
        if notation == 'phys':
            return eri_mo.transpose(0, 2, 1, 3)
        raise ValueError("notation must be 'chem' or 'phys'.")

    def to_uhf(self):
        # transform a RHF to UHF format with spin orbitals
        from .uhf import UHF

        uhf = UHF(self.mol, init_guess=self.init_guess)
        uhf.e_tot = self.e_tot
        uhf.e_nuc = self.e_nuc
        uhf.hcore = (self.hcore, self.hcore)
        uhf.mo_energy = (self.mo_energy.copy(), self.mo_energy.copy())
        uhf.mo_coeff = (self.mo_coeff.copy(), self.mo_coeff.copy())

        mo_occ_a = np.zeros_like(self.mo_occ, dtype=float)
        mo_occ_b = np.zeros_like(self.mo_occ, dtype=float)
        mo_occ_a[:self.nocc] = 1.0
        mo_occ_b[:self.nocc] = 1.0
        uhf.mo_occ = (mo_occ_a, mo_occ_b)
        uhf.dm = np.array((
            make_rdm1(uhf.mo_coeff[0], mo_occ_a),
            make_rdm1(uhf.mo_coeff[1], mo_occ_b),
        ))
        uhf.vhf = uhf.get_veff(uhf.dm)
        uhf.converged = self.e_tot is not None
        uhf.na = self.nocc
        uhf.nb = self.nocc
        uhf.nocc = (self.nocc, self.nocc)
        uhf.nvir = (self.nvir, self.nvir)
        return uhf

    def RTTDHF(self, interaction_ao=None, field=None, **kwargs):
        """
        Convenience constructor for real-time TDHF propagation.
        """
        from pyqed.qchem.rttdhf import RTTDHF
        return RTTDHF(self, interaction_ao=interaction_ao, field=field, **kwargs)


    def energy_elec(self,dm=None):
        if dm is None:
            dm = self.make_rdm1()

        return energy_elec(dm, self.hcore, self.vhf)

    def energy_nuc(self):
        return self.e_nuc

    def eri_asymm(self):
        """
        antisymmetrized electron-repulsion integral in physicists' notation

        .. math::

            \langle ij||kl \rangle =  \langle ij | kl \rangle - \langle ij | lk \rangle

        Parameters
        ----------
        notation : TYPE, optional
            DESCRIPTION. The default is 'phys'.

        Returns
        -------
        None.

        """
        eri = self.get_eri_mo(notation='phys')
        return eri - np.transpose(eri, (0,1,3,2))


# def get_hcore(mol):
#     '''Core Hamiltonian
#     Examples:
#     >>> from pyscf import gto, scf
#     >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.1')
#     >>> scf.hf.get_hcore(mol)
#     array([[-0.93767904, -0.59316327],
#            [-0.59316327, -0.93767904]])

#     From Pyscf.
#     '''
#     h = mol.intor_symmetric('int1e_kin')

#     if mol._pseudo:
#         # Although mol._pseudo for GTH PP is only available in Cell, GTH PP
#         # may exist if mol is converted from cell object.
#         from pyscf.gto import pp_int
#         h += pp_int.get_gth_pp(mol)
#     else:
#         h+= mol.intor_symmetric('int1e_nuc')

#     if len(mol._ecpbas) > 0:
#         h += mol.intor_symmetric('ECPscalar')
#     return h


def get_veff(mol, dm, dm_last=None, vhf_last=None, hermi=1, vhfopt=None,
             eri_factors=None):
    '''Unrestricted Hartree-Fock potential matrix for the given density matrix

    .. math::

        V_{ij}^\alpha &= \sum_{kl} (ij|kl)(\gamma_{lk}^\alpha+\gamma_{lk}^\beta)
                       - \sum_{kl} (il|kj)\gamma_{lk}^\alpha \\
        V_{ij}^\beta  &= \sum_{kl} (ij|kl)(\gamma_{lk}^\alpha+\gamma_{lk}^\beta)
                       - \sum_{kl} (il|kj)\gamma_{lk}^\beta

    Args:
        mol : an instance of :class:`Mole`
        dm : ndarray or list of ndarrays
            A density matrix or a list of density matrices
    Kwargs:
        dm_last : ndarray or a list of ndarrays or 0
            The density matrix baseline.  If not 0, this function computes the
            increment of HF potential w.r.t. the reference HF potential matrix.
        vhf_last : ndarray or a list of ndarrays or 0
            The reference HF potential matrix.
        hermi : int
            Whether J, K matrix is hermitian
            | 0 : no hermitian or symmetric
            | 1 : hermitian
            | 2 : anti-hermitian
        vhfopt :
            A class which holds precomputed quantities to optimize the
            computation of J, K matrices
    Returns:
        matrix Vhf = 2*J - K.  Vhf can be a list matrices, corresponding to the
        input density matrices.
    Examples:
    >>> import np
    >>> from pyscf import gto, scf
    >>> from pyscf.scf import _vhf
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.1')
    >>> dm0 = np.random.random((mol.nao_nr(),mol.nao_nr()))
    >>> vhf0 = scf.hf.get_veff(mol, dm0, hermi=0)
    >>> dm1 = np.random.random((mol.nao_nr(),mol.nao_nr()))
    >>> vhf1 = scf.hf.get_veff(mol, dm1, hermi=0)
    >>> vhf2 = scf.hf.get_veff(mol, dm1, dm_last=dm0, vhf_last=vhf0, hermi=0)
    >>> np.allclose(vhf1, vhf2)
    True
    '''
    if dm_last is None:
        vj, vk = get_jk(mol, np.asarray(dm), eri_factors=eri_factors)
        return vj - vk * .5
    else:
        ddm = np.asarray(dm) - np.asarray(dm_last)
        vj, vk = get_jk(mol, ddm, eri_factors=eri_factors)
        return vj - vk * .5 + np.asarray(vhf_last)


# def get_jk(mol, dm, hermi=1, vhfopt=None, with_j=True, with_k=True, omega=None):
#     '''Compute J, K matrices for all input density matrices
#     Args:
#         mol : an instance of :class:`Mole`
#         dm : ndarray or list of ndarrays
#             A density matrix or a list of density matrices
#     Kwargs:
#         hermi : int
#             Whether J, K matrix is hermitian
#             | 0 : not hermitian and not symmetric
#             | 1 : hermitian or symmetric
#             | 2 : anti-hermitian
#         vhfopt :
#             A class which holds precomputed quantities to optimize the
#             computation of J, K matrices
#         with_j : boolean
#             Whether to compute J matrices
#         with_k : boolean
#             Whether to compute K matrices
#         omega : float
#             Parameter of range-seperated Coulomb operator: erf( omega * r12 ) / r12.
#             If specified, integration are evaluated based on the long-range
#             part of the range-seperated Coulomb operator.
#     Returns:
#         Depending on the given dm, the function returns one J and one K matrix,
#         or a list of J matrices and a list of K matrices, corresponding to the
#         input density matrices.
#     Examples:
#     >>> from pyscf import gto, scf
#     >>> from pyscf.scf import _vhf
#     >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.1')
#     >>> dms = np.random.random((3,mol.nao_nr(),mol.nao_nr()))
#     >>> j, k = scf.hf.get_jk(mol, dms, hermi=0)
#     >>> print(j.shape)
#     (3, 2, 2)
#     '''
#     dm = np.asarray(dm, order='C')
#     dm_shape = dm.shape
#     dm_dtype = dm.dtype
#     nao = dm_shape[-1]

#     if dm_dtype == np.complex128:
#         dm = np.vstack((dm.real, dm.imag)).reshape(-1,nao,nao)
#         hermi = 0

#     if omega is None:
#         vj, vk = _vhf.direct(dm, mol._atom, mol._bas, mol._env,
#                              vhfopt, hermi, mol.cart, with_j, with_k)
#     else:
#         # The vhfopt of standard Coulomb operator can be used here as an approximate
#         # integral prescreening conditioner since long-range part Coulomb is always
#         # smaller than standard Coulomb.  It's safe to filter LR integrals with the
#         # integral estimation from standard Coulomb.
#         with mol.with_range_coulomb(omega):
#             vj, vk = _vhf.direct(dm, mol._atm, mol._bas, mol._env,
#                                  vhfopt, hermi, mol.cart, with_j, with_k)

#     if dm_dtype == np.complex128:
#         if with_j:
#             vj = vj.reshape((2,) + dm_shape)
#             vj = vj[0] + vj[1] * 1j
#         if with_k:
#             vk = vk.reshape((2,) + dm_shape)
#             vk = vk[0] + vk[1] * 1j
#     else:
#         if with_j:
#             vj = vj.reshape(dm_shape)
#         if with_k:
#             vk = vk.reshape(dm_shape)
#     return vj, vk


def pivoted_cholesky_eri(eri, tol=1e-8, max_rank=None, init_pivots=None, return_pivots=False):
    """
    Pivoted-Cholesky factorization of the AO ERI pair matrix.

    The ERI tensor ``(ij|kl)`` is reshaped to the full AO-pair space and
    approximated as ``(ij|kl) ~= sum_P L_P[i,j] L_P[k,l]``.
    ``init_pivots`` may be supplied to warm-start nearby geometries.
    """
    eri = np.asarray(eri)
    nao = eri.shape[0]
    npair = nao * nao
    pair_mat = eri.reshape(npair, npair)

    if max_rank is None:
        max_rank = npair
    max_rank = min(int(max_rank), npair)

    diag = np.real(np.diag(pair_mat)).copy()
    chol = np.zeros((npair, max_rank), dtype=eri.dtype)
    rank = 0
    pivots = []

    if init_pivots is not None:
        used = set()
        for pivot in init_pivots:
            if rank >= max_rank:
                break
            pivot = int(pivot)
            if pivot < 0 or pivot >= npair or pivot in used:
                continue
            used.add(pivot)

            delta = float(diag[pivot])
            if delta <= tol:
                continue

            col = np.array(pair_mat[:, pivot], copy=True)
            if rank > 0:
                col -= chol[:, :rank] @ chol[pivot, :rank].conj()

            delta = float(np.real(col[pivot]))
            if delta <= tol:
                diag[pivot] = 0.0
                continue

            chol[:, rank] = col / np.sqrt(delta)
            diag -= np.real(chol[:, rank] * chol[:, rank].conj())
            diag = np.maximum(diag, 0.0)
            pivots.append(pivot)
            rank += 1

    for _ in range(rank, max_rank):
        pivot = int(np.argmax(diag))
        delta = float(diag[pivot])
        if delta <= tol:
            break

        col = np.array(pair_mat[:, pivot], copy=True)
        if rank > 0:
            col -= chol[:, :rank] @ chol[pivot, :rank].conj()

        delta = float(np.real(col[pivot]))
        if delta <= tol:
            diag[pivot] = 0.0
            continue

        chol[:, rank] = col / np.sqrt(delta)
        diag -= np.real(chol[:, rank] * chol[:, rank].conj())
        diag = np.maximum(diag, 0.0)
        pivots.append(pivot)
        rank += 1

    factors = chol[:, :rank].T.reshape(rank, nao, nao)
    if return_pivots:
        return factors, tuple(pivots)
    return factors


def _low_rank_cache_family_key(mol):
    return (
        tuple(mol.atom_symbols()),
        repr(mol.basis),
        int(mol.charge),
        int(mol.spin),
        getattr(mol, '_build_driver', None),
        int(mol.nao),
    )


def _low_rank_cache_geometry_key(mol):
    return (
        _low_rank_cache_family_key(mol),
        mol.geometry_hash() if hasattr(mol, 'geometry_hash') else None,
        tuple(np.asarray(mol.atom_coords(), dtype=float).shape),
    )


def get_or_build_low_rank_eri_factors(mol, tol=1e-8, max_rank=None, warm_start=True):
    """
    Return cached pivoted-Cholesky AO ERI factors for the requested settings.

    The cache is geometry-aware. Repeated calls at the same geometry reuse the
    exact factors, while nearby geometries can warm-start from the most recent
    entry in the same molecular/basis family.
    """
    existing = getattr(mol, 'eri_factors', None)
    if getattr(mol, 'eri', None) is None:
        if existing is not None:
            return existing
        if getattr(mol, 'eri_s4', None) is not None:
            from pyqed.qchem.basis import unpack_eri_s4
            mol.eri = unpack_eri_s4(mol.eri_s4, mol.nao)
        elif getattr(mol, 'eri_s8', None) is not None:
            from pyqed.qchem.basis import unpack_eri_s8
            mol.eri = unpack_eri_s8(mol.eri_s8, mol.nao)
        else:
            raise ValueError("mol.eri, mol.eri_s4, mol.eri_s8, or mol.eri_factors is required for the builtin low-rank J/K path.")

    cache = getattr(mol, '_low_rank_eri_cache', None)
    if cache is None:
        cache = {}
        setattr(mol, '_low_rank_eri_cache', cache)

    settings_key = (float(tol), None if max_rank is None else int(max_rank))
    family_key = _low_rank_cache_family_key(mol)
    geom_key = _low_rank_cache_geometry_key(mol)
    key = (settings_key, family_key)

    bucket = cache.get(key)
    if bucket is None:
        bucket = {'entries': {}, 'recent_geom_key': None}
        cache[key] = bucket

    entries = bucket['entries']
    if geom_key in entries:
        bucket['recent_geom_key'] = geom_key
        mol._low_rank_eri_last_info = {
            'mode': 'exact',
            'settings_key': settings_key,
            'geometry_key': geom_key,
            'rank': entries[geom_key]['factors'].shape[0],
        }
        return entries[geom_key]['factors']

    init_pivots = None
    if warm_start and entries:
        recent_geom_key = bucket.get('recent_geom_key')
        if recent_geom_key is not None and recent_geom_key in entries:
            init_pivots = entries[recent_geom_key].get('pivots')
        else:
            init_pivots = next(reversed(entries.values())).get('pivots')

    factors, pivots = pivoted_cholesky_eri(
        mol.eri,
        tol=tol,
        max_rank=max_rank,
        init_pivots=init_pivots,
        return_pivots=True,
    )
    entries[geom_key] = {'factors': factors, 'pivots': pivots}
    bucket['recent_geom_key'] = geom_key

    # Keep a short per-family history so geometry scans do not grow forever.
    while len(entries) > 8:
        oldest_key = next(iter(entries))
        if oldest_key == geom_key:
            break
        del entries[oldest_key]

    mol._low_rank_eri_last_info = {
        'mode': 'warm' if init_pivots is not None else 'cold',
        'settings_key': settings_key,
        'geometry_key': geom_key,
        'rank': factors.shape[0],
    }
    return factors


def get_jk(mol, dm, eri_factors=None):
    """
    get the Colomb and exchange terms in the Fock matrix

    .. math::

        G[i,j] = P[k,l] * (v[i,j,k,l]  - 0.5 * v[i,l,k,j])

    Parameters
    ----------
    mol : TYPE
        DESCRIPTION.
    dm : TYPE
        DESCRIPTION.

    Returns
    -------
    vj : TYPE
        DESCRIPTION.
    vk : TYPE
        DESCRIPTION.

    """
    if eri_factors is not None:
        from pyqed.qchem.basis import contract_jk_ri
        return contract_jk_ri(eri_factors, dm, mol.nao)

    eri_s4 = getattr(mol, 'eri_s4', None)
    if eri_s4 is not None:
        from pyqed.qchem.basis import contract_jk_s4
        return contract_jk_s4(eri_s4, dm, mol.nao)

    eri_s8 = getattr(mol, 'eri_s8', None)
    if eri_s8 is not None:
        from pyqed.qchem.basis import contract_jk_s8
        return contract_jk_s8(eri_s8, dm, mol.nao)

    direct_jk_data = getattr(mol, '_builtin_direct_jk_data', None)
    if direct_jk_data is not None:
        from pyqed.qchem.basis import _basis_cy

        transform = getattr(mol, '_ao_cart2sph', None)
        if transform is None and direct_jk_data.get("cache_aosym") == "s8" and hasattr(_basis_cy, "compute_eri_s8"):
            from pyqed.qchem.basis import contract_jk_s8

            eri_s8, _computed, _skipped = _basis_cy.compute_eri_s8(
                direct_jk_data["shells"],
                direct_jk_data["origins"],
                direct_jk_data["exps"],
                direct_jk_data["weights"],
                direct_jk_data["nprim"],
                direct_jk_data["pair_bounds"],
                float(direct_jk_data.get("screen_tol", 0.0)),
            )
            mol.eri_s8 = eri_s8
            mol._builtin_direct_jk_data = None
            return contract_jk_s8(eri_s8, dm, mol.nao)

        dm_work = dm
        if transform is not None:
            dm_work = np.einsum('pa,ab,qb->pq', transform, dm, transform, optimize=True)
        vj, vk = _basis_cy.direct_jk(
            direct_jk_data["shells"],
            direct_jk_data["origins"],
            direct_jk_data["exps"],
            direct_jk_data["weights"],
            direct_jk_data["nprim"],
            direct_jk_data["pair_bounds"],
            np.ascontiguousarray(dm_work, dtype=np.float64),
            float(direct_jk_data.get("screen_tol", 0.0)),
        )
        if transform is not None:
            vj = np.einsum('pa,pq,qb->ab', transform, vj, transform, optimize=True)
            vk = np.einsum('pa,pq,qb->ab', transform, vk, transform, optimize=True)
        return vj, vk

    eri = mol.eri
    if eri is None:
        raise ValueError("get_jk requires mol.eri, mol.eri_s4, mol.eri_s8, mol._builtin_direct_jk_data, or eri_factors.")

    vj = contract('lk, ijkl -> ij', dm, eri)
    vk = contract('lk, ilkj -> ij', dm, eri)
    return vj, vk




def energy_elec(dm, h1e=None, vhf=None):
    r'''Electronic part of Hartree-Fock energy, for given core hamiltonian and
    HF potential

    ... math::
        E = \sum_{ij}h_{ij} \gamma_{ji}
          + \frac{1}{2}\sum_{ijkl} \gamma_{ji}\gamma_{lk} \langle ik||jl\rangle
    Note this function has side effects which cause mf.scf_summary updated.
    Args:
        mf : an instance of SCF class
    Kwargs:
        dm : 2D ndarray
            one-partical density matrix
        h1e : 2D ndarray
            Core hamiltonian
        vhf : 2D ndarray
            HF potential
    Returns:
        Hartree-Fock electronic energy and the Coulomb energy
    Examples:
    >>> from pyscf import gto, scf
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.1')
    >>> mf = scf.RHF(mol)
    >>> mf.scf()
    >>> dm = mf.make_rdm1()
    >>> scf.hf.energy_elec(mf, dm)
    (-1.5176090667746334, 0.60917167853723675)
    >>> mf.energy_elec(dm)
    (-1.5176090667746334, 0.60917167853723675)
    '''
    # if dm is None: dm = mf.make_rdm1()
    # if h1e is None: h1e = mf.get_hcore()
    # if vhf is None: vhf = mf.get_veff(mf.mol, dm)
    e1 = np.einsum('ij,ji->', h1e, dm).real
    e_coul = np.einsum('ij,ji->', vhf, dm).real * .5
    # mf.scf_summary['e1'] = e1
    # mf.scf_summary['e2'] = e_coul
    # logger.debug(mf, 'E1 = %s  E_coul = %s', e1, e_coul)
    return e1+e_coul #, e_coul




def make_rdm1(mo_coeff, mo_occ, **kwargs):
    '''One-particle density matrix in AO representation
    Args:
        mo_coeff : 2D ndarray
            Orbital coefficients. Each column is one orbital.
        mo_occ : 1D ndarray
            Occupancy
    Returns:
        One-particle density matrix, 2D ndarray
    '''

    mocc = mo_coeff[:,mo_occ>0]
# DO NOT make tag_array for dm1 here because this DM array may be modified and
# passed to functions like get_jk, get_vxc.  These functions may take the tags
# (mo_coeff, mo_occ) to compute the potential if tags were found in the DM
# array and modifications to DM array may be ignored.
    return np.dot(mocc*mo_occ[mo_occ>0], mocc.conj().T)




def hartree_fock(mol, dm0=None, init_guess='hcore', max_cycle=50, tol=1e-8,
                 low_rank_jk=False, low_rank_tol=1e-8, low_rank_max_rank=None,
                 verbose=0):
    verbose = int(verbose)

    #calculate the overlap matrix S
    #the matrix should be symmetric with diagonal entries equal to one
    logging.info("building overlap matrix")

    # S = mol.intor_symmetric('int1e_ovlp')
    S = mol.overlap

    # for i in range(len(phi)):
    #     for j in range( (i+1),len(phi)):
    #         S[i,j] = S[j,i] = overlap_integral_sto(phi[i], phi[j])

    # print("S: ", S)


    # #calculate the kinetic energy matrix T
    # print("building kinetic energy matrix")
    # T = np.zeros((K,K))

    # #print('test', phi[0].g[0].center)
    # #print('test', phi[1].g[1].center)

    # for i in range(len(phi)):
    #     for j in range(i, len(phi)):
    #         T[i,j] = T[j,i] = kinetic_energy_integral(phi[i], phi[j])


    # #print("building nuclear attraction matrices")

    # V = np.zeros((K,K))

    # for A in range(K):
    #     for i in range(K):
    #         for j in range(i,K):
    #             v = nuclear_attraction_integral(Z[A], R[A], phi[i], phi[j])
    #             V[i,j] += v
    #             if i != j:
    #                 V[j,i] += v
    # #print("V: ", V)

    #build core-Hamiltonian matrix

    # #print("building core-Hamiltonian matrix")
    # Hcore = T + V

    # hcore = get_hcore(mol)

    hcore = mol.hcore

    # print("Hcore: ", Hcore)

    #diagonalize overlap matrix to get transformation matrix X
    #print("diagonalizing overlap matrix")
    s, U = eigh(S)

    # building transformation matrix S^{-1/2}
    X = U.dot(np.diagflat(s**(-0.5)).dot(dagger(U)))


    #calculate all of the two-electron integrals
    #print("building two_electron Coulomb and exchange integrals")

    # two_electron = np.zeros((K,K,K,K))

    # for mu in range(K):
    #     for v in range(K):
    #         for lamb in range(K):
    #             for sigma in range(K):
    #                 two_electron[mu,v,sigma,lamb] = \
    #                     two_electron_integral(phi[mu], phi[v], phi[sigma], phi[lamb])

#                    coulomb  = two_electron_integral(phi[mu], phi[v], \
#                                                     phi[sigma], phi[lamb])
#                    two_electron[mu,v,sigma,lamb] = coulomb
                    #print("coulomb  ( ", mu, v, '|', sigma, lamb,"): ",coulomb)
#                    exchange = two_electron_integral(phi[mu], phi[lamb], \
#                                                     phi[sigma], phi[v])
#                    #print("exchange ( ", mu, lamb, '|', sigma, v, "): ",exchange)
#                    two_electron[mu,lamb,sigma,v] = exchange

    # P = np.zeros((K,K))

    total_energy = 0.0
    old_energy = 0.0
    electronic_energy = 0.0

    nocc = mol.nelec // 2
    mo_occ = np.zeros(mol.nao)
    mo_occ[:nocc] = 2

    # dm = init_guess_by_hcore(hcore)
    def init_guess_by_h1e(h):
        h = dag(X) @ h @ X
        mo_energy, C = eigh(h)
        return make_rdm1(C, mo_occ)

    if dm0 is None:
        dm = init_guess_by_h1e(hcore)

        # dm = mf.get_init_guess(mol, mf.init_guess, s1e=s1e, **kwargs)
    elif init_guess == 'hcore':
        dm = dm0
    else:
        raise ValueError('Invalid init_guess.')


    ### DIIS
    nbas = mol.nao

    # diis storage
    maxdiis = 6
    diis_error_convergence = 1.0e-5

    diis_error_matrices = np.zeros((maxdiis, nbas, nbas))
    diis_fock_matrices = np.zeros_like(diis_error_matrices)

    def diis(fock, dens, overlap, orth, iter):
        """
        Extrapolate new fock matrix based on input fock matrix
            and previous fock-matrices.

        Arguments:
            fock -- current fock matrix

        Returns:
            (fock, error) -- interpolated fock matrix and diis-error
        """
        diis_fock = np.zeros_like(fock)

        if iter <= 1:
            return fock, 0.0

        # copy data down to lower storage
        for k in reversed(range(1, min(iter, maxdiis))):

            diis_error_matrices[k] = diis_error_matrices[k-1][:]
            diis_fock_matrices[k] = diis_fock_matrices[k-1][:]

        # calculate error matrix
        error_mat = reduce(np.dot, (fock, dens, overlap))
        error_mat -= error_mat.T

        # put orthogonal error matrix in storage
        # pulay use S^(-1/2) but here we choose whatever the user has defined
        diis_error_matrices[0]  = reduce(np.dot, (orth.T, error_mat, orth))
        diis_fock_matrices[0] = fock[:]
        diis_error_index = np.abs(diis_error_matrices[0]).argmax()
        diis_error = math.fabs(np.ravel(diis_error_matrices[0])[diis_error_index])

        # calculate B-matrix and solve for coefficients that reduces error
        bsize = min(iter, maxdiis)-1
        bmat = -1.0 * np.ones((bsize+1,bsize+1))
        rhs = np.zeros(bsize+1)
        bmat[bsize, bsize] = 0
        rhs[bsize] = -1
        for b1 in range(bsize):
            for b2 in range(bsize):
                bmat[b1, b2] = np.trace(diis_error_matrices[b1].dot(diis_error_matrices[b2]))
        try:
            C = np.linalg.solve(bmat, rhs)
        except np.linalg.LinAlgError:
            return fock, diis_error

        # form new interpolated diis fock matrix
        for i, k in enumerate(C[:-1]):
            diis_fock += k*diis_fock_matrices[i]

        return diis_fock, diis_error


    # nuclear energy
    # nuclear_energy = 0.0
    # for A in range(len(Z)):
    #     for B in range(A+1,len(Z)):
    #         nuclear_energy += Z[A]*Z[B]/abs(R[A]-R[B])

    nuclear_energy = mol.energy_nuc()

    if verbose >= 1:
        print("E_nclr = ", nuclear_energy)

    if verbose >= 2:
        logging.info("\n {:4s} {:13s} de\n".format("iter", "total energy"))

    eri_factors = None
    if low_rank_jk:
        eri_factors = get_or_build_low_rank_eri_factors(
            mol,
            tol=low_rank_tol,
            max_rank=low_rank_max_rank,
        )

    conv = False
    for scf_iter in range(max_cycle):

        # calculate the two electron part of the Fock matrix
        vhf = get_veff(mol, dm, eri_factors=eri_factors)
        F = hcore + vhf

        # obtain better (interpolated) fock matrix through diis accelleration

        # print('DIIS')
        F, diis_error = diis(F, dm, S, X, scf_iter)


        electronic_energy = energy_elec(dm, hcore, vhf)

        #print("E_elec = ", electronic_energy)

        total_energy = electronic_energy + nuclear_energy

        if verbose >= 2:
            logging.info("{:3} {:12.8f} {:12.4e} ".format(scf_iter, total_energy,\
                   total_energy - old_energy))

        if scf_iter > 2 and abs(old_energy - total_energy) < tol:
            conv = True
            break

        #println("F: ", F)
        #Fprime = X' * F * X
        Fprime = dagger(X).dot(F).dot(X)
        #println("F': $Fprime")
        mo_energy, Cprime = eigh(Fprime)
        # print("epsilon: ", epsilon)
        #print("C': ", Cprime)
        mo_coeff = np.real(np.dot(X,Cprime))
        # print("C: ", C)


        # new density matrix in original basis
        # P = np.zeros(Hcore.shape)
        # for mu in range(len(phi)):
        #     for v in range(len(phi)):
        #         P[mu,v] = 2. * C[mu,0] * C[v,0]
        dm = make_rdm1(mo_coeff, mo_occ)

        old_energy = total_energy

    if not conv: sys.exit('SCF not converged.')

    if verbose >= 1:
        print('E(HF) = ', total_energy)

    # check if this hartree-fock calculation is for configuration interaction
    # or not, if yes, output the essential information
    # if CI == False:
    return total_energy, nuclear_energy, mo_energy, mo_coeff, mo_occ, hcore, vhf,\
        dm
    # else:
    # return C, Hcore, nuclear_energy, two_electron


def pyscf_density_fit_rhf(mol, dm0=None, init_guess='hcore', max_cycle=50,
                          tol=1e-8, auxbasis=None, verbose=0):
    """
    Run a PySCF-backed density-fitted RHF calculation and return pyqed-style
    SCF data plus the underlying PySCF mean-field object.
    """
    from pyscf import scf

    pmol = mol.topyscf()
    pmol.build(verbose=0)

    mf = scf.RHF(pmol).density_fit(auxbasis=auxbasis)
    mf.max_cycle = max_cycle
    mf.conv_tol = tol
    mf.verbose = int(verbose)

    if dm0 is None:
        key = {'hcore': '1e', 'h1e': '1e'}.get(str(init_guess).lower(), init_guess)
        dm0 = mf.get_init_guess(key=key)

    total_energy = mf.kernel(dm0=dm0)
    if not mf.converged:
        raise RuntimeError("PySCF density-fitted RHF did not converge.")

    dm = mf.make_rdm1()
    hcore = mf.get_hcore()
    vhf = mf.get_veff(dm=dm)

    mol.nao = pmol.nao
    mol.nmo = pmol.nao
    mol.nbas = pmol.nbas
    mol.hcore = hcore
    mol.overlap = mf.get_ovlp()
    mol.cart = pmol.cart
    mol._atm = pmol._atm
    mol._bas = pmol._bas
    mol._env = pmol._env

    return (
        total_energy,
        pmol.energy_nuc(),
        mf.mo_energy,
        mf.mo_coeff,
        mf.mo_occ,
        hcore,
        vhf,
        dm,
        mf,
    )


class RHFScanner:
    """
    Minimal scanner wrapper for repeated RHF evaluations on nearby geometries.
    """

    def __init__(self, mf, build_driver=None):
        self.mf = mf
        self.mol = mf.mol
        self.build_driver = build_driver or getattr(self.mol, '_build_driver', None) or 'gbasis'

    def _build_molecule(self, mol):
        driver = getattr(mol, '_build_driver', None) or self.build_driver
        if driver == 'builtin':
            mol.build(driver=driver, options=getattr(mol, 'builtin_options', None))
        else:
            mol.build(driver=driver)

    def __call__(self, mol_or_geom):
        if isinstance(mol_or_geom, np.ndarray):
            mol = self.mol
            mol.set_geom(np.asarray(mol_or_geom, dtype=float).reshape(mol.natom, 3))
            self._build_molecule(mol)
        else:
            mol = mol_or_geom
            if getattr(mol, 'eri', None) is None or getattr(mol, 'hcore', None) is None:
                self._build_molecule(mol)

        run_kwargs = {
            'dm0': None if self.mf.dm is None else np.array(self.mf.dm, copy=True),
            'init_guess': 'hcore',
            'max_cycle': self.mf.max_cycle,
        }

        if self.mf.density_fit:
            run_kwargs['density_fit'] = True
            run_kwargs['auxbasis'] = self.mf.auxbasis
        elif self.mf.cholesky_jk:
            run_kwargs['cholesky_jk'] = True
            run_kwargs['cholesky_tol'] = self.mf.cholesky_tol
            run_kwargs['cholesky_max_rank'] = self.mf.cholesky_max_rank

        new_mf = RHF(mol, init_guess=self.mf.init_guess)
        new_mf.max_cycle = self.mf.max_cycle
        new_mf.run(**run_kwargs)

        self.mf = new_mf
        self.mol = mol
        return new_mf.e_tot


def ao2mo(op, C):
    """
    transform 1e operators from AO to MO representation

    Parameters
    ----------
    op : TYPE
        DESCRIPTION.
    C : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return dag(C) @ op @ C

if __name__ == '__main__':
    # from pyscf import gto, scf
    # mol = gto.M(atom='H 0 0 0; H 0 0 1.1', unit='b', basis='631g')
    # conv, e, mo_e, mo, mo_occ = scf.hf.kernel(scf.hf.SCF(mol), dm0=np.eye(mol.nao_nr()))
    # print('conv = %s, E(HF) = %.12f' % (conv, e))
    # conv = True, E(HF) = -1.081170784378

    from pyqed import Molecule
    mol = Molecule(atom='H 0 0 0; H 0 0 1.1', unit='a', basis='6311g**')
    mol.build()
    # hartree_fock(mol)
    hf = RHF(mol)
    hf.run()
