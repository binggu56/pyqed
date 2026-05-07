#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MP2
"""

import numpy as np
from opt_einsum import contract

from pyqed import dag
from pyqed.optimize import energy as orbital_objective
from pyqed.optimize import minimize as orbital_minimize
from pyqed.qchem.hf import RHF, UHF


def _transform_rdm2_rhf_to_ao(dm2, mo_coeff):
    return contract('pi,qj,ijkl,rk,sl->pqrs', mo_coeff, mo_coeff, dm2, mo_coeff, mo_coeff)


def _transform_rdm2_uhf_to_ao(dm2aa, dm2ab, dm2bb, mo_coeff):
    ca, cb = mo_coeff
    dm2aa_ao = contract('pi,qj,ijkl,rk,sl->pqrs', ca, ca, dm2aa, ca, ca)
    dm2ab_ao = contract('pi,qj,ijkl,rk,sl->pqrs', ca, ca, dm2ab, cb, cb)
    dm2bb_ao = contract('pi,qj,ijkl,rk,sl->pqrs', cb, cb, dm2bb, cb, cb)
    return dm2aa_ao, dm2ab_ao, dm2bb_ao


def _get_rhf_eri_factors(mf):
    eri_factors = getattr(mf, 'eri_factors', None)
    if eri_factors is None:
        eri_factors = getattr(getattr(mf, 'mol', None), 'eri_factors', None)
    return eri_factors


def _get_uhf_eri_factors(mf):
    eri_factors = getattr(mf, 'eri_factors', None)
    if eri_factors is None:
        eri_factors = getattr(getattr(mf, 'mol', None), 'eri_factors', None)
    return eri_factors


def _transform_eri_factors_to_mo_pair(eri_factors, mo_left, mo_right=None):
    if mo_right is None:
        mo_right = mo_left
    from pyqed.qchem.basis import transform_ri_factors_to_mo_pair
    return transform_ri_factors_to_mo_pair(eri_factors, mo_left, mo_right)


def _reference_density_rhf(mo_coeff, nocc):
    cocc = np.asarray(mo_coeff[:, :nocc])
    return 2.0 * cocc @ cocc.conj().T


def _reference_energy_rhf(hcore_ao, veff_ao, dm_ao, e_nuc):
    e1 = contract('pq,qp->', hcore_ao, dm_ao).real
    e2 = 0.5 * contract('pq,qp->', veff_ao, dm_ao).real
    return e_nuc + e1 + e2


def _orthogonalize_rotation(u, eps=1.0e-14):
    gram = u.conj().T @ u
    eigvals, eigvecs = np.linalg.eigh(gram)
    eigvals = np.clip(eigvals.real, eps, None)
    inv_sqrt = eigvecs @ np.diag(eigvals**-0.5) @ eigvecs.conj().T
    return u @ inv_sqrt


def _damp_rotation(u, alpha):
    eye = np.eye(u.shape[0], dtype=u.dtype)
    return _orthogonalize_rotation((1.0 - alpha) * eye + alpha * u)


def _semicanonicalize_rhf(mo_coeff, fock_mo, nocc):
    fock_mo = 0.5 * (np.asarray(fock_mo) + np.asarray(fock_mo).conj().T)
    foo = fock_mo[:nocc, :nocc]
    fvv = fock_mo[nocc:, nocc:]
    e_occ, u_occ = np.linalg.eigh(foo)
    e_vir, u_vir = np.linalg.eigh(fvv)
    u = np.eye(fock_mo.shape[0], dtype=fock_mo.dtype)
    u[:nocc, :nocc] = u_occ
    u[nocc:, nocc:] = u_vir
    mo_coeff_semi = np.asarray(mo_coeff) @ u
    fock_mo_semi = dag(u) @ fock_mo @ u
    mo_energy = np.concatenate((e_occ.real, e_vir.real))
    return mo_coeff_semi, fock_mo_semi, mo_energy, u


def _rmp2_kernel(nocc, nmo, mo_energy, eri, with_t2=True):
    nvir = nmo - nocc
    ovov = eri[:nocc, nocc:, :nocc, nocc:]
    gi = ovov.transpose(0, 2, 1, 3)
    eia = mo_energy[:nocc, None] - mo_energy[None, nocc:]
    denom = eia[:, None, :, None] + eia[None, :, None, :]
    t2 = gi.conj() / denom if with_t2 else None
    e_corr = (2 * contract('ijab,ijab->', t2 if t2 is not None else gi.conj() / denom, gi)
              - contract('ijab,ijba->', t2 if t2 is not None else gi.conj() / denom, gi)).real
    return e_corr, t2


def _rmp2_kernel_factors(nocc, nmo, mo_energy, pair_factors_ov, with_t2=True):
    eia = mo_energy[:nocc, None] - mo_energy[None, nocc:]
    denom = eia[:, None, :, None] + eia[None, :, None, :]
    gi = contract('Pia,Pjb->ijab', pair_factors_ov, pair_factors_ov)
    t2 = gi.conj() / denom if with_t2 else None
    amp = t2 if t2 is not None else gi.conj() / denom
    e_corr = (
        2 * contract('ijab,ijab->', amp, gi)
        - contract('ijab,ijba->', amp, gi)
    ).real
    return e_corr, t2


def _rmp2_gamma1_intermediates(t2):
    l2 = t2.conj()
    dm1vir = contract('ijac,ijbc->ab', l2, t2) * 2
    dm1vir -= contract('ijac,ijcb->ab', l2, t2)
    dm1occ = contract('ijab,ikab->jk', l2, t2) * 2
    dm1occ -= contract('ijab,ikba->jk', l2, t2)
    return -dm1occ, dm1vir


def _rmp2_make_rdm1(nocc, nmo, t2):
    nvir = nmo - nocc
    doo, dvv = _rmp2_gamma1_intermediates(t2)
    dm1 = np.zeros((nmo, nmo), dtype=np.result_type(t2))
    dm1[:nocc, :nocc] = doo + doo.conj().T
    dm1[nocc:, nocc:] = dvv + dvv.conj().T
    dm1[np.diag_indices(nocc)] += 2
    return dm1


def _rmp2_make_rdm2(nocc, nmo, t2):
    nvir = nmo - nocc
    dm1 = _rmp2_make_rdm1(nocc, nmo, t2).copy()
    dm1[np.diag_indices(nocc)] -= 2
    dm2 = np.zeros((nmo, nmo, nmo, nmo), dtype=np.result_type(t2))

    for i in range(nocc):
        t2i = t2[i]
        dovov = t2i.transpose(1, 0, 2) * 2 - t2i.transpose(2, 0, 1)
        dovov *= 2
        dm2[i, nocc:, :nocc, nocc:] = dovov
        dm2[nocc:, i, nocc:, :nocc] = dovov.conj().transpose(0, 2, 1)

    for i in range(nocc):
        dm2[i, i, :, :] += dm1.T * 2
        dm2[:, :, i, i] += dm1.T * 2
        dm2[:, i, i, :] -= dm1.T
        dm2[i, :, :, i] -= dm1

    for i in range(nocc):
        for j in range(nocc):
            dm2[i, i, j, j] += 4
            dm2[i, j, j, i] -= 2
    return dm2


def _ump2_kernel(nocc, nmo, mo_energy, eri, with_t2=True):
    nocca, noccb = nocc
    nmoa, nmob = nmo
    nvira, nvirb = nmoa - nocca, nmob - noccb
    mo_ea, mo_eb = mo_energy

    eia_a = mo_ea[:nocca, None] - mo_ea[None, nocca:]
    eia_b = mo_eb[:noccb, None] - mo_eb[None, noccb:]

    aa = eri[0, 0]
    ab = eri[0, 1]
    bb = eri[1, 1]

    gi_aa = aa[:nocca, nocca:, :nocca, nocca:].transpose(0, 2, 1, 3)
    gi_ab = ab[:nocca, nocca:, :noccb, noccb:].transpose(0, 2, 1, 3)
    gi_bb = bb[:noccb, noccb:, :noccb, noccb:].transpose(0, 2, 1, 3)

    denom_aa = (
        (mo_ea[:nocca, None] - mo_ea[None, nocca:])[:, None, :, None]
        + (mo_ea[:nocca, None] - mo_ea[None, nocca:])[None, :, None, :]
    )
    denom_ab = (
        (mo_ea[:nocca, None] - mo_ea[None, nocca:])[:, None, :, None]
        + (mo_eb[:noccb, None] - mo_eb[None, noccb:])[None, :, None, :]
    )
    denom_bb = (
        (mo_eb[:noccb, None] - mo_eb[None, noccb:])[:, None, :, None]
        + (mo_eb[:noccb, None] - mo_eb[None, noccb:])[None, :, None, :]
    )

    raw_aa = gi_aa.conj() / denom_aa
    raw_ab = gi_ab.conj() / denom_ab
    raw_bb = gi_bb.conj() / denom_bb

    emp2_ss = 0.5 * contract('ijab,ijab->', raw_aa, gi_aa).real
    emp2_ss -= 0.5 * contract('ijab,ijba->', raw_aa, gi_aa).real
    emp2_ss += 0.5 * contract('ijab,ijab->', raw_bb, gi_bb).real
    emp2_ss -= 0.5 * contract('ijab,ijba->', raw_bb, gi_bb).real
    emp2_os = contract('ijab,ijab->', raw_ab, gi_ab).real
    e_corr = emp2_ss + emp2_os

    if not with_t2:
        return e_corr, None

    t2aa = raw_aa - raw_aa.transpose(0, 1, 3, 2)
    t2ab = raw_ab
    t2bb = raw_bb - raw_bb.transpose(0, 1, 3, 2)
    return e_corr, (t2aa, t2ab, t2bb)


def _ump2_kernel_factors(nocc, nmo, mo_energy, pair_factors_ov_a, pair_factors_ov_b, with_t2=True):
    nocca, noccb = nocc
    mo_ea, mo_eb = mo_energy

    gi_aa = contract('Pia,Pjb->ijab', pair_factors_ov_a, pair_factors_ov_a)
    gi_ab = contract('Pia,Pjb->ijab', pair_factors_ov_a, pair_factors_ov_b)
    gi_bb = contract('Pia,Pjb->ijab', pair_factors_ov_b, pair_factors_ov_b)

    denom_aa = (
        (mo_ea[:nocca, None] - mo_ea[None, nocca:])[:, None, :, None]
        + (mo_ea[:nocca, None] - mo_ea[None, nocca:])[None, :, None, :]
    )
    denom_ab = (
        (mo_ea[:nocca, None] - mo_ea[None, nocca:])[:, None, :, None]
        + (mo_eb[:noccb, None] - mo_eb[None, noccb:])[None, :, None, :]
    )
    denom_bb = (
        (mo_eb[:noccb, None] - mo_eb[None, noccb:])[:, None, :, None]
        + (mo_eb[:noccb, None] - mo_eb[None, noccb:])[None, :, None, :]
    )

    raw_aa = gi_aa.conj() / denom_aa
    raw_ab = gi_ab.conj() / denom_ab
    raw_bb = gi_bb.conj() / denom_bb

    emp2_ss = 0.5 * contract('ijab,ijab->', raw_aa, gi_aa).real
    emp2_ss -= 0.5 * contract('ijab,ijba->', raw_aa, gi_aa).real
    emp2_ss += 0.5 * contract('ijab,ijab->', raw_bb, gi_bb).real
    emp2_ss -= 0.5 * contract('ijab,ijba->', raw_bb, gi_bb).real
    emp2_os = contract('ijab,ijab->', raw_ab, gi_ab).real
    e_corr = emp2_ss + emp2_os

    if not with_t2:
        return e_corr, None

    t2aa = raw_aa - raw_aa.transpose(0, 1, 3, 2)
    t2ab = raw_ab
    t2bb = raw_bb - raw_bb.transpose(0, 1, 3, 2)
    return e_corr, (t2aa, t2ab, t2bb)


def _ump2_gamma1_intermediates(t2):
    t2aa, t2ab, t2bb = t2
    dooa = contract('imef,jmef->ij', t2aa.conj(), t2aa) * -0.5
    dooa -= contract('imef,jmef->ij', t2ab.conj(), t2ab)
    doob = contract('imef,jmef->ij', t2bb.conj(), t2bb) * -0.5
    doob -= contract('mief,mjef->ij', t2ab.conj(), t2ab)

    dvva = contract('mnae,mnbe->ba', t2aa.conj(), t2aa) * 0.5
    dvva += contract('mnae,mnbe->ba', t2ab.conj(), t2ab)
    dvvb = contract('mnae,mnbe->ba', t2bb.conj(), t2bb) * 0.5
    dvvb += contract('mnea,mneb->ba', t2ab.conj(), t2ab)
    return (dooa, doob), (dvva, dvvb)


def _ump2_make_rdm1(nocc, nmo, t2):
    (dooa, doob), (dvva, dvvb) = _ump2_gamma1_intermediates(t2)
    nocca, noccb = nocc
    nmoa, nmob = nmo
    dm1a = np.zeros((nmoa, nmoa), dtype=np.result_type(t2[0], t2[1], t2[2]))
    dm1b = np.zeros((nmob, nmob), dtype=np.result_type(t2[0], t2[1], t2[2]))
    dm1a[:nocca, :nocca] = dooa + dooa.conj().T
    dm1a[nocca:, nocca:] = dvva + dvva.conj().T
    dm1b[:noccb, :noccb] = doob + doob.conj().T
    dm1b[noccb:, noccb:] = dvvb + dvvb.conj().T
    dm1a *= 0.5
    dm1b *= 0.5
    dm1a[np.diag_indices(nocca)] += 1
    dm1b[np.diag_indices(noccb)] += 1
    return dm1a, dm1b


def _ump2_make_rdm2(nocc, nmo, t2):
    nocca, noccb = nocc
    nmoa, nmob = nmo
    t2aa, t2ab, t2bb = t2

    dm2aa = np.zeros((nmoa, nmoa, nmoa, nmoa), dtype=np.result_type(t2aa, t2ab, t2bb))
    dm2ab = np.zeros((nmoa, nmoa, nmob, nmob), dtype=np.result_type(t2aa, t2ab, t2bb))
    dm2bb = np.zeros((nmob, nmob, nmob, nmob), dtype=np.result_type(t2aa, t2ab, t2bb))

    tmp = t2aa.transpose(0, 2, 1, 3)
    dm2aa[:nocca, nocca:, :nocca, nocca:] = tmp
    dm2aa[nocca:, :nocca, nocca:, :nocca] = tmp.conj().transpose(1, 0, 3, 2)

    tmp = t2bb.transpose(0, 2, 1, 3)
    dm2bb[:noccb, noccb:, :noccb, noccb:] = tmp
    dm2bb[noccb:, :noccb, noccb:, :noccb] = tmp.conj().transpose(1, 0, 3, 2)

    dm2ab[:nocca, nocca:, :noccb, noccb:] = t2ab.transpose(0, 2, 1, 3)
    dm2ab[nocca:, :nocca, noccb:, :noccb] = t2ab.transpose(2, 0, 3, 1).conj()

    dm1a, dm1b = _ump2_make_rdm1(nocc, nmo, t2)
    dm1a[np.diag_indices(nocca)] -= 1
    dm1b[np.diag_indices(noccb)] -= 1

    for i in range(nocca):
        dm2aa[i, i, :, :] += dm1a.T
        dm2aa[:, :, i, i] += dm1a.T
        dm2aa[:, i, i, :] -= dm1a.T
        dm2aa[i, :, :, i] -= dm1a
        dm2ab[i, i, :, :] += dm1b.T
    for i in range(noccb):
        dm2bb[i, i, :, :] += dm1b.T
        dm2bb[:, :, i, i] += dm1b.T
        dm2bb[:, i, i, :] -= dm1b.T
        dm2bb[i, :, :, i] -= dm1b
        dm2ab[:, :, i, i] += dm1a.T

    for i in range(nocca):
        for j in range(nocca):
            dm2aa[i, i, j, j] += 1
            dm2aa[i, j, j, i] -= 1
    for i in range(noccb):
        for j in range(noccb):
            dm2bb[i, i, j, j] += 1
            dm2bb[i, j, j, i] -= 1
    for i in range(nocca):
        for j in range(noccb):
            dm2ab[i, i, j, j] += 1

    return dm2aa, dm2ab, dm2bb


class MP2:
    def __init__(self, mf):
        if not isinstance(mf, RHF):
            raise TypeError('MP2 requires a pyqed RHF reference.')

        self.mf = mf
        self.nocc = mf.nocc
        self.nmo = mf.nmo
        self.nvir = mf.nvir

        self.mo_energy = np.asarray(mf.mo_energy)
        self.mo_coeff = np.asarray(mf.mo_coeff)
        self.mo_occ = np.asarray(mf.mo_occ)

        self.e_corr = None
        self.e_tot = None
        self.t2 = None
        self.eri_backend = None

    def run(self):
        eri_factors = _get_rhf_eri_factors(self.mf)
        if eri_factors is not None:
            cocc = self.mo_coeff[:, :self.nocc]
            cvir = self.mo_coeff[:, self.nocc:]
            pair_factors_ov = _transform_eri_factors_to_mo_pair(eri_factors, cocc, cvir)
            self.e_corr, self.t2 = _rmp2_kernel_factors(
                self.nocc,
                self.nmo,
                self.mo_energy,
                pair_factors_ov,
                with_t2=True,
            )
            self.eri_backend = 'factors'
        else:
            eri_mo = self.mf.get_eri_mo(notation='chem')
            self.e_corr, self.t2 = _rmp2_kernel(self.nocc, self.nmo, self.mo_energy, eri_mo, with_t2=True)
            self.eri_backend = 'dense'
        self.e_tot = self.e_corr + self.mf.e_tot
        return self

    def make_rdm1(self, ao_repr=False):
        if self.t2 is None:
            raise ValueError('Run MP2 before requesting RDMs.')
        dm1 = _rmp2_make_rdm1(self.nocc, self.nmo, self.t2)
        if ao_repr:
            return self.mo_coeff @ dm1 @ dag(self.mo_coeff)
        return dm1

    def make_rdm2(self, ao_repr=False):
        if self.t2 is None:
            raise ValueError('Run MP2 before requesting RDMs.')
        dm2 = _rmp2_make_rdm2(self.nocc, self.nmo, self.t2)
        if ao_repr:
            return _transform_rdm2_rhf_to_ao(dm2, self.mo_coeff)
        return dm2

    def make_rdm12(self, ao_repr=False):
        return self.make_rdm1(ao_repr=ao_repr), self.make_rdm2(ao_repr=ao_repr)


class COMP2(MP2):
    """
    Constrained orbital-relaxed MP2 via alternating MP2 and orbital updates.

    This is a macro-iterative approximation built on the fixed-RDM orbital
    objective in ``pyqed.optimize``.  It is not a fully variational OOMP2
    implementation: after each orbital update, the occupied and virtual
    subspaces are semicanonicalized separately and the MP2 amplitudes are
    rebuilt in that basis.
    """

    def __init__(
        self,
        mf,
        max_cycle=20,
        tol=1.0e-8,
        optimizer='RCG',
        optimizer_history=7,
        optimizer_tol=1.0e-5,
        optimizer_max_steps=100,
        optimizer_max_step_norm=None,
        macro_backtrack=0.5,
        macro_min_step=1.0e-3,
        macro_improvement_tol=1.0e-10,
    ):
        super().__init__(mf)
        self.max_cycle = int(max_cycle)
        self.tol = float(tol)
        self.optimizer = str(optimizer).upper()
        self.optimizer_history = int(optimizer_history)
        self.optimizer_tol = float(optimizer_tol)
        self.optimizer_max_steps = int(optimizer_max_steps)
        self.optimizer_max_step_norm = (
            None if optimizer_max_step_norm is None else float(optimizer_max_step_norm)
        )
        self.macro_backtrack = float(macro_backtrack)
        self.macro_min_step = float(macro_min_step)
        self.macro_improvement_tol = float(macro_improvement_tol)

        self.converged = False
        self.energy_history = []
        self.objective_history = []
        self.rotation_history = []
        self.step_history = []
        self.fock_mo = None
        self.dm = None
        self.semicanonical_transform = None
        self.eri_backend = None

    def _build_macro_state(self, mo_coeff):
        hcore_ao = np.asarray(self.mf.get_hcore())
        dm_ref = _reference_density_rhf(mo_coeff, self.nocc)
        veff_ao = np.asarray(self.mf.get_veff(dm_ref))
        fock_ao = hcore_ao + veff_ao
        fock_mo = dag(mo_coeff) @ fock_ao @ mo_coeff
        mo_coeff, fock_mo, mo_energy, u_semi = _semicanonicalize_rhf(mo_coeff, fock_mo, self.nocc)

        h1_mo = np.asarray(self.mf.get_hcore_mo(mo_coeff))
        eri_factors = _get_rhf_eri_factors(self.mf)
        if eri_factors is not None:
            pair_factors_mo = _transform_eri_factors_to_mo_pair(eri_factors, mo_coeff)
            pair_factors_ov = pair_factors_mo[:, :self.nocc, self.nocc:]
            e_corr, t2 = _rmp2_kernel_factors(
                self.nocc,
                self.nmo,
                mo_energy,
                pair_factors_ov,
                with_t2=True,
            )
            eri_repr = pair_factors_mo
            eri_backend = 'factors'
        else:
            eri_repr = np.asarray(self.mf.get_eri_mo(mo_coeff, notation='chem'))
            e_corr, t2 = _rmp2_kernel(self.nocc, self.nmo, mo_energy, eri_repr, with_t2=True)
            eri_backend = 'dense'
        dm1 = _rmp2_make_rdm1(self.nocc, self.nmo, t2)
        dm2 = _rmp2_make_rdm2(self.nocc, self.nmo, t2)
        e_ref = _reference_energy_rhf(hcore_ao, veff_ao, dm_ref, self.mf.e_nuc)

        return {
            'mo_coeff': np.asarray(mo_coeff),
            'h1_mo': h1_mo,
            'eri_mo': eri_repr,
            'dm_ref': dm_ref,
            'fock_mo': fock_mo,
            'mo_energy': mo_energy,
            'e_ref': e_ref,
            'e_corr': e_corr,
            'e_tot': e_ref + e_corr,
            't2': t2,
            'dm1': dm1,
            'dm2': dm2,
            'u_semi': u_semi,
            'eri_backend': eri_backend,
        }

    def run(self):
        mo_coeff = np.asarray(self.mo_coeff).copy()
        previous_energy = None
        final_state = None

        self.converged = False
        self.energy_history = []
        self.objective_history = []
        self.rotation_history = []
        self.step_history = []

        for _ in range(self.max_cycle):
            state = self._build_macro_state(mo_coeff)
            mo_coeff = state['mo_coeff']
            self.energy_history.append(float(state['e_tot']))

            if previous_energy is not None and abs(state['e_tot'] - previous_energy) < self.tol:
                self.converged = True
                final_state = state
                break

            U0 = np.eye(self.nmo)
            U, objective = orbital_minimize(
                orbital_objective,
                U0,
                args=(state['h1_mo'], state['eri_mo'], state['dm1'], state['dm2']),
                algorithm=self.optimizer,
                history_size=self.optimizer_history,
                epsilon=self.optimizer_tol,
                max_iterations=self.optimizer_max_steps,
                max_step_norm=self.optimizer_max_step_norm,
            )
            self.objective_history.append(float(np.real(objective)))
            alpha = 1.0
            accepted_rotation = None
            accepted_state = None
            accepted_alpha = 0.0
            while alpha >= self.macro_min_step:
                trial_u = _damp_rotation(U, alpha)
                trial_coeff = mo_coeff @ trial_u
                trial_state = self._build_macro_state(trial_coeff)
                if trial_state['e_tot'] <= state['e_tot'] - self.macro_improvement_tol:
                    accepted_rotation = trial_u
                    accepted_state = trial_state
                    accepted_alpha = alpha
                    break
                alpha *= self.macro_backtrack

            if accepted_rotation is None:
                final_state = state
                self.rotation_history.append(0.0)
                self.step_history.append(0.0)
                break

            self.rotation_history.append(float(np.linalg.norm(accepted_rotation - np.eye(self.nmo))))
            self.step_history.append(float(accepted_alpha))
            mo_coeff = accepted_state['mo_coeff']
            previous_energy = state['e_tot']
            final_state = accepted_state

        final_state = self._build_macro_state(mo_coeff)
        if not self.energy_history or abs(final_state['e_tot'] - self.energy_history[-1]) > 1.0e-14:
            self.energy_history.append(float(final_state['e_tot']))
        self.mo_coeff = final_state['mo_coeff']
        self.mo_energy = final_state['mo_energy']
        self.fock_mo = final_state['fock_mo']
        self.dm = final_state['dm_ref']
        self.semicanonical_transform = final_state['u_semi']
        self.t2 = final_state['t2']
        self.eri_backend = final_state['eri_backend']
        self.e_corr = final_state['e_corr']
        self.e_tot = final_state['e_tot']
        return self


class UMP2:
    def __init__(self, mf):
        if not isinstance(mf, UHF):
            raise TypeError('UMP2 requires a pyqed UHF reference.')

        self.mf = mf
        self.nocc = mf.nocc
        self.nmo = (mf.mo_coeff[0].shape[1], mf.mo_coeff[1].shape[1])
        self.nvir = mf.nvir

        self.mo_energy = tuple(np.asarray(x) for x in mf.mo_energy)
        self.mo_coeff = tuple(np.asarray(x) for x in mf.mo_coeff)
        self.mo_occ = tuple(np.asarray(x) for x in mf.mo_occ)

        self.e_corr = None
        self.e_tot = None
        self.t2 = None
        self.eri_backend = None

    def run(self):
        eri_factors = _get_uhf_eri_factors(self.mf)
        if eri_factors is not None:
            ca, cb = self.mo_coeff
            nocca, noccb = self.nocc
            pair_factors_ov_a = _transform_eri_factors_to_mo_pair(
                eri_factors,
                ca[:, :nocca],
                ca[:, nocca:],
            )
            pair_factors_ov_b = _transform_eri_factors_to_mo_pair(
                eri_factors,
                cb[:, :noccb],
                cb[:, noccb:],
            )
            self.e_corr, self.t2 = _ump2_kernel_factors(
                self.nocc,
                self.nmo,
                self.mo_energy,
                pair_factors_ov_a,
                pair_factors_ov_b,
                with_t2=True,
            )
            self.eri_backend = 'factors'
        else:
            eri_mo = self.mf.get_eri_mo(notation='chem')
            self.e_corr, self.t2 = _ump2_kernel(self.nocc, self.nmo, self.mo_energy, eri_mo, with_t2=True)
            self.eri_backend = 'dense'
        self.e_tot = self.e_corr + self.mf.e_tot
        return self

    def make_rdm1(self, ao_repr=False):
        if self.t2 is None:
            raise ValueError('Run UMP2 before requesting RDMs.')
        dm1a, dm1b = _ump2_make_rdm1(self.nocc, self.nmo, self.t2)
        if ao_repr:
            ca, cb = self.mo_coeff
            dm1a = ca @ dm1a @ dag(ca)
            dm1b = cb @ dm1b @ dag(cb)
        return dm1a, dm1b

    def make_rdm2(self, ao_repr=False):
        if self.t2 is None:
            raise ValueError('Run UMP2 before requesting RDMs.')
        dm2aa, dm2ab, dm2bb = _ump2_make_rdm2(self.nocc, self.nmo, self.t2)
        if ao_repr:
            return _transform_rdm2_uhf_to_ao(dm2aa, dm2ab, dm2bb, self.mo_coeff)
        return dm2aa, dm2ab, dm2bb

    def make_rdm12(self, ao_repr=False):
        return self.make_rdm1(ao_repr=ao_repr), self.make_rdm2(ao_repr=ao_repr)
