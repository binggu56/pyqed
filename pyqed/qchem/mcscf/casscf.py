#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Native first-order CASSCF built on top of the existing CASCI solvers.

This module lives beside the original constrained-optimization implementation
now exposed as ``pyqed.qchem.COCASCI``.
"""

import copy

import numpy as np

from .direct_ci import CASCI
from .orbopt import (
    diagonal_inverse_hessian,
    diagonal_preconditioned_vector,
    embed_rdm2,
    generalized_fock,
    gradient_norm,
    lbfgs_direction,
    limit_step_norm,
    orbital_step,
    pack_nonredundant,
    rotate_orbitals,
    unpack_nonredundant,
    update_lbfgs_history,
)


class OrbitalDIIS:
    """Pulay extrapolation for anti-Hermitian orbital-rotation generators."""

    def __init__(self, max_space=6, start=2, regularization=1.0e-10):
        self.max_space = int(max_space)
        self.start = int(start)
        self.regularization = float(regularization)
        self.vectors = []
        self.errors = []

    def update(self, vector, error):
        self.vectors.append(np.array(vector, copy=True))
        self.errors.append(np.array(error, copy=True))

        if len(self.vectors) > self.max_space:
            self.vectors.pop(0)
            self.errors.pop(0)

        if len(self.errors) < self.start:
            return vector

        bsize = len(self.errors)
        bmat = -1.0 * np.ones((bsize + 1, bsize + 1), dtype=float)
        rhs = np.zeros(bsize + 1, dtype=float)
        bmat[bsize, bsize] = 0.0
        rhs[bsize] = -1.0

        for i in range(bsize):
            for j in range(bsize):
                bmat[i, j] = np.vdot(self.errors[i], self.errors[j]).real

        try:
            coeff = np.linalg.solve(bmat, rhs)
        except np.linalg.LinAlgError:
            try:
                bmat[:-1, :-1] += np.eye(bsize) * self.regularization
                coeff = np.linalg.solve(bmat, rhs)
            except np.linalg.LinAlgError:
                return vector

        mixed = np.zeros_like(vector, dtype=vector.dtype)
        for weight, stored in zip(coeff[:-1], self.vectors):
            mixed += weight * stored

        # Preserve the orbital-generator manifold after the linear combination.
        return 0.5 * (mixed - mixed.conj().T)


class CASSCF:
    """
    State-specific first-order CASSCF with a diagonal-preconditioned orbital step.

    Notes
    -----
    The implementation is intentionally lightweight:

    - reuses the existing native CASCI solver for the active-space problem
    - optimizes orbitals by repeated CASCI + generalized-Fock updates
    - uses a simple backtracking line search over orbital-rotation steps

    It is meant to complement the original constrained-optimization
    implementation now exposed as ``COCASCI``.
    """

    def __init__(
        self,
        mf,
        ncas,
        nelecas,
        max_cycle=12,
        conv_tol=1.0e-7,
        conv_tol_grad=1.0e-5,
        conv_tol_grad_relaxed=1.0e-3,
        conv_tol_step=1.0e-3,
        level_shift=1.0e-3,
        step_size=0.5,
        max_step=0.25,
        optimizer="DIAG",
        optimizer_history=7,
        diis=True,
        diis_space=6,
        diis_start=2,
        ci_method="direct_ci",
    ):
        self.mf = mf
        self.mol = mf.mol
        self.ncas = int(ncas)
        self.nelecas = nelecas
        self.max_cycle = int(max_cycle)
        self.conv_tol = float(conv_tol)
        self.conv_tol_grad = float(conv_tol_grad)
        self.conv_tol_grad_relaxed = float(conv_tol_grad_relaxed)
        self.conv_tol_step = float(conv_tol_step)
        self.level_shift = float(level_shift)
        self.step_size = float(step_size)
        self.max_step = float(max_step)
        self.optimizer = str(optimizer).upper()
        self.optimizer_history = int(optimizer_history)
        self.diis = bool(diis)
        self.diis_space = int(diis_space)
        self.diis_start = int(diis_start)
        self.ci_method = ci_method
        if self.optimizer not in {"DIAG", "LBFGS"}:
            raise ValueError(
                "Unknown orbital optimizer '{}'. Use 'DIAG' or 'LBFGS'.".format(
                    self.optimizer
                )
            )

        self.nmo = getattr(mf, "nmo", None)
        if self.nmo is None and hasattr(mf.mol, "nao"):
            self.nmo = mf.mol.nao
        if self.nmo is None:
            raise ValueError("Could not determine the number of molecular orbitals.")

        self.ncore = None
        self.nstates = None
        self.state_id = 0
        self.weights = None
        self.mo_coeff = None
        self.e_tot = None
        self.ci = None
        self.history = []
        self.converged = False
        self.casci = None
        self.orbital_diis = None
        self.lbfgs_s = []
        self.lbfgs_y = []
        self.spin_purification = False
        self.ss = None
        self.shift = None

    @staticmethod
    def _copy_ci_guess(ci):
        if ci is None:
            return None
        return copy.deepcopy(ci)

    def state_average(self, weights):
        weights = np.asarray(weights, dtype=float)
        if weights.ndim != 1 or weights.size == 0:
            raise ValueError("weights must be a non-empty 1D array.")
        if np.any(weights < 0):
            raise ValueError("state-average weights must be non-negative.")
        total = float(np.sum(weights))
        if total <= 0.0:
            raise ValueError("state-average weights must sum to a positive value.")
        self.weights = weights / total
        return self

    def fix_spin(self, s=None, ss=0, shift=0.2):
        probe = CASCI(self.mf, ncas=self.ncas, nelecas=self.nelecas)
        probe.fix_spin(s=s, ss=ss, shift=shift)
        self.spin_purification = probe.spin_purification
        self.ss = probe.ss
        self.shift = probe.shift
        return self

    def _make_casci(self, mo_coeff, nstates, ci0=None):
        mc = CASCI(self.mf, ncas=self.ncas, nelecas=self.nelecas)
        if self.spin_purification:
            mc.spin_purification = self.spin_purification
            mc.ss = self.ss
            mc.shift = self.shift
        mc.run(nstates=nstates, mo_coeff=mo_coeff, method=self.ci_method, ci0=ci0)
        self.ncore = mc.ncore
        return mc

    def _get_integrals(self, mo_coeff):
        if not hasattr(self.mf, "get_hcore_mo") or not hasattr(self.mf, "get_eri_mo"):
            raise NotImplementedError(
                "CASSCF currently requires a reference with "
                "`get_hcore_mo()` and `get_eri_mo()` methods."
            )
        h1_mo = self.mf.get_hcore_mo(mo_coeff)
        eri_mo = self.mf.get_eri_mo(mo_coeff, notation="chem")
        return h1_mo, eri_mo

    def _effective_rdms(self, mc, state_id):
        if self.weights is None:
            dm1 = mc.make_rdm1(
                state_id,
                with_core=True,
                with_vir=True,
                representation="mo",
            )
            dm2_small = mc.make_rdm2(state_id, with_core=True)
            return dm1, embed_rdm2(dm2_small, self.nmo)

        dm1 = np.zeros((self.nmo, self.nmo), dtype=float)
        dm2 = np.zeros((self.nmo, self.nmo, self.nmo, self.nmo), dtype=float)
        for root, weight in enumerate(self.weights):
            dm1 += weight * mc.make_rdm1(
                root,
                with_core=True,
                with_vir=True,
                representation="mo",
            )
            dm2_small = mc.make_rdm2(root, with_core=True)
            dm2 += weight * embed_rdm2(dm2_small, self.nmo)
        return dm1, dm2

    def _objective_energy(self, mc, state_id):
        if self.weights is None:
            return float(np.real(mc.e_tot[state_id]))
        return float(np.real(np.dot(self.weights, mc.e_tot)))

    def _evaluate(self, mo_coeff, nstates, state_id, ci0=None):
        mc = self._make_casci(mo_coeff, nstates=nstates, ci0=ci0)
        dm1, dm2 = self._effective_rdms(mc, state_id)
        h1_mo, eri_mo = self._get_integrals(mo_coeff)
        fock = generalized_fock(h1_mo, eri_mo, dm1, dm2)
        grad = orbital_step(
            fock,
            mc.ncore,
            mc.ncas,
            step_size=self.step_size,
            level_shift=self.level_shift,
            max_step=self.max_step,
        )[1]
        return mc, fock, grad

    def _line_search(self, mo_coeff, kappa, energy, ci0=None):
        scale = float(self.step_size)
        min_scale = max(1.0e-6, 1.0e-4 * abs(self.step_size))
        best = None
        guess = self._copy_ci_guess(ci0)

        while scale >= min_scale:
            trial_coeff = rotate_orbitals(mo_coeff, scale * kappa)
            trial_mc = self._make_casci(trial_coeff, nstates=self.nstates, ci0=guess)
            trial_energy = self._objective_energy(trial_mc, self.state_id)
            guess = self._copy_ci_guess(trial_mc.ci)

            if best is None or trial_energy < best[1]:
                best = (trial_coeff, trial_energy, scale, trial_mc)

            if trial_energy < energy - 1.0e-9:
                return True, trial_coeff, trial_energy, scale, trial_mc

            scale *= 0.5

        if best is None:
            return False, mo_coeff, energy, 0.0, None
        return False, best[0], best[1], best[2], best[3]

    def run(self, nstates=1, state_id=0, mo_coeff=None):
        if isinstance(self.mf.mo_coeff, tuple):
            raise NotImplementedError(
                "CASSCF currently supports restricted references only."
            )

        if self.weights is not None:
            if nstates == 1:
                nstates = len(self.weights)
            elif int(nstates) != len(self.weights):
                raise ValueError(
                    "nstates={} is inconsistent with {} state-average weights.".format(
                        nstates, len(self.weights)
                    )
                )
        self.nstates = int(nstates)
        self.state_id = int(state_id)
        # ``run()`` should behave like a fresh calculation even when the same
        # object is reused, so clear any cached convergence state from a
        # previous attempt before rebuilding the macroiteration history.
        self.history = []
        self.converged = False
        self.casci = None
        self.mo_coeff = None
        self.e_tot = None
        self.ci = None
        self.orbital_diis = (
            OrbitalDIIS(max_space=self.diis_space, start=self.diis_start)
            if self.diis else None
        )
        self.lbfgs_s = []
        self.lbfgs_y = []
        if mo_coeff is None:
            mo_coeff = np.array(self.mf.mo_coeff, copy=True)
        else:
            mo_coeff = np.array(mo_coeff, copy=True)
        prev_energy = None
        prev_step_norm = None
        ci_guess = None
        prev_grad_vec = None
        accepted_step_vec = None

        for cycle in range(1, self.max_cycle + 1):
            mc, fock, grad = self._evaluate(
                mo_coeff,
                self.nstates,
                self.state_id,
                ci0=ci_guess,
            )
            energy = self._objective_energy(mc, self.state_id)
            gnorm = gradient_norm(grad, mc.ncore, mc.ncas, self.nmo)
            grad_vec = pack_nonredundant(grad, mc.ncore, mc.ncas, self.nmo)
            if (
                self.optimizer == "LBFGS"
                and accepted_step_vec is not None
                and prev_grad_vec is not None
                and len(accepted_step_vec) == len(grad_vec)
            ):
                update_lbfgs_history(
                    self.lbfgs_s,
                    self.lbfgs_y,
                    accepted_step_vec,
                    grad_vec - prev_grad_vec,
                    self.optimizer_history,
                )
                accepted_step_vec = None
            self.history.append(
                {
                    "cycle": cycle,
                    "energy": energy,
                    "gradient_norm": gnorm,
                    "step_norm": prev_step_norm,
                }
            )

            if (
                prev_energy is not None
                and abs(energy - prev_energy) < self.conv_tol
                and (
                    gnorm < self.conv_tol_grad
                    or (
                        gnorm < self.conv_tol_grad_relaxed
                        and (
                            prev_step_norm is None
                            or prev_step_norm < self.max_step
                        )
                    )
                )
            ):
                self.converged = True
                self.casci = mc
                break

            kappa, _ = orbital_step(
                fock,
                mc.ncore,
                mc.ncas,
                step_size=1.0,
                level_shift=self.level_shift,
                max_step=self.max_step,
            )
            if self.optimizer == "LBFGS":
                if len(grad_vec) > 0:
                    diag_step = diagonal_preconditioned_vector(
                        grad,
                        fock,
                        mc.ncore,
                        mc.ncas,
                        level_shift=self.level_shift,
                    )
                    h0_diag = diagonal_inverse_hessian(
                        fock,
                        mc.ncore,
                        mc.ncas,
                        level_shift=self.level_shift,
                    )
                    if self.lbfgs_s:
                        step_vec = -lbfgs_direction(
                            grad_vec,
                            self.lbfgs_s,
                            self.lbfgs_y,
                            h0_diag=h0_diag,
                        )
                    else:
                        step_vec = diag_step
                    if np.dot(step_vec, grad_vec) >= 0.0:
                        step_vec = diag_step
                    step_vec = limit_step_norm(step_vec, self.max_step)
                    kappa = unpack_nonredundant(
                        step_vec,
                        mc.ncore,
                        mc.ncas,
                        self.nmo,
                        max_step=self.max_step,
                    )
                else:
                    step_vec = np.zeros(0, dtype=float)
            else:
                step_vec = pack_nonredundant(kappa, mc.ncore, mc.ncas, self.nmo)
            kappa_diis = None
            if self.orbital_diis is not None:
                kappa_diis = self.orbital_diis.update(kappa, grad)

            accepted, trial_coeff, _, _, trial_mc = self._line_search(
                mo_coeff,
                kappa_diis if kappa_diis is not None else kappa,
                energy,
                ci0=mc.ci,
            )
            if (
                not accepted
                and kappa_diis is not None
                and not np.allclose(kappa_diis, kappa)
            ):
                accepted, trial_coeff, _, _, trial_mc = self._line_search(
                    mo_coeff,
                    kappa,
                    energy,
                    ci0=mc.ci,
                )

            self.casci = mc
            if accepted:
                mo_coeff = trial_coeff
                prev_energy = energy
                ci_guess = self._copy_ci_guess(trial_mc.ci)
                accepted_step_vec = step_vec.copy()
                prev_step_norm = float(np.max(np.abs(step_vec))) if len(step_vec) > 0 else 0.0
            else:
                ci_guess = self._copy_ci_guess(mc.ci)
                if trial_mc is not None:
                    self.casci = trial_mc
                    ci_guess = self._copy_ci_guess(trial_mc.ci)
                if gnorm < self.conv_tol_grad:
                    self.converged = True
                    mo_coeff = self.casci.mo_coeff
                    break
                raise RuntimeError(
                    "CASSCF orbital line search failed before reaching the "
                    "gradient tolerance."
                )
            prev_grad_vec = grad_vec.copy()

        if not self.converged:
            raise RuntimeError("Max macro steps reached. CASSCF not converged.")

        if self.casci is None or not np.allclose(mo_coeff, self.casci.mo_coeff):
            self.casci = self._make_casci(mo_coeff, nstates=self.nstates, ci0=ci_guess)

        self.mo_coeff = self.casci.mo_coeff
        self.ci = self.casci.ci
        self.e_tot = self.casci.e_tot
        self.ncore = self.casci.ncore
        return self

    def spin_square(self, state_id=0):
        if self.casci is None:
            raise ValueError("Run CASSCF before requesting spin diagnostics.")
        return self.casci.spin_square(state_id)

    def make_rdm1(self, state_id=0, **kwargs):
        if self.casci is None:
            raise ValueError("Run CASSCF before requesting RDMs.")
        return self.casci.make_rdm1(state_id, **kwargs)

    def make_rdm2(self, state_id=0, **kwargs):
        if self.casci is None:
            raise ValueError("Run CASSCF before requesting RDMs.")
        return self.casci.make_rdm2(state_id, **kwargs)


# Backward-compatible alias for the earlier explicit name.
FirstOrderCASSCF = CASSCF
