#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Native first-order CASSCF built on top of the existing CASCI solvers.

This module lives beside the original constrained-optimization implementation
now exposed as ``pyqed.qchem.COCAS``.
"""

import copy
import math

import numpy as np
from scipy.linalg import expm

from .casci import (
    _get_mf_cholesky_factors,
    _resolve_use_cholesky_integrals,
    transform_eri_factors_to_mo_pair,
)
from .direct_ci import CASCI
from .orbopt import (
    augmented_hessian_direction,
    davidson_augmented_hessian_direction,
    diagonal_hessian,
    diagonal_inverse_hessian,
    diagonal_preconditioned_vector,
    embed_rdm2,
    generalized_fock,
    generalized_fock_from_factors,
    gradient_norm,
    lbfgs_direction,
    limit_step_norm,
    orbital_gradient,
    orbital_step,
    orbital_hessian_action_from_integrals,
    pack_nonredundant,
    quadratic_model_change,
    rotate_orbitals,
    unpack_nonredundant,
    update_lbfgs_history,
)
from .reduced_ci import ReducedCISubspace, _transition_rdms_with_core, ci_diagonal


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
    State-specific native CASSCF with pluggable orbital optimizers.

    Notes
    -----
    The implementation is intentionally lightweight:

    - reuses the existing native CASCI solver for the active-space problem
    - optimizes orbitals by repeated CASCI + generalized-Fock updates
    - uses a simple backtracking line search over orbital-rotation steps

    It is meant to complement the original constrained-optimization
    implementation now exposed as ``COCAS``.
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
        ah_max_cycle=4,
        ah_max_subspace=8,
        ah_dense_threshold=0,
        ah_fd_step=1.0e-3,
        ah_hessian="analytic",
        ci_method="direct_ci",
        use_cholesky=None,
        max_cycles=None,
    ):
        if max_cycles is not None:
            if int(max_cycle) != 12 and int(max_cycle) != int(max_cycles):
                raise ValueError(
                    "Received conflicting values for max_cycle={} and "
                    "max_cycles={}.".format(max_cycle, max_cycles)
                )
            max_cycle = max_cycles

        self.mf = mf
        self.mol = mf.mol
        self.ncas = int(ncas)
        self.nelecas = nelecas
        self.max_cycle = int(max_cycle)
        self.max_cycles = self.max_cycle
        self.conv_tol = float(conv_tol)
        self.conv_tol_grad = float(conv_tol_grad)
        self.conv_tol_grad_relaxed = float(conv_tol_grad_relaxed)
        self.conv_tol_step = float(conv_tol_step)
        self.level_shift = float(level_shift)
        self.step_size = float(step_size)
        self.max_step = float(max_step)
        self.optimizer = str(optimizer).upper().replace("-", "_")
        if self.optimizer == "AUGMENTED_HESSIAN":
            self.optimizer = "AH"
        self.optimizer_history = int(optimizer_history)
        self.diis = bool(diis)
        self.diis_space = int(diis_space)
        self.diis_start = int(diis_start)
        self.ah_max_cycle = int(ah_max_cycle)
        self.ah_max_subspace = int(ah_max_subspace)
        self.ah_dense_threshold = int(ah_dense_threshold)
        self.ah_fd_step = float(ah_fd_step)
        self.ah_hessian = str(ah_hessian).lower().replace("-", "_")
        self.ci_method = ci_method
        self.use_cholesky = use_cholesky
        if self.ah_hessian not in {"analytic", "finite_difference"}:
            raise ValueError(
                "Unknown ah_hessian '{}'. Use 'analytic' or 'finite_difference'.".format(
                    ah_hessian
                )
            )
        if self.optimizer not in {"DIAG", "LBFGS", "AH"}:
            raise ValueError(
                "Unknown orbital optimizer '{}'. Use 'DIAG', 'LBFGS', or 'AH'.".format(
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
        self.use_cholesky_integrals = False
        self._casci_binary_cache = None
        self._casci_direct_connectivity_cache = None
        self._casci_sc1_cache = None
        self._casci_sc2_cache = None
        self._ah_trust_radius = self.max_step
        self._ah_reference_cache = None
        self.ci_root_tracking = False
        self.ci_root_cushion = 3

    def _casci_ndet(self):
        if isinstance(self.nelecas, (tuple, list)):
            na, nb = self.nelecas
        else:
            if int(self.nelecas) % 2:
                raise ValueError(
                    "Closed-shell CASSCF needs an even active electron count or "
                    "explicit (nalpha, nbeta)."
                )
            na = nb = int(self.nelecas) // 2
        return math.comb(self.ncas, int(na)) * math.comb(self.ncas, int(nb))

    def _active_nelectron_count(self):
        if isinstance(self.nelecas, (tuple, list)):
            return int(self.nelecas[0]) + int(self.nelecas[1])
        return int(self.nelecas)

    def _reference_nelectron_count(self):
        nelec = getattr(self.mf, "nelec", None)
        if nelec is not None:
            if isinstance(nelec, (tuple, list)):
                return int(nelec[0]) + int(nelec[1])
            return int(nelec)
        if getattr(self.mf, "mo_occ", None) is None:
            raise ValueError("Cannot infer the number of reference electrons.")
        return int(round(float(np.sum(self.mf.mo_occ))))

    def _default_ncore(self):
        ncore2 = self._reference_nelectron_count() - self._active_nelectron_count()
        if ncore2 < 0 or ncore2 % 2:
            raise ValueError(
                "Inconsistent reference/active electron counts for restricted CASSCF."
            )
        return ncore2 // 2

    def reorder_mo_for_active_orbitals(self, mo_coeff, active_orbitals):
        """
        Move selected original MO columns into the active block.

        ``active_orbitals`` are zero-based indices in the supplied ``mo_coeff``.
        The remaining orbitals keep their original order, with the first
        ``ncore`` columns used as doubly occupied core orbitals.
        """
        if active_orbitals is None:
            return np.array(mo_coeff, copy=True)
        active = [int(i) for i in active_orbitals]
        if len(active) != self.ncas:
            raise ValueError(
                "active_orbitals must contain exactly ncas={} entries.".format(
                    self.ncas
                )
            )
        if len(set(active)) != len(active):
            raise ValueError("active_orbitals contains duplicate indices.")
        nmo = np.asarray(mo_coeff).shape[1]
        if min(active) < 0 or max(active) >= nmo:
            raise ValueError("active_orbitals contains an out-of-range MO index.")
        ncore = self._default_ncore()
        rest = [idx for idx in range(nmo) if idx not in set(active)]
        if len(rest) < ncore:
            raise ValueError("Not enough remaining orbitals to form the core block.")
        order = rest[:ncore] + active + rest[ncore:]
        return np.array(mo_coeff[:, order], copy=True)

    def _ci_tracking_nstates(self, nstates, ci0):
        nstates = int(nstates)
        if (
            not self.ci_root_tracking
            or self.weights is not None
            or ci0 is None
            or nstates < 1
        ):
            return nstates
        return min(self._casci_ndet(), max(nstates, nstates + int(self.ci_root_cushion)))

    def _reorder_tracked_ci_root(self, mc, requested_nstates, ci0):
        if (
            not self.ci_root_tracking
            or self.weights is not None
            or ci0 is None
            or requested_nstates < 1
            or len(mc.ci) <= requested_nstates
        ):
            return mc

        target_id = min(self.state_id, len(ci0) - 1) if isinstance(ci0, (list, tuple)) else 0
        target = np.asarray(ci0[target_id] if isinstance(ci0, (list, tuple)) else ci0)
        if target.ndim != 1 or target.shape[0] != len(mc.ci[0]):
            return mc

        overlaps = np.array([abs(np.vdot(target, root)) for root in mc.ci], dtype=float)
        selected = int(np.argmax(overlaps))
        remaining = [idx for idx in range(len(mc.ci)) if idx != selected]
        remaining.sort(key=lambda idx: float(np.real(mc.e_tot[idx])))
        order = [selected] + remaining
        order = order[: int(requested_nstates)]

        mc.e_tot = np.asarray(mc.e_tot)[order]
        mc.ci = [mc.ci[idx] for idx in order]
        mc.nstates = int(requested_nstates)
        return mc

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
        if self._casci_binary_cache is not None:
            mc.binary = self._casci_binary_cache
        if self._casci_direct_connectivity_cache is not None:
            mc.direct_connectivity = self._casci_direct_connectivity_cache
        if self._casci_sc1_cache is not None and self._casci_sc2_cache is not None:
            mc.SC1 = self._casci_sc1_cache
            mc.SC2 = self._casci_sc2_cache
        if self.spin_purification:
            mc.spin_purification = self.spin_purification
            mc.ss = self.ss
            mc.shift = self.shift
        requested_nstates = int(nstates)
        solve_nstates = self._ci_tracking_nstates(requested_nstates, ci0)
        mc.run(
            nstates=solve_nstates,
            mo_coeff=mo_coeff,
            method=self.ci_method,
            ci0=ci0,
            use_cholesky=self.use_cholesky_integrals,
        )
        self._reorder_tracked_ci_root(mc, requested_nstates, ci0)
        self.ncore = mc.ncore
        self._update_casci_cache(mc)
        return mc

    def _update_casci_cache(self, mc):
        if getattr(mc, "binary", None) is not None:
            self._casci_binary_cache = mc.binary
        if getattr(mc, "direct_connectivity", None) is not None:
            self._casci_direct_connectivity_cache = mc.direct_connectivity
        if getattr(mc, "SC1", None) is not None and getattr(mc, "SC2", None) is not None:
            self._casci_sc1_cache = mc.SC1
            self._casci_sc2_cache = mc.SC2

    def _resolve_use_cholesky(self, use_cholesky=None):
        if use_cholesky is None:
            use_cholesky = self.use_cholesky
        if use_cholesky is None:
            use_cholesky = bool(getattr(self.mf, "cholesky_jk", False))
        return _resolve_use_cholesky_integrals(self.mf, use_cholesky)

    def _get_integrals(self, mo_coeff):
        if not hasattr(self.mf, "get_hcore_mo") or not hasattr(self.mf, "get_eri_mo"):
            raise NotImplementedError(
                "CASSCF currently requires a reference with "
                "`get_hcore_mo()` and `get_eri_mo()` methods."
            )
        h1_mo = self.mf.get_hcore_mo(mo_coeff)
        eri_mo = self.mf.get_eri_mo(mo_coeff, notation="chem")
        return h1_mo, eri_mo

    def _get_hcore_mo(self, mo_coeff):
        if not hasattr(self.mf, "get_hcore_mo"):
            raise NotImplementedError(
                "CASSCF currently requires a reference with `get_hcore_mo()`."
            )
        return self.mf.get_hcore_mo(mo_coeff)

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

    def _effective_rdms_occ(self, mc, state_id):
        if self.weights is None:
            dm1 = mc.make_rdm1(state_id, with_core=True)
            dm2 = mc.make_rdm2(state_id, with_core=True)
            self._update_casci_cache(mc)
            return dm1, dm2

        nocc_like = mc.ncore + mc.ncas
        dm1 = np.zeros((nocc_like, nocc_like), dtype=float)
        dm2 = np.zeros((nocc_like, nocc_like, nocc_like, nocc_like), dtype=float)
        for root, weight in enumerate(self.weights):
            dm1 += weight * mc.make_rdm1(root, with_core=True)
            dm2 += weight * mc.make_rdm2(root, with_core=True)
        self._update_casci_cache(mc)
        return dm1, dm2

    def _objective_energy(self, mc, state_id):
        if self.weights is None:
            return float(np.real(mc.e_tot[state_id]))
        return float(np.real(np.dot(self.weights, mc.e_tot)))

    def _invalidate_ah_reference_cache(self):
        self._ah_reference_cache = None

    def _set_ah_reference_data(self, mo_coeff, mc, dm1, dm2, h1_mo, eri_mo):
        if self.ah_hessian != "analytic" or self.use_cholesky_integrals:
            return
        self._ah_reference_cache = {
            "mo_coeff_ref": mo_coeff,
            "mc_ref": mc,
            "dm1": dm1,
            "dm2": dm2,
            "h1_mo": h1_mo,
            "eri_mo": eri_mo,
        }

    def _get_ah_reference_data(self, mo_coeff, mc):
        if self.ah_hessian != "analytic" or self.use_cholesky_integrals:
            return None

        cache = self._ah_reference_cache
        if (
            cache is not None
            and cache["mo_coeff_ref"] is mo_coeff
            and cache["mc_ref"] is mc
        ):
            return cache

        dm1, dm2 = self._effective_rdms(mc, self.state_id)
        h1_mo, eri_mo = self._get_integrals(mo_coeff)
        cache = {
            "mo_coeff_ref": mo_coeff,
            "mc_ref": mc,
            "dm1": dm1,
            "dm2": dm2,
            "h1_mo": h1_mo,
            "eri_mo": eri_mo,
        }
        self._ah_reference_cache = cache
        return cache

    def _evaluate(self, mo_coeff, nstates, state_id, ci0=None):
        mc = self._make_casci(mo_coeff, nstates=nstates, ci0=ci0)
        if self.use_cholesky_integrals:
            dm1_occ, dm2_occ = self._effective_rdms_occ(mc, state_id)
            h1_mo = self._get_hcore_mo(mo_coeff)
            occ_mo = mo_coeff[:, :mc.ncore + mc.ncas]
            pair_factors = transform_eri_factors_to_mo_pair(
                _get_mf_cholesky_factors(self.mf),
                mo_coeff,
                occ_mo,
            )
            fock = generalized_fock_from_factors(h1_mo, pair_factors, dm1_occ, dm2_occ)
        else:
            dm1, dm2 = self._effective_rdms(mc, state_id)
            h1_mo, eri_mo = self._get_integrals(mo_coeff)
            self._set_ah_reference_data(mo_coeff, mc, dm1, dm2, h1_mo, eri_mo)
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

    def _line_search(
        self,
        mo_coeff,
        kappa,
        energy,
        ci0=None,
        start_scale=None,
        min_scale=None,
        accept_delta=1.0e-9,
    ):
        if start_scale is None:
            start_scale = self.step_size
        if min_scale is None:
            min_scale = max(1.0e-6, 1.0e-4 * abs(start_scale))

        scale = float(start_scale)
        best = None
        guess = self._copy_ci_guess(ci0)

        while scale >= min_scale:
            trial_coeff = rotate_orbitals(mo_coeff, scale * kappa)
            trial_mc = self._make_casci(trial_coeff, nstates=self.nstates, ci0=guess)
            trial_energy = self._objective_energy(trial_mc, self.state_id)
            guess = self._copy_ci_guess(trial_mc.ci)

            if best is None or trial_energy < best[1]:
                best = (trial_coeff, trial_energy, scale, trial_mc)

            if trial_energy < energy - float(accept_delta):
                return True, trial_coeff, trial_energy, scale, trial_mc

            scale *= 0.5

        if best is None:
            return False, mo_coeff, energy, 0.0, None
        return False, best[0], best[1], best[2], best[3]

    def _update_ah_trust_radius(self, radius, ratio, accepted_scale, step_vec):
        """Adapt the AH trust radius based on actual vs predicted reduction."""
        radius = float(min(radius, self.max_step))
        min_radius = min(self.max_step, max(5.0e-3, 0.02 * self.max_step))
        max_radius = self.max_step
        step_peak = float(np.max(np.abs(step_vec))) if len(step_vec) > 0 else 0.0
        used_peak = float(accepted_scale * step_peak)

        if ratio < 0.25:
            radius = max(min_radius, 0.5 * max(used_peak, min_radius))
        elif ratio > 0.75 and accepted_scale > 0.9 and step_peak >= 0.8 * radius:
            radius = min(max_radius, max(radius, used_peak) * 1.5)
        else:
            radius = min(max_radius, max(min_radius, max(radius, used_peak)))

        self._ah_trust_radius = radius

    def _ah_line_search(self, mo_coeff, mc, energy, grad_vec, hess_diag, step_vec, ci0=None):
        """
        Trust-radius acceptance loop for AH steps.

        Unlike the generic backtracking path, this routine explicitly compares
        actual and predicted energy reductions and shrinks the trust radius when
        the model is unreliable.
        """
        hess_model = np.maximum(np.abs(np.asarray(hess_diag, dtype=float)), self.level_shift)
        radius = float(min(getattr(self, "_ah_trust_radius", self.max_step), self.max_step))
        min_radius = min(self.max_step, max(5.0e-3, 0.02 * self.max_step))
        best = None

        for _ in range(4):
            limited_vec = limit_step_norm(step_vec, radius)
            if limited_vec.size == 0:
                break

            kappa = unpack_nonredundant(
                limited_vec,
                mc.ncore,
                mc.ncas,
                self.nmo,
                max_step=radius,
            )
            accepted, trial_coeff, trial_energy, accepted_scale, trial_mc = self._line_search(
                mo_coeff,
                kappa,
                energy,
                ci0=ci0,
                start_scale=1.0,
                min_scale=0.125,
                accept_delta=0.0,
            )

            if trial_mc is not None and (best is None or trial_energy < best[1]):
                best = (trial_coeff, trial_energy, accepted_scale, trial_mc, limited_vec.copy())

            if trial_mc is None:
                radius = max(min_radius, 0.5 * radius)
                continue

            actual_reduction = energy - trial_energy
            scaled_vec = accepted_scale * limited_vec
            predicted_reduction = -quadratic_model_change(scaled_vec, grad_vec, hess_model)
            if predicted_reduction <= 1.0e-12:
                ratio = -np.inf if actual_reduction <= 0.0 else np.inf
            else:
                ratio = actual_reduction / predicted_reduction

            if actual_reduction > 0.0:
                self._update_ah_trust_radius(radius, ratio, accepted_scale, limited_vec)
                return True, (trial_coeff, trial_energy, accepted_scale, trial_mc, limited_vec)

            radius = max(min_radius, 0.5 * radius)
            self._ah_trust_radius = radius
            ci0 = self._copy_ci_guess(trial_mc.ci)

        if best is not None:
            return False, best
        return False, (mo_coeff, energy, 0.0, None, np.asarray(step_vec, dtype=float))

    def _orbital_hessian_action(self, mo_coeff, mc, grad_vec, direction_vec):
        """Approximate the packed orbital Hessian action with a finite-difference gradient."""
        if self.ah_hessian == "analytic" and not self.use_cholesky_integrals:
            reference = self._get_ah_reference_data(mo_coeff, mc)
            direction_kappa = unpack_nonredundant(
                direction_vec,
                mc.ncore,
                mc.ncas,
                self.nmo,
            )
            grad_mat = orbital_hessian_action_from_integrals(
                reference["h1_mo"],
                reference["eri_mo"],
                reference["dm1"],
                reference["dm2"],
                direction_kappa,
            )
            return pack_nonredundant(grad_mat, mc.ncore, mc.ncas, self.nmo)

        direction_vec = np.asarray(direction_vec, dtype=float)
        if direction_vec.size == 0:
            return np.zeros(0, dtype=float)

        peak = float(np.max(np.abs(direction_vec)))
        if peak == 0.0:
            return np.zeros_like(direction_vec)

        fd_scale = min(self.ah_fd_step, 0.1 / peak)
        fd_scale = max(fd_scale, 1.0e-5)

        direction_kappa = unpack_nonredundant(
            direction_vec,
            mc.ncore,
            mc.ncas,
            self.nmo,
        )
        trial_coeff = rotate_orbitals(mo_coeff, fd_scale * direction_kappa)
        trial_mc, _, trial_grad = self._evaluate(
            trial_coeff,
            self.nstates,
            self.state_id,
            ci0=self._copy_ci_guess(mc.ci),
        )
        self.casci = trial_mc
        trial_grad_vec = pack_nonredundant(
            trial_grad,
            trial_mc.ncore,
            trial_mc.ncas,
            self.nmo,
        )
        return (trial_grad_vec - grad_vec) / fd_scale

    def _fallback_step_vectors(self, step_vec, grad_vec):
        """Generate progressively safer packed orbital steps after a rejection."""
        step_vec = np.asarray(step_vec, dtype=float)
        grad_vec = np.asarray(grad_vec, dtype=float)
        if step_vec.size == 0:
            return []

        radii = []
        for radius in (
            min(self.max_step, 0.02),
            min(self.max_step, 0.01),
            min(self.max_step, 0.005),
        ):
            if radius > 0.0:
                radii.append(float(radius))

        candidates = []
        seen = []

        def add_candidate(vec):
            vec = np.asarray(vec, dtype=float)
            if vec.size == 0:
                return
            for other in seen:
                if np.allclose(vec, other):
                    return
            seen.append(vec.copy())
            candidates.append(vec)

        for radius in radii:
            add_candidate(limit_step_norm(step_vec, radius))

        grad_peak = float(np.max(np.abs(grad_vec))) if grad_vec.size > 0 else 0.0
        if grad_peak > 0.0:
            descent_dir = -grad_vec / grad_peak
            for radius in radii:
                add_candidate(limit_step_norm(descent_dir, radius))

        return candidates

    def _format_stall_message(self, reason):
        """Build a high-signal convergence diagnostic for stalled runs."""
        lines = [reason]
        lines.append(
            "Optimizer: {}  cycles: {}/{}".format(
                self.optimizer,
                len(self.history),
                self.max_cycle,
            )
        )

        if self.history:
            last = self.history[-1]
            best = min(self.history, key=lambda item: item["energy"])
            step_norm = last.get("step_norm")
            step_text = "n/a" if step_norm is None else "{:.3e}".format(step_norm)
            lines.append(
                "Last cycle: {cycle}  energy={energy:.12f}  |grad|_inf={grad:.3e}  step={step}".format(
                    cycle=last["cycle"],
                    energy=last["energy"],
                    grad=last["gradient_norm"],
                    step=step_text,
                )
            )
            lines.append(
                "Best cycle: {cycle}  energy={energy:.12f}".format(
                    cycle=best["cycle"],
                    energy=best["energy"],
                )
            )

        if self.weights is not None:
            lines.append(
                "State averaging: {} roots with weights {}".format(
                    len(self.weights),
                    np.array2string(self.weights, precision=3),
                )
            )

        suggestions = [
            "Inspect mc.history for the per-cycle energy/gradient trend.",
            "Try a smaller orbital step or more damping: reduce step_size/max_step or increase level_shift.",
        ]
        if getattr(self.mol, "_build_driver", None) == "pyscf":
            suggestions.append(
                "Compare against pure PySCF with examples/qchem/casscf_compare_vs_pyscf.py."
            )
        lines.append("Next steps: " + " ".join(suggestions))
        return "\n".join(lines)

    def run(
        self,
        nstates=1,
        state_id=0,
        mo_coeff=None,
        use_cholesky=None,
        active_orbitals=None,
    ):
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
        self._full_derivative_cache = None
        self._full_derivative_sigma_cache = None
        self._invalidate_ah_reference_cache()
        self._casci_binary_cache = None
        self._casci_direct_connectivity_cache = None
        self._casci_sc1_cache = None
        self._casci_sc2_cache = None
        self._ah_trust_radius = self.max_step
        self.use_cholesky_integrals = self._resolve_use_cholesky(use_cholesky)
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
        mo_coeff = self.reorder_mo_for_active_orbitals(mo_coeff, active_orbitals)
        prev_energy = None
        prev_step_norm = None
        ci_guess = None
        prev_grad_vec = None
        accepted_step_vec = None

        for cycle in range(1, self.max_cycle + 1):
            self._invalidate_ah_reference_cache()
            mc, fock, grad = self._evaluate(
                mo_coeff,
                self.nstates,
                self.state_id,
                ci0=ci_guess,
            )
            energy = self._objective_energy(mc, self.state_id)
            gnorm = gradient_norm(grad, mc.ncore, mc.ncas, self.nmo)
            grad_vec = pack_nonredundant(grad, mc.ncore, mc.ncas, self.nmo)
            hess_diag = None
            if len(grad_vec) > 0 and self.optimizer == "AH":
                hess_diag = diagonal_hessian(
                    fock,
                    mc.ncore,
                    mc.ncas,
                    level_shift=self.level_shift,
                )
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
            elif self.optimizer == "AH":
                if len(grad_vec) > 0:
                    step_limit = min(self._ah_trust_radius, self.max_step)
                    diag_step = diagonal_preconditioned_vector(
                        grad,
                        fock,
                        mc.ncore,
                        mc.ncas,
                        level_shift=self.level_shift,
                    )
                    step_vec = augmented_hessian_direction(
                        grad_vec,
                        hess_diag,
                        max_step=step_limit,
                        regularization=self.level_shift,
                        fallback_step=diag_step,
                    )
                    step_vec = davidson_augmented_hessian_direction(
                        grad_vec,
                        hess_diag,
                        matvec=lambda vec: self._orbital_hessian_action(
                            mo_coeff,
                            mc,
                            grad_vec,
                            vec,
                        ),
                        max_step=step_limit,
                        regularization=self.level_shift,
                        max_cycle=self.ah_max_cycle,
                        max_subspace=self.ah_max_subspace,
                        tol=max(self.conv_tol_grad, 1.0e-4),
                        guess=step_vec,
                        fallback_step=diag_step,
                    )
                    if np.dot(step_vec, grad_vec) >= 0.0:
                        step_vec = diag_step
                    step_vec = limit_step_norm(step_vec, step_limit)
                    kappa = unpack_nonredundant(
                        step_vec,
                        mc.ncore,
                        mc.ncas,
                        self.nmo,
                        max_step=step_limit,
                    )
                else:
                    step_vec = np.zeros(0, dtype=float)
            else:
                step_vec = pack_nonredundant(kappa, mc.ncore, mc.ncas, self.nmo)
            kappa_diis = None
            if self.orbital_diis is not None:
                kappa_diis = self.orbital_diis.update(kappa, grad)

            accepted = False
            trial_coeff = mo_coeff
            trial_mc = None
            accepted_scale = 0.0
            used_step_vec = step_vec
            reset_optimizer_history = False

            if self.optimizer == "AH":
                accepted, ah_result = self._ah_line_search(
                    mo_coeff,
                    mc,
                    energy,
                    grad_vec,
                    hess_diag,
                    used_step_vec,
                    ci0=mc.ci,
                )
                trial_coeff, _, accepted_scale, trial_mc, used_step_vec = ah_result
            else:
                accepted, trial_coeff, _, accepted_scale, trial_mc = self._line_search(
                    mo_coeff,
                    kappa_diis if kappa_diis is not None else kappa,
                    energy,
                    ci0=mc.ci,
                )
                if accepted and kappa_diis is not None:
                    used_step_vec = pack_nonredundant(
                        kappa_diis,
                        mc.ncore,
                        mc.ncas,
                        self.nmo,
                    )

                if (
                    not accepted
                    and kappa_diis is not None
                    and not np.allclose(kappa_diis, kappa)
                ):
                    accepted, trial_coeff, _, accepted_scale, trial_mc = self._line_search(
                        mo_coeff,
                        kappa,
                        energy,
                        ci0=mc.ci,
                    )
                    if accepted:
                        used_step_vec = step_vec

            if not accepted:
                for fallback_vec in self._fallback_step_vectors(step_vec, grad_vec):
                    fallback_kappa = unpack_nonredundant(
                        fallback_vec,
                        mc.ncore,
                        mc.ncas,
                        self.nmo,
                    )
                    accepted, trial_coeff, _, accepted_scale, trial_mc = self._line_search(
                        mo_coeff,
                        fallback_kappa,
                        energy,
                        ci0=mc.ci,
                    )
                    if accepted:
                        used_step_vec = fallback_vec
                        reset_optimizer_history = True
                        break

            self.casci = mc
            if accepted:
                mo_coeff = trial_coeff
                prev_energy = energy
                ci_guess = self._copy_ci_guess(trial_mc.ci)
                accepted_step_vec = used_step_vec.copy()
                prev_step_norm = (
                    float(accepted_scale * np.max(np.abs(used_step_vec)))
                    if len(used_step_vec) > 0
                    else 0.0
                )
                if reset_optimizer_history:
                    if self.orbital_diis is not None:
                        self.orbital_diis = OrbitalDIIS(
                            max_space=self.diis_space,
                            start=self.diis_start,
                        )
                    self.lbfgs_s = []
                    self.lbfgs_y = []
            else:
                ci_guess = self._copy_ci_guess(mc.ci)
                if trial_mc is not None:
                    self.casci = trial_mc
                    ci_guess = self._copy_ci_guess(trial_mc.ci)
                if gnorm < self.conv_tol_grad:
                    self.converged = True
                    mo_coeff = self.casci.mo_coeff
                    break
                if self.optimizer == "AH":
                    min_radius = min(self.max_step, max(5.0e-3, 0.02 * self.max_step))
                    if self._ah_trust_radius > min_radius * (1.0 + 1.0e-12):
                        prev_energy = energy
                        prev_step_norm = 0.0
                        continue
                raise RuntimeError(
                    self._format_stall_message(
                        "CASSCF orbital line search failed before reaching the "
                        "gradient tolerance."
                    )
                )
            prev_grad_vec = grad_vec.copy()

        if not self.converged:
            raise RuntimeError(
                self._format_stall_message(
                    "Max macro steps reached before the CASSCF optimizer converged."
                )
            )

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


class SecondOrderCASSCF(CASSCF):
    """
    Native second-order CASSCF with fixed-integral microiterations.

    Notes
    -----
    This implementation follows the macro/microiteration architecture of
    second-order CASSCF: within each macroiteration, MO integrals are frozen and
    the CI coefficients and orbital rotations are optimized in microiterations.

    Coupling modes are:

    - ``"qn"``: default BFGS/QN coupling to the CI relaxation observed across
      fixed-integral microiterations.
    - ``"uncoupled"``: alternating CI/orbital optimization with an orbital-only
      Hessian action.
    - ``"relaxed_fd"``: expensive finite-difference Hessian action that
      re-solves CASCI at each displaced orbital point.
    - ``"partial"``: dense partial coupled-AH solve that can eliminate reduced
      CI rotations by a Schur complement.  Extra CASCI roots and finite-
      difference orbital-response vectors are opt-in because low excited roots
      are not always a reliable CI-response basis.
    - ``"full"``: experimental state-specific matrix-free coupled AH solve over
      orbital rotations and the full determinant-space CI response vector.
    """

    def __init__(
        self,
        mf,
        ncas,
        nelecas,
        max_cycle=16,
        max_micro_cycle=8,
        conv_tol=1.0e-8,
        conv_tol_grad=5.0e-6,
        conv_tol_grad_relaxed=5.0e-4,
        conv_tol_step=1.0e-4,
        level_shift=5.0e-4,
        step_size=0.25,
        max_step=0.10,
        optimizer="AH",
        optimizer_history=7,
        diis=False,
        diis_space=6,
        diis_start=2,
        ah_max_cycle=6,
        ah_max_subspace=12,
        ah_dense_threshold=0,
        ah_fd_step=5.0e-4,
        ah_hessian="analytic",
        ci_method="direct_ci",
        use_cholesky=None,
        coupling="qn",
        coupled_fd_step=5.0e-4,
        coupled_ci_roots=0,
        coupled_qspace_cycles=2,
        coupled_qspace_max_vectors=None,
        coupled_response_vectors=0,
        coupled_response_fd_step=5.0e-4,
        auto_active_restarts=True,
        active_restart_window=2,
        active_restart_max=4,
        max_cycles=None,
    ):
        super().__init__(
            mf,
            ncas,
            nelecas,
            max_cycle=max_cycle,
            conv_tol=conv_tol,
            conv_tol_grad=conv_tol_grad,
            conv_tol_grad_relaxed=conv_tol_grad_relaxed,
            conv_tol_step=conv_tol_step,
            level_shift=level_shift,
            step_size=step_size,
            max_step=max_step,
            optimizer=optimizer,
            optimizer_history=optimizer_history,
            diis=diis,
            diis_space=diis_space,
            diis_start=diis_start,
            ah_max_cycle=ah_max_cycle,
            ah_max_subspace=ah_max_subspace,
            ah_dense_threshold=ah_dense_threshold,
            ah_fd_step=ah_fd_step,
            ah_hessian=ah_hessian,
            ci_method=ci_method,
            use_cholesky=use_cholesky,
            max_cycles=max_cycles,
        )
        self.max_micro_cycle = int(max_micro_cycle)
        self.coupling = str(coupling).lower().replace("-", "_")
        if self.coupling not in {
            "uncoupled",
            "qn",
            "quasi_newton",
            "relaxed_fd",
            "coupled_fd",
            "partial",
            "partial_coupled",
            "full",
            "full_coupled",
        }:
            raise ValueError(
                "coupling must be 'uncoupled', 'qn', 'relaxed_fd', 'partial', "
                "or 'full'."
            )
        if self.coupling == "quasi_newton":
            self.coupling = "qn"
        if self.coupling == "coupled_fd":
            self.coupling = "relaxed_fd"
        if self.coupling == "partial_coupled":
            self.coupling = "partial"
        if self.coupling == "full_coupled":
            self.coupling = "full"
        self.coupled_fd_step = float(coupled_fd_step)
        self.coupled_ci_roots = int(coupled_ci_roots)
        self.coupled_qspace_cycles = int(coupled_qspace_cycles)
        self.coupled_qspace_max_vectors = (
            None
            if coupled_qspace_max_vectors is None
            else int(coupled_qspace_max_vectors)
        )
        self.coupled_response_vectors = (
            0 if coupled_response_vectors is None else int(coupled_response_vectors)
        )
        self.coupled_response_fd_step = float(coupled_response_fd_step)
        self.ah_dense_threshold = int(ah_dense_threshold)
        self.auto_active_restarts = bool(auto_active_restarts)
        self.active_restart_window = int(active_restart_window)
        self.active_restart_max = int(active_restart_max)
        self.active_restart_history = []
        self.micro_history = []
        self._qn_updates = []
        self._full_derivative_cache = None
        self._full_derivative_sigma_cache = None

    class _FrozenIntegralRHF:
        def __init__(self, parent_mf, h1_mo, eri_mo, mo_coeff):
            self.mol = parent_mf.mol
            self.nmo = mo_coeff.shape[1]
            self.mo_coeff = np.array(mo_coeff, copy=True)
            self.mo_occ = np.array(parent_mf.mo_occ, copy=True)
            self.nelec = int(np.rint(np.sum(self.mo_occ)))
            self._energy_nuc = float(np.real(parent_mf.energy_nuc()))
            self._h1_mo = np.array(h1_mo, copy=True)
            self.eri = np.array(eri_mo, copy=True)

        def energy_nuc(self):
            return self._energy_nuc

        def get_hcore(self):
            return np.array(self._h1_mo, copy=True)

        def get_hcore_mo(self, mo_coeff=None):
            if mo_coeff is None:
                mo_coeff = self.mo_coeff
            return np.array(mo_coeff.conj().T @ self._h1_mo @ mo_coeff, copy=False)

        def get_eri_mo(self, mo_coeff=None, notation="chem"):
            if mo_coeff is None:
                mo_coeff = self.mo_coeff
            eri = np.einsum(
                "pi,qj,pqrs,rk,sl->ijkl",
                mo_coeff.conj(),
                mo_coeff,
                self.eri,
                mo_coeff.conj(),
                mo_coeff,
                optimize=True,
            )
            if notation == "chem":
                return eri
            raise NotImplementedError("Only chem notation is supported.")

        def get_veff(self, dm):
            dm = np.asarray(dm)
            j = np.einsum("rs,pqrs->pq", dm, self.eri, optimize=True)
            k = np.einsum("rs,prqs->pq", dm, self.eri, optimize=True)
            return j - 0.5 * k

    def _transform_frozen_integrals(self, h1_ref, eri_ref, U):
        h1 = U.conj().T @ h1_ref @ U
        eri = np.einsum(
            "pi,qj,pqrs,rk,sl->ijkl",
            U.conj(),
            U,
            eri_ref,
            U.conj(),
            U,
            optimize=True,
        )
        return h1, eri

    def _make_integral_casci(self, h1_mo, eri_mo, mo_coeff, nstates, ci0=None):
        frozen_mf = self._FrozenIntegralRHF(self.mf, h1_mo, eri_mo, mo_coeff)
        mc = CASCI(frozen_mf, ncas=self.ncas, nelecas=self.nelecas)
        if self.spin_purification:
            mc.spin_purification = self.spin_purification
            mc.ss = self.ss
            mc.shift = self.shift
        if self._casci_binary_cache is not None:
            mc.binary = self._casci_binary_cache
        if self._casci_sc1_cache is not None and self._casci_sc2_cache is not None:
            mc.SC1 = self._casci_sc1_cache
            mc.SC2 = self._casci_sc2_cache
        requested_nstates = int(nstates)
        solve_nstates = self._ci_tracking_nstates(requested_nstates, ci0)
        mc.run(
            nstates=solve_nstates,
            mo_coeff=np.eye(self.nmo),
            method=self.ci_method,
            ci0=ci0,
            use_cholesky=False,
        )
        self._reorder_tracked_ci_root(mc, requested_nstates, ci0)
        self._update_casci_cache(mc)
        return mc

    def _micro_line_search(self, h1_ref, eri_ref, U, kappa, energy, ci0):
        scale = 1.0
        best = None
        while scale >= 0.125:
            step_u = expm(scale * kappa)
            trial_U = U @ step_u
            h1_trial, eri_trial = self._transform_frozen_integrals(h1_ref, eri_ref, trial_U)
            trial_mc = self._make_integral_casci(
                h1_trial,
                eri_trial,
                self.mo_coeff_ref,
                self.nstates,
                ci0=ci0,
            )
            trial_energy = self._objective_energy(trial_mc, self.state_id)
            if best is None or trial_energy < best[1]:
                best = (trial_U, trial_energy, trial_mc, scale)
            if trial_energy < energy - 1.0e-10:
                return True, trial_U, trial_energy, trial_mc, scale
            scale *= 0.5
        if best is None:
            return False, U, energy, None, 0.0
        return False, best[0], best[1], best[2], best[3]

    def _qn_hessian_action(self, vec, base_matvec):
        out = np.asarray(base_matvec(vec), dtype=float).copy()
        for y, s, bs, denom_y, denom_b in self._qn_updates:
            out += y * (np.dot(y, vec) / denom_y)
            out -= bs * (np.dot(bs, vec) / denom_b)
        return out

    def _append_qn_update(self, step_vec, delta_grad_vec, base_matvec):
        step_vec = np.asarray(step_vec, dtype=float)
        delta_grad_vec = np.asarray(delta_grad_vec, dtype=float)
        if step_vec.size == 0 or delta_grad_vec.size == 0:
            return
        denom_y = float(np.dot(delta_grad_vec, step_vec))
        if abs(denom_y) < 1.0e-12:
            return
        bs = self._qn_hessian_action(step_vec, base_matvec)
        denom_b = float(np.dot(bs, step_vec))
        if abs(denom_b) < 1.0e-12:
            return
        self._qn_updates.append(
            (
                delta_grad_vec.copy(),
                step_vec.copy(),
                bs.copy(),
                denom_y,
                denom_b,
            )
        )
        if len(self._qn_updates) > self.optimizer_history:
            self._qn_updates.pop(0)

    def _relaxed_ci_hessian_action(
        self,
        h1_ref,
        eri_ref,
        U,
        mo_coeff,
        grad_vec,
        mc,
        ci0,
        vec,
    ):
        """
        Hessian-vector product with CI relaxation by finite difference.

        This expensive opt-in action re-solves CASCI at the displaced orbital
        point before differencing the orbital gradient. It therefore includes
        the response of the CI coefficients to the orbital perturbation.
        """
        vec = np.asarray(vec, dtype=float)
        if vec.size == 0:
            return np.zeros(0, dtype=float)
        eps = self.coupled_fd_step
        if eps <= 0.0:
            raise ValueError("coupled_fd_step must be positive.")

        kappa = unpack_nonredundant(vec, mc.ncore, mc.ncas, self.nmo)
        trial_U = U @ expm(eps * kappa)
        h1_trial, eri_trial = self._transform_frozen_integrals(h1_ref, eri_ref, trial_U)
        trial_mc = self._make_integral_casci(
            h1_trial,
            eri_trial,
            mo_coeff,
            self.nstates,
            ci0=ci0,
        )
        if self.nstates == 1:
            grad_trial_vec = self._exact_orbital_gradient_vector(
                trial_mc,
                h1_trial,
                eri_trial,
                trial_mc.ci[self.state_id],
            )
        else:
            dm1_trial, dm2_trial = self._effective_rdms(trial_mc, self.state_id)
            fock_trial = generalized_fock(h1_trial, eri_trial, dm1_trial, dm2_trial)
            grad_trial = orbital_gradient(fock_trial)
            grad_trial_vec = pack_nonredundant(
                grad_trial,
                trial_mc.ncore,
                trial_mc.ncas,
                self.nmo,
            )
        return (grad_trial_vec - grad_vec) / eps

    def _ci_orbital_response_vectors(
        self,
        mc,
        h1_mo,
        eri_mo,
        nstates,
        weights=None,
        max_vectors=None,
        tol=1.0e-10,
    ):
        """
        Build compact preconditioned CI response vectors.

        Extra CASCI eigenroots alone can miss the CI-response space, especially
        when the next root has the wrong spin symmetry.  Following the
        Davidson-space construction used in second-order CASSCF, these vectors
        approximate ``-(H-E)^{-1} Q (dH/dkappa) |C_m>`` for each optimized root
        and orbital rotation, then SVD-compress the resulting response space.
        """
        nstates = int(nstates)
        if nstates < 1 or getattr(mc, "ci", None) is None:
            return None
        eps = self.coupled_response_fd_step
        if eps <= 0.0:
            raise ValueError("coupled_response_fd_step must be positive.")

        nstates = min(nstates, len(mc.ci))
        if weights is None:
            weights = np.ones(nstates, dtype=float) / float(nstates)
        weights = np.asarray(weights, dtype=float)
        if max_vectors is None:
            max_vectors = max(1, min(4 * nstates, len(mc.ci[0]) - len(mc.ci)))
        max_vectors = max(0, int(max_vectors))
        if max_vectors == 0:
            return None

        nvar = pack_nonredundant(
            np.zeros((self.nmo, self.nmo)),
            mc.ncore,
            mc.ncas,
            self.nmo,
        ).size
        if nvar == 0:
            return None

        root_mat = np.column_stack([np.asarray(c, dtype=float) for c in mc.ci])
        diag = ci_diagonal(mc)
        active_energies = np.asarray(mc.e_tot[:nstates], dtype=float) - float(mc.e_core)
        columns = []
        eye = np.eye(nvar)
        for ivec in range(nvar):
            kappa = unpack_nonredundant(
                eye[:, ivec],
                mc.ncore,
                mc.ncas,
                self.nmo,
            )
            rot_plus = expm(eps * kappa)
            rot_minus = expm(-eps * kappa)
            h1_plus, eri_plus = self._transform_frozen_integrals(
                h1_mo,
                eri_mo,
                rot_plus,
            )
            h1_minus, eri_minus = self._transform_frozen_integrals(
                h1_mo,
                eri_mo,
                rot_minus,
            )
            plus_mc = self._make_integral_casci(
                h1_plus,
                eri_plus,
                self.mo_coeff_ref,
                len(mc.ci),
                ci0=mc.ci,
            )
            minus_mc = self._make_integral_casci(
                h1_minus,
                eri_minus,
                self.mo_coeff_ref,
                len(mc.ci),
                ci0=mc.ci,
            )
            for m in range(nstates):
                vec = (plus_mc.ci_sigma(mc.ci[m]) - minus_mc.ci_sigma(mc.ci[m])) / (
                    2.0 * eps
                )
                vec -= root_mat @ (root_mat.conj().T @ vec)
                denom = active_energies[m] - diag
                safe = np.where(
                    np.abs(denom) > 1.0e-10,
                    denom,
                    np.where(denom >= 0.0, 1.0e-10, -1.0e-10),
                )
                vec = vec / safe
                vec -= root_mat @ (root_mat.conj().T @ vec)
                vec *= np.sqrt(max(float(weights[m]), 0.0))
                for prev in columns:
                    vec -= prev * np.vdot(prev, vec)
                norm = np.linalg.norm(vec)
                if norm > tol:
                    columns.append(vec / norm)

        if not columns:
            return None
        mat = np.column_stack(columns)
        try:
            u, singular_values, _ = np.linalg.svd(mat, full_matrices=False)
        except np.linalg.LinAlgError:
            return mat[:, :max_vectors]
        keep = singular_values > tol
        if not np.any(keep):
            return None
        return u[:, keep][:, :max_vectors]

    def _dense_orbital_hessian(self, grad_vec, hessian_action):
        n = int(np.asarray(grad_vec).size)
        if n == 0:
            return np.zeros((0, 0), dtype=float)
        eye = np.eye(n)
        cols = [np.asarray(hessian_action(eye[:, i]), dtype=float) for i in range(n)]
        hess = np.column_stack(cols)
        return 0.5 * (hess + hess.T)

    def _dense_augmented_hessian_step(
        self,
        grad_vec,
        hess,
        max_step,
        fallback_step,
    ):
        grad_vec = np.asarray(grad_vec, dtype=float)
        hess = np.asarray(hess, dtype=float)
        if grad_vec.size == 0:
            return np.zeros(0, dtype=float)

        hess = 0.5 * (hess + hess.T)
        hess = hess + self.level_shift * np.eye(hess.shape[0])
        ah = np.zeros((grad_vec.size + 1, grad_vec.size + 1), dtype=float)
        ah[0, 1:] = grad_vec
        ah[1:, 0] = grad_vec
        ah[1:, 1:] = hess

        eigvals, eigvecs = np.linalg.eigh(ah)
        best_step = None
        best_model = None
        for root in np.argsort(eigvals):
            alpha = float(eigvecs[0, root])
            if abs(alpha) < 1.0e-10:
                continue
            coeff = eigvecs[1:, root] / alpha
            if alpha < 0.0:
                coeff = -coeff
            if np.dot(coeff, grad_vec) >= -1.0e-12:
                continue
            model = float(np.dot(grad_vec, coeff) + 0.5 * np.dot(coeff, hess @ coeff))
            if best_step is None or model < best_model:
                best_step = coeff
                best_model = model
        if best_step is None:
            best_step = np.asarray(fallback_step, dtype=float)
        if max_step is not None and best_step.size > 0:
            peak = float(np.max(np.abs(best_step)))
            if peak > max_step and peak > 0.0:
                best_step = best_step * (max_step / peak)
        return np.asarray(best_step, dtype=float)

    def _ci_relaxed_orbital_hessian(self, orb_hess, ci_hess, hoc):
        """
        Eliminate CI rotations and return the relaxed orbital Hessian.
        """
        orb_hess = np.asarray(orb_hess, dtype=float)
        ci_hess = np.asarray(ci_hess, dtype=float)
        hoc = np.asarray(hoc, dtype=float)
        if ci_hess.size == 0 or hoc.size == 0:
            return orb_hess, lambda orbital_step: np.zeros(0, dtype=float)

        reg = ci_hess + self.level_shift * np.eye(ci_hess.shape[0])
        reg = 0.5 * (reg + reg.T)
        try:
            ci_response = np.linalg.solve(reg, hoc.T)
        except np.linalg.LinAlgError:
            ci_response = np.linalg.pinv(reg, rcond=1.0e-10) @ hoc.T

        relaxed = orb_hess - hoc @ ci_response
        relaxed = 0.5 * (relaxed + relaxed.T)

        def solve_ci_step(orbital_step):
            return -ci_response @ np.asarray(orbital_step, dtype=float)

        return relaxed, solve_ci_step

    def _ci_orbital_coupling_fd(
        self,
        subspace,
        mc,
        h1_mo,
        eri_mo,
        nstates,
        weights,
        eps=None,
    ):
        """
        Finite-difference derivative of reduced CI gradients by orbital rotations.
        """
        if eps is None:
            eps = self.coupled_response_fd_step
        if eps <= 0.0:
            raise ValueError("coupled_response_fd_step must be positive.")
        nvar = pack_nonredundant(
            np.zeros((self.nmo, self.nmo)),
            mc.ncore,
            mc.ncas,
            self.nmo,
        ).size
        if nvar == 0:
            return np.zeros((0, 0), dtype=float)

        cols = []
        eye = np.eye(nvar)
        for iorb in range(nvar):
            kappa = unpack_nonredundant(
                eye[:, iorb],
                mc.ncore,
                mc.ncas,
                self.nmo,
            )
            h1_plus, eri_plus = self._transform_frozen_integrals(
                h1_mo,
                eri_mo,
                expm(eps * kappa),
            )
            h1_minus, eri_minus = self._transform_frozen_integrals(
                h1_mo,
                eri_mo,
                expm(-eps * kappa),
            )
            plus_mc = self._make_integral_casci(
                h1_plus,
                eri_plus,
                self.mo_coeff_ref,
                len(mc.ci),
                ci0=mc.ci,
            )
            minus_mc = self._make_integral_casci(
                h1_minus,
                eri_minus,
                self.mo_coeff_ref,
                len(mc.ci),
                ci0=mc.ci,
            )
            grad_plus, _ = ReducedCISubspace.from_basis(
                plus_mc,
                subspace.basis,
            ).rotation_gradient(nstates=nstates, weights=weights)
            grad_minus, _ = ReducedCISubspace.from_basis(
                minus_mc,
                subspace.basis,
            ).rotation_gradient(nstates=nstates, weights=weights)
            cols.append((grad_plus - grad_minus) / (2.0 * eps))
        return np.vstack(cols)

    def _project_ci_response(self, vec, roots):
        vec = np.asarray(vec, dtype=float).copy()
        for root in roots:
            root = np.asarray(root, dtype=float)
            vec -= root * np.dot(root, vec)
        return vec

    def _active_integrals_from_full_mo(self, h1_mo, eri_mo, ncore, ncas):
        ncore = int(ncore)
        ncas = int(ncas)
        nocc = ncore + ncas
        h1_mo = np.asarray(h1_mo)
        eri_mo = np.asarray(eri_mo)
        active = slice(ncore, nocc)
        h1_active = np.array(h1_mo[active, active], copy=True)
        if ncore > 0:
            core = slice(0, ncore)
            core_j = 2.0 * np.einsum(
                "pqii->pq",
                eri_mo[active, active, core, core],
                optimize=True,
            )
            core_k = np.einsum(
                "piqi->pq",
                eri_mo[active, core, active, core],
                optimize=True,
            )
            h1_active = h1_active + core_j - core_k
        eri_active = np.array(eri_mo[active, active, active, active], copy=True)
        return h1_active, eri_active

    def _full_mo_integral_derivatives(self, h1_mo, eri_mo, kappa):
        """
        Differentiate frozen full-MO integrals under ``U = exp(kappa)`` at U=I.
        """
        h1_mo = np.asarray(h1_mo)
        eri_mo = np.asarray(eri_mo)
        kappa = np.asarray(kappa)
        dh1 = (
            np.einsum("pi,pj->ij", kappa, h1_mo, optimize=True)
            + np.einsum("qj,iq->ij", kappa, h1_mo, optimize=True)
        )
        deri = (
            np.einsum("pi,pjkl->ijkl", kappa, eri_mo, optimize=True)
            + np.einsum("qj,iqkl->ijkl", kappa, eri_mo, optimize=True)
            + np.einsum("rk,ijrl->ijkl", kappa, eri_mo, optimize=True)
            + np.einsum("sl,ijks->ijkl", kappa, eri_mo, optimize=True)
        )
        return dh1, deri

    def _active_integral_derivatives_from_orbital_step(
        self,
        h1_mo,
        eri_mo,
        kappa,
        ncore,
        ncas,
    ):
        dh1_full, deri_full = self._full_mo_integral_derivatives(h1_mo, eri_mo, kappa)
        return self._active_integrals_from_full_mo(
            dh1_full,
            deri_full,
            ncore,
            ncas,
        )

    def _active_integral_derivative_basis(self, mc, h1_mo, eri_mo):
        key = (id(h1_mo), id(eri_mo), int(mc.ncore), int(mc.ncas), int(self.nmo))
        if (
            self._full_derivative_cache is not None
            and self._full_derivative_cache.get("key") == key
        ):
            return (
                self._full_derivative_cache["dh1"],
                self._full_derivative_cache["deri"],
            )

        nvar = pack_nonredundant(
            np.zeros((self.nmo, self.nmo)),
            mc.ncore,
            mc.ncas,
            self.nmo,
        ).size
        dh1_cols = []
        deri_cols = []
        eye = np.eye(nvar)
        for iorb in range(nvar):
            kappa = unpack_nonredundant(eye[:, iorb], mc.ncore, mc.ncas, self.nmo)
            dh1, deri = self._active_integral_derivatives_from_orbital_step(
                h1_mo,
                eri_mo,
                kappa,
                mc.ncore,
                mc.ncas,
            )
            dh1_cols.append(dh1)
            deri_cols.append(deri)
        dh1_basis = np.asarray(dh1_cols)
        deri_basis = np.asarray(deri_cols)
        self._full_derivative_cache = {
            "key": key,
            "dh1": dh1_basis,
            "deri": deri_basis,
        }
        self._full_derivative_sigma_cache = None
        return dh1_basis, deri_basis

    def _core_energy_derivative(self, dh1_mo, deri_mo, ncore):
        ncore = int(ncore)
        if ncore <= 0:
            return 0.0
        core = range(ncore)
        out = 0.0
        for i in core:
            out += 2.0 * dh1_mo[i, i]
        for i in core:
            for j in core:
                out += 2.0 * deri_mo[i, i, j, j] - deri_mo[i, j, j, i]
        return float(np.real(out))

    def _core_energy_derivative_basis(self, mc, h1_mo, eri_mo):
        key = (id(h1_mo), id(eri_mo), int(mc.ncore), int(mc.ncas), int(self.nmo), "core")
        if (
            self._full_derivative_cache is not None
            and self._full_derivative_cache.get("core_key") == key
        ):
            return self._full_derivative_cache["de_core"]

        nvar = pack_nonredundant(
            np.zeros((self.nmo, self.nmo)),
            mc.ncore,
            mc.ncas,
            self.nmo,
        ).size
        vals = []
        eye = np.eye(nvar)
        for iorb in range(nvar):
            kappa = unpack_nonredundant(eye[:, iorb], mc.ncore, mc.ncas, self.nmo)
            dh1, deri = self._full_mo_integral_derivatives(h1_mo, eri_mo, kappa)
            vals.append(self._core_energy_derivative(dh1, deri, mc.ncore))
        vals = np.asarray(vals, dtype=float)
        if self._full_derivative_cache is None:
            self._full_derivative_cache = {}
        self._full_derivative_cache["core_key"] = key
        self._full_derivative_cache["de_core"] = vals
        return vals

    def _derivative_sigma_basis(self, mc, h1_mo, eri_mo, c0):
        dh1_basis, deri_basis = self._active_integral_derivative_basis(mc, h1_mo, eri_mo)
        key = (id(h1_mo), id(eri_mo), id(c0), int(mc.ncore), int(mc.ncas), int(self.nmo))
        if (
            self._full_derivative_sigma_cache is not None
            and self._full_derivative_sigma_cache.get("key") == key
        ):
            return self._full_derivative_sigma_cache["sigma"]
        cols = []
        for iorb in range(dh1_basis.shape[0]):
            deriv_mc = self._make_active_sigma_casci(
                mc,
                dh1_basis[iorb],
                deri_basis[iorb],
            )
            cols.append(deriv_mc.ci_sigma(c0))
        sigma_basis = np.asarray(cols)
        self._full_derivative_sigma_cache = {
            "key": key,
            "sigma": sigma_basis,
        }
        return sigma_basis

    def _exact_orbital_gradient_vector(self, mc, h1_mo, eri_mo, ci):
        """
        State-specific orbital energy derivative from CI Hamiltonian derivatives.

        This avoids relying on the spin-traced 2-RDM convention, which is not
        currently reliable for all spin states in the native CASCI code.
        """
        ci = np.asarray(ci, dtype=float)
        sigma_basis = self._derivative_sigma_basis(mc, h1_mo, eri_mo, ci)
        grad = sigma_basis @ ci
        grad = grad + self._core_energy_derivative_basis(mc, h1_mo, eri_mo)
        return np.asarray(grad, dtype=float)

    def _gradient_matrix_from_vector(self, grad_vec, ncore, ncas, nmo):
        return unpack_nonredundant(grad_vec, ncore, ncas, nmo)

    def _make_integral_sigma_casci(self, mc, h1_mo, eri_mo):
        """
        Lightweight CASCI-like object for CI sigma at supplied full MO integrals.

        This reuses the determinant basis and connectivity from ``mc`` and does
        not diagonalize the active-space Hamiltonian.
        """
        sigma_mc = copy.copy(mc)
        h1_active, eri_active = self._active_integrals_from_full_mo(
            h1_mo,
            eri_mo,
            mc.ncore,
            mc.ncas,
        )
        h1a, h1b = h1_active, h1_active
        sigma_mc.hcore = np.asarray([h1a, h1b])
        sigma_mc.h2e_cas = eri_active
        sigma_mc.eri_so = None
        sigma_mc._direct_spatial_h1 = h1_active
        sigma_mc._direct_spatial_eri = eri_active
        sigma_mc._direct_same_spin_eri = eri_active - eri_active.swapaxes(1, 3)
        sigma_mc._direct_cross_spin_eri = eri_active
        sigma_mc._direct_factor_H_diag = None
        sigma_mc._direct_factor_H_A = None
        sigma_mc._direct_factor_H_B = None
        sigma_mc._direct_factor_H_AA = None
        sigma_mc._direct_factor_H_BB = None
        sigma_mc._direct_factor_H_AB = None
        sigma_mc.direct_connectivity = mc.direct_connectivity
        sigma_mc.binary = mc.binary
        return sigma_mc

    def _make_active_sigma_casci(self, mc, h1_active, eri_active):
        """
        Lightweight CASCI-like object for CI sigma with active-space integrals.
        """
        sigma_mc = copy.copy(mc)
        h1a, h1b = h1_active, h1_active
        sigma_mc.hcore = np.asarray([h1a, h1b])
        sigma_mc.h2e_cas = eri_active
        sigma_mc.eri_so = None
        sigma_mc._direct_spatial_h1 = h1_active
        sigma_mc._direct_spatial_eri = eri_active
        sigma_mc._direct_same_spin_eri = eri_active - eri_active.swapaxes(1, 3)
        sigma_mc._direct_cross_spin_eri = eri_active
        sigma_mc._direct_factor_H_diag = None
        sigma_mc._direct_factor_H_A = None
        sigma_mc._direct_factor_H_B = None
        sigma_mc._direct_factor_H_AA = None
        sigma_mc._direct_factor_H_BB = None
        sigma_mc._direct_factor_H_AB = None
        sigma_mc.direct_connectivity = mc.direct_connectivity
        sigma_mc.binary = mc.binary
        return sigma_mc

    def _orbital_gradient_from_ci_response(self, mc, h1_mo, eri_mo, c0, dc):
        dc = self._project_ci_response(dc, [c0])
        if np.linalg.norm(dc) <= 1.0e-14:
            nvar = pack_nonredundant(
                np.zeros((self.nmo, self.nmo)),
                mc.ncore,
                mc.ncas,
                self.nmo,
            ).size
            return np.zeros(nvar, dtype=float)
        tdm1_dc, tdm2_dc = _transition_rdms_with_core(mc, dc, c0, nmo=self.nmo)
        tdm1_cd, tdm2_cd = _transition_rdms_with_core(mc, c0, dc, nmo=self.nmo)
        dm1_delta = tdm1_dc + tdm1_cd
        dm2_delta = tdm2_dc + tdm2_cd
        fock_delta = generalized_fock(h1_mo, eri_mo, dm1_delta, dm2_delta)
        return pack_nonredundant(
            orbital_gradient(fock_delta),
            mc.ncore,
            mc.ncas,
            self.nmo,
        )

    def _ci_gradient_from_orbital_response(self, mc, h1_mo, eri_mo, c0, orb_step):
        orb_step = np.asarray(orb_step, dtype=float)
        if orb_step.size == 0:
            return np.zeros_like(c0, dtype=float)
        sigma_basis = self._derivative_sigma_basis(mc, h1_mo, eri_mo, c0)
        vec = np.tensordot(orb_step, sigma_basis, axes=(0, 0))
        return self._project_ci_response(vec, [c0])

    def _orbital_gradient_from_ci_response_adjoint(self, mc, h1_mo, eri_mo, c0, dc):
        dc = self._project_ci_response(dc, [c0])
        if np.linalg.norm(dc) <= 1.0e-14:
            nvar = pack_nonredundant(
                np.zeros((self.nmo, self.nmo)),
                mc.ncore,
                mc.ncas,
                self.nmo,
            ).size
            return np.zeros(nvar, dtype=float)
        sigma_basis = self._derivative_sigma_basis(mc, h1_mo, eri_mo, c0)
        return sigma_basis @ dc

    def _full_coupled_step(
        self,
        mc,
        h1_mo,
        eri_mo,
        grad_vec,
        hess_diag,
        hessian_action,
        fallback_step,
        max_step,
    ):
        """
        Matrix-free state-specific coupled orbital/full-CI-response AH step.
        """
        if self.nstates != 1:
            raise NotImplementedError(
                "coupling='full' currently supports state-specific CASSCF only."
            )
        c0 = np.asarray(mc.ci[self.state_id], dtype=float)
        ndet = c0.size
        n_orb = grad_vec.size
        if ndet == 0 or n_orb == 0:
            return np.asarray(fallback_step, dtype=float), self._copy_ci_guess(mc.ci[:1])

        active_energy = float(mc.e_tot[self.state_id] - mc.e_core)
        ci_diag = np.asarray(ci_diagonal(mc), dtype=float)
        ci_hdiag = ci_diag - active_energy
        orb_hdiag = np.maximum(np.abs(np.asarray(hess_diag, dtype=float)), self.level_shift)
        precond_diag = np.concatenate(
            (
                orb_hdiag,
                np.maximum(np.abs(ci_hdiag), self.level_shift),
            )
        )
        total_grad = np.concatenate((np.asarray(grad_vec, dtype=float), np.zeros(ndet)))

        def split(vec):
            vec = np.asarray(vec, dtype=float)
            return vec[:n_orb], self._project_ci_response(vec[n_orb:], [c0])

        def matvec(vec):
            orb_part, ci_part = split(vec)
            out_orb = np.asarray(hessian_action(orb_part), dtype=float)
            out_orb += self._orbital_gradient_from_ci_response_adjoint(
                mc,
                h1_mo,
                eri_mo,
                c0,
                ci_part,
            )
            out_ci = mc.ci_sigma(ci_part) - active_energy * ci_part
            out_ci += self._ci_gradient_from_orbital_response(
                mc,
                h1_mo,
                eri_mo,
                c0,
                orb_part,
            )
            out_ci = self._project_ci_response(out_ci, [c0])
            return np.concatenate((out_orb, out_ci))

        seed = np.concatenate((np.asarray(fallback_step, dtype=float), np.zeros(ndet)))
        diag_step = -total_grad / precond_diag
        seeds = []
        for col in (seed, diag_step):
            norm = np.linalg.norm(col)
            if norm > 1.0e-12:
                seeds.append(col / norm)
        if not seeds:
            seeds = [-total_grad / np.linalg.norm(total_grad)]

        def orth(cols):
            clean = []
            for col in cols:
                col = np.asarray(col, dtype=float).copy()
                orb, ci = split(col)
                col = np.concatenate((orb, ci))
                for prev in clean:
                    col -= prev * np.dot(prev, col)
                norm = np.linalg.norm(col)
                if norm > 1.0e-12:
                    clean.append(col / norm)
            if not clean:
                return np.zeros((total_grad.size, 0), dtype=float)
            return np.column_stack(clean)

        V = orth(seeds)
        W = np.column_stack([matvec(V[:, i]) for i in range(V.shape[1])])
        best = None
        max_cycle = max(1, self.ah_max_cycle)
        max_subspace = max(2, self.ah_max_subspace)

        for _ in range(max_cycle):
            h_proj = 0.5 * (V.T @ W + (V.T @ W).T)
            g_proj = V.T @ total_grad
            ah = np.zeros((V.shape[1] + 1, V.shape[1] + 1), dtype=float)
            ah[0, 1:] = g_proj
            ah[1:, 0] = g_proj
            ah[1:, 1:] = h_proj
            eigvals, eigvecs = np.linalg.eigh(ah)
            candidate = None
            for root in np.argsort(eigvals):
                alpha = float(eigvecs[0, root])
                coeff = eigvecs[1:, root]
                if alpha < 0.0:
                    alpha = -alpha
                    coeff = -coeff
                if abs(alpha) < 1.0e-10:
                    continue
                raw_step = V @ (coeff / alpha)
                raw_hv = W @ (coeff / alpha)
                orb_raw, ci_raw = split(raw_step)
                scale = 1.0
                if max_step is not None and orb_raw.size > 0:
                    peak = np.max(np.abs(orb_raw))
                    if peak > max_step and peak > 0.0:
                        scale = max_step / peak
                step = scale * raw_step
                hv = scale * raw_hv
                deriv = float(np.dot(step, total_grad))
                if deriv >= -1.0e-12:
                    continue
                model = float(np.dot(total_grad, step) + 0.5 * np.dot(step, hv))
                residual = alpha * total_grad + raw_hv - float(eigvals[root]) * raw_step
                scalar_residual = float(np.dot(total_grad, raw_step) - float(eigvals[root]) * alpha)
                residual_norm = float(
                    np.sqrt(np.dot(residual, residual) + scalar_residual ** 2)
                )
                candidate = {
                    "model": model,
                    "residual_norm": residual_norm,
                    "eigenvalue": float(eigvals[root]),
                    "step": step,
                    "residual": residual,
                }
                break
            if candidate is None:
                break
            if best is None or candidate["model"] < best["model"]:
                best = candidate
            if candidate["residual_norm"] < max(self.conv_tol_grad, 1.0e-4):
                break

            denom = precond_diag - candidate["eigenvalue"]
            safe = np.where(
                np.abs(denom) > 1.0e-8,
                denom,
                np.where(denom >= 0.0, 1.0e-8, -1.0e-8),
            )
            correction = -candidate["residual"] / safe
            correction -= V @ (V.T @ correction)
            corr_norm = np.linalg.norm(correction)
            if corr_norm <= 1.0e-12:
                break
            correction /= corr_norm

            if V.shape[1] + 1 > max_subspace:
                V = orth([candidate["step"], diag_step])
                W = np.column_stack([matvec(V[:, i]) for i in range(V.shape[1])])
            else:
                V = np.column_stack((V, correction))
                W = np.column_stack((W, matvec(correction).reshape(-1, 1)))

        if best is None:
            return np.asarray(fallback_step, dtype=float), self._copy_ci_guess(mc.ci[:1])

        orb_step, ci_step = split(best["step"])
        ci_guess = c0 + ci_step
        ci_guess = self._project_ci_response(ci_guess, [])
        norm = np.linalg.norm(ci_guess)
        if norm > 1.0e-12:
            ci_guess = ci_guess / norm
        else:
            ci_guess = c0.copy()
        return np.asarray(orb_step, dtype=float), [ci_guess]

    def _partial_coupled_step(
        self,
        mc,
        h1_mo,
        eri_mo,
        grad_vec,
        hess_diag,
        hessian_action,
        fallback_step,
        max_step,
    ):
        """
        Dense partial coupled-AH step over orbital and reduced CI variables.

        Returns the orbital step and an updated CI guess from the CI part of the
        coupled AH eigenvector.
        """
        nroots = max(1, self.nstates)
        if len(mc.ci) <= nroots:
            step = davidson_augmented_hessian_direction(
                grad_vec,
                hess_diag,
                matvec=hessian_action,
                max_step=max_step,
                regularization=self.level_shift,
                max_cycle=self.ah_max_cycle,
                max_subspace=self.ah_max_subspace,
                tol=max(self.conv_tol_grad, 1.0e-4),
                guess=fallback_step,
                fallback_step=fallback_step,
            )
            return step, self._copy_ci_guess(mc.ci[:nroots])

        weights = self.weights
        if weights is None:
            weights = np.ones(nroots, dtype=float) / float(nroots)
        orb_hess = self._dense_orbital_hessian(grad_vec, hessian_action)
        if self.coupled_response_vectors is None:
            max_response = max(1, min(4 * nroots, len(mc.ci[0]) - len(mc.ci)))
        else:
            max_response = self.coupled_response_vectors
        response_vectors = self._ci_orbital_response_vectors(
            mc,
            h1_mo,
            eri_mo,
            nstates=nroots,
            weights=weights,
            max_vectors=max_response,
        )
        subspace = ReducedCISubspace.from_casci(
            mc,
            root_ids=range(len(mc.ci)),
            extra_vectors=response_vectors,
        )
        orbital_step = np.asarray(fallback_step, dtype=float)
        ci_guess_mat = np.column_stack(mc.ci[:nroots])
        q_cycles = max(1, self.coupled_qspace_cycles)

        for q_cycle in range(q_cycles):
            ci_grad, ci_pairs = subspace.rotation_gradient(nstates=nroots, weights=weights)
            ci_hess, hess_pairs = subspace.rotation_hessian(nstates=nroots, weights=weights)
            hoc, coupling_pairs = subspace.orbital_coupling(
                mc,
                h1_mo,
                eri_mo,
                nstates=nroots,
                weights=weights,
                nmo=self.nmo,
            )
            if ci_pairs != hess_pairs or ci_pairs != coupling_pairs or ci_grad.size == 0:
                step = davidson_augmented_hessian_direction(
                    grad_vec,
                    hess_diag,
                    matvec=hessian_action,
                    max_step=max_step,
                    regularization=self.level_shift,
                    max_cycle=self.ah_max_cycle,
                    max_subspace=self.ah_max_subspace,
                    tol=max(self.conv_tol_grad, 1.0e-4),
                    guess=fallback_step,
                    fallback_step=fallback_step,
                )
                return step, self._copy_ci_guess(mc.ci[:nroots])

            if max_response > 0:
                hco_fd = self._ci_orbital_coupling_fd(
                    subspace,
                    mc,
                    h1_mo,
                    eri_mo,
                    nstates=nroots,
                    weights=weights,
                )
                if hco_fd.shape == hoc.shape:
                    hoc = 0.5 * (hoc + hco_fd)

            relaxed_hess, solve_ci_step = self._ci_relaxed_orbital_hessian(
                orb_hess,
                ci_hess,
                hoc,
            )
            orbital_step = self._dense_augmented_hessian_step(
                grad_vec,
                relaxed_hess,
                max_step=max_step,
                fallback_step=fallback_step,
            )
            if np.dot(orbital_step, grad_vec) >= 0.0:
                orbital_step = np.asarray(fallback_step, dtype=float)
            ci_step = solve_ci_step(orbital_step)
            ci_guess_mat = subspace.rotated_state_vectors(
                ci_step,
                ci_pairs,
                nstates=nroots,
            )

            if q_cycle + 1 >= q_cycles:
                break
            energies = subspace.rayleigh_energies(mc, ci_guess_mat, include_core=True)
            max_new = self.coupled_qspace_max_vectors
            if max_new is None:
                max_new = nroots
            subspace, n_added = subspace.expand_with_residuals(
                mc,
                ci_guess_mat,
                energies,
                max_vectors=max_new,
                precondition=True,
            )
            if n_added == 0:
                break

        return orbital_step, [ci_guess_mat[:, i].copy() for i in range(ci_guess_mat.shape[1])]

    def _active_restart_candidates(self, mo_coeff):
        if self.active_restart_window <= 0 or self.active_restart_max <= 0:
            return []
        nmo = int(np.asarray(mo_coeff).shape[1])
        ncore = self._default_ncore()
        first_active = ncore
        last_active = ncore + self.ncas - 1
        if first_active < 0 or last_active >= nmo:
            return []

        default_active = list(range(first_active, last_active + 1))
        candidates = []

        for offset in range(1, self.active_restart_window + 1):
            repl = first_active - offset
            if repl >= 0:
                cand = default_active.copy()
                cand[0] = repl
                candidates.append(tuple(cand))

        for offset in range(1, self.active_restart_window + 1):
            repl = last_active + offset
            if repl < nmo:
                cand = default_active.copy()
                cand[-1] = repl
                candidates.append(tuple(cand))

        # Also allow one simultaneous lower-core/upper-virtual swap.  This is a
        # cheap way to cross active-space boundaries when two frontier orbitals
        # are entangled, but keep it after the single-boundary restarts.
        for lower in range(1, self.active_restart_window + 1):
            low_repl = first_active - lower
            if low_repl < 0:
                continue
            for upper in range(1, self.active_restart_window + 1):
                high_repl = last_active + upper
                if high_repl >= nmo:
                    continue
                cand = default_active.copy()
                cand[0] = low_repl
                cand[-1] = high_repl
                candidates.append(tuple(cand))

        unique = []
        seen = {tuple(default_active)}
        for cand in candidates:
            if cand in seen or len(set(cand)) != len(cand):
                continue
            seen.add(cand)
            unique.append(cand)
            if len(unique) >= self.active_restart_max:
                break
        return unique

    def _restart_solver(self):
        trial = SecondOrderCASSCF(
            self.mf,
            self.ncas,
            self.nelecas,
            max_cycle=self.max_cycle,
            max_micro_cycle=self.max_micro_cycle,
            conv_tol=self.conv_tol,
            conv_tol_grad=self.conv_tol_grad,
            conv_tol_grad_relaxed=self.conv_tol_grad_relaxed,
            conv_tol_step=self.conv_tol_step,
            level_shift=self.level_shift,
            step_size=self.step_size,
            max_step=self.max_step,
            optimizer=self.optimizer,
            optimizer_history=self.optimizer_history,
            diis=self.diis,
            diis_space=self.diis_space,
            diis_start=self.diis_start,
            ah_max_cycle=self.ah_max_cycle,
            ah_max_subspace=self.ah_max_subspace,
            ah_dense_threshold=self.ah_dense_threshold,
            ah_fd_step=self.ah_fd_step,
            ah_hessian=self.ah_hessian,
            ci_method=self.ci_method,
            use_cholesky=self.use_cholesky,
            coupling=self.coupling,
            coupled_fd_step=self.coupled_fd_step,
            coupled_ci_roots=self.coupled_ci_roots,
            coupled_qspace_cycles=self.coupled_qspace_cycles,
            coupled_qspace_max_vectors=self.coupled_qspace_max_vectors,
            coupled_response_vectors=self.coupled_response_vectors,
            coupled_response_fd_step=self.coupled_response_fd_step,
            auto_active_restarts=False,
        )
        trial.weights = None if self.weights is None else np.array(self.weights, copy=True)
        trial.spin_purification = self.spin_purification
        trial.ss = self.ss
        trial.shift = self.shift
        trial.ci_root_tracking = self.ci_root_tracking
        trial.ci_root_cushion = self.ci_root_cushion
        return trial

    def _try_active_restarts(self, initial_mo_coeff, best_energy):
        self.active_restart_history = []
        if (
            not self.auto_active_restarts
            or self.weights is not None
            or self.nstates != 1
            or self.state_id != 0
        ):
            return
        if not self.history:
            return
        last_grad = float(self.history[-1].get("gradient_norm", 0.0))
        if last_grad <= self.conv_tol_grad:
            return

        best = None
        accept_drop = max(1.0e-5, 10.0 * self.conv_tol)
        for active in self._active_restart_candidates(initial_mo_coeff):
            trial = self._restart_solver()
            row = {
                "active_orbitals": active,
                "ok": False,
                "energy": np.nan,
                "gradient_norm": np.nan,
            }
            try:
                trial.run(
                    nstates=self.nstates,
                    state_id=self.state_id,
                    mo_coeff=initial_mo_coeff,
                    active_orbitals=active,
                )
            except Exception as exc:
                row["error"] = "{}: {}".format(type(exc).__name__, exc)
            else:
                row["ok"] = bool(trial.converged)
                row["energy"] = float(np.ravel(trial.e_tot)[0])
                row["gradient_norm"] = float(trial.history[-1]["gradient_norm"])
                if trial.converged and row["energy"] < best_energy - max(self.conv_tol, 1.0e-10):
                    if best is None or row["energy"] < best[0]:
                        best = (row["energy"], trial, active)
            self.active_restart_history.append(row)
            if best is not None and best_energy - best[0] >= accept_drop:
                break

        if best is None:
            return
        _, trial, active = best
        self.mo_coeff = np.array(trial.mo_coeff, copy=True)
        self.casci = trial.casci
        self.ci = trial.ci
        self.e_tot = np.array(trial.e_tot, copy=True)
        self.ncore = trial.ncore
        self.history = list(trial.history)
        self.micro_history = list(trial.micro_history)
        self.converged = trial.converged
        self.active_orbitals = active

    def run(
        self,
        nstates=1,
        state_id=0,
        mo_coeff=None,
        use_cholesky=None,
        active_orbitals=None,
    ):
        if isinstance(self.mf.mo_coeff, tuple):
            raise NotImplementedError(
                "SecondOrderCASSCF currently supports restricted references only."
            )
        if use_cholesky:
            raise NotImplementedError(
                "SecondOrderCASSCF currently requires dense MO integrals in the "
                "microiterations."
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
        self.history = []
        self.micro_history = []
        self.converged = False
        self.casci = None
        self.mo_coeff = None
        self.e_tot = None
        self.ci = None
        self.active_restart_history = []
        self.active_orbitals = active_orbitals
        self._invalidate_ah_reference_cache()
        self._casci_binary_cache = None
        self._casci_direct_connectivity_cache = None
        self._casci_sc1_cache = None
        self._casci_sc2_cache = None
        self._ah_trust_radius = self.max_step
        self._qn_updates = []

        if mo_coeff is None:
            mo_coeff = np.array(self.mf.mo_coeff, copy=True)
        else:
            mo_coeff = np.array(mo_coeff, copy=True)
        initial_mo_coeff = np.array(mo_coeff, copy=True)
        mo_coeff = self.reorder_mo_for_active_orbitals(mo_coeff, active_orbitals)

        prev_energy = None
        ci_guess = None

        for macro in range(1, self.max_cycle + 1):
            self.mo_coeff_ref = mo_coeff
            h1_ref = self.mf.get_hcore_mo(mo_coeff)
            eri_ref = self.mf.get_eri_mo(mo_coeff, notation="chem")

            U = np.eye(self.nmo)
            micro_mc = None
            micro_energy = None
            micro_gnorm = None
            micro_step = None
            local_ci_guess = self._copy_ci_guess(ci_guess)
            prev_micro_grad_vec = None
            prev_micro_step_vec = None
            qn_base_hessian_action = None
            self._qn_updates = []

            for micro in range(1, self.max_micro_cycle + 1):
                h1_cur, eri_cur = self._transform_frozen_integrals(h1_ref, eri_ref, U)
                solve_nstates = self.nstates
                if self.coupling == "partial":
                    solve_nstates += max(0, self.coupled_ci_roots)
                mc = self._make_integral_casci(
                    h1_cur,
                    eri_cur,
                    mo_coeff,
                    solve_nstates,
                    ci0=local_ci_guess,
                )
                energy = self._objective_energy(mc, self.state_id)
                dm1, dm2 = self._effective_rdms(mc, self.state_id)
                fock = generalized_fock(h1_cur, eri_cur, dm1, dm2)
                if self.nstates == 1:
                    grad_vec = self._exact_orbital_gradient_vector(
                        mc,
                        h1_cur,
                        eri_cur,
                        mc.ci[self.state_id],
                    )
                    grad = self._gradient_matrix_from_vector(
                        grad_vec,
                        mc.ncore,
                        mc.ncas,
                        self.nmo,
                    )
                    gnorm = float(np.max(np.abs(grad_vec))) if grad_vec.size else 0.0
                else:
                    grad = orbital_gradient(fock)
                    gnorm = gradient_norm(grad, mc.ncore, mc.ncas, self.nmo)
                    grad_vec = pack_nonredundant(grad, mc.ncore, mc.ncas, self.nmo)

                def base_hessian_action(vec):
                    return pack_nonredundant(
                        orbital_hessian_action_from_integrals(
                            h1_cur,
                            eri_cur,
                            dm1,
                            dm2,
                            unpack_nonredundant(vec, mc.ncore, mc.ncas, self.nmo),
                        ),
                        mc.ncore,
                        mc.ncas,
                        self.nmo,
                    )

                if self.coupling == "qn" and qn_base_hessian_action is None:
                    h1_qn = np.array(h1_cur, copy=True)
                    eri_qn = np.array(eri_cur, copy=True)
                    dm1_qn = np.array(dm1, copy=True)
                    dm2_qn = np.array(dm2, copy=True)
                    ncore_qn = mc.ncore
                    ncas_qn = mc.ncas
                    nmo_qn = self.nmo

                    def qn_base_hessian_action(vec):
                        return pack_nonredundant(
                            orbital_hessian_action_from_integrals(
                                h1_qn,
                                eri_qn,
                                dm1_qn,
                                dm2_qn,
                                unpack_nonredundant(vec, ncore_qn, ncas_qn, nmo_qn),
                            ),
                            ncore_qn,
                            ncas_qn,
                            nmo_qn,
                        )

                if self.coupling == "qn":
                    hessian_action = lambda vec: self._qn_hessian_action(
                        vec,
                        qn_base_hessian_action,
                    )
                elif self.coupling == "relaxed_fd":
                    hessian_action = lambda vec: self._relaxed_ci_hessian_action(
                        h1_ref,
                        eri_ref,
                        U,
                        mo_coeff,
                        grad_vec,
                        mc,
                        mc.ci,
                        vec,
                    )
                else:
                    hessian_action = base_hessian_action

                if (
                    self.coupling == "qn"
                    and prev_micro_step_vec is not None
                    and prev_micro_grad_vec is not None
                ):
                    self._append_qn_update(
                        prev_micro_step_vec,
                        grad_vec - prev_micro_grad_vec,
                        qn_base_hessian_action,
                    )

                self.micro_history.append(
                    {
                        "macro": macro,
                        "micro": micro,
                        "energy": energy,
                        "gradient_norm": gnorm,
                    }
                )

                micro_mc = mc
                micro_energy = energy
                micro_gnorm = gnorm
                local_ci_guess = self._copy_ci_guess(mc.ci)

                if gnorm < self.conv_tol_grad:
                    micro_step = 0.0
                    break

                hess_diag = diagonal_hessian(
                    fock,
                    mc.ncore,
                    mc.ncas,
                    level_shift=self.level_shift,
                )
                diag_step = diagonal_preconditioned_vector(
                    grad,
                    fock,
                    mc.ncore,
                    mc.ncas,
                    level_shift=self.level_shift,
                )
                step_limit = min(self._ah_trust_radius, self.max_step)
                step_vec = augmented_hessian_direction(
                    grad_vec,
                    hess_diag,
                    max_step=step_limit,
                    regularization=self.level_shift,
                    fallback_step=diag_step,
                )
                coupled_ci_guess = None
                if self.coupling == "partial":
                    step_vec, coupled_ci_guess = self._partial_coupled_step(
                        mc,
                        h1_cur,
                        eri_cur,
                        grad_vec,
                        hess_diag,
                        hessian_action,
                        step_vec,
                        step_limit,
                    )
                elif self.coupling == "full":
                    step_vec, coupled_ci_guess = self._full_coupled_step(
                        mc,
                        h1_cur,
                        eri_cur,
                        grad_vec,
                        hess_diag,
                        hessian_action,
                        step_vec,
                        step_limit,
                    )
                else:
                    if (
                        self.ah_dense_threshold > 0
                        and grad_vec.size <= self.ah_dense_threshold
                    ):
                        dense_hess = self._dense_orbital_hessian(
                            grad_vec,
                            hessian_action,
                        )
                        step_vec = self._dense_augmented_hessian_step(
                            grad_vec,
                            dense_hess,
                            max_step=step_limit,
                            fallback_step=step_vec,
                        )
                    else:
                        step_vec = davidson_augmented_hessian_direction(
                            grad_vec,
                            hess_diag,
                            matvec=hessian_action,
                            max_step=step_limit,
                            regularization=self.level_shift,
                            max_cycle=self.ah_max_cycle,
                            max_subspace=self.ah_max_subspace,
                            tol=max(self.conv_tol_grad, 1.0e-4),
                            guess=step_vec,
                            fallback_step=diag_step,
                        )
                if np.dot(step_vec, grad_vec) >= 0.0:
                    step_vec = diag_step
                step_vec = limit_step_norm(step_vec, step_limit)
                kappa = unpack_nonredundant(
                    step_vec,
                    mc.ncore,
                    mc.ncas,
                    self.nmo,
                    max_step=step_limit,
                )
                accepted, U, _, trial_mc, accepted_scale = self._micro_line_search(
                    h1_ref,
                    eri_ref,
                    U,
                    kappa,
                    energy,
                    coupled_ci_guess if coupled_ci_guess is not None else mc.ci[: self.nstates],
                )
                micro_step = (
                    float(accepted_scale * np.max(np.abs(step_vec)))
                    if step_vec.size > 0
                    else 0.0
                )
                if not accepted:
                    break
                micro_mc = trial_mc
                local_ci_guess = self._copy_ci_guess(trial_mc.ci)
                if self.coupling in {"partial", "full"} and coupled_ci_guess is not None:
                    local_ci_guess = self._copy_ci_guess(coupled_ci_guess)
                prev_micro_grad_vec = grad_vec.copy()
                prev_micro_step_vec = accepted_scale * step_vec

            mo_coeff = mo_coeff @ U
            self.history.append(
                {
                    "cycle": macro,
                    "energy": float(micro_energy),
                    "gradient_norm": float(micro_gnorm),
                    "step_norm": 0.0 if micro_step is None else float(micro_step),
                    "micro_cycles": micro,
                }
            )
            self.casci = micro_mc

            if (
                prev_energy is not None
                and abs(micro_energy - prev_energy) < self.conv_tol
                and micro_gnorm < self.conv_tol_grad_relaxed
            ):
                self.converged = True
                break

            prev_energy = float(micro_energy)
            ci_guess = self._copy_ci_guess(micro_mc.ci)

        if not self.converged:
            finite_energies = [
                float(entry["energy"])
                for entry in self.history
                if np.isfinite(float(entry.get("energy", np.nan)))
            ]
            if finite_energies:
                self._try_active_restarts(initial_mo_coeff, min(finite_energies))
            if self.converged:
                return self
            raise RuntimeError(
                self._format_stall_message(
                    "Max macro steps reached before the second-order CASSCF "
                    "optimizer converged."
                )
            )

        self.mo_coeff = mo_coeff
        # Rebuild final CASCI on the actual reference object for a consistent
        # public-facing result container.
        self.casci = self._make_casci(
            mo_coeff,
            nstates=self.nstates,
            ci0=ci_guess,
        )
        self.ci = self.casci.ci
        self.e_tot = self.casci.e_tot
        self.ncore = self.casci.ncore
        if active_orbitals is None:
            self._try_active_restarts(initial_mo_coeff, float(np.ravel(self.e_tot)[0]))
        return self
