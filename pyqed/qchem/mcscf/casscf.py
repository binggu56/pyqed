#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Native first-order CASSCF built on top of the existing CASCI solvers.

This module lives beside the original constrained-optimization implementation
now exposed as ``pyqed.qchem.COCAS``.
"""

import copy

import numpy as np

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
    orbital_step,
    orbital_hessian_action_from_integrals,
    pack_nonredundant,
    quadratic_model_change,
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
        mc.run(
            nstates=nstates,
            mo_coeff=mo_coeff,
            method=self.ci_method,
            ci0=ci0,
            use_cholesky=self.use_cholesky_integrals,
        )
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

    def run(self, nstates=1, state_id=0, mo_coeff=None, use_cholesky=None):
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
