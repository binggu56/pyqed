#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Native CASSCF drivers built on top of the existing CASCI solvers.

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
from .direct_ci import (
    CASCI,
    build_direct_connectivity,
    _compute_diag_compact_factors,
    _compute_double_cross_values_from_factors,
    _compute_double_same_values_from_factors,
    _compute_single_values_from_factors,
    _sigma_compact_derivative_batch_numba,
)
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
    limit_step_trust_radius,
    nonredundant_pairs,
    orbital_gradient,
    orbital_step,
    orbital_hessian_action_from_integrals,
    pack_nonredundant,
    quadratic_model_change,
    rotate_orbitals,
    shifted_hessian_trust_step,
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


class FirstOrderCASSCF:
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
        verbose=0,
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
        self.verbose = int(verbose)
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
        self._casci_spin_string_connectivity_cache = None
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

    def _casci_verbose(self):
        """Keep CASSCF-level verbosity from leaking raw internal CASCI solves."""
        return max(0, self.verbose - 1)

    def _log_casscf_cycle(self, cycle, energy, gnorm, step_norm, micro_cycles=None):
        if self.verbose < 1:
            return
        step_text = "None" if step_norm is None else "{:.3e}".format(float(step_norm))
        fields = [
            "CASSCF cycle {:3d}".format(int(cycle)),
            "E = {:.10f}".format(float(energy)),
            "|g| = {:.3e}".format(float(gnorm)),
            "step = {}".format(step_text),
        ]
        if micro_cycles is not None:
            fields.append("micro = {}".format(int(micro_cycles)))
        print("  ".join(fields))

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
        probe = CASCI(
            self.mf,
            ncas=self.ncas,
            nelecas=self.nelecas,
            verbose=self._casci_verbose(),
        )
        probe.fix_spin(s=s, ss=ss, shift=shift)
        self.spin_purification = probe.spin_purification
        self.ss = probe.ss
        self.shift = probe.shift
        return self

    def _make_casci(self, mo_coeff, nstates, ci0=None):
        mc = CASCI(
            self.mf,
            ncas=self.ncas,
            nelecas=self.nelecas,
            verbose=self._casci_verbose(),
        )
        if self._casci_binary_cache is not None:
            mc.binary = self._casci_binary_cache
        if self._casci_direct_connectivity_cache is not None:
            mc.direct_connectivity = self._casci_direct_connectivity_cache
        if self._casci_spin_string_connectivity_cache is not None:
            mc.spin_string_connectivity = self._casci_spin_string_connectivity_cache
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
        if getattr(mc, "spin_string_connectivity", None) is not None:
            self._casci_spin_string_connectivity_cache = mc.spin_string_connectivity
        if getattr(mc, "SC1", None) is not None and getattr(mc, "SC2", None) is not None:
            self._casci_sc1_cache = mc.SC1
            self._casci_sc2_cache = mc.SC2

    def _resolve_use_cholesky(self, use_cholesky=None):
        if use_cholesky is None:
            use_cholesky = self.use_cholesky
        if use_cholesky is None:
            use_cholesky = bool(getattr(self.mf, "cholesky_jk", False))
        return _resolve_use_cholesky_integrals(self.mf, use_cholesky)

    def orbital_rotation_pairs(self, ncore=None, ncas=None, nmo=None):
        if ncore is None:
            ncore = self.ncore if self.ncore is not None else self._default_ncore()
        if ncas is None:
            ncas = self.ncas
        if nmo is None:
            nmo = self.nmo
        return nonredundant_pairs(ncore, ncas, nmo)

    def _pack_orbitals(self, matrix, ncore, ncas, nmo):
        return pack_nonredundant(matrix, ncore, ncas, nmo)

    def _unpack_orbitals(self, vec, ncore, ncas, nmo, max_step=None):
        return unpack_nonredundant(vec, ncore, ncas, nmo, max_step=max_step)

    def _gradient_norm(self, gradient, ncore, ncas, nmo):
        return gradient_norm(gradient, ncore, ncas, nmo)

    def _diagonal_hessian(self, fock, ncore, ncas, level_shift=1.0e-3):
        return diagonal_hessian(fock, ncore, ncas, level_shift=level_shift)

    def _diagonal_preconditioned_vector(
        self,
        gradient,
        fock,
        ncore,
        ncas,
        level_shift=1.0e-3,
    ):
        return diagonal_preconditioned_vector(
            gradient,
            fock,
            ncore,
            ncas,
            level_shift=level_shift,
        )

    def _diagonal_inverse_hessian(self, fock, ncore, ncas, level_shift=1.0e-3):
        return diagonal_inverse_hessian(fock, ncore, ncas, level_shift=level_shift)

    def _orbital_step(
        self,
        fock,
        ncore,
        ncas,
        step_size=1.0,
        level_shift=1.0e-3,
        max_step=0.25,
    ):
        return orbital_step(
            fock,
            ncore,
            ncas,
            step_size=step_size,
            level_shift=level_shift,
            max_step=max_step,
        )

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
            if hasattr(self.mf, "mo_factors"):
                pair_factors = self.mf.mo_factors(mo_coeff, occ_mo)
            else:
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
        self._full_coupled_seed = None
        self._joint_trial_sigma_cache = {}
        self._invalidate_ah_reference_cache()
        self._casci_binary_cache = None
        self._casci_direct_connectivity_cache = None
        self._casci_spin_string_connectivity_cache = None
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
            self._log_casscf_cycle(cycle, energy, gnorm, prev_step_norm)

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

    def overlap(self, other):
        """Electronic state overlap with another completed CASSCF object.

        The optimized CASSCF wavefunction is represented by the final CASCI
        solve in the optimized orbital basis, so the existing CASCI overlap
        implementation is the correct backend for CASSCF LDR links.
        """
        if self.casci is None:
            raise ValueError("Run CASSCF before requesting overlaps.")
        other_casci = getattr(other, "casci", other)
        if other_casci is None:
            raise ValueError("Run the other CASSCF object before requesting overlaps.")
        return self.casci.overlap(other_casci)


class SecondOrderCASSCF(FirstOrderCASSCF):
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
        max_cycle=50,
        max_micro_cycle=8,
        conv_tol=1.0e-7,
        conv_tol_grad=None,
        conv_tol_grad_relaxed=None,
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
        ah_pspace_size=12,
        ah_pspace_max_cycle=6,
        ah_trust_metric="component",
        ah_adaptive_trust=False,
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
        coupled_accept_min_ratio=0.05,
        coupled_fallback=True,
        coupled_reuse_subspace=False,
        orbital_parameterization="exponential",
        internal_preopt_steps=0,
        internal_preopt_max_step=None,
        internal_preopt_hessian="finite_difference",
        internal_preopt_solver="dense",
        internal_preopt_space="core_active",
        internal_preopt_guard_cycles=0,
        internal_optimization=False,
        internal_max_cycle=None,
        internal_conv_tol_grad=None,
        internal_conv_tol_step=None,
        internal_conv_tol_energy=None,
        auto_active_restarts=True,
        active_restart_window=2,
        active_restart_max=4,
        exact_state_specific_gradient=False,
        max_cycles=None,
        verbose=0,
    ):
        if max_cycles is not None:
            if int(max_cycle) != 50 and int(max_cycle) != int(max_cycles):
                raise ValueError(
                    "Received conflicting values for max_cycle={} and "
                    "max_cycles={}.".format(max_cycle, max_cycles)
                )
            max_cycle = max_cycles
        if conv_tol_grad is None:
            conv_tol_grad = math.sqrt(float(conv_tol))
        if conv_tol_grad_relaxed is None:
            conv_tol_grad_relaxed = conv_tol_grad

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
            ah_fd_step=ah_fd_step,
            ah_hessian=ah_hessian,
            ci_method=ci_method,
            use_cholesky=use_cholesky,
            max_cycles=None,
            verbose=verbose,
        )
        self.max_micro_cycle = int(max_micro_cycle)
        self.ah_pspace_size = int(ah_pspace_size)
        self.ah_pspace_max_cycle = int(ah_pspace_max_cycle)
        self.ah_trust_metric = str(ah_trust_metric).lower().replace("-", "_")
        if self.ah_trust_metric not in {"component", "norm"}:
            raise ValueError("ah_trust_metric must be 'component' or 'norm'.")
        self.ah_adaptive_trust = bool(ah_adaptive_trust)
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
            "simultaneous",
            "simultaneous_full",
            "simultaneous_partial",
            "simultaneous_reduced",
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
        if self.coupling in {"simultaneous", "simultaneous_full"}:
            self.coupling = "full"
        if self.coupling in {"simultaneous_partial", "simultaneous_reduced"}:
            self.coupling = "partial"
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
        self.coupled_accept_min_ratio = float(coupled_accept_min_ratio)
        self.coupled_fallback = bool(coupled_fallback)
        self.coupled_reuse_subspace = bool(coupled_reuse_subspace)
        self.orbital_parameterization = str(orbital_parameterization).lower().replace(
            "-",
            "_",
        )
        if self.orbital_parameterization in {"exp", "expm"}:
            self.orbital_parameterization = "exponential"
        if self.orbital_parameterization in {"cayley", "wmk_cayley"}:
            self.orbital_parameterization = "wmk"
        if self.orbital_parameterization not in {"exponential", "wmk"}:
            raise ValueError(
                "orbital_parameterization must be 'exponential' or 'wmk'."
            )
        self.internal_preopt_steps = int(internal_preopt_steps)
        self.internal_preopt_max_step = (
            None
            if internal_preopt_max_step is None
            else float(internal_preopt_max_step)
        )
        self.internal_preopt_hessian = str(internal_preopt_hessian).lower().replace(
            "-",
            "_",
        )
        if self.internal_preopt_hessian not in {
            "diagonal",
            "analytic",
            "finite_difference",
            "coupled",
            "coupled_fd",
        }:
            raise ValueError(
                "internal_preopt_hessian must be 'diagonal', 'analytic', "
                "'finite_difference', 'coupled', or 'coupled_fd'."
            )
        self.internal_preopt_solver = str(internal_preopt_solver).lower().replace(
            "-",
            "_",
        )
        if self.internal_preopt_solver not in {"dense", "davidson"}:
            raise ValueError("internal_preopt_solver must be 'dense' or 'davidson'.")
        self.internal_preopt_space = str(internal_preopt_space).lower().replace(
            "-",
            "_",
        )
        if self.internal_preopt_space in {"all", "all_nonredundant"}:
            self.internal_preopt_space = "nonredundant"
        if self.internal_preopt_space not in {"core_active", "nonredundant"}:
            raise ValueError(
                "internal_preopt_space must be 'core_active' or 'nonredundant'."
            )
        self.internal_preopt_guard_cycles = int(internal_preopt_guard_cycles)
        self.internal_optimization = bool(internal_optimization)
        if internal_max_cycle is None:
            internal_max_cycle = 20 if self.internal_optimization else self.internal_preopt_steps
        self.internal_max_cycle = int(internal_max_cycle)
        self.internal_conv_tol_grad = (
            self.conv_tol_grad
            if internal_conv_tol_grad is None
            else float(internal_conv_tol_grad)
        )
        self.internal_conv_tol_step = (
            self.conv_tol_step
            if internal_conv_tol_step is None
            else float(internal_conv_tol_step)
        )
        self.internal_conv_tol_energy = (
            self.conv_tol
            if internal_conv_tol_energy is None
            else float(internal_conv_tol_energy)
        )
        self.internal_optimization_converged = False
        self.auto_active_restarts = bool(auto_active_restarts)
        self.active_restart_window = int(active_restart_window)
        self.active_restart_max = int(active_restart_max)
        self.exact_state_specific_gradient = bool(exact_state_specific_gradient)
        self.active_restart_history = []
        self.internal_preopt_history = []
        self.micro_history = []
        self._qn_updates = []
        self._full_derivative_cache = None
        self._full_derivative_sigma_cache = None
        self._full_coupled_seed = None
        self._joint_trial_sigma_cache = {}

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

    class _FrozenFactorRHF:
        def __init__(self, parent_mf, h1_mo, pair_factors, mo_coeff):
            self.mol = parent_mf.mol
            self.nmo = mo_coeff.shape[1]
            self.mo_coeff = np.array(mo_coeff, copy=True)
            self.mo_occ = np.array(parent_mf.mo_occ, copy=True)
            self.nelec = int(np.rint(np.sum(self.mo_occ)))
            self._energy_nuc = float(np.real(parent_mf.energy_nuc()))
            self._h1_mo = np.array(h1_mo, copy=True)
            self.eri_factors = np.array(pair_factors, copy=True)
            self.cholesky_jk = True
            self.cholesky_tol = getattr(parent_mf, "cholesky_tol", None)
            self.cholesky_max_rank = getattr(parent_mf, "cholesky_max_rank", None)
            self.low_rank_tol = getattr(parent_mf, "low_rank_tol", None)
            self.low_rank_max_rank = getattr(parent_mf, "low_rank_max_rank", None)

        def energy_nuc(self):
            return self._energy_nuc

        def get_hcore(self):
            return np.array(self._h1_mo, copy=True)

        def get_hcore_mo(self, mo_coeff=None):
            if mo_coeff is None:
                mo_coeff = self.mo_coeff
            return np.array(mo_coeff.conj().T @ self._h1_mo @ mo_coeff, copy=False)

        def get_eri_mo(self, mo_coeff=None, notation="chem"):
            raise AssertionError("dense MO ERIs should not be built in factorized CASSCF")

        def get_veff(self, dm):
            dm = np.asarray(dm)
            factors = self.eri_factors
            coeff = np.einsum("Pkl,lk->P", factors, dm, optimize=True)
            j = np.einsum("P,Pij->ij", coeff, factors, optimize=True)
            k = np.einsum("Pil,lk,Pkj->ij", factors, dm, factors, optimize=True)
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

    def _transform_frozen_factor_integrals(self, h1_ref, pair_ref, U):
        h1 = U.conj().T @ h1_ref @ U
        pair_factors = np.einsum(
            "pi,Ppq,qj->Pij",
            U.conj(),
            pair_ref,
            U,
            optimize=True,
        )
        return h1, pair_factors

    def _orbital_unitary(self, kappa):
        kappa = np.asarray(kappa, dtype=float)
        if self.orbital_parameterization == "exponential":
            return expm(kappa)
        eye = np.eye(kappa.shape[0], dtype=kappa.dtype)
        trial = eye + kappa
        metric = trial.T @ trial
        metric = 0.5 * (metric + metric.T)
        eigvals, eigvecs = np.linalg.eigh(metric)
        safe = np.maximum(eigvals, 1.0e-14)
        invsqrt = (eigvecs / np.sqrt(safe)) @ eigvecs.T
        # WMK treats the orbital variable as T = U - I.  The polar factor of
        # I + T gives a unitary completion with U = I + T + 1/2 T^2 + O(T^3)
        # for anti-Hermitian T, matching the WMK second-order model.
        return trial @ invsqrt

    def _apply_orbital_update(self, mo_or_u, kappa):
        return np.real_if_close(np.asarray(mo_or_u) @ self._orbital_unitary(kappa))

    def _make_integral_casci(self, h1_mo, eri_mo, mo_coeff, nstates, ci0=None):
        frozen_mf = self._FrozenIntegralRHF(self.mf, h1_mo, eri_mo, mo_coeff)
        mc = CASCI(
            frozen_mf,
            ncas=self.ncas,
            nelecas=self.nelecas,
            verbose=self._casci_verbose(),
        )
        if self.spin_purification:
            mc.spin_purification = self.spin_purification
            mc.ss = self.ss
            mc.shift = self.shift
        if self._casci_binary_cache is not None:
            mc.binary = self._casci_binary_cache
        if self._casci_spin_string_connectivity_cache is not None:
            mc.spin_string_connectivity = self._casci_spin_string_connectivity_cache
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

    def _make_factor_integral_casci(
        self,
        h1_mo,
        pair_factors,
        mo_coeff,
        nstates,
        ci0=None,
    ):
        frozen_mf = self._FrozenFactorRHF(self.mf, h1_mo, pair_factors, mo_coeff)
        mc = CASCI(
            frozen_mf,
            ncas=self.ncas,
            nelecas=self.nelecas,
            verbose=self._casci_verbose(),
        )
        if self.spin_purification:
            mc.spin_purification = self.spin_purification
            mc.ss = self.ss
            mc.shift = self.shift
        if self._casci_binary_cache is not None:
            mc.binary = self._casci_binary_cache
        if self._casci_direct_connectivity_cache is not None:
            mc.direct_connectivity = self._casci_direct_connectivity_cache
        if self._casci_spin_string_connectivity_cache is not None:
            mc.spin_string_connectivity = self._casci_spin_string_connectivity_cache
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
            use_cholesky=True,
        )
        self._reorder_tracked_ci_root(mc, requested_nstates, ci0)
        self._update_casci_cache(mc)
        return mc

    def _micro_line_search(self, h1_ref, eri_ref, U, kappa, energy, ci0):
        scale = 1.0
        best = None
        while scale >= 0.125:
            trial_U = self._apply_orbital_update(U, scale * kappa)
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

    def _factor_micro_line_search(self, h1_ref, pair_ref, U, kappa, energy, ci0):
        scale = 1.0
        best = None
        while scale >= 0.125:
            trial_U = self._apply_orbital_update(U, scale * kappa)
            h1_trial, pair_trial = self._transform_frozen_factor_integrals(
                h1_ref,
                pair_ref,
                trial_U,
            )
            trial_mc = self._make_factor_integral_casci(
                h1_trial,
                pair_trial,
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

    def _joint_ci_orbital_trial(
        self,
        h1_ref,
        eri_ref,
        U,
        kappa,
        mc,
        ci_base,
        ci_target,
        scale,
    ):
        trial_U = self._apply_orbital_update(U, float(scale) * kappa)
        h1_trial, eri_trial = self._transform_frozen_integrals(h1_ref, eri_ref, trial_U)
        sigma_mc = self._cached_integral_sigma_casci(mc, h1_trial, eri_trial)
        ci_roots = self._scaled_ci_guess_roots(ci_base, ci_target, scale)
        energies = [
            float(sigma_mc.e_core + np.real(np.vdot(ci, sigma_mc.ci_sigma(ci))))
            for ci in ci_roots
        ]
        total_energy = self._objective_energy_from_values(energies)
        trial_mc = copy.copy(sigma_mc)
        trial_mc.ci = ci_roots
        trial_mc.e_tot = np.asarray(energies, dtype=float)
        trial_mc.nstates = len(ci_roots)
        return trial_U, total_energy, trial_mc

    def _factor_joint_ci_orbital_trial(
        self,
        h1_ref,
        pair_ref,
        U,
        kappa,
        mc,
        ci_base,
        ci_target,
        scale,
    ):
        trial_U = self._apply_orbital_update(U, float(scale) * kappa)
        h1_trial, pair_trial = self._transform_frozen_factor_integrals(
            h1_ref,
            pair_ref,
            trial_U,
        )
        sigma_mc = self._cached_factor_integral_sigma_casci(mc, h1_trial, pair_trial)
        ci_roots = self._scaled_ci_guess_roots(ci_base, ci_target, scale)
        energies = [
            float(sigma_mc.e_core + np.real(np.vdot(ci, sigma_mc.ci_sigma(ci))))
            for ci in ci_roots
        ]
        total_energy = self._objective_energy_from_values(energies)
        trial_mc = copy.copy(sigma_mc)
        trial_mc.ci = ci_roots
        trial_mc.e_tot = np.asarray(energies, dtype=float)
        trial_mc.nstates = len(ci_roots)
        return trial_U, total_energy, trial_mc

    def _ci_guess_root_list(self, roots):
        if roots is None:
            return None
        if isinstance(roots, np.ndarray) and roots.ndim == 1:
            return [np.asarray(roots, dtype=float)]
        return [np.asarray(root, dtype=float) for root in roots]

    def _orthonormalize_ci_roots(self, roots, fallback_roots=None):
        fallback_roots = self._ci_guess_root_list(fallback_roots) or []
        out = []
        for idx, root in enumerate(self._ci_guess_root_list(roots) or []):
            vec = np.asarray(root, dtype=float).copy()
            for prev in out:
                vec -= prev * np.dot(prev, vec)
            norm = np.linalg.norm(vec)
            if norm <= 1.0e-12 and idx < len(fallback_roots):
                vec = np.asarray(fallback_roots[idx], dtype=float).copy()
                for prev in out:
                    vec -= prev * np.dot(prev, vec)
                norm = np.linalg.norm(vec)
            if norm <= 1.0e-12:
                continue
            out.append(vec / norm)
        return out

    def _scaled_ci_guess_roots(self, ci_base, ci_target, scale):
        base_roots = self._ci_guess_root_list(ci_base) or []
        target_roots = self._ci_guess_root_list(ci_target) or base_roots
        nroots = min(len(base_roots), len(target_roots))
        trial = [
            base_roots[root] + float(scale) * (target_roots[root] - base_roots[root])
            for root in range(nroots)
        ]
        return self._orthonormalize_ci_roots(trial, fallback_roots=base_roots)

    def _objective_energy_from_values(self, energies):
        energies = np.asarray(energies, dtype=float)
        if self.weights is None:
            return float(energies[min(self.state_id, len(energies) - 1)])
        weights = np.asarray(self.weights, dtype=float)[: len(energies)]
        if weights.size == 0:
            return float(energies[0])
        weights = weights / float(np.sum(weights))
        return float(np.dot(weights, energies[: weights.size]))

    def _joint_trust_region_micro_search(
        self,
        h1_ref,
        eri_ref,
        U,
        kappa,
        energy,
        mc,
        ci_target,
        model_reduction,
        model_linear=None,
        model_quadratic=None,
    ):
        ci_base = self._ci_guess_root_list(mc.ci[: self.nstates])
        min_scale = 0.125
        best = None
        scale = 1.0
        model_reduction = float(model_reduction)
        model_linear = None if model_linear is None else float(model_linear)
        model_quadratic = None if model_quadratic is None else float(model_quadratic)
        while scale >= min_scale:
            trial_U, trial_energy, trial_mc = self._joint_ci_orbital_trial(
                h1_ref,
                eri_ref,
                U,
                kappa,
                mc,
                ci_base,
                ci_target,
                scale,
            )
            actual = float(energy - trial_energy)
            if model_linear is not None and model_quadratic is not None:
                predicted = -(
                    scale * model_linear
                    + 0.5 * scale * scale * model_quadratic
                )
            else:
                predicted = model_reduction * scale * scale
            if predicted <= 1.0e-12:
                ratio = np.inf if actual > 0.0 else -np.inf
            else:
                ratio = actual / predicted
            row = (trial_U, trial_energy, trial_mc, scale, actual, predicted, ratio)
            if best is None or trial_energy < best[1]:
                best = row
            ratio_ok = np.isfinite(ratio) and ratio >= self.coupled_accept_min_ratio
            model_free_ok = not np.isfinite(ratio) and actual > 0.0
            if actual > 0.0 and (ratio_ok or model_free_ok):
                return True, row
            scale *= 0.5
        if best is None:
            return False, (U, energy, None, 0.0, 0.0, model_reduction, -np.inf)
        return False, best

    def _factor_joint_trust_region_micro_search(
        self,
        h1_ref,
        pair_ref,
        U,
        kappa,
        energy,
        mc,
        ci_target,
        model_reduction,
        model_linear=None,
        model_quadratic=None,
    ):
        ci_base = self._ci_guess_root_list(mc.ci[: self.nstates])
        min_scale = 0.125
        best = None
        scale = 1.0
        model_reduction = float(model_reduction)
        model_linear = None if model_linear is None else float(model_linear)
        model_quadratic = None if model_quadratic is None else float(model_quadratic)
        while scale >= min_scale:
            trial_U, trial_energy, trial_mc = self._factor_joint_ci_orbital_trial(
                h1_ref,
                pair_ref,
                U,
                kappa,
                mc,
                ci_base,
                ci_target,
                scale,
            )
            actual = float(energy - trial_energy)
            if model_linear is not None and model_quadratic is not None:
                predicted = -(
                    scale * model_linear
                    + 0.5 * scale * scale * model_quadratic
                )
            else:
                predicted = model_reduction * scale * scale
            if predicted <= 1.0e-12:
                ratio = np.inf if actual > 0.0 else -np.inf
            else:
                ratio = actual / predicted
            row = (trial_U, trial_energy, trial_mc, scale, actual, predicted, ratio)
            if best is None or trial_energy < best[1]:
                best = row
            ratio_ok = np.isfinite(ratio) and ratio >= self.coupled_accept_min_ratio
            model_free_ok = not np.isfinite(ratio) and actual > 0.0
            if actual > 0.0 and (ratio_ok or model_free_ok):
                return True, row
            scale *= 0.5
        if best is None:
            return False, (U, energy, None, 0.0, 0.0, model_reduction, -np.inf)
        return False, best

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

        kappa = self._unpack_orbitals(vec, mc.ncore, mc.ncas, self.nmo)
        trial_U = self._apply_orbital_update(U, eps * kappa)
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
            grad_trial_vec = self._pack_orbitals(
                grad_trial,
                trial_mc.ncore,
                trial_mc.ncas,
                self.nmo,
            )
        return (grad_trial_vec - grad_vec) / eps

    def _factor_relaxed_ci_hessian_action(
        self,
        h1_ref,
        pair_ref,
        U,
        mo_coeff,
        grad_vec,
        mc,
        ci0,
        vec,
    ):
        vec = np.asarray(vec, dtype=float)
        if vec.size == 0:
            return np.zeros(0, dtype=float)
        eps = self.coupled_fd_step
        if eps <= 0.0:
            raise ValueError("coupled_fd_step must be positive.")

        kappa = self._unpack_orbitals(vec, mc.ncore, mc.ncas, self.nmo)
        trial_U = self._apply_orbital_update(U, eps * kappa)
        h1_trial, pair_trial = self._transform_frozen_factor_integrals(
            h1_ref,
            pair_ref,
            trial_U,
        )
        trial_mc = self._make_factor_integral_casci(
            h1_trial,
            pair_trial,
            mo_coeff,
            self.nstates,
            ci0=ci0,
        )
        dm1_trial, dm2_trial = self._effective_rdms_occ(trial_mc, self.state_id)
        grad_trial_vec = self._factor_frozen_orbital_gradient_vector(
            trial_mc,
            h1_trial,
            pair_trial,
            dm1_trial,
            dm2_trial,
        )
        return (grad_trial_vec - grad_vec) / eps

    def _frozen_orbital_gradient_vector(
        self,
        mc,
        h1_mo,
        eri_mo,
        dm1,
        dm2,
        ci,
    ):
        if self.nstates == 1:
            return self._exact_orbital_gradient_vector(
                mc,
                h1_mo,
                eri_mo,
                ci,
            )
        fock = generalized_fock(h1_mo, eri_mo, dm1, dm2)
        grad = orbital_gradient(fock)
        return self._pack_orbitals(grad, mc.ncore, mc.ncas, self.nmo)

    def _factor_frozen_orbital_gradient_vector(
        self,
        mc,
        h1_mo,
        pair_factors,
        dm1_occ,
        dm2_occ,
    ):
        nocc_like = mc.ncore + mc.ncas
        fock = generalized_fock_from_factors(
            h1_mo,
            pair_factors[:, :, :nocc_like],
            dm1_occ,
            dm2_occ,
        )
        grad = orbital_gradient(fock)
        return self._pack_orbitals(grad, mc.ncore, mc.ncas, self.nmo)

    def _pair_factor_response_slices(self, pair_factors, kappa, nocc_like):
        pair_factors = np.asarray(pair_factors)
        kappa = np.asarray(kappa)
        nocc_like = int(nocc_like)
        kappa_occ = kappa[:, :nocc_like]
        d_full_occ = (
            np.einsum(
                "Ppj,pi->Pij",
                pair_factors[:, :, :nocc_like],
                kappa,
                optimize=True,
            )
            + np.einsum(
                "Piq,qj->Pij",
                pair_factors,
                kappa_occ,
                optimize=True,
            )
        )
        return d_full_occ, d_full_occ[:, :nocc_like, :]

    def _generalized_fock_factor_product_sliced(self, left_full_occ, right_occ_occ, dm2_occ):
        contracted = np.einsum(
            "Pst,rqst->Prq",
            right_occ_occ,
            dm2_occ,
            optimize=True,
        )
        return np.einsum(
            "Ppr,Prq->pq",
            left_full_occ,
            contracted,
            optimize=True,
        )

    def _orbital_hessian_action_from_factors(
        self,
        h1_mo,
        pair_factors,
        dm1_occ,
        dm2_occ,
        kappa,
    ):
        h1_mo = np.asarray(h1_mo)
        pair_factors = np.asarray(pair_factors)
        dm1_occ = np.asarray(dm1_occ)
        dm2_occ = np.asarray(dm2_occ)
        kappa = np.asarray(kappa)
        nocc_like = int(dm1_occ.shape[0])

        dh1 = h1_mo @ kappa - kappa @ h1_mo
        pair_full_occ = pair_factors[:, :, :nocc_like]
        pair_occ_occ = pair_full_occ[:, :nocc_like, :]
        d_full_occ, d_occ_occ = self._pair_factor_response_slices(
            pair_factors,
            kappa,
            nocc_like,
        )

        dfock = np.zeros_like(h1_mo, dtype=np.result_type(h1_mo, pair_factors, dm2_occ))
        dfock[:, :nocc_like] = np.einsum(
            "pr,rq->pq",
            dh1[:, :nocc_like],
            dm1_occ,
            optimize=True,
        )
        dfock[:, :nocc_like] += self._generalized_fock_factor_product_sliced(
            d_full_occ,
            pair_occ_occ,
            dm2_occ,
        )
        dfock[:, :nocc_like] += self._generalized_fock_factor_product_sliced(
            pair_full_occ,
            d_occ_occ,
            dm2_occ,
        )
        return orbital_gradient(dfock)

    def _parameterized_orbital_hessian_action(
        self,
        h1_mo,
        eri_mo,
        dm1,
        dm2,
        mc,
        grad_vec,
        ci,
        vec,
    ):
        """
        Frozen-RDM Hessian action using the active orbital parameterization.

        This is the parameterization-consistent counterpart of the analytic
        integral-response Hessian.  It is used for WMK/Cayley mode and for the
        explicit finite-difference AH Hessian option.
        """
        vec = np.asarray(vec, dtype=float)
        if vec.size == 0:
            return np.zeros(0, dtype=float)
        peak = float(np.max(np.abs(vec)))
        if peak == 0.0:
            return np.zeros_like(vec)
        eps = min(self.ah_fd_step, 0.1 / peak)
        eps = max(eps, 1.0e-5)

        kappa = self._unpack_orbitals(vec, mc.ncore, mc.ncas, self.nmo)
        plus_u = self._orbital_unitary(eps * kappa)
        minus_u = self._orbital_unitary(-eps * kappa)
        h1_plus, eri_plus = self._transform_frozen_integrals(h1_mo, eri_mo, plus_u)
        h1_minus, eri_minus = self._transform_frozen_integrals(h1_mo, eri_mo, minus_u)
        grad_plus = self._frozen_orbital_gradient_vector(
            mc,
            h1_plus,
            eri_plus,
            dm1,
            dm2,
            ci,
        )
        grad_minus = self._frozen_orbital_gradient_vector(
            mc,
            h1_minus,
            eri_minus,
            dm1,
            dm2,
            ci,
        )
        return (grad_plus - grad_minus) / (2.0 * eps)

    def _factor_parameterized_orbital_hessian_action(
        self,
        h1_mo,
        pair_factors,
        dm1_occ,
        dm2_occ,
        mc,
        vec,
    ):
        vec = np.asarray(vec, dtype=float)
        if vec.size == 0:
            return np.zeros(0, dtype=float)
        peak = float(np.max(np.abs(vec)))
        if peak == 0.0:
            return np.zeros_like(vec)
        eps = min(self.ah_fd_step, 0.1 / peak)
        eps = max(eps, 1.0e-5)

        kappa = self._unpack_orbitals(vec, mc.ncore, mc.ncas, self.nmo)
        plus_u = self._orbital_unitary(eps * kappa)
        minus_u = self._orbital_unitary(-eps * kappa)
        h1_plus, pair_plus = self._transform_frozen_factor_integrals(
            h1_mo,
            pair_factors,
            plus_u,
        )
        h1_minus, pair_minus = self._transform_frozen_factor_integrals(
            h1_mo,
            pair_factors,
            minus_u,
        )
        grad_plus = self._factor_frozen_orbital_gradient_vector(
            mc,
            h1_plus,
            pair_plus,
            dm1_occ,
            dm2_occ,
        )
        grad_minus = self._factor_frozen_orbital_gradient_vector(
            mc,
            h1_minus,
            pair_minus,
            dm1_occ,
            dm2_occ,
        )
        return (grad_plus - grad_minus) / (2.0 * eps)

    def _analytic_orbital_hessian_action(
        self,
        h1_mo,
        eri_mo,
        dm1,
        dm2,
        mc,
        vec,
    ):
        """
        Analytic local second-order orbital Hessian action.

        For WMK/Cayley and exponential parameterizations this local Hessian is
        the same at the current point because both unitary maps have identical
        first and second derivatives at zero: ``U = I + K + 1/2 K^2 + O(K^3)``.
        """
        return self._pack_orbitals(
            orbital_hessian_action_from_integrals(
                h1_mo,
                eri_mo,
                dm1,
                dm2,
                self._unpack_orbitals(vec, mc.ncore, mc.ncas, self.nmo),
            ),
            mc.ncore,
            mc.ncas,
            self.nmo,
        )

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

        nvar = self._pack_orbitals(
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
            kappa = self._unpack_orbitals(
                eye[:, ivec],
                mc.ncore,
                mc.ncas,
                self.nmo,
            )
            rot_plus = self._orbital_unitary(eps * kappa)
            rot_minus = self._orbital_unitary(-eps * kappa)
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
        nvar = self._pack_orbitals(
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
            kappa = self._unpack_orbitals(
                eye[:, iorb],
                mc.ncore,
                mc.ncas,
                self.nmo,
            )
            h1_plus, eri_plus = self._transform_frozen_integrals(
                h1_mo,
                eri_mo,
                self._orbital_unitary(eps * kappa),
            )
            h1_minus, eri_minus = self._transform_frozen_integrals(
                h1_mo,
                eri_mo,
                self._orbital_unitary(-eps * kappa),
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

    def _active_integrals_from_full_mo_factors(self, h1_mo, pair_factors, ncore, ncas):
        h1_active, pair_active = self._active_h1_pair_from_full_mo_factors(
            h1_mo,
            pair_factors,
            ncore,
            ncas,
        )
        eri_active = np.einsum("Ppq,Prs->pqrs", pair_active, pair_active, optimize=True)
        return h1_active, eri_active

    def _active_h1_pair_from_full_mo_factors(self, h1_mo, pair_factors, ncore, ncas):
        ncore = int(ncore)
        ncas = int(ncas)
        nocc = ncore + ncas
        h1_mo = np.asarray(h1_mo)
        pair_factors = np.asarray(pair_factors)
        active = slice(ncore, nocc)
        h1_active = np.array(h1_mo[active, active], copy=True)
        if ncore > 0:
            core = slice(0, ncore)
            pair_aa = pair_factors[:, active, active]
            pair_cc = pair_factors[:, core, core]
            pair_ac = pair_factors[:, active, core]
            core_j = 2.0 * np.einsum("Ppq,Pii->pq", pair_aa, pair_cc, optimize=True)
            core_k = np.einsum("Ppi,Pqi->pq", pair_ac, pair_ac, optimize=True)
            h1_active = h1_active + core_j - core_k
        pair_active = pair_factors[:, active, active]
        return h1_active, pair_active

    def _active_integral_derivatives_from_factor_step(
        self,
        dh1_mo,
        pair_factors,
        dpair_factors,
        ncore,
        ncas,
    ):
        ncore = int(ncore)
        ncas = int(ncas)
        nocc = ncore + ncas
        dh1_mo = np.asarray(dh1_mo)
        pair_factors = np.asarray(pair_factors)
        dpair_factors = np.asarray(dpair_factors)
        active = slice(ncore, nocc)
        dh1_active = np.array(dh1_mo[active, active], copy=True)
        if ncore > 0:
            core = slice(0, ncore)
            b_aa = pair_factors[:, active, active]
            db_aa = dpair_factors[:, active, active]
            b_cc = pair_factors[:, core, core]
            db_cc = dpair_factors[:, core, core]
            b_ac = pair_factors[:, active, core]
            db_ac = dpair_factors[:, active, core]
            core_j = 2.0 * (
                np.einsum("Ppq,Pii->pq", db_aa, b_cc, optimize=True)
                + np.einsum("Ppq,Pii->pq", b_aa, db_cc, optimize=True)
            )
            core_k = (
                np.einsum("Ppi,Pqi->pq", db_ac, b_ac, optimize=True)
                + np.einsum("Ppi,Pqi->pq", b_ac, db_ac, optimize=True)
            )
            dh1_active = dh1_active + core_j - core_k
        b_act = pair_factors[:, active, active]
        db_act = dpair_factors[:, active, active]
        deri_active = (
            np.einsum("Ppq,Prs->pqrs", db_act, b_act, optimize=True)
            + np.einsum("Ppq,Prs->pqrs", b_act, db_act, optimize=True)
        )
        return dh1_active, deri_active

    def _core_energy_from_full_mo(self, h1_mo, eri_mo, ncore):
        ncore = int(ncore)
        energy = float(np.real(self.mf.energy_nuc()))
        if ncore <= 0:
            return energy
        core = slice(0, ncore)
        h1_core = np.asarray(h1_mo)[core, core]
        eri_core = np.asarray(eri_mo)[core, core, core, core]
        energy += 2.0 * float(np.real(np.trace(h1_core)))
        energy += 2.0 * float(np.real(np.einsum("iijj->", eri_core, optimize=True)))
        energy -= float(np.real(np.einsum("ijji->", eri_core, optimize=True)))
        return energy

    def _core_energy_from_full_mo_factors(self, h1_mo, pair_factors, ncore):
        ncore = int(ncore)
        energy = float(np.real(self.mf.energy_nuc()))
        if ncore <= 0:
            return energy
        core = slice(0, ncore)
        h1_core = np.asarray(h1_mo)[core, core]
        pair_core = np.asarray(pair_factors)[:, core, core]
        energy += 2.0 * float(np.real(np.trace(h1_core)))
        energy += 2.0 * float(np.real(np.einsum("Pii,Pjj->", pair_core, pair_core, optimize=True)))
        energy -= float(np.real(np.einsum("Pij,Pji->", pair_core, pair_core, optimize=True)))
        return energy

    def _core_energy_derivative_factors(self, dh1_mo, pair_factors, dpair_factors, ncore):
        ncore = int(ncore)
        if ncore <= 0:
            return 0.0
        core = slice(0, ncore)
        dh1_core = np.asarray(dh1_mo)[core, core]
        b_core = np.asarray(pair_factors)[:, core, core]
        db_core = np.asarray(dpair_factors)[:, core, core]
        out = 2.0 * float(np.real(np.trace(dh1_core)))
        out += 2.0 * float(
            np.real(
                np.einsum("Pii,Pjj->", db_core, b_core, optimize=True)
                + np.einsum("Pii,Pjj->", b_core, db_core, optimize=True)
            )
        )
        out -= float(
            np.real(
                np.einsum("Pij,Pji->", db_core, b_core, optimize=True)
                + np.einsum("Pij,Pji->", b_core, db_core, optimize=True)
            )
        )
        return out

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

    def _full_mo_factor_derivatives(self, h1_mo, pair_factors, kappa):
        """
        Differentiate frozen full-MO one-electron integrals and MO-pair CD factors.
        """
        h1_mo = np.asarray(h1_mo)
        pair_factors = np.asarray(pair_factors)
        kappa = np.asarray(kappa)
        dh1 = (
            np.einsum("pi,pj->ij", kappa, h1_mo, optimize=True)
            + np.einsum("qj,iq->ij", kappa, h1_mo, optimize=True)
        )
        dpair = (
            np.einsum("pi,Ppj->Pij", kappa, pair_factors, optimize=True)
            + np.einsum("qj,Piq->Pij", kappa, pair_factors, optimize=True)
        )
        return dh1, dpair

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

        nvar = self._pack_orbitals(
            np.zeros((self.nmo, self.nmo)),
            mc.ncore,
            mc.ncas,
            self.nmo,
        ).size
        dh1_cols = []
        deri_cols = []
        eye = np.eye(nvar)
        for iorb in range(nvar):
            kappa = self._unpack_orbitals(eye[:, iorb], mc.ncore, mc.ncas, self.nmo)
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

    def _active_integral_derivative_basis_factors(self, mc, h1_mo, pair_factors):
        key = (
            "factor",
            id(h1_mo),
            id(pair_factors),
            int(mc.ncore),
            int(mc.ncas),
            int(self.nmo),
        )
        if (
            self._full_derivative_cache is not None
            and self._full_derivative_cache.get("key") == key
        ):
            return (
                self._full_derivative_cache["dh1"],
                self._full_derivative_cache["deri"],
            )

        nvar = self._pack_orbitals(
            np.zeros((self.nmo, self.nmo)),
            mc.ncore,
            mc.ncas,
            self.nmo,
        ).size
        dh1_cols = []
        deri_cols = []
        eye = np.eye(nvar)
        for iorb in range(nvar):
            kappa = self._unpack_orbitals(eye[:, iorb], mc.ncore, mc.ncas, self.nmo)
            dh1, dpair = self._full_mo_factor_derivatives(h1_mo, pair_factors, kappa)
            h1_active, eri_active = self._active_integral_derivatives_from_factor_step(
                dh1,
                pair_factors,
                dpair,
                mc.ncore,
                mc.ncas,
            )
            dh1_cols.append(h1_active)
            deri_cols.append(eri_active)
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

        nvar = self._pack_orbitals(
            np.zeros((self.nmo, self.nmo)),
            mc.ncore,
            mc.ncas,
            self.nmo,
        ).size
        vals = []
        eye = np.eye(nvar)
        for iorb in range(nvar):
            kappa = self._unpack_orbitals(eye[:, iorb], mc.ncore, mc.ncas, self.nmo)
            dh1, deri = self._full_mo_integral_derivatives(h1_mo, eri_mo, kappa)
            vals.append(self._core_energy_derivative(dh1, deri, mc.ncore))
        vals = np.asarray(vals, dtype=float)
        if self._full_derivative_cache is None:
            self._full_derivative_cache = {}
        self._full_derivative_cache["core_key"] = key
        self._full_derivative_cache["de_core"] = vals
        return vals

    def _core_energy_derivative_basis_factors(self, mc, h1_mo, pair_factors):
        key = (
            "factor",
            id(h1_mo),
            id(pair_factors),
            int(mc.ncore),
            int(mc.ncas),
            int(self.nmo),
            "core",
        )
        if (
            self._full_derivative_cache is not None
            and self._full_derivative_cache.get("core_key") == key
        ):
            return self._full_derivative_cache["de_core"]

        nvar = self._pack_orbitals(
            np.zeros((self.nmo, self.nmo)),
            mc.ncore,
            mc.ncas,
            self.nmo,
        ).size
        vals = []
        eye = np.eye(nvar)
        for iorb in range(nvar):
            kappa = self._unpack_orbitals(eye[:, iorb], mc.ncore, mc.ncas, self.nmo)
            dh1, dpair = self._full_mo_factor_derivatives(h1_mo, pair_factors, kappa)
            vals.append(
                self._core_energy_derivative_factors(
                    dh1,
                    pair_factors,
                    dpair,
                    mc.ncore,
                )
            )
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
        sigma_basis = self._batched_derivative_sigma(mc, dh1_basis, deri_basis, c0)
        self._full_derivative_sigma_cache = {
            "key": key,
            "sigma": sigma_basis,
        }
        return sigma_basis

    def _batched_derivative_sigma(self, mc, dh1_basis, deri_basis, c0):
        if dh1_basis.shape[0] == 0:
            return np.zeros((0, np.asarray(c0).size), dtype=float)
        if getattr(mc, "direct_connectivity", None) is None:
            mc.direct_connectivity = build_direct_connectivity(mc.binary)
        conn = mc.direct_connectivity
        try:
            return _sigma_compact_derivative_batch_numba(
                np.ascontiguousarray(dh1_basis),
                np.ascontiguousarray(deri_basis),
                np.ascontiguousarray(c0),
                mc.binary,
                conn.I_A, conn.J_A, conn.p_A, conn.q_A, conn.phase_A,
                conn.I_B, conn.J_B, conn.p_B, conn.q_B, conn.phase_B,
                conn.I_AA, conn.J_AA, conn.p_AA, conn.q_AA, conn.r_AA, conn.s_AA, conn.phase_AA,
                conn.I_BB, conn.J_BB, conn.p_BB, conn.q_BB, conn.r_BB, conn.s_BB, conn.phase_BB,
                conn.I_AB, conn.J_AB, conn.p_AB, conn.q_AB, conn.r_AB, conn.s_AB, conn.phase_AB,
            )
        except Exception:
            cols = []
            for iorb in range(dh1_basis.shape[0]):
                deriv_mc = self._make_active_sigma_casci(
                    mc,
                    dh1_basis[iorb],
                    deri_basis[iorb],
                )
                cols.append(deriv_mc.ci_sigma(c0))
            return np.asarray(cols)

    def _factor_derivative_sigma_basis(self, mc, h1_mo, pair_factors, c0):
        dh1_basis, deri_basis = self._active_integral_derivative_basis_factors(
            mc,
            h1_mo,
            pair_factors,
        )
        key = (
            "factor",
            id(h1_mo),
            id(pair_factors),
            id(c0),
            int(mc.ncore),
            int(mc.ncas),
            int(self.nmo),
        )
        if (
            self._full_derivative_sigma_cache is not None
            and self._full_derivative_sigma_cache.get("key") == key
        ):
            return self._full_derivative_sigma_cache["sigma"]
        sigma_basis = self._batched_derivative_sigma(mc, dh1_basis, deri_basis, c0)
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

    def _factor_exact_orbital_gradient_vector(self, mc, h1_mo, pair_factors, ci):
        ci = np.asarray(ci, dtype=float)
        sigma_basis = self._factor_derivative_sigma_basis(mc, h1_mo, pair_factors, ci)
        grad = sigma_basis @ ci
        grad = grad + self._core_energy_derivative_basis_factors(
            mc,
            h1_mo,
            pair_factors,
        )
        return np.asarray(grad, dtype=float)

    def _gradient_matrix_from_vector(self, grad_vec, ncore, ncas, nmo):
        return self._unpack_orbitals(grad_vec, ncore, ncas, nmo)

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
        sigma_mc.e_core = self._core_energy_from_full_mo(h1_mo, eri_mo, mc.ncore)
        return sigma_mc

    def _cached_integral_sigma_casci(self, mc, h1_mo, eri_mo):
        key = (id(mc), id(h1_mo), id(eri_mo))
        cached = self._joint_trial_sigma_cache.get(key)
        if cached is not None:
            return cached
        sigma_mc = self._make_integral_sigma_casci(mc, h1_mo, eri_mo)
        self._joint_trial_sigma_cache[key] = sigma_mc
        return sigma_mc

    def _make_factor_integral_sigma_casci(self, mc, h1_mo, pair_factors):
        h1_active, pair_active = self._active_h1_pair_from_full_mo_factors(
            h1_mo,
            pair_factors,
            mc.ncore,
            mc.ncas,
        )
        sigma_mc = self._make_active_factor_sigma_casci(mc, h1_active, pair_active)
        sigma_mc.e_core = self._core_energy_from_full_mo_factors(
            h1_mo,
            pair_factors,
            mc.ncore,
        )
        return sigma_mc

    def _cached_factor_integral_sigma_casci(self, mc, h1_mo, pair_factors):
        key = ("factor", id(mc), id(h1_mo), id(pair_factors))
        cached = self._joint_trial_sigma_cache.get(key)
        if cached is not None:
            return cached
        sigma_mc = self._make_factor_integral_sigma_casci(mc, h1_mo, pair_factors)
        self._joint_trial_sigma_cache[key] = sigma_mc
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

    def _make_active_factor_sigma_casci(self, mc, h1_active, pair_active):
        """
        Lightweight CASCI-like object for CI sigma with active-space pair factors.
        """
        sigma_mc = copy.copy(mc)
        h1_active = np.asarray(h1_active)
        pair_active = np.asarray(pair_active)
        sigma_mc.hcore = np.asarray([h1_active, h1_active])
        sigma_mc.h2e_cas = None
        sigma_mc.eri_so = None
        sigma_mc._direct_spatial_h1 = h1_active
        sigma_mc._direct_spatial_eri = None
        sigma_mc._direct_same_spin_eri = None
        sigma_mc._direct_cross_spin_eri = None
        sigma_mc._direct_pair_factors = pair_active
        sigma_mc.direct_connectivity = mc.direct_connectivity
        sigma_mc.binary = mc.binary
        if sigma_mc.direct_connectivity is None:
            sigma_mc.direct_connectivity = build_direct_connectivity(sigma_mc.binary)
        conn = sigma_mc.direct_connectivity
        sigma_mc._direct_factor_H_diag = _compute_diag_compact_factors(
            h1_active,
            pair_active,
            sigma_mc.binary,
        )
        sigma_mc._direct_factor_H_A = _compute_single_values_from_factors(
            conn.J_A,
            conn.p_A,
            conn.q_A,
            conn.phase_A,
            h1_active,
            pair_active,
            sigma_mc.binary,
            0,
        )
        sigma_mc._direct_factor_H_B = _compute_single_values_from_factors(
            conn.J_B,
            conn.p_B,
            conn.q_B,
            conn.phase_B,
            h1_active,
            pair_active,
            sigma_mc.binary,
            1,
        )
        sigma_mc._direct_factor_H_AA = _compute_double_same_values_from_factors(
            conn.p_AA,
            conn.q_AA,
            conn.r_AA,
            conn.s_AA,
            conn.phase_AA,
            pair_active,
        )
        sigma_mc._direct_factor_H_BB = _compute_double_same_values_from_factors(
            conn.p_BB,
            conn.q_BB,
            conn.r_BB,
            conn.s_BB,
            conn.phase_BB,
            pair_active,
        )
        sigma_mc._direct_factor_H_AB = _compute_double_cross_values_from_factors(
            conn.p_AB,
            conn.q_AB,
            conn.r_AB,
            conn.s_AB,
            conn.phase_AB,
            pair_active,
        )
        return sigma_mc

    def _orbital_gradient_from_ci_response(self, mc, h1_mo, eri_mo, c0, dc):
        dc = self._project_ci_response(dc, [c0])
        if np.linalg.norm(dc) <= 1.0e-14:
            nvar = self._pack_orbitals(
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
        return self._pack_orbitals(
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

    def _factor_ci_gradient_from_orbital_response(
        self,
        mc,
        h1_mo,
        pair_factors,
        c0,
        orb_step,
    ):
        orb_step = np.asarray(orb_step, dtype=float)
        if orb_step.size == 0:
            return np.zeros_like(c0, dtype=float)
        sigma_basis = self._factor_derivative_sigma_basis(mc, h1_mo, pair_factors, c0)
        vec = np.tensordot(orb_step, sigma_basis, axes=(0, 0))
        return self._project_ci_response(vec, [c0])

    def _orbital_gradient_from_ci_response_adjoint(self, mc, h1_mo, eri_mo, c0, dc):
        dc = self._project_ci_response(dc, [c0])
        if np.linalg.norm(dc) <= 1.0e-14:
            nvar = self._pack_orbitals(
                np.zeros((self.nmo, self.nmo)),
                mc.ncore,
                mc.ncas,
                self.nmo,
            ).size
            return np.zeros(nvar, dtype=float)
        sigma_basis = self._derivative_sigma_basis(mc, h1_mo, eri_mo, c0)
        return sigma_basis @ dc

    def _factor_orbital_gradient_from_ci_response_adjoint(
        self,
        mc,
        h1_mo,
        pair_factors,
        c0,
        dc,
    ):
        dc = self._project_ci_response(dc, [c0])
        if np.linalg.norm(dc) <= 1.0e-14:
            nvar = self._pack_orbitals(
                np.zeros((self.nmo, self.nmo)),
                mc.ncore,
                mc.ncas,
                self.nmo,
            ).size
            return np.zeros(nvar, dtype=float)
        sigma_basis = self._factor_derivative_sigma_basis(mc, h1_mo, pair_factors, c0)
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
        return_info=False,
        pair_factors=None,
    ):
        """
        Matrix-free coupled orbital/full-CI-response AH step.
        """
        nroots = max(1, int(self.nstates))
        c_roots = [np.asarray(root, dtype=float) for root in mc.ci[:nroots]]
        if not c_roots:
            out = (np.asarray(fallback_step, dtype=float), self._copy_ci_guess(mc.ci[:nroots]))
            if return_info:
                return out[0], out[1], {"model_reduction": np.nan}
            return out
        if self.weights is None:
            root_weights = np.zeros(len(c_roots), dtype=float)
            root_weights[min(self.state_id, len(c_roots) - 1)] = 1.0
        else:
            root_weights = np.asarray(self.weights, dtype=float)[: len(c_roots)]
            root_weights = root_weights / float(np.sum(root_weights))
        ndet = c_roots[0].size
        n_orb = grad_vec.size
        if ndet == 0 or n_orb == 0:
            out = (np.asarray(fallback_step, dtype=float), self._copy_ci_guess(mc.ci[:nroots]))
            if return_info:
                return out[0], out[1], {"model_reduction": np.nan}
            return out

        active_energies = np.asarray(mc.e_tot[: len(c_roots)], dtype=float) - float(mc.e_core)
        ci_diag = np.asarray(ci_diagonal(mc), dtype=float)
        ci_hdiags = [ci_diag - float(active_energy) for active_energy in active_energies]
        orb_hdiag = np.maximum(np.abs(np.asarray(hess_diag, dtype=float)), self.level_shift)
        ci_precond = [
            np.maximum(abs(float(weight)) * np.abs(ci_hdiag), self.level_shift)
            for weight, ci_hdiag in zip(root_weights, ci_hdiags)
        ]
        precond_diag = np.concatenate((orb_hdiag, *ci_precond))
        total_grad = np.concatenate(
            (
                np.asarray(grad_vec, dtype=float),
                np.zeros(ndet * len(c_roots), dtype=float),
            )
        )

        def split(vec):
            vec = np.asarray(vec, dtype=float)
            orb = vec[:n_orb]
            ci_flat = vec[n_orb:]
            ci_parts = []
            for root in range(len(c_roots)):
                start = root * ndet
                stop = start + ndet
                ci_parts.append(
                    self._project_ci_response(ci_flat[start:stop], c_roots)
                )
            return orb, ci_parts

        def matvec(vec):
            orb_part, ci_parts = split(vec)
            out_orb = np.asarray(hessian_action(orb_part), dtype=float)
            out_ci_parts = []
            for weight, c0, ci_part, active_energy in zip(
                root_weights,
                c_roots,
                ci_parts,
                active_energies,
            ):
                if pair_factors is None:
                    out_orb += float(weight) * self._orbital_gradient_from_ci_response_adjoint(
                        mc,
                        h1_mo,
                        eri_mo,
                        c0,
                        ci_part,
                    )
                    out_ci = mc.ci_sigma(ci_part) - float(active_energy) * ci_part
                    out_ci += self._ci_gradient_from_orbital_response(
                        mc,
                        h1_mo,
                        eri_mo,
                        c0,
                        orb_part,
                    )
                else:
                    out_orb += float(weight) * self._factor_orbital_gradient_from_ci_response_adjoint(
                        mc,
                        h1_mo,
                        pair_factors,
                        c0,
                        ci_part,
                    )
                    out_ci = mc.ci_sigma(ci_part) - float(active_energy) * ci_part
                    out_ci += self._factor_ci_gradient_from_orbital_response(
                        mc,
                        h1_mo,
                        pair_factors,
                        c0,
                        orb_part,
                    )
                out_ci_parts.append(
                    float(weight) * self._project_ci_response(out_ci, c_roots)
                )
            return np.concatenate((out_orb, *out_ci_parts))

        seed = np.concatenate(
            (
                np.asarray(fallback_step, dtype=float),
                np.zeros(ndet * len(c_roots), dtype=float),
            )
        )
        diag_step = -total_grad / precond_diag
        seeds = []
        if (
            self.coupled_reuse_subspace
            and
            self._full_coupled_seed is not None
            and np.asarray(self._full_coupled_seed).shape == total_grad.shape
        ):
            seeds.append(np.asarray(self._full_coupled_seed, dtype=float))
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
                col = np.concatenate((orb, *ci))
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
                orb_raw, _ = split(raw_step)
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
                model_linear = float(np.dot(total_grad, step))
                model_quadratic = float(np.dot(step, hv))
                residual = alpha * total_grad + raw_hv - float(eigvals[root]) * raw_step
                scalar_residual = float(np.dot(total_grad, raw_step) - float(eigvals[root]) * alpha)
                residual_norm = float(
                    np.sqrt(np.dot(residual, residual) + scalar_residual ** 2)
                )
                candidate = {
                    "model": model,
                    "model_linear": model_linear,
                    "model_quadratic": model_quadratic,
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
            out = (np.asarray(fallback_step, dtype=float), self._copy_ci_guess(mc.ci[:1]))
            if return_info:
                return out[0], out[1], {"model_reduction": np.nan}
            return out

        orb_step, ci_steps = split(best["step"])
        if self.coupled_reuse_subspace:
            self._full_coupled_seed = np.asarray(best["step"], dtype=float).copy()
        ci_guess = self._orthonormalize_ci_roots(
            [
                c0 + ci_step
                for c0, ci_step in zip(c_roots, ci_steps)
            ],
            fallback_roots=c_roots,
        )
        info = {
            "model": float(best["model"]),
            "model_reduction": float(-best["model"]),
            "model_linear": float(best["model_linear"]),
            "model_quadratic": float(best["model_quadratic"]),
            "residual_norm": float(best["residual_norm"]),
            "eigenvalue": float(best["eigenvalue"]),
            "joint_step_norm": float(np.linalg.norm(best["step"])),
            "ci_step_norm": float(
                np.sqrt(sum(float(np.dot(step, step)) for step in ci_steps))
            ),
        }
        if return_info:
            return np.asarray(orb_step, dtype=float), ci_guess, info
        return np.asarray(orb_step, dtype=float), ci_guess

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

    def _orbital_pspace_guess(self, grad_vec, hess_diag, primary_step=None):
        """
        Build a rotational P-space seed for the orbital AH solver.

        The paper's improved orbital optimizer explicitly includes the most
        critical rotations: first those with negative Hessian diagonals, then
        the largest ``|g_i / h_ii|`` rotations.  We use the same criterion to
        seed the Davidson AH subspace.
        """
        grad_vec = np.asarray(grad_vec, dtype=float)
        hess_diag = np.asarray(hess_diag, dtype=float)
        nvar = grad_vec.size
        columns = []
        selected = set()

        if primary_step is not None:
            step = np.asarray(primary_step, dtype=float)
            if step.shape == grad_vec.shape:
                norm = np.linalg.norm(step)
                if norm > 1.0e-14:
                    columns.append(step / norm)

        max_size = min(max(0, int(self.ah_pspace_size)), nvar)
        if max_size == 0 or nvar == 0:
            return None if not columns else np.column_stack(columns)

        safe = np.where(
            np.abs(hess_diag) > self.level_shift,
            np.abs(hess_diag),
            self.level_shift,
        )
        ranked = list(np.argsort(np.abs(grad_vec) / safe)[::-1])
        negative = list(np.flatnonzero(hess_diag < 0.0))

        for idx in negative + ranked:
            idx = int(idx)
            if idx in selected:
                continue
            selected.add(idx)
            col = np.zeros(nvar, dtype=float)
            col[idx] = 1.0
            columns.append(col)
            if len(selected) >= max_size:
                break

        if not columns:
            return None
        return np.column_stack(columns)

    def _core_active_mask(self, ncore, ncas, nmo):
        pairs = nonredundant_pairs(ncore, ncas, nmo)
        return np.asarray(
            [
                p < ncore and ncore <= q < ncore + ncas
                for p, q in pairs
            ],
            dtype=bool,
        )

    def _internal_preopt_mask(self, ncore, ncas, nmo):
        pairs = nonredundant_pairs(ncore, ncas, nmo)
        if self.internal_preopt_space == "nonredundant":
            return np.ones(len(pairs), dtype=bool)
        return self._core_active_mask(ncore, ncas, nmo)

    def _internal_preopt_evaluate(self, mo_coeff, ci_guess, solve_nstates=None):
        if solve_nstates is None:
            solve_nstates = self.nstates
        mc = self._make_casci(mo_coeff, nstates=solve_nstates, ci0=ci_guess)
        energy = self._objective_energy(mc, self.state_id)
        h1_mo = self.mf.get_hcore_mo(mo_coeff)
        eri_mo = self.mf.get_eri_mo(mo_coeff, notation="chem")
        dm1, dm2 = self._effective_rdms(mc, self.state_id)
        fock = generalized_fock(h1_mo, eri_mo, dm1, dm2)
        if self.nstates == 1:
            grad_vec = self._exact_orbital_gradient_vector(
                mc,
                h1_mo,
                eri_mo,
                mc.ci[self.state_id],
            )
        else:
            grad = orbital_gradient(fock)
            grad_vec = pack_nonredundant(grad, mc.ncore, mc.ncas, self.nmo)
        return mc, energy, fock, grad_vec

    def _internal_preopt_finite_difference_hessian(
        self,
        mo_coeff,
        mc,
        ci_guess,
        mask,
    ):
        internal_idx = np.flatnonzero(mask)
        if internal_idx.size == 0:
            return np.zeros((0, 0), dtype=float)
        eps = self.ah_fd_step
        if eps <= 0.0:
            raise ValueError("ah_fd_step must be positive.")

        cols = []
        eye = np.eye(mask.size)
        for idx in internal_idx:
            kappa = unpack_nonredundant(
                eye[:, idx],
                mc.ncore,
                mc.ncas,
                self.nmo,
            )
            plus_mo = self._apply_orbital_update(mo_coeff, eps * kappa)
            minus_mo = self._apply_orbital_update(mo_coeff, -eps * kappa)
            _, _, _, grad_plus = self._internal_preopt_evaluate(
                plus_mo,
                ci_guess,
            )
            _, _, _, grad_minus = self._internal_preopt_evaluate(
                minus_mo,
                ci_guess,
            )
            cols.append((grad_plus[mask] - grad_minus[mask]) / (2.0 * eps))
        hess = np.column_stack(cols)
        return 0.5 * (hess + hess.T)

    def _internal_preopt_analytic_hessian_action(
        self,
        mc,
        h1_mo,
        eri_mo,
        grad_vec,
        mask,
        vec,
    ):
        dm1, dm2 = self._effective_rdms(mc, self.state_id)
        full_vec = np.zeros_like(grad_vec)
        full_vec[mask] = np.asarray(vec, dtype=float)
        out = self._analytic_orbital_hessian_action(
            h1_mo,
            eri_mo,
            dm1,
            dm2,
            mc,
            full_vec,
        )
        return np.asarray(out[mask], dtype=float)

    def _internal_preopt_analytic_hessian(
        self,
        mc,
        h1_mo,
        eri_mo,
        grad_vec,
        mask,
    ):
        internal_grad = grad_vec[mask]

        def hessian_action_internal(vec):
            return self._internal_preopt_analytic_hessian_action(
                mc,
                h1_mo,
                eri_mo,
                grad_vec,
                mask,
                vec,
            )

        return self._dense_orbital_hessian(internal_grad, hessian_action_internal)

    def _internal_preopt_coupled_step(
        self,
        mc,
        mo_coeff,
        h1_mo,
        eri_mo,
        grad_vec,
        hess_diag,
        mask,
        diag_step,
        max_step,
        use_fd_orbital_hessian=False,
    ):
        """
        Reduced CI-orbital coupled internal-preopt step.

        This builds a dense orbital Hessian in the selected internal orbital
        subspace, relaxes it by eliminating reduced CI rotations, then embeds
        the resulting orbital step back into the full nonredundant vector.
        """
        nroots = max(1, self.nstates)
        if len(mc.ci) <= nroots:
            return diag_step, self._copy_ci_guess(mc.ci[:nroots]), None

        weights = self.weights
        if weights is None:
            weights = np.ones(nroots, dtype=float) / float(nroots)
        selected = np.flatnonzero(mask)
        if selected.size == 0:
            return diag_step, self._copy_ci_guess(mc.ci[:nroots]), None

        dm1, dm2 = self._effective_rdms(mc, self.state_id)

        if use_fd_orbital_hessian:
            orb_hess = self._internal_preopt_finite_difference_hessian(
                mo_coeff,
                mc,
                mc.ci[:nroots],
                mask,
            )
            internal_grad = grad_vec[mask]
            orbital_hessian_model = "finite_difference"
        else:
            def hessian_action_internal(vec):
                full_vec = np.zeros_like(grad_vec)
                full_vec[mask] = vec
                out = self._analytic_orbital_hessian_action(
                    h1_mo,
                    eri_mo,
                    dm1,
                    dm2,
                    mc,
                    full_vec,
                )
                return out[mask]

            internal_grad = grad_vec[mask]
            orb_hess = self._dense_orbital_hessian(
                internal_grad,
                hessian_action_internal,
            )
            orbital_hessian_model = "analytic"
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
        ci_guess_mat = np.column_stack(mc.ci[:nroots])
        q_cycles = max(1, self.coupled_qspace_cycles)
        internal_step = np.asarray(diag_step[mask], dtype=float)
        info = {
            "ci_dim": 0,
            "q_cycles": 0,
            "relaxed_min_eig": np.nan,
            "orbital_hessian": orbital_hessian_model,
            "orbital_min_eig": (
                float(np.min(np.linalg.eigvalsh(orb_hess))) if orb_hess.size else np.nan
            ),
        }

        for q_cycle in range(q_cycles):
            ci_grad, ci_pairs = subspace.rotation_gradient(
                nstates=nroots,
                weights=weights,
            )
            ci_hess, hess_pairs = subspace.rotation_hessian(
                nstates=nroots,
                weights=weights,
            )
            hoc_full, coupling_pairs = subspace.orbital_coupling(
                mc,
                h1_mo,
                eri_mo,
                nstates=nroots,
                weights=weights,
                nmo=self.nmo,
            )
            if ci_pairs != hess_pairs or ci_pairs != coupling_pairs or ci_grad.size == 0:
                break
            hoc = hoc_full[mask, :]
            relaxed_hess, solve_ci_step = self._ci_relaxed_orbital_hessian(
                orb_hess,
                ci_hess,
                hoc,
            )
            internal_step = self._dense_augmented_hessian_step(
                internal_grad,
                relaxed_hess,
                max_step=max_step,
                fallback_step=diag_step[mask],
            )
            if np.dot(internal_step, internal_grad) >= 0.0:
                internal_step = np.asarray(diag_step[mask], dtype=float)
            ci_step = solve_ci_step(internal_step)
            ci_guess_mat = subspace.rotated_state_vectors(
                ci_step,
                ci_pairs,
                nstates=nroots,
            )
            info["ci_dim"] = int(ci_hess.shape[0])
            info["q_cycles"] = int(q_cycle + 1)
            if relaxed_hess.size:
                info["relaxed_min_eig"] = float(np.min(np.linalg.eigvalsh(relaxed_hess)))

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

        full_step = np.zeros_like(grad_vec)
        full_step[mask] = internal_step
        return full_step, [ci_guess_mat[:, i].copy() for i in range(ci_guess_mat.shape[1])], info

    def _internal_preopt_solve_nstates(self):
        solve_nstates = self.nstates
        if self.internal_preopt_hessian in {"coupled", "coupled_fd"}:
            solve_nstates += max(1, self.coupled_ci_roots)
        return solve_nstates

    def _internal_preopt_step(self, mo_coeff, ci_guess, macro, internal_cycle=None):
        """
        Run one exact-energy core-active orbital preoptimization step.

        The second-order model can underestimate inactive-active rotations.
        This restricted line search resolves CASCI at every trial point, so it
        only changes the macro reference when the actual CASSCF energy drops.
        """
        solve_nstates = self._internal_preopt_solve_nstates()
        mc, energy, fock, grad_vec = self._internal_preopt_evaluate(
            mo_coeff,
            ci_guess,
            solve_nstates=solve_nstates,
        )
        h1_mo = self.mf.get_hcore_mo(mo_coeff)
        eri_mo = self.mf.get_eri_mo(mo_coeff, notation="chem")

        mask = self._internal_preopt_mask(mc.ncore, mc.ncas, self.nmo)
        if grad_vec.size == 0 or not np.any(mask):
            return False, mo_coeff, energy, mc, 0.0, 0.0

        internal_grad = grad_vec[mask]
        gnorm = float(np.linalg.norm(internal_grad)) if internal_grad.size else 0.0
        if gnorm < self.conv_tol_grad:
            return False, mo_coeff, energy, mc, gnorm, 0.0

        hess_diag = diagonal_hessian(
            fock,
            mc.ncore,
            mc.ncas,
            level_shift=self.level_shift,
        )
        internal_hdiag = np.maximum(
            np.abs(hess_diag[mask]),
            self.level_shift,
        )
        diag_step = -internal_grad / internal_hdiag
        step_limit = self.max_step
        if self.internal_preopt_max_step is not None:
            step_limit = min(step_limit, self.internal_preopt_max_step)
        coupled_info = None
        coupled_ci_guess = None
        if self.internal_preopt_hessian in {"coupled", "coupled_fd"}:
            full_diag_step = np.zeros_like(grad_vec)
            full_diag_step[mask] = diag_step
            full_step, coupled_ci_guess, coupled_info = self._internal_preopt_coupled_step(
                mc,
                mo_coeff,
                h1_mo,
                eri_mo,
                grad_vec,
                hess_diag,
                mask,
                full_diag_step,
                step_limit,
                use_fd_orbital_hessian=self.internal_preopt_hessian == "coupled_fd",
            )
            internal_hess = None
            internal_step = full_step[mask]
            internal_solver = "dense_coupled"
            internal_solver_info = {}
        elif (
            self.internal_preopt_hessian == "analytic"
            and self.internal_preopt_solver == "davidson"
        ):
            internal_hess = None

            def hessian_action_internal(vec):
                return self._internal_preopt_analytic_hessian_action(
                    mc,
                    h1_mo,
                    eri_mo,
                    grad_vec,
                    mask,
                    vec,
                )

            internal_step, internal_solver_info = davidson_augmented_hessian_direction(
                internal_grad,
                hess_diag[mask],
                matvec=hessian_action_internal,
                max_step=step_limit,
                regularization=self.level_shift,
                max_cycle=self.ah_max_cycle,
                max_subspace=max(self.ah_max_subspace, min(2, internal_grad.size)),
                tol=max(self.conv_tol_grad_relaxed, 1.0e-8),
                fallback_step=diag_step,
                return_info=True,
            )
            internal_solver = "davidson"
        elif self.internal_preopt_hessian == "analytic":
            internal_hess = self._internal_preopt_analytic_hessian(
                mc,
                h1_mo,
                eri_mo,
                grad_vec,
                mask,
            )
            internal_step = self._dense_augmented_hessian_step(
                internal_grad,
                internal_hess,
                max_step=step_limit,
                fallback_step=diag_step,
            )
            internal_solver = "dense"
            internal_solver_info = {}
        elif self.internal_preopt_hessian == "finite_difference":
            internal_hess = self._internal_preopt_finite_difference_hessian(
                mo_coeff,
                mc,
                mc.ci[: self.nstates],
                mask,
            )
            internal_step = self._dense_augmented_hessian_step(
                internal_grad,
                internal_hess,
                max_step=step_limit,
                fallback_step=diag_step,
            )
            internal_solver = "dense"
            internal_solver_info = {}
        else:
            internal_hess = None
            internal_step = diag_step
            internal_solver = "diagonal"
            internal_solver_info = {}
        internal_step = limit_step_norm(internal_step, step_limit)
        if np.dot(internal_step, internal_grad) >= -1.0e-14:
            internal_step = limit_step_norm(diag_step, step_limit)
        if np.dot(internal_step, internal_grad) >= -1.0e-14:
            return False, mo_coeff, energy, mc, gnorm, 0.0

        full_step = np.zeros_like(grad_vec)
        full_step[mask] = internal_step
        kappa = unpack_nonredundant(
            full_step,
            mc.ncore,
            mc.ncas,
            self.nmo,
            max_step=step_limit,
        )
        accepted, trial_coeff, trial_energy, accepted_scale, trial_mc = self._line_search(
            mo_coeff,
            kappa,
            energy,
            ci0=coupled_ci_guess if coupled_ci_guess is not None else mc.ci[: self.nstates],
            start_scale=1.0,
            min_scale=0.125,
            accept_delta=0.0,
        )
        used_fallback = False
        if not accepted and coupled_info is not None:
            fallback_internal = limit_step_norm(diag_step, step_limit)
            if np.dot(fallback_internal, internal_grad) < -1.0e-14:
                fallback_full = np.zeros_like(grad_vec)
                fallback_full[mask] = fallback_internal
                fallback_kappa = unpack_nonredundant(
                    fallback_full,
                    mc.ncore,
                    mc.ncas,
                    self.nmo,
                    max_step=step_limit,
                )
                (
                    fallback_accepted,
                    fallback_coeff,
                    fallback_energy,
                    fallback_scale,
                    fallback_mc,
                ) = self._line_search(
                    mo_coeff,
                    fallback_kappa,
                    energy,
                    ci0=mc.ci[: self.nstates],
                    start_scale=1.0,
                    min_scale=0.125,
                    accept_delta=0.0,
                )
                if fallback_accepted:
                    accepted = fallback_accepted
                    trial_coeff = fallback_coeff
                    trial_energy = fallback_energy
                    accepted_scale = fallback_scale
                    trial_mc = fallback_mc
                    full_step = fallback_full
                    used_fallback = True
        step_norm = (
            float(accepted_scale * np.max(np.abs(full_step)))
            if full_step.size > 0
            else 0.0
        )
        self.internal_preopt_history.append(
            {
                "macro": int(macro),
                "internal_cycle": (
                    int(internal_cycle) if internal_cycle is not None else len(self.internal_preopt_history)
                ),
                "internal_optimization": bool(self.internal_optimization),
                "energy": float(energy),
                "trial_energy": float(trial_energy),
                "accepted": bool(accepted),
                "gradient_norm": float(gnorm),
                "step_norm": float(step_norm),
                "hessian": self.internal_preopt_hessian,
                "solver": internal_solver,
                "space": self.internal_preopt_space,
                "hessian_dim": int(internal_grad.size),
                "solver_converged": bool(
                    internal_solver_info.get("converged", False)
                ),
                "solver_iterations": int(
                    internal_solver_info.get("iterations", 0)
                ),
                "solver_residual_norm": float(
                    internal_solver_info.get("residual_norm", np.nan)
                ),
                "solver_subspace_dim": int(
                    internal_solver_info.get("subspace_dim", 0)
                ),
                "hessian_min_eig": (
                    float(np.min(np.linalg.eigvalsh(internal_hess)))
                    if internal_hess is not None and internal_hess.size
                    else np.nan
                ),
                "coupled_ci_dim": (
                    int(coupled_info["ci_dim"]) if coupled_info is not None else 0
                ),
                "coupled_q_cycles": (
                    int(coupled_info["q_cycles"]) if coupled_info is not None else 0
                ),
                "coupled_relaxed_min_eig": (
                    float(coupled_info["relaxed_min_eig"])
                    if coupled_info is not None
                    else np.nan
                ),
                "coupled_orbital_hessian": (
                    str(coupled_info["orbital_hessian"])
                    if coupled_info is not None
                    else ""
                ),
                "coupled_orbital_min_eig": (
                    float(coupled_info["orbital_min_eig"])
                    if coupled_info is not None
                    else np.nan
                ),
                "coupled_fallback_diagonal": bool(used_fallback),
            }
        )
        if not accepted or trial_mc is None:
            return False, mo_coeff, energy, mc, gnorm, step_norm
        return True, trial_coeff, float(trial_energy), trial_mc, gnorm, step_norm

    def _internal_preopt_preview_energy(self, mo_coeff, guard_cycles):
        trial = self._restart_solver()
        trial.max_cycle = int(guard_cycles)
        trial.max_cycles = int(guard_cycles)
        trial.internal_preopt_steps = 0
        trial.internal_preopt_guard_cycles = 0
        trial.auto_active_restarts = False
        try:
            trial.run(
                nstates=self.nstates,
                state_id=self.state_id,
                mo_coeff=mo_coeff,
            )
        except RuntimeError:
            pass
        energies = [
            float(entry["energy"])
            for entry in trial.history
            if np.isfinite(float(entry.get("energy", np.nan)))
        ]
        if trial.e_tot is not None:
            energies.append(float(np.ravel(trial.e_tot)[0]))
        if not energies:
            return np.inf
        return float(min(energies))

    def _internal_preopt_guard_accepts(self, before_mo, after_mo, record):
        guard_cycles = int(self.internal_preopt_guard_cycles)
        if guard_cycles <= 0:
            return True
        before_energy = self._internal_preopt_preview_energy(before_mo, guard_cycles)
        after_energy = self._internal_preopt_preview_energy(after_mo, guard_cycles)
        record["guard_cycles"] = guard_cycles
        record["guard_before_energy"] = float(before_energy)
        record["guard_after_energy"] = float(after_energy)
        accepted = after_energy <= before_energy + max(self.conv_tol, 1.0e-10)
        record["guard_accepted"] = bool(accepted)
        return bool(accepted)

    def _internal_preopt(self, mo_coeff, ci_guess, macro):
        enabled = self.internal_optimization or self.internal_preopt_steps > 0
        if not enabled:
            return mo_coeff, ci_guess
        max_cycle = (
            self.internal_max_cycle
            if self.internal_optimization
            else self.internal_preopt_steps
        )
        if max_cycle <= 0:
            return mo_coeff, ci_guess
        self.internal_optimization_converged = False
        local_ci = self._copy_ci_guess(ci_guess)
        for internal_cycle in range(max_cycle):
            before_mo = np.array(mo_coeff, copy=True)
            before_ci = self._copy_ci_guess(local_ci)
            accepted, trial_mo, _, mc, _, _ = self._internal_preopt_step(
                mo_coeff,
                local_ci,
                macro,
                internal_cycle=internal_cycle,
            )
            local_ci = self._copy_ci_guess(mc.ci)
            if not accepted:
                if self.internal_optimization and self.internal_preopt_history:
                    record = self.internal_preopt_history[-1]
                    record["internal_stop_reason"] = "line_search_rejected"
                break
            record = self.internal_preopt_history[-1]
            if not self._internal_preopt_guard_accepts(before_mo, trial_mo, record):
                record["accepted"] = False
                record["rejected_by_guard"] = True
                record["internal_stop_reason"] = "guard_rejected"
                local_ci = before_ci
                break
            mo_coeff = trial_mo
            if self.internal_optimization:
                post_mc, post_energy, _, post_grad = self._internal_preopt_evaluate(
                    mo_coeff,
                    local_ci,
                    solve_nstates=self._internal_preopt_solve_nstates(),
                )
                local_ci = self._copy_ci_guess(post_mc.ci)
                mask = self._internal_preopt_mask(post_mc.ncore, post_mc.ncas, self.nmo)
                if post_grad.size and np.any(mask):
                    post_gnorm = float(np.linalg.norm(post_grad[mask]))
                else:
                    post_gnorm = 0.0
                energy_drop = float(record["energy"] - record["trial_energy"])
                record["post_energy"] = float(post_energy)
                record["post_gradient_norm"] = post_gnorm
                record["energy_drop"] = energy_drop
                if post_gnorm <= self.internal_conv_tol_grad:
                    record["internal_converged"] = True
                    record["internal_stop_reason"] = "gradient"
                    self.internal_optimization_converged = True
                    break
                if float(record["step_norm"]) <= self.internal_conv_tol_step:
                    record["internal_converged"] = True
                    record["internal_stop_reason"] = "step"
                    self.internal_optimization_converged = True
                    break
                if abs(energy_drop) <= self.internal_conv_tol_energy:
                    record["internal_converged"] = True
                    record["internal_stop_reason"] = "energy"
                    self.internal_optimization_converged = True
                    break
                record["internal_converged"] = False
                record["internal_stop_reason"] = "continue"
        if self.internal_optimization and self.internal_preopt_history:
            record = self.internal_preopt_history[-1]
            if (
                record.get("macro") == int(macro)
                and record.get("internal_stop_reason") == "continue"
            ):
                record["internal_stop_reason"] = "max_cycle"
        return mo_coeff, local_ci

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
        for lower in range(1, self.active_restart_window + 1):
            low = first_active - lower
            if low < 0:
                continue
            for upper in range(1, self.active_restart_window + 1):
                high = last_active + upper
                if high >= nmo:
                    continue
                cand = default_active.copy()
                cand[0] = low
                cand[-1] = high
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
            ah_pspace_size=self.ah_pspace_size,
            ah_pspace_max_cycle=self.ah_pspace_max_cycle,
            ah_trust_metric=self.ah_trust_metric,
            ah_adaptive_trust=self.ah_adaptive_trust,
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
            coupled_accept_min_ratio=self.coupled_accept_min_ratio,
            coupled_fallback=self.coupled_fallback,
            coupled_reuse_subspace=self.coupled_reuse_subspace,
            orbital_parameterization=self.orbital_parameterization,
            internal_preopt_steps=self.internal_preopt_steps,
            internal_preopt_max_step=self.internal_preopt_max_step,
            internal_preopt_hessian=self.internal_preopt_hessian,
            internal_preopt_solver=self.internal_preopt_solver,
            internal_preopt_space=self.internal_preopt_space,
            internal_preopt_guard_cycles=self.internal_preopt_guard_cycles,
            internal_optimization=self.internal_optimization,
            internal_max_cycle=self.internal_max_cycle,
            internal_conv_tol_grad=self.internal_conv_tol_grad,
            internal_conv_tol_step=self.internal_conv_tol_step,
            internal_conv_tol_energy=self.internal_conv_tol_energy,
            auto_active_restarts=False,
            exact_state_specific_gradient=self.exact_state_specific_gradient,
            verbose=self.verbose,
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
        if self.history:
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
        self.internal_preopt_history = list(trial.internal_preopt_history)
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
        self.use_cholesky_integrals = self._resolve_use_cholesky(use_cholesky)
        if self.use_cholesky_integrals:
            if self.coupling == "partial":
                raise NotImplementedError(
                    "Factorized SecondOrderCASSCF currently supports the "
                    "'qn', 'uncoupled', 'relaxed_fd', and 'full' coupling paths. "
                    "The partial coupled path still requires dense "
                    "CI-orbital derivative tensors."
                )
            if self.internal_optimization or self.internal_preopt_steps > 0:
                raise NotImplementedError(
                    "Factorized SecondOrderCASSCF does not yet support internal "
                    "preoptimization; run without internal_preopt_steps or use "
                    "the dense integral path."
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
        self.internal_preopt_history = []
        self.internal_optimization_converged = False
        self.active_orbitals = active_orbitals
        self._invalidate_ah_reference_cache()
        self._casci_binary_cache = None
        self._casci_direct_connectivity_cache = None
        self._casci_spin_string_connectivity_cache = None
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
            mo_coeff, ci_guess = self._internal_preopt(mo_coeff, ci_guess, macro)
            self._full_derivative_cache = None
            self._full_derivative_sigma_cache = None
            self._full_coupled_seed = None
            self._joint_trial_sigma_cache = {}
            self.mo_coeff_ref = mo_coeff
            h1_ref = self.mf.get_hcore_mo(mo_coeff)
            if self.use_cholesky_integrals:
                pair_ref = transform_eri_factors_to_mo_pair(
                    _get_mf_cholesky_factors(self.mf),
                    mo_coeff,
                )
                eri_ref = None
            else:
                pair_ref = None
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
                if self.use_cholesky_integrals:
                    h1_cur, pair_cur = self._transform_frozen_factor_integrals(
                        h1_ref,
                        pair_ref,
                        U,
                    )
                    eri_cur = None
                else:
                    h1_cur, eri_cur = self._transform_frozen_integrals(h1_ref, eri_ref, U)
                    pair_cur = None
                solve_nstates = self.nstates
                if self.coupling == "partial":
                    solve_nstates += max(0, self.coupled_ci_roots)
                if self.use_cholesky_integrals:
                    mc = self._make_factor_integral_casci(
                        h1_cur,
                        pair_cur,
                        mo_coeff,
                        solve_nstates,
                        ci0=local_ci_guess,
                    )
                else:
                    mc = self._make_integral_casci(
                        h1_cur,
                        eri_cur,
                        mo_coeff,
                        solve_nstates,
                        ci0=local_ci_guess,
                    )
                energy = self._objective_energy(mc, self.state_id)
                if self.use_cholesky_integrals:
                    dm1_occ, dm2_occ = self._effective_rdms_occ(mc, self.state_id)
                    nocc_like = mc.ncore + mc.ncas
                    fock = generalized_fock_from_factors(
                        h1_cur,
                        pair_cur[:, :, :nocc_like],
                        dm1_occ,
                        dm2_occ,
                    )
                    grad = orbital_gradient(fock)
                    gnorm = self._gradient_norm(grad, mc.ncore, mc.ncas, self.nmo)
                    grad_vec = self._pack_orbitals(grad, mc.ncore, mc.ncas, self.nmo)
                    dm1 = dm1_occ
                    dm2 = dm2_occ
                else:
                    dm1, dm2 = self._effective_rdms(mc, self.state_id)
                    fock = generalized_fock(h1_cur, eri_cur, dm1, dm2)
                    grad = orbital_gradient(fock)
                    gnorm = self._gradient_norm(grad, mc.ncore, mc.ncas, self.nmo)
                    grad_vec = self._pack_orbitals(grad, mc.ncore, mc.ncas, self.nmo)
                if (
                    self.nstates == 1
                    and self.exact_state_specific_gradient
                    and not self.use_cholesky_integrals
                ):
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
                    gnorm = float(np.linalg.norm(grad_vec)) if grad_vec.size else 0.0

                use_parameterized_hessian = self.ah_hessian == "finite_difference"
                if use_parameterized_hessian:
                    orbital_hessian_model = "parameterized_finite_difference"
                elif self.use_cholesky_integrals:
                    orbital_hessian_model = "factorized_analytic_integral_response"
                elif self.orbital_parameterization == "wmk":
                    orbital_hessian_model = "analytic_wmk_second_order"
                else:
                    orbital_hessian_model = "analytic_integral_response"

                def base_hessian_action(vec):
                    if self.use_cholesky_integrals and not use_parameterized_hessian:
                        return self._pack_orbitals(
                            self._orbital_hessian_action_from_factors(
                                h1_cur,
                                pair_cur,
                                dm1,
                                dm2,
                                self._unpack_orbitals(vec, mc.ncore, mc.ncas, self.nmo),
                            ),
                            mc.ncore,
                            mc.ncas,
                            self.nmo,
                        )
                    if self.use_cholesky_integrals:
                        return self._factor_parameterized_orbital_hessian_action(
                            h1_cur,
                            pair_cur,
                            dm1,
                            dm2,
                            mc,
                            vec,
                        )
                    if use_parameterized_hessian:
                        return self._parameterized_orbital_hessian_action(
                            h1_cur,
                            eri_cur,
                            dm1,
                            dm2,
                            mc,
                            grad_vec,
                            mc.ci[self.state_id],
                            vec,
                        )
                    return self._analytic_orbital_hessian_action(
                        h1_cur,
                        eri_cur,
                        dm1,
                        dm2,
                        mc,
                        vec,
                    )

                if self.coupling == "qn" and qn_base_hessian_action is None:
                    h1_qn = np.array(h1_cur, copy=True)
                    if self.use_cholesky_integrals:
                        pair_qn = np.array(pair_cur, copy=True)
                        eri_qn = None
                    else:
                        pair_qn = None
                        eri_qn = np.array(eri_cur, copy=True)
                    dm1_qn = np.array(dm1, copy=True)
                    dm2_qn = np.array(dm2, copy=True)
                    ncore_qn = mc.ncore
                    ncas_qn = mc.ncas
                    nmo_qn = self.nmo

                    ci_qn = np.array(mc.ci[self.state_id], copy=True)
                    grad_qn = np.array(grad_vec, copy=True)

                    def qn_base_hessian_action(vec):
                        if self.use_cholesky_integrals and not use_parameterized_hessian:
                            return self._pack_orbitals(
                                self._orbital_hessian_action_from_factors(
                                    h1_qn,
                                    pair_qn,
                                    dm1_qn,
                                    dm2_qn,
                                    self._unpack_orbitals(vec, mc.ncore, mc.ncas, self.nmo),
                                ),
                                mc.ncore,
                                mc.ncas,
                                self.nmo,
                            )
                        if self.use_cholesky_integrals:
                            return self._factor_parameterized_orbital_hessian_action(
                                h1_qn,
                                pair_qn,
                                dm1_qn,
                                dm2_qn,
                                mc,
                                vec,
                            )
                        if use_parameterized_hessian:
                            return self._parameterized_orbital_hessian_action(
                                h1_qn,
                                eri_qn,
                                dm1_qn,
                                dm2_qn,
                                mc,
                                grad_qn,
                                ci_qn,
                                vec,
                            )
                        return self._analytic_orbital_hessian_action(
                            h1_qn,
                            eri_qn,
                            dm1_qn,
                            dm2_qn,
                            mc,
                            vec,
                        )

                if self.coupling == "qn":
                    hessian_action = lambda vec: self._qn_hessian_action(
                        vec,
                        qn_base_hessian_action,
                    )
                elif self.coupling == "relaxed_fd":
                    if self.use_cholesky_integrals:
                        hessian_action = lambda vec: self._factor_relaxed_ci_hessian_action(
                            h1_ref,
                            pair_ref,
                            U,
                            mo_coeff,
                            grad_vec,
                            mc,
                            mc.ci,
                            vec,
                        )
                    else:
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

                micro_record = {
                    "macro": macro,
                    "micro": micro,
                    "energy": energy,
                    "gradient_norm": gnorm,
                    "orbital_parameterization": self.orbital_parameterization,
                    "orbital_hessian_model": orbital_hessian_model,
                }
                self.micro_history.append(micro_record)

                micro_mc = mc
                micro_energy = energy
                micro_gnorm = gnorm
                local_ci_guess = self._copy_ci_guess(mc.ci)

                if gnorm < self.conv_tol_grad:
                    micro_step = 0.0
                    break

                hess_diag = self._diagonal_hessian(
                    fock,
                    mc.ncore,
                    mc.ncas,
                    level_shift=self.level_shift,
                )
                step_limit = min(self._ah_trust_radius, self.max_step)
                use_norm_trust = self.ah_trust_metric == "norm"
                step_trust_radius = (
                    step_limit * math.sqrt(max(1, grad_vec.size))
                    if use_norm_trust
                    else step_limit
                )
                if use_norm_trust:
                    diag_step, ah_diag_shift = shifted_hessian_trust_step(
                        grad_vec,
                        hess_diag,
                        trust_radius=step_trust_radius,
                        regularization=self.level_shift,
                    )
                else:
                    diag_step = self._diagonal_preconditioned_vector(
                        grad,
                        fock,
                        mc.ncore,
                        mc.ncas,
                        level_shift=self.level_shift,
                    )
                    ah_diag_shift = 0.0
                micro_record["ah_diagonal_shift"] = float(ah_diag_shift)
                micro_record["ah_trust_radius"] = float(step_trust_radius)
                micro_record["ah_trust_metric"] = self.ah_trust_metric
                micro_record["ah_adaptive_trust"] = bool(self.ah_adaptive_trust)
                ah_step_bound = None if use_norm_trust else step_limit
                step_vec = augmented_hessian_direction(
                    grad_vec,
                    hess_diag,
                    max_step=ah_step_bound,
                    regularization=self.level_shift,
                    fallback_step=diag_step,
                )
                orbital_fallback_step_vec = np.asarray(step_vec, dtype=float).copy()
                coupled_ci_guess = None
                coupled_info = None
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
                    step_vec, coupled_ci_guess, coupled_info = self._full_coupled_step(
                        mc,
                        h1_cur,
                        eri_cur,
                        grad_vec,
                        hess_diag,
                        hessian_action,
                        step_vec,
                        step_limit,
                        return_info=True,
                        pair_factors=pair_cur if self.use_cholesky_integrals else None,
                    )
                else:
                    ah_guess = self._orbital_pspace_guess(
                        grad_vec,
                        hess_diag,
                        primary_step=step_vec,
                    )
                    if ah_guess is None:
                        ah_guess = step_vec
                        ah_max_subspace = self.ah_max_subspace
                    else:
                        ah_max_subspace = max(
                            self.ah_max_subspace,
                            ah_guess.shape[1] + 2,
                        )
                    ah_max_cycle = self.ah_max_cycle
                    if self.ah_pspace_size > 0 and ah_guess is not None:
                        ah_max_cycle = max(1, self.ah_pspace_max_cycle)
                    ah_pspace_dim = 0
                    if ah_guess is not None:
                        ah_pspace_dim = (
                            1 if np.asarray(ah_guess).ndim == 1 else ah_guess.shape[1]
                        )
                    step_vec, ah_info = davidson_augmented_hessian_direction(
                        grad_vec,
                        hess_diag,
                        matvec=hessian_action,
                        max_step=ah_step_bound,
                        regularization=self.level_shift,
                        max_cycle=ah_max_cycle,
                        max_subspace=ah_max_subspace,
                        tol=max(self.conv_tol_grad, 1.0e-4),
                        guess=ah_guess,
                        fallback_step=diag_step,
                        return_info=True,
                    )
                    micro_record.update(
                        {
                            "ah_converged": bool(ah_info["converged"]),
                            "ah_iterations": int(ah_info["iterations"]),
                            "ah_residual_norm": float(ah_info["residual_norm"]),
                            "ah_eigenvalue": float(ah_info["eigenvalue"]),
                            "ah_model": float(ah_info["model"]),
                            "ah_subspace_dim": int(ah_info["subspace_dim"]),
                            "ah_pspace_dim": int(ah_pspace_dim),
                            "ah_used_fallback": bool(ah_info["used_fallback"]),
                        }
                    )
                def limited_candidate(raw_step):
                    candidate = np.asarray(raw_step, dtype=float)
                    if np.dot(candidate, grad_vec) >= 0.0:
                        candidate = np.asarray(diag_step, dtype=float)
                    if use_norm_trust:
                        return limit_step_trust_radius(candidate, step_trust_radius), None
                    return limit_step_norm(candidate, step_limit), step_limit

                def evaluate_micro_candidate(label, raw_step, ci_guess):
                    candidate_step, candidate_limit = limited_candidate(raw_step)
                    kappa = self._unpack_orbitals(
                        candidate_step,
                        mc.ncore,
                        mc.ncas,
                        self.nmo,
                        max_step=candidate_limit,
                    )
                    if (
                        label == "coupled"
                        and self.coupling == "full"
                        and ci_guess is not None
                        and coupled_info is not None
                    ):
                        predicted = float(coupled_info.get("model_reduction", np.nan))
                        if not np.isfinite(predicted) or predicted <= 0.0:
                            step_hv = np.asarray(hessian_action(candidate_step), dtype=float)
                            predicted = -float(
                                np.dot(grad_vec, candidate_step)
                                + 0.5 * np.dot(candidate_step, step_hv)
                            )
                        if self.use_cholesky_integrals:
                            trial_accepted, joint = self._factor_joint_trust_region_micro_search(
                                h1_ref,
                                pair_ref,
                                U,
                                kappa,
                                energy,
                                mc,
                                ci_guess,
                                predicted,
                                model_linear=coupled_info.get("model_linear"),
                                model_quadratic=coupled_info.get("model_quadratic"),
                            )
                        else:
                            trial_accepted, joint = self._joint_trust_region_micro_search(
                                h1_ref,
                                eri_ref,
                                U,
                                kappa,
                                energy,
                                mc,
                                ci_guess,
                                predicted,
                                model_linear=coupled_info.get("model_linear"),
                                model_quadratic=coupled_info.get("model_quadratic"),
                            )
                        (
                            trial_U,
                            trial_energy,
                            trial_mc,
                            trial_scale,
                            actual,
                            scaled_predicted,
                            ratio,
                        ) = joint
                        predicted = (
                            float(scaled_predicted) / (float(trial_scale) ** 2)
                            if trial_scale != 0.0
                            else float(predicted)
                        )
                    else:
                        step_hv = np.asarray(hessian_action(candidate_step), dtype=float)
                        predicted = -float(
                            np.dot(grad_vec, candidate_step)
                            + 0.5 * np.dot(candidate_step, step_hv)
                        )
                        if self.use_cholesky_integrals:
                            trial_accepted, trial_U, _, trial_mc, trial_scale = (
                                self._factor_micro_line_search(
                                    h1_ref,
                                    pair_ref,
                                    U,
                                    kappa,
                                    energy,
                                    ci_guess,
                                )
                            )
                        else:
                            trial_accepted, trial_U, _, trial_mc, trial_scale = (
                                self._micro_line_search(
                                    h1_ref,
                                    eri_ref,
                                    U,
                                    kappa,
                                    energy,
                                    ci_guess,
                                )
                            )
                        if trial_mc is None:
                            actual = 0.0
                            trial_energy = energy
                        else:
                            trial_energy = self._objective_energy(trial_mc, self.state_id)
                            actual = float(energy - trial_energy)
                        scaled_predicted = predicted * float(trial_scale) ** 2
                        if scaled_predicted <= 1.0e-12:
                            ratio = np.inf if actual > 0.0 else -np.inf
                        else:
                            ratio = actual / scaled_predicted
                    return {
                        "label": label,
                        "accepted": bool(trial_accepted),
                        "U": trial_U,
                        "trial_energy": float(trial_energy),
                        "trial_mc": trial_mc,
                        "accepted_scale": float(trial_scale),
                        "step_vec": candidate_step,
                        "predicted_reduction": float(predicted),
                        "actual_reduction": float(actual),
                        "ratio": float(ratio),
                    }

                coupled_mode = self.coupling in {"partial", "full"}
                primary_label = "coupled" if coupled_mode else "orbital"
                primary_ci = (
                    coupled_ci_guess
                    if coupled_ci_guess is not None
                    else mc.ci[: self.nstates]
                )
                primary_result = evaluate_micro_candidate(
                    primary_label,
                    step_vec,
                    primary_ci,
                )
                chosen_result = primary_result
                fallback_result = None

                if coupled_mode:
                    ratio = primary_result["ratio"]
                    ratio_ok = (
                        np.isfinite(ratio)
                        and ratio >= self.coupled_accept_min_ratio
                    )
                    model_free_ok = (
                        not np.isfinite(ratio)
                        and primary_result["actual_reduction"] > 0.0
                    )
                    coupled_trusted = bool(
                        primary_result["accepted"] and (ratio_ok or model_free_ok)
                    )
                    if self.coupled_fallback and not coupled_trusted:
                        fallback_result = evaluate_micro_candidate(
                            "orbital_fallback",
                            orbital_fallback_step_vec,
                            mc.ci[: self.nstates],
                        )
                        if fallback_result["accepted"]:
                            primary_good = bool(primary_result["accepted"])
                            fallback_better = (
                                fallback_result["trial_energy"]
                                <= primary_result["trial_energy"]
                                + max(self.conv_tol, 1.0e-10)
                            )
                            if (not primary_good) or fallback_better:
                                chosen_result = fallback_result
                    micro_record.update(
                        {
                            "coupled_step_attempted": True,
                            "coupled_acceptance": str(chosen_result["label"]),
                            "coupled_trial_accepted": bool(primary_result["accepted"]),
                            "coupled_trial_ratio": float(primary_result["ratio"]),
                            "coupled_trial_actual_reduction": float(
                                primary_result["actual_reduction"]
                            ),
                            "coupled_trial_predicted_reduction": float(
                                primary_result["predicted_reduction"]
                            ),
                            "coupled_joint_trust_region": bool(
                                self.coupling == "full"
                            ),
                            "coupled_fallback_used": bool(
                                chosen_result["label"] == "orbital_fallback"
                            ),
                            "coupled_fallback_enabled": bool(self.coupled_fallback),
                        }
                    )
                    if fallback_result is not None:
                        micro_record.update(
                            {
                                "coupled_fallback_accepted": bool(
                                    fallback_result["accepted"]
                                ),
                                "coupled_fallback_ratio": float(
                                    fallback_result["ratio"]
                                ),
                                "coupled_fallback_actual_reduction": float(
                                    fallback_result["actual_reduction"]
                                ),
                            }
                        )

                step_vec = chosen_result["step_vec"]
                accepted = bool(chosen_result["accepted"])
                U = chosen_result["U"]
                trial_mc = chosen_result["trial_mc"]
                accepted_scale = float(chosen_result["accepted_scale"])
                predicted_reduction = float(chosen_result["predicted_reduction"])
                actual_reduction = float(chosen_result["actual_reduction"])
                ah_ratio = float(chosen_result["ratio"])
                micro_step = (
                    float(accepted_scale * np.max(np.abs(step_vec)))
                    if step_vec.size > 0
                    else 0.0
                )
                micro_record["ah_predicted_reduction"] = predicted_reduction
                if not accepted:
                    micro_record["ah_actual_reduction"] = 0.0
                    micro_record["ah_ratio"] = -np.inf
                    if self.ah_adaptive_trust:
                        self._update_ah_trust_radius(step_limit, -np.inf, 0.0, step_vec)
                    break
                micro_record["ah_actual_reduction"] = actual_reduction
                micro_record["ah_ratio"] = ah_ratio
                if self.ah_adaptive_trust:
                    self._update_ah_trust_radius(
                        step_limit,
                        ah_ratio,
                        accepted_scale,
                        step_vec,
                    )
                micro_mc = trial_mc
                local_ci_guess = self._copy_ci_guess(trial_mc.ci)
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
            self._log_casscf_cycle(
                macro,
                micro_energy,
                micro_gnorm,
                0.0 if micro_step is None else float(micro_step),
                micro_cycles=micro,
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
            if active_orbitals is None and finite_energies:
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


# Public CASSCF defaults to the more robust second-order driver.  The original
# macroiteration-only implementation remains available explicitly as
# FirstOrderCASSCF for tests, debugging, and compatibility-sensitive workflows.
CASSCF = SecondOrderCASSCF
