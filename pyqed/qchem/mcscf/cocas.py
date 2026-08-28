#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 22:07:30 2025

@author: bingg
"""
import numpy as np
import time
from scipy.linalg import eigh
# from pyqed.qchem.mcscf.casci import CASCI
from opt_einsum import contract
from pyqed.qchem.mcscf.direct_ci import CASCI
from pyqed.qchem.mcscf.casci import (
    _get_mf_cholesky_factors,
    _resolve_use_cholesky_integrals,
    transform_eri_factors_to_mo_pair,
)
# from pyqed.qchem.mcscf.casci import CASCI


from pyqed.optimize import OrbitalContractionPlan, minimize
from pyqed.optimize import grad as opt_grad
from pyqed.optimize import gradient as opt_gradient
from pyqed.optimize import norm as opt_norm


def _orthonormalize_columns(U, eps=1.0e-12):
    """Project a rectangular orbital transform back onto the Stiefel manifold.

    The DIIS extrapolation step builds a linear combination of previous
    ``U``-matrices, which generally drifts slightly away from the orthonormal
    column constraint required by the orbital optimizer.  We restore
    orthonormal columns with a symmetric Lowdin-like normalization using the
    Gram matrix of the extrapolated vectors.
    """

    gram = U.conj().T @ U
    eigvals, eigvecs = eigh(gram)
    eigvals = np.clip(eigvals.real, eps, None)
    inv_sqrt = eigvecs @ np.diag(eigvals**-0.5) @ eigvecs.conj().T
    return U @ inv_sqrt


class OrbitalDIIS:
    """Pulay extrapolation for the CASSCF orbital subspace transform ``U``.

    The main branch carried a simple DIIS accelerator based on recent orbital
    updates.  On ``bg`` we keep the newer optimizer backends and wrap the DIIS
    logic in a small helper so both state-specific and state-averaged CASSCF
    can reuse it without duplicating the bookkeeping.
    r"""

    def __init__(self, max_space=6, start=2, regularization=1.0e-10):
        self.max_space = max_space
        self.start = start
        self.regularization = regularization
        self.vectors = []
        self.errors = []

    def reset(self):
        self.vectors.clear()
        self.errors.clear()

    def update(self, base, candidate, *, ncore, ncas, active_active):
        """Extrapolate the outer CO fixed-point residual ``candidate - base``."""

        candidate = _align_redundant_gauge(
            base,
            candidate,
            ncore,
            ncas,
            active_active=active_active,
        )
        self.vectors.append(candidate.copy())
        self.errors.append((candidate - base).copy())

        if len(self.errors) > self.max_space:
            self.errors.pop(0)
            self.vectors.pop(0)

        if len(self.errors) < self.start:
            return candidate

        bsize = len(self.errors)
        bmat = -1.0 * np.ones((bsize + 1, bsize + 1), dtype=float)
        rhs = np.zeros(bsize + 1, dtype=float)
        bmat[bsize, bsize] = 0.0
        rhs[bsize] = -1.0

        for i in range(bsize):
            for j in range(bsize):
                bmat[i, j] = np.vdot(self.errors[i], self.errors[j]).real
        bmat[:bsize, :bsize] += np.eye(bsize) * self.regularization

        try:
            coeff = np.linalg.solve(bmat, rhs)
        except np.linalg.LinAlgError:
            return candidate
        if np.max(np.abs(coeff[:-1])) > 5.0:
            return candidate

        U_new = np.zeros_like(candidate, dtype=candidate.dtype)
        for weight, vector in zip(coeff[:-1], self.vectors):
            U_new += weight * vector

        U_new = _orthonormalize_columns(U_new)
        return _align_redundant_gauge(
            base,
            U_new,
            ncore,
            ncas,
            active_active=active_active,
        )


def _apply_orbital_diis(
    diis_helper, base, candidate, *, ncore, ncas, active_active
):
    """Apply residual-based DIIS to the outer CO fixed-point map."""

    if diis_helper is None:
        return candidate
    return diis_helper.update(
        base,
        candidate,
        ncore=ncore,
        ncas=ncas,
        active_active=active_active,
    )


def _align_redundant_gauge(base, candidate, ncore, ncas, *, active_active):
    """Align only orbital blocks that are redundant for the active solver."""

    base = np.asarray(base)
    aligned = np.asarray(candidate).copy()
    blocks = [(0, int(ncore))]
    if not active_active:
        blocks.append((int(ncore), int(ncore) + int(ncas)))
    for start, stop in blocks:
        if stop - start <= 0:
            continue
        block = slice(start, stop)
        overlap = aligned[:, block].conj().T @ base[:, block]
        left, _, right_h = np.linalg.svd(overlap, full_matrices=False)
        aligned[:, block] = aligned[:, block] @ (left @ right_h)
    return aligned


def _fresh_casci_like(source, *, solver_cls=None):
    """Build a fresh CASCI object while preserving solver configuration."""

    if hasattr(source, "D"):
        cls = source.__class__ if solver_cls is None else solver_cls
        mc = cls(
            source.mf,
            ncas=source.ncas,
            nelecas=source.nelecas,
            D=source.D,
            init_guess=getattr(source, "init_guess", "hf"),
            m_warmup=getattr(source, "m_warmup", None),
            tol=getattr(source, "dmrg_conv_tol", getattr(source, "tol", 1.0e-6)),
            low_rank_mpo=getattr(source, "low_rank_mpo", False),
            low_rank_mpo_bond=getattr(source, "low_rank_mpo_bond", None),
            low_rank_mpo_batch_size=getattr(source, "low_rank_mpo_batch_size", 4),
            site=getattr(source, "site", getattr(source, "site_basis", "spin_orbital")),
            spatial_reduced_mpo=getattr(source, "spatial_reduced_mpo", None),
            symmetry=getattr(source, "symmetry", None),
            spatial_site_basis=getattr(source, "spatial_site_basis", "canonical"),
            integral_backend=getattr(source, "integral_backend_override", None),
            spatial_abelian_mpo=getattr(source, "spatial_abelian_mpo", "auto"),
            spatial_abelian_symbolic_algo=getattr(
                source,
                "spatial_abelian_symbolic_algo",
                "Hopcroft-Karp",
            ),
            spatial_family_environment_backend=getattr(
                source,
                "spatial_family_environment_backend",
                "block2_table",
            ),
            spatial_native_p_grouping=getattr(
                source,
                "spatial_native_p_grouping",
                "first_site_order",
            ),
            spatial_block2_table_p_split_metric=getattr(
                source,
                "spatial_block2_table_p_split_metric",
                "auto",
            ),
            spatial_block2_table_p_split_groups=getattr(
                source,
                "spatial_block2_table_p_split_groups",
                "auto",
            ),
            spatial_block2_table_native_p=getattr(
                source,
                "spatial_block2_table_native_p",
                False,
            ),
            spatial_complementary_payload_tensor_matvec=getattr(
                source,
                "spatial_complementary_payload_tensor_matvec",
                True,
            ),
            spatial_precontracted_family_environment=getattr(
                source,
                "spatial_precontracted_family_environment",
                True,
            ),
            spatial_boundary_table_max_dim=getattr(
                source,
                "spatial_boundary_table_max_dim",
                32,
            ),
            spatial_exact_component_compression_policy=getattr(
                source,
                "spatial_exact_component_compression_policy",
                "auto",
            ),
            spatial_exact_component_compression_validate=getattr(
                source,
                "spatial_exact_component_compression_validate",
                True,
            ),
            spatial_exact_component_compression_validation_vectors=getattr(
                source,
                "spatial_exact_component_compression_validation_vectors",
                1,
            ),
            spatial_exact_component_compression_min_reduction=getattr(
                source,
                "spatial_exact_component_compression_min_reduction",
                1,
            ),
            spatial_exact_component_compression_max_group_size=getattr(
                source,
                "spatial_exact_component_compression_max_group_size",
                64,
            ),
            spatial_enable_cpp_boundary_r=getattr(
                source,
                "spatial_enable_cpp_boundary_r",
                False,
            ),
            spatial_validate_cpp_boundary_r=getattr(
                source,
                "spatial_validate_cpp_boundary_r",
                True,
            ),
            spatial_enable_cpp_boundary_p=getattr(
                source,
                "spatial_enable_cpp_boundary_p",
                True,
            ),
            spatial_validate_cpp_boundary_p=getattr(
                source,
                "spatial_validate_cpp_boundary_p",
                True,
            ),
            spatial_cpp_boundary_p_validation_policy=getattr(
                source,
                "spatial_cpp_boundary_p_validation_policy",
                "first_pass",
            ),
            spatial_direct_operator_batch_min_entries=getattr(
                source,
                "spatial_direct_operator_batch_min_entries",
                2,
            ),
            dmrg_performance=getattr(source, "dmrg_performance", "auto"),
            abelian_matvec_options=getattr(source, "abelian_matvec_options", None),
            debug_complementary_action_check=getattr(
                source,
                "debug_complementary_action_check",
                False,
            ),
            debug_complementary_action_check_tol=getattr(
                source,
                "debug_complementary_action_check_tol",
                1.0e-10,
            ),
            debug_complementary_action_check_limit=getattr(
                source,
                "debug_complementary_action_check_limit",
                32,
            ),
            debug_spatial_family_hamiltonian_check=getattr(
                source,
                "debug_spatial_family_hamiltonian_check",
                False,
            ),
            orb_sym=getattr(source, "orb_sym", None),
            verbose=getattr(source, "verbose", 0),
        )
    else:
        mc = CASCI(
            source.mf,
            ncas=source.ncas,
            nelecas=source.nelecas,
            tol=source.tol,
            verbose=getattr(source, "verbose", 0),
        )
    mc.spin_purification = source.spin_purification
    mc.ss = source.ss
    mc.shift = source.shift
    for name in (
        "direct_ci_dense_fallback_ndets",
        "direct_ci_eigensolver",
        "direct_ci_max_cycle",
        "direct_ci_max_subspace",
        "direct_ci_reuse_guess",
    ):
        value = getattr(source, name, None)
        if value is not None:
            setattr(mc, name, value)
    mc.use_cholesky_integrals = getattr(source, 'use_cholesky_integrals', False)
    mc._su2_runtime = getattr(
        source,
        "_su2_runtime",
        getattr(source, "_active_hamiltonian", None),
    )
    mc.binary = getattr(source, 'binary', None)
    mc.direct_connectivity = getattr(source, 'direct_connectivity', None)
    mc.SC1 = getattr(source, 'SC1', None)
    mc.SC2 = getattr(source, 'SC2', None)
    return mc


def _is_su2_dmrg(mc):
    """Return whether ``mc`` owns the reduced SU(2) DMRG runtime."""

    symmetry = getattr(mc, "symmetry", ()) or ()
    if isinstance(symmetry, str):
        symmetry = (symmetry,)
    return bool(
        hasattr(mc, "D")
        and "su2" in symmetry
        and getattr(mc, "spatial_reduced_mpo", False)
    )


def _fresh_macro_casci(source, *, rebuild_runtime=False):
    """Create an orbital-trial solver, optionally rebuilding SU(2) routes."""

    mc = _fresh_casci_like(source)
    if rebuild_runtime and _is_su2_dmrg(mc):
        mc._su2_runtime = None
    return mc


def _run_macro_casci(
    source,
    *args,
    warm_start=True,
    method="direct_ci",
    **kwargs,
):
    """Run one CO active solve with a clean retry for stale SU(2) routes."""

    def is_stale_route_error(exc):
        message = str(exc).lower()
        return (
            "route" in message
            and ("incompatible" in message or "inconsistent" in message)
            and (
                "shape" in message
                or "dimension" in message
                or "topology" in message
            )
        )

    def solve(owner):
        trial = _fresh_macro_casci(owner)
        if warm_start:
            _wguess(owner, trial)
        try:
            _run_casci_like(trial, *args, method=method, **kwargs)
        except ValueError as exc:
            if not _is_su2_dmrg(trial) or not is_stale_route_error(exc):
                raise
            trial = _fresh_macro_casci(owner, rebuild_runtime=True)
            if warm_start:
                _wguess(owner, trial)
            _run_casci_like(trial, *args, method=method, **kwargs)
            trial._co_su2_runtime_rebuilt = True
        else:
            trial._co_su2_runtime_rebuilt = False
        return trial

    trial = solve(source)
    solver_retried = bool(
        _is_su2_dmrg(trial) and not _solver_converged(trial)
    )
    if solver_retried:
        rebuilt = bool(getattr(trial, "_co_su2_runtime_rebuilt", False))
        trial = solve(trial)
        trial._co_su2_runtime_rebuilt = bool(
            rebuilt or getattr(trial, "_co_su2_runtime_rebuilt", False)
        )
    trial._co_solver_retried = solver_retried
    return trial


def _run_casci_like(mc, *args, method="direct_ci", **kwargs):
    """
    Run either a CI-like CASCI object or a DMRG-backed CASCI object.

    DMRG.run does not accept the CI ``method`` keyword; CASCI.run does.
    """

    if hasattr(mc, "D"):
        kwargs.pop("use_cholesky", None)
        return mc.run(*args, **kwargs)
    return mc.run(*args, method=method, **kwargs)


def _wguess(src, dst, state=0):
    """Reuse the accepted DMRG MPS as the next macroiteration guess."""

    if not hasattr(src, "export_ground_state"):
        return
    try:
        dst.init_guess = src.export_ground_state(state=state)
    except Exception:
        return


def _physical_orbital_gradient(
    U, euclidean_gradient, ncore, ncas, *, active_active
):
    """Remove only gauge blocks redundant for the chosen active solver."""

    components = _orbital_gradient_components(
        U,
        euclidean_gradient,
        ncore,
        ncas,
    )
    gradient = components["nonredundant"]
    if active_active:
        gradient = gradient + components["active_active"]
    return gradient


def _orbital_gradient_components(U, euclidean_gradient, ncore, ncas):
    """Return nonredundant, active-active, and discarded core-core blocks."""

    tangent = opt_grad(U, euclidean_gradient)
    vertical = U.conj().T @ tangent
    horizontal = tangent - U @ vertical

    ncore = int(ncore)
    ncas = int(ncas)
    core_active = np.zeros_like(vertical)
    active_internal = np.zeros_like(vertical)
    core_internal = np.zeros_like(vertical)
    active = slice(ncore, ncore + ncas)
    core = slice(0, ncore)
    core_active[core, active] = vertical[core, active]
    core_active[active, core] = vertical[active, core]
    active_internal[active, active] = vertical[active, active]
    core_internal[core, core] = vertical[core, core]
    return {
        "nonredundant": horizontal + U @ core_active,
        "active_active": U @ active_internal,
        "core_core": U @ core_internal,
    }


def _gn_details(
    U,
    h1e,
    eri,
    dm1,
    dm2,
    contraction_plan=None,
    *,
    ncore=0,
    ncas,
    active_active=False,
):
    gradient_fn = (
        opt_gradient if contraction_plan is None else contraction_plan.gradient
    )
    euclidean = gradient_fn(U, h1e, eri, dm1, dm2)
    components = _orbital_gradient_components(U, euclidean, ncore, ncas)
    nonredundant = float(opt_norm(components["nonredundant"]))
    active_norm = float(opt_norm(components["active_active"]))
    total = float(
        opt_norm(
            components["nonredundant"]
            + (components["active_active"] if active_active else 0.0)
        )
    )
    return {
        "total": total,
        "nonredundant": nonredundant,
        "active_active": active_norm,
        "core_core_discarded": float(opt_norm(components["core_core"])),
    }


def _gn(
    U,
    h1e,
    eri,
    dm1,
    dm2,
    contraction_plan=None,
    *,
    ncore=0,
    ncas=None,
    active_active=False,
):
    gradient_fn = (
        opt_gradient if contraction_plan is None else contraction_plan.gradient
    )
    if ncas is None:
        g = gradient_fn(U, h1e, eri, dm1, dm2)
        return float(opt_norm(opt_grad(U, g)))
    return _gn_details(
        U,
        h1e,
        eri,
        dm1,
        dm2,
        contraction_plan,
        ncore=ncore,
        ncas=ncas,
        active_active=active_active,
    )["total"]


def _limit_stiefel_displacement(base, candidate, trust_radius):
    """Apply a trust radius to the complete CO macro displacement."""

    base = np.asarray(base)
    candidate = _orthonormalize_columns(np.asarray(candidate))
    displacement = float(opt_norm(candidate - base))
    if trust_radius is None or displacement <= float(trust_radius):
        return candidate, displacement

    radius = float(trust_radius)
    if radius <= 0.0:
        return base.copy(), 0.0

    direction = candidate - base
    lo = 0.0
    hi = 1.0
    limited = base.copy()
    limited_norm = 0.0
    # Polar projection can change the chord length slightly. A short bisection
    # makes the returned displacement a genuine hard macro trust radius.
    for _ in range(32):
        fraction = 0.5 * (lo + hi)
        trial = _orthonormalize_columns(base + fraction * direction)
        trial_norm = float(opt_norm(trial - base))
        if trial_norm <= radius:
            lo = fraction
            limited = trial
            limited_norm = trial_norm
        else:
            hi = fraction
    return limited, limited_norm


def _sdiag(mc):
    dmrg = getattr(mc, "dmrg", None)
    hist = getattr(dmrg, "sweep_history", None)
    out = {"solver": True if dmrg is None else bool(getattr(dmrg, "converged", False))}
    if not hist:
        return out
    out["nsw"] = len(hist)
    out["sweep_E"] = hist[-1].get("energy")
    if len(hist) > 1:
        e0 = hist[-2].get("energy")
        e1 = hist[-1].get("energy")
        try:
            out["sweep_dE"] = float(e1 - e0)
        except Exception:
            pass
    out["trunc"] = hist[-1].get("truncation")
    out["kept"] = hist[-1].get("states_kept")
    return out


def _solver_converged(mc):
    """Return whether the active-space solver attached to ``mc`` converged."""

    dmrg = getattr(mc, "dmrg", None)
    if dmrg is None:
        return True
    return bool(getattr(dmrg, "converged", False))


def _set_convergence_metadata(mc, *, macro_converged, macro_iterations):
    """Store separate macro and active-space solver convergence flags."""

    mc.macro_converged = bool(macro_converged)
    mc.macro_iterations = int(macro_iterations)
    mc.solver_converged = _solver_converged(mc)
    mc.converged = bool(mc.macro_converged and mc.solver_converged)


class COCAS(CASCI):
    """

    Using the OptOrbFCI algorithm to optimize orbitals
    (better than conventional CASSCF algorithm)



    """
    def __init__(self, mf, ncas, nelecas, max_cycles=30,
                 optimizer='RCG', optimizer_history=7,
                 diis=True, diis_space=6, diis_start=2,
                 ci_method='direct_ci', direct_ci_dense_fallback_ndets=0,
                 optimizer_tol=1.0e-4,
                 optimizer_max_steps=200,
                 optimizer_max_step_norm=None,
                 macro_tol=1.0e-6,
                 ci_tol=0.0,
                 orb_grad_tol=None,
                 reject_macro_energy=True,
                 macro_energy_rise_tol=1.0e-8,
                 macro_reject_max=8,
                 macro_trust_radius=0.25,
                 macro_trust_min=1.0e-4,
                 macro_trust_max=1.0,
                 macro_trust_shrink=0.5,
                 macro_trust_grow=1.5,
                 warm_start_dmrg=True,
                 use_cholesky=None,
                 verbose=0,
                 **kwargs):
        super().__init__(mf, ncas, nelecas, tol=ci_tol, verbose=verbose, **kwargs)

        self.max_cycles = max_cycles # macroiterations
        self.macro_tol = float(macro_tol)
        self.mo_coeff = None # opt orb
        # Orbital optimization backend for the U-matrix formulation.
        self.optimizer = optimizer.upper()
        self.optimizer_history = optimizer_history
        self.optimizer_tol = float(optimizer_tol)
        self.optimizer_max_steps = (
            None if optimizer_max_steps is None else int(optimizer_max_steps)
        )
        self.optimizer_max_step_norm = (
            None if optimizer_max_step_norm is None else float(optimizer_max_step_norm)
        )
        self.orb_grad_tol = (
            None if orb_grad_tol is None else float(orb_grad_tol)
        )
        self.reject_macro_energy = bool(reject_macro_energy)
        self.macro_energy_rise_tol = float(macro_energy_rise_tol)
        self.macro_reject_max = int(macro_reject_max)
        self.macro_trust_radius = (
            None if macro_trust_radius is None else float(macro_trust_radius)
        )
        self.macro_trust_min = float(macro_trust_min)
        self.macro_trust_max = float(macro_trust_max)
        self.macro_trust_shrink = float(macro_trust_shrink)
        self.macro_trust_grow = float(macro_trust_grow)
        self.warm_start_dmrg = bool(warm_start_dmrg)
        # Optional DIIS mixing over the optimized U matrices.  This mirrors the
        # main-branch accelerator while keeping it configurable on bg.
        self.diis = diis
        self.diis_space = diis_space
        self.diis_start = diis_start
        self.ci_method = ci_method
        self.use_cholesky = use_cholesky
        self.use_cholesky_integrals = False
        # Keep a consistent CI backend across macroiterations; switching
        # between dense and direct solvers changes the effective orbital
        # objective enough to confuse convergence comparisons.
        self.direct_ci_dense_fallback_ndets = direct_ci_dense_fallback_ndets


        self.weights = None
        self.nstates = 1
        self.e_history = []
        self.converged = False
        self.macro_converged = False
        self.solver_converged = False
        self.macro_iterations = 0


    def run(self, nstates= None, weights = None, use_cholesky=None):
        mf = self.mf

        # canonical molecular orbs
        C0 = mf.mo_coeff

        # CASCI roots
        if nstates == None:
            nstates = self.nstates
        else:
            self.nstates = nstates
        if weights != None:
            self.weights = weights
            if nstates != len(self.weights):
                raise ValueError("the nstates you requires does not align with the nstates indicated by the weights. check input.")
        nmo = self.mf.nao
        ncas = self.ncas
        nelecas = self.nelecas
        ncore = self.ncore

        if use_cholesky is None:
            use_cholesky = self.use_cholesky
        if use_cholesky is None:
            use_cholesky = bool(getattr(mf, "cholesky_jk", False))
        self.use_cholesky_integrals = _resolve_use_cholesky_integrals(mf, use_cholesky)

        mc = _fresh_casci_like(self)
        # spin
        _run_casci_like(
            mc,
            nstates,
            method=self.ci_method,
            use_cholesky=self.use_cholesky_integrals,
        )


        # matrix elements in CMOs
        h1e = mf.get_hcore_mo()
        if self.use_cholesky_integrals:
            eri = transform_eri_factors_to_mo_pair(_get_mf_cholesky_factors(mf), C0)
        else:
            eri = mf.get_eri_mo()

        U0 = np.zeros((nmo, ncas+ncore))
        for i in range(ncas+ncore):
            U0[i, i] = 1.

        if nstates == 1: # ground state only
            C, mc = kernel(
                mc, U0, nelecas, ncas, C0, h1e, eri,
                max_cycles=self.max_cycles,
                optimizer=self.optimizer,
                optimizer_history=self.optimizer_history,
                optimizer_tol=self.optimizer_tol,
                optimizer_max_steps=self.optimizer_max_steps,
                optimizer_max_step_norm=self.optimizer_max_step_norm,
                tol=self.macro_tol,
                orb_grad_tol=self.orb_grad_tol,
                reject_macro_energy=self.reject_macro_energy,
                macro_energy_rise_tol=self.macro_energy_rise_tol,
                macro_reject_max=self.macro_reject_max,
                macro_trust_radius=self.macro_trust_radius,
                macro_trust_min=self.macro_trust_min,
                macro_trust_max=self.macro_trust_max,
                macro_trust_shrink=self.macro_trust_shrink,
                macro_trust_grow=self.macro_trust_grow,
                warm_start_dmrg=self.warm_start_dmrg,
                diis=self.diis,
                diis_space=self.diis_space,
                diis_start=self.diis_start,
                ci_method=self.ci_method,
                use_cholesky=self.use_cholesky_integrals,
            )

        elif nstates > 1:
            if self.weights is None:
                self.state_average(weights = np.ones(nstates)/nstates)
            if len(self.weights) != nstates: 
                self.state_average(weights = np.ones(nstates)/nstates)

            C, mc = kernel_state_average(
                mc, weights=self.weights, U0=U0, nelecas=nelecas, ncas=ncas,
                C0=C0, h1e=h1e, eri=eri,
                optimizer=self.optimizer,
                optimizer_history=self.optimizer_history,
                optimizer_tol=self.optimizer_tol,
                optimizer_max_steps=self.optimizer_max_steps,
                optimizer_max_step_norm=self.optimizer_max_step_norm,
                tol=self.macro_tol,
                orb_grad_tol=self.orb_grad_tol,
                reject_macro_energy=self.reject_macro_energy,
                macro_energy_rise_tol=self.macro_energy_rise_tol,
                macro_reject_max=self.macro_reject_max,
                macro_trust_radius=self.macro_trust_radius,
                macro_trust_min=self.macro_trust_min,
                macro_trust_max=self.macro_trust_max,
                macro_trust_shrink=self.macro_trust_shrink,
                macro_trust_grow=self.macro_trust_grow,
                warm_start_dmrg=self.warm_start_dmrg,
                diis=self.diis,
                diis_space=self.diis_space,
                diis_start=self.diis_start,
                ci_method=self.ci_method,
                use_cholesky=self.use_cholesky_integrals,
            )

        self.mo_coeff = C
        self.e_tot = mc.e_tot
        self.ci = mc.ci
        self.e_history = getattr(mc, 'e_history', [self.e_tot])
        self.macro_diagnostics = getattr(mc, "macro_diagnostics", [])
        self.e_core = mc.e_core
        self.hcore = mc.hcore
        self.h1e = getattr(mc, 'h1e', None)
        self.h2e = getattr(mc, 'h2e', None)
        self.h2e_cas = getattr(mc, 'h2e_cas', None)
        self.eri_so = getattr(mc, 'eri_so', None)
        self.binary = getattr(mc, 'binary', None)
        self.SC1 = getattr(mc, 'SC1', None)
        self.SC2 = getattr(mc, 'SC2', None)
        self.direct_connectivity = getattr(mc, 'direct_connectivity', None)
        self.mo_core = getattr(mc, 'mo_core', None)
        self.mo_cas = getattr(mc, 'mo_cas', None)
        self.nstates = mc.nstates
        self.solver_backend = getattr(mc, 'solver_backend', None)
        self.converged = bool(getattr(mc, "converged", False))
        self.macro_converged = bool(getattr(mc, "macro_converged", False))
        self.solver_converged = bool(getattr(mc, "solver_converged", False))
        self.macro_iterations = int(getattr(mc, "macro_iterations", 0))

        return self

    def state_average(self, weights):
        self.nstates = len(weights)
        self.weights = weights
        return self


def energy(U, h1e, eri, dm1, dm2):
    """
    electronic energy

    Parameters
    ----------
    U : ndarray of (n, p < n/2)
        transformation matrix
    h1e : TYPE
        core Hamiltonian in canonical MO
    eri : TYPE
        DESCRIPTION.
    dm1 : TYPE
        DESCRIPTION.
    dm2 : TYPE
        DESCRIPTION.

    Returns
    -------
    e : TYPE
        DESCRIPTION.

    """
    e = contract('pq, pa, qb, ab ->', h1e, U, U, dm1)
    if np.ndim(eri) == 3:
        transformed = contract('Ppq,pa,qb->Pab', eri, U, U)
        e += 0.5 * contract('Pab,Pcd,abcd->', transformed, transformed, dm2)
    else:
        e += 0.5 * (contract('pqrs, pa, qb, rc, sd, abcd ->', eri, U, U, U, U, dm2))
    return e



def kernel(mc, U0, nelecas, ncas, C0, h1e, eri, max_cycles=30, tol=1e-6,
           optimizer='RCG', optimizer_history=7, optimizer_tol=1.0e-4,
           optimizer_max_steps=200, optimizer_max_step_norm=None,
           diis=True,
           diis_space=6, diis_start=2, ci_method='direct_ci',
           reject_macro_energy=True, macro_energy_rise_tol=1.0e-8,
           macro_reject_max=8,
           orb_grad_tol=None, macro_trust_radius=0.25,
           macro_trust_min=1.0e-4, macro_trust_max=1.0,
           macro_trust_shrink=0.5, macro_trust_grow=1.5,
           warm_start_dmrg=True,
           raise_on_nonconvergence=True, **kwargs):
    r"""
    complete active space orbital optimization with orthonomality constraint

    .. math::
        U^\top U = I_N

        E = \sum_{p,q=1}^N t_{pq} U_{pp'} U_{q q'} \gamma_{p'q'} +
        1/2 v_{pqrs} \Gamma_{p'q'r's'} U_{pp'}U_{qq'}U_{rr'}U_{ss'}

    where U is a M x N (M > N) matrix.

    .. math::
        U_{k+1} = orth(U_k - \tau_k G_k)

    where G_k = \nabla P(U_k) is the gradient.

    Parameters
    ----------
    h1e : TYPE
        DESCRIPTION.
    h2e : TYPE
        DESCRIPTION.
    U0: ndarray
        initial guess of orbitals
    dm1 : TYPE
        DESCRIPTION.
    dm2 : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    if mc.ncore > 0:
        with_core = True
    else:
        with_core = False

    timing = {
        "solver_seconds": 0.0,
        "rdm_seconds": 0.0,
        "orbital_gradient_seconds": 0.0,
        "orbital_opt_seconds": 0.0,
    }
    timing_start = time.perf_counter()
    dm1, dm2 = mc.make_rdm12(0, with_core=with_core)
    timing["rdm_seconds"] += time.perf_counter() - timing_start
    contraction_plan = OrbitalContractionPlan(
        h1e, eri, U0.shape, dm1.shape, dm2.shape
    )

    # eri = mc.eri_so[0, 0] # for spin-restricted calculation
    # nmo = self.nmo

    # U0 = np.zeros((nmo, ncas))
    # for i in range(ncas):
    #     U0[i, i] = 1

    orbital_diis = None
    if diis:
        orbital_diis = OrbitalDIIS(max_space=diis_space, start=diis_start)

    cap0 = optimizer_max_step_norm
    gt = optimizer_tol if orb_grad_tol is None else float(orb_grad_tol)
    tr = None if macro_trust_radius is None else float(macro_trust_radius)
    tr_min = float(macro_trust_min)
    tr_max = float(macro_trust_max)
    tr_dn = float(macro_trust_shrink)
    tr_up = float(macro_trust_grow)
    active_active = hasattr(mc, "D")
    diag = []

    def opt_u(u, d1, d2, use_diis=True):
        timing_start = time.perf_counter()
        u1, e1 = minimize(
            contraction_plan.energy,
            u,
            args=(h1e, eri, d1, d2),
            tau=1.0,
            algorithm=optimizer, history_size=optimizer_history,
            epsilon=optimizer_tol,
            max_iterations=optimizer_max_steps,
            max_step_norm=cap0,
            gradient_fn=contraction_plan.gradient,
        )
        u1 = _align_redundant_gauge(
            u,
            u1,
            mc.ncore,
            ncas,
            active_active=active_active,
        )
        if use_diis:
            u1 = _apply_orbital_diis(
                orbital_diis,
                u,
                u1,
                ncore=mc.ncore,
                ncas=ncas,
                active_active=active_active,
            )
        timing["orbital_opt_seconds"] += time.perf_counter() - timing_start
        return u1

    U_acc = U0
    timing_start = time.perf_counter()
    gn_components = _gn_details(
        U_acc,
        h1e,
        eri,
        dm1,
        dm2,
        contraction_plan,
        ncore=mc.ncore,
        ncas=ncas,
        active_active=active_active,
    )
    gn = gn_components["total"]
    timing["orbital_gradient_seconds"] += time.perf_counter() - timing_start
    U_target = opt_u(U_acc, dm1, dm2)
    U, step_norm = _limit_stiefel_displacement(U_acc, U_target, tr)

    k = 0

    e_old = mc.e_tot
    e_history = [mc.e_tot]
    last_mo_coeff = C0 @ U_acc
    best_e = float(np.real(np.asarray(mc.e_tot).reshape(-1)[0]))
    best_mc = mc
    best_C = last_mo_coeff

    converged = False
    while k < max_cycles:

        ok = False
        rej = 0
        for ir in range(int(macro_reject_max) + 1):
            mo_coeff = C0 @ U

            timing_start = time.perf_counter()
            current_mc = _run_macro_casci(
                mc,
                mo_coeff=mo_coeff,
                warm_start=warm_start_dmrg,
                method=ci_method,
                **kwargs,
            )
            timing["solver_seconds"] += time.perf_counter() - timing_start

            if _is_su2_dmrg(current_mc) and not _solver_converged(current_mc):
                diag.append(
                    {
                        "macro": k + 1,
                        "energy": float(
                            np.real(np.asarray(current_mc.e_tot).reshape(-1)[0])
                        ),
                        "accepted": False,
                        "reason": "active_solver_unconverged",
                        "tr": tr,
                        "rej": rej,
                        "active_active_optimized": active_active,
                        "su2_runtime_rebuilt": bool(
                            getattr(
                                current_mc,
                                "_co_su2_runtime_rebuilt",
                                False,
                            )
                        ),
                        "solver_retried": bool(
                            getattr(current_mc, "_co_solver_retried", False)
                        ),
                        "solver": False,
                    }
                )
                break

            if (not reject_macro_energy) or current_mc.e_tot <= e_old + macro_energy_rise_tol:
                ok = True
                break

            rej += 1
            tr = None if tr is None else max(tr_min, tr * tr_dn)
            U, step_norm = _limit_stiefel_displacement(U_acc, U_target, tr)

        if not ok:
            break

        last_mo_coeff = mo_coeff
        e_history.append(current_mc.e_tot)
        de = float(np.real(np.asarray(current_mc.e_tot - e_old).reshape(-1)[0]))
        e_now = float(np.real(np.asarray(current_mc.e_tot).reshape(-1)[0]))
        timing_start = time.perf_counter()
        dm1_new, dm2_new = current_mc.make_rdm12(0, with_core=with_core)
        timing["rdm_seconds"] += time.perf_counter() - timing_start
        timing_start = time.perf_counter()
        gn_components_new = _gn_details(
            U,
            h1e,
            eri,
            dm1_new,
            dm2_new,
            contraction_plan,
            ncore=current_mc.ncore,
            ncas=ncas,
            active_active=active_active,
        )
        gn_new = gn_components_new["total"]
        timing["orbital_gradient_seconds"] += time.perf_counter() - timing_start
        row = {
            "macro": k + 1,
            "energy": e_now,
            "accepted": True,
            "dE": de,
            "gn": gn_new,
            "gn_nonredundant": gn_components_new["nonredundant"],
            "gn_active_active": gn_components_new["active_active"],
            "gn_core_core_discarded": gn_components_new[
                "core_core_discarded"
            ],
            "gn_start": gn,
            "gn_start_nonredundant": gn_components["nonredundant"],
            "gn_start_active_active": gn_components["active_active"],
            "step": step_norm,
            "tr": tr,
            "rej": rej,
            "active_active_optimized": active_active,
            "su2_runtime_rebuilt": bool(
                getattr(current_mc, "_co_su2_runtime_rebuilt", False)
            ),
            "solver_retried": bool(
                getattr(current_mc, "_co_solver_retried", False)
            ),
        }
        row.update(_sdiag(current_mc))
        gradient_spike = bool(gn > 0.0 and gn_new > 1.5 * gn)
        row["diis_reset"] = gradient_spike
        diag.append(row)
        if e_now < best_e:
            best_e = e_now
            best_mc = current_mc
            best_C = mo_coeff

        if gradient_spike and orbital_diis is not None:
            orbital_diis.reset()
        if tr is not None and rej == 0 and not gradient_spike:
            tr = min(tr_max, tr * tr_up)

        if abs(current_mc.e_tot - e_old) < tol and gn_new < gt:
            if getattr(mc, "verbose", 0) >= 1:
                print('\nCASSCF converged at macroiteration {}'.format(k))
                print("E(CASSCF) = {}".format(current_mc.e_tot))
            mc = current_mc
            converged = True
            break

        U_acc = U
        mc = current_mc
        e_old = mc.e_tot
        dm1, dm2 = dm1_new, dm2_new
        gn = gn_new
        gn_components = gn_components_new

        k += 1
        if k >= max_cycles:
            break

        U_target = opt_u(U_acc, dm1, dm2)
        U, step_norm = _limit_stiefel_displacement(U_acc, U_target, tr)
        # print(E + mol.energy_nuc())

    if not converged:
        if raise_on_nonconvergence:
            if diag and diag[-1].get("reason") == "active_solver_unconverged":
                raise RuntimeError(
                    "CO active-space DMRG did not converge after its warm "
                    "continuation; the orbital macro step was not accepted."
                )
            raise RuntimeError('Max macro steps reached. CASSCF not converged.')
        mc = best_mc
        mc.e_history = e_history
        mc.macro_diagnostics = diag
        mc.dmrgscf_timing = dict(timing)
        _set_convergence_metadata(
            mc,
            macro_converged=False,
            macro_iterations=k,
        )
        return best_C, mc

    # Rebuild the final CASCI result from scratch at the returned orbitals.
    # Reusing the same CASCI object across many macroiterations can leave the
    # final reported energy out of sync with the best orbitals, while a fresh
    # CASCI solve at ``mo_coeff`` is consistent.
    final_mc = _fresh_casci_like(mc)
    final_su2_runtime_rebuilt = getattr(final_mc, "_su2_runtime", None) is not None
    if final_su2_runtime_rebuilt:
        # The verification solve is a correctness boundary. Its MPS can have
        # different sector multiplicities from the accepted macro state, so a
        # clean owner avoids carrying bond-contextual execution plans across
        # that topology change. Macroiterations still use the normal reuse path.
        final_mc._su2_runtime = None
    final_mc.spin_purification = mc.spin_purification
    final_mc.ss = mc.ss
    final_mc.shift = mc.shift
    for name in (
        "direct_ci_dense_fallback_ndets",
        "direct_ci_eigensolver",
        "direct_ci_max_cycle",
        "direct_ci_max_subspace",
        "direct_ci_reuse_guess",
    ):
        value = getattr(mc, name, None)
        if value is not None:
            setattr(final_mc, name, value)
    if warm_start_dmrg:
        _wguess(mc, final_mc)
    _run_casci_like(final_mc, mo_coeff=mo_coeff, method=ci_method, **kwargs)
    if getattr(final_mc, "build_info", None) is not None:
        final_mc.build_info[
            "final_su2_runtime_rebuilt"
        ] = bool(final_su2_runtime_rebuilt)
    final_mc.e_history = e_history
    final_mc.macro_diagnostics = diag
    final_mc.dmrgscf_timing = dict(timing)
    _set_convergence_metadata(
        final_mc,
        macro_converged=True,
        macro_iterations=k + 1,
    )

    return mo_coeff, final_mc


def kernel_state_average(mc, weights, U0, nelecas, ncas, C0, h1e, eri,
                         max_cycles=50, tol=1e-6, optimizer='RCG',
                         optimizer_history=7, optimizer_tol=1.0e-4,
                         optimizer_max_steps=200,
                         optimizer_max_step_norm=None,
                         diis=True, diis_space=6,
                         diis_start=2, ci_method='direct_ci',
                         reject_macro_energy=True,
                         macro_energy_rise_tol=1.0e-8,
                         macro_reject_max=8,
                         orb_grad_tol=None, macro_trust_radius=0.25,
                         macro_trust_min=1.0e-4, macro_trust_max=1.0,
                         macro_trust_shrink=0.5, macro_trust_grow=1.5,
                         warm_start_dmrg=True,
                         raise_on_nonconvergence=True, **kwargs):

    if mc.ncore > 0:
        with_core = True
    else:
        with_core = False

    nstates = mc.nstates
    e_history = [mc.e_tot]
    
    orbital_diis = None
    if diis:
        orbital_diis = OrbitalDIIS(max_space=diis_space, start=diis_start)

    dm1 = 0
    dm2 = 0
    for n in range(nstates):
        _dm1, _dm2 = mc.make_rdm12(n, with_core=with_core)
        dm1 += _dm1 * weights[n]
        dm2 += _dm2 * weights[n]
    contraction_plan = OrbitalContractionPlan(
        h1e, eri, U0.shape, dm1.shape, dm2.shape
    )

    # State-averaged CASSCF uses the same ``U`` variable as the state-specific
    # kernel, so it should also keep improving the latest orbital transform
    # rather than restarting from ``U0`` in every macroiteration.
    cap0 = optimizer_max_step_norm
    gt = optimizer_tol if orb_grad_tol is None else float(orb_grad_tol)
    tr = None if macro_trust_radius is None else float(macro_trust_radius)
    tr_min = float(macro_trust_min)
    tr_max = float(macro_trust_max)
    tr_dn = float(macro_trust_shrink)
    tr_up = float(macro_trust_grow)
    active_active = hasattr(mc, "D")
    diag = []

    def opt_u(u, d1, d2, use_diis=True):
        u1, e1 = minimize(
            contraction_plan.energy,
            u,
            args=(h1e, eri, d1, d2),
            tau=1.0,
            algorithm=optimizer, history_size=optimizer_history,
            epsilon=optimizer_tol,
            max_iterations=optimizer_max_steps,
            max_step_norm=cap0,
            gradient_fn=contraction_plan.gradient,
        )
        u1 = _align_redundant_gauge(
            u,
            u1,
            mc.ncore,
            ncas,
            active_active=active_active,
        )
        if use_diis:
            u1 = _apply_orbital_diis(
                orbital_diis,
                u,
                u1,
                ncore=mc.ncore,
                ncas=ncas,
                active_active=active_active,
            )
        return u1


    e_old = sum(weights * mc.e_tot)
    U_acc = U0
    gn_components = _gn_details(
        U_acc,
        h1e,
        eri,
        dm1,
        dm2,
        contraction_plan,
        ncore=mc.ncore,
        ncas=ncas,
        active_active=active_active,
    )
    gn = gn_components["total"]
    U_target = opt_u(U_acc, dm1, dm2)
    U, step_norm = _limit_stiefel_displacement(U_acc, U_target, tr)
    last_mo_coeff = C0 @ U_acc
    best_e = float(np.real(np.asarray(e_old).reshape(-1)[0]))
    best_mc = mc
    best_C = last_mo_coeff

    converged = False
    k = 0
    while k < max_cycles:

        ok = False
        rej = 0
        for ir in range(int(macro_reject_max) + 1):
            mo_coeff = C0 @ U

            current_mc = _run_macro_casci(
                mc,
                nstates,
                mo_coeff=mo_coeff,
                warm_start=warm_start_dmrg,
                method=ci_method,
                **kwargs,
            )
            current_mc.nstates = nstates

            eAve = sum(weights * current_mc.e_tot)
            if _is_su2_dmrg(current_mc) and not _solver_converged(current_mc):
                diag.append(
                    {
                        "macro": k + 1,
                        "energy": float(np.real(np.asarray(eAve).reshape(-1)[0])),
                        "accepted": False,
                        "reason": "active_solver_unconverged",
                        "tr": tr,
                        "rej": rej,
                        "active_active_optimized": active_active,
                        "su2_runtime_rebuilt": bool(
                            getattr(
                                current_mc,
                                "_co_su2_runtime_rebuilt",
                                False,
                            )
                        ),
                        "solver_retried": bool(
                            getattr(current_mc, "_co_solver_retried", False)
                        ),
                        "solver": False,
                    }
                )
                break
            if (not reject_macro_energy) or eAve <= e_old + macro_energy_rise_tol:
                ok = True
                break

            rej += 1
            tr = None if tr is None else max(tr_min, tr * tr_dn)
            U, step_norm = _limit_stiefel_displacement(U_acc, U_target, tr)

        if not ok:
            break

        last_mo_coeff = mo_coeff
        e_history.append(current_mc.e_tot)
        de = float(np.real(np.asarray(eAve - e_old).reshape(-1)[0]))
        e_now = float(np.real(np.asarray(eAve).reshape(-1)[0]))
        dm1_new = 0
        dm2_new = 0
        for n in range(nstates):
            _dm1, _dm2 = current_mc.make_rdm12(n, with_core=with_core)
            dm1_new += _dm1 * weights[n]
            dm2_new += _dm2 * weights[n]
        gn_components_new = _gn_details(
            U,
            h1e,
            eri,
            dm1_new,
            dm2_new,
            contraction_plan,
            ncore=current_mc.ncore,
            ncas=ncas,
            active_active=active_active,
        )
        gn_new = gn_components_new["total"]
        row = {
            "macro": k + 1,
            "energy": e_now,
            "accepted": True,
            "dE": de,
            "gn": gn_new,
            "gn_nonredundant": gn_components_new["nonredundant"],
            "gn_active_active": gn_components_new["active_active"],
            "gn_core_core_discarded": gn_components_new[
                "core_core_discarded"
            ],
            "gn_start": gn,
            "gn_start_nonredundant": gn_components["nonredundant"],
            "gn_start_active_active": gn_components["active_active"],
            "step": step_norm,
            "tr": tr,
            "rej": rej,
            "active_active_optimized": active_active,
            "su2_runtime_rebuilt": bool(
                getattr(current_mc, "_co_su2_runtime_rebuilt", False)
            ),
            "solver_retried": bool(
                getattr(current_mc, "_co_solver_retried", False)
            ),
        }
        row.update(_sdiag(current_mc))
        gradient_spike = bool(gn > 0.0 and gn_new > 1.5 * gn)
        row["diis_reset"] = gradient_spike
        diag.append(row)
        if e_now < best_e:
            best_e = e_now
            best_mc = current_mc
            best_C = mo_coeff

        if gradient_spike and orbital_diis is not None:
            orbital_diis.reset()
        if tr is not None and rej == 0 and not gradient_spike:
            tr = min(tr_max, tr * tr_up)

        if abs(eAve - e_old) < tol and gn_new < gt:
            if getattr(mc, "verbose", 0) >= 1:
                print('CASSCF converged at macroiteration {}'.format(k))
                print("E(CASSCF) = {}".format(current_mc.e_tot))
            mc = current_mc
            converged = True
            break

        U_acc = U
        mc = current_mc
        e_old = eAve
        dm1, dm2 = dm1_new, dm2_new
        gn = gn_new
        gn_components = gn_components_new

        # Reuse the more conservative restart step from the state-specific
        # kernel.  The state-averaged surface is typically flatter, so jumping
        # back to the global default ``tau=2`` every macroiteration is often
        # too aggressive.
        U_target = opt_u(U_acc, dm1, dm2)
        U, step_norm = _limit_stiefel_displacement(U_acc, U_target, tr)
        # print(E + mol.energy_nuc())

        k += 1

    if not converged:
        if raise_on_nonconvergence:
            if diag and diag[-1].get("reason") == "active_solver_unconverged":
                raise RuntimeError(
                    "CO active-space DMRG did not converge after its warm "
                    "continuation; the orbital macro step was not accepted."
                )
            raise RuntimeError('Max macro steps reached. CASSCF not converged.')
        mc = best_mc
        mc.e_history = e_history
        mc.macro_diagnostics = diag
        _set_convergence_metadata(
            mc,
            macro_converged=False,
            macro_iterations=k,
        )
        return best_C, mc

    # As in the state-specific kernel, build a fresh final CASCI result so the
    # returned state-averaged orbitals and energies are self-consistent.
    final_mc = _fresh_casci_like(mc)
    final_su2_runtime_rebuilt = getattr(final_mc, "_su2_runtime", None) is not None
    if final_su2_runtime_rebuilt:
        final_mc._su2_runtime = None
    final_mc.spin_purification = mc.spin_purification
    final_mc.ss = mc.ss
    final_mc.shift = mc.shift
    final_mc.nstates = nstates
    for name in (
        "direct_ci_dense_fallback_ndets",
        "direct_ci_eigensolver",
        "direct_ci_max_cycle",
        "direct_ci_max_subspace",
        "direct_ci_reuse_guess",
    ):
        value = getattr(mc, name, None)
        if value is not None:
            setattr(final_mc, name, value)
    if warm_start_dmrg:
        _wguess(mc, final_mc)
    _run_casci_like(final_mc, nstates, mo_coeff=mo_coeff, method=ci_method, **kwargs)
    if getattr(final_mc, "build_info", None) is not None:
        final_mc.build_info[
            "final_su2_runtime_rebuilt"
        ] = bool(final_su2_runtime_rebuilt)
    final_mc.e_history = e_history
    final_mc.macro_diagnostics = diag
    _set_convergence_metadata(
        final_mc,
        macro_converged=True,
        macro_iterations=k + 1,
    )

    return mo_coeff, final_mc


# def constrained_optimization(U, h1e, h2e, dm1, dm2, max_steps=50):
#     """
#     complete active space orbital optimization with orthonomality constraint

#     .. math::
#         U^\top U = I_N

#         E = \sum_{p,q=1}^N t_{pq} U_{pp'} U_{q q'} \gamma_{p'q'} +
#         1/2 v_{pqrs} \Gamma_{p'q'r's'} U_{pp'}U_{qq'}U_{rr'}U_{ss'}

#     where U is a M x N (M > N) matrix.

#     .. math::
#         U_{k+1} = orth(U_k - \tau_k G_k)

#     Parameters
#     ----------
#     h1e : TYPE
#         DESCRIPTION.
#     h2e : TYPE
#         ERI.
#     dm1 : TYPE
#         1RDM.
#     dm2 : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     None.

#     """

#     # orb opt
#     converged = False
#     k = 0

#     # add random noise
#     U += 0.1 * np.random.randn(U.shape)
#     U = orth(U)

#     U_old = U.copy()
#     for k in range(max_steps):

#         G = gradient(U, h1e, h2e, dm1, dm2)
#         U = orth(U - stepsize(k) * G)

#         if 1 - abs(inner(U_old, U)) < 1e-3:
#             converged = True
#             break

#         U_old = U.copy()
#         k += 1

#     if converged:
#         return U
#     else:
#         raise RuntimeError('Constrained optimization not converged.')


def gradient(U, h1e, h2e, dm1, dm2):
    g = h1e @ U @ dm1.T + h1e.T @ U @ dm1  # these two terms are probably the same
    g += 0.5 * (contract('pqrs, qb, rc, sd, abcd -> pa', h2e, U, U, U, dm2) + \
        contract('pqrs, pa, rc, sd, abcd -> qb', h2e, U, U, U, dm2) + \
        contract('pqrs, pa, qb, sd, abcd -> rc', h2e, U, U, U, dm2) + \
        contract('pqrs, pa, qb, rc, abcd -> sd', h2e, U, U, U, dm2) )
    return g



class CASPT2(COCAS):
    """
    CASSCF
    """
    pass



if __name__=='__main__':

    from pyqed import Molecule
    # from pyqed.qchem.mcscf.direct_ci import CASCI

    mol = Molecule(atom='Li 0 0 0; F 0 0 1.4', unit='b', basis='6311g')
    mol.build()

    mf = mol.RHF().run()

    mc = COCAS(mf, ncas=6, nelecas=6, max_cycles=50)

    nstates = 2
    mc.state_average(weights = np.ones(nstates)/nstates)
    mc.fix_spin(ss=0, shift=0.2)
    mc.run()

    # correct result is E(CASSCF) = [-7.67160344]
    # energy logs for you to use
    print(mc.e_tot[0]) #ground state energy
    print(mc.e_tot[1]) #fitst excited state
    print([list(h) for h in mc.e_history]) #whole energy log in list
    print(mc.e_history) #whole energy log in array


# Backward-compatible alias for older imports.
COCASCI = COCAS
