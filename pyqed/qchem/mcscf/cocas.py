#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 22:07:30 2025

@author: bingg
"""
import numpy as np
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


from pyqed.optimize import minimize
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

    def update(self, U):
        """Store the latest ``U`` and return a DIIS-mixed candidate if ready."""

        self.vectors.append(U.copy())
        if len(self.vectors) > 1:
            self.errors.append((self.vectors[-1] - self.vectors[-2]).copy())

        # ``errors`` is always one item shorter than ``vectors`` because each
        # error is defined as a difference between consecutive iterates.
        if len(self.errors) > self.max_space:
            self.errors.pop(0)
            self.vectors.pop(0)

        # The first extrapolated guess needs at least two error vectors, which
        # matches the original main-branch behavior.
        if len(self.errors) < self.start:
            return U

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
                return U

        U_new = np.zeros_like(U, dtype=U.dtype)
        for i, weight in enumerate(coeff[:-1]):
            U_new += weight * self.vectors[i + 1]

        return _orthonormalize_columns(U_new)


def _apply_orbital_diis(diis_helper, U, h1e, eri, dm1, dm2, current_energy):
    """Accept a DIIS-mixed ``U`` only if it improves the current objective.

    The historical main-branch DIIS scheme extrapolated blindly, which can
    overshoot when combined with the newer RCG/L-BFGS optimizers on ``bg``.
    We keep the same Pulay idea, but guard it with the same orbital objective
    that the minimizer is currently solving.
    """

    if diis_helper is None:
        return U

    U_diis = diis_helper.update(U)
    if U_diis is U:
        return U

    candidate_energy = energy(U_diis, h1e, eri, dm1, dm2)
    current_energy = np.asarray(current_energy).real.item()
    candidate_energy = np.asarray(candidate_energy).real.item()

    if np.isfinite(candidate_energy) and candidate_energy <= current_energy + 1.0e-10:
        return U_diis
    return U


def _fresh_casci_like(source):
    """Build a fresh CASCI object while preserving solver configuration."""

    if hasattr(source, "D"):
        mc = source.__class__(
            source.mf,
            ncas=source.ncas,
            nelecas=source.nelecas,
            D=source.D,
            init_guess=getattr(source, "init_guess", "hf"),
            m_warmup=getattr(source, "m_warmup", None),
            tol=getattr(source, "tol", 1.0e-6),
            low_rank_mpo=getattr(source, "low_rank_mpo", False),
            low_rank_mpo_bond=getattr(source, "low_rank_mpo_bond", None),
            low_rank_mpo_batch_size=getattr(source, "low_rank_mpo_batch_size", 4),
            site=getattr(source, "site", getattr(source, "site_basis", "spin_orbital")),
            spatial_reduced_mpo=getattr(source, "spatial_reduced_mpo", None),
            symmetry=getattr(source, "symmetry", None),
            spatial_site_basis=getattr(source, "spatial_site_basis", "canonical"),
            integral_backend=getattr(source, "integral_backend", "auto"),
            spatial_abelian_mpo=getattr(source, "spatial_abelian_mpo", "spatial"),
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
            spatial_enable_native_boundary_p=getattr(
                source,
                "spatial_enable_native_boundary_p",
                True,
            ),
            spatial_validate_native_boundary_p=getattr(
                source,
                "spatial_validate_native_boundary_p",
                True,
            ),
            spatial_native_boundary_p_validation_policy=getattr(
                source,
                "spatial_native_boundary_p_validation_policy",
                "first_pass",
            ),
            spatial_direct_operator_batch_min_entries=getattr(
                source,
                "spatial_direct_operator_batch_min_entries",
                2,
            ),
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
    mc.binary = getattr(source, 'binary', None)
    mc.direct_connectivity = getattr(source, 'direct_connectivity', None)
    mc.SC1 = getattr(source, 'SC1', None)
    mc.SC2 = getattr(source, 'SC2', None)
    return mc


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

    if not hasattr(src, "export_initial_guess"):
        return
    try:
        dst.init_guess = src.export_initial_guess(state=state)
    except Exception:
        return


def _cap(cap0, tr):
    if tr is None:
        return cap0
    if cap0 is None:
        return tr
    return min(float(cap0), float(tr))


def _gn(U, h1e, eri, dm1, dm2):
    g = opt_gradient(U, h1e, eri, dm1, dm2)
    return float(opt_norm(opt_grad(U, g)))


def _sdiag(mc):
    dmrg = getattr(mc, "dmrg", None)
    hist = getattr(dmrg, "sweep_history", None)
    out = {"solver": bool(getattr(dmrg, "converged", False))}
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
        super().__init__(mf, ncas, nelecas, verbose=verbose, **kwargs)

        self.max_cycles = max_cycles # macroiterations
        self.tol = float(macro_tol) # macro energy tol
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
                tol=self.tol,
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
                tol=self.tol,
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

    dm1, dm2 = mc.make_rdm12(0, with_core=with_core)

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
    diag = []

    def opt_u(u, d1, d2, st, cap, use_diis=True):
        u1, e1 = minimize(
            energy, u, args=(h1e, eri, d1, d2), tau=st,
            algorithm=optimizer, history_size=optimizer_history,
            epsilon=optimizer_tol,
            max_iterations=optimizer_max_steps,
            max_step_norm=cap,
        )
        if use_diis:
            u1 = _apply_orbital_diis(orbital_diis, u1, h1e, eri, d1, d2, e1)
        return u1

    U_acc = U0
    gn = _gn(U_acc, h1e, eri, dm1, dm2)
    U = opt_u(U_acc, dm1, dm2, 1.0, _cap(cap0, tr))

    k = 0

    e_old = mc.e_tot
    e_history = [mc.e_tot]
    last_mo_coeff = C0 @ U_acc
    best_e = float(np.real(np.asarray(mc.e_tot).reshape(-1)[0]))
    best_mc = mc
    best_C = last_mo_coeff

    converged = False
    while k < max_cycles:

        st = 1.0
        cap = _cap(cap0, tr)
        ok = False
        rej = 0
        for ir in range(int(macro_reject_max) + 1):
            mo_coeff = C0 @ U

            current_mc = _fresh_casci_like(mc)
            if warm_start_dmrg:
                _wguess(mc, current_mc)
            _run_casci_like(current_mc, mo_coeff=mo_coeff, method=ci_method, **kwargs)

            if (not reject_macro_energy) or current_mc.e_tot <= e_old + macro_energy_rise_tol:
                ok = True
                break

            rej += 1
            st *= 0.5
            tr = None if tr is None else max(tr_min, tr * tr_dn)
            cap = _cap(cap0, tr)
            U = opt_u(U_acc, dm1, dm2, st, cap, use_diis=False)

        if not ok:
            break

        last_mo_coeff = mo_coeff
        e_history.append(current_mc.e_tot)
        de = float(np.real(np.asarray(current_mc.e_tot - e_old).reshape(-1)[0]))
        e_now = float(np.real(np.asarray(current_mc.e_tot).reshape(-1)[0]))
        row = {"macro": k + 1, "energy": e_now, "dE": de, "gn": gn, "tr": tr, "rej": rej}
        row.update(_sdiag(current_mc))
        diag.append(row)
        if e_now < best_e:
            best_e = e_now
            best_mc = current_mc
            best_C = mo_coeff

        if tr is not None and rej == 0:
            tr = min(tr_max, tr * tr_up)

        if abs(current_mc.e_tot - e_old) < tol and gn < gt:
            if getattr(mc, "verbose", 0) >= 1:
                print('\nCASSCF converged at macroiteration {}'.format(k))
                print("E(CASSCF) = {}".format(current_mc.e_tot))
            mc = current_mc
            converged = True
            break

        U_acc = U
        mc = current_mc
        e_old = mc.e_tot


        dm1, dm2 = mc.make_rdm12(0, with_core=with_core)
        gn = _gn(U_acc, h1e, eri, dm1, dm2)

        U = opt_u(U_acc, dm1, dm2, 1.0, _cap(cap0, tr))
        # print(E + mol.energy_nuc())

        k += 1

    if not converged:
        if raise_on_nonconvergence:
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

    # Rebuild the final CASCI result from scratch at the returned orbitals.
    # Reusing the same CASCI object across many macroiterations can leave the
    # final reported energy out of sync with the best orbitals, while a fresh
    # CASCI solve at ``mo_coeff`` is consistent.
    final_mc = _fresh_casci_like(mc)
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
    final_mc.e_history = e_history
    final_mc.macro_diagnostics = diag
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
    diag = []

    def opt_u(u, d1, d2, st, cap, use_diis=True):
        u1, e1 = minimize(
            energy, u, args=(h1e, eri, d1, d2), tau=st,
            algorithm=optimizer, history_size=optimizer_history,
            epsilon=optimizer_tol,
            max_iterations=optimizer_max_steps,
            max_step_norm=cap,
        )
        if use_diis:
            u1 = _apply_orbital_diis(orbital_diis, u1, h1e, eri, d1, d2, e1)
        return u1


    e_old = sum(weights * mc.e_tot)
    U_acc = U0
    gn = _gn(U_acc, h1e, eri, dm1, dm2)
    U = opt_u(U_acc, dm1, dm2, 1.0, _cap(cap0, tr))
    last_mo_coeff = C0 @ U_acc
    best_e = float(np.real(np.asarray(e_old).reshape(-1)[0]))
    best_mc = mc
    best_C = last_mo_coeff

    converged = False
    k = 0
    while k < max_cycles:

        st = 1.0
        cap = _cap(cap0, tr)
        ok = False
        rej = 0
        for ir in range(int(macro_reject_max) + 1):
            mo_coeff = C0 @ U

            current_mc = _fresh_casci_like(mc)
            if warm_start_dmrg:
                _wguess(mc, current_mc)
            _run_casci_like(
                current_mc,
                nstates,
                mo_coeff=mo_coeff,
                method=ci_method,
                **kwargs,
            )
            current_mc.nstates = nstates

            eAve = sum(weights * current_mc.e_tot)
            if (not reject_macro_energy) or eAve <= e_old + macro_energy_rise_tol:
                ok = True
                break

            rej += 1
            st *= 0.5
            tr = None if tr is None else max(tr_min, tr * tr_dn)
            cap = _cap(cap0, tr)
            U = opt_u(U_acc, dm1, dm2, st, cap, use_diis=False)

        if not ok:
            break

        last_mo_coeff = mo_coeff
        e_history.append(current_mc.e_tot)
        de = float(np.real(np.asarray(eAve - e_old).reshape(-1)[0]))
        e_now = float(np.real(np.asarray(eAve).reshape(-1)[0]))
        row = {"macro": k + 1, "energy": e_now, "dE": de, "gn": gn, "tr": tr, "rej": rej}
        row.update(_sdiag(current_mc))
        diag.append(row)
        if e_now < best_e:
            best_e = e_now
            best_mc = current_mc
            best_C = mo_coeff

        if tr is not None and rej == 0:
            tr = min(tr_max, tr * tr_up)

        if abs(eAve - e_old) < tol and gn < gt:
            if getattr(mc, "verbose", 0) >= 1:
                print('CASSCF converged at macroiteration {}'.format(k))
                print("E(CASSCF) = {}".format(current_mc.e_tot))
            mc = current_mc
            converged = True
            break

        U_acc = U
        mc = current_mc
        e_old = eAve

        # update 1- and 2-RDMs
        dm1 = 0
        dm2 = 0
        for n in range(nstates):
            _dm1, _dm2 = mc.make_rdm12(n, with_core=with_core)
            dm1 += _dm1 * weights[n]
            dm2 += _dm2 * weights[n]
        gn = _gn(U_acc, h1e, eri, dm1, dm2)

        # Reuse the more conservative restart step from the state-specific
        # kernel.  The state-averaged surface is typically flatter, so jumping
        # back to the global default ``tau=2`` every macroiteration is often
        # too aggressive.
        U = opt_u(U_acc, dm1, dm2, 1.0, _cap(cap0, tr))
        # print(E + mol.energy_nuc())

        k += 1

    if not converged:
        if raise_on_nonconvergence:
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
    mol.build(driver='pyscf')

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
