#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 18:10:50 2026

DMRGSCF

@author: Bing Gu (gubing at westlake dot edu dot cn)
"""
# TODO: so since we are sharing CASSCF optimization code, currently after the DMRGSCF, final print get E(CASSCF) = xxxxxxx, it might be better if we fix that.
from pyqed.qchem import CASSCF
from pyqed.qchem.dmrg.dmrg import QCDMRG
from pyqed.qchem.mcscf.cocas import kernel, kernel_state_average
import numpy as np


def _ao_overlap(mf):
    if hasattr(mf, "get_ovlp"):
        overlap = mf.get_ovlp()
    else:
        overlap = getattr(getattr(mf, "mol", None), "overlap", None)
    if overlap is None:
        return np.eye(int(getattr(mf, "nao")))
    return np.asarray(overlap)


def _s_orthonormalize(coeff, overlap, *, thresh=1.0e-10, orth_tol=1.0e-8):
    metric = coeff.conj().T @ overlap @ coeff
    eye = np.eye(metric.shape[0], dtype=metric.dtype)
    if metric.shape[0] == metric.shape[1] and np.linalg.norm(metric - eye) < orth_tol:
        return np.real_if_close(coeff)
    eig, vec = np.linalg.eigh(metric)
    if np.all(eig > thresh):
        chol = np.linalg.cholesky(metric)
        ortho = coeff @ np.linalg.inv(chol.conj().T)
        return np.real_if_close(ortho)
    keep = eig > thresh
    if not np.any(keep):
        raise ValueError("No linearly independent MO vectors remain after S-orthogonalization.")
    ortho = coeff @ (vec[:, keep] / np.sqrt(eig[keep]))
    return np.real_if_close(ortho)


def _complete_mo_basis(mf, mo_coeff):
    """
    Return a full MO basis whose leading columns span ``mo_coeff``.

    DMRGSCF stores the optimized core+active block.  When that block is reused
    as the next orbital-optimization start, complete it with the current HF MO
    basis so the optimizer can still rotate into the external space.
    """
    if mo_coeff is None:
        return np.asarray(mf.mo_coeff)

    coeff = np.asarray(mo_coeff)
    if coeff.ndim != 2:
        raise ValueError("mo_coeff must be a two-dimensional array.")
    nao = int(getattr(mf, "nao", coeff.shape[0]))
    if coeff.shape[0] != nao:
        raise ValueError(
            f"mo_coeff row dimension {coeff.shape[0]} does not match mf.nao={nao}."
        )
    if coeff.shape[1] > nao:
        raise ValueError(
            f"mo_coeff has too many columns ({coeff.shape[1]}) for nao={nao}."
        )

    overlap = _ao_overlap(mf)
    coeff = _s_orthonormalize(coeff, overlap)
    if coeff.shape[1] == nao:
        return coeff

    reference = np.asarray(mf.mo_coeff)
    if reference.shape[0] != nao:
        raise ValueError("mf.mo_coeff row dimension is inconsistent with mf.nao.")

    # Project the reference MO basis into the S-orthogonal complement of the
    # supplied block, then S-orthonormalize that complement.
    residual = reference - coeff @ (coeff.conj().T @ overlap @ reference)
    complement = _s_orthonormalize(residual, overlap)
    ncomp = nao - coeff.shape[1]
    if complement.shape[1] < ncomp:
        raise ValueError("Could not complete mo_coeff to a full-rank MO basis.")
    full = np.concatenate([coeff, complement[:, :ncomp]], axis=1)
    return np.real_if_close(full)


class DMRGSCF(QCDMRG):
    def __init__(
        self,
        mf,
        ncas,
        nelecas,
        D=20,
        max_cycles=30,
        macro_tol=1e-6,
        dmrg_conv_tol=1e-7,
        integral_backend="auto",
        **kwargs,
    ):
       
        super().__init__(
            mf,
            ncas,
            nelecas,
            D,
            integral_backend=integral_backend,
            **kwargs,
        )

        self.max_cycles = max_cycles # macroiterations
        self.tol = float(macro_tol) # macro energy tol
        self.dmrg_conv_tol = float(dmrg_conv_tol)
        self.mo_coeff = None # opt orb


        self.weights = None
        self.nstates = 1
        self.converged = False
        self.macro_converged = False
        self.solver_converged = False
        self.macro_iterations = 0


    def run(self, nstates=1, weights = None, require_conv=True, mo_coeff=None, **kwargs):
        mf = self.mf
        rej = kwargs.pop("reject_macro_energy", True)
        rise = kwargs.pop("macro_energy_rise_tol", 1.0e-8)
        rmax = kwargs.pop("macro_reject_max", 8)
        mtol = kwargs.pop("macro_tol", self.tol)
        gtol = kwargs.pop("orb_grad_tol", None)
        tr = kwargs.pop("macro_trust_radius", 0.25)
        tr_min = kwargs.pop("macro_trust_min", 1.0e-4)
        tr_max = kwargs.pop("macro_trust_max", 1.0)
        tr_dn = kwargs.pop("macro_trust_shrink", 0.5)
        tr_up = kwargs.pop("macro_trust_grow", 1.5)
        symmetry_labels = tuple(getattr(self, "symmetry", ()) or ())
        default_warm_start = not (
            int(nstates or self.nstates) > 1 and "su2" in symmetry_labels
        )
        warm = kwargs.pop("warm_start_dmrg", default_warm_start)
        sw_tol = kwargs.pop("sweep_tol", kwargs.pop("conv_tol", self.dmrg_conv_tol))
        ldense = kwargs.pop("local_dense_max_dim", 0)

        # Starting molecular orbitals for orbital optimization.  By default this
        # is the HF MO basis; callers can pass a previous DMRGSCF ``mo_coeff``
        # to continue from optimized orbitals.
        C0 = _complete_mo_basis(mf, mo_coeff)

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

        mc = QCDMRG(
            mf,
            ncas=ncas,
            nelecas=nelecas,
            D=self.D,
            site=getattr(self, "site", getattr(self, "site_basis", "spin_orbital")),
            spatial_reduced_mpo=getattr(self, "spatial_reduced_mpo", None),
            symmetry=getattr(self, "symmetry", None),
            spatial_site_basis=getattr(self, "spatial_site_basis", "canonical"),
            spatial_abelian_mpo=getattr(self, "spatial_abelian_mpo", "grouped"),
            spatial_abelian_symbolic_algo=getattr(
                self,
                "spatial_abelian_symbolic_algo",
                "Hopcroft-Karp",
            ),
            spatial_family_environment_backend=getattr(
                self,
                "spatial_family_environment_backend",
                "block2_table",
            ),
            spatial_native_p_grouping=getattr(
                self,
                "spatial_native_p_grouping",
                "first_site_order",
            ),
            spatial_block2_table_p_split_metric=getattr(
                self,
                "spatial_block2_table_p_split_metric",
                "auto",
            ),
            spatial_block2_table_p_split_groups=getattr(
                self,
                "spatial_block2_table_p_split_groups",
                "auto",
            ),
            spatial_block2_table_native_p=getattr(
                self,
                "spatial_block2_table_native_p",
                False,
            ),
            spatial_complementary_payload_tensor_matvec=getattr(
                self,
                "spatial_complementary_payload_tensor_matvec",
                True,
            ),
            spatial_precontracted_family_environment=getattr(
                self,
                "spatial_precontracted_family_environment",
                True,
            ),
            spatial_boundary_table_max_dim=getattr(
                self,
                "spatial_boundary_table_max_dim",
                32,
            ),
            spatial_exact_component_compression_policy=getattr(
                self,
                "spatial_exact_component_compression_policy",
                "auto",
            ),
            spatial_exact_component_compression_validate=getattr(
                self,
                "spatial_exact_component_compression_validate",
                True,
            ),
            spatial_exact_component_compression_validation_vectors=getattr(
                self,
                "spatial_exact_component_compression_validation_vectors",
                1,
            ),
            spatial_exact_component_compression_min_reduction=getattr(
                self,
                "spatial_exact_component_compression_min_reduction",
                1,
            ),
            spatial_exact_component_compression_max_group_size=getattr(
                self,
                "spatial_exact_component_compression_max_group_size",
                64,
            ),
            spatial_enable_native_boundary_p=getattr(
                self,
                "spatial_enable_native_boundary_p",
                True,
            ),
            spatial_validate_native_boundary_p=getattr(
                self,
                "spatial_validate_native_boundary_p",
                True,
            ),
            spatial_native_boundary_p_validation_policy=getattr(
                self,
                "spatial_native_boundary_p_validation_policy",
                "first_pass",
            ),
            spatial_direct_operator_batch_min_entries=getattr(
                self,
                "spatial_direct_operator_batch_min_entries",
                2,
            ),
            dmrg_performance=getattr(self, "dmrg_performance", "block2-like"),
            abelian_matvec_options=getattr(self, "abelian_matvec_options", None),
            debug_complementary_action_check=getattr(
                self,
                "debug_complementary_action_check",
                False,
            ),
            debug_complementary_action_check_tol=getattr(
                self,
                "debug_complementary_action_check_tol",
                1.0e-10,
            ),
            debug_complementary_action_check_limit=getattr(
                self,
                "debug_complementary_action_check_limit",
                32,
            ),
            debug_spatial_family_hamiltonian_check=getattr(
                self,
                "debug_spatial_family_hamiltonian_check",
                False,
            ),
            integral_backend=getattr(self, "integral_backend", "auto"),
            verbose=getattr(self, "verbose", 0),
        )

        # spin
        mc.spin_purification = self.spin_purification
        mc.ss = self.ss
        mc.shift = self.shift

        kwargs.setdefault("sweep_tol", sw_tol)
        kwargs.setdefault("local_dense_max_dim", ldense)

        mc.run(nstates=self.nstates, weights=self.weights, mo_coeff=C0, **kwargs)
        # matrix elements in CMOs
        h1e = mf.get_hcore_mo(C0)
        eri = mf.get_eri_mo(C0)

        U0 = np.zeros((nmo, ncas+ncore))
        for i in range(ncas+ncore):
            U0[i, i] = 1.

        if nstates == 1: # ground state only
            C, mc = kernel(
                mc,
                U0,
                nelecas,
                ncas,
                C0,
                h1e,
                eri,
                max_cycles=self.max_cycles,
                tol=mtol,
                orb_grad_tol=gtol,
                reject_macro_energy=rej,
                macro_energy_rise_tol=rise,
                macro_reject_max=rmax,
                macro_trust_radius=tr,
                macro_trust_min=tr_min,
                macro_trust_max=tr_max,
                macro_trust_shrink=tr_dn,
                macro_trust_grow=tr_up,
                warm_start_dmrg=warm,
                raise_on_nonconvergence=require_conv,
                **kwargs,
            )

        elif nstates > 1:
            if self.weights is None:
                self.state_average(weights = np.ones(nstates)/nstates)
            if len(self.weights) != nstates: 
                self.state_average(weights = np.ones(nstates)/nstates)
            mc.nstates = self.nstates
            C, mc = kernel_state_average(
                mc,
                weights=self.weights,
                U0=U0,
                nelecas=nelecas,
                ncas=ncas,
                C0=C0,
                h1e=h1e,
                eri=eri,
                max_cycles=self.max_cycles,
                tol=mtol,
                orb_grad_tol=gtol,
                reject_macro_energy=rej,
                macro_energy_rise_tol=rise,
                macro_reject_max=rmax,
                macro_trust_radius=tr,
                macro_trust_min=tr_min,
                macro_trust_max=tr_max,
                macro_trust_shrink=tr_dn,
                macro_trust_grow=tr_up,
                warm_start_dmrg=warm,
                raise_on_nonconvergence=require_conv,
                **kwargs,
            )

        self.mo_coeff = C
        self.e_tot = mc.e_tot
        self.ci = getattr(mc, "ci", None)
        self.e_history = getattr(mc, 'e_history', [self.e_tot])
        self.macro_diagnostics = getattr(mc, "macro_diagnostics", [])
        self.converged = bool(getattr(mc, "converged", False))
        self.macro_converged = bool(getattr(mc, "macro_converged", False))
        self.solver_converged = bool(getattr(mc, "solver_converged", False))
        self.macro_iterations = int(getattr(mc, "macro_iterations", 0))
        self.dmrg = getattr(mc, "dmrg", None)
        self.H = getattr(mc, "H", None)
        self.H_raw = getattr(mc, "H_raw", None)
        self.e_core = getattr(mc, "e_core", None)
        self.casci = mc

        if require_conv and not self.solver_converged:
            raise RuntimeError(
                "Final DMRGSCF active-space DMRG did not converge. "
                "Increase nsweeps or D, loosen conv_tol, or pass "
                "require_conv=False for debugging."
            )

        return self

    def state_average(self, weights):
        self.nstates = len(weights)
        self.weights = weights
        return self

if __name__=='__main__':

    from pyqed import Molecule

    mol = Molecule(atom='Li 0 0 0; F 0 0 1.4', unit='b', basis='6311g')
    mol.build(driver='pyscf')

    mf = mol.RHF().run()

    mc = DMRGSCF(mf, ncas=6, nelecas=6, D=60, max_cycles=50)

    mc.fix_spin(ss=0, shift=0.2)
    mc.run(
        nstates=1,
        symmetry_list=['charge', 'sz'], 
        initial_guess='cid'
    )
