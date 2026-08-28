"""Density-matrix-renormalization-group self-consistent field."""

import numpy as np

from pyqed.qchem.dmrg.dmrg import QCDMRG
from pyqed.qchem.mcscf.casscf import FirstOrderCASSCF, SecondOrderCASSCF
from pyqed.qchem.mcscf.cocas import _fresh_casci_like, kernel, kernel_state_average
from pyqed.qchem.mcscf.orbopt import embed_rdm2
from pyqed.qchem.mcscf.casci import (
    _get_mf_cholesky_factors,
    _resolve_use_cholesky_integrals,
    transform_eri_factors_to_mo_pair,
)


class _DMRGFirstOrderCASSCF(FirstOrderCASSCF):
    """Nonredundant orbital optimizer backed by the production DMRG solver."""

    def __init__(self, prototype, dmrg_options, **kwargs):
        super().__init__(prototype.mf, prototype.ncas, prototype.nelecas, **kwargs)
        self.prototype = prototype
        self.dmrg_options = dict(dmrg_options)
        self.weights = prototype.weights
        self.spin_purification = prototype.spin_purification
        self.ss = prototype.ss
        self.shift = prototype.shift

    @staticmethod
    def _copy_ci_guess(ci):
        if ci is None:
            return None
        return list(ci) if isinstance(ci, (list, tuple)) else [ci]

    def _objective_energy(self, mc, state_id):
        energies = np.asarray(mc.e_tot, dtype=float).reshape(-1)
        if self.weights is None:
            return float(energies[int(state_id)])
        return float(np.dot(np.asarray(self.weights, dtype=float), energies))

    def _effective_rdms(self, mc, state_id):
        dm1_occ, dm2_occ = self._effective_rdms_occ(mc, state_id)
        dm1 = np.zeros((self.nmo, self.nmo), dtype=dm1_occ.dtype)
        size = dm1_occ.shape[0]
        dm1[:size, :size] = dm1_occ
        return dm1, embed_rdm2(dm2_occ, self.nmo)

    def _make_casci(self, mo_coeff, nstates, ci0=None):
        mc = _fresh_casci_like(self.prototype, solver_cls=QCDMRG)
        mc._su2_runtime = None
        if ci0:
            mc.init_guess = ci0[0]
        options = dict(self.dmrg_options)
        options["require_convergence"] = False
        QCDMRG.run(
            mc,
            nstates=int(nstates),
            weights=self.weights,
            mo_coeff=mo_coeff,
            **options,
        )
        mc.ci = [mc.export_ground_state(state=root) for root in range(int(nstates))]
        self.ncore = mc.ncore
        return mc


class _DMRGSecondOrderCASSCF(SecondOrderCASSCF):
    """One-keyframe augmented-Hessian CASSCF backed by reduced DMRG."""

    def __init__(self, prototype, dmrg_options, **kwargs):
        super().__init__(prototype.mf, prototype.ncas, prototype.nelecas, **kwargs)
        self.prototype = prototype
        self.dmrg_options = dict(dmrg_options)
        self.weights = prototype.weights
        self.spin_purification = prototype.spin_purification
        self.ss = prototype.ss
        self.shift = prototype.shift

    _copy_ci_guess = staticmethod(_DMRGFirstOrderCASSCF._copy_ci_guess)
    _objective_energy = _DMRGFirstOrderCASSCF._objective_energy
    _effective_rdms = _DMRGFirstOrderCASSCF._effective_rdms

    def _run_dmrg(self, mean_field, mo_coeff, nstates, ci0):
        mc = _fresh_casci_like(self.prototype, solver_cls=QCDMRG)
        mc.mf = mean_field
        mc.mol = mean_field.mol
        mc._su2_runtime = None
        if ci0:
            mc.init_guess = ci0[0]
        options = dict(self.dmrg_options)
        options["require_convergence"] = False
        QCDMRG.run(
            mc,
            nstates=int(nstates),
            weights=self.weights,
            mo_coeff=mo_coeff,
            **options,
        )
        mc.ci = [mc.export_ground_state(state=root) for root in range(int(nstates))]
        self.ncore = mc.ncore
        return mc

    def _make_casci(self, mo_coeff, nstates, ci0=None):
        return self._run_dmrg(self.mf, mo_coeff, nstates, ci0)

    def _make_factor_integral_casci(
        self, h1_mo, pair_factors, mo_coeff, nstates, ci0=None
    ):
        frozen = self._FrozenFactorRHF(self.mf, h1_mo, pair_factors, mo_coeff)
        return self._run_dmrg(frozen, np.eye(self.nmo), nstates, ci0)

    def _make_integral_casci(
        self, h1_mo, eri_mo, mo_coeff, nstates, ci0=None
    ):
        frozen = self._FrozenIntegralRHF(self.mf, h1_mo, eri_mo, mo_coeff)
        return self._run_dmrg(frozen, np.eye(self.nmo), nstates, ci0)


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
        integral_backend=None,
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
        self.use_cholesky_integrals = False


        self.weights = None
        self.nstates = 1
        self.converged = False
        self.macro_converged = False
        self.solver_converged = False
        self.macro_iterations = 0


    def run(self, nstates=1, weights=None, require_conv=True, mo_coeff=None, **kwargs):
        mf = self.mf
        orbital_driver = str(kwargs.pop("orbital_driver", "constrained")).lower().replace(
            "-", "_"
        )
        if orbital_driver not in {"constrained", "nonredundant", "second_order"}:
            raise ValueError(
                "orbital_driver must be 'constrained', 'nonredundant', or "
                "'second_order'."
            )
        orbital_options = {
            "optimizer": kwargs.pop("optimizer", "RCG"),
            "optimizer_history": kwargs.pop("optimizer_history", 7),
            "optimizer_tol": kwargs.pop("optimizer_tol", 1.0e-4),
            "optimizer_max_steps": kwargs.pop("optimizer_max_steps", 200),
            "optimizer_max_step_norm": kwargs.pop("optimizer_max_step_norm", None),
            "diis": kwargs.pop("diis", True),
            "diis_space": kwargs.pop("diis_space", 6),
            "diis_start": kwargs.pop("diis_start", 2),
            "ci_method": kwargs.pop("ci_method", "direct_ci"),
        }
        orbital_micro_cycles = int(kwargs.pop("orbital_micro_cycles", 1))
        if orbital_micro_cycles < 1:
            raise ValueError("orbital_micro_cycles must be positive.")
        rej = kwargs.pop("reject_macro_energy", True)
        rise = kwargs.pop("macro_energy_rise_tol", 1.0e-8)
        rmax = kwargs.pop("macro_reject_max", 8)
        mtol = kwargs.pop("macro_tol", self.tol)
        gtol = kwargs.pop("orb_grad_tol", None)
        gtol_relaxed = kwargs.pop("orb_grad_tol_relaxed", gtol)
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
        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates
        if weights is not None:
            self.weights = weights
            if nstates != len(self.weights):
                raise ValueError("nstates must match the number of state-average weights.")

        nmo = self.mf.nao
        ncas = self.ncas
        nelecas = self.nelecas
        ncore = self.ncore

        kwargs.setdefault("sweep_tol", sw_tol)
        kwargs.setdefault("local_dense_max_dim", ldense)
        kwargs["require_convergence"] = False

        if orbital_driver in {"nonredundant", "second_order"}:
            orbital_use_cholesky = _resolve_use_cholesky_integrals(mf)
            self.use_cholesky_integrals = orbital_use_cholesky
            optimizer = orbital_options["optimizer"].upper()
            if optimizer not in {"DIAG", "LBFGS", "AH"}:
                optimizer = "LBFGS"
            max_step = orbital_options["optimizer_max_step_norm"]
            if max_step is None:
                max_step = 0.10 if tr is None else min(0.10, float(tr))
            common = dict(
                max_cycle=self.max_cycles,
                conv_tol=mtol,
                conv_tol_grad=1.0e-4 if gtol is None else gtol,
                conv_tol_grad_relaxed=(
                    1.0e-4 if gtol_relaxed is None else gtol_relaxed
                ),
                max_step=max_step,
                use_cholesky=orbital_use_cholesky,
                verbose=getattr(self, "verbose", 0),
            )
            if orbital_driver == "second_order":
                driver = _DMRGSecondOrderCASSCF(
                    self,
                    kwargs,
                    max_micro_cycle=orbital_micro_cycles,
                    coupling="qn",
                    micro_ci_mode="full",
                    optimizer="AH",
                    diis=False,
                    auto_active_restarts=False,
                    **common,
                )
            else:
                driver = _DMRGFirstOrderCASSCF(
                    self,
                    kwargs,
                    step_size=1.0,
                    optimizer=optimizer,
                    optimizer_history=orbital_options["optimizer_history"],
                    diis=orbital_options["diis"],
                    diis_space=orbital_options["diis_space"],
                    diis_start=orbital_options["diis_start"],
                    **common,
                )
            try:
                driver.run(
                    nstates=nstates,
                    mo_coeff=C0,
                    use_cholesky=orbital_use_cholesky,
                )
            except RuntimeError:
                if require_conv:
                    raise

            mc = driver.casci
            self.mo_coeff = driver.mo_coeff
            self.e_tot = mc.e_tot
            self.ci = mc.ci
            self.e_history = [row["energy"] for row in driver.history]
            self.macro_diagnostics = []
            previous = None
            for row in driver.history:
                energy = float(row["energy"])
                self.macro_diagnostics.append(
                    {
                        "macro": int(row["cycle"]),
                        "energy": energy,
                        "dE": None if previous is None else energy - previous,
                        "gn": float(row["gradient_norm"]),
                        "step": row.get("step_norm"),
                        "solver": bool(getattr(mc.dmrg, "converged", False)),
                    }
                )
                previous = energy
            self.macro_converged = bool(driver.converged)
            self.solver_converged = bool(getattr(mc.dmrg, "converged", False))
            self.converged = bool(self.macro_converged and self.solver_converged)
            self.macro_iterations = len(driver.history)
            self.dmrg = mc.dmrg
            self.H = getattr(mc, "H", None)
            self.H_raw = getattr(mc, "H_raw", None)
            self.e_core = getattr(mc, "e_core", None)
            self.integral_mode = getattr(
                mc,
                "integral_mode",
                self.integral_mode,
            )
            self.casci = mc
            if require_conv and not self.converged:
                raise RuntimeError(
                    "Nonredundant DMRGSCF did not converge both its orbital and "
                    "active-space solver criteria."
                )
            return self

        mc = _fresh_casci_like(self, solver_cls=QCDMRG)

        # DMRGSCF owns the final convergence policy so it can distinguish the
        # active-space solve from macro-iteration convergence in its error.
        mc.run(nstates=self.nstates, weights=self.weights, mo_coeff=C0, **kwargs)
        # matrix elements in CMOs
        h1e = mf.get_hcore_mo(C0)
        self.use_cholesky_integrals = _resolve_use_cholesky_integrals(mf)
        if self.use_cholesky_integrals:
            eri = transform_eri_factors_to_mo_pair(
                _get_mf_cholesky_factors(mf),
                C0,
            )
        else:
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
                **orbital_options,
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
                **orbital_options,
                **kwargs,
            )

        self.mo_coeff = C
        self.e_tot = mc.e_tot
        self.ci = getattr(mc, "ci", None)
        self.e_history = getattr(mc, 'e_history', [self.e_tot])
        self.macro_diagnostics = getattr(mc, "macro_diagnostics", [])
        self.dmrgscf_timing = getattr(mc, "dmrgscf_timing", {})
        self.converged = bool(getattr(mc, "converged", False))
        self.macro_converged = bool(getattr(mc, "macro_converged", False))
        self.solver_converged = bool(getattr(mc, "solver_converged", False))
        self.macro_iterations = int(getattr(mc, "macro_iterations", 0))
        self.dmrg = getattr(mc, "dmrg", None)
        self.H = getattr(mc, "H", None)
        self.H_raw = getattr(mc, "H_raw", None)
        self.e_core = getattr(mc, "e_core", None)
        self.integral_mode = getattr(
            mc,
            "integral_mode",
            self.integral_mode,
        )
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
