"""State-averaged CASSCF nonadiabatic couplings.

This module provides a small backend for computing adiabatic derivative
coupling matrices

    D[a, i, j] = <Psi_i(R)|d Psi_j(R) / d R_a>

from state overlaps or SA-CASSCF response equations.  The public shape matches
the existing Ehrenfest NAC convention.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable

import numpy as np

from pyqed.qchem.mol import Molecule
from pyqed.qchem.basis_derivatives import (
    eri_derivatives,
    one_electron_derivatives,
    one_index_one_electron_derivatives,
)
from pyqed.qchem.grad import sacasscf as sacasscf_grad
from pyqed.qchem.mcscf.casci import CASCI, overlap
from pyqed.qchem.mcscf.orbopt import (
    generalized_fock,
    orbital_gradient,
    pack_nonredundant,
    unpack_nonredundant,
)
from pyqed.qchem.mcscf.reduced_ci import _transition_rdms_with_core
from pyqed.qchem.mcscf.zvector import MCSCFZVector, NACRHS, PropertyRHS


@dataclass
class FixedOrbitalCASCIDriver:
    """Minimal driver adapter for fixed-orbital CASCI NACs."""

    mf: object
    mo_coeff: np.ndarray
    ncore: int
    ncas: int
    weights: np.ndarray | None = None
    state_id: int = 0

    def __post_init__(self) -> None:
        self.mo_coeff = np.asarray(self.mo_coeff, dtype=float)
        self.ncore = int(self.ncore)
        self.ncas = int(self.ncas)
        self.nmo = int(self.mo_coeff.shape[1])

    def _get_integrals(self, mo_coeff):
        if not hasattr(self.mf, "get_hcore_mo") or not hasattr(self.mf, "get_eri_mo"):
            raise NotImplementedError(
                "CASCI NAC requires a reference with get_hcore_mo() and get_eri_mo()."
            )
        return (
            self.mf.get_hcore_mo(mo_coeff),
            self.mf.get_eri_mo(mo_coeff, notation="chem"),
        )

    def _active_integrals_from_full_mo(self, h1_mo, eri_mo, ncore, ncas):
        ncore = int(ncore)
        ncas = int(ncas)
        nocc = ncore + ncas
        active = slice(ncore, nocc)
        core = slice(0, ncore)
        h1_mo = np.asarray(h1_mo)
        eri_mo = np.asarray(eri_mo)
        h1_active = np.array(h1_mo[active, active], copy=True)
        if ncore > 0:
            core_j = 2.0 * np.einsum("pqii->pq", eri_mo[active, active, core, core], optimize=True)
            core_k = np.einsum("piqi->pq", eri_mo[active, core, active, core], optimize=True)
            h1_active = h1_active + core_j - core_k
        return h1_active, np.array(eri_mo[active, active, active, active], copy=True)

    def _make_active_sigma_casci(self, mc, h1_active, eri_active):
        if not hasattr(mc, "ci_sigma"):
            raise NotImplementedError(
                "CASCI NAC currently requires the direct-CI CASCI backend with ci_sigma()."
            )
        sigma_mc = copy.copy(mc)
        h1_active = np.asarray(h1_active, dtype=float)
        eri_active = np.asarray(eri_active, dtype=float)
        sigma_mc.hcore = np.asarray([h1_active, h1_active])
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
        return sigma_mc

    def _core_energy_derivative(self, dh1_mo, deri_mo, ncore):
        ncore = int(ncore)
        if ncore <= 0:
            return 0.0
        out = 0.0
        for i in range(ncore):
            out += 2.0 * dh1_mo[i, i]
        for i in range(ncore):
            for j in range(ncore):
                out += 2.0 * deri_mo[i, i, j, j] - deri_mo[i, j, j, i]
        return float(np.real(out))


def _one_sided_overlap_derivatives(mol: Molecule, *, step: float = 1.0e-4) -> np.ndarray:
    """Return d <AO(R)|AO(R + x)> / dx at the reference geometry."""

    _ = step
    return one_index_one_electron_derivatives(mol, "overlap", index="ket").reshape(
        -1,
        mol.nao,
        mol.nao,
    )


@dataclass
class ResponseBackend:
    """Backend adapter for SA-CASSCF NAC response operations.

    The default implementation wraps the native state-averaged CASSCF/CASCI
    objects and exposes the active-space data needed by the NAC/Z-vector
    builders.
    """

    driver: object
    mc: object
    nroots: int | None = None

    def __post_init__(self) -> None:
        if self.nroots is None:
            self.nroots = len(getattr(self.mc, "ci", ()))
        self.nroots = int(self.nroots)
        if self.nroots <= 0:
            raise ValueError("ResponseBackend requires at least one root.")

    @classmethod
    def from_driver(cls, driver, mc, *, nroots: int | None = None) -> "ResponseBackend":
        return cls(driver=driver, mc=mc, nroots=nroots)

    @property
    def mf(self):
        return self.driver.mf

    @property
    def nmo(self) -> int:
        return int(self.driver.nmo)

    @property
    def ncore(self) -> int:
        return int(self.mc.ncore)

    @property
    def ncas(self) -> int:
        return int(self.mc.ncas)

    @property
    def energies(self) -> np.ndarray:
        return np.asarray(self.mc.e_tot, dtype=float)[: self.nroots]

    @property
    def weights(self) -> np.ndarray:
        weights = getattr(self.driver, "weights", None)
        if weights is None:
            out = np.zeros(self.nroots, dtype=float)
            out[min(int(getattr(self.driver, "state_id", 0)), self.nroots - 1)] = 1.0
            return out
        out = np.asarray(weights, dtype=float)[: self.nroots]
        total = float(np.sum(out))
        if abs(total) <= 1.0e-14:
            raise ValueError("MCSCF response weights sum to zero.")
        return out / total

    @property
    def roots(self) -> list[np.ndarray]:
        return [np.asarray(root, dtype=float) for root in self.mc.ci[: self.nroots]]

    @property
    def ndet(self) -> int:
        return int(self.roots[0].size)

    @property
    def orbital_size(self) -> int:
        return pack_nonredundant(
            np.zeros((self.nmo, self.nmo)),
            self.ncore,
            self.ncas,
            self.nmo,
        ).size

    def active_integrals(self, h1_mo: np.ndarray, eri_mo: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.driver._active_integrals_from_full_mo(
            h1_mo,
            eri_mo,
            self.ncore,
            self.ncas,
        )

    def sigma_operator(self, h1_mo: np.ndarray, eri_mo: np.ndarray):
        h1_active, eri_active = self.active_integrals(h1_mo, eri_mo)
        return self.driver._make_active_sigma_casci(self.mc, h1_active, eri_active).ci_sigma

    def core_derivative(self, h1_mo: np.ndarray, eri_mo: np.ndarray) -> float:
        if hasattr(self.driver, "_core_energy_derivative"):
            return float(self.driver._core_energy_derivative(h1_mo, eri_mo, self.ncore))
        return 0.0

    def orbital_derivative_sigma_basis(
        self,
        h1_mo: np.ndarray,
        eri_mo: np.ndarray,
        ci: np.ndarray,
    ) -> np.ndarray:
        nvar = self.orbital_size
        if nvar == 0:
            return np.zeros((0, np.asarray(ci).size), dtype=float)
        eye = np.eye(nvar)
        dh1_cols = []
        deri_cols = []
        for iorb in range(nvar):
            kappa = unpack_nonredundant(eye[:, iorb], self.ncore, self.ncas, self.nmo)
            dh1, deri = self.driver._active_integral_derivatives_from_orbital_step(
                h1_mo,
                eri_mo,
                kappa,
                self.ncore,
                self.ncas,
            )
            dh1_cols.append(dh1)
            deri_cols.append(deri)
        return self.driver._batched_derivative_sigma(
            self.mc,
            np.asarray(dh1_cols),
            np.asarray(deri_cols),
            np.asarray(ci, dtype=float),
        )

    def matrix_element(
        self,
        h1_mo: np.ndarray,
        eri_mo: np.ndarray,
        bra: int,
        ket: int,
        *,
        include_core: bool = False,
    ) -> float:
        op = self.sigma_operator(h1_mo, eri_mo)
        value = float(np.dot(self.roots[int(bra)], op(self.roots[int(ket)])))
        if include_core and int(bra) == int(ket):
            value += self.core_derivative(h1_mo, eri_mo)
        return value

    def h_derivative_matrix(
        self,
        h1_mo: np.ndarray,
        eri_mo: np.ndarray,
        *,
        include_core: bool = True,
    ) -> np.ndarray:
        out = np.zeros((self.nroots, self.nroots), dtype=float)
        op = self.sigma_operator(h1_mo, eri_mo)
        roots = self.roots
        for bra in range(self.nroots):
            for ket in range(self.nroots):
                out[bra, ket] = float(np.dot(roots[bra], op(roots[ket])))
        if include_core:
            out += np.eye(self.nroots) * self.core_derivative(h1_mo, eri_mo)
        return out

    def energy_gradients(self, h1_mo: np.ndarray, eri_mo: np.ndarray) -> np.ndarray:
        return np.diag(self.h_derivative_matrix(h1_mo, eri_mo))

    def active_energy_gradients(self, h1_mo: np.ndarray, eri_mo: np.ndarray) -> np.ndarray:
        return np.diag(self.h_derivative_matrix(h1_mo, eri_mo, include_core=False))

    def stationarity_derivative(
        self,
        h1_mo: np.ndarray,
        eri_mo: np.ndarray,
        zvector: MCSCFZVector,
        *,
        project_ci: bool = True,
    ) -> np.ndarray:
        """Derivative of packed MCSCF stationarity equations for one coordinate."""

        if zvector.orbital_size != self.orbital_size:
            raise ValueError(
                f"zvector orbital_size {zvector.orbital_size} != backend orbital size {self.orbital_size}."
            )
        if zvector.nroots > self.nroots:
            raise ValueError("zvector requests more roots than backend provides.")
        roots = self.roots[: zvector.nroots]
        weights = self.weights[: zvector.nroots]
        op = self.sigma_operator(h1_mo, eri_mo)

        orbital = np.zeros(zvector.orbital_size, dtype=float)
        if zvector.orbital_size:
            for weight, root in zip(weights, roots, strict=True):
                if abs(weight) <= 1.0e-14:
                    continue
                orbital += float(weight) * self.driver._exact_orbital_gradient_vector(
                    self.mc,
                    h1_mo,
                    eri_mo,
                    root,
                )

        active_e_x = self.active_energy_gradients(h1_mo, eri_mo)[: zvector.nroots]
        ci_parts = []
        for weight, root, e_x in zip(weights, roots, active_e_x, strict=True):
            ci_part = op(root) - float(e_x) * root
            if project_ci:
                ci_part = _project_against_roots(ci_part, roots)
            ci_parts.append(float(weight) * np.asarray(ci_part, dtype=float))
        return np.concatenate((orbital, *ci_parts))


def nac_from_hamiltonian_derivatives(
    energies: np.ndarray,
    h_derivatives: np.ndarray,
    *,
    gap_threshold: float = 1.0e-8,
    antisymmetrize: bool = True,
) -> np.ndarray:
    """Return adiabatic NACs from off-diagonal Hamiltonian derivatives.

    Parameters
    ----------
    energies
        Adiabatic energies, shape ``(nstates,)``.
    h_derivatives
        Electronic Hamiltonian derivative matrix elements with shape
        ``(nstates, nstates, ncoord)`` and convention
        ``h_derivatives[beta, alpha, j] =
        <Psi_beta|dH/dq_j|Psi_alpha>``.

    Notes
    -----
    For nondegenerate adiabatic states,

    ``D[beta, alpha, j] = H'[beta, alpha, j] / (E_alpha - E_beta)``.

    This is the Hellmann-Feynman form.  For MCSCF, it is complete when the
    supplied Hamiltonian derivatives include the required orbital/CI response
    terms for the chosen state model.
    """

    energies = np.asarray(energies, dtype=float)
    h_derivatives = np.asarray(h_derivatives, dtype=complex)
    if energies.ndim != 1:
        raise ValueError("energies must be one-dimensional.")
    if h_derivatives.ndim != 3 or h_derivatives.shape[:2] != (energies.size, energies.size):
        raise ValueError(
            "h_derivatives must have shape (nstates, nstates, ncoord); "
            f"got {h_derivatives.shape} for {energies.size} states."
        )

    nstates, _, ncoord = h_derivatives.shape
    nac = np.zeros((nstates, nstates, ncoord), dtype=complex)
    for beta in range(nstates):
        for alpha in range(nstates):
            if beta == alpha:
                continue
            gap = energies[alpha] - energies[beta]
            if abs(gap) <= gap_threshold:
                continue
            nac[beta, alpha] = h_derivatives[beta, alpha] / gap
    if antisymmetrize:
        nac = 0.5 * (nac - np.swapaxes(nac.conj(), 0, 1))
        for state in range(nstates):
            nac[state, state] = 0.0
    return nac.real if np.max(np.abs(nac.imag)) < 1.0e-10 else nac


def analytic_nac(
    state_model,
    *,
    state_ids=None,
    modes=None,
    gap_threshold: float = 1.0e-8,
) -> np.ndarray:
    """Return Hellmann-Feynman NACs from CASSCF/CASCI vibronic couplings.

    ``state_model`` must provide ``e_tot`` and ``vibronic_couplings``.  The
    result has shape ``(nstates, nstates, ncoord)`` where ``ncoord`` is either
    ``3 * natom`` or the number of projected modes.
    """

    if not hasattr(state_model, "vibronic_couplings"):
        raise ValueError(
            f"{type(state_model).__name__} does not provide vibronic_couplings()."
        )
    if not hasattr(state_model, "e_tot"):
        raise ValueError(f"{type(state_model).__name__} does not provide e_tot.")

    if state_ids is None:
        energies = np.asarray(state_model.e_tot, dtype=float)
        state_ids = tuple(range(energies.size))
    else:
        state_ids = tuple(int(state) for state in state_ids)
        energies = np.asarray(state_model.e_tot, dtype=float)[list(state_ids)]

    f, _ = state_model.vibronic_couplings(state_ids=state_ids, modes=modes)
    f = np.asarray(f, dtype=complex)
    if modes is None:
        if f.ndim != 4:
            raise ValueError(f"Cartesian vibronic couplings must have ndim=4; got {f.shape}.")
        h_derivatives = f.reshape(f.shape[0], f.shape[1], -1)
    else:
        if f.ndim != 3:
            raise ValueError(f"Projected vibronic couplings must have ndim=3; got {f.shape}.")
        h_derivatives = f
    return nac_from_hamiltonian_derivatives(
        energies,
        h_derivatives,
        gap_threshold=gap_threshold,
    )


def _apply_ci_operator(operator, vector):
    if callable(operator):
        return np.asarray(operator(vector), dtype=float)
    return np.asarray(operator, dtype=float) @ np.asarray(vector, dtype=float)


def _project_against_roots(vector, roots):
    out = np.asarray(vector, dtype=float).copy()
    for root in roots:
        root = np.asarray(root, dtype=float)
        out -= root * float(np.dot(root, out))
    return out


def _symmetrized_transition_rdms_with_core(
    mc,
    cibra: np.ndarray,
    ciket: np.ndarray,
    *,
    nmo: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the real SA-CASSCF transition RDM convention used for NAC RHSs.

    Native CASCI stores the active 2-RDM as
    ``E = h[p,q] D[p,q] + 0.5 g[p,q,r,s] Gamma[p,q,r,s]``.  For NAC response
    the state-pair orbital RHS uses the Hermitian/symmetrized transition
    density, equivalent to ``0.5 * (Gamma + Gamma.transpose(1,0,3,2))`` for
    real CI vectors.
    """

    tdm1_ba, tdm2_ba = _transition_rdms_with_core(mc, cibra, ciket, nmo=nmo)
    tdm1_ab, tdm2_ab = _transition_rdms_with_core(mc, ciket, cibra, nmo=nmo)
    dm1 = 0.5 * (tdm1_ba + tdm1_ab)
    dm2 = 0.5 * (tdm2_ba + tdm2_ab)
    return dm1, dm2


def nac_rhs_from_hamiltonian_derivative(
    zvector: MCSCFZVector,
    energies: np.ndarray,
    derivative_operator,
    ci_roots,
    *,
    state_pair: tuple[int, int],
    orbital_gradient: np.ndarray | None = None,
    gap_threshold: float = 1.0e-8,
    project_ci: bool = True,
) -> NACRHS:
    """Build a NAC Z-vector RHS for one derivative Hamiltonian.

    The state pair follows the Hellmann-Feynman convention
    ``state_pair=(beta, alpha)`` for

    ``D[beta, alpha] = <Psi_beta|H'|Psi_alpha> / (E_alpha - E_beta)``.

    This helper fills the property-gradient blocks with respect to the packed
    orbital and CI response variables.  The optional ``orbital_gradient`` is the
    derivative of the numerator with respect to the packed orbital variables.
    CI blocks are built analytically as ``H'|other state> / gap`` and projected
    out of the optimized root space by default.
    """

    energies = np.asarray(energies, dtype=float)
    roots = [np.asarray(root, dtype=float).reshape(-1) for root in ci_roots]
    beta, alpha = (int(state_pair[0]), int(state_pair[1]))
    if beta == alpha:
        raise ValueError("NAC state_pair must contain two different states.")
    if beta < 0 or alpha < 0 or beta >= len(roots) or alpha >= len(roots):
        raise ValueError("state_pair indices must be available in ci_roots.")
    if energies.ndim != 1 or energies.size <= max(beta, alpha):
        raise ValueError("energies must contain both states in state_pair.")

    gap = float(energies[alpha] - energies[beta])
    if abs(gap) <= gap_threshold:
        raise ValueError(
            f"Cannot build NAC RHS for near-degenerate pair {state_pair}; gap={gap:.3e}."
        )

    if orbital_gradient is None:
        orbital = np.zeros(zvector.orbital_size)
    else:
        orbital = np.asarray(orbital_gradient, dtype=float).reshape(-1) / gap
        if orbital.size != zvector.orbital_size:
            raise ValueError(f"orbital_gradient size {orbital.size} != {zvector.orbital_size}.")

    ci_beta = _apply_ci_operator(derivative_operator, roots[alpha]) / gap
    ci_alpha = _apply_ci_operator(derivative_operator, roots[beta]) / gap
    if project_ci:
        ci_beta = _project_against_roots(ci_beta, roots[: zvector.nroots])
        ci_alpha = _project_against_roots(ci_alpha, roots[: zvector.nroots])

    return NACRHS.from_ci_state_pair(
        zvector,
        ci_beta,
        ci_alpha,
        state_pair=(beta, alpha),
        orbital=orbital,
    )


def nac_rhs_from_integrals(
    backend: ResponseBackend,
    zvector: MCSCFZVector,
    h1_derivative_mo: np.ndarray,
    eri_derivative_mo: np.ndarray,
    *,
    state_pair: tuple[int, int],
    energies: np.ndarray | None = None,
    gap_threshold: float = 1.0e-8,
    project_ci: bool = True,
) -> NACRHS:
    """Build a first analytic SA-CASSCF NAC RHS from derivative MO integrals.

    ``h1_derivative_mo`` and ``eri_derivative_mo`` are full-MO derivatives of
    the electronic Hamiltonian for one nuclear Cartesian coordinate or normal
    mode.  The returned RHS can be passed directly to ``MCSCFZVector.solve``.
    """

    if not isinstance(backend, ResponseBackend):
        raise TypeError("nac_rhs_from_integrals expects an ResponseBackend.")
    energies = backend.energies if energies is None else np.asarray(energies, dtype=float)
    beta, alpha = (int(state_pair[0]), int(state_pair[1]))
    roots = backend.roots
    if len(roots) < zvector.nroots:
        raise ValueError("backend does not contain zvector.nroots CI roots.")
    derivative_operator = backend.sigma_operator(h1_derivative_mo, eri_derivative_mo)

    gap = float(energies[alpha] - energies[beta])
    if abs(gap) <= gap_threshold:
        raise ValueError(
            f"Cannot build NAC RHS for near-degenerate pair {state_pair}; gap={gap:.3e}."
        )

    nvar = backend.orbital_size
    if nvar != zvector.orbital_size:
        raise ValueError(f"zvector orbital_size {zvector.orbital_size} != SA-CASSCF orbital size {nvar}.")

    orbital_gradient = np.zeros(nvar, dtype=float)
    if nvar:
        sigma_basis = backend.orbital_derivative_sigma_basis(
            h1_derivative_mo,
            eri_derivative_mo,
            roots[alpha],
        )
        orbital_gradient = sigma_basis @ roots[beta]

    return nac_rhs_from_hamiltonian_derivative(
        zvector,
        energies,
        derivative_operator,
        roots,
        state_pair=(beta, alpha),
        orbital_gradient=orbital_gradient,
        gap_threshold=gap_threshold,
        project_ci=project_ci,
    )


def mo_derivs(
    mf,
    mo_coeff=None,
    *,
    with_eri: bool = True,
    moving_basis: bool | str = True,
    overlap_step: float = 1.0e-4,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return first nuclear derivative integrals in an orthonormal MO frame.

    The first axis is flattened Cartesian nuclear coordinates
    ``atom * 3 + xyz``.  This uses the native builtin integral-derivative layer.
    By default, it uses the one-sided cross-overlap derivative
    ``d <AO(R)|AO(R + dR)>`` for the MO-frame transport, matching the gauge used
    by finite-difference state overlaps.  Pass ``moving_basis="symmetric"`` to
    recover the older ``-0.5 dS`` same-geometry transport, or ``False`` to leave
    the MO frame fixed.
    """

    if mo_coeff is None:
        mo_coeff = getattr(mf, "mo_coeff", None)
    if mo_coeff is None:
        raise ValueError("mo_coeff must be supplied or available as mf.mo_coeff.")
    mo_coeff = np.asarray(mo_coeff, dtype=float)
    mol = getattr(mf, "mol", None)
    if mol is None:
        raise ValueError("mf must provide mol for derivative integrals.")

    h1_ao = one_electron_derivatives(mol, "hcore", order=1).reshape(-1, mol.nao, mol.nao)
    h1_mo = np.einsum("pi,xpq,qj->xij", mo_coeff, h1_ao, mo_coeff, optimize=True)
    kappa_metric = None
    if moving_basis:
        transport = "one-sided" if moving_basis is True else str(moving_basis).lower().replace("_", "-")
        if transport in {"one-sided", "cross", "overlap"}:
            s1_ao = _one_sided_overlap_derivatives(mol, step=overlap_step)
            metric_factor = -1.0
        elif transport in {"symmetric", "same-geometry"}:
            s1_ao = one_electron_derivatives(mol, "overlap", order=1).reshape(-1, mol.nao, mol.nao)
            metric_factor = -0.5
        else:
            raise ValueError(
                "moving_basis must be False, True, 'one-sided', or 'symmetric'; "
                f"got {moving_basis!r}."
            )
        s1_mo = np.einsum("pi,xpq,qj->xij", mo_coeff, s1_ao, mo_coeff, optimize=True)
        h0_mo = np.asarray(mf.get_hcore_mo(mo_coeff), dtype=float)
        kappa_metric = metric_factor * s1_mo
        h1_mo = h1_mo + np.einsum("xai,aj->xij", kappa_metric, h0_mo, optimize=True)
        h1_mo = h1_mo + np.einsum("ia,xaj->xij", h0_mo, kappa_metric, optimize=True)
    if not with_eri:
        return h1_mo, None

    eri_ao = eri_derivatives(mol, order=1).reshape(
        -1,
        mol.nao,
        mol.nao,
        mol.nao,
        mol.nao,
    )
    eri_mo = np.einsum(
        "pi,qj,xpqrs,rk,sl->xijkl",
        mo_coeff,
        mo_coeff,
        eri_ao,
        mo_coeff,
        mo_coeff,
        optimize=True,
    )
    if kappa_metric is not None:
        eri0_mo = np.asarray(mf.get_eri_mo(mo_coeff, notation="chem"), dtype=float)
        eri_mo = eri_mo + np.einsum("xai,ajkl->xijkl", kappa_metric, eri0_mo, optimize=True)
        eri_mo = eri_mo + np.einsum("xaj,iakl->xijkl", kappa_metric, eri0_mo, optimize=True)
        eri_mo = eri_mo + np.einsum("xak,ijal->xijkl", kappa_metric, eri0_mo, optimize=True)
        eri_mo = eri_mo + np.einsum("xal,ijka->xijkl", kappa_metric, eri0_mo, optimize=True)
    return h1_mo, eri_mo


def nac_rhs_cartesian(
    backend: ResponseBackend,
    zvector: MCSCFZVector,
    *,
    state_pair: tuple[int, int],
    mo_coeff=None,
    h1_mo: np.ndarray | None = None,
    eri_mo: np.ndarray | None = None,
    gap_threshold: float = 1.0e-8,
    project_ci: bool = True,
) -> list[NACRHS]:
    """Build one NAC RHS per Cartesian nuclear coordinate."""

    if not isinstance(backend, ResponseBackend):
        raise TypeError("nac_rhs_cartesian expects an ResponseBackend.")
    if h1_mo is None or eri_mo is None:
        if mo_coeff is None:
            mo_coeff = getattr(backend.driver, "mo_coeff", None)
        h1_mo, eri_mo = mo_derivs(backend.mf, mo_coeff=mo_coeff, with_eri=True)
    h1_mo = np.asarray(h1_mo, dtype=float)
    eri_mo = np.asarray(eri_mo, dtype=float)
    if h1_mo.ndim != 3:
        raise ValueError("h1_mo must have shape (ncoord, nmo, nmo).")
    if eri_mo.ndim != 5 or eri_mo.shape[0] != h1_mo.shape[0]:
        raise ValueError("eri_mo must have shape (ncoord, nmo, nmo, nmo, nmo).")

    rhs = []
    for coord in range(h1_mo.shape[0]):
        rhs.append(
            nac_rhs_from_integrals(
                backend,
                zvector,
                h1_mo[coord],
                eri_mo[coord],
                state_pair=state_pair,
                gap_threshold=gap_threshold,
                project_ci=project_ci,
            )
        )
    return rhs


def nac_csf_cartesian(
    backend: ResponseBackend,
    *,
    mo_coeff=None,
    overlap_derivatives_ao: np.ndarray | None = None,
    state_pairs: list[tuple[int, int]] | tuple[tuple[int, int], ...] | None = None,
    overlap_step: float = 1.0e-4,
) -> np.ndarray:
    """Return the explicit CSF/basis-motion NAC contribution.

    The convention matches :func:`nac_from_hamiltonian_derivatives`:
    ``out[beta, alpha, x] = <Psi_beta|d Psi_alpha/dR_x>``.  This is the
    non-ETFS contribution in the full SA-CASSCF NAC implementation, expressed
    with the one-sided AO-overlap derivative used by the overlap NAC path.
    """

    if not isinstance(backend, ResponseBackend):
        raise TypeError("nac_csf_cartesian expects an ResponseBackend.")
    if mo_coeff is None:
        mo_coeff = getattr(backend.mf, "mo_coeff", None)
    if mo_coeff is None:
        raise ValueError("mo_coeff must be supplied or available as backend.mf.mo_coeff.")
    mo_coeff = np.asarray(mo_coeff, dtype=float)

    mol = getattr(backend.mf, "mol", None)
    if mol is None:
        raise ValueError("backend.mf must provide mol for AO-overlap derivatives.")
    if overlap_derivatives_ao is None:
        overlap_derivatives_ao = _one_sided_overlap_derivatives(mol, step=overlap_step)
    overlap_derivatives_ao = np.asarray(overlap_derivatives_ao, dtype=float)
    if overlap_derivatives_ao.ndim != 3:
        raise ValueError("overlap_derivatives_ao must have shape (ncoord, nao, nao).")

    ncoord = overlap_derivatives_ao.shape[0]
    out = np.zeros((backend.nroots, backend.nroots, ncoord), dtype=float)
    if state_pairs is None:
        state_pairs = [
            (beta, alpha)
            for beta in range(backend.nroots)
            for alpha in range(beta + 1, backend.nroots)
        ]

    ncore = backend.ncore
    ncas = backend.ncas
    mo_cas = mo_coeff[:, ncore : ncore + ncas]
    for beta, alpha in state_pairs:
        beta = int(beta)
        alpha = int(alpha)
        gamma = np.asarray(backend.mc.make_tdm1(beta, alpha), dtype=float)
        anti = gamma.T - gamma
        tm1_ao = mo_cas @ anti @ mo_cas.T
        values = 0.5 * np.einsum("xij,ij->x", overlap_derivatives_ao, tm1_ao, optimize=True)
        out[beta, alpha] = values
        out[alpha, beta] = -values
    return out


def nac_state_pair_response_rhs(
    backend: ResponseBackend,
    zvector: MCSCFZVector,
    *,
    state_pair: tuple[int, int],
    h1_mo: np.ndarray | None = None,
    eri_mo: np.ndarray | None = None,
    symmetrize_transition: bool = True,
) -> PropertyRHS:
    """Build the coordinate-independent SA-MCSCF NAC wavefunction RHS.

    This mirrors the usual SA-CASSCF NAC wavefunction-response structure: the
    RHS depends on the pair of electronic states, not on a nuclear derivative.
    Its solution must be contracted with the nuclear derivative of the
    state-averaged stationarity equations.  The current native contraction is
    still approximate, but this RHS is the correct object to expose and test.
    """

    if not isinstance(backend, ResponseBackend):
        raise TypeError("nac_state_pair_response_rhs expects an ResponseBackend.")
    if zvector.orbital_size != backend.orbital_size:
        raise ValueError(f"zvector orbital_size {zvector.orbital_size} != backend orbital_size {backend.orbital_size}.")
    if zvector.nroots > backend.nroots:
        raise ValueError("zvector requests more roots than backend provides.")
    if h1_mo is None or eri_mo is None:
        mo_coeff = getattr(backend.driver, "mo_coeff", None)
        if mo_coeff is None:
            mo_coeff = getattr(backend.mf, "mo_coeff", None)
        h1_mo, eri_mo = backend.driver._get_integrals(mo_coeff)

    beta, alpha = (int(state_pair[0]), int(state_pair[1]))
    roots = backend.roots
    if beta == alpha:
        raise ValueError("NAC state_pair must contain two different states.")
    if beta < 0 or alpha < 0 or beta >= len(roots) or alpha >= len(roots):
        raise ValueError("state_pair indices must be available in backend roots.")

    if symmetrize_transition:
        dm1, dm2 = _symmetrized_transition_rdms_with_core(
            backend.mc,
            roots[beta],
            roots[alpha],
            nmo=backend.nmo,
        )
    else:
        tdm1_ba, tdm2_ba = _transition_rdms_with_core(
            backend.mc,
            roots[beta],
            roots[alpha],
            nmo=backend.nmo,
        )
        dm1 = tdm1_ba
        dm2 = tdm2_ba

    fock = generalized_fock(h1_mo, eri_mo, dm1, dm2)
    orbital = pack_nonredundant(
        orbital_gradient(fock),
        backend.ncore,
        backend.ncas,
        backend.nmo,
    )

    # For converged SA-CASSCF roots these CI blocks should be tiny, but keeping
    # them makes the layout match the full state-pair Lagrange equation.
    ci_blocks = [np.zeros(zvector.ci_size) for _ in range(zvector.nroots)]
    active_energies = backend.energies[: zvector.nroots] - float(getattr(backend.mc, "e_core", 0.0))
    if beta < zvector.nroots:
        ci_blocks[beta] = 0.5 * (
            backend.mc.ci_sigma(roots[alpha]) - float(active_energies[alpha]) * roots[alpha]
        )
    if alpha < zvector.nroots:
        ci_blocks[alpha] = 0.5 * (
            backend.mc.ci_sigma(roots[beta]) - float(active_energies[beta]) * roots[beta]
        )
    for idx, block in enumerate(ci_blocks):
        if block.size != zvector.ci_size:
            raise ValueError(f"CI block {idx} size {block.size} != zvector.ci_size {zvector.ci_size}.")
        ci_blocks[idx] = _project_against_roots(block, roots[: zvector.nroots])

    return PropertyRHS.from_blocks(orbital, ci_blocks, state_pair=(beta, alpha))


@dataclass
class NACResult:
    """Analytic NAC data and optional Z-vector response solves."""

    energies: np.ndarray
    gradients: np.ndarray
    nac: np.ndarray
    explicit_nac: np.ndarray
    csf: np.ndarray
    correction: np.ndarray
    orbital_correction: np.ndarray
    ci_correction: np.ndarray
    h_derivatives: np.ndarray
    stationarity_derivatives: np.ndarray | None
    rhs: dict[tuple[int, int], list[NACRHS] | PropertyRHS]
    z: dict[tuple[int, int], list[object]]


def relaxed_nac(
    backend: ResponseBackend,
    zvector: MCSCFZVector | None = None,
    *,
    h1_mo: np.ndarray | None = None,
    eri_mo: np.ndarray | None = None,
    state_pairs: list[tuple[int, int]] | tuple[tuple[int, int], ...] | None = None,
    gap_threshold: float = 1.0e-8,
    solve_response: bool = True,
    include_csf: bool = False,
    response_rhs: str = "state-pair",
    response_contraction: str = "mo",
    ao_level_shift: float = 1.0e-8,
    moving_basis: bool | str = True,
    nac_gauge: str | None = None,
) -> NACResult:
    """Assemble analytic Cartesian NACs and optional Z-vector response data.

    The returned ``nac`` starts from the Hellmann-Feynman derivative-coupling
    matrix.  ``nac_gauge`` is the preferred high-level gauge selector:
    ``"overlap"`` keeps the one-sided overlap gauge, ``"full"`` selects the
    full NAC path, and ``"etfs"`` selects the ETFS path without the explicit
    CSF contribution.  The older ``include_csf`` and
    ``response_contraction`` options remain available for diagnostics.  When
    ``zvector`` is supplied,
    ``response_rhs="state-pair"`` builds the
    coordinate-independent SA-MCSCF NAC wavefunction RHS for each state pair.
    ``response_rhs="property"`` keeps the older coordinate-dependent property
    RHS for diagnostics.  With ``response_rhs="state-pair"``,
    ``response_contraction="ao"`` evaluates the final Lagrange contraction in
    AO-gradient form.  The default ``"mo"`` keeps the lightweight
    transported-MO stationarity derivative path.
    """

    if not isinstance(backend, ResponseBackend):
        raise TypeError("relaxed_nac expects an ResponseBackend.")
    if nac_gauge is not None:
        nac_gauge = str(nac_gauge).lower().replace("_", "-")
        if nac_gauge not in {"overlap", "full", "etfs"}:
            raise ValueError("nac_gauge must be 'overlap', 'full', or 'etfs'.")
        if nac_gauge == "overlap":
            include_csf = False
            moving_basis = True
        elif nac_gauge == "full":
            include_csf = True
            moving_basis = "symmetric"
            response_contraction = "ao"
        else:
            include_csf = False
            moving_basis = "symmetric"
            response_contraction = "ao"
    if h1_mo is None or eri_mo is None:
        mo_coeff = getattr(backend.driver, "mo_coeff", None)
        derivative_moving_basis = (
            "symmetric" if include_csf and moving_basis is True else moving_basis
        )
        h1_mo, eri_mo = mo_derivs(
            backend.mf,
            mo_coeff=mo_coeff,
            with_eri=True,
            moving_basis=derivative_moving_basis,
        )
    h1_mo = np.asarray(h1_mo, dtype=float)
    eri_mo = np.asarray(eri_mo, dtype=float)
    if h1_mo.ndim != 3:
        raise ValueError("h1_mo must have shape (ncoord, nmo, nmo).")
    if eri_mo.ndim != 5 or eri_mo.shape[0] != h1_mo.shape[0]:
        raise ValueError("eri_mo must have shape (ncoord, nmo, nmo, nmo, nmo).")

    ncoord = h1_mo.shape[0]
    h_derivatives = np.zeros((backend.nroots, backend.nroots, ncoord), dtype=float)
    for coord in range(ncoord):
        h_derivatives[:, :, coord] = backend.h_derivative_matrix(h1_mo[coord], eri_mo[coord])
    energies = backend.energies
    gradients = np.moveaxis(np.diagonal(h_derivatives, axis1=0, axis2=1), -1, 0)
    explicit_nac = nac_from_hamiltonian_derivatives(
        energies,
        h_derivatives,
        gap_threshold=gap_threshold,
    )
    csf = np.zeros_like(explicit_nac)
    if include_csf:
        csf = nac_csf_cartesian(
            backend,
            mo_coeff=getattr(backend.driver, "mo_coeff", None),
            state_pairs=state_pairs,
        )
    nac = np.array(explicit_nac - csf, copy=True)
    correction = np.zeros_like(nac, dtype=np.result_type(nac, float))
    orbital_correction = np.zeros_like(correction)
    ci_correction = np.zeros_like(correction)

    rhs_by_pair: dict[tuple[int, int], list[NACRHS] | PropertyRHS] = {}
    z_by_pair: dict[tuple[int, int], list[object]] = {}
    stationarity_derivatives = None
    if zvector is not None:
        response_rhs = str(response_rhs).lower().replace("_", "-")
        if response_rhs not in {"state-pair", "property"}:
            raise ValueError("response_rhs must be 'state-pair' or 'property'.")
        response_contraction = str(response_contraction).lower().replace("_", "-")
        if response_contraction not in {"mo", "ao"}:
            raise ValueError("response_contraction must be 'mo' or 'ao'.")
        if response_rhs == "property" and response_contraction == "ao":
            raise ValueError("response_contraction='ao' is only available with response_rhs='state-pair'.")
        stationarity_derivatives = np.asarray(
            [
                backend.stationarity_derivative(h1_mo[coord], eri_mo[coord], zvector)
                for coord in range(ncoord)
            ],
            dtype=float,
        )
        if state_pairs is None:
            state_pairs = [
                (beta, alpha)
                for beta in range(backend.nroots)
                for alpha in range(beta + 1, backend.nroots)
            ]
        for pair in state_pairs:
            beta, alpha = (int(pair[0]), int(pair[1]))
            if response_rhs == "property":
                pair_rhs = nac_rhs_cartesian(
                    backend,
                    zvector,
                    state_pair=pair,
                    h1_mo=h1_mo,
                    eri_mo=eri_mo,
                    gap_threshold=gap_threshold,
                )
                rhs_by_pair[tuple(pair)] = pair_rhs
                if solve_response:
                    pair_z = [zvector.solve(item) for item in pair_rhs]
                    z_by_pair[tuple(pair)] = pair_z
                    for coord, z_result in enumerate(pair_z):
                        orbital_value = float(
                            np.dot(
                                z_result.solution[: zvector.orbital_size],
                                stationarity_derivatives[coord, : zvector.orbital_size],
                            )
                        )
                        ci_value = float(
                            np.dot(
                                z_result.solution[zvector.orbital_size :],
                                stationarity_derivatives[coord, zvector.orbital_size :],
                            )
                        )
                        orbital_correction[beta, alpha, coord] += orbital_value
                        orbital_correction[alpha, beta, coord] -= orbital_value
                        ci_correction[beta, alpha, coord] += ci_value
                        ci_correction[alpha, beta, coord] -= ci_value
                        correction[beta, alpha, coord] += orbital_value + ci_value
                        correction[alpha, beta, coord] -= orbital_value + ci_value
            else:
                mo_coeff = getattr(backend.driver, "mo_coeff", None)
                if mo_coeff is None:
                    mo_coeff = getattr(backend.mf, "mo_coeff", None)
                h0_mo, eri0_mo = backend.driver._get_integrals(mo_coeff)
                pair_rhs = nac_state_pair_response_rhs(
                    backend,
                    zvector,
                    state_pair=pair,
                    h1_mo=h0_mo,
                    eri_mo=eri0_mo,
                )
                rhs_by_pair[tuple(pair)] = pair_rhs
                if solve_response:
                    z_kwargs = {"level_shift": float(ao_level_shift)} if response_contraction == "ao" else {}
                    z_result = zvector.solve(pair_rhs, **z_kwargs)
                    z_by_pair[tuple(pair)] = [z_result]
                    gap = float(energies[alpha] - energies[beta])
                    if abs(gap) > gap_threshold:
                        if response_contraction == "ao":
                            orbital_values = (
                                sacasscf_grad.lorb_dot_dgorb_cartesian(
                                    backend,
                                    z_result.solution[: zvector.orbital_size],
                                    mo_coeff=mo_coeff,
                                )
                                / gap
                            )
                            ci_values = (
                                sacasscf_grad.lci_dot_dgci_cartesian(
                                    backend,
                                    z_result.solution[zvector.orbital_size :],
                                    mo_coeff=mo_coeff,
                                )
                                / gap
                            )
                        else:
                            orbital_values = (
                                stationarity_derivatives[:, : zvector.orbital_size]
                                @ z_result.solution[: zvector.orbital_size]
                                / gap
                            )
                            ci_values = (
                                stationarity_derivatives[:, zvector.orbital_size :]
                                @ z_result.solution[zvector.orbital_size :]
                                / gap
                            )
                        values = orbital_values + ci_values
                        orbital_correction[beta, alpha] += orbital_values
                        orbital_correction[alpha, beta] -= orbital_values
                        ci_correction[beta, alpha] += ci_values
                        ci_correction[alpha, beta] -= ci_values
                        correction[beta, alpha] += values
                        correction[alpha, beta] -= values
        nac = nac + correction

    return NACResult(
        energies=energies,
        gradients=gradients,
        nac=nac,
        explicit_nac=explicit_nac,
        csf=csf,
        correction=correction,
        orbital_correction=orbital_correction,
        ci_correction=ci_correction,
        h_derivatives=h_derivatives,
        stationarity_derivatives=stationarity_derivatives,
        rhs=rhs_by_pair,
        z=z_by_pair,
    )


def casci_nac(
    mc,
    *,
    mf=None,
    mo_coeff=None,
    nroots: int | None = None,
    state_pairs: list[tuple[int, int]] | tuple[tuple[int, int], ...] | None = None,
    gap_threshold: float = 1.0e-8,
    include_csf: bool = False,
    moving_basis: bool | str = True,
    nac_gauge: str | None = None,
) -> NACResult:
    """Compute fixed-orbital CASCI NACs from native Hamiltonian derivatives.

    This is the CASCI analogue of :func:`relaxed_nac` with no orbital or
    SA-CASSCF Z-vector relaxation.  The orbitals are treated as fixed input
    orbitals, while the CI eigenvectors respond through the off-diagonal
    Hamiltonian derivative relation.
    """

    if mf is None:
        mf = getattr(mc, "mf", None)
    if mf is None:
        raise ValueError("mf must be supplied or available as mc.mf.")
    if mo_coeff is None:
        mo_coeff = getattr(mc, "mo_coeff", None)
    if mo_coeff is None:
        mo_coeff = getattr(mf, "mo_coeff", None)
    if mo_coeff is None:
        raise ValueError("mo_coeff must be supplied or available on mc/mf.")
    if not hasattr(mc, "ci_sigma"):
        raise NotImplementedError(
            "casci_nac currently requires the direct-CI CASCI backend with ci_sigma()."
        )

    driver = FixedOrbitalCASCIDriver(
        mf=mf,
        mo_coeff=np.asarray(mo_coeff, dtype=float),
        ncore=int(mc.ncore),
        ncas=int(mc.ncas),
    )
    backend = ResponseBackend.from_driver(driver, mc, nroots=nroots)
    return relaxed_nac(
        backend,
        zvector=None,
        state_pairs=state_pairs,
        gap_threshold=gap_threshold,
        solve_response=False,
        include_csf=include_csf,
        moving_basis=moving_basis,
        nac_gauge=nac_gauge,
    )


@dataclass
class NACScanner:
    """Scanner wrapper returning ``(energies, gradients, nac)``."""

    point_builder: Callable[[np.ndarray | None], object]
    solve_response: bool = False
    gap_threshold: float = 1.0e-8

    def _point(self, coords=None) -> tuple[ResponseBackend, MCSCFZVector | None]:
        point = self.point_builder(coords)
        if isinstance(point, ResponseBackend):
            return point, None
        if isinstance(point, tuple):
            if len(point) == 2:
                backend, zvector = point
                if not isinstance(backend, ResponseBackend):
                    raise TypeError("point_builder tuple must start with ResponseBackend.")
                return backend, zvector
        raise TypeError(
            "point_builder must return ResponseBackend or "
            "(ResponseBackend, MCSCFZVector)."
        )

    def evaluate(self, coords=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        backend, zvector = self._point(coords)
        result = relaxed_nac(
            backend,
            zvector,
            gap_threshold=self.gap_threshold,
            solve_response=self.solve_response,
        )
        return result.energies, result.gradients, result.nac

    def __call__(self, coords=None):
        return self.evaluate(coords)

    def as_scanner(self):
        return self


def _phase_align_columns(overlaps: np.ndarray) -> np.ndarray:
    aligned = np.array(overlaps, dtype=complex, copy=True)
    for state in range(min(aligned.shape)):
        diagonal = aligned[state, state]
        if abs(diagonal) > 1.0e-14:
            aligned[:, state] *= np.exp(-1j * np.angle(diagonal))
    return aligned


def nac_from_displaced_overlaps(
    overlap_plus: np.ndarray,
    overlap_minus: np.ndarray,
    step: float,
    *,
    phase_align: bool = True,
    antisymmetrize: bool = True,
) -> np.ndarray:
    """Return one derivative-coupling matrix from displaced overlaps.

    ``overlap_plus`` and ``overlap_minus`` must both use the current geometry
    as the bra and displaced geometries as kets:

        S_plus[i,j]  = <Psi_i(R)|Psi_j(R + step)>
        S_minus[i,j] = <Psi_i(R)|Psi_j(R - step)>
    """

    overlap_plus = np.asarray(overlap_plus, dtype=complex)
    overlap_minus = np.asarray(overlap_minus, dtype=complex)
    if overlap_plus.shape != overlap_minus.shape:
        raise ValueError(
            f"overlap_plus shape {overlap_plus.shape} != overlap_minus shape {overlap_minus.shape}."
        )
    if overlap_plus.ndim != 2 or overlap_plus.shape[0] != overlap_plus.shape[1]:
        raise ValueError("displaced overlaps must be square state-overlap matrices.")
    if step <= 0.0:
        raise ValueError("step must be positive.")

    if phase_align:
        overlap_plus = _phase_align_columns(overlap_plus)
        overlap_minus = _phase_align_columns(overlap_minus)
    nac = (overlap_plus - overlap_minus) / (2.0 * float(step))
    if antisymmetrize:
        nac = 0.5 * (nac - nac.conj().T)
        np.fill_diagonal(nac, 0.0)
    return nac


@dataclass
class OverlapNACDriver:
    """Finite-difference CASSCF/CASCI NAC driver.

    Parameters
    ----------
    mol
        Reference molecule.  Coordinates are assumed to be in bohr.
    ncas, nelecas, nstates
        CASCI active space and number of adiabatic roots.
    step
        Cartesian displacement in bohr.
    point_builder
        Optional callable ``point_builder(coords) -> mc`` for custom CASSCF or
        state-averaged CASSCF points.  If omitted, native RHF/CASCI is used.
    """

    mol: Molecule
    ncas: int
    nelecas: int | tuple[int, int]
    nstates: int
    step: float = 1.0e-3
    method: str = "direct_ci"
    mf_max_cycle: int = 80
    verbose: int = 0
    point_builder: Callable[[np.ndarray], object] | None = None

    def __post_init__(self) -> None:
        self.coords0 = np.asarray(self.mol.atom_coords(), dtype=float)
        if self.coords0.ndim != 2 or self.coords0.shape[1] != 3:
            raise ValueError("mol.atom_coords() must have shape (natom, 3).")
        if self.step <= 0.0:
            raise ValueError("step must be positive.")

    @property
    def ndof(self) -> int:
        return int(self.coords0.size)

    def _copy_molecule(self, coords: np.ndarray) -> Molecule:
        coords = np.asarray(coords, dtype=float).reshape(self.coords0.shape)
        mol = Molecule(
            atom=[
                [symbol, tuple(coord)]
                for symbol, coord in zip(self.mol.atom_symbols(), coords, strict=True)
            ],
            charge=self.mol.charge,
            spin=self.mol.spin,
            basis=self.mol.basis,
            unit="bohr",
        )
        mol.build()
        return mol

    def point(self, coords: np.ndarray):
        coords = np.asarray(coords, dtype=float).reshape(self.coords0.shape)
        if self.point_builder is not None:
            return self.point_builder(coords)
        mol = self._copy_molecule(coords)
        mf = mol.RHF(verbose=self.verbose).run(max_cycle=self.mf_max_cycle)
        return CASCI(mf, ncas=self.ncas, nelecas=self.nelecas, verbose=self.verbose).run(
            nstates=self.nstates,
            method=self.method,
        )

    def state_overlap(self, left, right) -> np.ndarray:
        if hasattr(left, "wavefunction_overlap"):
            value = left.wavefunction_overlap(right)
        elif hasattr(left, "overlap"):
            value = left.overlap(right)
        else:
            value = overlap(left, right)
        value = np.asarray(value, dtype=complex)
        return value[: self.nstates, : self.nstates]

    def nac(self, coords: np.ndarray | None = None, *, reference=None) -> np.ndarray:
        coords = self.coords0 if coords is None else np.asarray(coords, dtype=float).reshape(self.coords0.shape)
        ref = self.point(coords) if reference is None else reference
        flat = coords.reshape(-1)
        nac = np.zeros((self.nstates, self.nstates, flat.size), dtype=complex)
        for dof in range(flat.size):
            disp = np.zeros_like(flat)
            disp[dof] = self.step
            plus = self.point(flat + disp)
            minus = self.point(flat - disp)
            overlap_plus = self.state_overlap(ref, plus)
            overlap_minus = self.state_overlap(ref, minus)
            nac[:, :, dof] = nac_from_displaced_overlaps(
                overlap_plus,
                overlap_minus,
                self.step,
            )
        return nac.real if np.max(np.abs(nac.imag)) < 1.0e-10 else nac

    def evaluate(self, coords: np.ndarray | None = None):
        coords = self.coords0 if coords is None else np.asarray(coords, dtype=float).reshape(self.coords0.shape)
        ref = self.point(coords)
        energies = np.asarray(ref.e_tot[: self.nstates], dtype=float)
        return energies, self.nac(coords, reference=ref)


@dataclass
class AnalyticNACDriver:
    """NAC driver from CASSCF/CASCI Hamiltonian derivative couplings.

    This class wraps a completed CASSCF/CASCI-like object that exposes
    ``vibronic_couplings``.  It returns the nondegenerate Hellmann-Feynman NAC
    matrix.  Use ``OverlapNACDriver`` when the response-complete derivative
    couplings are not available.
    """

    state_model: object
    state_ids: tuple[int, ...] | None = None
    modes: np.ndarray | None = None
    gap_threshold: float = 1.0e-8

    def nac(self) -> np.ndarray:
        return analytic_nac(
            self.state_model,
            state_ids=self.state_ids,
            modes=self.modes,
            gap_threshold=self.gap_threshold,
        )

    def evaluate(self):
        if self.state_ids is None:
            energies = np.asarray(self.state_model.e_tot, dtype=float)
        else:
            energies = np.asarray(self.state_model.e_tot, dtype=float)[list(self.state_ids)]
        return energies, self.nac()
