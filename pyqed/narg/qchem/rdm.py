"""Dense RDM helpers for qchem NARG drivers."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from pyqed.mps.fermion import SpinHalfFermionChain
from pyqed.narg.core import narg_state_vector


def dense_root_from_narg_tensors(tensors, *, state_id: int = 0) -> np.ndarray:
    """Reconstruct one active-space root from stored NARG tensors."""
    if tensors is None or len(tensors) < 2:
        raise ValueError(
            "NARG RDMs require stored NARG tensors; rerun with store_tensors=True."
        )
    return narg_state_vector(tensors[:-1], tensors[-1], root=int(state_id))


@lru_cache(maxsize=32)
def _fermion_ops(ncas: int):
    ncas = int(ncas)
    h1e = np.zeros((ncas, ncas), dtype=float)
    eri = np.zeros((ncas, ncas, ncas, ncas), dtype=float)
    model = SpinHalfFermionChain(h1e, eri)
    model.jordan_wigner(forward=False)
    return tuple(model.Cu), tuple(model.Cd)


def _normalized_state(psi: np.ndarray) -> np.ndarray:
    psi = np.asarray(psi, dtype=complex).reshape(-1)
    norm = np.vdot(psi, psi)
    if abs(norm) <= 0.0:
        raise ValueError("cannot build RDMs from a zero-norm NARG state")
    return psi / np.sqrt(norm)


def spin_traced_rdm1_from_state(psi, ncas: int) -> np.ndarray:
    """Return active-space spin-traced 1-RDM from a dense Fock-space vector."""
    psi = _normalized_state(psi)
    ncas = int(ncas)
    cu, cd = _fermion_ops(ncas)
    annihilated = (
        [op @ psi for op in cu],
        [op @ psi for op in cd],
    )
    dm1 = np.zeros((ncas, ncas), dtype=complex)
    for p in range(ncas):
        for q in range(ncas):
            dm1[p, q] = (
                np.vdot(annihilated[0][p], annihilated[0][q])
                + np.vdot(annihilated[1][p], annihilated[1][q])
            )
    return np.real_if_close(dm1)


def spin_traced_rdm2_from_state(psi, ncas: int) -> np.ndarray:
    """Return active-space spin-traced 2-RDM from a dense Fock-space vector."""
    psi = _normalized_state(psi)
    ncas = int(ncas)
    cu, cd = _fermion_ops(ncas)
    annihilators = (cu, cd)
    first = tuple([op @ psi for op in ops] for ops in annihilators)

    pair = {}
    for sigma in range(2):
        for p in range(ncas):
            ket = first[sigma][p]
            for tau in range(2):
                for r in range(ncas):
                    pair[(sigma, p, tau, r)] = annihilators[tau][r] @ ket

    dm2 = np.zeros((ncas, ncas, ncas, ncas), dtype=complex)
    for p in range(ncas):
        for q in range(ncas):
            for r in range(ncas):
                for s in range(ncas):
                    value = 0.0j
                    for sigma in range(2):
                        for tau in range(2):
                            value += np.vdot(
                                pair[(sigma, p, tau, r)],
                                pair[(sigma, q, tau, s)],
                            )
                    dm2[p, q, r, s] = value
    return np.real_if_close(dm2)


def active_rdm1_from_narg(driver, state_id: int = 0) -> np.ndarray:
    ncas = _driver_ncas(driver)
    psi = dense_root_from_narg_tensors(driver.tensors, state_id=state_id)
    return spin_traced_rdm1_from_state(psi, ncas)


def active_rdm2_from_narg(driver, state_id: int = 0) -> np.ndarray:
    ncas = _driver_ncas(driver)
    psi = dense_root_from_narg_tensors(driver.tensors, state_id=state_id)
    return spin_traced_rdm2_from_state(psi, ncas)


def make_rdm1_from_narg(
    driver,
    state_id: int = 0,
    *,
    with_core: bool = False,
    with_vir: bool = False,
    representation: str = "mo",
    repr=None,
) -> np.ndarray:
    """Return a CASCI-style spin-traced 1-RDM for a NARG driver."""
    if repr is not None:
        representation = repr
    representation = str(representation).lower()
    if representation not in {"mo", "ao"}:
        raise ValueError("representation must be 'mo' or 'ao'.")

    dm1 = active_rdm1_from_narg(driver, state_id)
    out = _embed_rdm1(driver, dm1, with_core=with_core, with_vir=with_vir)
    if representation == "mo":
        return out
    coeff = _rdm_mo_coeff(driver, with_core=with_core, with_vir=with_vir)
    return coeff @ out @ coeff.conj().T


def make_rdm2_from_narg(
    driver,
    state_id: int = 0,
    *,
    with_core: bool = False,
    with_vir: bool = False,
) -> np.ndarray:
    """Return a CASCI-style spin-traced 2-RDM for a NARG driver."""
    dm1 = active_rdm1_from_narg(driver, state_id)
    dm2 = active_rdm2_from_narg(driver, state_id)
    return _embed_rdm2(driver, dm1, dm2, with_core=with_core, with_vir=with_vir)


def _driver_ncas(driver) -> int:
    ncas = getattr(driver, "ncas", None)
    if ncas is not None:
        return int(ncas)
    h1e = getattr(driver, "h1e", None)
    if h1e is None:
        raise ValueError("cannot infer active-space size for NARG RDMs")
    return int(np.asarray(h1e).shape[0])


def _driver_ncore(driver) -> int:
    return int(getattr(driver, "ncore", 0) or 0)


def _driver_nmo(driver, ncore: int, ncas: int, with_vir: bool) -> int:
    if with_vir:
        mf = getattr(driver, "mf", None)
        mo_coeff = getattr(mf, "mo_coeff", None)
        if mo_coeff is not None:
            return int(np.asarray(mo_coeff).shape[1])
        nmo = getattr(mf, "nmo", None)
        if nmo is not None:
            return int(nmo)
    return int(ncore) + int(ncas)


def _active_slice(driver, ncas: int) -> slice:
    return slice(_driver_ncore(driver), _driver_ncore(driver) + int(ncas))


def _embed_rdm1(driver, dm1, *, with_core: bool, with_vir: bool) -> np.ndarray:
    if not with_core and not with_vir:
        return dm1
    ncas = int(dm1.shape[0])
    ncore = _driver_ncore(driver)
    nmo = _driver_nmo(driver, ncore, ncas, with_vir)
    out = np.zeros((nmo, nmo), dtype=dm1.dtype)
    if with_core and ncore:
        out[np.arange(ncore), np.arange(ncore)] = 2.0
    active = _active_slice(driver, ncas)
    out[active, active] = dm1
    return out


def _embed_rdm2(driver, dm1, dm2, *, with_core: bool, with_vir: bool) -> np.ndarray:
    if not with_core and not with_vir:
        return dm2
    ncas = int(dm2.shape[0])
    ncore = _driver_ncore(driver)
    nmo = _driver_nmo(driver, ncore, ncas, with_vir)
    out = np.zeros((nmo, nmo, nmo, nmo), dtype=dm2.dtype)
    active = _active_slice(driver, ncas)

    if with_core and ncore:
        identity = np.eye(ncore, dtype=dm2.dtype)
        out[:ncore, :ncore, :ncore, :ncore] = (
            4.0 * np.einsum("ij,kl->ijkl", identity, identity)
            - 2.0 * np.einsum("ps,rq->pqrs", identity, identity)
        )
        for i in range(ncore):
            out[i, i, active, active] = 2.0 * dm1
            out[active, active, i, i] = 2.0 * dm1
            out[i, active, i, active] = -dm1
            out[active, i, active, i] = -dm1

    out[active, active, active, active] = dm2
    return out


def _rdm_mo_coeff(driver, *, with_core: bool, with_vir: bool) -> np.ndarray:
    mf = getattr(driver, "mf", None)
    mo_coeff = getattr(mf, "mo_coeff", None)
    if with_core or with_vir:
        if mo_coeff is None:
            raise ValueError("AO RDMs require mf.mo_coeff when core/virtual orbitals are requested")
        mo_coeff = np.asarray(mo_coeff)
        if with_vir:
            return mo_coeff
        nmo = _driver_ncore(driver) + _driver_ncas(driver)
        return mo_coeff[:, :nmo]
    mo_cas = getattr(driver, "mo_cas", None)
    if mo_cas is not None:
        return np.asarray(mo_cas)
    if mo_coeff is None:
        raise ValueError("AO active-space RDMs require active-space mo_coeff")
    ncore = _driver_ncore(driver)
    ncas = _driver_ncas(driver)
    return np.asarray(mo_coeff)[:, ncore : ncore + ncas]
