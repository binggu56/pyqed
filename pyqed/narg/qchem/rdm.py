"""Reduced-density-matrix helpers for qchem NARG drivers."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from pyqed import SpinHalfFermionOperators
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


def _numerical_rank(singular_values, shape) -> int:
    singular_values = np.asarray(singular_values)
    if singular_values.size == 0 or singular_values[0] == 0:
        return 0
    threshold = (
        np.finfo(singular_values.dtype).eps
        * max(shape)
        * singular_values[0]
    )
    return int(np.count_nonzero(singular_values > threshold))


class _CarriedOccupationFactor:
    """Implicit ``(s, alpha)`` output bond for a conditional NARG factor."""

    def __init__(self, base):
        self.base = np.asarray(base)
        if self.base.ndim != 3 or self.base.shape[1] != 4:
            raise ValueError("a carried NARG factor must have shape (left, 4, right).")

    @property
    def shape(self):
        left, physical, right = self.base.shape
        return left, physical, physical * right


def _factor_storage_bytes(factor) -> int:
    if isinstance(factor, _CarriedOccupationFactor):
        return int(factor.base.nbytes)
    return int(np.asarray(factor).nbytes)


def _narg_spatial_mps(driver, state_id: int):
    """Convert one-site NARG factors to an exact conventional d=4 MPS."""
    tensors = getattr(driver, "tensors", None)
    if tensors is None or len(tensors) < 2:
        return None
    factors = [np.asarray(tensor) for tensor in tensors[:-1]]
    coeff = np.asarray(tensors[-1])
    if any(factor.ndim != 3 for factor in factors):
        return None

    n0 = getattr(driver, "n0", None)
    if n0 is None:
        return None
    n0 = int(n0)
    first = factors[0]
    if n0 < 1 or first.shape[0] != 4**n0:
        return None
    if coeff.ndim != 3 or coeff.shape[0] not in {1, 4}:
        return None
    state_id = int(state_id)
    if state_id < 0 or state_id >= coeff.shape[2]:
        raise IndexError("state_id is out of range for stored NARG roots.")

    # The first NARG factor spans the exact n0-orbital prefix and the first
    # appended orbital. Split that finite prefix exactly by TT-SVD.
    prefix = first.reshape(*(4,) * n0, first.shape[1], 4)
    order = (*range(n0), n0 + 1, n0)
    remainder = prefix.transpose(order)[None, ...]
    mps = []
    left_dim = 1
    for _site in range(n0):
        matrix = remainder.reshape(left_dim * 4, -1)
        left, singular, right = np.linalg.svd(matrix, full_matrices=False)
        rank = _numerical_rank(singular, matrix.shape)
        if rank == 0:
            raise ValueError("stored NARG prefix has zero norm.")
        mps.append(left[:, :rank].reshape(left_dim, 4, rank))
        remainder = (singular[:rank, None] * right[:rank]).reshape(
            rank, *remainder.shape[2:]
        )
        left_dim = rank

    if remainder.ndim != 3:
        raise ValueError("stored NARG prefix has an invalid physical layout.")
    mps.append(remainder)
    right_dim = remainder.shape[2]

    # General detached/CC projectors are compact append factors. Standard NARG
    # factors additionally depend on the previous occupation. Carry that label
    # implicitly on the left bond, avoiding a zero-padded tensor four times the
    # size of its conditional factor.
    for factor in factors[1:]:
        if factor.shape[2] != 4:
            return None
        if factor.shape[0] == 4 * right_dim:
            mps[-1] = _CarriedOccupationFactor(mps[-1])
        elif factor.shape[0] != right_dim:
            return None
        site_tensor = factor.transpose(0, 2, 1)
        mps.append(site_tensor)
        right_dim = site_tensor.shape[2]

    if coeff.shape[1] != right_dim:
        return None
    if coeff.shape[0] == 1:
        mps[-1] = np.einsum(
            "lpr,r->lp",
            mps[-1],
            coeff[0, :, state_id],
            optimize=True,
        )[..., None]
    else:
        mps[-1] = np.einsum(
            "lpr,pr->lp",
            mps[-1],
            coeff[:, :, state_id],
            optimize=True,
        )[..., None]
    if len(mps) != _driver_ncas(driver):
        return None
    return mps


class _SpatialMPSContractions:
    """Memory-bounded spatial-fermion string contractions for one NARG root."""

    def __init__(self, factors):
        self.factors = [
            factor
            if isinstance(factor, _CarriedOccupationFactor)
            else np.asarray(factor)
            for factor in factors
        ]
        local = SpinHalfFermionOperators()
        self.local_ops = {
            "I": np.eye(4, dtype=complex),
            "JW": np.asarray(local["JW"], dtype=complex),
            ("ann", 0): np.asarray(local["Cu"], dtype=complex),
            ("ann", 1): np.asarray(local["Cd"], dtype=complex),
            ("cre", 0): np.asarray(local["Cdu"], dtype=complex),
            ("cre", 1): np.asarray(local["Cdd"], dtype=complex),
        }
        self._site_op_cache = {}
        self.left_env, self.right_env = self._identity_environments()
        self.factor_storage_bytes = sum(
            _factor_storage_bytes(factor) for factor in self.factors
        )
        self.environment_storage_bytes = sum(
            environment.nbytes
            for environment in (*self.left_env, *self.right_env)
            if environment is not None
        )
        self.implicit_factor_count = sum(
            isinstance(factor, _CarriedOccupationFactor)
            for factor in self.factors
        )
        self.norm = self._transfer(
            self.left_env[-1],
            self.factors[-1],
            self.local_ops["I"],
        )[0, 0]
        if abs(self.norm) <= 1.0e-14:
            raise ValueError("cannot build RDMs from a zero-norm NARG state")

    @staticmethod
    def _transfer(environment, factor, operator):
        if isinstance(factor, _CarriedOccupationFactor):
            base = factor.base
            if environment.ndim == 1:
                transferred = np.einsum(
                    "i,isa,st,itb->satb",
                    environment,
                    base.conj(),
                    operator,
                    base,
                    optimize=True,
                )
            else:
                transferred = np.einsum(
                    "ij,isa,st,jtb->satb",
                    environment,
                    base.conj(),
                    operator,
                    base,
                    optimize=True,
                )
            return transferred.reshape(4 * base.shape[2], 4 * base.shape[2])
        if environment.ndim == 1:
            return np.einsum(
                "a,asr,st,atu->ru",
                environment,
                factor.conj(),
                operator,
                factor,
                optimize=True,
            )
        return np.einsum(
            "ab,asr,st,btu->ru",
            environment,
            factor.conj(),
            operator,
            factor,
            optimize=True,
        )

    @staticmethod
    def _transfer_right(environment, factor, operator):
        if isinstance(factor, _CarriedOccupationFactor):
            base = factor.base
            environment = environment.reshape(
                4, base.shape[2], 4, base.shape[2]
            )
            return np.einsum(
                "satb,isa,st,jtb->ij",
                environment,
                base.conj(),
                operator,
                base,
                optimize=True,
            )
        return np.einsum(
            "ru,asr,st,btu->ab",
            environment,
            factor.conj(),
            operator,
            factor,
            optimize=True,
        )

    def _identity_environments(self):
        left = [np.array([1.0 + 0.0j])]
        for factor in self.factors[:-1]:
            environment = self._transfer(left[-1], factor, self.local_ops["I"])
            diagonal = np.diag(environment)
            total_norm_sq = float(np.vdot(environment, environment).real)
            diagonal_norm_sq = float(np.vdot(diagonal, diagonal).real)
            off_diagonal_norm = np.sqrt(
                max(total_norm_sq - diagonal_norm_sq, 0.0)
            )
            if off_diagonal_norm <= 1.0e-12 * max(
                np.sqrt(total_norm_sq), 1.0
            ):
                environment = diagonal.copy()
            left.append(environment)

        right = [None] * len(self.factors)
        right[-1] = np.array([[1.0 + 0.0j]])
        for site in range(len(self.factors) - 1, 0, -1):
            factor = self.factors[site]
            right[site - 1] = self._transfer_right(
                right[site],
                factor,
                self.local_ops["I"],
            )
        return left, right

    def _site_operator(self, op_specs, site):
        key = []
        for kind, spin, op_site in op_specs:
            if site > op_site:
                key.append("JW")
            elif site == op_site:
                key.append((kind, spin))
        key = tuple(key)
        operator = self._site_op_cache.get(key)
        if operator is None:
            operator = self.local_ops["I"]
            for part in key:
                operator = operator @ self.local_ops[part]
            self._site_op_cache[key] = operator
        return operator

    def _close(self, environment, site):
        return np.sum(environment * self.right_env[site])

    def expect_string(self, op_specs):
        first = min(site for _kind, _spin, site in op_specs)
        last = max(site for _kind, _spin, site in op_specs)
        environment = self.left_env[first]
        for site in range(first, last + 1):
            environment = self._transfer(
                environment,
                self.factors[site],
                self._site_operator(op_specs, site),
            )
        return self._close(environment, last) / self.norm

    def make_rdm1(self):
        ncas = len(self.factors)
        dm1 = np.zeros((ncas, ncas), dtype=complex)
        for p in range(ncas):
            for q in range(p, ncas):
                value = sum(
                    self.expect_string((("cre", spin, p), ("ann", spin, q)))
                    for spin in range(2)
                )
                dm1[p, q] = value
                dm1[q, p] = value.conjugate()
        return np.real_if_close(dm1)

    def make_rdm2(self):
        ncas = len(self.factors)
        dm2 = np.zeros((ncas, ncas, ncas, ncas), dtype=complex)
        pairs = [(p, r) for p in range(ncas) for r in range(ncas)]
        for spin in range(2):
            for other_spin in range(2):
                for left_index, (p, r) in enumerate(pairs):
                    for q, s in pairs[left_index:]:
                        value = self.expect_string(
                            (
                                ("cre", spin, p),
                                ("cre", other_spin, r),
                                ("ann", other_spin, s),
                                ("ann", spin, q),
                            )
                        )
                        dm2[p, q, r, s] += value
                        if (p, r) != (q, s):
                            dm2[q, p, s, r] += value.conjugate()
        return np.real_if_close(dm2)


def _rdm_cache(driver):
    tensors = getattr(driver, "tensors", None)
    signature = id(tensors)
    cache = getattr(driver, "_narg_rdm_cache", None)
    if cache is None or cache.get("signature") != signature:
        cache = {"signature": signature, "roots": {}}
        driver._narg_rdm_cache = cache
    return cache["roots"]


def _root_rdm_cache(driver, state_id: int):
    roots = _rdm_cache(driver)
    state_id = int(state_id)
    root = roots.get(state_id)
    if root is None:
        factors = _narg_spatial_mps(driver, state_id)
        root = {
            "contractions": (
                None if factors is None else _SpatialMPSContractions(factors)
            )
        }
        roots[state_id] = root
    return root


def _set_tensor_backend_diagnostics(driver, contractions):
    driver.rdm_backend = "tensor"
    driver.rdm_factor_storage_bytes = contractions.factor_storage_bytes
    driver.rdm_environment_storage_bytes = contractions.environment_storage_bytes
    driver.rdm_implicit_factor_count = contractions.implicit_factor_count


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
    cache = _root_rdm_cache(driver, state_id)
    if "dm1" in cache:
        return cache["dm1"]
    contractions = cache["contractions"]
    if contractions is not None:
        dm1 = contractions.make_rdm1()
        _set_tensor_backend_diagnostics(driver, contractions)
        cache["dm1"] = dm1
        return dm1
    psi = dense_root_from_narg_tensors(driver.tensors, state_id=state_id)
    dm1 = spin_traced_rdm1_from_state(psi, ncas)
    driver.rdm_backend = "dense"
    cache["dm1"] = dm1
    return dm1


def active_rdm2_from_narg(driver, state_id: int = 0) -> np.ndarray:
    ncas = _driver_ncas(driver)
    cache = _root_rdm_cache(driver, state_id)
    if "dm2" in cache:
        return cache["dm2"]
    contractions = cache["contractions"]
    if contractions is not None:
        dm2 = contractions.make_rdm2()
        _set_tensor_backend_diagnostics(driver, contractions)
        cache["dm2"] = dm2
        return dm2
    psi = dense_root_from_narg_tensors(driver.tensors, state_id=state_id)
    dm2 = spin_traced_rdm2_from_state(psi, ncas)
    driver.rdm_backend = "dense"
    cache["dm2"] = dm2
    return dm2


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
