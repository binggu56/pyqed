#!/usr/bin/env python3
"""Quantum-chemistry adapter for the generic NARG LETTA optimizer."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from pyqed.letta import LETTA as GenericLETTA

_DEFAULT_MPO_CACHE = {}
_DEFAULT_MPO_CACHE_MAXSIZE = 4


def _first_attr(obj, names):
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return None


@dataclass
class LETTA:
    """Quantum-chemistry LETTA driver.

    This class is intentionally a thin domain adapter: quantum-chemistry code
    is responsible for preparing NARG factors, Hamiltonians, or MPOs, while the
    tensor optimization is delegated to :class:`pyqed.letta.LETTA`.
    """

    engine: GenericLETTA
    hamiltonian: object = None
    mpo: object = None
    mol: object = None
    source: object = None
    site: str = "spatial"
    e_tot: object = None
    spin_info: object = None
    qn_info: object = None

    @classmethod
    def from_integrals(
        cls,
        h1e,
        eri=None,
        *,
        symmetry="su2",
        **kwargs,
    ):
        """Build the native reduced qchem LETTA selected by ``symmetry``.

        The first native path is SU(2): ties condition on invariant local
        multiplet labels and the Hamiltonian remains a rank-coupled reduced
        MPO throughout contraction and optimization.
        """
        key = str(symmetry).lower().replace("-", "").replace("_", "")
        if key != "su2":
            raise NotImplementedError(
                "LETTA.from_integrals currently implements symmetry='su2'; "
                "use FrontierLETTA with a canonical Hamiltonian for dense/U(1) work."
            )
        from pyqed.letta import SU2LETTA

        return SU2LETTA.from_integrals(h1e, eri=eri, **kwargs)

    @classmethod
    def from_narg(
        cls,
        narg,
        coeff=None,
        *,
        dims=None,
        site="spatial",
        root=0,
        symmetry=None,
        preserve_support=None,
        support_tol=1e-12,
        hamiltonian=None,
        mpo=None,
        bond_dim=None,
        overlap=None,
        seed=None,
        mol=None,
        build_mpo=False,
        **kwargs,
    ):
        """Initialize qchem LETTA from a qchem NARG object.

        Parameters
        ----------
        narg
            NARG object exposing ``tensors``.  The last element of
            ``narg.tensors`` is the terminal root coefficient tensor with shape
            ``(4, D, nroots)`` for ``site="spatial"``.
        site
            Local qchem site convention.  ``"spatial"`` means one spatial
            orbital per site with local states ``|0>``, ``|alpha>``,
            ``|beta>``, and ``|alpha beta>``.
        """
        if coeff is None:
            tensors = _first_attr(narg, ("tensors", "narg_tensors", "factors"))
            if tensors is None:
                raise ValueError(
                    "qchem.LETTA.from_narg expects a NARG object with a 'tensors' "
                    "or 'narg_tensors' attribute. The last tensor must be the "
                    "terminal coefficient tensor C with shape (4, D, nroots) for "
                    "site='spatial'."
                )
            factors, coeff = _split_narg_tensors(tensors, site)
            if dims is None:
                dims = _dims_from_site(narg, factors, site)
        else:
            factors = [np.asarray(tensor) for tensor in narg]
            coeff = np.asarray(coeff)
            if not factors:
                raise ValueError("at least one NARG factor is required.")
            if dims is None:
                dims = (factors[0].shape[0],) + tuple(tensor.shape[2] for tensor in factors)

        if bond_dim is None:
            bond_dim = _first_attr(narg, ("bond_dim", "D"))
        if mol is None:
            mol = _first_attr(narg, ("mol",))
        if hamiltonian is None:
            hamiltonian = _first_attr(narg, ("hamiltonian", "H"))
        if mpo is None:
            mpo = _first_attr(narg, ("mpo",))
        if build_mpo and hamiltonian is None and mpo is None:
            mpo = _build_default_mpo(narg, dims, site)
        qn_info = _first_attr(narg, ("tensor_qns", "qn_info"))
        if preserve_support is None:
            preserve_support = _preserve_support_from_symmetry(symmetry)
        local_masks = kwargs.pop("local_masks", None)
        append_terminal = kwargs.pop("append_terminal", True)
        if preserve_support and local_masks is None and qn_info is not None:
            local_masks = _local_masks_from_qn_info(qn_info, factors, dims)

        engine = GenericLETTA.from_narg(
            factors,
            _coeff_matrix(coeff),
            dims=dims,
            root=root,
            hamiltonian=hamiltonian,
            bond_dim=bond_dim,
            overlap=overlap,
            seed=seed,
            preserve_support=preserve_support,
            support_tol=support_tol,
            local_masks=local_masks,
            append_terminal=append_terminal,
            **kwargs,
        )
        return cls(
            engine=engine,
            hamiltonian=hamiltonian,
            mpo=mpo,
            mol=mol,
            source=narg,
            e_tot=_first_attr(narg, ("e_tot",)),
            spin_info=_first_attr(narg, ("spin_info",)),
            qn_info=qn_info,
            site=str(site).lower(),
        )

    @property
    def dims(self):
        return self.engine.dims

    @property
    def tensors(self):
        return self.engine.tensors

    @property
    def history(self):
        return self.engine.history

    @property
    def energy(self):
        if self.engine.energy is None:
            return None
        return self.engine.energy + self._energy_shift()

    def _energy_shift(self):
        if self.mol is None or not hasattr(self.mol, "energy_nuc"):
            return 0.0
        return float(self.mol.energy_nuc())

    @property
    def converged(self):
        return self.engine.converged

    @property
    def ncompleted(self):
        return self.engine.ncompleted

    def operator(self, operator=None):
        """Return the supplied operator or the qchem default operator."""
        if operator is not None:
            return operator
        if self.mpo is not None:
            return self.mpo
        if self.hamiltonian is not None:
            return self.hamiltonian
        mpo = _build_default_mpo(self.source, self.dims, self.site)
        if mpo is not None:
            self.mpo = mpo
            return self.mpo
        n0 = _first_attr(self.source, ("n0",))
        if n0 is not None and int(n0) != 1:
            raise ValueError(
                "No Hamiltonian/MPO is available. This LETTA was initialized from "
                f"qchem NARG with n0={int(n0)}, so the first site is a compressed "
                "multi-orbital block rather than one spatial orbital. Re-run NARG "
                "with n0=1 for automatic qchem MPO construction, or pass an "
                "operator explicitly to letta.run(operator, ...)."
            )
        raise ValueError(
            "No Hamiltonian/MPO is available. Re-run NARG first so h1e/eri are "
            "stored on the NARG object, or pass an operator explicitly to "
            "letta.run(operator, ...)."
        )

    def expect(self, operator=None):
        value = self.engine.expect(self.operator(operator))
        if operator is None:
            value += self._energy_shift()
        return value

    def expectation(self, operator=None):
        return self.expect(operator)

    def run(self, operator=None, **kwargs):
        if self.source is not None:
            kwargs.setdefault("start_direction", "rl")
        self.engine.run(self.operator(operator), **kwargs)
        return self

    def state_vector(self):
        return self.engine.state_vector()

    def correlation(self, *args, **kwargs):
        return self.engine.correlation(*args, **kwargs)

    def copy(self):
        return type(self)(
            engine=self.engine.copy(),
            hamiltonian=self.hamiltonian,
            mpo=self.mpo,
            mol=self.mol,
            source=self.source,
            site=self.site,
            e_tot=self.e_tot,
            spin_info=self.spin_info,
            qn_info=self.qn_info,
        )


def _dims_from_site(narg, tensors, site):
    dims = _first_attr(narg, ("dims", "local_dims"))
    if dims is not None:
        dims = tuple(int(dim) for dim in dims)
        if len(dims) == len(tensors) + 1:
            return dims

    key = str(site).lower().replace("-", "_")
    if key in {"spatial", "spatial_orbital", "orbital"}:
        n0 = int(_first_attr(narg, ("n0",)) or 1)
        if n0 > 1:
            return (4**n0,) + (4,) * len(tensors)
        return (4,) * (len(tensors) + 1)
    raise ValueError(f"Unsupported qchem LETTA site={site!r}; only 'spatial' is implemented.")


def _build_default_mpo(narg, dims, site):
    key = str(site).lower().replace("-", "_")
    if key not in {"spatial", "spatial_orbital", "orbital"}:
        return None
    n0 = int(_first_attr(narg, ("n0",)) or 1)
    if n0 != 1 or any(int(dim) != 4 for dim in dims):
        return None
    h1e = _first_attr(narg, ("h1e",))
    eri = _first_attr(narg, ("eri",))
    if h1e is None or eri is None:
        return None

    from pyqed.qchem.dmrg.dmrg import _build_spatial_hamiltonian_tensor_mpo

    h1e = np.asarray(h1e)
    eri = np.asarray(eri)
    if h1e.ndim == 2:
        h1e = np.stack((h1e, h1e))
    if eri.ndim == 4:
        eri = np.stack(
            (
                np.stack((eri, eri)),
                np.stack((eri, eri)),
            )
        )
    symbolic_algo = _first_attr(narg, ("spatial_abelian_symbolic_algo",)) or "Hopcroft-Karp"
    cache_key = (_array_digest(h1e), _array_digest(eri), str(symbolic_algo))
    cached = _first_attr(narg, ("_letta_default_mpo",))
    cached_key = _first_attr(narg, ("_letta_default_mpo_cache_key",))
    if cached is not None and cached_key == cache_key:
        return cached
    cached = _DEFAULT_MPO_CACHE.get(cache_key)
    if cached is not None:
        _attach_default_mpo_cache(narg, cache_key, cached)
        return cached

    tensor_mpo, _term_count, _spin_term_count = _build_spatial_hamiltonian_tensor_mpo(
        h1e,
        eri,
        symbolic_algo=symbolic_algo,
    )
    factors = [np.asarray(factor) for factor in tensor_mpo.factors]
    _store_default_mpo_cache(cache_key, factors)
    _attach_default_mpo_cache(narg, cache_key, factors)
    return factors


def _array_digest(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(array.shape).encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _store_default_mpo_cache(cache_key, factors):
    if cache_key in _DEFAULT_MPO_CACHE:
        _DEFAULT_MPO_CACHE.pop(cache_key)
    elif len(_DEFAULT_MPO_CACHE) >= _DEFAULT_MPO_CACHE_MAXSIZE:
        _DEFAULT_MPO_CACHE.pop(next(iter(_DEFAULT_MPO_CACHE)))
    _DEFAULT_MPO_CACHE[cache_key] = factors


def _attach_default_mpo_cache(narg, cache_key, factors):
    try:
        setattr(narg, "_letta_default_mpo_cache_key", cache_key)
        setattr(narg, "_letta_default_mpo", factors)
    except Exception:
        pass


def _split_narg_tensors(tensors, site):
    tensors = list(tensors)
    if len(tensors) < 2:
        raise ValueError("narg.tensors must contain NARG factors followed by terminal coefficient tensor C.")
    coeff = np.asarray(tensors[-1])
    key = str(site).lower().replace("-", "_")
    if key in {"spatial", "spatial_orbital", "orbital"} and coeff.ndim != 3:
        raise ValueError("For site='spatial', narg.tensors[-1] must have shape (4, D, nroots).")
    if key in {"spatial", "spatial_orbital", "orbital"} and coeff.shape[0] != 4:
        raise ValueError("For site='spatial', narg.tensors[-1].shape[0] must be 4.")
    return tensors[:-1], coeff


def _coeff_matrix(coeff):
    coeff = np.asarray(coeff)
    if coeff.ndim == 3:
        return coeff.reshape(coeff.shape[0] * coeff.shape[1], coeff.shape[2])
    return coeff


def _local_masks_from_qn_info(qn_info, factors, dims):
    factors = [np.asarray(factor) for factor in factors]
    dims = tuple(int(dim) for dim in dims)
    factor_qns = qn_info.get("factors") if isinstance(qn_info, dict) else None
    if factor_qns is None or len(factor_qns) != len(factors):
        return None

    masks = []
    for i, (factor, info) in enumerate(zip(factors, factor_qns)):
        left_dim = 1 if i == 0 else factors[i - 1].shape[1]
        right_dim = factor.shape[1]
        row_qn = np.asarray(info["row_qn"], dtype=int)
        right_qn = np.asarray(info["right_qn_by_next"], dtype=int)
        if row_qn.shape[0] != dims[i] * left_dim:
            return None
        if right_qn.shape[:2] != (dims[i + 1], right_dim):
            return None

        factor_mask = np.zeros((left_dim, dims[i], dims[i + 1], right_dim), dtype=bool)
        for si in range(dims[i]):
            for left in range(left_dim):
                row = si * left_dim + left
                qn = row_qn[row]
                for sj in range(dims[i + 1]):
                    factor_mask[left, si, sj, :] = np.all(right_qn[sj] == qn, axis=1)
        masks.append(factor_mask)

    terminal_qn = qn_info.get("terminal_total_qn_by_site") if isinstance(qn_info, dict) else None
    target_qn = qn_info.get("target_qn") if isinstance(qn_info, dict) else None
    if terminal_qn is not None and target_qn is not None:
        terminal_qn = np.asarray(terminal_qn, dtype=int)
        target_qn = np.asarray(target_qn, dtype=int)
        if terminal_qn.shape[:2] == (dims[-1], factors[-1].shape[1]):
            masks.append(np.all(terminal_qn == target_qn, axis=2))
            return masks
    masks.append(None)
    return masks


def _preserve_support_from_symmetry(symmetry):
    if symmetry is None or symmetry is False:
        return False
    key = str(symmetry).lower().replace("-", "_")
    return key in {"abelian", "u1", "u1xu1", "u1_u1", "support", "support_mask"}
