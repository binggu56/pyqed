#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""XLETTA: LETTA with variational lifted shared legs.

``XLETTA`` generalizes finite-chain LETTA by replacing each hard shared
physical leg with a variational tensor ``W[u, v, s]``.  Setting every ``W`` to
the copy tensor ``delta[u, s] delta[v, s]`` recovers ordinary terminal LETTA.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import issparse

from .abelian import Layout, XLayout
from .core import (
    LETTA,
    _hermitian_sqrt_pair,
    _lowest_hermitian_eigenpair,
    _metric_basis,
    _normalize_with_metric,
    _validate_dims,
)


def _whitened_lowest_generalized_eigenpair(hamiltonian, metric, *, metric_tol=1.0e-12):
    """Solve ``H x = E N x`` after explicitly whitening the local metric."""
    hamiltonian = 0.5 * (hamiltonian + hamiltonian.conj().T)
    metric = 0.5 * (metric + metric.conj().T)
    basis = _metric_basis(metric, metric_tol=metric_tol)
    reduced_h = basis.conj().T @ hamiltonian @ basis
    energy, reduced_vector = _lowest_hermitian_eigenpair(reduced_h)
    vector = basis @ reduced_vector
    vector = _normalize_with_metric(vector, metric)
    return energy, vector, {
        "raw_dim": int(metric.shape[0]),
        "whitened_dim": int(basis.shape[1]),
    }


def _as_mpo_factors(operator):
    if hasattr(operator, "factors"):
        return list(operator.factors)
    if isinstance(operator, (list, tuple)):
        return list(operator)
    return None


def _looks_like_mpo(operator) -> bool:
    factors = _as_mpo_factors(operator)
    return factors is not None and bool(factors) and all(np.asarray(site).ndim == 4 for site in factors)


def _validate_mpo(operator, dims: tuple[int, ...]) -> list[np.ndarray]:
    factors = _as_mpo_factors(operator)
    if factors is None:
        raise ValueError("MPO must be a list/tuple of site tensors or expose a factors attribute.")
    if len(factors) != len(dims):
        raise ValueError("MPO length must match the number of physical sites.")
    mpo = [np.asarray(site) for site in factors]
    for i, site in enumerate(mpo):
        if site.ndim != 4:
            raise ValueError("each MPO tensor must have shape (left, right, bra, ket).")
        if site.shape[2] != dims[i] or site.shape[3] != dims[i]:
            raise ValueError(f"MPO tensor {i} physical dimensions do not match dims[{i}].")
        if i == 0 and site.shape[0] != 1:
            raise ValueError("first MPO tensor must have left bond dimension 1.")
        if i == len(dims) - 1 and site.shape[1] != 1:
            raise ValueError("last MPO tensor must have right bond dimension 1.")
        if i and mpo[i - 1].shape[1] != site.shape[0]:
            raise ValueError(f"MPO bond mismatch between tensors {i - 1} and {i}.")
    return mpo


def _mpo_to_dense_operator(operator, dims: tuple[int, ...]) -> np.ndarray:
    mpo = _validate_mpo(operator, dims)
    cores = [site.transpose(0, 2, 3, 1) for site in mpo]
    tensor = cores[0]
    for core in cores[1:]:
        tensor = np.tensordot(tensor, core, axes=([-1], [0]))
    tensor = np.squeeze(tensor, axis=(0, -1))
    nsites = len(dims)
    permutation = list(range(0, 2 * nsites, 2)) + list(range(1, 2 * nsites, 2))
    tensor = np.transpose(tensor, axes=permutation)
    dim = int(np.prod(dims))
    return tensor.reshape(dim, dim)


def _mpo_site_sparse_entries(mpo_site, *, drop_tol: float = 0.0):
    mpo_site = np.asarray(mpo_site)
    nz = np.argwhere(np.abs(mpo_site) > float(drop_tol))
    if nz.size == 0:
        return nz.reshape(0, 4), np.asarray([], dtype=mpo_site.dtype)
    values = mpo_site[nz[:, 0], nz[:, 1], nz[:, 2], nz[:, 3]]
    return nz.astype(np.int64, copy=False), values


def _as_operator(operator, dims: tuple[int, ...] | None = None):
    if operator is None:
        return None
    if _looks_like_mpo(operator):
        if dims is None:
            raise ValueError("dims are required to validate an MPO operator.")
        return _mpo_to_dense_operator(operator, dims)
    if issparse(operator):
        if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
            raise ValueError("operator must be a square matrix.")
        return operator
    matrix = np.asarray(operator)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("operator must be a square matrix.")
    return matrix


def _validate_view_dims(view_dim, dims: tuple[int, ...]) -> tuple[int, ...]:
    if len(dims) < 2:
        raise ValueError("XLETTA needs at least two physical sites.")
    if view_dim is None:
        return tuple(dims[1:])
    if np.isscalar(view_dim):
        q = int(view_dim)
        if q < 1:
            raise ValueError("view_dim must be positive.")
        view_dims = (q,) * (len(dims) - 1)
    else:
        view_dims = tuple(int(q) for q in view_dim)
        if len(view_dims) != len(dims) - 1:
            raise ValueError("view_dim must have one entry for each shared site.")
    for site, (q, d) in enumerate(zip(view_dims, dims[1:]), start=1):
        if q < d:
            raise ValueError(f"view_dim for site {site} must be at least the local dimension {d}.")
    return view_dims


def _as_xlayout(layout, *, view_dim=None) -> XLayout | None:
    if layout is None:
        return None
    if isinstance(layout, XLayout):
        if view_dim is not None and _validate_view_dims(view_dim, layout.dims) != layout.view_dims:
            raise ValueError("view_dim does not match the Abelian XLETTA layout view dimensions.")
        return layout
    if isinstance(layout, Layout):
        return XLayout.from_letta_layout(layout, view_dim=view_dim)
    raise TypeError("abelian_layout must be a pyqed.letta.XLayout or pyqed.letta.Layout instance.")


class XLETTA:
    r"""
    Dense-reference LETTA with variational lifted shared legs.

    The represented wavefunction is

    .. math::

        \Psi(s_0,\ldots,s_{L-1}) =
        T_0(s_0, a_1, u_1)
        \prod_{i=1}^{L-2} T_i(a_i, v_i, u_{i+1}, a_{i+1})
        T_{L-1}(a_{L-1}, v_{L-1})
        \prod_{i=1}^{L-1} W_i(u_i, v_i, s_i),

    with all repeated indices summed.  The copy choice
    ``W_i[u, v, s] = delta[u, s] delta[v, s]`` embeds terminal LETTA exactly.

    Dense operators use explicit product-basis projectors.  MPO operators use
    environment contractions on the effective MPS induced by the XLETTA tensors
    and project the one-site MPS local problem back onto either a ``T`` tensor
    or a ``W`` tensor.
    """

    def __init__(
        self,
        hamiltonian,
        dims,
        *,
        bond_dim=4,
        view_dim=None,
        overlap=None,
        tensors=None,
        w_tensors=None,
        tensor_masks=None,
        w_masks=None,
        abelian_layout=None,
        seed=None,
    ):
        self.dims = _validate_dims(dims)
        if len(self.dims) < 2:
            raise ValueError("XLETTA needs at least two physical sites.")
        self.full_dim = int(np.prod(self.dims))

        self.hamiltonian = None
        self.operator_mpo = None
        if _looks_like_mpo(hamiltonian):
            self.operator_mpo = _validate_mpo(hamiltonian, self.dims)
        else:
            self.hamiltonian = _as_operator(hamiltonian, self.dims)
        if self.hamiltonian is not None and self.hamiltonian.shape != (self.full_dim, self.full_dim):
            raise ValueError(
                f"hamiltonian shape {self.hamiltonian.shape} does not match product dimension {self.full_dim}."
            )
        self.overlap = _as_operator(overlap, self.dims)
        if self.overlap is not None and self.overlap.shape != (self.full_dim, self.full_dim):
            raise ValueError("overlap shape must match product dimension.")

        self.abelian_layout = _as_xlayout(abelian_layout, view_dim=view_dim)
        if self.abelian_layout is not None:
            if self.abelian_layout.dims != self.dims:
                raise ValueError("Abelian XLETTA layout dimensions do not match dims.")
            if view_dim is None:
                view_dim = self.abelian_layout.view_dims

        self.bond_dim = int(bond_dim)
        if self.bond_dim < 1:
            raise ValueError("bond_dim must be positive.")
        self.view_dims = _validate_view_dims(view_dim, self.dims)
        self.view_dim = self.view_dims[0] if len(set(self.view_dims)) == 1 else self.view_dims

        self.rng = np.random.default_rng(seed)
        if tensors is None and self.abelian_layout is not None:
            self.tensors = self._random_tensors_from_shapes(self.abelian_layout.tensor_shapes())
        else:
            self.tensors = self._random_tensors() if tensors is None else self._validate_tensors(tensors)
        self.w_tensors = self._copy_w_tensors() if w_tensors is None else self._validate_w_tensors(w_tensors)
        if self.abelian_layout is not None:
            layout_tensor_masks, layout_w_masks = self.abelian_layout.masks()
            if tensor_masks is None:
                tensor_masks = layout_tensor_masks
            if w_masks is None:
                w_masks = layout_w_masks
        self.tensor_masks = self._validate_variable_masks(tensor_masks, self.tensors, "tensor_masks")
        self.w_masks = self._validate_variable_masks(w_masks, self.w_tensors, "w_masks")
        self._apply_variable_masks()
        self.history = []
        self.converged = False
        self.last_energy = None
        self.normalize()

    @property
    def nsites(self) -> int:
        return len(self.dims)

    @property
    def nshared(self) -> int:
        return self.nsites - 1

    @property
    def ncompleted(self) -> int:
        return len(self.history)

    @staticmethod
    def copy_w(local_dim: int, view_dim: int) -> np.ndarray:
        """Return ``W[u, v, s] = delta[u, s] delta[v, s]``."""
        local_dim = int(local_dim)
        view_dim = int(view_dim)
        if view_dim < local_dim:
            raise ValueError("view_dim must be at least local_dim.")
        w = np.zeros((view_dim, view_dim, local_dim), dtype=float)
        for s in range(local_dim):
            w[s, s, s] = 1.0
        return w

    @classmethod
    def from_standard_tensors(
        cls,
        hamiltonian,
        dims,
        tensors,
        *,
        view_dim=None,
        overlap=None,
        seed=None,
    ) -> "XLETTA":
        """Embed terminal LETTA tensors with copy ``W`` tensors.

        ``tensors`` use the terminal LETTA convention
        ``T0[s0, a1, s1]``, ``Ti[ai, si, s{i+1}, a{i+1}]``, and
        ``Tlast[a_last, s_last]``.
        """
        dims = _validate_dims(dims)
        tensors = [np.asarray(tensor) for tensor in tensors]
        if len(tensors) != len(dims):
            raise ValueError("standard LETTA tensors must have one tensor per physical site.")
        view_dims = _validate_view_dims(view_dim, dims)

        first = tensors[0]
        last = tensors[-1]
        if first.ndim != 3 or first.shape[0] != dims[0] or first.shape[2] != dims[1]:
            raise ValueError(f"first tensor must have shape ({dims[0]}, bond, {dims[1]}).")
        if last.ndim != 2 or last.shape[1] != dims[-1]:
            raise ValueError(f"last tensor must have shape (bond, {dims[-1]}).")

        converted = []
        dtype = np.result_type(*[tensor.dtype for tensor in tensors])
        out0 = np.zeros((dims[0], first.shape[1], view_dims[0]), dtype=dtype)
        out0[:, :, : dims[1]] = first
        converted.append(out0)

        previous_bond = first.shape[1]
        for site in range(1, len(dims) - 1):
            tensor = tensors[site]
            if tensor.ndim != 4 or tensor.shape[:3] != (previous_bond, dims[site], dims[site + 1]):
                raise ValueError(
                    f"tensor {site} must have shape "
                    f"({previous_bond}, {dims[site]}, {dims[site + 1]}, bond)."
                )
            out = np.zeros((previous_bond, view_dims[site - 1], view_dims[site], tensor.shape[3]), dtype=dtype)
            out[:, : dims[site], : dims[site + 1], :] = tensor
            converted.append(out)
            previous_bond = tensor.shape[3]

        if last.shape[0] != previous_bond:
            raise ValueError("last tensor bond dimension does not match the preceding tensor.")
        out_last = np.zeros((previous_bond, view_dims[-1]), dtype=dtype)
        out_last[:, : dims[-1]] = last
        converted.append(out_last)

        bond_sizes = [converted[0].shape[1], converted[-1].shape[0]]
        bond_sizes.extend(tensor.shape[3] for tensor in converted[1:-1])
        bond_dim = max(bond_sizes)
        return cls(
            hamiltonian,
            dims,
            bond_dim=bond_dim,
            view_dim=view_dims,
            overlap=overlap,
            tensors=converted,
            seed=seed,
        )

    @classmethod
    def from_letta(
        cls,
        letta: LETTA,
        *,
        hamiltonian=None,
        view_dim=None,
        overlap=None,
        seed=None,
    ) -> "XLETTA":
        """Embed a :class:`LETTA` instance with copy ``W`` tensors."""
        if not isinstance(letta, LETTA):
            raise TypeError("from_letta expects a pyqed.letta.LETTA instance.")
        dims = letta.dims
        pair_tensors = letta.tensors[: letta.npairs]
        standard = []
        first = pair_tensors[0]
        standard.append(np.transpose(first[0], (0, 2, 1)).copy())
        for tensor in pair_tensors[1:]:
            standard.append(tensor.copy())
        if letta.has_terminal_tensor:
            standard.append(letta.tensors[-1].T.copy())
        else:
            standard.append(np.ones((pair_tensors[-1].shape[3], dims[-1]), dtype=pair_tensors[-1].dtype))
        if hamiltonian is None:
            hamiltonian = letta.hamiltonian
        if overlap is None:
            overlap = letta.overlap
        return cls.from_standard_tensors(
            hamiltonian,
            dims,
            standard,
            view_dim=view_dim,
            overlap=overlap,
            seed=seed,
        )

    @classmethod
    def from_mps(
        cls,
        mps,
        *,
        hamiltonian=None,
        dims=None,
        view_dim=None,
        overlap=None,
        abelian_layout=None,
        seed=None,
    ) -> "XLETTA":
        """Embed an open-boundary MPS into XLETTA with copy ``W`` tensors."""
        factors = mps.factors if hasattr(mps, "factors") else mps
        factors = [np.asarray(factor) for factor in factors]
        if len(factors) < 2:
            raise ValueError("at least two MPS tensors are required.")
        if dims is None:
            dims = tuple(int(factor.shape[1]) for factor in factors)
        dims = _validate_dims(dims)
        if len(dims) != len(factors):
            raise ValueError("dims must have one entry per MPS tensor.")
        for site, (factor, dim) in enumerate(zip(factors, dims)):
            if factor.ndim != 3 or factor.shape[1] != dim:
                raise ValueError(f"MPS tensor {site} must have shape (left, {dim}, right).")
            if site == 0 and factor.shape[0] != 1:
                raise ValueError("first MPS tensor must have left bond dimension 1.")
            if site == len(factors) - 1 and factor.shape[2] != 1:
                raise ValueError("last MPS tensor must have right bond dimension 1.")
            if site and factors[site - 1].shape[2] != factor.shape[0]:
                raise ValueError(f"MPS bond mismatch between tensors {site - 1} and {site}.")

        xlayout = _as_xlayout(abelian_layout, view_dim=view_dim)
        view_dims = xlayout.view_dims if xlayout is not None else _validate_view_dims(view_dim, dims)
        dtype = np.result_type(*[factor.dtype for factor in factors])

        tensors = []
        first = np.zeros((dims[0], factors[0].shape[2], view_dims[0]), dtype=dtype)
        first[:, :, : dims[1]] = factors[0][0, :, :, None]
        tensors.append(first)

        for site in range(1, len(dims) - 1):
            factor = factors[site]
            tensor = np.zeros((factor.shape[0], view_dims[site - 1], view_dims[site], factor.shape[2]), dtype=dtype)
            tensor[:, : dims[site], : dims[site + 1], :] = factor[:, :, None, :]
            tensors.append(tensor)

        last = np.zeros((factors[-1].shape[0], view_dims[-1]), dtype=dtype)
        last[:, : dims[-1]] = factors[-1][:, :, 0]
        tensors.append(last)

        bond_dim = max(max(tensor.shape[1], tensor.shape[-1]) for tensor in tensors)
        return cls(
            hamiltonian,
            dims,
            bond_dim=bond_dim,
            view_dim=view_dims,
            overlap=overlap,
            tensors=tensors,
            abelian_layout=xlayout,
            seed=seed,
        )

    def _random_tensors(self) -> list[np.ndarray]:
        bonds = [1] + [self.bond_dim] * (self.nsites - 1)
        tensors = []
        first_shape = (self.dims[0], bonds[1], self.view_dims[0])
        first = self.rng.normal(size=first_shape) / np.sqrt(np.prod(first_shape))
        tensors.append(first.astype(float))
        for site in range(1, self.nsites - 1):
            shape = (bonds[site], self.view_dims[site - 1], self.view_dims[site], bonds[site + 1])
            tensor = self.rng.normal(size=shape) / np.sqrt(np.prod(shape))
            tensors.append(tensor.astype(float))
        last_shape = (bonds[-1], self.view_dims[-1])
        last = self.rng.normal(size=last_shape) / np.sqrt(np.prod(last_shape))
        tensors.append(last.astype(float))
        return tensors

    def _random_tensors_from_shapes(self, shapes) -> list[np.ndarray]:
        tensors = []
        for shape in shapes:
            shape = tuple(int(dim) for dim in shape)
            scale = np.sqrt(max(1, int(np.prod(shape))))
            tensors.append((self.rng.normal(size=shape) / scale).astype(float))
        return self._validate_tensors(tensors)

    def _validate_tensors(self, tensors) -> list[np.ndarray]:
        tensors = [np.asarray(tensor, dtype=complex if np.iscomplexobj(tensor) else float) for tensor in tensors]
        if len(tensors) != self.nsites:
            raise ValueError("XLETTA needs one tensor per physical site.")
        first = tensors[0]
        if first.ndim != 3 or first.shape[0] != self.dims[0] or first.shape[2] != self.view_dims[0]:
            raise ValueError(f"first tensor must have shape ({self.dims[0]}, bond, {self.view_dims[0]}).")
        previous_bond = first.shape[1]
        for site in range(1, self.nsites - 1):
            expected_prefix = (previous_bond, self.view_dims[site - 1], self.view_dims[site])
            if tensors[site].ndim != 4 or tensors[site].shape[:3] != expected_prefix:
                raise ValueError(f"tensor {site} has incompatible shape {tensors[site].shape}.")
            previous_bond = tensors[site].shape[3]
        last = tensors[-1]
        if last.ndim != 2 or last.shape != (previous_bond, self.view_dims[-1]):
            raise ValueError(f"last tensor must have shape ({previous_bond}, {self.view_dims[-1]}).")
        return [tensor.copy() for tensor in tensors]

    def _copy_w_tensors(self) -> list[np.ndarray]:
        return [self.copy_w(d, q) for d, q in zip(self.dims[1:], self.view_dims)]

    def _validate_w_tensors(self, w_tensors) -> list[np.ndarray]:
        w_tensors = [np.asarray(w, dtype=complex if np.iscomplexobj(w) else float) for w in w_tensors]
        if len(w_tensors) != self.nshared:
            raise ValueError("w_tensors must have one tensor for each shared physical site.")
        for site, (w, q, d) in enumerate(zip(w_tensors, self.view_dims, self.dims[1:]), start=1):
            if w.ndim != 3 or w.shape != (q, q, d):
                raise ValueError(f"W tensor for site {site} must have shape ({q}, {q}, {d}).")
        return [w.copy() for w in w_tensors]

    @staticmethod
    def _validate_variable_masks(masks, variables, name: str):
        if masks is None:
            return [None] * len(variables)
        masks = list(masks)
        if len(masks) != len(variables):
            raise ValueError(f"{name} must have one entry for each corresponding XLETTA variable.")
        validated = []
        for index, (mask, variable) in enumerate(zip(masks, variables)):
            if mask is None:
                validated.append(None)
                continue
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != variable.shape:
                raise ValueError(f"{name}[{index}] shape {mask.shape} does not match {variable.shape}.")
            validated.append(mask.copy())
        return validated

    def _apply_variable_masks(self) -> None:
        for index, mask in enumerate(self.tensor_masks):
            if mask is not None:
                self.tensors[index] = np.where(mask, self.tensors[index], 0.0)
        for index, mask in enumerate(self.w_masks):
            if mask is not None:
                self.w_tensors[index] = np.where(mask, self.w_tensors[index], 0.0)

    def _variable_mask(self, kind: str, index: int):
        if kind == "tensor":
            return self.tensor_masks[int(index)]
        if kind == "w":
            return self.w_masks[int(index)]
        raise ValueError("kind must be 'tensor' or 'w'.")

    def _variable_support(self, kind: str, index: int) -> np.ndarray:
        variable = self._variable(kind, index)
        mask = self._variable_mask(kind, index)
        if mask is None:
            return np.arange(variable.size, dtype=np.int64)
        support = np.flatnonzero(mask.reshape(-1))
        if support.size == 0:
            raise ValueError(f"{kind} variable {index} has empty variational support.")
        return support

    def _variable(self, kind: str, index: int) -> np.ndarray:
        if kind == "tensor":
            return self.tensors[int(index)]
        if kind == "w":
            return self.w_tensors[int(index)]
        raise ValueError("kind must be 'tensor' or 'w'.")

    def _set_variable(self, kind: str, index: int, value: np.ndarray) -> None:
        if kind == "tensor":
            index = int(index)
            value = np.asarray(value).reshape(self.tensors[index].shape)
            mask = self._variable_mask(kind, index) if hasattr(self, "tensor_masks") else None
            self.tensors[index] = value if mask is None else np.where(mask, value, 0.0)
            return
        if kind == "w":
            index = int(index)
            value = np.asarray(value).reshape(self.w_tensors[index].shape)
            mask = self._variable_mask(kind, index) if hasattr(self, "w_masks") else None
            self.w_tensors[index] = value if mask is None else np.where(mask, value, 0.0)
            return
        raise ValueError("kind must be 'tensor' or 'w'.")

    def _with_override(self, override):
        tensors = self.tensors
        w_tensors = self.w_tensors
        if override is None:
            return tensors, w_tensors
        kind, index, value = override
        index = int(index)
        if kind == "tensor":
            tensors = list(tensors)
            tensors[index] = value
        elif kind == "w":
            w_tensors = list(w_tensors)
            w_tensors[index] = value
        else:
            raise ValueError("override kind must be 'tensor' or 'w'.")
        return tensors, w_tensors

    def _amplitude(self, config, override=None):
        tensors, w_tensors = self._with_override(override)
        current_u = tensors[0][config[0]]
        for site in range(1, self.nsites):
            current_v = np.einsum("bu,uv->bv", current_u, w_tensors[site - 1][:, :, config[site]], optimize=True)
            if site == self.nsites - 1:
                return np.einsum("bv,bv->", current_v, tensors[-1], optimize=True)
            current_u = np.einsum("bv,bvuc->cu", current_v, tensors[site], optimize=True)
        raise RuntimeError("unreachable")

    def _effective_mps_core(self, site: int) -> np.ndarray:
        """Return one ordinary MPS core induced by the current XLETTA tensors."""
        site = int(site)
        if site == 0:
            first = self.tensors[0]
            return first.reshape(1, self.dims[0], first.shape[1] * first.shape[2])
        if site == self.nsites - 1:
            last = self.tensors[-1]
            w = self.w_tensors[-1]
            core = np.einsum("uvs,av->aus", w, last, optimize=True)
            return core.reshape(last.shape[0] * w.shape[0], self.dims[-1], 1)
        tensor = self.tensors[site]
        w = self.w_tensors[site - 1]
        core = np.einsum("uvs,avxb->ausbx", w, tensor, optimize=True)
        return core.reshape(tensor.shape[0] * w.shape[0], self.dims[site], tensor.shape[3] * tensor.shape[2])

    def effective_mps_cores(self) -> list[np.ndarray]:
        """Return the ordinary MPS cores induced by the current XLETTA tensors."""
        return [self._effective_mps_core(site) for site in range(self.nsites)]

    def state_vector(self, override=None) -> np.ndarray:
        """Return the dense product-basis state vector."""
        dtype = np.result_type(
            *[tensor.dtype for tensor in self.tensors],
            *[w.dtype for w in self.w_tensors],
        )
        psi = np.empty(self.full_dim, dtype=dtype)
        for flat, config in enumerate(np.ndindex(*self.dims)):
            psi[flat] = self._amplitude(config, override=override)
        return psi

    def state_sector(self) -> np.ndarray:
        """Compatibility alias for dense benchmark helpers."""
        return self.state_vector()

    def norm(self) -> float:
        if self.overlap is not None:
            psi = self.state_vector()
            return float(np.real(np.vdot(psi, self.overlap @ psi)))
        cores = self.effective_mps_cores()
        env = np.ones((1, 1), dtype=np.result_type(*[core.dtype for core in cores]))
        for core in cores:
            env = np.einsum("ab,asc,bsd->cd", env, core.conj(), core, optimize=True)
        norm2 = float(np.real(env[0, 0]))
        return 0.0 if -1.0e-12 < norm2 < 0.0 else norm2

    def normalize(self) -> "XLETTA":
        norm = np.sqrt(self.norm())
        if norm < 1.0e-14:
            raise ValueError("Cannot normalize a numerically zero XLETTA state.")
        self.tensors[0] = self.tensors[0] / norm
        return self

    def expectation(self, operator=None) -> float:
        """Return the normalized expectation value of a dense, sparse, or MPO operator."""
        if operator is None:
            if self.operator_mpo is not None:
                return self.expectation_mpo(self.operator_mpo)
            operator = self.hamiltonian
        if _looks_like_mpo(operator):
            return self.expectation_mpo(operator)
        operator = _as_operator(operator, self.dims)
        if operator is None:
            raise ValueError("operator is not available.")
        if operator.shape != (self.full_dim, self.full_dim):
            raise ValueError("operator shape does not match product dimension.")
        psi = self.state_vector()
        denom = np.vdot(psi, psi) if self.overlap is None else np.vdot(psi, self.overlap @ psi)
        if abs(denom) < 1.0e-14:
            raise ValueError("State norm is numerically zero.")
        return float(np.real(np.vdot(psi, operator @ psi) / denom))

    def expect(self, operator=None) -> float:
        return self.expectation(operator)

    def energy(self) -> float:
        """Compatibility alias for ``expectation()``."""
        return self.expectation()

    @staticmethod
    def _advance_left_metric_environment(left, core):
        return np.einsum("ab,asc,bsd->cd", left, core.conj(), core, optimize=True)

    @staticmethod
    def _advance_right_metric_environment(right, core):
        return np.einsum("cd,asc,bsd->ab", right, core.conj(), core, optimize=True)

    @staticmethod
    def _advance_left_mpo_environment(left, mpo_site, core):
        entries, values = _mpo_site_sparse_entries(mpo_site)
        if entries.size and entries.shape[0] < np.asarray(mpo_site).size // 2:
            out = np.zeros(
                (core.shape[2], core.shape[2], mpo_site.shape[1]),
                dtype=np.result_type(left.dtype, core.dtype, mpo_site.dtype),
            )
            for (m, n, s, t), value in zip(entries, values):
                out[:, :, n] += value * (core[:, s, :].conj().T @ left[:, :, m] @ core[:, t, :])
            return out
        return np.einsum("abm,asc,mnst,btd->cdn", left, core.conj(), mpo_site, core, optimize=True)

    @staticmethod
    def _advance_right_mpo_environment(right, mpo_site, core):
        entries, values = _mpo_site_sparse_entries(mpo_site)
        if entries.size and entries.shape[0] < np.asarray(mpo_site).size // 2:
            out = np.zeros(
                (core.shape[0], core.shape[0], mpo_site.shape[0]),
                dtype=np.result_type(right.dtype, core.dtype, mpo_site.dtype),
            )
            for (m, n, s, t), value in zip(entries, values):
                out[:, :, m] += value * (core[:, s, :].conj() @ right[:, :, n] @ core[:, t, :].T)
            return out
        return np.einsum("cdn,asc,mnst,btd->abm", right, core.conj(), mpo_site, core, optimize=True)

    def _left_metric_environments(self, cores: list[np.ndarray] | None = None) -> list[np.ndarray]:
        cores = self.effective_mps_cores() if cores is None else cores
        envs = [None] * (self.nsites + 1)
        envs[0] = np.ones((1, 1), dtype=np.result_type(*[core.dtype for core in cores]))
        for site, core in enumerate(cores):
            envs[site + 1] = self._advance_left_metric_environment(envs[site], core)
        return envs

    def _right_metric_environments(self, cores: list[np.ndarray] | None = None) -> list[np.ndarray]:
        cores = self.effective_mps_cores() if cores is None else cores
        envs = [None] * (self.nsites + 1)
        envs[-1] = np.ones((1, 1), dtype=np.result_type(*[core.dtype for core in cores]))
        for site in reversed(range(self.nsites)):
            core = cores[site]
            envs[site] = self._advance_right_metric_environment(envs[site + 1], core)
        return envs

    def _left_mpo_environments(self, mpo, cores: list[np.ndarray] | None = None) -> list[np.ndarray]:
        mpo = _validate_mpo(mpo, self.dims)
        cores = self.effective_mps_cores() if cores is None else cores
        dtype = np.result_type(*[core.dtype for core in cores], *[site.dtype for site in mpo])
        envs = [None] * (self.nsites + 1)
        envs[0] = np.ones((1, 1, 1), dtype=dtype)
        for site, core in enumerate(cores):
            envs[site + 1] = self._advance_left_mpo_environment(envs[site], mpo[site], core)
        return envs

    def _right_mpo_environments(self, mpo, cores: list[np.ndarray] | None = None) -> list[np.ndarray]:
        mpo = _validate_mpo(mpo, self.dims)
        cores = self.effective_mps_cores() if cores is None else cores
        dtype = np.result_type(*[core.dtype for core in cores], *[site.dtype for site in mpo])
        envs = [None] * (self.nsites + 1)
        envs[-1] = np.ones((1, 1, 1), dtype=dtype)
        for site in reversed(range(self.nsites)):
            core = cores[site]
            envs[site] = self._advance_right_mpo_environment(envs[site + 1], mpo[site], core)
        return envs

    def _mpo_matrix_element(self, mpo) -> complex:
        mpo = _validate_mpo(mpo, self.dims)
        cores = self.effective_mps_cores()
        return self._left_mpo_environments(mpo, cores)[-1][0, 0, 0]

    def expectation_mpo(self, mpo) -> float:
        denom = self.norm()
        if abs(denom) < 1.0e-14:
            raise ValueError("State norm is numerically zero.")
        return float(np.real(self._mpo_matrix_element(mpo) / denom))

    def copy_deviation(self) -> tuple[float, ...]:
        """Return ``||W_i - W_copy||`` for each shared site."""
        return tuple(
            float(np.linalg.norm(w - self.copy_w(d, q)))
            for w, d, q in zip(self.w_tensors, self.dims[1:], self.view_dims)
        )

    def physical_isometry_error(self) -> tuple[float, ...]:
        """Return ``||W_i^dagger W_i - I||`` after flattening ``(u, v)``."""
        errors = []
        for w, d in zip(self.w_tensors, self.dims[1:]):
            matrix = w.reshape(w.shape[0] * w.shape[1], d)
            gram = matrix.conj().T @ matrix
            errors.append(float(np.linalg.norm(gram - np.eye(d))))
        return tuple(errors)

    @staticmethod
    def _axis_transform(array, axis: int, matrix: np.ndarray) -> np.ndarray:
        """Apply ``array[..., old, ...] @ matrix[old, new]`` on one axis."""
        axis = int(axis)
        matrix = np.asarray(matrix)
        moved = np.moveaxis(array, axis, -1)
        transformed = np.tensordot(moved, matrix, axes=([-1], [0]))
        return np.moveaxis(transformed, -1, axis)

    @classmethod
    def _axis_transform_subset(cls, array, axis: int, group, matrix: np.ndarray) -> np.ndarray:
        """Apply an axis transform inside one index group, leaving other entries unchanged."""
        axis = int(axis)
        group = np.asarray(group, dtype=np.int64)
        out = np.array(array, copy=True)
        selector = [slice(None)] * out.ndim
        selector[axis] = group
        subset = out[tuple(selector)]
        subset = cls._axis_transform(subset, axis, matrix)
        out[tuple(selector)] = subset
        return out

    @staticmethod
    def _groups_from_labels(labels) -> tuple[np.ndarray, ...]:
        groups = {}
        for index, label in enumerate(labels):
            groups.setdefault(tuple(label), []).append(index)
        return tuple(np.asarray(indices, dtype=np.int64) for indices in groups.values() if indices)

    def _prefix_bond_groups(self, bond: int) -> tuple[np.ndarray, ...]:
        bond = int(bond)
        dim = self.tensors[bond].shape[0]
        if self.abelian_layout is not None and hasattr(self.abelian_layout, "prefix_qns"):
            labels = self.abelian_layout.prefix_qns[bond][:dim]
            return self._groups_from_labels(labels)
        return (np.arange(dim, dtype=np.int64),)

    def _view_leg_groups(self, shared_index: int) -> tuple[np.ndarray, ...]:
        shared_index = int(shared_index)
        dim = self.view_dims[shared_index]
        if self.abelian_layout is not None and hasattr(self.abelian_layout, "view_qns"):
            labels = self.abelian_layout.view_qns[shared_index][:dim]
            return self._groups_from_labels(labels)
        return (np.arange(dim, dtype=np.int64),)

    @staticmethod
    def _left_prefix_axis(tensor_index: int) -> int:
        return 1 if int(tensor_index) == 0 else 3

    @staticmethod
    def _right_prefix_axis(_tensor_index: int) -> int:
        return 0

    @staticmethod
    def _left_view_axis(_shared_index: int) -> int:
        return 2

    @staticmethod
    def _right_view_axis(_shared_index: int) -> int:
        return 1

    def _prefix_left_matrix(self, bond: int, group) -> np.ndarray:
        tensor = self.tensors[int(bond) - 1]
        axis = self._left_prefix_axis(int(bond) - 1)
        group = np.asarray(group, dtype=np.int64)
        selector = [slice(None)] * tensor.ndim
        selector[axis] = group
        block = tensor[tuple(selector)]
        return np.moveaxis(block, axis, -1).reshape(-1, group.size)

    def _prefix_right_matrix(self, bond: int, group) -> np.ndarray:
        tensor = self.tensors[int(bond)]
        axis = self._right_prefix_axis(int(bond))
        group = np.asarray(group, dtype=np.int64)
        selector = [slice(None)] * tensor.ndim
        selector[axis] = group
        block = tensor[tuple(selector)]
        return np.moveaxis(block, axis, 0).reshape(group.size, -1)

    def prefix_bond_orthogonality_error(self, direction: str = "lr") -> tuple[float, ...]:
        """Return sector-block orthogonality errors for the prefix ``a`` bonds."""
        direction = str(direction).lower()
        if direction not in {"lr", "rl"}:
            raise ValueError("direction must be 'lr' or 'rl'.")
        errors = []
        for bond in range(1, self.nsites):
            block_errors = []
            for group in self._prefix_bond_groups(bond):
                if direction == "lr":
                    matrix = self._prefix_left_matrix(bond, group)
                    gram = matrix.conj().T @ matrix
                else:
                    matrix = self._prefix_right_matrix(bond, group)
                    gram = matrix @ matrix.conj().T
                block_errors.append(float(np.linalg.norm(gram - np.eye(group.size))))
            errors.append(max(block_errors, default=0.0))
        return tuple(errors)

    def canonicalize_prefix_bond(
        self,
        bond: int,
        *,
        direction: str = "lr",
        eps: float = 1.0e-14,
        rcond: float = 1.0e-12,
        normalize: bool = False,
    ) -> "XLETTA":
        """Whiten one prefix bond with a state-preserving sector gauge."""
        bond = int(bond)
        if bond < 1 or bond >= self.nsites:
            raise IndexError("prefix bond index must satisfy 1 <= bond < nsites.")
        direction = str(direction).lower()
        if direction not in {"lr", "rl"}:
            raise ValueError("direction must be 'lr' or 'rl'.")
        left_index = bond - 1
        right_index = bond
        left_axis = self._left_prefix_axis(left_index)
        right_axis = self._right_prefix_axis(right_index)

        for group in self._prefix_bond_groups(bond):
            if group.size == 0:
                continue
            if direction == "lr":
                matrix = self._prefix_left_matrix(bond, group)
                gram = matrix.conj().T @ matrix
                sqrt, inv_sqrt = _hermitian_sqrt_pair(gram, eps=eps, rcond=rcond)
                self.tensors[left_index] = self._axis_transform_subset(
                    self.tensors[left_index],
                    left_axis,
                    group,
                    inv_sqrt,
                )
                self.tensors[right_index] = self._axis_transform_subset(
                    self.tensors[right_index],
                    right_axis,
                    group,
                    sqrt.T,
                )
            else:
                matrix = self._prefix_right_matrix(bond, group)
                gram = matrix @ matrix.conj().T
                sqrt, inv_sqrt = _hermitian_sqrt_pair(gram, eps=eps, rcond=rcond)
                self.tensors[left_index] = self._axis_transform_subset(
                    self.tensors[left_index],
                    left_axis,
                    group,
                    sqrt,
                )
                self.tensors[right_index] = self._axis_transform_subset(
                    self.tensors[right_index],
                    right_axis,
                    group,
                    inv_sqrt.T,
                )
        self._apply_variable_masks()
        if normalize:
            self.normalize()
        return self

    def canonicalize_prefix_bonds(
        self,
        *,
        direction: str = "lr",
        eps: float = 1.0e-14,
        rcond: float = 1.0e-12,
        normalize: bool = True,
    ) -> "XLETTA":
        """Apply a moving-center prefix-bond gauge across the XLETTA chain."""
        direction = str(direction).lower()
        if direction not in {"lr", "rl"}:
            raise ValueError("direction must be 'lr' or 'rl'.")
        bonds = range(1, self.nsites)
        if direction == "rl":
            bonds = reversed(list(bonds))
        for bond in bonds:
            self.canonicalize_prefix_bond(
                bond,
                direction=direction,
                eps=eps,
                rcond=rcond,
                normalize=False,
            )
        if normalize:
            self.normalize()
        return self

    def _w_view_matrix(self, shared_index: int, leg: str, group) -> np.ndarray:
        leg = str(leg).lower()
        axis = 0 if leg == "u" else 1 if leg == "v" else None
        if axis is None:
            raise ValueError("leg must be 'u' or 'v'.")
        w = self.w_tensors[int(shared_index)]
        group = np.asarray(group, dtype=np.int64)
        selector = [slice(None)] * w.ndim
        selector[axis] = group
        block = w[tuple(selector)]
        return np.moveaxis(block, axis, 0).reshape(group.size, -1)

    def view_leg_isometry_error(self, leg: str = "u") -> tuple[float, ...]:
        """Return row-isometry errors for one lifted view leg of each ``W``."""
        leg = str(leg).lower()
        if leg not in {"u", "v"}:
            raise ValueError("leg must be 'u' or 'v'.")
        errors = []
        for shared in range(self.nshared):
            block_errors = []
            for group in self._view_leg_groups(shared):
                matrix = self._w_view_matrix(shared, leg, group)
                gram = matrix @ matrix.conj().T
                block_errors.append(float(np.linalg.norm(gram - np.eye(group.size))))
            errors.append(max(block_errors, default=0.0))
        return tuple(errors)

    def canonicalize_w_view_leg(
        self,
        shared_index: int,
        *,
        leg: str = "u",
        eps: float = 1.0e-14,
        rcond: float = 1.0e-12,
        normalize: bool = False,
    ) -> "XLETTA":
        """Whiten one lifted view leg of ``W`` with a compensating A-tensor gauge."""
        shared_index = int(shared_index)
        if shared_index < 0 or shared_index >= self.nshared:
            raise IndexError("shared_index out of range.")
        leg = str(leg).lower()
        if leg not in {"u", "v"}:
            raise ValueError("leg must be 'u' or 'v'.")

        if leg == "u":
            w_axis = 0
            tensor_index = shared_index
            tensor_axis = self._left_view_axis(shared_index)
        else:
            w_axis = 1
            tensor_index = shared_index + 1
            tensor_axis = self._right_view_axis(shared_index)

        for group in self._view_leg_groups(shared_index):
            if group.size == 0:
                continue
            matrix = self._w_view_matrix(shared_index, leg, group)
            gram = matrix @ matrix.conj().T
            sqrt, inv_sqrt = _hermitian_sqrt_pair(gram, eps=eps, rcond=rcond)
            self.w_tensors[shared_index] = self._axis_transform_subset(
                self.w_tensors[shared_index],
                w_axis,
                group,
                inv_sqrt.T,
            )
            self.tensors[tensor_index] = self._axis_transform_subset(
                self.tensors[tensor_index],
                tensor_axis,
                group,
                sqrt,
            )
        self._apply_variable_masks()
        if normalize:
            self.normalize()
        return self

    def canonicalize_w_view_legs(
        self,
        *,
        legs=("u", "v"),
        iterations: int = 1,
        eps: float = 1.0e-14,
        rcond: float = 1.0e-12,
        normalize: bool = True,
    ) -> "XLETTA":
        """Apply state-preserving gauges that whiten the lifted ``W`` view legs."""
        if isinstance(legs, str):
            legs = (legs,)
        legs = tuple(str(leg).lower() for leg in legs)
        if any(leg not in {"u", "v"} for leg in legs):
            raise ValueError("legs must contain only 'u' and/or 'v'.")
        for _ in range(int(iterations)):
            for shared in range(self.nshared):
                for leg in legs:
                    self.canonicalize_w_view_leg(
                        shared,
                        leg=leg,
                        eps=eps,
                        rcond=rcond,
                        normalize=False,
                    )
        if normalize:
            self.normalize()
        return self

    def canonicalize_gauge(
        self,
        *,
        prefix: bool = True,
        view: bool = True,
        direction: str = "lr",
        view_legs=("u", "v"),
        view_iterations: int = 1,
        eps: float = 1.0e-14,
        rcond: float = 1.0e-12,
        normalize: bool = True,
    ) -> "XLETTA":
        """Canonicalize the native XLETTA internal gauge.

        The operation is state preserving: prefix bonds are whitened like MPS
        virtual bonds, and lifted ``W`` view-leg gauges are compensated by the
        neighboring XLETTA tensors.  The physical ``s`` leg is intentionally not
        transformed because that would change the represented wavefunction in
        the fixed physical basis.
        """
        if prefix:
            self.canonicalize_prefix_bonds(
                direction=direction,
                eps=eps,
                rcond=rcond,
                normalize=False,
            )
        if view:
            self.canonicalize_w_view_legs(
                legs=view_legs,
                iterations=view_iterations,
                eps=eps,
                rcond=rcond,
                normalize=False,
            )
        self._apply_variable_masks()
        if normalize:
            self.normalize()
        return self

    def _local_core_matrices_from_environments(
        self,
        site: int,
        mpo_site,
        core,
        left,
        right,
        metric_left,
        metric_right,
    ):
        dim = int(np.prod(core.shape))

        heff = np.einsum(
            "abm,mnst,cdn->ascbtd",
            left,
            mpo_site,
            right,
            optimize=True,
        ).reshape(dim, dim)
        eye = np.eye(core.shape[1], dtype=np.result_type(core.dtype, float))
        metric = np.einsum(
            "ab,cd,st->ascbtd",
            metric_left,
            metric_right,
            eye,
            optimize=True,
        ).reshape(dim, dim)
        heff = 0.5 * (heff + heff.conj().T)
        metric = 0.5 * (metric + metric.conj().T)
        return heff, metric

    def _local_core_matrices(self, site: int, mpo, cores=None):
        site = int(site)
        mpo = _validate_mpo(mpo, self.dims)
        cores = self.effective_mps_cores() if cores is None else cores
        left = self._left_mpo_environments(mpo, cores)[site]
        right = self._right_mpo_environments(mpo, cores)[site + 1]
        metric_left = self._left_metric_environments(cores)[site]
        metric_right = self._right_metric_environments(cores)[site + 1]
        return self._local_core_matrices_from_environments(
            site,
            mpo[site],
            cores[site],
            left,
            right,
            metric_left,
            metric_right,
        )

    def _core_projection(self, kind: str, index: int, cores=None, support=None):
        kind = str(kind)
        index = int(index)
        cores = self.effective_mps_cores() if cores is None else cores
        if kind == "tensor":
            site = index
        elif kind == "w":
            site = index + 1
        else:
            raise ValueError("kind must be 'tensor' or 'w'.")
        core_shape = cores[site].shape
        variable = self._variable(kind, index)
        dtype = np.result_type(variable.dtype, float)
        if kind == "tensor" and index > 0:
            dtype = np.result_type(dtype, self.w_tensors[index - 1].dtype)
        if kind == "w":
            dtype = np.result_type(dtype, self.tensors[site].dtype)
        support = self._variable_support(kind, index) if support is None else np.asarray(support, dtype=np.int64)
        projection = np.zeros((int(np.prod(core_shape)), support.size), dtype=dtype)

        if kind == "tensor" and index == 0:
            for col, flat in enumerate(support):
                s, a, u = np.unravel_index(int(flat), variable.shape)
                row = np.ravel_multi_index((0, s, a * variable.shape[2] + u), core_shape)
                projection[row, col] = 1.0
            return site, projection

        if kind == "tensor" and index == self.nsites - 1:
            w = self.w_tensors[-1]
            for col, flat in enumerate(support):
                a, v = np.unravel_index(int(flat), variable.shape)
                for u, s in np.ndindex(w.shape[0], w.shape[2]):
                    row = np.ravel_multi_index((a * w.shape[0] + u, s, 0), core_shape)
                    projection[row, col] = w[u, v, s]
            return site, projection

        if kind == "tensor":
            w = self.w_tensors[index - 1]
            tensor = variable
            for col, flat in enumerate(support):
                a, v, x, b = np.unravel_index(int(flat), tensor.shape)
                for u, s in np.ndindex(w.shape[0], w.shape[2]):
                    row = np.ravel_multi_index((a * w.shape[0] + u, s, b * tensor.shape[2] + x), core_shape)
                    projection[row, col] = w[u, v, s]
            return site, projection

        if kind == "w" and site == self.nsites - 1:
            tensor = self.tensors[-1]
            w = variable
            for col, flat in enumerate(support):
                u, v, s = np.unravel_index(int(flat), w.shape)
                for a in range(tensor.shape[0]):
                    row = np.ravel_multi_index((a * w.shape[0] + u, s, 0), core_shape)
                    projection[row, col] = tensor[a, v]
            return site, projection

        tensor = self.tensors[site]
        w = variable
        for col, flat in enumerate(support):
            u, v, s = np.unravel_index(int(flat), w.shape)
            for a, x, b in np.ndindex(tensor.shape[0], tensor.shape[2], tensor.shape[3]):
                row = np.ravel_multi_index((a * w.shape[0] + u, s, b * tensor.shape[2] + x), core_shape)
                projection[row, col] = tensor[a, v, x, b]
        return site, projection

    def _abelian_variable_blocks(self, kind: str, index: int) -> tuple:
        layout = self.abelian_layout
        if layout is None or not hasattr(layout, "local_variable_blocks"):
            return ()
        try:
            return tuple(layout.local_variable_blocks(str(kind), int(index)))
        except Exception:
            return ()

    def _block_variable_support(self, kind: str, index: int):
        blocks = self._abelian_variable_blocks(kind, index)
        if not blocks:
            return None, ()
        support = np.concatenate([np.asarray(block.flat_indices, dtype=np.int64) for block in blocks])
        if support.size == 0 or np.unique(support).size != support.size:
            return None, ()
        mask_support = self._variable_support(kind, index)
        if support.size != mask_support.size or not np.array_equal(np.sort(support), mask_support):
            return None, ()
        return support, blocks

    def _projected_core_entries(self, kind: str, index: int, cores=None, support=None, *, drop_tol: float = 0.0):
        """Return sparse projected-core rows for one XLETTA variable.

        Each local variational coordinate can contribute to several entries of
        the induced ordinary-MPS core.  The dense fallback materializes that
        projection as a full matrix; this routine keeps only the nonzero rows.
        """
        kind = str(kind)
        index = int(index)
        cores = self.effective_mps_cores() if cores is None else cores
        if kind == "tensor":
            site = index
        elif kind == "w":
            site = index + 1
        else:
            raise ValueError("kind must be 'tensor' or 'w'.")
        variable = self._variable(kind, index)
        support = self._variable_support(kind, index) if support is None else np.asarray(support, dtype=np.int64)
        core_shape = cores[site].shape
        tol = float(drop_tol)

        columns = []
        left_rows = []
        physical_rows = []
        right_rows = []
        values = []

        def append(column, left, physical, right, value):
            if tol > 0.0 and abs(value) <= tol:
                return
            if value == 0:
                return
            columns.append(int(column))
            left_rows.append(int(left))
            physical_rows.append(int(physical))
            right_rows.append(int(right))
            values.append(value)

        if kind == "tensor" and index == 0:
            for col, flat in enumerate(support):
                s, a, u = np.unravel_index(int(flat), variable.shape)
                append(col, 0, s, a * variable.shape[2] + u, 1.0)
        elif kind == "tensor" and index == self.nsites - 1:
            w = self.w_tensors[-1]
            for col, flat in enumerate(support):
                a, v = np.unravel_index(int(flat), variable.shape)
                for u, s in np.ndindex(w.shape[0], w.shape[2]):
                    append(col, a * w.shape[0] + u, s, 0, w[u, v, s])
        elif kind == "tensor":
            w = self.w_tensors[index - 1]
            tensor = variable
            for col, flat in enumerate(support):
                a, v, x, b = np.unravel_index(int(flat), tensor.shape)
                for u, s in np.ndindex(w.shape[0], w.shape[2]):
                    append(col, a * w.shape[0] + u, s, b * tensor.shape[2] + x, w[u, v, s])
        elif kind == "w" and site == self.nsites - 1:
            tensor = self.tensors[-1]
            w = variable
            for col, flat in enumerate(support):
                u, v, s = np.unravel_index(int(flat), w.shape)
                for a in range(tensor.shape[0]):
                    append(col, a * w.shape[0] + u, s, 0, tensor[a, v])
        else:
            tensor = self.tensors[site]
            w = variable
            for col, flat in enumerate(support):
                u, v, s = np.unravel_index(int(flat), w.shape)
                for a, x, b in np.ndindex(tensor.shape[0], tensor.shape[2], tensor.shape[3]):
                    append(col, a * w.shape[0] + u, s, b * tensor.shape[2] + x, tensor[a, v, x, b])

        if not values:
            raise ValueError(f"{kind} variable {index} has no support in the induced MPS core.")
        left_rows = np.asarray(left_rows, dtype=np.int64)
        physical_rows = np.asarray(physical_rows, dtype=np.int64)
        right_rows = np.asarray(right_rows, dtype=np.int64)
        if np.any(left_rows < 0) or np.any(left_rows >= core_shape[0]):
            raise ValueError("projected left-row index is out of bounds.")
        if np.any(physical_rows < 0) or np.any(physical_rows >= core_shape[1]):
            raise ValueError("projected physical index is out of bounds.")
        if np.any(right_rows < 0) or np.any(right_rows >= core_shape[2]):
            raise ValueError("projected right-row index is out of bounds.")
        return site, {
            "columns": np.asarray(columns, dtype=np.int64),
            "left": left_rows,
            "physical": physical_rows,
            "right": right_rows,
            "values": np.asarray(values, dtype=np.result_type(*[np.asarray(v).dtype for v in values])),
            "ncols": int(support.size),
            "support": support,
        }

    @staticmethod
    def _accumulate_projected_matrix(out, row_cols, col_cols, values):
        if values.size == 0:
            return
        np.add.at(out, (row_cols[:, None], col_cols[None, :]), values)

    def _projected_local_matrices_from_environments(
        self,
        kind: str,
        index: int,
        mpo_site,
        cores,
        left,
        right,
        metric_left,
        metric_right,
        *,
        support=None,
        drop_tol: float = 0.0,
    ):
        """Build ``P^dag H_eff P`` without materializing the full core matrix."""
        site, entries = self._projected_core_entries(
            kind,
            index,
            cores=cores,
            support=support,
            drop_tol=drop_tol,
        )
        ncols = int(entries["ncols"])
        dtype = np.result_type(
            entries["values"].dtype,
            np.asarray(left).dtype,
            np.asarray(right).dtype,
            np.asarray(mpo_site).dtype,
        )
        heff = np.zeros((ncols, ncols), dtype=dtype)
        metric = np.zeros((ncols, ncols), dtype=np.result_type(dtype, np.asarray(metric_left).dtype, np.asarray(metric_right).dtype))

        columns = entries["columns"]
        lrows = entries["left"]
        prows = entries["physical"]
        rrows = entries["right"]
        coeff = entries["values"]
        groups = [
            np.flatnonzero(prows == physical)
            for physical in range(int(mpo_site.shape[2]))
        ]

        nz = np.argwhere(np.abs(mpo_site) > float(drop_tol))
        for m, n, bra_phys, ket_phys in nz:
            bra = groups[int(bra_phys)]
            ket = groups[int(ket_phys)]
            if bra.size == 0 or ket.size == 0:
                continue
            block = (
                mpo_site[int(m), int(n), int(bra_phys), int(ket_phys)]
                * left[lrows[bra, None], lrows[ket][None, :], int(m)]
                * right[rrows[bra, None], rrows[ket][None, :], int(n)]
                * coeff[bra].conj()[:, None]
                * coeff[ket][None, :]
            )
            self._accumulate_projected_matrix(heff, columns[bra], columns[ket], block)

        for physical, group in enumerate(groups):
            if group.size == 0:
                continue
            block = (
                metric_left[lrows[group, None], lrows[group][None, :]]
                * metric_right[rrows[group, None], rrows[group][None, :]]
                * coeff[group].conj()[:, None]
                * coeff[group][None, :]
            )
            self._accumulate_projected_matrix(metric, columns[group], columns[group], block)

        heff = 0.5 * (heff + heff.conj().T)
        metric = 0.5 * (metric + metric.conj().T)
        return site, heff, metric, {
            "local_solver": "irrep_projected",
            "projected_core_entries": int(coeff.size),
            "projected_variable_dim": ncols,
        }

    def _pad_with_noise(self, array, shape, rng, noise: float):
        shape = tuple(int(dim) for dim in shape)
        out = np.zeros(shape, dtype=array.dtype)
        old = tuple(slice(0, dim) for dim in array.shape)
        out[old] = array
        if float(noise) > 0.0:
            mask = np.ones(shape, dtype=bool)
            mask[old] = False
            if np.iscomplexobj(out):
                values = rng.normal(scale=float(noise), size=mask.sum())
                values = values + 1j * rng.normal(scale=float(noise), size=mask.sum())
            else:
                values = rng.normal(scale=float(noise), size=mask.sum())
            out[mask] = values
        return out

    def _pad_mask(self, mask, shape):
        if mask is None:
            return None
        shape = tuple(int(dim) for dim in shape)
        out = np.zeros(shape, dtype=bool)
        old = tuple(slice(0, dim) for dim in mask.shape)
        out[old] = mask
        return out

    def expand_view_dim(
        self,
        view_dim=None,
        *,
        increment: int | tuple[int, ...] | None = None,
        max_view_dim=None,
        noise: float = 0.0,
        seed=None,
    ) -> "XLETTA":
        """Increase lifted shared-leg dimensions while preserving the old state.

        With ``noise=0`` the expansion is an exact zero-padding.  A small noise
        value opens new variational directions, giving a simple subspace
        expansion mechanism for later sweeps.
        """
        if view_dim is None:
            if increment is None:
                increment = 1
            if np.isscalar(increment):
                increments = (int(increment),) * self.nshared
            else:
                increments = tuple(int(value) for value in increment)
                if len(increments) != self.nshared:
                    raise ValueError("increment must have one entry for each shared site.")
            targets = tuple(q + dq for q, dq in zip(self.view_dims, increments))
        else:
            targets = _validate_view_dims(view_dim, self.dims)

        if max_view_dim is not None:
            if np.isscalar(max_view_dim):
                maxima = (int(max_view_dim),) * self.nshared
            else:
                maxima = tuple(int(value) for value in max_view_dim)
                if len(maxima) != self.nshared:
                    raise ValueError("max_view_dim must have one entry for each shared site.")
            targets = tuple(min(q, qmax) for q, qmax in zip(targets, maxima))

        targets = tuple(max(q, d) for q, d in zip(targets, self.dims[1:]))
        if all(q_new == q_old for q_new, q_old in zip(targets, self.view_dims)):
            return self
        if any(q_new < q_old for q_new, q_old in zip(targets, self.view_dims)):
            raise ValueError("expand_view_dim only supports increasing view dimensions.")

        rng = self.rng if seed is None else np.random.default_rng(seed)
        tensors = []
        tensor_masks = []
        first = self.tensors[0]
        shape = (first.shape[0], first.shape[1], targets[0])
        tensors.append(self._pad_with_noise(first, shape, rng, noise))
        tensor_masks.append(self._pad_mask(self.tensor_masks[0], shape))
        for site in range(1, self.nsites - 1):
            tensor = self.tensors[site]
            shape = (tensor.shape[0], targets[site - 1], targets[site], tensor.shape[3])
            tensors.append(
                self._pad_with_noise(
                    tensor,
                    shape,
                    rng,
                    noise,
                )
            )
            tensor_masks.append(self._pad_mask(self.tensor_masks[site], shape))
        last = self.tensors[-1]
        shape = (last.shape[0], targets[-1])
        tensors.append(self._pad_with_noise(last, shape, rng, noise))
        tensor_masks.append(self._pad_mask(self.tensor_masks[-1], shape))

        w_tensors = []
        w_masks = []
        for shared, w in enumerate(self.w_tensors):
            shape = (targets[shared], targets[shared], w.shape[2])
            w_tensors.append(
                self._pad_with_noise(
                    w,
                    shape,
                    rng,
                    noise,
                )
            )
            w_masks.append(self._pad_mask(self.w_masks[shared], shape))

        self.view_dims = targets
        self.view_dim = self.view_dims[0] if len(set(self.view_dims)) == 1 else self.view_dims
        self.tensors = tensors
        self.w_tensors = w_tensors
        self.tensor_masks = tensor_masks
        self.w_masks = w_masks
        self._apply_variable_masks()
        self.normalize()
        return self

    def variable_order(self, direction: str = "lr", *, symmetric: bool = True):
        forward = [("tensor", 0)]
        for shared in range(self.nshared):
            forward.append(("w", shared))
            forward.append(("tensor", shared + 1))
        if direction.lower() == "rl":
            forward = list(reversed(forward))
        elif direction.lower() != "lr":
            raise ValueError("direction must be 'lr' or 'rl'.")
        if symmetric and len(forward) > 2:
            return forward + list(reversed(forward[1:-1]))
        return forward

    def projector(self, kind: str, index: int, support=None) -> np.ndarray:
        original = self._variable(kind, index)
        support = self._variable_support(kind, index) if support is None else np.asarray(support, dtype=np.int64)
        columns = []
        dtype = np.result_type(original.dtype, float)
        for flat in support:
            trial = np.zeros(original.shape, dtype=dtype)
            trial.reshape(-1)[flat] = 1.0
            columns.append(self.state_vector(override=(kind, index, trial)))
        return np.column_stack(columns)

    @staticmethod
    def _metric_normalize_vector(vector, metric):
        norm2 = np.vdot(vector, metric @ vector)
        norm = np.sqrt(float(np.real(norm2)))
        if norm < 1.0e-14:
            raise ValueError("Cannot normalize a numerically zero local update.")
        return vector / norm

    def optimize_variable(self, kind: str, index: int, *, metric_tol: float = 1.0e-12) -> dict:
        if self.hamiltonian is None:
            raise ValueError("dense or sparse hamiltonian is not available.")
        support = self._variable_support(kind, index)
        projector = self.projector(kind, index, support=support)
        h_projector = self.hamiltonian @ projector
        heff = projector.conj().T @ h_projector
        metric_projector = projector if self.overlap is None else self.overlap @ projector
        metric = projector.conj().T @ metric_projector
        heff = 0.5 * (heff + heff.conj().T)
        metric = 0.5 * (metric + metric.conj().T)
        local_energy, vector, info = _whitened_lowest_generalized_eigenpair(
            heff,
            metric,
            metric_tol=metric_tol,
        )
        full_vector = np.zeros(self._variable(kind, index).size, dtype=vector.dtype)
        full_vector[support] = vector
        self._set_variable(kind, index, full_vector.reshape(self._variable(kind, index).shape))
        self.normalize()
        return {"kind": kind, "index": int(index), "local_energy": float(local_energy), **info}

    def optimize_variable_mpo(
        self,
        mpo,
        kind: str,
        index: int,
        *,
        cores=None,
        left_mpo_environment=None,
        right_mpo_environment=None,
        left_metric_environment=None,
        right_metric_environment=None,
        metric_tol: float = 1.0e-12,
        w_copy_mix: float = 0.0,
        normalize: bool = True,
        block_sparse: bool | None = None,
    ) -> dict:
        """Optimize one XLETTA variable with MPO environments."""
        mpo = _validate_mpo(mpo, self.dims)
        cores = self.effective_mps_cores() if cores is None else cores
        if kind == "tensor":
            site = int(index)
        elif kind == "w":
            site = int(index) + 1
        else:
            raise ValueError("kind must be 'tensor' or 'w'.")
        support = None
        blocks = ()
        if block_sparse is None:
            block_sparse = self.abelian_layout is not None
        if block_sparse:
            support, blocks = self._block_variable_support(kind, index)
        if support is None:
            support = self._variable_support(kind, index)
        if (
            left_mpo_environment is None
            or right_mpo_environment is None
            or left_metric_environment is None
            or right_metric_environment is None
        ):
            left_mpo_environment = self._left_mpo_environments(mpo, cores)[site]
            right_mpo_environment = self._right_mpo_environments(mpo, cores)[site + 1]
            left_metric_environment = self._left_metric_environments(cores)[site]
            right_metric_environment = self._right_metric_environments(cores)[site + 1]
        if block_sparse:
            site, heff, metric, projection_info = self._projected_local_matrices_from_environments(
                kind,
                index,
                mpo[site],
                cores,
                left_mpo_environment,
                right_mpo_environment,
                left_metric_environment,
                right_metric_environment,
                support=support,
            )
        else:
            site, projection = self._core_projection(kind, index, cores, support=support)
            core_heff, core_metric = self._local_core_matrices_from_environments(
                site,
                mpo[site],
                cores[site],
                left_mpo_environment,
                right_mpo_environment,
                left_metric_environment,
                right_metric_environment,
            )
            heff = projection.conj().T @ core_heff @ projection
            metric = projection.conj().T @ core_metric @ projection
            heff = 0.5 * (heff + heff.conj().T)
            metric = 0.5 * (metric + metric.conj().T)
            projection_info = {
                "local_solver": "dense_core_projected",
                "projected_core_entries": int(np.count_nonzero(projection)),
                "projected_variable_dim": int(support.size),
            }
        local_energy, vector, info = _whitened_lowest_generalized_eigenpair(
            heff,
            metric,
            metric_tol=metric_tol,
        )
        if kind == "w" and float(w_copy_mix) > 0.0:
            mix = min(1.0, max(0.0, float(w_copy_mix)))
            reference = self.copy_w(self.dims[site], self.view_dims[index]).reshape(-1)[support]
            vector = (1.0 - mix) * vector + mix * reference
            vector = self._metric_normalize_vector(vector, metric)
        full_vector = np.zeros(self._variable(kind, index).size, dtype=vector.dtype)
        full_vector[support] = vector
        self._set_variable(kind, index, full_vector.reshape(self._variable(kind, index).shape))
        if normalize:
            self.normalize()
        return {
            "kind": kind,
            "index": int(index),
            "site": site,
            "local_energy": float(local_energy),
            "irrep_blocks": int(len(blocks)),
            **projection_info,
            **info,
        }

    def optimize_tensor(self, tensor_index: int) -> dict:
        return self.optimize_variable("tensor", int(tensor_index))

    def optimize_w(self, shared_index: int) -> dict:
        return self.optimize_variable("w", int(shared_index))

    def optimize_tensor_mpo(self, mpo, tensor_index: int) -> dict:
        return self.optimize_variable_mpo(mpo, "tensor", int(tensor_index))

    def optimize_w_mpo(self, mpo, shared_index: int) -> dict:
        return self.optimize_variable_mpo(mpo, "w", int(shared_index))

    def sweep(
        self,
        direction: str = "lr",
        operator=None,
        *,
        symmetric: bool = True,
        cached: bool = True,
        metric_tol: float = 1.0e-12,
        w_copy_mix: float = 0.0,
    ) -> list[dict]:
        """Run one variational sweep.

        If ``operator`` is provided, it is treated as an MPO and optimized with
        cached environment updates. Without an operator, this performs the dense
        local tensor sweep.
        """
        if operator is not None:
            return self.sweep_mpo(
                operator,
                direction,
                symmetric=symmetric,
                cached=cached,
                metric_tol=metric_tol,
                w_copy_mix=w_copy_mix,
            )

        return [
            self.optimize_variable(kind, index, metric_tol=metric_tol)
            for kind, index in self.variable_order(direction, symmetric=symmetric)
        ]

    def _sweep_mpo_cached_direction(
        self,
        mpo,
        direction: str,
        order,
        *,
        metric_tol: float = 1.0e-12,
        w_copy_mix: float = 0.0,
    ) -> list[dict]:
        """Run one monotone MPO pass while incrementally updating environments."""
        direction = direction.lower()
        if direction not in {"lr", "rl"}:
            raise ValueError("direction must be 'lr' or 'rl'.")
        mpo = _validate_mpo(mpo, self.dims)
        order = list(order)
        if not order:
            return []

        cores = self.effective_mps_cores()
        dtype = np.result_type(*[core.dtype for core in cores], *[site.dtype for site in mpo])
        metric_dtype = np.result_type(*[core.dtype for core in cores])
        updates = []

        if direction == "lr":
            right_mpo = self._right_mpo_environments(mpo, cores)
            right_metric = self._right_metric_environments(cores)
            left_mpo = [None] * (self.nsites + 1)
            left_metric = [None] * (self.nsites + 1)
            left_mpo[0] = np.ones((1, 1, 1), dtype=dtype)
            left_metric[0] = np.ones((1, 1), dtype=metric_dtype)

            def ensure_left(site):
                for cursor in range(self.nsites):
                    if cursor >= site:
                        break
                    if left_mpo[cursor + 1] is None:
                        left_mpo[cursor + 1] = self._advance_left_mpo_environment(
                            left_mpo[cursor],
                            mpo[cursor],
                            cores[cursor],
                        )
                        left_metric[cursor + 1] = self._advance_left_metric_environment(
                            left_metric[cursor],
                            cores[cursor],
                        )

            for kind, index in order:
                site = index if kind == "tensor" else index + 1
                ensure_left(site)
                update = self.optimize_variable_mpo(
                    mpo,
                    kind,
                    index,
                    cores=cores,
                    left_mpo_environment=left_mpo[site],
                    right_mpo_environment=right_mpo[site + 1],
                    left_metric_environment=left_metric[site],
                    right_metric_environment=right_metric[site + 1],
                    metric_tol=metric_tol,
                    w_copy_mix=w_copy_mix,
                    normalize=False,
                )
                cores[site] = self._effective_mps_core(site)
                left_mpo[site + 1] = self._advance_left_mpo_environment(left_mpo[site], mpo[site], cores[site])
                left_metric[site + 1] = self._advance_left_metric_environment(left_metric[site], cores[site])
                for stale in range(site + 2, self.nsites + 1):
                    left_mpo[stale] = None
                    left_metric[stale] = None
                updates.append(update)
            return updates

        left_mpo = self._left_mpo_environments(mpo, cores)
        left_metric = self._left_metric_environments(cores)
        right_mpo = [None] * (self.nsites + 1)
        right_metric = [None] * (self.nsites + 1)
        right_mpo[-1] = np.ones((1, 1, 1), dtype=dtype)
        right_metric[-1] = np.ones((1, 1), dtype=metric_dtype)

        def ensure_right(site):
            for cursor in reversed(range(self.nsites)):
                if cursor <= site:
                    break
                if right_mpo[cursor] is None:
                    right_mpo[cursor] = self._advance_right_mpo_environment(
                        right_mpo[cursor + 1],
                        mpo[cursor],
                        cores[cursor],
                    )
                    right_metric[cursor] = self._advance_right_metric_environment(
                        right_metric[cursor + 1],
                        cores[cursor],
                    )

        for kind, index in order:
            site = index if kind == "tensor" else index + 1
            ensure_right(site)
            update = self.optimize_variable_mpo(
                mpo,
                kind,
                index,
                cores=cores,
                left_mpo_environment=left_mpo[site],
                right_mpo_environment=right_mpo[site + 1],
                left_metric_environment=left_metric[site],
                right_metric_environment=right_metric[site + 1],
                metric_tol=metric_tol,
                w_copy_mix=w_copy_mix,
                normalize=False,
            )
            cores[site] = self._effective_mps_core(site)
            right_mpo[site] = self._advance_right_mpo_environment(right_mpo[site + 1], mpo[site], cores[site])
            right_metric[site] = self._advance_right_metric_environment(right_metric[site + 1], cores[site])
            for stale in range(site):
                right_mpo[stale] = None
                right_metric[stale] = None
            updates.append(update)
        return updates

    def sweep_mpo(
        self,
        mpo,
        direction: str = "lr",
        *,
        symmetric: bool = True,
        cached: bool = True,
        metric_tol: float = 1.0e-12,
        w_copy_mix: float = 0.0,
    ) -> list[dict]:
        """Run one local variational sweep using MPO environments.

        Compatibility entrypoint; new code should call ``sweep(..., operator=mpo)``.
        """
        mpo = _validate_mpo(mpo, self.dims)
        if not cached:
            return [
                self.optimize_variable_mpo(
                    mpo,
                    kind,
                    index,
                    metric_tol=metric_tol,
                    w_copy_mix=w_copy_mix,
                )
                for kind, index in self.variable_order(direction, symmetric=symmetric)
            ]
        order = self.variable_order(direction, symmetric=False)
        updates = self._sweep_mpo_cached_direction(
            mpo,
            direction,
            order,
            metric_tol=metric_tol,
            w_copy_mix=w_copy_mix,
        )
        if symmetric and len(order) > 2:
            reverse_direction = "rl" if direction.lower() == "lr" else "lr"
            updates.extend(
                self._sweep_mpo_cached_direction(
                    mpo,
                    reverse_direction,
                    reversed(order[1:-1]),
                    metric_tol=metric_tol,
                    w_copy_mix=w_copy_mix,
                )
            )
        self.normalize()
        return updates

    def run(
        self,
        operator=None,
        nsweeps=None,
        *,
        sweeps=None,
        start_direction: str = "lr",
        alternate: bool = False,
        symmetric: bool = True,
        cached_mpo: bool = True,
        metric_tol: float = 1.0e-12,
        adaptive_view_dim: bool = False,
        expand_every: int = 1,
        expand_by: int = 1,
        max_view_dim=None,
        expand_noise: float = 1.0e-8,
        w_regularization: float = 0.0,
        w_regularization_decay: float = 1.0,
        canonicalize: bool = False,
        canonicalize_every: int = 1,
        canonicalize_view_legs=("u",),
        canonicalize_view_iterations: int = 1,
        tol: float = 1.0e-10,
        verbose: bool | int = False,
        label: str = "xletta",
    ) -> "XLETTA":
        """Run dense one-site variational sweeps.

        ``operator`` may be a dense/sparse matrix or an MPO with site tensors
        shaped ``(left, right, bra, ket)``.  MPOs use effective-MPS
        environments; dense/sparse matrices use the dense projector path.
        """
        if operator is not None and np.isscalar(operator) and nsweeps is None and sweeps is None:
            nsweeps = int(operator)
            operator = None
        if operator is not None:
            if _looks_like_mpo(operator):
                self.operator_mpo = _validate_mpo(operator, self.dims)
                self.hamiltonian = None
            else:
                self.hamiltonian = _as_operator(operator, self.dims)
                self.operator_mpo = None
        if nsweeps is None:
            nsweeps = sweeps
        if nsweeps is None:
            nsweeps = 4
        if int(nsweeps) < 1:
            raise ValueError("nsweeps must be positive.")

        direction = start_direction.lower()
        previous_energy = None
        self.history = []
        self.converged = False
        for sweep_idx in range(int(nsweeps)):
            did_canonicalize = False
            if bool(canonicalize) and int(canonicalize_every) > 0 and sweep_idx % int(canonicalize_every) == 0:
                self.canonicalize_gauge(
                    direction=direction,
                    view_legs=canonicalize_view_legs,
                    view_iterations=canonicalize_view_iterations,
                    normalize=True,
                )
                did_canonicalize = True
            w_copy_mix = float(w_regularization) * (float(w_regularization_decay) ** sweep_idx)
            if self.operator_mpo is not None:
                updates = self.sweep(
                    direction,
                    self.operator_mpo,
                    symmetric=symmetric,
                    cached=cached_mpo,
                    metric_tol=metric_tol,
                    w_copy_mix=w_copy_mix,
                )
                energy = self.expectation_mpo(self.operator_mpo)
            else:
                updates = self.sweep(direction, symmetric=symmetric, metric_tol=metric_tol)
                energy = self.expectation()
            delta = None if previous_energy is None else abs(energy - previous_energy)
            expanded_from = self.view_dims
            expanded_to = None
            self.history.append(
                {
                    "sweep": sweep_idx,
                    "direction": direction,
                    "energy": energy,
                    "delta_energy": delta,
                    "updates": updates,
                    "w_copy_mix": w_copy_mix,
                    "canonicalized": did_canonicalize,
                    "expanded_view_dims": expanded_to,
                }
            )
            if int(verbose) > 0:
                print(
                    f"{label} sweep={sweep_idx:2d} E={energy: .12f} "
                    f"dE={'-' if delta is None else f'{delta:.3e}'}"
                )
            if delta is not None and delta <= float(tol):
                self.converged = True
                break
            if (
                adaptive_view_dim
                and max_view_dim is not None
                and sweep_idx + 1 < int(nsweeps)
                and int(expand_every) > 0
                and (sweep_idx + 1) % int(expand_every) == 0
                and any(q < qmax for q, qmax in zip(self.view_dims, _validate_view_dims(max_view_dim, self.dims)))
            ):
                self.expand_view_dim(increment=expand_by, max_view_dim=max_view_dim, noise=expand_noise)
                if self.view_dims != expanded_from:
                    expanded_to = self.view_dims
                    self.history[-1]["expanded_view_dims"] = expanded_to
                    previous_energy = None
                else:
                    previous_energy = energy
            else:
                previous_energy = energy
            if alternate:
                direction = "rl" if direction == "lr" else "lr"
        self.last_energy = self.history[-1]["energy"]
        return self


class AbelianXLETTA(XLETTA):
    """Abelian XLETTA with Irrep-block projected local solves.

    This class enforces Abelian charge sectors through :class:`XLayout`
    support masks for both the XLETTA ``A`` tensors and the lifted ``W``
    tensors.  MPO sweeps use the layout's Irrep block coordinates to project
    local effective Hamiltonians directly into the legal sector space instead
    of materializing full dense induced-MPS core matrices.
    """

    def __init__(
        self,
        hamiltonian,
        layout,
        *,
        bond_dim=None,
        view_dim=None,
        overlap=None,
        tensors=None,
        w_tensors=None,
        seed=None,
    ):
        xlayout = _as_xlayout(layout, view_dim=view_dim)
        if xlayout is None:
            raise ValueError("AbelianXLETTA requires an Abelian XLayout or LETTA Layout.")
        if bond_dim is None:
            bond_dim = max(len(bond) for bond in xlayout.prefix_qns)
        super().__init__(
            hamiltonian,
            xlayout.dims,
            bond_dim=bond_dim,
            view_dim=xlayout.view_dims,
            overlap=overlap,
            tensors=tensors,
            w_tensors=w_tensors,
            abelian_layout=xlayout,
            seed=seed,
        )

    @classmethod
    def from_local_charges(
        cls,
        hamiltonian,
        local_qns,
        *,
        target,
        left_boundary=None,
        prefix_qns=None,
        view_qns=None,
        view_dim=None,
        bond_dim=None,
        overlap=None,
        tensors=None,
        w_tensors=None,
        seed=None,
    ) -> "AbelianXLETTA":
        layout = XLayout.from_local_charges(
            local_qns,
            target=target,
            left_boundary=left_boundary,
            prefix_qns=prefix_qns,
            view_qns=view_qns,
            view_dim=view_dim,
        )
        return cls(
            hamiltonian,
            layout,
            bond_dim=bond_dim,
            overlap=overlap,
            tensors=tensors,
            w_tensors=w_tensors,
            seed=seed,
        )

    @classmethod
    def from_letta_layout(
        cls,
        hamiltonian,
        layout: Layout,
        *,
        view_qns=None,
        view_dim=None,
        bond_dim=None,
        overlap=None,
        tensors=None,
        w_tensors=None,
        seed=None,
    ) -> "AbelianXLETTA":
        xlayout = XLayout.from_letta_layout(layout, view_qns=view_qns, view_dim=view_dim)
        return cls(
            hamiltonian,
            xlayout,
            bond_dim=bond_dim,
            overlap=overlap,
            tensors=tensors,
            w_tensors=w_tensors,
            seed=seed,
        )

    @classmethod
    def from_mps(
        cls,
        mps,
        layout,
        *,
        hamiltonian=None,
        dims=None,
        view_dim=None,
        overlap=None,
        seed=None,
    ) -> "AbelianXLETTA":
        xlayout = _as_xlayout(layout, view_dim=view_dim)
        base = XLETTA.from_mps(
            mps,
            hamiltonian=None,
            dims=dims,
            view_dim=xlayout.view_dims,
            overlap=None,
            abelian_layout=xlayout,
            seed=seed,
        )
        return cls(
            hamiltonian,
            xlayout,
            bond_dim=base.bond_dim,
            overlap=overlap,
            tensors=base.tensors,
            w_tensors=base.w_tensors,
            seed=seed,
        )
