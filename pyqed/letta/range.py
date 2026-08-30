"""Range-2 physical-leg tied LETTA prototypes.

This module keeps the next-nearest-neighbor tied ansatz separate from the
production nearest-neighbor :class:`pyqed.letta.LETTA` implementation.  The
local eigensolves use range-2 MPO prefix/suffix environments that carry the two
open shared physical legs required by the ansatz.  Physical-pair-conditioned
gauges and adaptive bond compression expose an identity norm metric on the
active local support.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

from .core import (
    _lowest_generalized_eigenpair,
    _lowest_hermitian_eigenpair,
    _metric_basis,
)


def _validate_dims(dims) -> tuple[int, ...]:
    dims = tuple(int(dim) for dim in dims)
    if any(dim < 1 for dim in dims):
        raise ValueError("dims must contain positive integers.")
    return dims


class NNNLETTA:
    r"""Next-nearest-neighbor physical-leg tied LETTA.

    The represented wavefunction is

    $$
    \Psi(\sigma_0,\ldots,\sigma_{L-1}) =
    \sum_{\alpha}
    \prod_i A_i(\alpha_{i-1},\sigma_i,\sigma_{i+1},\sigma_{i+2},\alpha_i).
    $$

    Adjacent tensors share two physical legs.  This experimental class supports
    one-tensor MPO sweeps, adaptive conditional ranks, and a conditional
    canonical gauge over the shared physical pair on each virtual bond.
    """

    tie_range = 2
    tensor_width = 3

    def __init__(self, dims, *, bond_dim=4, tensors=None, local_masks=None, seed=None):
        self.dims = _validate_dims(dims)
        if len(self.dims) < self.tensor_width:
            raise ValueError("NNNLETTA needs at least three physical sites.")
        self.bond_dim = int(bond_dim)
        if self.bond_dim < 1:
            raise ValueError("bond_dim must be positive.")
        self.rng = np.random.default_rng(seed)
        self.tensors = self._random_tensors() if tensors is None else self._validate_tensors(tensors)
        self.local_masks = self._validate_local_masks(local_masks)
        self._apply_local_masks()
        self.history: list[dict] = []
        self.converged = False
        self.energy = None
        self.normalize()

    @property
    def nsites(self) -> int:
        return len(self.dims)

    @property
    def nlocal_tensors(self) -> int:
        return self.nsites - self.tensor_width + 1

    def _default_bonds(self) -> list[int]:
        return [1] + [self.bond_dim] * max(0, self.nlocal_tensors - 1) + [1]

    def _random_tensors(self) -> list[np.ndarray]:
        bonds = self._default_bonds()
        tensors = []
        for site in range(self.nlocal_tensors):
            shape = (
                bonds[site],
                self.dims[site],
                self.dims[site + 1],
                self.dims[site + 2],
                bonds[site + 1],
            )
            tensor = self.rng.normal(size=shape) / np.sqrt(np.prod(shape))
            tensors.append(tensor.astype(float))
        return tensors

    def _validate_tensors(self, tensors) -> list[np.ndarray]:
        tensors = [np.asarray(tensor, dtype=complex if np.iscomplexobj(tensor) else float) for tensor in tensors]
        if len(tensors) != self.nlocal_tensors:
            raise ValueError("number of NNNLETTA tensors must be len(dims)-2.")
        for site, tensor in enumerate(tensors):
            expected_phys = self.dims[site : site + self.tensor_width]
            if tensor.ndim != 5 or tensor.shape[1:4] != expected_phys:
                raise ValueError(
                    f"tensor {site} must have shape "
                    f"(left, {expected_phys[0]}, {expected_phys[1]}, {expected_phys[2]}, right)."
                )
            if site == 0 and tensor.shape[0] != 1:
                raise ValueError("first NNNLETTA tensor must have left bond dimension 1.")
            if site == len(tensors) - 1 and tensor.shape[-1] != 1:
                raise ValueError("last NNNLETTA tensor must have right bond dimension 1.")
            if site and tensors[site - 1].shape[-1] != tensor.shape[0]:
                raise ValueError(f"bond mismatch between NNNLETTA tensors {site - 1} and {site}.")
        return tensors

    def _validate_local_masks(self, local_masks):
        if local_masks is None:
            return None
        masks = [np.asarray(mask, dtype=bool) for mask in local_masks]
        if len(masks) != self.nlocal_tensors:
            raise ValueError("local_masks must have one entry per NNNLETTA tensor.")
        for site, (mask, tensor) in enumerate(zip(masks, self.tensors)):
            if mask.shape != tensor.shape:
                raise ValueError(f"local mask {site} shape does not match tensor shape.")
            if not np.any(mask):
                raise ValueError(f"local mask {site} has no active entries.")
        return masks

    def _apply_local_masks(self) -> None:
        if self.local_masks is None:
            return
        for site, mask in enumerate(self.local_masks):
            self.tensors[site] = np.where(mask, self.tensors[site], 0.0)

    def to_state_dict(self, *, metadata=None):
        """Return a pickle-friendly state payload for restarting NNN-LETTA sweeps."""
        return {
            "format": "pyqed.letta.NNNLETTA.state",
            "version": 1,
            "dims": tuple(int(dim) for dim in self.dims),
            "bond_dim": int(self.bond_dim),
            "tensors": [tensor.copy() for tensor in self.tensors],
            "local_masks": None
            if self.local_masks is None
            else [mask.copy() for mask in self.local_masks],
            "history": list(self.history),
            "converged": bool(self.converged),
            "energy": self.energy,
            "metadata": {} if metadata is None else dict(metadata),
        }

    @classmethod
    def from_state_dict(cls, payload) -> "NNNLETTA":
        """Reconstruct an NNN-LETTA state saved by :meth:`to_state_dict`."""
        if payload.get("format") != "pyqed.letta.NNNLETTA.state":
            raise ValueError("not a pyqed NNNLETTA state payload.")
        out = cls(
            payload["dims"],
            bond_dim=payload.get("bond_dim", 4),
            tensors=payload["tensors"],
            local_masks=payload.get("local_masks"),
        )
        out.history = list(payload.get("history", []))
        out.converged = bool(payload.get("converged", False))
        out.energy = payload.get("energy")
        out.state_metadata = dict(payload.get("metadata", {}))
        return out

    def save(self, path, *, metadata=None):
        """Save this NNN-LETTA state to ``path`` for later continuation."""
        output = Path(path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as handle:
            pickle.dump(self.to_state_dict(metadata=metadata), handle, protocol=pickle.HIGHEST_PROTOCOL)
        return output

    @classmethod
    def load(cls, path) -> "NNNLETTA":
        """Load an NNN-LETTA state saved by :meth:`save`."""
        source = Path(path).expanduser()
        with source.open("rb") as handle:
            payload = pickle.load(handle)
        return cls.from_state_dict(payload)

    @classmethod
    def from_mps(cls, mps, *, dims=None, local_masks=None, seed=None) -> "NNNLETTA":
        """Embed an open-boundary MPS exactly into NNN-LETTA form."""
        if hasattr(mps, "to_order"):
            factors = [np.asarray(tensor) for tensor in mps.to_order(["lv", "p", "rv"]).factors]
        elif hasattr(mps, "factors"):
            factors = [np.asarray(tensor) for tensor in mps.factors]
        else:
            factors = [np.asarray(tensor) for tensor in mps]
        if len(factors) < cls.tensor_width:
            raise ValueError("NNNLETTA.from_mps needs at least three MPS sites.")
        for site, tensor in enumerate(factors):
            if tensor.ndim != 3:
                raise ValueError(f"MPS factor {site} must have shape (left, physical, right).")
            if site == 0 and tensor.shape[0] != 1:
                raise ValueError("first MPS factor must have left bond dimension 1.")
            if site == len(factors) - 1 and tensor.shape[2] != 1:
                raise ValueError("last MPS factor must have right bond dimension 1.")
            if site and factors[site - 1].shape[2] != tensor.shape[0]:
                raise ValueError(f"MPS bond mismatch between factors {site - 1} and {site}.")
        inferred_dims = tuple(int(tensor.shape[1]) for tensor in factors)
        if dims is not None and tuple(int(dim) for dim in dims) != inferred_dims:
            raise ValueError("provided dims do not match MPS physical dimensions.")
        dtype = np.result_type(*[tensor.dtype for tensor in factors])
        tensors = []
        for site in range(len(factors) - cls.tensor_width):
            core = np.asarray(factors[site], dtype=dtype)
            shape = (
                core.shape[0],
                inferred_dims[site],
                inferred_dims[site + 1],
                inferred_dims[site + 2],
                core.shape[2],
            )
            tensor = np.broadcast_to(core[:, :, None, None, :], shape).copy()
            tensors.append(tensor)
        left = len(factors) - cls.tensor_width
        tail = np.einsum(
            "asb,btc,cud->astud",
            np.asarray(factors[left], dtype=dtype),
            np.asarray(factors[left + 1], dtype=dtype),
            np.asarray(factors[left + 2], dtype=dtype),
            optimize=True,
        )
        tensors.append(tail)
        bond_dim = max(max(tensor.shape[0], tensor.shape[-1]) for tensor in tensors)
        return cls(inferred_dims, bond_dim=bond_dim, tensors=tensors, local_masks=local_masks, seed=seed)

    def _matrix_for_config(self, tensor_index: int, config) -> np.ndarray:
        tensor_index = int(tensor_index)
        tensor = self.tensors[tensor_index]
        phys = tuple(int(config[tensor_index + offset]) for offset in range(self.tensor_width))
        return tensor[(slice(None),) + phys + (slice(None),)]

    def _amplitude(self, config) -> complex:
        vec = np.ones(1, dtype=np.result_type(*[tensor.dtype for tensor in self.tensors]))
        for tensor_index in range(self.nlocal_tensors):
            vec = vec @ self._matrix_for_config(tensor_index, config)
        return vec[0]

    def state_vector(self) -> np.ndarray:
        dtype = np.result_type(*[tensor.dtype for tensor in self.tensors])
        psi = np.empty(int(np.prod(self.dims)), dtype=dtype)
        for flat, config in enumerate(np.ndindex(*self.dims)):
            psi[flat] = self._amplitude(config)
        return psi

    def norm(self) -> float:
        psi = self.state_vector()
        value = np.vdot(psi, psi).real
        return float(0.0 if -1.0e-12 < value < 0.0 else value)

    def normalize(self):
        norm = np.sqrt(self.norm())
        if norm < 1.0e-14:
            raise ValueError("Cannot normalize a numerically zero NNNLETTA state.")
        self.tensors[0] = self.tensors[0] / norm
        return self

    def _validate_mpo(self, mpo) -> list[np.ndarray]:
        mpo = [np.asarray(site) for site in mpo]
        if len(mpo) != self.nsites:
            raise ValueError("MPO length must match dims.")
        for site, tensor in enumerate(mpo):
            if tensor.ndim != 4:
                raise ValueError("each MPO tensor must have shape (left, right, bra, ket).")
            if tensor.shape[2] != self.dims[site] or tensor.shape[3] != self.dims[site]:
                raise ValueError(f"MPO tensor {site} physical dimensions do not match dims.")
            if site == 0 and tensor.shape[0] != 1:
                raise ValueError("first MPO tensor must have left bond dimension 1.")
            if site == self.nsites - 1 and tensor.shape[1] != 1:
                raise ValueError("last MPO tensor must have right bond dimension 1.")
            if site and mpo[site - 1].shape[1] != tensor.shape[0]:
                raise ValueError(f"MPO bond mismatch between sites {site - 1} and {site}.")
        return mpo

    def apply_mpo(self, mpo, vector) -> np.ndarray:
        """Apply an MPO to a dense product-basis vector."""
        mpo = self._validate_mpo(mpo)
        tmp = np.asarray(vector).reshape(self.dims)[None, ...]
        for site, operator in enumerate(mpo):
            nout = site
            rem_after = self.nsites - site - 1
            tmp = np.tensordot(tmp, operator, axes=([0, nout + 1], [0, 3]))
            right_axis = nout + rem_after
            current_output_axis = right_axis + 1
            order = [right_axis] + list(range(nout)) + [current_output_axis] + list(
                range(nout, nout + rem_after)
            )
            tmp = np.transpose(tmp, order)
        return tmp[0].reshape(-1)

    def expectation_mpo(self, mpo) -> float:
        mpo = self._validate_mpo(mpo)
        value = self._mpo_matrix_element(mpo)
        denom = self._identity_matrix_element()
        if abs(denom) < 1.0e-14:
            raise ValueError("State norm is numerically zero.")
        return float(np.real(value / denom))

    def expectation_product_operator(self, operators) -> complex:
        """Return the expectation value of a product of one-site operators."""
        operators = [np.asarray(operator) for operator in operators]
        if len(operators) != self.nsites:
            raise ValueError("number of operators must match dims.")
        mpo = []
        for site, operator in enumerate(operators):
            if operator.shape != (self.dims[site], self.dims[site]):
                raise ValueError(f"operator {site} shape does not match local dimension.")
            mpo.append(operator.reshape(1, 1, self.dims[site], self.dims[site]))
        return self.expectation_mpo(mpo)

    def identity_mpo(self) -> list[np.ndarray]:
        return [np.eye(dim, dtype=self.tensors[0].dtype).reshape(1, 1, dim, dim) for dim in self.dims]

    def _initial_left_environment(self, mpo) -> np.ndarray:
        dtype = np.result_type(*[tensor.dtype for tensor in self.tensors], *[site.dtype for site in mpo])
        return np.ones(
            (
                1,
                1,
                mpo[0].shape[0],
                self.dims[0],
                self.dims[0],
                self.dims[1],
                self.dims[1],
            ),
            dtype=dtype,
        )

    def _terminal_right_environment(self, mpo) -> np.ndarray:
        dtype = np.result_type(*[tensor.dtype for tensor in self.tensors], *[site.dtype for site in mpo])
        return np.ones(
            (
                1,
                1,
                mpo[-1].shape[1],
                self.dims[-2],
                self.dims[-2],
                self.dims[-1],
                self.dims[-1],
            ),
            dtype=dtype,
        )

    @staticmethod
    def _advance_left_environment(env, mpo_site, tensor):
        return np.einsum(
            "abmxyuv,mnxy,axupc,byvqd->cdnuvpq",
            env,
            mpo_site,
            tensor.conj(),
            tensor,
            optimize=True,
        )

    @staticmethod
    def _advance_right_environment(env, mpo_site, tensor):
        return np.einsum(
            "cdnuvpq,mnpq,axupc,byvqd->abmxyuv",
            env,
            mpo_site,
            tensor.conj(),
            tensor,
            optimize=True,
        )

    def _left_local_environments(self, mpo) -> list[np.ndarray]:
        mpo = self._validate_mpo(mpo)
        left = []
        env = self._initial_left_environment(mpo)
        left.append(env)
        for tensor_index in range(self.nlocal_tensors - 1):
            env = self._advance_left_environment(env, mpo[tensor_index], self.tensors[tensor_index])
            left.append(env)
        return left

    def _right_local_environments(self, mpo) -> list[np.ndarray]:
        mpo = self._validate_mpo(mpo)
        right = [None] * self.nlocal_tensors
        env = self._terminal_right_environment(mpo)
        right[-1] = env
        for tensor_index in range(self.nlocal_tensors - 1, 0, -1):
            mpo_site = tensor_index + self.tensor_width - 1
            env = self._advance_right_environment(env, mpo[mpo_site], self.tensors[tensor_index])
            right[tensor_index - 1] = env
        return right

    def _local_effective_from_environments(self, mpo, tensor_index: int, left_envs, right_envs) -> np.ndarray:
        tensor_index = int(tensor_index)
        shape = self.tensors[tensor_index].shape
        local = np.einsum(
            "abmijkl,mnij,nokl,opqr,cdpklqr->aikqcbjlrd",
            left_envs[tensor_index],
            mpo[tensor_index],
            mpo[tensor_index + 1],
            mpo[tensor_index + 2],
            right_envs[tensor_index],
            optimize=True,
        )
        return local.reshape(int(np.prod(shape)), int(np.prod(shape)))

    def _apply_local_effective_from_environments(self, mpo, tensor_index: int, left_envs, right_envs, vector):
        """Apply the range-2 local Hamiltonian without forming ``Heff``."""
        tensor_index = int(tensor_index)
        shape = self.tensors[tensor_index].shape
        theta = np.asarray(vector).reshape(shape)
        out = np.einsum(
            "abmijkl,mnij,nokl,opqr,cdpklqr,bjlrd->aikqc",
            left_envs[tensor_index],
            mpo[tensor_index],
            mpo[tensor_index + 1],
            mpo[tensor_index + 2],
            right_envs[tensor_index],
            theta,
            optimize=True,
        )
        return out.reshape(-1)

    def _mpo_matrix_element(self, mpo) -> complex:
        mpo = self._validate_mpo(mpo)
        env = self._initial_left_environment(mpo)
        for tensor_index, tensor in enumerate(self.tensors):
            env = self._advance_left_environment(env, mpo[tensor_index], tensor)
        return np.einsum(
            "abmxyuv,mnxy,nouv->",
            env,
            mpo[-2],
            mpo[-1],
            optimize=True,
        )

    def _identity_matrix_element(self) -> complex:
        return self._mpo_matrix_element(self.identity_mpo())

    def _prefix_vector(self, config, stop: int) -> np.ndarray:
        vec = np.ones(1, dtype=np.result_type(*[tensor.dtype for tensor in self.tensors]))
        for tensor_index in range(int(stop)):
            vec = vec @ self._matrix_for_config(tensor_index, config)
        return vec

    def _suffix_vector(self, config, start: int) -> np.ndarray:
        vec = np.ones(1, dtype=np.result_type(*[tensor.dtype for tensor in self.tensors]))
        for tensor_index in range(self.nlocal_tensors - 1, int(start) - 1, -1):
            vec = self._matrix_for_config(tensor_index, config) @ vec
        return vec

    def one_tensor_projector(self, tensor_index: int) -> np.ndarray:
        """Return the dense local projector for one tied tensor."""
        tensor_index = int(tensor_index)
        if tensor_index < 0 or tensor_index >= self.nlocal_tensors:
            raise IndexError("tensor_index out of range.")
        shape = self.tensors[tensor_index].shape
        projector = np.zeros((int(np.prod(self.dims)), int(np.prod(shape))), dtype=self.tensors[tensor_index].dtype)
        for row, config in enumerate(np.ndindex(*self.dims)):
            left = self._prefix_vector(config, tensor_index)
            right = self._suffix_vector(config, tensor_index + 1)
            phys = tuple(int(config[tensor_index + offset]) for offset in range(self.tensor_width))
            for left_bond in range(shape[0]):
                if left_bond >= left.size:
                    continue
                for right_bond in range(shape[-1]):
                    col = np.ravel_multi_index(
                        (left_bond,) + phys + (right_bond,),
                        shape,
                    )
                    projector[row, col] = left[left_bond] * right[right_bond]
        return projector

    def _support_indices(self, tensor_index: int):
        if self.local_masks is None:
            return None
        return np.flatnonzero(self.local_masks[int(tensor_index)].reshape(-1))

    @staticmethod
    def _restrict_matrix(matrix, support):
        if support is None:
            return matrix
        return matrix[np.ix_(support, support)]

    @staticmethod
    def _expand_supported_vector(vector, support, local_dim: int):
        if support is None:
            return vector
        full = np.zeros(int(local_dim), dtype=np.asarray(vector).dtype)
        full[support] = vector
        return full

    def _solve_tensor_mpo_dense(
        self,
        mpo,
        tensor_index: int,
        left,
        right,
        metric_left,
        metric_right,
        *,
        assume_identity_metric=False,
    ):
        heff = self._local_effective_from_environments(mpo, tensor_index, left, right)
        support = self._support_indices(tensor_index)
        local_dim = heff.shape[0]
        heff = self._restrict_matrix(heff, support)
        heff = 0.5 * (heff + heff.T.conj())
        if assume_identity_metric:
            energy, vector = _lowest_hermitian_eigenpair(heff)
        else:
            metric = self._local_effective_from_environments(
                self.identity_mpo(),
                tensor_index,
                metric_left,
                metric_right,
            )
            energy, vector = _lowest_generalized_eigenpair(
                heff,
                self._restrict_matrix(metric, support),
            )
        return energy, self._expand_supported_vector(vector, support, local_dim)

    def _solve_tensor_mpo_matrix_free(
        self,
        mpo,
        tensor_index: int,
        left,
        right,
        metric_left,
        metric_right,
        *,
        tol=1.0e-9,
        maxiter=None,
        assume_identity_metric=False,
    ):
        shape = self.tensors[tensor_index].shape
        local_dim = int(np.prod(shape))
        support = self._support_indices(tensor_index)
        active_dim = local_dim if support is None else int(support.size)
        dtype = np.result_type(
            self.tensors[tensor_index].dtype,
            *[site.dtype for site in mpo],
            complex,
        )

        def matvec(vector):
            full_vector = self._expand_supported_vector(np.asarray(vector), support, local_dim)
            return self._apply_local_effective_from_environments(
                mpo,
                tensor_index,
                left,
                right,
                full_vector,
            ).reshape(-1)[slice(None) if support is None else support].astype(dtype, copy=False)

        if assume_identity_metric:
            metric = None
            basis = None
            reduced_dim = active_dim
        else:
            metric = self._local_effective_from_environments(
                self.identity_mpo(),
                tensor_index,
                metric_left,
                metric_right,
            )
            metric = self._restrict_matrix(metric, support)
            metric = 0.5 * (metric + metric.T.conj())
            basis = _metric_basis(metric)
            reduced_dim = basis.shape[1]

        def reduced_matvec(vector):
            if basis is None:
                return matvec(vector)
            full_vector = basis @ np.asarray(vector)
            return basis.conj().T @ matvec(full_vector)

        v0 = np.asarray(self.tensors[tensor_index]).reshape(-1)
        if support is not None:
            v0 = v0[support]
        reduced_v0 = v0 if basis is None else basis.conj().T @ (metric @ v0)
        try:
            if reduced_dim <= 2:
                reduced_matrix = np.column_stack(
                    [reduced_matvec(unit) for unit in np.eye(reduced_dim, dtype=dtype)]
                )
                reduced_matrix = 0.5 * (reduced_matrix + reduced_matrix.T.conj())
                evals, evecs = np.linalg.eigh(reduced_matrix)
                vector = evecs[:, 0] if basis is None else basis @ evecs[:, 0]
                return float(np.real(evals[0])), self._expand_supported_vector(vector, support, local_dim)
            reduced_operator = LinearOperator(
                (reduced_dim, reduced_dim),
                matvec=reduced_matvec,
                dtype=dtype if basis is None else np.result_type(dtype, basis.dtype),
            )
            if np.linalg.norm(reduced_v0) < 1.0e-14:
                reduced_v0 = None
            evals, evecs = eigsh(
                reduced_operator,
                k=1,
                which="SA",
                tol=float(tol),
                maxiter=maxiter,
                v0=reduced_v0,
            )
            vector = evecs[:, 0] if basis is None else basis @ evecs[:, 0]
            return float(np.real(evals[0])), self._expand_supported_vector(vector, support, local_dim)
        except Exception:
            heff = self._local_effective_from_environments(mpo, tensor_index, left, right)
            heff = self._restrict_matrix(heff, support)
            heff = 0.5 * (heff + heff.T.conj())
            if assume_identity_metric:
                energy, vector = _lowest_hermitian_eigenpair(heff)
            else:
                energy, vector = _lowest_generalized_eigenpair(heff, metric)
            return energy, self._expand_supported_vector(vector, support, local_dim)

    def _local_metric_is_identity(
        self,
        tensor_index: int,
        metric_left,
        metric_right,
        *,
        atol=1.0e-10,
    ) -> bool:
        """Check the active range-2 norm blocks without building the full metric."""
        tensor_index = int(tensor_index)
        left = metric_left[tensor_index]
        right = metric_right[tensor_index]
        if left.shape[2] != 1 or right.shape[2] != 1:
            return False
        mask = None if self.local_masks is None else self.local_masks[tensor_index]
        for s0 in range(self.dims[tensor_index]):
            for s1 in range(self.dims[tensor_index + 1]):
                left_block = left[:, :, 0, s0, s0, s1, s1]
                for s2 in range(self.dims[tensor_index + 2]):
                    right_block = right[:, :, 0, s1, s1, s2, s2]
                    if mask is None:
                        if not np.allclose(
                            left_block,
                            np.eye(left_block.shape[0], dtype=left_block.dtype),
                            rtol=0.0,
                            atol=float(atol),
                        ):
                            return False
                        if not np.allclose(
                            right_block,
                            np.eye(right_block.shape[0], dtype=right_block.dtype),
                            rtol=0.0,
                            atol=float(atol),
                        ):
                            return False
                        continue
                    coords = np.argwhere(mask[:, s0, s1, s2, :])
                    if not coords.size:
                        continue
                    left_indices = coords[:, 0]
                    right_indices = coords[:, 1]
                    block = (
                        left_block[left_indices[:, None], left_indices[None, :]]
                        * right_block[right_indices[:, None], right_indices[None, :]]
                    )
                    if not np.allclose(
                        block,
                        np.eye(coords.shape[0], dtype=block.dtype),
                        rtol=0.0,
                        atol=float(atol),
                    ):
                        return False
        return True

    def optimize_tensor_mpo(
        self,
        mpo,
        tensor_index: int,
        *,
        local_solver="auto",
        matrix_free_threshold=4096,
        matrix_free_tol=1.0e-9,
        matrix_free_maxiter=None,
        identity_metric=False,
        metric_tol=1.0e-10,
    ) -> dict:
        """Optimize one local tensor using range-2 MPO environments."""
        mpo = self._validate_mpo(mpo)
        tensor_index = int(tensor_index)
        if tensor_index < 0 or tensor_index >= self.nlocal_tensors:
            raise IndexError("tensor_index out of range.")
        left = self._left_local_environments(mpo)
        right = self._right_local_environments(mpo)

        identity = self.identity_mpo()
        metric_left = self._left_local_environments(identity)
        metric_right = self._right_local_environments(identity)
        use_identity_metric = bool(identity_metric) and self._local_metric_is_identity(
            tensor_index,
            metric_left,
            metric_right,
            atol=metric_tol,
        )
        local_dim = int(np.prod(self.tensors[tensor_index].shape))
        solver = str(local_solver).lower()
        if solver not in {"auto", "dense", "matrix_free"}:
            raise ValueError("local_solver must be 'auto', 'dense', or 'matrix_free'.")
        use_matrix_free = solver == "matrix_free" or (
            solver == "auto"
            and matrix_free_threshold is not None
            and local_dim >= int(matrix_free_threshold)
        )
        if use_matrix_free:
            energy, vector = self._solve_tensor_mpo_matrix_free(
                mpo,
                tensor_index,
                left,
                right,
                metric_left,
                metric_right,
                tol=matrix_free_tol,
                maxiter=matrix_free_maxiter,
                assume_identity_metric=use_identity_metric,
            )
        else:
            energy, vector = self._solve_tensor_mpo_dense(
                mpo,
                tensor_index,
                left,
                right,
                metric_left,
                metric_right,
                assume_identity_metric=use_identity_metric,
            )
        tensor = vector.reshape(self.tensors[tensor_index].shape)
        if self.local_masks is not None:
            tensor = np.where(self.local_masks[tensor_index], tensor, 0.0)
        self.tensors[tensor_index] = tensor
        return {
            "tensor": tensor_index,
            "local_energy": float(np.real(energy)),
            "identity_metric": bool(use_identity_metric),
        }

    def _virtual_bond_groups(self, bond: int) -> list[np.ndarray]:
        """Return virtual channels with compatible mask support."""
        bond = int(bond)
        left = self.tensors[bond]
        right = self.tensors[bond + 1]
        shared = min(left.shape[-1], right.shape[0])
        if self.local_masks is None:
            return [np.arange(shared, dtype=np.int64)]
        left_mask = self.local_masks[bond]
        right_mask = self.local_masks[bond + 1]
        groups = {}
        for index in range(shared):
            left_signature = np.ascontiguousarray(left_mask[..., index]).tobytes()
            right_signature = np.ascontiguousarray(right_mask[index, ...]).tobytes()
            groups.setdefault((left_signature, right_signature), []).append(index)
        return [np.asarray(indices, dtype=np.int64) for indices in groups.values() if indices]

    def _conditional_bond_matrices(self, bond, s1, s2, group):
        left = self.tensors[int(bond)]
        right = self.tensors[int(bond) + 1]
        group = np.asarray(group, dtype=np.int64)
        left_block = left[:, :, int(s1), int(s2), :][:, :, group]
        right_block = right[:, int(s1), int(s2), :, :][group, :, :]
        return (
            left_block.reshape(-1, group.size),
            right_block.reshape(group.size, -1),
        )

    def compress_conditional_bond(
        self,
        bond: int,
        *,
        direction="lr",
        rtol=1.0e-12,
        atol=0.0,
        max_bond_dim=None,
    ) -> dict:
        """Compress a bond independently for every shared physical pair."""
        bond = int(bond)
        if bond < 0 or bond + 1 >= self.nlocal_tensors:
            raise IndexError("bond out of range.")
        direction = str(direction).lower()
        if direction not in {"lr", "rl", "balanced"}:
            raise ValueError("direction must be 'lr', 'rl', or 'balanced'.")
        rtol = float(rtol)
        atol = float(atol)
        if rtol < 0.0 or atol < 0.0:
            raise ValueError("rtol and atol must be nonnegative.")
        if max_bond_dim is not None:
            max_bond_dim = int(max_bond_dim)
            if max_bond_dim < 1:
                raise ValueError("max_bond_dim must be positive when provided.")

        left = self.tensors[bond]
        right = self.tensors[bond + 1]
        old_dim = int(left.shape[-1])
        if right.shape[0] != old_dim:
            raise ValueError("neighboring tensors have incompatible virtual bond dimensions.")
        shared_dims = (self.dims[bond + 1], self.dims[bond + 2])
        pairs = tuple(np.ndindex(*shared_dims))
        groups = self._virtual_bond_groups(bond)
        records = []
        numeric_ranks = []
        total_weight = 0.0
        for group in groups:
            group_records = []
            group_ranks = []
            for s1, s2 in pairs:
                left_matrix, right_matrix = self._conditional_bond_matrices(
                    bond,
                    s1,
                    s2,
                    group,
                )
                u, values, vh = np.linalg.svd(left_matrix @ right_matrix, full_matrices=False)
                scale = float(values[0]) if values.size else 0.0
                rank = int(np.count_nonzero(values > max(atol, rtol * scale)))
                group_records.append((u, values, vh, rank))
                group_ranks.append(rank)
                total_weight += float(np.sum(np.abs(values) ** 2))
            records.append(group_records)
            numeric_ranks.append(group_ranks)

        capacities = [max(ranks, default=0) for ranks in numeric_ranks]
        exact_dim = int(sum(capacities))
        if max_bond_dim is not None and exact_dim > max_bond_dim:
            capacities = [0] * len(records)
            for _ in range(max_bond_dim):
                gains = []
                for group_index, group_records in enumerate(records):
                    channel = capacities[group_index]
                    gains.append(
                        sum(
                            float(abs(values[channel]) ** 2)
                            for _u, values, _vh, rank in group_records
                            if channel < rank
                        )
                    )
                selected = int(np.argmax(gains)) if gains else 0
                if not gains or gains[selected] <= 0.0:
                    break
                capacities[selected] += 1
        new_dim = int(sum(capacities))
        if new_dim < 1:
            raise ValueError("conditional compression removed every virtual channel.")

        dtype = np.result_type(left.dtype, right.dtype)
        new_left = np.zeros(left.shape[:-1] + (new_dim,), dtype=dtype)
        new_right = np.zeros((new_dim,) + right.shape[1:], dtype=dtype)
        new_left_mask = np.zeros(new_left.shape, dtype=bool)
        new_right_mask = np.zeros(new_right.shape, dtype=bool)
        old_left_mask = None if self.local_masks is None else self.local_masks[bond]
        old_right_mask = None if self.local_masks is None else self.local_masks[bond + 1]
        retained_ranks = np.zeros(shared_dims, dtype=np.int64)
        discarded_weight = 0.0
        cursor = 0
        for group, group_records, capacity in zip(groups, records, capacities):
            group = np.asarray(group, dtype=np.int64)
            capacity = int(capacity)
            if capacity == 0:
                for _u, values, _vh, _rank in group_records:
                    discarded_weight += float(np.sum(np.abs(values) ** 2))
                continue
            left_allowed = (
                np.ones(left.shape[:-1], dtype=bool)
                if old_left_mask is None
                else np.any(old_left_mask[..., group], axis=-1)
            )
            right_allowed = (
                np.ones(right.shape[1:], dtype=bool)
                if old_right_mask is None
                else np.any(old_right_mask[group, ...], axis=0)
            )
            for (s1, s2), (u, values, vh, numeric_rank) in zip(pairs, group_records):
                rank = min(int(numeric_rank), capacity)
                retained_ranks[s1, s2] += rank
                discarded_weight += float(np.sum(np.abs(values[rank:]) ** 2))
                if rank == 0:
                    continue
                if direction == "lr":
                    left_factor = u[:, :rank]
                    right_factor = values[:rank, None] * vh[:rank, :]
                elif direction == "rl":
                    left_factor = u[:, :rank] * values[None, :rank]
                    right_factor = vh[:rank, :]
                else:
                    roots = np.sqrt(values[:rank])
                    left_factor = u[:, :rank] * roots[None, :]
                    right_factor = roots[:, None] * vh[:rank, :]
                new_left[:, :, s1, s2, cursor : cursor + rank] = left_factor.reshape(
                    left.shape[0],
                    left.shape[1],
                    rank,
                )
                new_right[cursor : cursor + rank, s1, s2, :, :] = right_factor.reshape(
                    rank,
                    right.shape[3],
                    right.shape[4],
                )
                new_left_mask[:, :, s1, s2, cursor : cursor + rank] = left_allowed[
                    :, :, s1, s2, None
                ]
                new_right_mask[cursor : cursor + rank, s1, s2, :, :] = right_allowed[
                    s1, s2, :, :
                ]
            cursor += capacity

        masks = (
            [np.ones(tensor.shape, dtype=bool) for tensor in self.tensors]
            if self.local_masks is None
            else [mask.copy() for mask in self.local_masks]
        )
        self.tensors[bond] = new_left
        self.tensors[bond + 1] = new_right
        masks[bond] = new_left_mask
        masks[bond + 1] = new_right_mask
        self.local_masks = None if all(np.all(mask) for mask in masks) else masks
        self._apply_local_masks()
        self.bond_dim = max(
            [1]
            + [int(tensor.shape[-1]) for tensor in self.tensors[:-1]]
            + [int(tensor.shape[0]) for tensor in self.tensors[1:]]
        )
        self.converged = False
        relative_discarded = 0.0 if total_weight <= 0.0 else discarded_weight / total_weight
        return {
            "bond": bond,
            "old_dim": old_dim,
            "new_dim": new_dim,
            "exact_dim": exact_dim,
            "sector_ranks": tuple(int(rank) for rank in retained_ranks.reshape(-1)),
            "discarded_weight": float(discarded_weight),
            "relative_discarded_weight": float(relative_discarded),
            "truncated": bool(discarded_weight > max(atol * atol, 1.0e-30)),
        }

    def compress_conditional_bonds(
        self,
        *,
        direction="lr",
        rtol=1.0e-12,
        atol=0.0,
        max_bond_dim=None,
    ) -> list[dict]:
        """Compress all range-2 bonds in sweep order."""
        direction = str(direction).lower()
        if direction not in {"lr", "rl"}:
            raise ValueError("direction must be 'lr' or 'rl'.")
        bonds = list(range(self.nlocal_tensors - 1))
        if direction == "rl":
            bonds.reverse()
        return [
            self.compress_conditional_bond(
                bond,
                direction=direction,
                rtol=rtol,
                atol=atol,
                max_bond_dim=max_bond_dim,
            )
            for bond in bonds
        ]

    def canonicalize_conditional_bond(self, bond: int, *, direction="lr", normalize=False):
        """QR-gauge one virtual bond for each shared physical pair.

        For bond ``i`` the two neighboring tensors share
        ``(sigma_{i+1}, sigma_{i+2})``.  In left-to-right mode the left tensor
        columns are orthonormal for each shared pair; in right-to-left mode the
        right tensor rows are orthonormal.  The represented state is preserved.
        """
        bond = int(bond)
        if bond < 0 or bond + 1 >= self.nlocal_tensors:
            raise IndexError("bond out of range.")
        direction = str(direction).lower()
        if direction not in {"lr", "rl"}:
            raise ValueError("direction must be 'lr' or 'rl'.")
        dtype = np.result_type(self.tensors[bond].dtype, self.tensors[bond + 1].dtype)
        left = self.tensors[bond].astype(dtype, copy=True)
        right = self.tensors[bond + 1].astype(dtype, copy=True)
        shared_dims = (self.dims[bond + 1], self.dims[bond + 2])
        shared_bond = left.shape[-1]
        if right.shape[0] != shared_bond:
            raise ValueError("neighboring tensors have incompatible virtual bond dimensions.")

        for s1 in range(shared_dims[0]):
            for s2 in range(shared_dims[1]):
                left_block = left[:, :, s1, s2, :]
                right_block = right[:, s1, s2, :, :]
                left_matrix = left_block.reshape(left_block.shape[0] * left_block.shape[1], shared_bond)
                right_matrix = right_block.reshape(shared_bond, right_block.shape[1] * right_block.shape[2])
                if self.local_masks is None:
                    if direction == "lr":
                        if left_matrix.shape[0] < shared_bond:
                            raise ValueError("left conditional block has too few rows for QR gauge.")
                        q, gauge = np.linalg.qr(left_matrix, mode="reduced")
                        left[:, :, s1, s2, :] = q.reshape(left_block.shape)
                        right[:, s1, s2, :, :] = (gauge @ right_matrix).reshape(right_block.shape)
                    else:
                        if right_matrix.shape[1] < shared_bond:
                            raise ValueError("right conditional block has too few columns for QR gauge.")
                        q, gauge = np.linalg.qr(right_matrix.T, mode="reduced")
                        left[:, :, s1, s2, :] = (left_matrix @ gauge.T).reshape(left_block.shape)
                        right[:, s1, s2, :, :] = q.T.reshape(right_block.shape)
                    continue

                left_mask = self.local_masks[bond][:, :, s1, s2, :].reshape(left_matrix.shape)
                right_mask = self.local_masks[bond + 1][:, s1, s2, :, :].reshape(right_matrix.shape)
                groups = {}
                for column in range(shared_bond):
                    row_support = tuple(np.flatnonzero(left_mask[:, column]).tolist())
                    col_support = tuple(np.flatnonzero(right_mask[column, :]).tolist())
                    if not row_support or not col_support:
                        continue
                    groups.setdefault((row_support, col_support), []).append(column)

                for (row_support, col_support), columns in groups.items():
                    rows = np.asarray(row_support, dtype=int)
                    cols = np.asarray(col_support, dtype=int)
                    alphas = np.asarray(columns, dtype=int)
                    can_left_qr = rows.size >= alphas.size
                    can_right_qr = cols.size >= alphas.size
                    if direction == "lr" and can_left_qr:
                        q, gauge = np.linalg.qr(left_matrix[np.ix_(rows, alphas)], mode="reduced")
                        left_matrix[np.ix_(rows, alphas)] = q[:, : alphas.size]
                        right_matrix[alphas, :] = gauge[: alphas.size, : alphas.size] @ right_matrix[alphas, :]
                    elif direction == "rl" and can_right_qr:
                        q, gauge = np.linalg.qr(right_matrix[np.ix_(alphas, cols)].T, mode="reduced")
                        left_matrix[:, alphas] = left_matrix[:, alphas] @ gauge[: alphas.size, : alphas.size].T
                        right_matrix[np.ix_(alphas, cols)] = q[:, : alphas.size].T
                    elif can_right_qr:
                        q, gauge = np.linalg.qr(right_matrix[np.ix_(alphas, cols)].T, mode="reduced")
                        left_matrix[:, alphas] = left_matrix[:, alphas] @ gauge[: alphas.size, : alphas.size].T
                        right_matrix[np.ix_(alphas, cols)] = q[:, : alphas.size].T
                    elif can_left_qr:
                        q, gauge = np.linalg.qr(left_matrix[np.ix_(rows, alphas)], mode="reduced")
                        left_matrix[np.ix_(rows, alphas)] = q[:, : alphas.size]
                        right_matrix[alphas, :] = gauge[: alphas.size, : alphas.size] @ right_matrix[alphas, :]
                    else:
                        raise ValueError("masked conditional block has too little support for QR gauge.")
                left[:, :, s1, s2, :] = left_matrix.reshape(left_block.shape)
                right[:, s1, s2, :, :] = right_matrix.reshape(right_block.shape)

        self.tensors[bond] = left
        self.tensors[bond + 1] = right
        self._apply_local_masks()
        if normalize:
            self.normalize()
        return self

    def canonicalize_conditional_center(self, tensor_index: int, *, normalize=True):
        """Build a physical-pair-conditioned mixed gauge around one tensor."""
        tensor_index = int(tensor_index)
        if tensor_index < 0 or tensor_index >= self.nlocal_tensors:
            raise IndexError("tensor_index out of range.")
        for bond in range(tensor_index):
            self.canonicalize_conditional_bond(bond, direction="lr", normalize=False)
        for bond in reversed(range(tensor_index, self.nlocal_tensors - 1)):
            self.canonicalize_conditional_bond(bond, direction="rl", normalize=False)
        if normalize:
            self.normalize()
        return self

    def sweep_mpo(
        self,
        mpo,
        direction="lr",
        *,
        gauge="conditional",
        local_solver="auto",
        matrix_free_threshold=4096,
        matrix_free_tol=1.0e-9,
        matrix_free_maxiter=None,
        identity_metric=None,
        metric_tol=1.0e-10,
        adapt_bonds=None,
        compress_rtol=1.0e-12,
        compress_atol=0.0,
        max_bond_dim=None,
    ) -> list[dict]:
        mpo = self._validate_mpo(mpo)
        direction = str(direction).lower()
        if direction not in {"lr", "rl"}:
            raise ValueError("direction must be 'lr' or 'rl'.")
        if gauge not in {None, "conditional"}:
            raise ValueError("NNNLETTA currently supports only gauge='conditional' or None.")
        if identity_metric is None:
            identity_metric = gauge == "conditional"
        identity_metric = bool(identity_metric)
        if identity_metric and gauge != "conditional":
            raise ValueError("identity_metric requires gauge='conditional'.")
        if adapt_bonds is None:
            adapt_bonds = identity_metric
        adapt_bonds = bool(adapt_bonds)
        if adapt_bonds and gauge != "conditional":
            raise ValueError("adapt_bonds requires gauge='conditional'.")
        precompression = []
        if adapt_bonds:
            precompression = self.compress_conditional_bonds(
                direction="rl" if direction == "lr" else "lr",
                rtol=compress_rtol,
                atol=compress_atol,
                max_bond_dim=max_bond_dim,
            )
        if identity_metric:
            start = 0 if direction == "lr" else self.nlocal_tensors - 1
            try:
                self.canonicalize_conditional_center(start, normalize=False)
            except ValueError:
                if adapt_bonds:
                    raise
        indices = range(self.nlocal_tensors)
        if direction == "rl":
            indices = reversed(list(indices))
        updates = []
        for tensor_index in indices:
            update = self.optimize_tensor_mpo(
                mpo,
                tensor_index,
                local_solver=local_solver,
                matrix_free_threshold=matrix_free_threshold,
                matrix_free_tol=matrix_free_tol,
                matrix_free_maxiter=matrix_free_maxiter,
                identity_metric=identity_metric,
                metric_tol=metric_tol,
            )
            if not updates and precompression:
                update["precompression"] = precompression
            updates.append(update)
            if gauge == "conditional":
                if direction == "lr" and tensor_index + 1 < self.nlocal_tensors:
                    if adapt_bonds:
                        update["compression"] = self.compress_conditional_bond(
                            tensor_index,
                            direction="lr",
                            rtol=compress_rtol,
                            atol=compress_atol,
                            max_bond_dim=max_bond_dim,
                        )
                    try:
                        self.canonicalize_conditional_bond(tensor_index, direction="lr")
                    except ValueError:
                        if adapt_bonds:
                            raise
                elif direction == "rl" and tensor_index > 0:
                    if adapt_bonds:
                        update["compression"] = self.compress_conditional_bond(
                            tensor_index - 1,
                            direction="rl",
                            rtol=compress_rtol,
                            atol=compress_atol,
                            max_bond_dim=max_bond_dim,
                        )
                    try:
                        self.canonicalize_conditional_bond(tensor_index - 1, direction="rl")
                    except ValueError:
                        if adapt_bonds:
                            raise
        self.normalize()
        return updates

    def run(
        self,
        mpo,
        *,
        nsweeps=4,
        start_direction="lr",
        alternate=True,
        tol=1.0e-10,
        gauge="conditional",
        local_solver="auto",
        matrix_free_threshold=4096,
        matrix_free_tol=1.0e-9,
        matrix_free_maxiter=None,
        identity_metric=None,
        metric_tol=1.0e-10,
        adapt_bonds=None,
        compress_rtol=1.0e-12,
        compress_atol=0.0,
        max_bond_dim=None,
        verbose=0,
    ):
        if int(nsweeps) < 1:
            raise ValueError("nsweeps must be positive.")
        direction = str(start_direction).lower()
        previous = None
        self.history = []
        self.converged = False
        for sweep in range(int(nsweeps)):
            updates = self.sweep_mpo(
                mpo,
                direction=direction,
                gauge=gauge,
                local_solver=local_solver,
                matrix_free_threshold=matrix_free_threshold,
                matrix_free_tol=matrix_free_tol,
                matrix_free_maxiter=matrix_free_maxiter,
                identity_metric=identity_metric,
                metric_tol=metric_tol,
                adapt_bonds=adapt_bonds,
                compress_rtol=compress_rtol,
                compress_atol=compress_atol,
                max_bond_dim=max_bond_dim,
            )
            energy = self.expectation_mpo(mpo)
            delta = None if previous is None else abs(energy - previous)
            self.history.append(
                {
                    "sweep": sweep,
                    "direction": direction,
                    "energy": float(energy),
                    "delta_energy": None if delta is None else float(delta),
                    "gauge": gauge,
                    "updates": updates,
                }
            )
            if int(verbose) > 0:
                print(
                    f"nnn-letta sweep {sweep:>2} | dir={direction} | "
                    f"E={energy:.12g} | dE={'-' if delta is None else f'{delta:.3e}'}",
                    flush=True,
                )
            if delta is not None and delta <= float(tol):
                self.converged = True
                break
            previous = energy
            if alternate:
                direction = "rl" if direction == "lr" else "lr"
        self.energy = self.history[-1]["energy"]
        return self


__all__ = ["NNNLETTA"]
