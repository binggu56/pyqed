"""Optional PyTorch kernels for window-2 LETTA real-time TDVP."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import weakref

import numpy as np
from opt_einsum import contract_expression
from scipy.linalg import eigh_tridiagonal

try:
    import torch
except ImportError as error:  # pragma: no cover - exercised without the extra
    raise ImportError(
        "PyTorch LETTA support requires the optional 'torch' dependency. "
        "Install pyqed[torch]."
    ) from error

from .core import LETTA
from .tdvp import Window2Hamiltonian


_OPERATOR_CACHE = weakref.WeakKeyDictionary()


@lru_cache(maxsize=256)
def _cached_expression(expression, shapes):
    return contract_expression(expression, *shapes, optimize="greedy")


def _contract(expression, *operands):
    shapes = tuple(tuple(operand.shape) for operand in operands)
    return _cached_expression(expression, shapes)(*operands, backend="torch")


def _torch_dtype(dtype):
    if dtype is None:
        return torch.complex128
    if isinstance(dtype, torch.dtype):
        result = dtype
    else:
        numpy_dtype = np.dtype(dtype)
        mapping = {
            np.dtype(np.complex64): torch.complex64,
            np.dtype(np.complex128): torch.complex128,
        }
        result = mapping.get(numpy_dtype)
    if result not in {torch.complex64, torch.complex128}:
        raise TypeError("Real-time LETTA TDVP needs complex64 or complex128 tensors.")
    return result


def torch_backend_capabilities():
    """Return runtime device availability for the optional Torch backend."""
    mps_backend = getattr(torch.backends, "mps", None)
    return {
        "torch_version": torch.__version__,
        "cpu": True,
        "cuda": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "mps": bool(mps_backend is not None and mps_backend.is_available()),
        "num_threads": int(torch.get_num_threads()),
    }


@dataclass
class TorchWindow2State:
    """Device-resident terminal-form window-2 LETTA state."""

    cores: list[torch.Tensor]
    dims: tuple[int, ...]
    max_bond: int | None = None

    def __post_init__(self):
        self.dims = tuple(int(value) for value in self.dims)
        if len(self.cores) != len(self.dims):
            raise ValueError("State core count and physical dimensions differ.")
        for site, core in enumerate(self.cores):
            expected = 3 if site == len(self.cores) - 1 else 4
            if core.ndim != expected:
                raise ValueError(f"State core {site} must have rank {expected}.")
            if not core.is_complex():
                raise TypeError("Real-time LETTA TDVP requires complex tensors.")
        devices = {core.device for core in self.cores}
        dtypes = {core.dtype for core in self.cores}
        if len(devices) != 1 or len(dtypes) != 1:
            raise ValueError("All Torch LETTA cores must share a device and dtype.")

    @property
    def device(self):
        return self.cores[0].device

    @property
    def dtype(self):
        return self.cores[0].dtype

    @property
    def ranks(self):
        return tuple(int(core.shape[-1]) for core in self.cores[:-1])

    @classmethod
    def from_letta(cls, state, *, device=None, dtype=None):
        if not isinstance(state, LETTA) or not state.has_terminal_tensor:
            raise TypeError("A terminal-form LETTA state is required.")
        target_device = torch.device("cpu" if device is None else device)
        target_dtype = _torch_dtype(dtype)
        cores = [
            torch.as_tensor(
                np.array(tensor, dtype=complex, copy=True),
                dtype=target_dtype,
                device=target_device,
            )
            for tensor in state.tensors[:-1]
        ]
        terminal = np.array(state.tensors[-1], dtype=complex, copy=True).T[:, :, None]
        cores.append(torch.as_tensor(terminal, dtype=target_dtype, device=target_device))
        return cls(cores, tuple(state.dims), int(state.bond_dim))

    def clone(self):
        return TorchWindow2State(
            [core.clone() for core in self.cores], self.dims, self.max_bond
        )

    def to(self, *, device=None, dtype=None):
        target_device = self.device if device is None else torch.device(device)
        target_dtype = self.dtype if dtype is None else _torch_dtype(dtype)
        if target_device == self.device and target_dtype == self.dtype:
            return self
        return TorchWindow2State(
            [core.to(device=target_device, dtype=target_dtype) for core in self.cores],
            self.dims,
            self.max_bond,
        )

    def to_letta(self):
        def numpy_copy(tensor):
            return (
                tensor.detach().resolve_conj().resolve_neg().cpu().numpy().copy()
            )

        arrays = [numpy_copy(core) for core in self.cores[:-1]]
        arrays.append(numpy_copy(self.cores[-1][:, :, 0].T))
        maximum = max(self.ranks, default=1)
        state = LETTA(
            None,
            self.dims,
            bond_dim=max(maximum, int(self.max_bond or 1)),
            tensors=arrays,
        )
        state.tensors = arrays
        state.local_masks = [None] * len(arrays)
        return state

    def system_reduced_density_matrix(self, *, return_info=False):
        return torch_system_reduced_density_matrix(self, return_info=return_info)

    def site_reduced_density_matrix(self, site, *, return_info=False):
        return torch_site_reduced_density_matrix(
            self, site, return_info=return_info
        )


@dataclass(frozen=True)
class TorchWindow2Hamiltonian:
    """Device-resident window-2 nearest-neighbour Hamiltonian."""

    cores: tuple[torch.Tensor, ...]
    dims: tuple[int, ...]
    transitions: tuple[tuple[tuple[int, int, torch.Tensor], ...], ...] | None
    factors: tuple[
        tuple[tuple[int, int, torch.Tensor, torch.Tensor | None], ...], ...
    ] | None

    @classmethod
    def from_numpy(
        cls, operator, *, device=None, dtype=None, factorize=True, cache=True
    ):
        if not isinstance(operator, Window2Hamiltonian):
            raise TypeError("operator must be a Window2Hamiltonian.")
        target_device = torch.device("cpu" if device is None else device)
        target_dtype = _torch_dtype(dtype)
        key = (str(target_device), target_dtype, bool(factorize))
        if cache:
            cached = _OPERATOR_CACHE.get(operator, {}).get(key)
            if cached is not None:
                return cached
        cores = tuple(
            torch.as_tensor(
                np.array(core, dtype=complex, copy=True),
                dtype=target_dtype,
                device=target_device,
            )
            for core in operator.cores
        )
        transitions = None
        factors = None
        if operator.transitions is not None:
            transitions = tuple(
                tuple(
                    (
                        int(source),
                        int(target),
                        torch.as_tensor(
                            np.array(value, dtype=complex, copy=True),
                            dtype=target_dtype,
                            device=target_device,
                        ),
                    )
                    for source, target, value in site
                )
                for site in operator.transitions
            )
            if factorize and operator.factors is not None:
                factors = tuple(
                    tuple(
                        (
                            int(source),
                            int(target),
                            torch.as_tensor(
                                np.array(left, dtype=complex, copy=True),
                                dtype=target_dtype,
                                device=target_device,
                            ),
                            None if right is None else torch.as_tensor(
                                np.array(right, dtype=complex, copy=True),
                                dtype=target_dtype,
                                device=target_device,
                            ),
                        )
                        for source, target, left, right in site
                    )
                    for site in operator.factors
                )
            elif factorize:
                factored_sites = []
                with torch.no_grad():
                    for site in transitions:
                        factored = []
                        for source, target, value in site:
                            if value.ndim == 2:
                                factored.append((source, target, value, None))
                                continue
                            p, q, r, s = value.shape
                            matrix = value.permute(0, 2, 1, 3).reshape(
                                p * r, q * s
                            )
                            u, singular_values, vh = torch.linalg.svd(
                                matrix, full_matrices=False
                            )
                            threshold = (
                                torch.finfo(singular_values.dtype).eps
                                * max(matrix.shape)
                                * singular_values[0]
                            )
                            rank = max(
                                1,
                                int(
                                    torch.count_nonzero(
                                        singular_values > threshold
                                    )
                                ),
                            )
                            root = torch.sqrt(singular_values[:rank])
                            left = (u[:, :rank] * root).T.reshape(rank, p, r)
                            right = (root[:, None] * vh[:rank]).reshape(
                                rank, q, s
                            )
                            factored.append((source, target, left, right))
                        factored_sites.append(tuple(factored))
                factors = tuple(factored_sites)
        result = cls(cores, operator.dims, transitions, factors)
        if cache:
            _OPERATOR_CACHE.setdefault(operator, {})[key] = result
        return result


def _validate(state, operator, *, copy=True):
    if not isinstance(state, TorchWindow2State):
        raise TypeError("state must be a TorchWindow2State.")
    if not isinstance(operator, TorchWindow2Hamiltonian):
        raise TypeError("operator must be a TorchWindow2Hamiltonian.")
    if state.dims != operator.dims:
        raise ValueError("State and Hamiltonian physical dimensions differ.")
    return [core.clone() for core in state.cores] if copy else state.cores


def _left_boundary(cores, operator):
    physical = cores[0].shape[1]
    return torch.ones(
        physical,
        physical,
        cores[0].shape[0],
        operator.cores[0].shape[0],
        cores[0].shape[0],
        dtype=cores[0].dtype,
        device=cores[0].device,
    )


def _right_boundary(cores, operator):
    return torch.ones(
        cores[-1].shape[-1],
        operator.cores[-1].shape[-1],
        cores[-1].shape[-1],
        dtype=cores[0].dtype,
        device=cores[0].device,
    )


def _step_left(core, mpo, left, transitions=None):
    if transitions is not None:
        output = torch.zeros(
            core.shape[2], core.shape[2], core.shape[-1], mpo.shape[-1],
            core.shape[-1], dtype=core.dtype, device=core.device,
        )
        if len(transitions[0]) == 4:
            for source, target, left_op, right_op in transitions:
                output[:, :, :, target, :] += torch.einsum(
                    "apqb,prac,kpr,kqs,crsd->qsbd",
                    core.conj(), left[:, :, :, source, :], left_op, right_op,
                    core,
                )
        else:
            for source, target, value in transitions:
                output[:, :, :, target, :] += torch.einsum(
                    "apqb,prac,pqrs,crsd->qsbd",
                    core.conj(), left[:, :, :, source, :], value, core,
                )
        return output
    return torch.einsum("apqb,pramc,mpqrsn,crsd->qsbnd", core.conj(), left, mpo, core)


def _step_right(core, mpo, right, transitions=None):
    if transitions is not None:
        output = torch.zeros(
            core.shape[1], core.shape[1], core.shape[0], mpo.shape[0],
            core.shape[0], dtype=core.dtype, device=core.device,
        )
        factored = len(transitions[0]) == 4
        if core.ndim == 3:
            for transition in transitions:
                source, target, value = transition[:3]
                output[:, :, :, source, :] += torch.einsum(
                    "apb,pr,crd,bd->prac",
                    core.conj(), value, core, right[:, target, :],
                )
        elif factored:
            for source, target, left_op, right_op in transitions:
                output[:, :, :, source, :] += torch.einsum(
                    "apqb,kpr,kqs,crsd,qsbd->prac",
                    core.conj(), left_op, right_op, core,
                    right[:, :, :, target, :],
                )
        else:
            for source, target, value in transitions:
                output[:, :, :, source, :] += torch.einsum(
                    "apqb,pqrs,crsd,qsbd->prac",
                    core.conj(), value, core, right[:, :, :, target, :],
                )
        return output
    if core.ndim == 3:
        return torch.einsum("apb,mprn,crd,bnd->pramc", core.conj(), mpo, core, right)
    return torch.einsum(
        "apqb,mpqrsn,crsd,qsbnd->pramc", core.conj(), mpo, core, right
    )


def _right_environments(cores, operator, transition_sets=None):
    blocks = [None] * (len(cores) + 1)
    blocks[-1] = _right_boundary(cores, operator)
    for site in reversed(range(len(cores))):
        transitions = None if transition_sets is None else transition_sets[site]
        blocks[site] = _step_right(
            cores[site], operator.cores[site], blocks[site + 1], transitions
        )
    return blocks


def _apply_local(left, right, mpo, core, transitions=None, out=None):
    if transitions is not None:
        output = torch.zeros_like(core) if out is None else out.zero_()
        factored = len(transitions[0]) == 4
        if core.ndim == 3:
            for transition in transitions:
                source, target, value = transition[:3]
                output += torch.einsum(
                    "prac,pr,crd,bd->apb",
                    left[:, :, :, source, :], value, core, right[:, target, :],
                )
        elif factored:
            for source, target, left_op, right_op in transitions:
                output += _contract(
                    "prac,kpr,kqs,crsd,qsbd->apqb",
                    left[:, :, :, source, :], left_op, right_op, core,
                    right[:, :, :, target, :],
                )
        else:
            for source, target, value in transitions:
                output += torch.einsum(
                    "prac,pqrs,crsd,qsbd->apqb",
                    left[:, :, :, source, :], value, core,
                    right[:, :, :, target, :],
                )
        return output
    if core.ndim == 3:
        return torch.einsum("pramc,mprn,crd,bnd->apb", left, mpo, core, right)
    return torch.einsum(
        "pramc,mpqrsn,crsd,qsbnd->apqb", left, mpo, core, right
    )


def _apply_bond(left, right, center):
    return torch.einsum("pramc,rcd,prbmd->pab", left, center, right)


def _merge(left, right):
    if right.ndim == 3:
        return torch.einsum("apqb,bqd->apqd", left, right).contiguous()
    return torch.einsum("apqb,bqud->apqud", left, right).contiguous()


def _apply_two_site(
    left, right, mpo0, mpo1, center, transitions0=None, transitions1=None,
    out=None,
):
    if transitions0 is not None and transitions1 is not None:
        output = torch.zeros_like(center) if out is None else out.zero_()
        by_source = {}
        factored = len(transitions0[0]) == 4
        if factored:
            for source, target, left_op, right_op in transitions1:
                by_source.setdefault(source, []).append(
                    (target, left_op, right_op)
                )
        else:
            for source, target, value in transitions1:
                by_source.setdefault(source, []).append((target, value))
        if center.ndim == 4 and factored:
            for source, middle, left0, right0 in transitions0:
                for target, value1, unused in by_source.get(middle, ()):
                    output += _contract(
                        "prac,kpr,kqs,qs,crsf,bf->apqb",
                        left[:, :, :, source, :], left0, right0, value1,
                        center, right[:, target, :],
                    )
        elif center.ndim == 4:
            for source, middle, value0 in transitions0:
                for target, value1 in by_source.get(middle, ()):
                    output += torch.einsum(
                        "prac,pqrs,qs,crsf,bf->apqb",
                        left[:, :, :, source, :], value0, value1, center,
                        right[:, target, :],
                    )
        elif factored:
            for source, middle, left0, right0 in transitions0:
                for target, left1, right1 in by_source.get(middle, ()):
                    output += _contract(
                        "prac,kpr,kqs,lqs,lux,crsxf,uxbf->apqub",
                        left[:, :, :, source, :], left0, right0, left1, right1,
                        center, right[:, :, :, target, :],
                    )
        else:
            for source, middle, value0 in transitions0:
                for target, value1 in by_source.get(middle, ()):
                    output += torch.einsum(
                        "prac,pqrs,qusx,crsxf,uxbf->apqub",
                        left[:, :, :, source, :], value0, value1, center,
                        right[:, :, :, target, :],
                    )
        return output
    if center.ndim == 4:
        return torch.einsum(
            "pramc,mpqrsn,nqso,crsf,bof->apqb",
            left,
            mpo0,
            mpo1,
            center,
            right,
        )
    p, r, a, m, c = left.shape
    _m, _p, q, _r, s, n = mpo0.shape
    _n, _q, u, _s, x, o = mpo1.shape
    _c, _r2, _s2, _x, f = center.shape
    _u, _x2, b, _o, _f = right.shape

    w0_batch = mpo0.permute(1, 3, 2, 4, 5, 0).reshape(p * r, q * s * n, m)
    left_batch = left.permute(0, 1, 3, 2, 4).reshape(p * r, m, a * c)
    first = (w0_batch @ left_batch).reshape(p, r, q, s, n, a, c)

    first_batch = first.permute(3, 0, 2, 4, 5, 1, 6).reshape(
        s, p * q * n * a, r * c
    )
    center_batch = center.permute(2, 1, 0, 3, 4).reshape(s, r * c, x * f)
    second = (first_batch @ center_batch).reshape(s, p, q, n, a, x, f)
    second = second.permute(1, 2, 0, 3, 4, 5, 6)

    w1_batch = mpo1.permute(1, 4, 2, 5, 0, 3).reshape(q * x, u * o, n * s)
    second_batch = second.permute(1, 5, 3, 2, 0, 4, 6).reshape(
        q * x, n * s, p * a * f
    )
    third = (w1_batch @ second_batch).reshape(q, x, u, o, p, a, f)
    third = third.permute(4, 0, 5, 1, 6, 2, 3)

    third_batch = third.permute(5, 0, 1, 2, 3, 4, 6).reshape(
        u, p * q * a, x * f * o
    )
    right_batch = right.permute(0, 1, 4, 3, 2).reshape(u, x * f * o, b)
    result = (third_batch @ right_batch).reshape(u, p, q, a, b)
    return result.permute(3, 1, 2, 0, 4)


def _tridiagonal_eigh(diagonal, off_diagonal):
    if diagonal.device.type == "cpu":
        values, vectors = eigh_tridiagonal(
            diagonal.detach().double().numpy(),
            off_diagonal.detach().double().numpy(),
        )
        return (
            torch.as_tensor(values, dtype=diagonal.dtype, device=diagonal.device),
            torch.as_tensor(vectors, dtype=diagonal.dtype, device=diagonal.device),
        )
    dimension = diagonal.numel()
    matrix = torch.diag(diagonal)
    if dimension > 1:
        matrix = matrix + torch.diag(off_diagonal, diagonal=1)
        matrix = matrix + torch.diag(off_diagonal, diagonal=-1)
    return torch.linalg.eigh(matrix)


def _expm_krylov(
    apply, vector, factor, maximum, tolerance, records, workspace_cache=None
):
    vector = vector.reshape(-1)
    norm = torch.linalg.vector_norm(vector)
    if not bool(torch.isfinite(norm)) or float(norm) <= 0.0:
        raise ValueError("The Krylov start vector must have finite nonzero norm.")
    maximum = min(int(maximum), vector.numel())
    if maximum < 1:
        raise ValueError("krylov_dim must be positive.")
    cache_key = (str(vector.device), vector.dtype, maximum, vector.numel())
    buffers = None if workspace_cache is None else workspace_cache.get(cache_key)
    if buffers is None:
        buffers = (
            torch.empty(
                maximum, vector.numel(), dtype=vector.dtype, device=vector.device
            ),
            torch.empty(maximum, dtype=vector.real.dtype, device=vector.device),
            torch.empty(
                max(0, maximum - 1),
                dtype=vector.real.dtype,
                device=vector.device,
            ),
        )
        if workspace_cache is not None:
            workspace_cache[cache_key] = buffers
    basis, alpha, beta = buffers
    basis.zero_()
    alpha.zero_()
    beta.zero_()
    basis[0] = vector / norm
    eps = torch.finfo(vector.real.dtype).eps
    output = None
    residual = float("inf")
    used = maximum
    for index in range(maximum):
        work = apply(basis[index]).reshape(-1)
        alpha[index] = torch.vdot(basis[index], work).real
        work = work - alpha[index].to(work.dtype) * basis[index]
        if index:
            work = work - beta[index - 1].to(work.dtype) * basis[index - 1]
        outside = torch.linalg.vector_norm(work).real
        dimension = index + 1
        check = dimension == maximum or (dimension >= 4 and dimension % 2 == 0)
        if check:
            previous = beta[index - 1].abs() if index else alpha[index].new_zeros(())
            scale = torch.maximum(
                alpha[index].abs(), torch.maximum(previous, alpha[index].new_ones(()))
            )
            invariant = bool(outside <= eps * maximum * scale)
            values, vectors = _tridiagonal_eigh(
                alpha[:dimension], beta[: dimension - 1]
            )
            vectors = vectors.to(vector.dtype)
            projected = vectors @ (
                norm * torch.exp(factor * values).to(vector.dtype) * vectors[0]
            )
            output = basis[:dimension].T @ projected
            denominator = max(float(torch.linalg.vector_norm(output)), torch.finfo(vector.real.dtype).tiny)
            residual = abs(complex(factor)) * float(outside) * float(projected[-1].abs()) / denominator
            if invariant or residual <= tolerance or dimension == maximum:
                used = dimension
                break
        if index < maximum - 1:
            beta[index] = outside
            basis[index + 1] = work / torch.clamp_min(
                outside, torch.finfo(vector.real.dtype).tiny
            )
    if output is None:
        raise RuntimeError("Krylov propagation failed to build a projection.")
    if records is not None:
        records.append(
            {
                "residual": float(residual),
                "iterations": int(used),
                "converged": bool(residual <= tolerance),
                "target": float(tolerance),
            }
        )
    return output


def _evolve_local(
    left, right, mpo, core, time, krylov_dim, tolerance, records,
    transitions=None, workspace=None, krylov_workspaces=None,
):
    shape = core.shape
    if transitions is not None and workspace is None:
        workspace = torch.empty_like(core)
    return _expm_krylov(
        lambda value: _apply_local(
            left, right, mpo, value.reshape(shape), transitions, workspace
        ),
        core,
        -1.0j * time,
        krylov_dim,
        tolerance,
        records,
        krylov_workspaces,
    ).reshape(shape)


def _evolve_bond(
    left, right, center, time, krylov_dim, tolerance, records,
    krylov_workspaces=None,
):
    shape = center.shape
    return _expm_krylov(
        lambda value: _apply_bond(left, right, value.reshape(shape)),
        center,
        -1.0j * time,
        krylov_dim,
        tolerance,
        records,
        krylov_workspaces,
    ).reshape(shape)


def _evolve_two_site(
    left, right, mpo0, mpo1, center, time, krylov_dim, tolerance, records,
    transitions0=None, transitions1=None, krylov_workspaces=None,
):
    shape = center.shape
    workspace = torch.empty_like(center) if transitions0 is not None else None
    return _expm_krylov(
        lambda value: _apply_two_site(
            left, right, mpo0, mpo1, value.reshape(shape),
            transitions0, transitions1, workspace,
        ),
        center,
        -1.0j * time,
        krylov_dim,
        tolerance,
        records,
        krylov_workspaces,
    ).reshape(shape)


def _left_factor(core):
    left_rank, physical, shared, right_rank = core.shape
    batches = core.reshape(left_rank * physical, shared, right_rank).permute(1, 0, 2)
    q, center = torch.linalg.qr(batches, mode="reduced")
    rank = q.shape[-1]
    return q.permute(1, 0, 2).reshape(left_rank, physical, shared, rank), center


def _right_factor(core):
    left_rank, shared = core.shape[:2]
    trailing = core.shape[2:-1]
    right_rank = core.shape[-1]
    trailing_size = math.prod(trailing)
    batches = core.reshape(left_rank, shared, trailing_size * right_rank).permute(1, 0, 2)
    qh, rh = torch.linalg.qr(batches.mH, mode="reduced")
    center = rh.mH
    right_batches = qh.mH
    rank = right_batches.shape[-2]
    right_core = right_batches.permute(1, 0, 2).reshape(
        rank, shared, *trailing, right_rank
    )
    return center, right_core


def _absorb_right(center, next_core):
    if next_core.ndim == 3:
        return torch.einsum("vab,bvc->avc", center, next_core)
    return torch.einsum("vab,bvwc->avwc", center, next_core)


def _absorb_left(previous_core, center):
    return torch.einsum("apub,ubc->apuc", previous_core, center)


def _right_canonicalize(cores, *, copy=True):
    cores = [core.clone() for core in cores] if copy else cores
    for site in reversed(range(1, len(cores))):
        center, cores[site] = _right_factor(cores[site])
        cores[site - 1] = _absorb_left(cores[site - 1], center)
    return cores


def _split(center, distribution, cutoff, max_bond):
    if distribution not in {"left", "right"}:
        raise ValueError("distribution must be 'left' or 'right'.")
    if center.ndim == 5:
        left_rank, left_dim, shared, right_dim, right_rank = center.shape
        matrices = center.permute(2, 0, 1, 3, 4).reshape(
            shared, left_rank * left_dim, right_dim * right_rank
        )
        right_shape = (right_dim, right_rank)
    elif center.ndim == 4:
        left_rank, left_dim, shared, right_rank = center.shape
        matrices = center.permute(2, 0, 1, 3).reshape(
            shared, left_rank * left_dim, right_rank
        )
        right_shape = (right_rank,)
    else:
        raise ValueError("A two-site center must have four or five axes.")
    u, all_values, vh = torch.linalg.svd(matrices, full_matrices=False)
    numerical = torch.finfo(all_values.dtype).eps * max(matrices.shape[-2:])
    threshold = max(float(cutoff), float(numerical))
    leading = all_values[:, :1]
    sector_ranks = torch.count_nonzero(all_values > threshold * leading, dim=1)
    sector_ranks = torch.where(
        leading[:, 0] == 0,
        torch.ones_like(sector_ranks),
        torch.clamp(sector_ranks, min=1),
    )
    common = int(sector_ranks.max())
    if max_bond is not None:
        common = min(common, int(max_bond))
    common = max(1, min(common, all_values.shape[-1]))
    u = u[:, :, :common]
    values = all_values[:, :common]
    vh = vh[:, :common, :]
    mask = torch.arange(common, device=center.device)[None, :] < sector_ranks[:, None]
    kept = torch.sum(torch.where(mask, values, torch.zeros_like(values)) ** 2)
    discarded = float(torch.clamp(torch.sum(all_values**2) - kept, min=0.0))
    values = torch.where(mask, values, torch.zeros_like(values))
    if distribution == "right":
        left_batches = u
        right_batches = values.to(vh.dtype).unsqueeze(-1) * vh
    else:
        left_batches = u * values.to(u.dtype).unsqueeze(-2)
        right_batches = vh
    left_core = left_batches.permute(1, 0, 2).reshape(
        left_rank, left_dim, shared, common
    )
    right_core = right_batches.permute(1, 0, 2).reshape(
        common, shared, *right_shape
    )
    return left_core.contiguous(), right_core.contiguous(), {
        "rank": common,
        "discarded_weight": discarded,
    }


def _rank_limit(max_bond, bond):
    if max_bond is None or isinstance(max_bond, (int, np.integer)):
        value = max_bond
    else:
        values = tuple(max_bond)
        value = values[min(bond, len(values) - 1)]
    if value is not None and int(value) < 1:
        raise ValueError("Every finite max_bond must be positive.")
    return None if value is None else int(value)


def _diagnostics(mode, cores, discarded, records):
    residuals = np.asarray([record["residual"] for record in records], dtype=float)
    return {
        "integrator": mode,
        "ranks": tuple(int(core.shape[-1]) for core in cores[:-1]),
        "discarded_weights": tuple(float(value) for value in discarded),
        "truncation_error": float(sum(discarded)),
        "krylov_residual_max": float(np.max(residuals, initial=0.0)),
        "krylov_residual_rms": float(np.sqrt(np.mean(residuals**2))) if residuals.size else 0.0,
        "krylov_iterations_max": max((record["iterations"] for record in records), default=0),
        "krylov_calls": len(records),
        "krylov_not_converged": sum(not record["converged"] for record in records),
    }


def _normalize_left_center(cores):
    norm = torch.linalg.vector_norm(cores[0])
    if not bool(torch.isfinite(norm)) or float(norm) <= 0.0:
        raise FloatingPointError("LETTA TDVP produced a zero or non-finite norm.")
    cores[0] = cores[0] / norm


@torch.inference_mode()
def torch_one_site_tdvp_step(
    state,
    operator,
    dt,
    *,
    krylov_dim=20,
    krylov_tol=1.0e-10,
    canonicalize=True,
    normalize=False,
    channel_mode="dense",
    copy_state=True,
    workspaces=None,
    krylov_workspaces=None,
    return_info=False,
):
    """Apply one symmetric fixed-rank TDVP step using PyTorch."""
    cores = _validate(state, operator, copy=copy_state)
    if canonicalize:
        cores = _right_canonicalize(cores, copy=False)
    if channel_mode not in {"dense", "sparse", "factorized"}:
        raise ValueError("Invalid Torch channel_mode.")
    transition_sets = {
        "dense": None,
        "sparse": operator.transitions,
        "factorized": operator.factors,
    }[channel_mode]
    if channel_mode == "factorized" and transition_sets is None:
        raise ValueError("The Torch operator was created without physical factors.")
    right = _right_environments(cores, operator, transition_sets)
    left = [None] * (len(cores) + 1)
    left[0] = _left_boundary(cores, operator)
    records = []
    transition_sets = (None,) * len(cores) if transition_sets is None else transition_sets
    if workspaces is None:
        workspaces = (None,) * len(cores)
    elif tuple(tuple(workspace.shape) for workspace in workspaces) != tuple(
        tuple(core.shape) for core in cores
    ):
        raise ValueError("TDVP1 workspace shapes do not match the fixed-rank state.")
    half = 0.5 * float(dt)
    for site in range(len(cores) - 1):
        cores[site] = _evolve_local(
            left[site], right[site + 1], operator.cores[site], cores[site],
            half, krylov_dim, krylov_tol, records, transition_sets[site],
            workspaces[site], krylov_workspaces,
        )
        cores[site], center = _left_factor(cores[site])
        left[site + 1] = _step_left(
            cores[site], operator.cores[site], left[site], transition_sets[site]
        )
        center = _evolve_bond(
            left[site + 1], right[site + 1], center,
            -half, krylov_dim, krylov_tol, records, krylov_workspaces,
        )
        cores[site + 1] = _absorb_right(center, cores[site + 1])
    last = len(cores) - 1
    cores[last] = _evolve_local(
        left[last], right[-1], operator.cores[last], cores[last],
        float(dt), krylov_dim, krylov_tol, records, transition_sets[last],
        workspaces[last], krylov_workspaces,
    )
    for site in reversed(range(1, len(cores))):
        center, cores[site] = _right_factor(cores[site])
        right[site] = _step_right(
            cores[site], operator.cores[site], right[site + 1],
            transition_sets[site],
        )
        center = _evolve_bond(
            left[site], right[site], center,
            -half, krylov_dim, krylov_tol, records, krylov_workspaces,
        )
        cores[site - 1] = _absorb_left(cores[site - 1], center)
        cores[site - 1] = _evolve_local(
            left[site - 1], right[site], operator.cores[site - 1], cores[site - 1],
            half, krylov_dim, krylov_tol, records, transition_sets[site - 1],
            workspaces[site - 1], krylov_workspaces,
        )
    if not all(bool(torch.isfinite(core).all()) for core in cores):
        raise FloatingPointError("LETTA TDVP produced a non-finite tensor.")
    if normalize:
        _normalize_left_center(cores)
    output = state if not copy_state else TorchWindow2State(
        cores, state.dims, state.max_bond
    )
    info = _diagnostics("tdvp1", cores, [0.0] * (len(cores) - 1), records)
    return (output, info) if return_info else output


@torch.inference_mode()
def torch_two_site_tdvp_step(
    state,
    operator,
    dt,
    *,
    max_bond=None,
    cutoff=0.0,
    krylov_dim=20,
    krylov_tol=1.0e-10,
    canonicalize=True,
    normalize=False,
    channel_mode="dense",
    copy_state=True,
    krylov_workspaces=None,
    return_info=False,
):
    """Apply one symmetric rank-adaptive TDVP step using PyTorch."""
    cores = _validate(state, operator, copy=copy_state)
    if canonicalize:
        cores = _right_canonicalize(cores, copy=False)
    if channel_mode not in {"dense", "sparse", "factorized"}:
        raise ValueError("Invalid Torch channel_mode.")
    transition_sets = {
        "dense": None,
        "sparse": operator.transitions,
        "factorized": operator.factors,
    }[channel_mode]
    if channel_mode == "factorized" and transition_sets is None:
        raise ValueError("The Torch operator was created without physical factors.")
    right = _right_environments(cores, operator, transition_sets)
    left = [None] * (len(cores) + 1)
    left[0] = _left_boundary(cores, operator)
    records = []
    transition_sets = (None,) * len(cores) if transition_sets is None else transition_sets
    discarded = [0.0] * (len(cores) - 1)
    half = 0.5 * float(dt)
    for bond in range(len(cores) - 1):
        center = _evolve_two_site(
            left[bond], right[bond + 2], operator.cores[bond], operator.cores[bond + 1],
            _merge(cores[bond], cores[bond + 1]), half,
            krylov_dim, krylov_tol, records,
            transition_sets[bond], transition_sets[bond + 1],
            krylov_workspaces,
        )
        cores[bond], cores[bond + 1], split = _split(
            center, "right", cutoff, _rank_limit(max_bond, bond)
        )
        discarded[bond] += split["discarded_weight"]
        left[bond + 1] = _step_left(
            cores[bond], operator.cores[bond], left[bond], transition_sets[bond]
        )
        if bond < len(cores) - 2:
            cores[bond + 1] = _evolve_local(
                left[bond + 1], right[bond + 2], operator.cores[bond + 1],
                cores[bond + 1], -half, krylov_dim, krylov_tol, records,
                transition_sets[bond + 1], None, krylov_workspaces,
            )
    for bond in reversed(range(len(cores) - 1)):
        center = _evolve_two_site(
            left[bond], right[bond + 2], operator.cores[bond], operator.cores[bond + 1],
            _merge(cores[bond], cores[bond + 1]), half,
            krylov_dim, krylov_tol, records,
            transition_sets[bond], transition_sets[bond + 1],
            krylov_workspaces,
        )
        cores[bond], cores[bond + 1], split = _split(
            center, "left", cutoff, _rank_limit(max_bond, bond)
        )
        discarded[bond] += split["discarded_weight"]
        right[bond + 1] = _step_right(
            cores[bond + 1], operator.cores[bond + 1], right[bond + 2],
            transition_sets[bond + 1],
        )
        if bond:
            cores[bond] = _evolve_local(
                left[bond], right[bond + 1], operator.cores[bond], cores[bond],
                -half, krylov_dim, krylov_tol, records, transition_sets[bond],
                None, krylov_workspaces,
            )
    if not all(bool(torch.isfinite(core).all()) for core in cores):
        raise FloatingPointError("LETTA TDVP produced a non-finite tensor.")
    if normalize:
        _normalize_left_center(cores)
    output = state if not copy_state else TorchWindow2State(
        cores, state.dims, state.max_bond
    )
    info = _diagnostics("tdvp2", cores, discarded, records)
    return (output, info) if return_info else output


class TorchLETTATDVPEngine:
    """Reusable device-resident PyTorch window-2 LETTA TDVP engine."""

    backend = "torch"

    def __init__(
        self,
        operator,
        *,
        integrator="tdvp2",
        max_bond=None,
        cutoff=0.0,
        krylov_dim=20,
        krylov_tol=1.0e-10,
        canonicalize_first=True,
        canonicalize_each_step=False,
        device=None,
        dtype=None,
        num_threads=None,
        channel_mode="auto",
        inplace=True,
    ):
        if num_threads is not None:
            num_threads = int(num_threads)
            if num_threads < 1:
                raise ValueError("num_threads must be positive.")
            torch.set_num_threads(num_threads)
        self.device = torch.device("cpu" if device is None else device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested for LETTA TDVP, but this PyTorch runtime "
                "has no available CUDA device."
            )
        if self.device.type == "mps":
            mps_backend = getattr(torch.backends, "mps", None)
            if mps_backend is None or not mps_backend.is_available():
                raise RuntimeError(
                    "MPS was requested for LETTA TDVP, but it is unavailable."
                )
        self.dtype = _torch_dtype(dtype)
        requested_channel_mode = str(channel_mode).lower()
        if requested_channel_mode == "auto":
            requested_channel_mode = (
                "factorized"
                if operator.nsites * max(operator.dims) ** 2 >= 512
                else "dense"
            )
        if requested_channel_mode not in {"dense", "sparse", "factorized"}:
            raise ValueError(
                "channel_mode must be 'auto', 'dense', 'sparse', or 'factorized'."
            )
        self.channel_mode = requested_channel_mode
        self.inplace = bool(inplace)
        self.operator = TorchWindow2Hamiltonian.from_numpy(
            operator,
            device=self.device,
            dtype=self.dtype,
            factorize=self.channel_mode == "factorized",
        )
        self.integrator = self._normalize_integrator(integrator)
        self.max_bond = max_bond
        self.cutoff = float(cutoff)
        self.krylov_dim = int(krylov_dim)
        self.krylov_tol = float(krylov_tol)
        self.canonicalize_first = bool(canonicalize_first)
        self.canonicalize_each_step = bool(canonicalize_each_step)
        self.num_threads = torch.get_num_threads()
        self.prepared = False
        self.state = None
        self.history = []
        self._tdvp1_shapes = None
        self._tdvp1_workspaces = None
        self.fixed_rank_plan_rebuilds = 0
        self._krylov_workspaces = {}

    @staticmethod
    def _normalize_integrator(value):
        key = str(value).lower().replace("_", "-")
        if key in {"tdvp", "tdvp1", "one-site", "one-site-tdvp"}:
            return "tdvp1"
        if key in {"tdvp2", "two-site", "two-site-tdvp"}:
            return "tdvp2"
        raise ValueError("integrator must be 'tdvp1' or 'tdvp2'.")

    def prepare_state(self, state):
        if isinstance(state, LETTA):
            return TorchWindow2State.from_letta(
                state, device=self.device, dtype=self.dtype
            )
        if isinstance(state, TorchWindow2State):
            if state.dims != self.operator.dims:
                raise ValueError("State and Hamiltonian physical dimensions differ.")
            return state.to(device=self.device, dtype=self.dtype)
        raise TypeError("state must be LETTA or TorchWindow2State.")

    def set_integrator(self, value):
        self.integrator = self._normalize_integrator(value)
        return self

    def reset(self):
        self.prepared = False
        self.state = None
        self.history = []
        self._tdvp1_shapes = None
        self._tdvp1_workspaces = None
        self.fixed_rank_plan_rebuilds = 0
        self._krylov_workspaces = {}

    def _fixed_rank_workspaces(self, state):
        shapes = tuple(tuple(core.shape) for core in state.cores)
        if shapes != self._tdvp1_shapes:
            self._tdvp1_shapes = shapes
            self._tdvp1_workspaces = [
                torch.empty_like(core) for core in state.cores
            ]
            self.fixed_rank_plan_rebuilds += 1
        return self._tdvp1_workspaces

    def step(self, state, dt, *, normalize=False, return_info=True):
        state = self.prepare_state(state)
        canonicalize = self.canonicalize_each_step or (
            self.canonicalize_first and not self.prepared
        )
        options = {
            "krylov_dim": self.krylov_dim,
            "krylov_tol": self.krylov_tol,
            "canonicalize": canonicalize,
            "normalize": normalize,
            "channel_mode": self.channel_mode,
            "copy_state": not self.inplace,
            "krylov_workspaces": self._krylov_workspaces,
            "return_info": True,
        }
        if self.integrator == "tdvp2":
            output, info = torch_two_site_tdvp_step(
                state,
                self.operator,
                dt,
                max_bond=self.max_bond,
                cutoff=self.cutoff,
                **options,
            )
        else:
            options["workspaces"] = self._fixed_rank_workspaces(state)
            output, info = torch_one_site_tdvp_step(
                state, self.operator, dt, **options
            )
        self.prepared = True
        self.state = output
        self.history.append(dict(info))
        return (output, info) if return_info else output


TorchTDVP = TorchLETTATDVPEngine


@torch.inference_mode()
def torch_system_reduced_density_matrix(state, *, return_info=False):
    """Contract the first-site RDM while leaving all large tensors on-device."""
    if not isinstance(state, TorchWindow2State):
        raise TypeError("state must be a TorchWindow2State.")

    def rescale(value, log_scale, context):
        scale = torch.max(torch.abs(value))
        scalar = float(scale)
        if not math.isfinite(scalar) or scalar <= 0.0:
            raise FloatingPointError(f"{context}: invalid contraction scale {scalar}")
        return value / scale, log_scale + math.log(scalar)

    first = state.cores[0]
    environment = torch.einsum("avmx,bumy->xymuv", first.conj(), first)
    environment, log_scale = rescale(environment, 0.0, "LETTA core 0")
    for site, core in enumerate(state.cores[1:-1], start=1):
        environment = torch.einsum(
            "xymuv,xmnp,ymnq->pqnuv", environment, core.conj(), core
        )
        environment, log_scale = rescale(
            environment, log_scale, f"LETTA core {site}"
        )
    tail = state.cores[-1]
    rho = torch.einsum("xymuv,xmp,ymq->uv", environment, tail.conj(), tail)
    rho, log_scale = rescale(rho, log_scale, "system RDM")
    trace = torch.trace(rho)
    trace_value = complex(trace)
    if trace_value.real <= 0.0 or abs(trace_value.imag) > 1.0e-10 * trace_value.real:
        raise FloatingPointError(f"Invalid scaled RDM trace {trace_value}")
    log_norm = log_scale + math.log(trace_value.real)
    rho = rho / trace.real
    hermiticity_error = float(torch.max(torch.abs(rho - rho.mH)))
    minimum_eigenvalue = float(torch.linalg.eigvalsh(0.5 * (rho + rho.mH))[0])
    result = rho.detach().cpu().numpy()
    info = {
        "log_norm": log_norm,
        "trace_error": abs(math.expm1(log_norm)),
        "hermiticity_error": hermiticity_error,
        "minimum_eigenvalue": minimum_eigenvalue,
    }
    return (result, info) if return_info else result


@torch.inference_mode()
def torch_site_reduced_density_matrix(state, site, *, return_info=False):
    """Contract an arbitrary one-site RDM without leaving the Torch device."""
    if not isinstance(state, TorchWindow2State):
        raise TypeError("state must be a TorchWindow2State.")
    site = int(site)
    if not 0 <= site < len(state.dims):
        raise IndexError(f"site must lie in [0, {len(state.dims)}).")

    def rescale(value, log_scale, context):
        scale = torch.max(torch.abs(value))
        scalar = float(scale)
        if not math.isfinite(scalar) or scalar <= 0.0:
            raise FloatingPointError(f"{context}: invalid contraction scale {scalar}")
        return value / scale, log_scale + math.log(scalar)

    identities = [
        torch.eye(dim, dtype=state.dtype, device=state.device)
        for dim in state.dims
    ]
    left = torch.ones(
        (1, 1, state.dims[0], state.dims[0]),
        dtype=state.dtype,
        device=state.device,
    )
    left_log = 0.0
    for index in range(site):
        core = state.cores[index]
        left = torch.einsum(
            "bkxy,xy,bxuc,kyvd->cduv",
            left,
            identities[index],
            core.conj(),
            core,
        )
        left, left_log = rescale(
            left, left_log, f"Torch LETTA left environment {index + 1}"
        )

    tail = state.cores[-1][:, :, 0]
    if site == len(state.dims) - 1:
        coefficient = torch.einsum(
            "bkxy,bx,ky->xy", left, tail.conj(), tail
        )
        log_scale = left_log
    else:
        right = torch.einsum(
            "xy,bx,ky->bkxy", identities[-1], tail.conj(), tail
        )
        right, right_log = rescale(
            right, 0.0, f"Torch LETTA right environment {len(state.dims) - 1}"
        )
        for index in range(len(state.dims) - 2, site, -1):
            core = state.cores[index]
            right = torch.einsum(
                "xy,bxuc,kyvd,cduv->bkxy",
                identities[index],
                core.conj(),
                core,
                right,
            )
            right, right_log = rescale(
                right, right_log, f"Torch LETTA right environment {index}"
            )
        core = state.cores[site]
        coefficient = torch.einsum(
            "bkxy,bxuc,kyvd,cduv->xy",
            left,
            core.conj(),
            core,
            right,
        )
        log_scale = left_log + right_log

    rho = coefficient.T
    rho, log_scale = rescale(rho, log_scale, f"Torch LETTA site {site} RDM")
    trace = torch.trace(rho)
    trace_value = complex(trace)
    if trace_value.real <= 0.0 or abs(trace_value.imag) > 1.0e-10 * trace_value.real:
        raise FloatingPointError(f"Invalid scaled RDM trace {trace_value}")
    log_norm = log_scale + math.log(trace_value.real)
    rho = rho / trace.real
    hermiticity_error = float(torch.max(torch.abs(rho - rho.mH)))
    minimum_eigenvalue = float(torch.linalg.eigvalsh(0.5 * (rho + rho.mH))[0])
    result = rho.detach().cpu().numpy()
    info = {
        "site": site,
        "log_norm": log_norm,
        "trace_error": abs(math.expm1(log_norm)),
        "hermiticity_error": hermiticity_error,
        "minimum_eigenvalue": minimum_eigenvalue,
    }
    return (result, info) if return_info else result


__all__ = [
    "TorchLETTATDVPEngine",
    "TorchTDVP",
    "TorchWindow2Hamiltonian",
    "TorchWindow2State",
    "torch_one_site_tdvp_step",
    "torch_backend_capabilities",
    "torch_site_reduced_density_matrix",
    "torch_system_reduced_density_matrix",
    "torch_two_site_tdvp_step",
]
