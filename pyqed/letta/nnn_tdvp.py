"""Fixed-rank real-time TDVP for range-2 physical-leg-tied LETTA states."""

from __future__ import annotations

import numpy as np

from .range import NNNLETTA
from .tdvp import _expm_krylov


def nnn_structural_rank_caps(dims, max_bond):
    """Return exact virtual-rank caps for the range-2 tied geometry."""
    dims = tuple(int(dim) for dim in dims)
    maximum = int(max_bond)
    if len(dims) < 3:
        raise ValueError("NNN-LETTA needs at least three physical sites.")
    if maximum < 1:
        raise ValueError("max_bond must be positive.")

    def capped_product(values):
        result = 1
        for value in values:
            result *= int(value)
            if result >= maximum:
                return maximum
        return result

    return tuple(
        min(
            maximum,
            capped_product(dims[: bond + 1]),
            capped_product(dims[bond + 3 :]),
        )
        for bond in range(len(dims) - 3)
    )


def nnn_product_state(local_factors, *, max_bond=1):
    """Embed a product state exactly with padded range-2 virtual ranks."""
    factors = [np.asarray(factor, dtype=complex).reshape(-1) for factor in local_factors]
    if len(factors) < 3:
        raise ValueError("NNN-LETTA needs at least three local factors.")
    if any(factor.size == 0 for factor in factors):
        raise ValueError("Local factors cannot be empty.")
    dims = tuple(factor.size for factor in factors)
    caps = nnn_structural_rank_caps(dims, max_bond)
    bonds = (1,) + caps + (1,)
    nlocal = len(dims) - 2
    tensors = []
    for site in range(nlocal - 1):
        tensor = np.zeros(
            (bonds[site], dims[site], dims[site + 1], dims[site + 2], bonds[site + 1]),
            dtype=complex,
        )
        tensor[0, :, :, :, 0] = factors[site][:, None, None]
        tensors.append(tensor)
    site = nlocal - 1
    tail = np.einsum(
        "i,j,k->ijk", factors[-3], factors[-2], factors[-1], optimize=True
    )
    tensor = np.zeros(
        (bonds[site], dims[site], dims[site + 1], dims[site + 2], 1),
        dtype=complex,
    )
    tensor[0, ..., 0] = tail
    tensors.append(tensor)
    return NNNLETTA(dims, bond_dim=max(1, int(max_bond)), tensors=tensors)


def _left_factor(core):
    left_rank, owner, shared0, shared1, right_rank = core.shape
    if left_rank * owner < right_rank:
        raise ValueError(
            "The NNN-LETTA bond exceeds its left conditional structural rank."
        )
    output = np.empty_like(core)
    center = np.empty(
        (right_rank, shared0, shared1, right_rank), dtype=core.dtype
    )
    for s0 in range(shared0):
        for s1 in range(shared1):
            matrix = core[:, :, s0, s1, :].reshape(left_rank * owner, right_rank)
            q, r = np.linalg.qr(matrix, mode="reduced")
            output[:, :, s0, s1, :] = q.reshape(left_rank, owner, right_rank)
            center[:, s0, s1, :] = r
    return output, center


def _right_factor(core):
    left_rank, shared0, shared1, owner, right_rank = core.shape
    if owner * right_rank < left_rank:
        raise ValueError(
            "The NNN-LETTA bond exceeds its right conditional structural rank."
        )
    output = np.empty_like(core)
    center = np.empty(
        (left_rank, shared0, shared1, left_rank), dtype=core.dtype
    )
    for s0 in range(shared0):
        for s1 in range(shared1):
            matrix = core[:, s0, s1, :, :].reshape(
                left_rank, owner * right_rank
            )
            q, r = np.linalg.qr(matrix.T.conj(), mode="reduced")
            output[:, s0, s1, :, :] = q.T.conj().reshape(
                left_rank, owner, right_rank
            )
            center[:, s0, s1, :] = r.T.conj()
    return center, output


def _absorb_right(center, core):
    return np.einsum("auvb,buvpc->auvpc", center, core, optimize=True)


def _absorb_left(core, center):
    return np.einsum("apuvb,buvc->apuvc", core, center, optimize=True)


def _apply_bond(left, right, mpo0, mpo1, center):
    return np.einsum(
        "abmijkl,mnij,nokl,cdoijkl,bjld->aikc",
        left,
        mpo0,
        mpo1,
        right,
        center,
        optimize=True,
    )


def _evolve_local(state, mpo, site, left, right, tensor, time, krylov_dim, tol, records):
    shape = tensor.shape
    return _expm_krylov(
        lambda vector: state._apply_local_effective_from_environments(
            mpo, site, [left] * state.nlocal_tensors, [right] * state.nlocal_tensors,
            vector,
        ),
        tensor,
        -1.0j * time,
        krylov_dim,
        tol,
        records,
    ).reshape(shape)


def _evolve_bond(left, right, mpo0, mpo1, center, time, krylov_dim, tol, records):
    shape = center.shape
    return _expm_krylov(
        lambda vector: _apply_bond(
            left, right, mpo0, mpo1, vector.reshape(shape)
        ).reshape(-1),
        center,
        -1.0j * time,
        krylov_dim,
        tol,
        records,
    ).reshape(shape)


def one_site_nnn_tdvp_step(
    state,
    mpo,
    dt,
    *,
    krylov_dim=20,
    krylov_tol=1.0e-10,
    canonicalize=True,
    normalize=False,
    return_info=False,
):
    """Apply one symmetric fixed-rank projector-splitting TDVP step."""
    if not isinstance(state, NNNLETTA):
        raise TypeError("state must be an NNNLETTA instance.")
    if state.local_masks is not None:
        raise NotImplementedError("masked NNN-LETTA real-time TDVP is not implemented.")
    work = state.copy()
    mpo = work._validate_mpo(mpo)
    if canonicalize:
        work.canonicalize_conditional_center(0, normalize=False)

    right = work._right_local_environments(mpo)
    left = [None] * work.nlocal_tensors
    left[0] = work._initial_left_environment(mpo)
    records = []
    half = 0.5 * float(dt)

    for site in range(work.nlocal_tensors - 1):
        work.tensors[site] = _evolve_local(
            work, mpo, site, left[site], right[site], work.tensors[site],
            half, krylov_dim, krylov_tol, records,
        )
        work.tensors[site], center = _left_factor(work.tensors[site])
        left[site + 1] = work._advance_left_environment(
            left[site], mpo[site], work.tensors[site]
        )
        center = _evolve_bond(
            left[site + 1], right[site], mpo[site + 1], mpo[site + 2],
            center, -half, krylov_dim, krylov_tol, records,
        )
        work.tensors[site + 1] = _absorb_right(center, work.tensors[site + 1])

    last = work.nlocal_tensors - 1
    work.tensors[last] = _evolve_local(
        work, mpo, last, left[last], right[last], work.tensors[last],
        float(dt), krylov_dim, krylov_tol, records,
    )

    for site in reversed(range(1, work.nlocal_tensors)):
        center, work.tensors[site] = _right_factor(work.tensors[site])
        right[site - 1] = work._advance_right_environment(
            right[site], mpo[site + 2], work.tensors[site]
        )
        center = _evolve_bond(
            left[site], right[site - 1], mpo[site], mpo[site + 1],
            center, -half, krylov_dim, krylov_tol, records,
        )
        work.tensors[site - 1] = _absorb_left(work.tensors[site - 1], center)
        work.tensors[site - 1] = _evolve_local(
            work, mpo, site - 1, left[site - 1], right[site - 1],
            work.tensors[site - 1], half, krylov_dim, krylov_tol, records,
        )

    if not all(np.all(np.isfinite(tensor)) for tensor in work.tensors):
        raise FloatingPointError("NNN-LETTA TDVP produced a non-finite tensor.")
    if normalize:
        work.normalize()
    residuals = np.asarray([record["residual"] for record in records], dtype=float)
    info = {
        "integrator": "nnn-tdvp1",
        "ranks": tuple(int(tensor.shape[-1]) for tensor in work.tensors[:-1]),
        "krylov_residual_max": float(np.max(residuals, initial=0.0)),
        "krylov_iterations_max": max(
            (record["iterations"] for record in records), default=0
        ),
        "krylov_calls": len(records),
        "krylov_not_converged": sum(not record["converged"] for record in records),
    }
    return (work, info) if return_info else work


class NNNLETTATDVPEngine:
    """Reusable fixed-rank real-time engine for :class:`NNNLETTA`."""

    backend = "numpy"
    integrator = "nnn-tdvp1"

    def __init__(
        self,
        mpo,
        *,
        krylov_dim=20,
        krylov_tol=1.0e-10,
        canonicalize_first=True,
        canonicalize_each_step=False,
    ):
        self.mpo = mpo
        self.krylov_dim = int(krylov_dim)
        self.krylov_tol = float(krylov_tol)
        self.canonicalize_first = bool(canonicalize_first)
        self.canonicalize_each_step = bool(canonicalize_each_step)
        self.prepared = False
        self.state = None
        self.history = []

    def reset(self):
        self.prepared = False
        self.state = None
        self.history = []
        return self

    def step(self, state, dt, *, normalize=False, return_info=True):
        output, info = one_site_nnn_tdvp_step(
            state,
            self.mpo,
            dt,
            krylov_dim=self.krylov_dim,
            krylov_tol=self.krylov_tol,
            canonicalize=self.canonicalize_each_step
            or (self.canonicalize_first and not self.prepared),
            normalize=normalize,
            return_info=True,
        )
        self.prepared = True
        self.state = output
        self.history.append(dict(info))
        return (output, info) if return_info else output


__all__ = [
    "NNNLETTATDVPEngine",
    "nnn_product_state",
    "nnn_structural_rank_caps",
    "one_site_nnn_tdvp_step",
]
