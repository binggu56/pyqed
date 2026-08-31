"""Projector-splitting TDVP for nearest-neighbour window-2 LETTA states.

The implementation uses :class:`pyqed.letta.LETTA` with its optional
terminal tensor.  Pair tensors have layout ``(left, p_i, p_{i+1}, right)``;
the terminal tensor owns the final physical factor and has layout
``(p_last, left)``.  All factorizations are conditioned on the physical leg
shared by neighbouring pair tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from opt_einsum import contract_expression
from scipy.linalg import eigh_tridiagonal

from .core import LETTA


@lru_cache(maxsize=128)
def _cached_expression(expression, shapes):
    return contract_expression(expression, *shapes, optimize="greedy")


def _contract(expression, *operands):
    shapes = tuple(tuple(np.shape(operand)) for operand in operands)
    return _cached_expression(expression, shapes)(*operands)


@dataclass(frozen=True, eq=False)
class Window2Hamiltonian:
    """Finite-state window-2 representation of a nearest-neighbour sum."""

    cores: tuple[np.ndarray, ...]
    dims: tuple[int, ...]
    transitions: tuple[tuple[tuple[int, int, np.ndarray], ...], ...] | None = None
    factors: tuple[
        tuple[tuple[int, int, np.ndarray, np.ndarray | None], ...], ...
    ] | None = None

    @property
    def nsites(self):
        return len(self.dims)


def window2_hamiltonian_from_mpo(mpo):
    """Lift a standard finite MPO to the duplicated-leg window-2 geometry.

    Each physical operator is applied on the owning leg of one LETTA core;
    its shared leg carries an all-ones equality channel.  The construction is
    exact and also supports finite-range terms spanning multiple windows.
    """
    raw_cores = getattr(mpo, "factors", mpo)
    raw_cores = [np.asarray(core, dtype=complex) for core in raw_cores]
    if len(raw_cores) < 2 or any(core.ndim != 4 for core in raw_cores):
        raise ValueError("A window-2 Hamiltonian needs an MPO with at least two sites.")
    if raw_cores[0].shape[0] != 1 or raw_cores[-1].shape[1] != 1:
        raise ValueError("The MPO must have open boundary ranks.")

    dims = tuple(int(core.shape[2]) for core in raw_cores)
    if any(core.shape[2] != core.shape[3] for core in raw_cores):
        raise ValueError("The MPO physical operators must be square.")
    for left, right in zip(raw_cores[:-1], raw_cores[1:]):
        if left.shape[1] != right.shape[0]:
            raise ValueError("Adjacent MPO ranks do not match.")

    cores = []
    transitions = []
    factors = []
    for site, mpo_core in enumerate(raw_cores[:-1]):
        shared_dim = dims[site + 1]
        equality = np.ones((shared_dim, shared_dim), dtype=complex)
        cores.append(np.einsum("abux,vy->auvxyb", mpo_core, equality))
        site_transitions = []
        site_factors = []
        for source in range(mpo_core.shape[0]):
            for target in range(mpo_core.shape[1]):
                operator = mpo_core[source, target]
                if not np.any(operator):
                    continue
                window_operator = np.einsum("ux,vy->uvxy", operator, equality)
                site_transitions.append((source, target, window_operator))
                site_factors.append(
                    (source, target, operator[None], equality[None])
                )
        transitions.append(tuple(site_transitions))
        factors.append(tuple(site_factors))

    terminal = raw_cores[-1].transpose(0, 2, 3, 1)
    cores.append(terminal)
    terminal_transitions = []
    terminal_factors = []
    for source in range(raw_cores[-1].shape[0]):
        operator = raw_cores[-1][source, 0]
        if not np.any(operator):
            continue
        terminal_transitions.append((source, 0, operator))
        terminal_factors.append((source, 0, operator, None))
    transitions.append(tuple(terminal_transitions))
    factors.append(tuple(terminal_factors))
    return Window2Hamiltonian(
        tuple(cores), dims, tuple(transitions), tuple(factors)
    )


def window2_product_state(local_factors, *, max_bond=1):
    """Return a terminal-form LETTA representing a product state exactly."""
    factors = [np.asarray(factor, dtype=complex).reshape(-1) for factor in local_factors]
    if len(factors) < 2:
        raise ValueError("A window-2 LETTA state needs at least two factors.")
    if any(factor.size == 0 for factor in factors):
        raise ValueError("Local factors cannot be empty.")
    tensors = []
    for left, right in zip(factors[:-1], factors[1:]):
        tensor = left[:, None] * np.ones((1, right.size), dtype=complex)
        tensors.append(tensor[None, :, :, None])
    tensors.append(factors[-1][:, None])
    return LETTA(
        None,
        tuple(factor.size for factor in factors),
        bond_dim=max(1, int(max_bond)),
        tensors=tensors,
    )


def letta_structural_rank_caps(dims, max_bond):
    """Return algebraic rank caps for every window-2 virtual bond."""
    dims = tuple(int(dim) for dim in dims)
    maximum = int(max_bond)
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
        min(maximum, capped_product(dims[: bond + 1]), capped_product(dims[bond + 2 :]))
        for bond in range(len(dims) - 1)
    )


def nearest_neighbor_hamiltonian(
    bond_hamiltonians,
    dims,
    *,
    check_hermitian=True,
):
    """Build the exact window-2 operator for ``sum_i h[i, i+1]``.

    Three finite-state channels encode the sum for any chain length.  The
    duplicated physical occurrence is passed with an all-ones matrix, because
    it labels an equality constraint in the state rather than a second copy
    of the physical Hilbert space.
    """
    dims = tuple(int(dim) for dim in dims)
    terms = list(bond_hamiltonians)
    if len(dims) < 2 or len(terms) != len(dims) - 1:
        raise ValueError("Expected one Hamiltonian for every nearest-neighbour bond.")
    local_terms = []
    local_factors = []
    for bond, (raw, left_dim, right_dim) in enumerate(zip(terms, dims[:-1], dims[1:])):
        matrix = np.asarray(raw, dtype=complex).reshape(
            left_dim * right_dim, left_dim * right_dim
        )
        if check_hermitian and not np.allclose(
            matrix, matrix.conj().T, rtol=1.0e-10, atol=1.0e-12
        ):
            error = np.max(np.abs(matrix - matrix.conj().T))
            raise ValueError(
                f"Bond Hamiltonian {bond} is not Hermitian "
                f"(max |h-h^dagger|={error:.3e})."
            )
        local_terms.append(matrix.reshape(left_dim, right_dim, left_dim, right_dim))
        unfolding = local_terms[-1].transpose(0, 2, 1, 3).reshape(
            left_dim * left_dim, right_dim * right_dim
        )
        u, singular_values, vh = np.linalg.svd(unfolding, full_matrices=False)
        threshold = (
            np.finfo(singular_values.dtype).eps
            * max(unfolding.shape)
            * singular_values[0]
        )
        rank = max(1, int(np.count_nonzero(singular_values > threshold)))
        root = np.sqrt(singular_values[:rank])
        local_factors.append(
            (
                (u[:, :rank] * root).T.reshape(rank, left_dim, left_dim),
                (root[:, None] * vh[:rank]).reshape(
                    rank, right_dim, right_dim
                ),
            )
        )

    if len(dims) == 2:
        terminal_value = np.ones((dims[-1], dims[-1]), dtype=complex)
        return Window2Hamiltonian(
            (
                local_terms[0][None, ..., None],
                terminal_value[None, ..., None],
            ),
            dims,
            (((0, 0, local_terms[0]),), ((0, 0, terminal_value),)),
            (
                ((0, 0, local_factors[0][0], local_factors[0][1]),),
                ((0, 0, terminal_value, None),),
            ),
        )

    cores = []
    transitions = []
    factors = []
    for site in range(len(dims) - 1):
        left_dim, right_dim = dims[site : site + 2]
        left_rank = 1 if site == 0 else 3
        core = np.zeros(
            (left_rank, left_dim, right_dim, left_dim, right_dim, 3),
            dtype=complex,
        )
        owner_identity = _contract(
            "ux,vy->uvxy", np.eye(left_dim), np.ones((right_dim, right_dim))
        )
        core[0, ..., 0] = owner_identity
        core[0, ..., 1] = local_terms[site]
        site_transitions = [
            (0, 0, owner_identity),
            (0, 1, local_terms[site]),
        ]
        if site:
            core[1, ..., 2] = 1.0
            core[2, ..., 2] = owner_identity
            site_transitions.extend(
                (
                    (1, 2, np.ones_like(owner_identity)),
                    (2, 2, owner_identity),
                )
            )
        cores.append(core)
        transitions.append(tuple(site_transitions))
        identity_left = np.eye(left_dim, dtype=complex)[None]
        identity_right = np.ones(
            (1, right_dim, right_dim), dtype=complex
        )
        site_factors = [
            (0, 0, identity_left, identity_right),
            (0, 1, local_factors[site][0], local_factors[site][1]),
        ]
        if site:
            site_factors.extend(
                (
                    (
                        1,
                        2,
                        np.ones((1, left_dim, left_dim), dtype=complex),
                        identity_right,
                    ),
                    (2, 2, identity_left, identity_right),
                )
            )
        factors.append(tuple(site_factors))
    terminal = np.zeros((3, dims[-1], dims[-1], 1), dtype=complex)
    terminal[1, ..., 0] = 1.0
    terminal[2, ..., 0] = np.eye(dims[-1])
    cores.append(terminal)
    transitions.append(
        (
            (1, 0, np.ones((dims[-1], dims[-1]), dtype=complex)),
            (2, 0, np.eye(dims[-1], dtype=complex)),
        )
    )
    factors.append(
        (
            (1, 0, np.ones((dims[-1], dims[-1]), dtype=complex), None),
            (2, 0, np.eye(dims[-1], dtype=complex), None),
        )
    )
    return Window2Hamiltonian(
        tuple(cores), dims, tuple(transitions), tuple(factors)
    )


def _state_cores(state):
    if not isinstance(state, LETTA):
        raise TypeError("state must be a LETTA instance.")
    if not state.has_terminal_tensor:
        raise ValueError("Window-2 TDVP requires LETTA's terminal-tensor form.")
    cores = [np.asarray(tensor, dtype=complex).copy() for tensor in state.tensors[:-1]]
    terminal = np.asarray(state.tensors[-1], dtype=complex).T[:, :, None]
    cores.append(terminal.copy())
    return cores


def _assign_state_cores(state, cores):
    state.tensors = [np.ascontiguousarray(core) for core in cores[:-1]]
    state.tensors.append(np.ascontiguousarray(cores[-1][:, :, 0].T))
    state.local_masks = [None] * len(state.tensors)
    return state


def _validate(state, operator):
    if not isinstance(operator, Window2Hamiltonian):
        raise TypeError("operator must be a Window2Hamiltonian.")
    if tuple(state.dims) != operator.dims:
        raise ValueError("State and Hamiltonian physical dimensions differ.")
    cores = _state_cores(state)
    if len(cores) != operator.nsites:
        raise ValueError("State and Hamiltonian lengths differ.")
    for site, core in enumerate(cores):
        expected = 3 if site == len(cores) - 1 else 4
        if core.ndim != expected:
            raise ValueError(f"State core {site} must have rank {expected}.")
    return cores


def _left_boundary(cores, operator):
    physical = cores[0].shape[1]
    return np.ones(
        (physical, physical, cores[0].shape[0], operator.cores[0].shape[0], cores[0].shape[0]),
        dtype=complex,
    )


def _right_boundary(cores, operator):
    return np.ones(
        (cores[-1].shape[-1], operator.cores[-1].shape[-1], cores[-1].shape[-1]),
        dtype=complex,
    )


def _step_left(core, mpo, left, transitions=None):
    if transitions is not None:
        output = np.zeros(
            (core.shape[2], core.shape[2], core.shape[-1], mpo.shape[-1], core.shape[-1]),
            dtype=core.dtype,
        )
        if len(transitions[0]) == 4:
            for source, target, left_op, right_op in transitions:
                output[:, :, :, target, :] += _contract(
                    "apqb,prac,kpr,kqs,crsd->qsbd",
                    core.conj(), left[:, :, :, source, :], left_op, right_op,
                    core,
                )
        else:
            for source, target, value in transitions:
                output[:, :, :, target, :] += _contract(
                    "apqb,prac,pqrs,crsd->qsbd",
                    core.conj(), left[:, :, :, source, :], value, core,
                )
        return output
    return _contract("apqb,pramc,mpqrsn,crsd->qsbnd", core.conj(), left, mpo, core)


def _step_right(core, mpo, right, transitions=None):
    if transitions is not None:
        output = np.zeros(
            (core.shape[1], core.shape[1], core.shape[0], mpo.shape[0], core.shape[0]),
            dtype=core.dtype,
        )
        factored = len(transitions[0]) == 4
        if core.ndim == 3:
            for transition in transitions:
                source, target, value = transition[:3]
                output[:, :, :, source, :] += _contract(
                    "apb,pr,crd,bd->prac",
                    core.conj(), value, core, right[:, target, :],
                )
        elif factored:
            for source, target, left_op, right_op in transitions:
                output[:, :, :, source, :] += _contract(
                    "apqb,kpr,kqs,crsd,qsbd->prac",
                    core.conj(), left_op, right_op, core,
                    right[:, :, :, target, :],
                )
        else:
            for source, target, value in transitions:
                output[:, :, :, source, :] += _contract(
                    "apqb,pqrs,crsd,qsbd->prac",
                    core.conj(), value, core, right[:, :, :, target, :],
                )
        return output
    if core.ndim == 3:
        return _contract("apb,mprn,crd,bnd->pramc", core.conj(), mpo, core, right)
    return _contract("apqb,mpqrsn,crsd,qsbnd->pramc", core.conj(), mpo, core, right)


def _right_environments(cores, operator, transition_sets=None):
    blocks = [None] * (len(cores) + 1)
    blocks[-1] = _right_boundary(cores, operator)
    for site in reversed(range(len(cores))):
        transitions = None if transition_sets is None else transition_sets[site]
        blocks[site] = _step_right(
            cores[site], operator.cores[site], blocks[site + 1], transitions
        )
    return blocks


def _apply_local(left, right, mpo, core, transitions=None):
    if transitions is not None:
        output = np.zeros_like(core)
        factored = len(transitions[0]) == 4
        if core.ndim == 3:
            for transition in transitions:
                source, target, value = transition[:3]
                output += _contract(
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
                output += _contract(
                    "prac,pqrs,crsd,qsbd->apqb",
                    left[:, :, :, source, :], value, core,
                    right[:, :, :, target, :],
                )
        return output
    if core.ndim == 3:
        return _contract("pramc,mprn,crd,bnd->apb", left, mpo, core, right)
    return _contract("pramc,mpqrsn,crsd,qsbnd->apqb", left, mpo, core, right)


def _apply_bond(left, right, center):
    return _contract("pramc,rcd,prbmd->pab", left, center, right)


def _merge(left, right):
    if right.ndim == 3:
        return _contract("apqb,bqd->apqd", left, right)
    return _contract("apqb,bqud->apqud", left, right)


def _apply_two_site(
    left, right, mpo0, mpo1, center, transitions0=None, transitions1=None
):
    if transitions0 is not None and transitions1 is not None:
        output = np.zeros_like(center)
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
                    output += _contract(
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
                    output += _contract(
                        "prac,pqrs,qusx,crsxf,uxbf->apqub",
                        left[:, :, :, source, :], value0, value1, center,
                        right[:, :, :, target, :],
                    )
        return output
    if center.ndim == 4:
        return _contract(
            "pramc,mpqrsn,nqso,crsf,bof->apqb",
            left, mpo0, mpo1, center, right,
        )
    # Interior contraction written as four dense matrix products.  NumPy's
    # generic einsum kernel does not recognize BLAS structure here and is
    # orders of magnitude slower for DVR dimensions around 12.
    p, r, a, m, c = left.shape
    _m, _p, q, _r, s, n = mpo0.shape
    _n, _q, u, _s, x, o = mpo1.shape
    _c, _r2, _s2, _x, f = center.shape
    _u, _x2, b, _o, _f = right.shape

    w0_batch = mpo0.transpose(1, 3, 2, 4, 5, 0).reshape(
        p * r, q * s * n, m
    )
    left_batch = left.transpose(0, 1, 3, 2, 4).reshape(
        p * r, m, a * c
    )
    first = (w0_batch @ left_batch).reshape(p, r, q, s, n, a, c)

    first_batch = first.transpose(3, 0, 2, 4, 5, 1, 6).reshape(
        s, p * q * n * a, r * c
    )
    center_batch = center.transpose(2, 1, 0, 3, 4).reshape(
        s, r * c, x * f
    )
    second = (first_batch @ center_batch).reshape(s, p, q, n, a, x, f)
    second = second.transpose(1, 2, 0, 3, 4, 5, 6)

    w1_batch = mpo1.transpose(1, 4, 2, 5, 0, 3).reshape(
        q * x, u * o, n * s
    )
    second_batch = second.transpose(1, 5, 3, 2, 0, 4, 6).reshape(
        q * x, n * s, p * a * f
    )
    third = (w1_batch @ second_batch).reshape(q, x, u, o, p, a, f)
    third = third.transpose(4, 0, 5, 1, 6, 2, 3)

    third_batch = third.transpose(5, 0, 1, 2, 3, 4, 6).reshape(
        u, p * q * a, x * f * o
    )
    right_batch = right.transpose(0, 1, 4, 3, 2).reshape(
        u, x * f * o, b
    )
    result = (third_batch @ right_batch).reshape(u, p, q, a, b)
    return result.transpose(3, 1, 2, 0, 4)


def _expm_krylov(apply, vector, factor, maximum, tolerance, records):
    vector = np.asarray(vector, dtype=complex).reshape(-1)
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("The Krylov start vector must have finite nonzero norm.")
    maximum = min(int(maximum), vector.size)
    if maximum < 1:
        raise ValueError("krylov_dim must be positive.")
    basis = np.zeros((maximum, vector.size), dtype=complex)
    basis[0] = vector / norm
    alpha = np.zeros(maximum)
    beta = np.zeros(max(0, maximum - 1))
    eps = np.finfo(float).eps
    output = None
    residual = np.inf
    used = maximum
    for index in range(maximum):
        work = np.asarray(apply(basis[index]), dtype=complex).reshape(-1)
        alpha[index] = np.vdot(basis[index], work).real
        work -= alpha[index] * basis[index]
        if index:
            work -= beta[index - 1] * basis[index - 1]
        outside = float(np.linalg.norm(work))
        dimension = index + 1
        scale = max(1.0, abs(alpha[index]), abs(beta[index - 1]) if index else 0.0)
        invariant = outside <= eps * maximum * scale
        check = invariant or dimension == maximum or (dimension >= 4 and dimension % 2 == 0)
        if check:
            values, vectors = eigh_tridiagonal(alpha[:dimension], beta[: dimension - 1])
            projected = vectors @ (norm * np.exp(factor * values) * vectors[0])
            output = basis[:dimension].T @ projected
            residual = abs(complex(factor)) * outside * abs(projected[-1]) / max(
                np.linalg.norm(output), np.finfo(float).tiny
            )
            if invariant or residual <= tolerance or dimension == maximum:
                used = dimension
                break
        if index < maximum - 1:
            beta[index] = outside
            basis[index + 1] = work / outside
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
    left, right, mpo, core, time, krylov_dim, tolerance, records, transitions=None
):
    shape = core.shape
    return _expm_krylov(
        lambda value: _apply_local(
            left, right, mpo, value.reshape(shape), transitions
        ).reshape(-1),
        core,
        -1.0j * time,
        krylov_dim,
        tolerance,
        records,
    ).reshape(shape)


def _evolve_bond(left, right, center, time, krylov_dim, tolerance, records):
    shape = center.shape
    return _expm_krylov(
        lambda value: _apply_bond(left, right, value.reshape(shape)).reshape(-1),
        center,
        -1.0j * time,
        krylov_dim,
        tolerance,
        records,
    ).reshape(shape)


def _evolve_two_site(
    left, right, mpo0, mpo1, center, time, krylov_dim, tolerance, records,
    transitions0=None, transitions1=None,
):
    shape = center.shape
    return _expm_krylov(
        lambda value: _apply_two_site(
            left, right, mpo0, mpo1, value.reshape(shape),
            transitions0, transitions1,
        ).reshape(-1),
        center,
        -1.0j * time,
        krylov_dim,
        tolerance,
        records,
    ).reshape(shape)


def _left_factor(core):
    left_rank, physical, shared, right_rank = core.shape
    batches = core.reshape(left_rank * physical, shared, right_rank).transpose(1, 0, 2)
    q, center = np.linalg.qr(batches, mode="reduced")
    rank = q.shape[-1]
    return q.transpose(1, 0, 2).reshape(left_rank, physical, shared, rank), center


def _right_factor(core):
    left_rank, shared = core.shape[:2]
    trailing = core.shape[2:-1]
    right_rank = core.shape[-1]
    batches = core.reshape(left_rank, shared, int(np.prod(trailing, dtype=int)) * right_rank)
    batches = batches.transpose(1, 0, 2)
    qh, rh = np.linalg.qr(batches.conj().transpose(0, 2, 1), mode="reduced")
    center = rh.conj().transpose(0, 2, 1)
    right_batches = qh.conj().transpose(0, 2, 1)
    rank = right_batches.shape[-2]
    right_core = right_batches.transpose(1, 0, 2).reshape(
        rank, shared, *trailing, right_rank
    )
    return center, right_core


def _absorb_right(center, next_core):
    if next_core.ndim == 3:
        return _contract("vab,bvc->avc", center, next_core)
    return _contract("vab,bvwc->avwc", center, next_core)


def _absorb_left(previous_core, center):
    return _contract("apub,ubc->apuc", previous_core, center)


def _right_canonicalize(cores):
    """Move the center to the left edge by exact conditional batch QR."""
    cores = [np.asarray(core, dtype=complex).copy() for core in cores]
    for site in reversed(range(1, len(cores))):
        center, cores[site] = _right_factor(cores[site])
        cores[site - 1] = _absorb_left(cores[site - 1], center)
    return cores


def _split(center, distribution, cutoff, max_bond):
    if distribution not in {"left", "right"}:
        raise ValueError("distribution must be 'left' or 'right'.")
    if center.ndim == 5:
        left_rank, left_dim, shared, right_dim, right_rank = center.shape
        matrices = center.transpose(2, 0, 1, 3, 4).reshape(
            shared, left_rank * left_dim, right_dim * right_rank
        )
        right_shape = (right_dim, right_rank)
    elif center.ndim == 4:
        left_rank, left_dim, shared, right_rank = center.shape
        matrices = center.transpose(2, 0, 1, 3).reshape(
            shared, left_rank * left_dim, right_rank
        )
        right_shape = (right_rank,)
    else:
        raise ValueError("A two-site center must have four or five axes.")
    u, all_values, vh = np.linalg.svd(matrices, full_matrices=False)
    numerical = np.finfo(all_values.dtype).eps * max(matrices.shape[-2:])
    threshold = max(float(cutoff), float(numerical))
    sector_ranks = []
    for values in all_values:
        rank = 1 if values[0] == 0 else max(1, int(np.count_nonzero(values > threshold * values[0])))
        sector_ranks.append(rank)
    common = max(sector_ranks)
    if max_bond is not None:
        common = min(common, int(max_bond))
    common = max(1, min(common, all_values.shape[-1]))
    u = u[:, :, :common]
    values = all_values[:, :common].copy()
    vh = vh[:, :common, :]
    mask = np.zeros_like(values, dtype=bool)
    for sector, rank in enumerate(sector_ranks):
        mask[sector, : min(rank, common)] = True
    kept = np.sum(np.where(mask, values, 0.0) ** 2)
    discarded = max(0.0, float(np.sum(all_values**2) - kept))
    values[~mask] = 0.0
    if distribution == "right":
        left_batches = u
        right_batches = values[:, :, None] * vh
    else:
        left_batches = u * values[:, None, :]
        right_batches = vh
    left_core = left_batches.transpose(1, 0, 2).reshape(
        left_rank, left_dim, shared, common
    )
    right_core = right_batches.transpose(1, 0, 2).reshape(
        common, shared, *right_shape
    )
    return left_core, right_core, {"rank": common, "discarded_weight": discarded}


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


def one_site_tdvp_step(
    state,
    operator,
    dt,
    *,
    krylov_dim=20,
    krylov_tol=1.0e-10,
    canonicalize=True,
    normalize=False,
    channel_mode="dense",
    return_info=False,
):
    """Apply one symmetric fixed-rank window-2 TDVP step."""
    work = state.copy()
    cores = _validate(work, operator)
    if canonicalize:
        cores = _right_canonicalize(cores)
    if channel_mode not in {"dense", "sparse", "factorized"}:
        raise ValueError("Invalid NumPy channel_mode.")
    transition_sets = {
        "dense": None,
        "sparse": operator.transitions,
        "factorized": operator.factors,
    }[channel_mode]
    if channel_mode == "factorized" and transition_sets is None:
        raise ValueError("The NumPy operator has no physical factors.")
    right = _right_environments(cores, operator, transition_sets)
    left = [None] * (len(cores) + 1)
    left[0] = _left_boundary(cores, operator)
    records = []
    transition_sets = (None,) * len(cores) if transition_sets is None else transition_sets
    half = 0.5 * float(dt)
    for site in range(len(cores) - 1):
        cores[site] = _evolve_local(
            left[site], right[site + 1], operator.cores[site], cores[site],
            half, krylov_dim, krylov_tol, records, transition_sets[site],
        )
        cores[site], center = _left_factor(cores[site])
        left[site + 1] = _step_left(
            cores[site], operator.cores[site], left[site], transition_sets[site]
        )
        center = _evolve_bond(
            left[site + 1], right[site + 1], center,
            -half, krylov_dim, krylov_tol, records,
        )
        cores[site + 1] = _absorb_right(center, cores[site + 1])
    last = len(cores) - 1
    cores[last] = _evolve_local(
        left[last], right[-1], operator.cores[last], cores[last],
        float(dt), krylov_dim, krylov_tol, records, transition_sets[last],
    )
    for site in reversed(range(1, len(cores))):
        center, cores[site] = _right_factor(cores[site])
        right[site] = _step_right(
            cores[site], operator.cores[site], right[site + 1],
            transition_sets[site],
        )
        center = _evolve_bond(
            left[site], right[site], center,
            -half, krylov_dim, krylov_tol, records,
        )
        cores[site - 1] = _absorb_left(cores[site - 1], center)
        cores[site - 1] = _evolve_local(
            left[site - 1], right[site], operator.cores[site - 1], cores[site - 1],
            half, krylov_dim, krylov_tol, records, transition_sets[site - 1],
        )
    if not all(np.all(np.isfinite(core)) for core in cores):
        raise FloatingPointError("LETTA TDVP produced a non-finite tensor.")
    _assign_state_cores(work, cores)
    if normalize:
        work.normalize()
    info = _diagnostics("tdvp1", cores, [0.0] * (len(cores) - 1), records)
    return (work, info) if return_info else work


def two_site_tdvp_step(
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
    return_info=False,
):
    """Apply one symmetric rank-adaptive window-2 TDVP step."""
    work = state.copy()
    cores = _validate(work, operator)
    if canonicalize:
        cores = _right_canonicalize(cores)
    if channel_mode not in {"dense", "sparse", "factorized"}:
        raise ValueError("Invalid NumPy channel_mode.")
    transition_sets = {
        "dense": None,
        "sparse": operator.transitions,
        "factorized": operator.factors,
    }[channel_mode]
    if channel_mode == "factorized" and transition_sets is None:
        raise ValueError("The NumPy operator has no physical factors.")
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
                left[bond + 1], right[bond + 2], operator.cores[bond + 1], cores[bond + 1],
                -half, krylov_dim, krylov_tol, records,
                transition_sets[bond + 1],
            )
    for bond in reversed(range(len(cores) - 1)):
        center = _evolve_two_site(
            left[bond], right[bond + 2], operator.cores[bond], operator.cores[bond + 1],
            _merge(cores[bond], cores[bond + 1]), half,
            krylov_dim, krylov_tol, records,
            transition_sets[bond], transition_sets[bond + 1],
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
            )
    if not all(np.all(np.isfinite(core)) for core in cores):
        raise FloatingPointError("LETTA TDVP produced a non-finite tensor.")
    _assign_state_cores(work, cores)
    if normalize:
        work.normalize()
    info = _diagnostics("tdvp2", cores, discarded, records)
    return (work, info) if return_info else work


class LETTATDVPEngine:
    """Reusable native window-2 LETTA TDVP engine."""

    backend = "numpy"

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
        channel_mode="dense",
    ):
        if not isinstance(operator, Window2Hamiltonian):
            raise TypeError("operator must be a Window2Hamiltonian.")
        self.operator = operator
        self.integrator = self._normalize_integrator(integrator)
        self.max_bond = max_bond
        self.cutoff = float(cutoff)
        self.krylov_dim = int(krylov_dim)
        self.krylov_tol = float(krylov_tol)
        self.canonicalize_first = bool(canonicalize_first)
        self.canonicalize_each_step = bool(canonicalize_each_step)
        self.channel_mode = str(channel_mode).lower()
        if self.channel_mode == "auto":
            self.channel_mode = (
                "factorized"
                if operator.nsites * max(operator.dims) ** 2 >= 512
                else "dense"
            )
        if self.channel_mode not in {"dense", "sparse", "factorized"}:
            raise ValueError(
                "channel_mode must be 'auto', 'dense', 'sparse', or 'factorized'."
            )
        self.prepared = False
        self.state = None
        self.history = []

    @staticmethod
    def _normalize_integrator(value):
        key = str(value).lower().replace("_", "-")
        if key in {"tdvp", "tdvp1", "one-site", "one-site-tdvp"}:
            return "tdvp1"
        if key in {"tdvp2", "two-site", "two-site-tdvp"}:
            return "tdvp2"
        raise ValueError("integrator must be 'tdvp1' or 'tdvp2'.")

    def set_integrator(self, value):
        self.integrator = self._normalize_integrator(value)
        return self

    def reset(self):
        self.prepared = False
        self.state = None
        self.history = []

    def step(self, state, dt, *, normalize=False, return_info=True):
        canonicalize = self.canonicalize_each_step or (
            self.canonicalize_first and not self.prepared
        )
        options = {
            "krylov_dim": self.krylov_dim,
            "krylov_tol": self.krylov_tol,
            "canonicalize": canonicalize,
            "normalize": normalize,
            "channel_mode": self.channel_mode,
            "return_info": True,
        }
        if self.integrator == "tdvp2":
            output, info = two_site_tdvp_step(
                state,
                self.operator,
                dt,
                max_bond=self.max_bond,
                cutoff=self.cutoff,
                **options,
            )
        else:
            output, info = one_site_tdvp_step(state, self.operator, dt, **options)
        self.prepared = True
        self.state = output
        self.history.append(dict(info))
        return (output, info) if return_info else output


NumPyTDVP = LETTATDVPEngine


__all__ = [
    "LETTATDVPEngine",
    "NumPyTDVP",
    "Window2Hamiltonian",
    "letta_structural_rank_caps",
    "nearest_neighbor_hamiltonian",
    "one_site_tdvp_step",
    "two_site_tdvp_step",
    "window2_hamiltonian_from_mpo",
    "window2_product_state",
]
