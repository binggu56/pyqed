"""Internal raw-array kernels for dense MPS/MPO algebra."""

from __future__ import annotations

import numpy as np

from .decompose import compress


def apply_mpo_uncompressed(mpo_factors, mps_factors):
    """Contract raw MPO/MPS factors into standard-order MPS factors."""
    mpo_factors = tuple(mpo_factors)
    mps_factors = tuple(mps_factors)
    if len(mpo_factors) != len(mps_factors):
        raise ValueError(
            "MPO and MPS lengths must match; got "
            f"{len(mpo_factors)} and {len(mps_factors)}."
        )
    if not mpo_factors:
        raise ValueError("MPO and MPS must contain at least one site.")

    result = []
    previous_mpo_right = previous_mps_right = None
    for site, (operator, state) in enumerate(zip(mpo_factors, mps_factors)):
        if getattr(operator, "ndim", None) not in (3,4):
            raise ValueError(f"MPO site {site} must have rank 3 or 4.")
        if getattr(state, "ndim", None) != 3:
            raise ValueError(f"MPS site {site} must have rank 3.")

        op_left, op_right, output_dim = operator.shape[:3]
        op_input_dim = operator.shape[-1]
        state_left, state_input_dim, state_right = state.shape
        if op_input_dim != state_input_dim:
            raise ValueError(
                f"Physical input dimension mismatch at site {site}: "
                f"MPO has {op_input_dim}, MPS has {state_input_dim}."
            )
        if site and op_left != previous_mpo_right:
            raise ValueError(f"Incompatible MPO virtual bond before site {site}.")
        if site and state_left != previous_mps_right:
            raise ValueError(f"Incompatible MPS virtual bond before site {site}.")

        contracted = np.einsum(
            "abi,kil->kailb" if operator.ndim==3 else "abij,kjl->kailb", operator, state, optimize=True
        )
        result.append(
            contracted.reshape(
                op_left * state_left,
                output_dim,
                state_right * op_right,
            )
        )
        previous_mpo_right = op_right
        previous_mps_right = state_right
    return result


def apply_mpo_factors(mpo_factors, mps_factors, max_bond):
    """Apply and truncate raw factors; object-level callers own metadata."""
    factors = apply_mpo_uncompressed(mpo_factors, mps_factors)
    return compress(factors, int(max_bond), renormalize=False)


def product_site_factors(left, right):
    """Compose two raw MPO site tensors as ``left @ right``."""
    if left.shape[3] != right.shape[2]:
        raise ValueError(
            "MPO physical composition mismatch: "
            f"{left.shape[3]} != {right.shape[2]}."
        )
    return np.reshape(
        np.einsum("abst,cdtu->acbdsu", left, right, optimize=True),
        (
            left.shape[0] * right.shape[0],
            left.shape[1] * right.shape[1],
            left.shape[2],
            right.shape[3],
        ),
    )


def product_mpo_factors(left_factors, right_factors):
    """Compose equal-length raw MPO factor sequences."""
    left_factors = tuple(left_factors)
    right_factors = tuple(right_factors)
    if len(left_factors) != len(right_factors):
        raise ValueError(
            "MPO lengths must match; got "
            f"{len(left_factors)} and {len(right_factors)}."
        )
    if not left_factors:
        raise ValueError("MPOs must contain at least one site.")
    return [
        product_site_factors(left, right)
        for left, right in zip(left_factors, right_factors)
    ]


def product_mpo_compressed(left_factors, right_factors, max_bond, *, rtol=0.):
    """Sequential local-SVD composition, without full product-bond cores.

    Operands are first right-canonicalized without truncation so arbitrary
    virtual-bond scales cannot decide which channels survive the local SVD.
    Incoming product bonds are contracted with the previous truncated transfer
    before forming each core. This is a local truncation approximation, not
    globally optimal MPO rounding; convergence with max_bond must be checked.
    QR changes only the virtual gauge; no normalization or polar projection
    is applied. This remains an approximate product compression, not a global
    error guarantee.
    """
    if int(max_bond) < 1:
        raise ValueError('max_bond must be positive')
    if not np.isfinite(rtol) or rtol<0:raise ValueError('rtol must be nonnegative and finite')
    def right_canonical(factors):
        factors = [np.asarray(x).copy() for x in factors]
        for site in range(len(factors)-1, 0, -1):
            core = factors[site]
            l, r, dout, din = core.shape
            q, transfer = np.linalg.qr(core.reshape(l, -1).T, mode='reduced')
            factors[site] = q.T.reshape(-1, r, dout, din)
            factors[site-1] = np.einsum('abij,bc->acij', factors[site-1], transfer.T)
        return factors

    left = right_canonical(left_factors)
    right = right_canonical(right_factors)
    if not left or len(left) != len(right):
        raise ValueError('MPOs must have matching nonzero lengths')
    transfer = np.ones((1, 1, 1), dtype=np.result_type(*left, *right))
    output = []
    for site, (a, b) in enumerate(zip(left, right)):
        if a.shape[3] != b.shape[2]:
            raise ValueError('MPO physical composition mismatch')
        # Contract the incoming transfer first; never form a*b bond cores.
        partial = np.einsum('kab,arij->kbrij', transfer, a, optimize=True)
        core = np.einsum('kbrij,bsjl->kilrs', partial, b, optimize=True)
        del partial
        k, dout, din, ra, rb = core.shape
        matrix = core.reshape(k*dout*din, ra*rb)
        if site == len(left)-1:
            output.append(matrix.reshape(k, dout, din, ra*rb).transpose(0, 3, 1, 2))
        else:
            u, s, vh = np.linalg.svd(matrix, full_matrices=False)
            rank = min(int(max_bond), len(s))
            if rtol:
                rank=min(rank,max(1,int(np.count_nonzero(s>rtol*s[0]))))
            output.append(u[:, :rank].reshape(k, dout, din, rank).transpose(0, 3, 1, 2).copy())
            transfer = (s[:rank, None]*vh[:rank]).reshape(rank, ra, rb)
    return output


__all__ = [
    "apply_mpo_factors",
    "apply_mpo_uncompressed",
    "product_mpo_factors",
    "product_site_factors",
]
