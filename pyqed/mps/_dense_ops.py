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
        if getattr(operator, "ndim", None) != 4:
            raise ValueError(f"MPO site {site} must have rank 4.")
        if getattr(state, "ndim", None) != 3:
            raise ValueError(f"MPS site {site} must have rank 3.")

        op_left, op_right, output_dim, op_input_dim = operator.shape
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
            "abij,kjl->kailb", operator, state, optimize=True
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


__all__ = [
    "apply_mpo_factors",
    "apply_mpo_uncompressed",
    "product_mpo_factors",
    "product_site_factors",
]
