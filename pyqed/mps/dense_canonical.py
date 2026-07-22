"""Standalone dense-MPS canonicalization helpers used by legacy callers."""

from __future__ import annotations

import numpy as np


def left_canonical(factors):
    """Return a normalized left-canonical copy of ``factors``."""
    result = [np.asarray(tensor).copy() for tensor in factors]
    for site, tensor in enumerate(result):
        left, physical, right = tensor.shape
        u, singular_values, vh = np.linalg.svd(
            tensor.reshape(left * physical, right),
            full_matrices=False,
        )
        result[site] = u.reshape(left, physical, u.shape[1])
        if site + 1 < len(result):
            transfer = singular_values[:, None] * vh
            result[site + 1] = np.tensordot(
                transfer, result[site + 1], axes=([1], [0])
            )
        else:
            result[site] = (u @ vh).reshape(left, physical, right)
    return result


def right_canonical(factors):
    """Return a normalized right-canonical copy of ``factors``."""
    result = [np.asarray(tensor).copy() for tensor in factors]
    for site in range(len(result) - 1, -1, -1):
        tensor = result[site]
        left, physical, right = tensor.shape
        u, singular_values, vh = np.linalg.svd(
            tensor.reshape(left, physical * right),
            full_matrices=False,
        )
        result[site] = vh.reshape(vh.shape[0], physical, right)
        if site:
            transfer = u * singular_values[None, :]
            result[site - 1] = np.tensordot(
                result[site - 1], transfer, axes=([2], [0])
            )
        else:
            result[site] = (u @ vh).reshape(left, physical, right)
    return result


# Historical public names.
LeftCanonical = left_canonical
RightCanonical = right_canonical

__all__ = ["LeftCanonical", "RightCanonical", "left_canonical", "right_canonical"]
