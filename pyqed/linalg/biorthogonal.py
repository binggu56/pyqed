"""Balanced pivoted LU factors for biorthonormal orbital bases.

Shared by CI and reduced tensor-network counter-transformations.
"""

import numpy as np
from scipy.linalg import lu, solve_triangular


def orbital_factors(metric):
    """Return orbital maps XL, XR and inverse coefficient maps A, B.

    More precisely XL=A^-1, XR=B^-1, A† B=s. No pseudoinverse is used:
    numerically singular cross metrics cannot define a biorthonormal basis.
    """
    n = len(metric)
    singular = np.linalg.svd(metric, compute_uv=False)
    if singular[-1] <= np.finfo(float).eps * n * singular[0]:
        raise np.linalg.LinAlgError('A singular orbital metric has no biorthonormal basis.')
    p, lower, upper = lu(metric)
    scale = np.sqrt(np.abs(np.diag(upper)))
    a = scale[:, None] * lower.conj().T @ p.conj().T
    b = upper / scale[:, None]
    xl = p @ solve_triangular(lower.conj().T, np.diag(1/scale), lower=False, unit_diagonal=True)
    xr = solve_triangular(upper, np.diag(scale), lower=False)
    return xl, xr, a, b
