#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone Rys-quadrature utilities for Gaussian ERIs.

This module currently implements a self-contained Rys prototype for the
primitive and contracted ``(s s|s s)`` electron-repulsion integral.  The root
and weight builder is written independently from PySCF/libcint and is intended
as a foundation for a future higher-angular-momentum kernel.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

try:
    from . import _rys_cy
except Exception:  # pragma: no cover - optional accelerator
    _rys_cy = None


PI = math.pi
ERI_PREFAC = 2.0 * (PI ** 2.5)


def _double_factorial(n: int) -> int:
    if n <= 0:
        return 1
    out = 1
    k = n
    while k > 0:
        out *= k
        k -= 2
    return out


@lru_cache(maxsize=16384)
def _weighted_moment(order: int, T: float) -> float:
    """
    Return ``mu_order(T) = int_0^1 u^order exp(-T u^2) du``.

    Even moments are exact Boys integrals. Odd moments are needed to build the
    orthogonal-polynomial recurrence for the quadrature rule.
    """
    if T < 1.0e-14:
        return 1.0 / (order + 1.0)

    if order % 2 == 0:
        n = order // 2
        return boys(n, T)

    # Reduce odd moments to the lower incomplete gamma function recurrence.
    # For order = 2m+1:
    #   int_0^1 u^(2m+1) exp(-T u^2) du = 1/(2 T^(m+1)) * gamma(m+1, 0, T)
    m = (order - 1) // 2
    series = 0.0
    term = 1.0
    for k in range(m + 1):
        if k:
            term *= T / k
        series += term
    return 0.5 * math.factorial(m) * (1.0 - math.exp(-T) * series) / (T ** (m + 1))


@lru_cache(maxsize=16384)
def boys(n: int, T: float) -> float:
    """
    Boys function reference for the standalone Rys prototype.

    Current scope only needs low-order values for the ``(s s|s s)`` kernel, so
    use the exact ``F_0`` expression plus upward recurrence.
    """
    if T < 1.0e-14:
        return 1.0 / (2.0 * n + 1.0)

    if T < 1.0e-8:
        value = 1.0 / (2.0 * n + 1.0)
        term = 1.0
        for k in range(1, 80):
            term *= -T / k
            add = term / (2.0 * n + 2.0 * k + 1.0)
            value += add
            if abs(add) < 1.0e-18:
                break
        return value

    sqrt_T = math.sqrt(T)
    value = 0.5 * math.sqrt(PI / T) * math.erf(sqrt_T)
    if n == 0:
        return value

    eT = math.exp(-T)
    for m in range(n):
        value = ((2.0 * m + 1.0) * value - eT) / (2.0 * T)
    return value


def _stieltjes_recurrence(nroots: int, T: float):
    """
    Build the Jacobi-matrix recurrence coefficients for the weight
    ``w_T(u) = exp(-T u^2)`` on ``[0, 1]``.
    """
    if nroots <= 0:
        raise ValueError("nroots must be positive.")

    # Gram-Schmidt on monomials using exact weighted moments.
    maxdeg = nroots
    poly = np.zeros((nroots + 1, maxdeg + 1), dtype=float)
    norms = np.zeros(nroots + 1, dtype=float)
    alpha = np.zeros(nroots, dtype=float)
    beta = np.zeros(nroots, dtype=float)

    poly[0, 0] = 1.0
    norms[0] = _weighted_moment(0, T)

    for k in range(nroots):
        xp = np.zeros(maxdeg + 1, dtype=float)
        xp[1:] = poly[k, :-1]
        alpha[k] = _poly_inner(xp, poly[k], T) / norms[k]
        q = xp - alpha[k] * poly[k]
        if k > 0:
            q -= beta[k - 1] * poly[k - 1]
        if k + 1 <= nroots:
            poly[k + 1] = q
            norms[k + 1] = _poly_inner(q, q, T)
            if norms[k + 1] <= 0.0:
                raise ValueError("Rys quadrature construction failed: nonpositive norm.")
            if k < nroots - 1:
                beta[k] = norms[k + 1] / norms[k]

    return alpha, beta, norms[0]


def _poly_inner(a: np.ndarray, b: np.ndarray, T: float) -> float:
    deg = len(a) + len(b) - 2
    out = 0.0
    for i, ai in enumerate(a):
        if ai == 0.0:
            continue
        for j, bj in enumerate(b):
            if bj == 0.0:
                continue
            out += ai * bj * _weighted_moment(i + j, T)
    return out


@lru_cache(maxsize=8192)
def rys_roots_weights(nroots: int, T: float):
    """
    Gaussian quadrature nodes/weights for the Rys weight ``exp(-T u^2)`` on
    ``[0, 1]``.
    """
    if int(nroots) == 1:
        mu0 = _weighted_moment(0, float(T))
        mu1 = _weighted_moment(1, float(T))
        return np.asarray([mu1 / mu0], dtype=float), np.asarray([mu0], dtype=float)

    alpha, beta, mu0 = _stieltjes_recurrence(int(nroots), float(T))
    J = np.diag(alpha)
    for k in range(nroots - 1):
        off = math.sqrt(beta[k])
        J[k, k + 1] = off
        J[k + 1, k] = off
    roots, vecs = np.linalg.eigh(J)
    weights = mu0 * (vecs[0, :] ** 2)
    return roots, weights


@lru_cache(maxsize=131072)
def _primitive_eri_ssss_rys_cached(
    a: float,
    Ax: float,
    Ay: float,
    Az: float,
    b: float,
    Bx: float,
    By: float,
    Bz: float,
    c: float,
    Cx: float,
    Cy: float,
    Cz: float,
    d: float,
    Dx: float,
    Dy: float,
    Dz: float,
):
    """
    Primitive ``(s s|s s)`` ERI via a 1-root Rys quadrature rule.
    """
    p = a + b
    q = c + d
    alpha = p * q / (p + q)

    Px = (a * Ax + b * Bx) / p
    Py = (a * Ay + b * By) / p
    Pz = (a * Az + b * Bz) / p
    Qx = (c * Cx + d * Dx) / q
    Qy = (c * Cy + d * Dy) / q
    Qz = (c * Cz + d * Dz) / q

    AB2 = (Ax - Bx) ** 2 + (Ay - By) ** 2 + (Az - Bz) ** 2
    CD2 = (Cx - Dx) ** 2 + (Cy - Dy) ** 2 + (Cz - Dz) ** 2
    PQ2 = (Px - Qx) ** 2 + (Py - Qy) ** 2 + (Pz - Qz) ** 2

    Kab = math.exp(-(a * b / p) * AB2)
    Kcd = math.exp(-(c * d / q) * CD2)
    T = alpha * PQ2

    # For ``(s s|s s)``, the polynomial factor is constant, so the 1-root
    # quadrature reduces to the weight ``F_0(T)``.
    return ERI_PREFAC * Kab * Kcd * boys(0, T) / (p * q * math.sqrt(p + q))


def primitive_eri_ssss_rys(
    a: float,
    A,
    b: float,
    B,
    c: float,
    C,
    d: float,
    D,
):
    return _primitive_eri_ssss_rys_cached(
        float(a),
        float(A[0]),
        float(A[1]),
        float(A[2]),
        float(b),
        float(B[0]),
        float(B[1]),
        float(B[2]),
        float(c),
        float(C[0]),
        float(C[1]),
        float(C[2]),
        float(d),
        float(D[0]),
        float(D[1]),
        float(D[2]),
    )


_P_SHELL_TO_AXIS = {
    (1, 0, 0): 0,
    (0, 1, 0): 1,
    (0, 0, 1): 2,
}

_VEC_NAME_TO_ID = {"AB": 0, "CD": 1, "PQ": 2}


def _p_axis(shell) -> int:
    shell = tuple(int(x) for x in shell)
    if shell not in _P_SHELL_TO_AXIS:
        raise NotImplementedError("Standalone Rys prototype currently supports only Cartesian p shells.")
    return _P_SHELL_TO_AXIS[shell]


def _primitive_ssss_common(a: float, A, b: float, B, c: float, C, d: float, D):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    D = np.asarray(D, dtype=float)

    p = a + b
    q = c + d
    alpha = p * q / (p + q)
    mu_ab = a * b / p
    mu_cd = c * d / q
    lam_a = a / p
    lam_b = b / p
    lam_c = c / q
    lam_d = d / q

    P = (a * A + b * B) / p
    Q = (c * C + d * D) / q
    AB = A - B
    CD = C - D
    PQ = P - Q

    AB2 = float(np.dot(AB, AB))
    CD2 = float(np.dot(CD, CD))
    T = alpha * float(np.dot(PQ, PQ))

    pref = ERI_PREFAC * math.exp(-mu_ab * AB2) * math.exp(-mu_cd * CD2) / (p * q * math.sqrt(p + q))
    return {
        "A": A,
        "B": B,
        "C": C,
        "D": D,
        "p": p,
        "q": q,
        "alpha": alpha,
        "mu_ab": mu_ab,
        "mu_cd": mu_cd,
        "lam_a": lam_a,
        "lam_b": lam_b,
        "lam_c": lam_c,
        "lam_d": lam_d,
        "AB": AB,
        "CD": CD,
        "PQ": PQ,
        "T": T,
        "pref": pref,
    }


def _base_ssss_terms():
    return {0: [(1.0, tuple())]}


def _vector_derivative_spec(center: str, params):
    if center == "A":
        return {"AB": 1.0, "CD": 0.0, "PQ": params["lam_a"]}
    if center == "B":
        return {"AB": -1.0, "CD": 0.0, "PQ": params["lam_b"]}
    if center == "C":
        return {"AB": 0.0, "CD": 1.0, "PQ": -params["lam_c"]}
    if center == "D":
        return {"AB": 0.0, "CD": -1.0, "PQ": -params["lam_d"]}
    raise ValueError(f"Unknown center {center!r}")


def _log_prefactor_spec(center: str, params):
    if center == "A":
        return params["mu_ab"], "AB", -2.0
    if center == "B":
        return params["mu_ab"], "AB", 2.0
    if center == "C":
        return params["mu_cd"], "CD", -2.0
    if center == "D":
        return params["mu_cd"], "CD", 2.0
    raise ValueError(f"Unknown center {center!r}")


def _dt_spec(center: str, params):
    if center == "A":
        return 2.0 * params["alpha"] * params["lam_a"]
    if center == "B":
        return 2.0 * params["alpha"] * params["lam_b"]
    if center == "C":
        return -2.0 * params["alpha"] * params["lam_c"]
    if center == "D":
        return -2.0 * params["alpha"] * params["lam_d"]
    raise ValueError(f"Unknown center {center!r}")


def _promotion_scale(center: str, exponents):
    return 1.0 / (2.0 * exponents[center])


def _promote_terms(terms_by_order, center: str, params, exponents, new_axis: int):
    promoted = {}
    vec_diff = _vector_derivative_spec(center, params)
    pref_mu, pref_vec, pref_sign = _log_prefactor_spec(center, params)
    dt_coeff = _dt_spec(center, params)
    scale = _promotion_scale(center, exponents)

    for order, terms in terms_by_order.items():
        for scalar, factors in terms:
            promoted.setdefault(order, []).append(
                (scale * scalar * pref_sign * pref_mu, factors + (("vec", new_axis, pref_vec),))
            )

            for idx, factor in enumerate(factors):
                if factor[0] != "vec":
                    continue
                _, axis, name = factor
                coeff = vec_diff.get(name, 0.0)
                if coeff == 0.0:
                    continue
                new_factors = list(factors)
                new_factors[idx] = ("delta", axis, new_axis)
                promoted.setdefault(order, []).append((scale * scalar * coeff, tuple(new_factors)))

            promoted.setdefault(order + 1, []).append(
                (scale * scalar * (-dt_coeff), factors + (("vec", new_axis, "PQ"),))
            )

    return promoted


def _evaluate_factors(factors, rank, vectors):
    shape = (3,) * rank
    if rank == 0:
        value = 1.0
    else:
        value = np.ones(shape, dtype=float)
    for factor in factors:
        kind = factor[0]
        if kind == "vec":
            _, axis, name = factor
            arr = np.asarray(vectors[name], dtype=float)
            reshape = [1] * rank
            reshape[axis] = 3
            value = value * arr.reshape(reshape)
        elif kind == "delta":
            _, ax1, ax2 = factor
            arr = np.eye(3, dtype=float)
            reshape = [1] * rank
            reshape[ax1] = 3
            reshape[ax2] = 3
            value = value * arr.reshape(reshape)
        else:
            raise ValueError(f"Unknown factor kind {kind!r}")
    return value


def _build_block_from_terms(terms_by_order, params, rank):
    vectors = {"AB": params["AB"], "CD": params["CD"], "PQ": params["PQ"]}
    max_order = max(terms_by_order)
    boys_values = [boys(n, params["T"]) for n in range(max_order + 1)]
    if rank == 0:
        block = 0.0
    else:
        block = np.zeros((3,) * rank, dtype=float)
    for order, terms in terms_by_order.items():
        for scalar, factors in terms:
            block = block + scalar * boys_values[order] * _evaluate_factors(factors, rank, vectors)
    return params["pref"] * block


def _freeze_terms(terms_by_order):
    max_factors = max((len(factors) for terms in terms_by_order.values() for _scalar, factors in terms), default=0)
    entries = []
    for order in sorted(terms_by_order):
        for scalar, factors in terms_by_order[order]:
            vecs = []
            deltas = []
            for factor in factors:
                if factor[0] == "vec":
                    _kind, axis, name = factor
                    vecs.append((axis, _VEC_NAME_TO_ID[name]))
                elif factor[0] == "delta":
                    _kind, ax1, ax2 = factor
                    deltas.append((ax1, ax2))
                else:
                    raise ValueError(f"Unknown factor kind {factor[0]!r}")
            entries.append((order, scalar, vecs, deltas))

    nterms = len(entries)
    orders = np.zeros(nterms, dtype=np.int64)
    scalars = np.zeros(nterms, dtype=float)
    nvec = np.zeros(nterms, dtype=np.int64)
    vec_axes = np.full((nterms, max_factors), -1, dtype=np.int64)
    vec_names = np.full((nterms, max_factors), -1, dtype=np.int64)
    ndelta = np.zeros(nterms, dtype=np.int64)
    delta_axis1 = np.full((nterms, max_factors), -1, dtype=np.int64)
    delta_axis2 = np.full((nterms, max_factors), -1, dtype=np.int64)

    for idx, (order, scalar, vecs, deltas) in enumerate(entries):
        orders[idx] = order
        scalars[idx] = scalar
        nvec[idx] = len(vecs)
        ndelta[idx] = len(deltas)
        for j, (axis, name_id) in enumerate(vecs):
            vec_axes[idx, j] = axis
            vec_names[idx, j] = name_id
        for j, (ax1, ax2) in enumerate(deltas):
            delta_axis1[idx, j] = ax1
            delta_axis2[idx, j] = ax2

    return {
        "orders": orders,
        "scalars": scalars,
        "nvec": nvec,
        "vec_axes": vec_axes,
        "vec_names": vec_names,
        "ndelta": ndelta,
        "delta_axis1": delta_axis1,
        "delta_axis2": delta_axis2,
    }


def _evaluate_block_dispatch(rank, terms_by_order, params):
    if _rys_cy is None:
        return _build_block_from_terms(terms_by_order, params, rank)

    key = tuple(sorted((order, tuple((scalar, factors) for scalar, factors in terms)) for order, terms in terms_by_order.items()))
    table = _TERM_TABLE_CACHE.get(key)
    if table is None:
        table = _freeze_terms(terms_by_order)
        _TERM_TABLE_CACHE[key] = table
    boys_values = np.asarray([boys(n, params["T"]) for n in range(int(np.max(table["orders"])) + 1)], dtype=float)
    return _rys_cy.evaluate_block(
        rank,
        table["orders"],
        table["scalars"],
        table["nvec"],
        table["vec_axes"],
        table["vec_names"],
        table["ndelta"],
        table["delta_axis1"],
        table["delta_axis2"],
        boys_values,
        float(params["pref"]),
        np.asarray(params["AB"], dtype=float),
        np.asarray(params["CD"], dtype=float),
        np.asarray(params["PQ"], dtype=float),
    )


_TERM_TABLE_CACHE = {}


def _evaluate_promoted_block_dispatch(centers, params, exponents):
    rank = len(centers)
    if _rys_cy is None:
        terms = _base_ssss_terms()
        for axis, center in enumerate(centers):
            terms = _promote_terms(terms, center, params, exponents, axis)
        return _build_block_from_terms(terms, params, rank)

    center_ids = np.asarray([{"A": 0, "B": 1, "C": 2, "D": 3}[c] for c in centers], dtype=np.int64)
    exponent_vec = np.asarray([exponents[c] for c in centers], dtype=float)
    boys_values = np.asarray([boys(n, params["T"]) for n in range(rank + 1)], dtype=float)
    return _rys_cy.evaluate_promoted_block(
        center_ids,
        exponent_vec,
        float(params["alpha"]),
        float(params["mu_ab"]),
        float(params["mu_cd"]),
        float(params["lam_a"]),
        float(params["lam_b"]),
        float(params["lam_c"]),
        float(params["lam_d"]),
        boys_values,
        float(params["pref"]),
        np.asarray(params["AB"], dtype=float),
        np.asarray(params["CD"], dtype=float),
        np.asarray(params["PQ"], dtype=float),
    )


def primitive_eri_psss_block_rys(
    a: float,
    A,
    b: float,
    B,
    c: float,
    C,
    d: float,
    D,
):
    """
    Primitive Cartesian ``(p s|s s)`` block on center ``A``.
    """
    params = _primitive_ssss_common(a, A, b, B, c, C, d, D)
    if _rys_cy is not None:
        boys_values = np.asarray([boys(0, params["T"]), boys(1, params["T"])], dtype=float)
        return _rys_cy.evaluate_psss(
            float(params["pref"]),
            float(params["mu_ab"]),
            float(params["alpha"]),
            float(params["lam_a"]),
            float(a),
            boys_values,
            np.asarray(params["AB"], dtype=float),
            np.asarray(params["PQ"], dtype=float),
        )
    return params["pref"] * (-params["mu_ab"] * params["AB"] * boys(0, params["T"]) - params["alpha"] * params["lam_a"] * params["PQ"] * boys(1, params["T"])) / a


def primitive_eri_psss_rys(
    shell_a,
    a: float,
    A,
    b: float,
    B,
    c: float,
    C,
    d: float,
    D,
):
    """
    Primitive Cartesian ``(p s|s s)`` scalar integral selected by ``shell_a``.
    """
    axis = _p_axis(shell_a)
    params = _primitive_ssss_common(a, A, b, B, c, C, d, D)
    if _rys_cy is not None:
        boys_values = np.asarray([boys(0, params["T"]), boys(1, params["T"])], dtype=float)
        return _rys_cy.evaluate_psss_scalar(
            float(params["pref"]),
            float(params["mu_ab"]),
            float(params["alpha"]),
            float(params["lam_a"]),
            float(a),
            boys_values,
            np.asarray(params["AB"], dtype=float),
            np.asarray(params["PQ"], dtype=float),
            int(axis),
        )
    return primitive_eri_psss_block_rys(a, A, b, B, c, C, d, D)[axis]


def primitive_eri_ppss_block_rys(
    a: float,
    A,
    b: float,
    B,
    c: float,
    C,
    d: float,
    D,
):
    """
    Primitive Cartesian ``(p p|s s)`` block on centers ``A`` and ``B``.
    """
    params = _primitive_ssss_common(a, A, b, B, c, C, d, D)
    if _rys_cy is not None:
        boys_values = np.asarray([boys(0, params["T"]), boys(1, params["T"]), boys(2, params["T"])], dtype=float)
        return _rys_cy.evaluate_ppss(
            float(params["pref"]),
            float(params["mu_ab"]),
            float(params["alpha"]),
            float(params["lam_a"]),
            float(params["lam_b"]),
            float(a),
            float(b),
            boys_values,
            np.asarray(params["AB"], dtype=float),
            np.asarray(params["PQ"], dtype=float),
        )
    delta = np.eye(3)
    f0 = boys(0, params["T"])
    f1 = boys(1, params["T"])
    f2 = boys(2, params["T"])
    term0 = (2.0 * params["mu_ab"] * delta - 4.0 * (params["mu_ab"] ** 2) * np.outer(params["AB"], params["AB"])) * f0
    term1 = (
        4.0 * params["mu_ab"] * params["alpha"] * params["lam_b"] * np.outer(params["AB"], params["PQ"])
        - 4.0 * params["mu_ab"] * params["alpha"] * params["lam_a"] * np.outer(params["PQ"], params["AB"])
        - 2.0 * params["alpha"] * params["lam_a"] * params["lam_b"] * delta
    ) * f1
    term2 = 4.0 * (params["alpha"] ** 2) * params["lam_a"] * params["lam_b"] * np.outer(params["PQ"], params["PQ"]) * f2
    return params["pref"] * (term0 + term1 + term2) / (4.0 * a * b)


def primitive_eri_ppss_rys(
    shell_a,
    a: float,
    A,
    shell_b,
    b: float,
    B,
    c: float,
    C,
    d: float,
    D,
):
    """
    Primitive Cartesian ``(p p|s s)`` scalar integral selected by ``shell_a``
    and ``shell_b``.
    """
    ia = _p_axis(shell_a)
    ib = _p_axis(shell_b)
    params = _primitive_ssss_common(a, A, b, B, c, C, d, D)
    if _rys_cy is not None:
        boys_values = np.asarray([boys(0, params["T"]), boys(1, params["T"]), boys(2, params["T"])], dtype=float)
        return _rys_cy.evaluate_ppss_scalar(
            float(params["pref"]),
            float(params["mu_ab"]),
            float(params["alpha"]),
            float(params["lam_a"]),
            float(params["lam_b"]),
            float(a),
            float(b),
            boys_values,
            np.asarray(params["AB"], dtype=float),
            np.asarray(params["PQ"], dtype=float),
            int(ia),
            int(ib),
        )
    return primitive_eri_ppss_block_rys(a, A, b, B, c, C, d, D)[ia, ib]


def primitive_eri_psps_block_rys(
    a: float,
    A,
    b: float,
    B,
    c: float,
    C,
    d: float,
    D,
):
    """
    Primitive Cartesian ``(p s|p s)`` block on centers ``A`` and ``C``.
    """
    params = _primitive_ssss_common(a, A, b, B, c, C, d, D)
    if _rys_cy is not None:
        boys_values = np.asarray([boys(0, params["T"]), boys(1, params["T"]), boys(2, params["T"])], dtype=float)
        return _rys_cy.evaluate_psps(
            float(params["pref"]),
            float(params["mu_ab"]),
            float(params["mu_cd"]),
            float(params["alpha"]),
            float(params["lam_a"]),
            float(params["lam_c"]),
            float(a),
            float(c),
            boys_values,
            np.asarray(params["AB"], dtype=float),
            np.asarray(params["CD"], dtype=float),
            np.asarray(params["PQ"], dtype=float),
        )
    f0 = boys(0, params["T"])
    f1 = boys(1, params["T"])
    f2 = boys(2, params["T"])
    delta = np.eye(3)
    term0 = params["mu_ab"] * params["mu_cd"] * np.outer(params["AB"], params["CD"]) * f0
    term1 = (
        params["alpha"] * params["lam_a"] * params["mu_cd"] * np.outer(params["PQ"], params["CD"])
        - params["alpha"] * params["lam_c"] * params["mu_ab"] * np.outer(params["AB"], params["PQ"])
        + 0.5 * params["alpha"] * params["lam_a"] * params["lam_c"] * delta
    ) * f1
    term2 = -(params["alpha"] ** 2) * params["lam_a"] * params["lam_c"] * np.outer(params["PQ"], params["PQ"]) * f2
    return params["pref"] * (term0 + term1 + term2) / (a * c)


def primitive_eri_psps_rys(
    shell_a,
    a: float,
    A,
    b: float,
    B,
    shell_c,
    c: float,
    C,
    d: float,
    D,
):
    """
    Primitive Cartesian ``(p s|p s)`` scalar integral selected by ``shell_a``
    and ``shell_c``.
    """
    ia = _p_axis(shell_a)
    ic = _p_axis(shell_c)
    params = _primitive_ssss_common(a, A, b, B, c, C, d, D)
    if _rys_cy is not None:
        boys_values = np.asarray([boys(0, params["T"]), boys(1, params["T"]), boys(2, params["T"])], dtype=float)
        return _rys_cy.evaluate_psps_scalar(
            float(params["pref"]),
            float(params["mu_ab"]),
            float(params["mu_cd"]),
            float(params["alpha"]),
            float(params["lam_a"]),
            float(params["lam_c"]),
            float(a),
            float(c),
            boys_values,
            np.asarray(params["AB"], dtype=float),
            np.asarray(params["CD"], dtype=float),
            np.asarray(params["PQ"], dtype=float),
            int(ia),
            int(ic),
        )
    return primitive_eri_psps_block_rys(a, A, b, B, c, C, d, D)[ia, ic]


def primitive_eri_ppps_block_rys(
    a: float,
    A,
    b: float,
    B,
    c: float,
    C,
    d: float,
    D,
):
    """
    Primitive Cartesian ``(p p|p s)`` block on centers ``A``, ``B``, ``C``.
    """
    params = _primitive_ssss_common(a, A, b, B, c, C, d, D)
    if _rys_cy is not None:
        boys_values = np.asarray([boys(0, params["T"]), boys(1, params["T"]), boys(2, params["T"]), boys(3, params["T"])], dtype=float)
        return _rys_cy.evaluate_ppps_specialized(
            float(params["pref"]),
            float(params["alpha"]),
            float(params["mu_ab"]),
            float(params["mu_cd"]),
            float(params["lam_a"]),
            float(params["lam_b"]),
            float(params["lam_c"]),
            float(params["lam_d"]),
            float(a),
            float(b),
            float(c),
            boys_values,
            np.asarray(params["AB"], dtype=float),
            np.asarray(params["CD"], dtype=float),
            np.asarray(params["PQ"], dtype=float),
        )
    return _evaluate_promoted_block_dispatch(("A", "B", "C"), params, {"A": a, "B": b, "C": c, "D": d})


def primitive_eri_ppps_rys(
    shell_a,
    a: float,
    A,
    shell_b,
    b: float,
    B,
    shell_c,
    c: float,
    C,
    d: float,
    D,
):
    ia = _p_axis(shell_a)
    ib = _p_axis(shell_b)
    ic = _p_axis(shell_c)
    params = _primitive_ssss_common(a, A, b, B, c, C, d, D)
    if _rys_cy is not None:
        boys_values = np.asarray([boys(0, params["T"]), boys(1, params["T"]), boys(2, params["T"]), boys(3, params["T"])], dtype=float)
        return _rys_cy.evaluate_ppps_scalar(
            float(params["pref"]),
            float(params["alpha"]),
            float(params["mu_ab"]),
            float(params["mu_cd"]),
            float(params["lam_a"]),
            float(params["lam_b"]),
            float(params["lam_c"]),
            float(params["lam_d"]),
            float(a),
            float(b),
            float(c),
            boys_values,
            np.asarray(params["AB"], dtype=float),
            np.asarray(params["CD"], dtype=float),
            np.asarray(params["PQ"], dtype=float),
            int(ia),
            int(ib),
            int(ic),
        )
    return primitive_eri_ppps_block_rys(a, A, b, B, c, C, d, D)[ia, ib, ic]


def primitive_eri_pppp_block_rys(
    a: float,
    A,
    b: float,
    B,
    c: float,
    C,
    d: float,
    D,
):
    """
    Primitive Cartesian ``(p p|p p)`` block on centers ``A``, ``B``, ``C``, ``D``.
    """
    params = _primitive_ssss_common(a, A, b, B, c, C, d, D)
    if _rys_cy is not None:
        boys_values = np.asarray([boys(0, params["T"]), boys(1, params["T"]), boys(2, params["T"]), boys(3, params["T"]), boys(4, params["T"])], dtype=float)
        return _rys_cy.evaluate_pppp_specialized(
            float(params["pref"]),
            float(params["alpha"]),
            float(params["mu_ab"]),
            float(params["mu_cd"]),
            float(params["lam_a"]),
            float(params["lam_b"]),
            float(params["lam_c"]),
            float(params["lam_d"]),
            float(a),
            float(b),
            float(c),
            float(d),
            boys_values,
            np.asarray(params["AB"], dtype=float),
            np.asarray(params["CD"], dtype=float),
            np.asarray(params["PQ"], dtype=float),
        )
    return _evaluate_promoted_block_dispatch(("A", "B", "C", "D"), params, {"A": a, "B": b, "C": c, "D": d})


def primitive_eri_pppp_rys(
    shell_a,
    a: float,
    A,
    shell_b,
    b: float,
    B,
    shell_c,
    c: float,
    C,
    shell_d,
    d: float,
    D,
):
    ia = _p_axis(shell_a)
    ib = _p_axis(shell_b)
    ic = _p_axis(shell_c)
    id_ = _p_axis(shell_d)
    params = _primitive_ssss_common(a, A, b, B, c, C, d, D)
    if _rys_cy is not None:
        boys_values = np.asarray([boys(0, params["T"]), boys(1, params["T"]), boys(2, params["T"]), boys(3, params["T"]), boys(4, params["T"])], dtype=float)
        return _rys_cy.evaluate_pppp_scalar(
            float(params["pref"]),
            float(params["alpha"]),
            float(params["mu_ab"]),
            float(params["mu_cd"]),
            float(params["lam_a"]),
            float(params["lam_b"]),
            float(params["lam_c"]),
            float(params["lam_d"]),
            float(a),
            float(b),
            float(c),
            float(d),
            boys_values,
            np.asarray(params["AB"], dtype=float),
            np.asarray(params["CD"], dtype=float),
            np.asarray(params["PQ"], dtype=float),
            int(ia),
            int(ib),
            int(ic),
            int(id_),
        )
    return primitive_eri_pppp_block_rys(a, A, b, B, c, C, d, D)[ia, ib, ic, id_]


def contracted_eri_ssss_rys(a, b, c, d):
    """
    Contracted ``(s s|s s)`` ERI via the primitive Rys kernel.

    Arguments are ``BasisFunction``-like objects with ``shell``, ``origin``,
    ``exps`` and ``prim_weights``.
    """
    if tuple(a.shell) != (0, 0, 0) or tuple(b.shell) != (0, 0, 0) or tuple(c.shell) != (0, 0, 0) or tuple(d.shell) != (0, 0, 0):
        raise NotImplementedError("Standalone Rys prototype currently supports only s shells.")

    eri = 0.0
    for ia, wa in enumerate(a.prim_weights):
        for ib, wb in enumerate(b.prim_weights):
            for ic, wc in enumerate(c.prim_weights):
                for id_, wd in enumerate(d.prim_weights):
                    eri += (
                        wa
                        * wb
                        * wc
                        * wd
                        * primitive_eri_ssss_rys(
                            a.exps[ia],
                            a.origin,
                            b.exps[ib],
                            b.origin,
                            c.exps[ic],
                            c.origin,
                            d.exps[id_],
                            d.origin,
                        )
                    )
    return eri


def contracted_eri_psss_rys(a, b, c, d):
    """
    Contracted Cartesian ``(p s|s s)`` scalar ERI.
    """
    if sum(a.shell) != 1 or tuple(b.shell) != (0, 0, 0) or tuple(c.shell) != (0, 0, 0) or tuple(d.shell) != (0, 0, 0):
        raise NotImplementedError("Standalone Rys prototype currently supports only (p s|s s) for this wrapper.")

    eri = 0.0
    for ia, wa in enumerate(a.prim_weights):
        for ib, wb in enumerate(b.prim_weights):
            for ic, wc in enumerate(c.prim_weights):
                for id_, wd in enumerate(d.prim_weights):
                    eri += (
                        wa
                        * wb
                        * wc
                        * wd
                        * primitive_eri_psss_rys(
                            a.shell,
                            a.exps[ia],
                            a.origin,
                            b.exps[ib],
                            b.origin,
                            c.exps[ic],
                            c.origin,
                            d.exps[id_],
                            d.origin,
                        )
                    )
    return eri


def contracted_eri_ppss_rys(a, b, c, d):
    """
    Contracted Cartesian ``(p p|s s)`` scalar ERI.
    """
    if sum(a.shell) != 1 or sum(b.shell) != 1 or tuple(c.shell) != (0, 0, 0) or tuple(d.shell) != (0, 0, 0):
        raise NotImplementedError("Standalone Rys prototype currently supports only (p p|s s) for this wrapper.")

    eri = 0.0
    for ia, wa in enumerate(a.prim_weights):
        for ib, wb in enumerate(b.prim_weights):
            for ic, wc in enumerate(c.prim_weights):
                for id_, wd in enumerate(d.prim_weights):
                    eri += (
                        wa
                        * wb
                        * wc
                        * wd
                        * primitive_eri_ppss_rys(
                            a.shell,
                            a.exps[ia],
                            a.origin,
                            b.shell,
                            b.exps[ib],
                            b.origin,
                            c.exps[ic],
                            c.origin,
                            d.exps[id_],
                            d.origin,
                        )
                    )
    return eri


def contracted_eri_psps_rys(a, b, c, d):
    """
    Contracted Cartesian ``(p s|p s)`` scalar ERI.
    """
    if sum(a.shell) != 1 or tuple(b.shell) != (0, 0, 0) or sum(c.shell) != 1 or tuple(d.shell) != (0, 0, 0):
        raise NotImplementedError("Standalone Rys prototype currently supports only (p s|p s) for this wrapper.")

    eri = 0.0
    for ia, wa in enumerate(a.prim_weights):
        for ib, wb in enumerate(b.prim_weights):
            for ic, wc in enumerate(c.prim_weights):
                for id_, wd in enumerate(d.prim_weights):
                    eri += (
                        wa
                        * wb
                        * wc
                        * wd
                        * primitive_eri_psps_rys(
                            a.shell,
                            a.exps[ia],
                            a.origin,
                            b.exps[ib],
                            b.origin,
                            c.shell,
                            c.exps[ic],
                            c.origin,
                            d.exps[id_],
                            d.origin,
                        )
                    )
    return eri


def contracted_eri_ppps_rys(a, b, c, d):
    """
    Contracted Cartesian ``(p p|p s)`` scalar ERI.
    """
    if sum(a.shell) != 1 or sum(b.shell) != 1 or sum(c.shell) != 1 or tuple(d.shell) != (0, 0, 0):
        raise NotImplementedError("Standalone Rys prototype currently supports only (p p|p s) for this wrapper.")

    eri = 0.0
    for ia, wa in enumerate(a.prim_weights):
        for ib, wb in enumerate(b.prim_weights):
            for ic, wc in enumerate(c.prim_weights):
                for id_, wd in enumerate(d.prim_weights):
                    eri += (
                        wa
                        * wb
                        * wc
                        * wd
                        * primitive_eri_ppps_rys(
                            a.shell,
                            a.exps[ia],
                            a.origin,
                            b.shell,
                            b.exps[ib],
                            b.origin,
                            c.shell,
                            c.exps[ic],
                            c.origin,
                            d.exps[id_],
                            d.origin,
                        )
                    )
    return eri


def contracted_eri_pppp_rys(a, b, c, d):
    """
    Contracted Cartesian ``(p p|p p)`` scalar ERI.
    """
    if sum(a.shell) != 1 or sum(b.shell) != 1 or sum(c.shell) != 1 or sum(d.shell) != 1:
        raise NotImplementedError("Standalone Rys prototype currently supports only (p p|p p) for this wrapper.")

    eri = 0.0
    for ia, wa in enumerate(a.prim_weights):
        for ib, wb in enumerate(b.prim_weights):
            for ic, wc in enumerate(c.prim_weights):
                for id_, wd in enumerate(d.prim_weights):
                    eri += (
                        wa
                        * wb
                        * wc
                        * wd
                        * primitive_eri_pppp_rys(
                            a.shell,
                            a.exps[ia],
                            a.origin,
                            b.shell,
                            b.exps[ib],
                            b.origin,
                            c.shell,
                            c.exps[ic],
                            c.origin,
                            d.shell,
                            d.exps[id_],
                            d.origin,
                        )
                    )
    return eri
