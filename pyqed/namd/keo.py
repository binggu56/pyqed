"""
Kinetic Energy Operator (KEO) Calculation using Vibrojet and JAX
===============================================================================
 https://github.com/robochimps/vibrojet/

"""


from numpy import kron

from pyqed.dvr.dvr_1d import SineDVR

try:
    from opt_einsum import contract
except ImportError:
    from numpy import einsum as contract
from typing import Callable
from functools import reduce
import operator
import jax
from jax import numpy as jnp

jax.config.update("jax_enable_x64", True)

EPS = jnp.array(
    [
        [[int((i - j) * (j - k) * (k - i) * 0.5) for k in range(3)] for j in range(3)]
        for i in range(3)
    ],
    dtype=jnp.float64,
)

import functools
from enum import Enum

import numpy as np
from jax import lax
from jax.core import ShapedArray
from jax.experimental import jet
from jax.extend.core import Primitive
from jax.interpreters import ad, batching, mlir



PARAMS = {
    "NO_ITERS_ECKART": 10,
    "EXP_TAYLOR_ORDER": 10,
    "EXP_TAYLOR_SQUARING": 4,
}


def set_params(**kw):
    for key, val in kw.items():
        if key in PARAMS:
            # print(f"set parameter '{key}' = {val}")
            PARAMS[key] = val
        else:
            raise KeyError(f"Unknown parameter name: {key}")


def fact(n):
    return lax.exp(lax.lgamma(n + 1.0))


# Define JAX.jet rules for missing primitive functions


##################
# matrix inversion
##################

inv_p = Primitive("_inv")


def inv(a, **kw):
    return inv_p.bind(a, **kw)


def inv_impl(a, **kw):
    return jnp.linalg.inv(a)
    # return jnp.linalg.pinv(a, hermitian=True)


inv_p.def_impl(inv_impl)


# @jax.jit
def inv_jvp(primals, tangents, **kw):
    (a,) = primals
    (da,) = tangents
    a_inv = inv(a)
    da_inv = -a_inv @ da @ a_inv
    return a_inv, da_inv


ad.primitive_jvps[inv_p] = inv_jvp


def inv_abstact_eval(a, **kw):
    shape = a.shape
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("Input to 'inv' must be a square matrix")
    N = shape[0]
    dtype = a.dtype
    return ShapedArray((N, N), dtype)


inv_p.def_abstract_eval(inv_abstact_eval)


def inv_lowering(ctx, a, **kw):
    return mlir.lower_fun(jnp.linalg.inv, multiple_results=False)(ctx, a)
    # return mlir.lower_fun(
    #     lambda x: jnp.linalg.pinv(x, hermitian=True), multiple_results=False
    # )(ctx, a)


mlir.register_lowering(inv_p, inv_lowering)


def inv_batch_rule(args, dims):
    (mat,) = args
    (dim,) = dims
    if dim is None:
        return inv_p(mat), None
    m = jax.vmap(jnp.linalg.inv)(mat)
    return m, dim


batching.primitive_batchers[inv_p] = inv_batch_rule


# @jax.jit
def _inverse_taylor_rule(primals_in, series_in, **kw):
    (x,) = primals_in
    (x_terms,) = series_in
    u = [x] + x_terms
    v = [None] * len(u)

    v[0] = inv(x)

    def scale(k, j):
        return 1.0 / (fact(k - j) * fact(j))

    for k in range(1, len(v)):
        v[k] = (
            -fact(k)
            * v[0]  # T?
            @ sum(scale(k, j) * u[j] @ v[k - j] for j in range(1, k + 1))
        )
    primal_out, *series_out = v
    return primal_out, series_out


jet.jet_rules[inv_p] = _inverse_taylor_rule


#################################
# matrix eigenvalue decomposition
#################################

eigh_p = Primitive("_eigh")


def eigh(a, **kw):
    e, v = eigh_p.bind(a, **kw)
    return e, v


def eigh_impl(a, **kw):
    e, v = jnp.linalg.eigh(a)
    return e, v


eigh_p.multiple_results = True
eigh_p.def_impl(eigh_impl)


# @jax.jit
def eigh_jvp(primals, tangents, **kw):
    (a,) = primals
    (da,) = tangents
    e, v = eigh(a)
    eye_ = jnp.eye(len(e))
    de = jnp.array([v[:, i] @ da @ v[:, i] for i in range(len(v))])
    inv_de = jnp.pow(e[:, None] - e[None, :] + eye_, -1) - eye_
    c = (v.T @ da @ v) * inv_de
    dv = -v @ c
    return (e, v), (de, dv)


ad.primitive_jvps[eigh_p] = eigh_jvp


def eigh_abstract_eval(a):
    shape = a.shape
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("Input to 'eigh' must be a square matrix")
    N = shape[0]
    dtype = a.dtype
    return ShapedArray((N,), dtype), ShapedArray((N, N), dtype)


eigh_p.def_abstract_eval(eigh_abstract_eval)


def eigh_lowering(ctx, a, **kw):
    return mlir.lower_fun(jnp.linalg.eigh, multiple_results=True)(ctx, a)


mlir.register_lowering(eigh_p, eigh_lowering)


def eigh_batch_rule(args, dims):
    (mat,) = args
    (dim,) = dims
    if dim is None:
        return eigh_p(mat), None
    e, v = jax.vmap(jnp.linalg.eigh)(mat)
    return (e, v), (dim, dim)


batching.primitive_batchers[eigh_p] = eigh_batch_rule


# @jax.jit
def _eigh_taylor_rule(primals_in, series_in, **kw):
    (x,) = primals_in
    (x_terms,) = series_in
    a = [x] + x_terms
    e = [None] * len(a)
    v = [None] * len(a)

    def scale(k, j):
        return 1.0 / (fact(k - j) * fact(j))

    e[0], v[0] = eigh(a[0])

    nprim, nvec = v[0].shape
    eye_ = jnp.eye(nprim)

    mi = [None] * nvec
    for i in range(nvec):
        m1 = jnp.concatenate((jnp.array([[0]]), v[0][:, i : i + 1].T), axis=-1)
        m2 = jnp.concatenate((v[0][:, i : i + 1], eye_ * e[0][i] - a[0]), axis=-1)
        m = jnp.concatenate((m1, m2), axis=0)
        mi[i] = inv(m)

    for k in range(1, len(a)):
        if k == 1:
            b1 = jnp.zeros((nvec, nvec), dtype=jnp.float64)
            b2 = a[k] @ v[0]
        else:
            b1 = (
                -0.5
                * fact(k)
                * sum(scale(k, m) * v[k - m].T @ v[m] for m in range(1, k))
            )
            b2 = fact(k) * (
                sum(scale(k, m) * a[k - m] @ v[m] for m in range(k))
                - sum(scale(k, m) * v[k - m] @ (eye_ * e[m]) for m in range(1, k))
            )

        e_, *v_ = jnp.array(
            [
                mi[i] @ jnp.concatenate((b1[i : i + 1, i], b2[:, i]), axis=0)
                for i in range(nvec)
            ]
        ).T
        e[k] = jnp.array(e_)
        v[k] = jnp.array(v_)

    e_primal_out, *e_series_out = e
    v_primal_out, *v_series_out = v
    return (e_primal_out, v_primal_out), (e_series_out, v_series_out)


jet.jet_rules[eigh_p] = _eigh_taylor_rule


#########################
# matrix LU decomposition
#########################

lu_p = Primitive("_lu")


def lu(a, **kw):
    l, u = lu_p.bind(a, **kw)
    return l, u


def lu_impl(a, **kw):
    l, u = jax.scipy.linalg.lu(a, permute_l=True)
    return l, u


lu_p.def_impl(lu_impl)
lu_p.multiple_results = True


# @jax.jit
def lu_jvp(primals, tangents, **kw):
    (a,) = primals
    (da,) = tangents
    l, u = lu(a)
    li = inv(l)
    ui = inv(u)
    f = li @ da @ ui
    du = jnp.triu(f) @ u
    dl = l @ jnp.tril(f, -1)
    return (l, u), (dl, du)


ad.primitive_jvps[lu_p] = lu_jvp


def lu_abstract_eval(a):
    shape = a.shape
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("Current implementation of 'lu' works only for square matrix")
    N = shape[0]
    dtype = a.dtype
    return (ShapedArray((N, N), dtype), ShapedArray((N, N), dtype))


lu_p.def_abstract_eval(lu_abstract_eval)


def lu_lowering(ctx, a, **kw):
    return mlir.lower_fun(
        lambda a: jax.scipy.linalg.lu(a, permute_l=True), multiple_results=True
    )(ctx, a)


mlir.register_lowering(lu_p, lu_lowering)


def lu_batch_rule(args, dims):
    (mat,) = args
    (dim,) = dims
    if dim is None:
        return lu_p(mat), None
    l, u = jax.vmap(lambda a: jax.scipy.linalg.lu(a, permute_l=True))(mat)
    return (l, u), (dim, dim)


batching.primitive_batchers[lu_p] = lu_batch_rule


# @jax.jit
def _lu_taylor_rule(primals_in, series_in, **kw):
    (x,) = primals_in
    (x_terms,) = series_in
    a = [x] + x_terms
    l = [None] * len(a)
    u = [None] * len(a)

    l[0], u[0] = lu(a[0])
    li = inv(l[0])
    ui = inv(u[0])

    def scale(k, j):
        return 1.0 / (fact(k - j) * fact(j))

    for k in range(1, len(a)):
        f = (
            li
            @ (a[k] - fact(k) * sum(scale(k, i) * l[i] @ u[k - i] for i in range(1, k)))
            @ ui
        )
        u[k] = jnp.triu(f) @ u[0]
        l[k] = l[0] @ jnp.tril(f, -1)

    l_primal_out, *l_series_out = l
    u_primal_out, *u_series_out = u
    return (l_primal_out, u_primal_out), (l_series_out, u_series_out)


jet.jet_rules[lu_p] = _lu_taylor_rule


#######
# acos
#######

acos_p = Primitive("_acos")


def acos(a, **kw):
    return acos_p.bind(a, **kw)


def acos_impl(a, **kw):
    return jnp.acos(a)


acos_p.def_impl(acos_impl)
acos_p.multiple_results = False


@jax.jit
def acos_jvp(primals, tangents, **kw):
    (a,) = primals
    (da,) = tangents
    x = acos(a)
    dx = -1 / jnp.sqrt(1 - a * a) * da
    return x, dx


ad.primitive_jvps[acos_p] = acos_jvp


def acos_abstract_eval(a):
    return ShapedArray(a.shape, a.dtype)


acos_p.def_abstract_eval(acos_abstract_eval)


def acos_lowering(ctx, a, **kw):
    return mlir.lower_fun(jnp.acos, multiple_results=False)(ctx, a)


mlir.register_lowering(acos_p, acos_lowering)


def acos_batch_rule(args, dims):
    (mat,) = args
    (dim,) = dims
    if dim is None:
        return acos_p(mat), None
    res = jax.vmap(jnp.acos)(mat)
    return res, dim


batching.primitive_batchers[acos_p] = acos_batch_rule


# @jax.jit
def _acos_taylor_rule(primals_in, series_in, **kw):
    (x,) = primals_in
    (series,) = series_in

    primal_out = jnp.acos(x)

    c0, cs = jet.jet(
        lambda x: lax.div(jnp.ones_like(x), -lax.sqrt(1 - lax.square(x))),
        (x,),
        (series,),
    )

    def scale_(k, j):
        return 1.0 / (fact(k - j) * fact(j - 1))

    c = [c0] + cs
    u = [x] + series
    v = [primal_out] + [None] * len(series)
    for k in range(1, len(v)):
        v[k] = fact(k - 1) * sum(
            scale_(k, j) * c[k - j] * u[j] for j in range(1, k + 1)
        )
    primal_out, *series_out = v
    return primal_out, series_out


jet.jet_rules[acos_p] = _acos_taylor_rule


##############
# Eckart kappa
##############

# Implementation of frame rotation matrix that satisfy the Eckart conditions
# A. Yachmenev, S. N. Yurchenko, J. Chem. Phys. 143, 014105 (2015),
# https://doi.org/10.1063/1.4923039

eckart_kappa_p = Primitive("_eckart_kappa")


def eckart_kappa(xyz, xyz_ref, masses, **kw):
    return eckart_kappa_p.bind(xyz, xyz_ref, masses, **kw)


@jax.jit
def _solve_eckart(xyz, xyz_ref, masses):
    u = jnp.sum(masses[:, None, None] * xyz_ref[:, :, None] * xyz[:, None, :], axis=0)
    umat = jnp.array(
        [
            [u[0, 0] + u[1, 1], u[1, 2], -u[0, 2]],
            [u[2, 1], u[0, 0] + u[2, 2], u[0, 1]],
            [-u[2, 0], u[1, 0], u[1, 1] + u[2, 2]],
        ]
    )
    inv_umat = inv(umat)

    exp_kappa = jnp.eye(3)
    kappa = jnp.zeros_like(exp_kappa)
    l = jnp.eye(3)

    for _ in range(PARAMS["NO_ITERS_ECKART"]):
        rhs = jnp.sum(
            jnp.array(
                [
                    l[0] * u[1] - l[1] * u[0],
                    l[0] * u[2] - l[2] * u[0],
                    l[1] * u[2] - l[2] * u[1],
                ]
            ),
            axis=-1,
        )
        kxy, kxz, kyz = inv_umat @ rhs
        kappa = jnp.array(
            [
                [0.0, kxy, kxz],
                [-kxy, 0.0, kyz],
                [-kxz, -kyz, 0.0],
            ]
        )
        exp_kappa = _expm_pade(-kappa)
        l = exp_kappa + kappa
    return exp_kappa, kappa


def eckart_kappa_impl(xyz, xyz_ref, masses, **kw):
    exp_kappa, _ = _solve_eckart(xyz, xyz_ref, masses, **kw)
    return exp_kappa


eckart_kappa_p.def_impl(eckart_kappa_impl)
eckart_kappa_p.multiple_results = False


@jax.jit
def eckart_kappa_jvp(primals, tangents, **kw):
    xyz, xyz_ref, masses = primals
    dxyz, dxyz_ref, dmasses = tangents

    exp_kappa, kappa = _solve_eckart(xyz, xyz_ref, masses, **kw)

    u = jnp.sum(masses[:, None, None] * xyz_ref[:, :, None] * xyz[:, None, :], axis=0)
    du = jnp.sum(masses[:, None, None] * xyz_ref[:, :, None] * dxyz[:, None, :], axis=0)

    umat = jnp.array(
        [
            [u[0, 0] + u[1, 1], u[1, 2], -u[0, 2]],
            [u[2, 1], u[0, 0] + u[2, 2], u[0, 1]],
            [-u[2, 0], u[1, 0], u[1, 1] + u[2, 2]],
        ]
    )
    inv_umat = inv(umat)

    dl = jnp.zeros((3, 3))
    dexp_kappa = jnp.zeros((3, 3))
    rhs = exp_kappa @ du.T - du @ exp_kappa.T

    for _ in range(PARAMS["NO_ITERS_ECKART"]):
        rhs_ = rhs + dl @ u.T - u @ dl.T
        dkxy, dkxz, dkyz = inv_umat @ jnp.array([rhs_[0, 1], rhs_[0, 2], rhs_[1, 2]])
        dkappa = jnp.array(
            [
                [0.0, dkxy, dkxz],
                [-dkxy, 0.0, dkyz],
                [-dkxz, -dkyz, 0.0],
            ]
        )
        dexp_kappa = _expm_taylor_squaring([-kappa, -dkappa])[1]
        dl = dexp_kappa + dkappa

    return exp_kappa, dexp_kappa


ad.primitive_jvps[eckart_kappa_p] = eckart_kappa_jvp


def eckart_kappa_abstact_eval(xyz, xyz_ref, masses, **kw):
    return ShapedArray((3, 3), xyz.dtype)


eckart_kappa_p.def_abstract_eval(eckart_kappa_abstact_eval)


def eckart_kappa_lowering(ctx, *ar, **kw):
    return mlir.lower_fun(
        lambda *ar, **kw: _solve_eckart(*ar, **kw)[0], multiple_results=False
    )(ctx, *ar, **kw)


mlir.register_lowering(eckart_kappa_p, eckart_kappa_lowering)


def eckart_kappa_batch_rule(args, dims):
    xyz, xyz_ref, masses = args
    dim1, dim2, dim3 = dims
    assert dim1 == 0
    assert dim2 is None
    assert dim3 is None
    out = jax.vmap(lambda x: eckart_kappa_impl(x, xyz_ref, masses))(xyz)
    return out, 0


batching.primitive_batchers[eckart_kappa_p] = eckart_kappa_batch_rule


@jax.jit
def eckart_kappa_taylor_rule(primals_in, series_in, **kw):
    xyz, xyz_ref, masses = primals_in
    dxyz, dxyz_ref, dmasses = series_in

    exp_kappa, kappa = _solve_eckart(xyz, xyz_ref, masses, **kw)

    u = jnp.sum(masses[:, None, None] * xyz_ref[:, :, None] * xyz[:, None, :], axis=0)
    du = [u] + [
        jnp.sum(masses[:, None, None] * xyz_ref[:, :, None] * elem[:, None, :], axis=0)
        for elem in dxyz
    ]

    umat = jnp.array(
        [
            [u[0, 0] + u[1, 1], u[1, 2], -u[0, 2]],
            [u[2, 1], u[0, 0] + u[2, 2], u[0, 1]],
            [-u[2, 0], u[1, 0], u[1, 1] + u[2, 2]],
        ]
    )
    inv_umat = inv(umat)

    dkappa = [kappa] + [None] * len(dxyz)
    dexp_kappa = [exp_kappa] + [None] * len(dxyz)

    def scale(k, j):
        return 1.0 / (fact(k - j) * fact(j))

    for k in range(1, len(du)):

        rhs = dexp_kappa[0] @ du[k].T - du[k] @ dexp_kappa[0].T

        if k > 1:
            rhs2 = fact(k) * sum(
                scale(k, m) * dexp_kappa[m] @ du[k - m].T for m in range(1, k)
            )
            rhs += rhs2 - rhs2.T

        dl = jnp.zeros((3, 3))
        for _ in range(PARAMS["NO_ITERS_ECKART"]):
            rhs_ = rhs + (dl @ du[0].T - du[0] @ dl.T)
            dkxy, dkxz, dkyz = inv_umat @ jnp.array(
                [rhs_[0, 1], rhs_[0, 2], rhs_[1, 2]]
            )
            dkappa[k] = jnp.array(
                [
                    [0.0, dkxy, dkxz],
                    [-dkxy, 0.0, dkyz],
                    [-dkxz, -dkyz, 0.0],
                ]
            )
            dexp_kappa[k] = _expm_taylor_squaring(
                [-elem for elem in dkappa[: k + 1]],
            )[k]
            dl = dexp_kappa[k] + dkappa[k]

    primal_out, *series_out = dexp_kappa
    return primal_out, series_out


jet.jet_rules[eckart_kappa_p] = eckart_kappa_taylor_rule


@jax.jit
def _expm_pade(a):
    b = jnp.array(
        [
            64764752532480000.0,
            32382376266240000.0,
            7771770303897600.0,
            1187353796428800.0,
            129060195264000.0,
            10559470521600.0,
            670442572800.0,
            33522128640.0,
            1323241920.0,
            40840800.0,
            960960.0,
            16380.0,
            182.0,
            1.0,
        ],
        dtype=jnp.float64,
    )

    a2 = a @ a
    a4 = a2 @ a2
    a6 = a2 @ a4

    u = a @ (
        b[13] * a6 @ a6
        + b[11] * a6 @ a4
        + b[9] * a6 @ a2
        + b[7] * a6
        + b[5] * a4
        + b[3] * a2
        + b[1] * jnp.eye(3)
    )

    v = (
        b[12] * a6 @ a6
        + b[10] * a6 @ a4
        + b[8] * a6 @ a2
        + b[6] * a6
        + b[4] * a4
        + b[2] * a2
        + b[0] * jnp.eye(3)
    )

    return inv(v - u) @ (v + u)


@jax.jit
def _matprod_taylor(a, b):
    ab = [None] * len(a)

    def scale(k, j):
        return 1.0 / (fact(k - j) * fact(j))

    for k in range(len(a)):
        ab[k] = fact(k) * sum(scale(k, m) * a[m] @ b[k - m] for m in range(k + 1))
    return ab


@jax.jit
def _expm_taylor(a, k):
    if k == 0:
        at = jnp.eye(len(a[k])) + a[k]
    else:
        at = a[k]
    ap = _matprod_taylor(a, a)
    fac = 0.5
    at = at + ap[k] * fac
    for i in range(3, PARAMS["EXP_TAYLOR_ORDER"]):
        ap = _matprod_taylor(ap, a)
        fac = fac / i
        at = at + ap[k] * fac
    return at


@jax.jit
def _expm_taylor_squaring(a):
    no_sq = PARAMS["EXP_TAYLOR_SQUARING"]
    sq = 1 / 2**no_sq
    a_sq = [elem * sq for elem in a]
    at = [elem for elem in a_sq]
    at[0] += jnp.eye(len(a[0]))
    ap = _matprod_taylor(a_sq, a_sq)
    fac = 0.5
    at = [el1 + el2 * fac for el1, el2 in zip(at, ap)]
    for i in range(3, PARAMS["EXP_TAYLOR_ORDER"]):
        ap = _matprod_taylor(ap, a_sq)
        fac = fac / i
        at = [el1 + el2 * fac for el1, el2 in zip(at, ap)]
    # squaring
    for _ in range(no_sq):
        at = _matprod_taylor(at, at)
    return at


jax.config.update("jax_enable_x64", True)


class EckartMethod(Enum):
    exp_kappa = "exp(-kappa) method from https://doi.org/10.1063/1.4923039"
    quaternion = "quaternion algebra method from https://doi.org/10.1063/1.4870936"
    exp_kappa_direct = (
        "exp(-kappa) method from https://doi.org/10.1063/1.4923039, "
        + "direct differentiation through iterative solver"
    )


def eckart_old(
    q_ref: np.ndarray,
    masses: np.ndarray,
    no_iters: int = 10,
    no_taylor: int = 10,
    no_squaring: int = 4,
    method: EckartMethod = EckartMethod.exp_kappa,
):

    def _wrapper(internal_to_cartesian):
        @functools.wraps(internal_to_cartesian)
        def wrapper_eckart(*args, **kwargs):

            masses_ = jnp.asarray(masses)

            xyz = internal_to_cartesian(*args, **kwargs)

            assert len(xyz) == len(masses), (
                "The number of elements in 'masses' (i.e., number of atoms) must match the leading"
                "dimension of the Cartesian coordinates array returned by the 'internal_to_cartesian'"
                "function"
            )

            com = masses_ @ xyz / jnp.sum(masses_)
            xyz -= com

            xyz_ref = internal_to_cartesian(q_ref, **kwargs)
            com_ref = masses_ @ xyz_ref / jnp.sum(masses_)
            xyz_ref -= com_ref

            if method == EckartMethod.exp_kappa:
                set_params(
                    NO_ITERS_ECKART=no_iters,
                    EXP_TAYLOR_ORDER=no_taylor,
                    EXP_TAYLOR_SQUARING=no_squaring,
                )
                rot_mat = _eckart_expkappa(xyz, xyz_ref, masses_)
            elif method == EckartMethod.quaternion:
                rot_mat = _eckart_quaternion(xyz, xyz_ref, masses_)
            else:
                rot_mat = _eckart_expkappa_direct(xyz, xyz_ref, masses_)

            return xyz @ rot_mat.T

        return wrapper_eckart

    return _wrapper


def _eckart_expkappa(xyz, xyz_ref, masses):
    rot_mat = eckart_kappa(xyz, xyz_ref, masses)
    return rot_mat


def _eckart_expkappa_direct(xyz, xyz_ref, masses):
    rot_mat, _ = _solve_eckart(xyz, xyz_ref, masses)
    return rot_mat


def _eckart_quaternion(xyz, xyz_ref, masses):
    xyz_ma = xyz_ref - xyz
    xyz_pa = xyz_ref + xyz
    x_ma, y_ma, z_ma = xyz_ma.T
    x_pa, y_pa, z_pa = xyz_pa.T

    c11 = jnp.sum(masses * (x_ma**2 + y_ma**2 + z_ma**2))
    c12 = jnp.sum(masses * (y_pa * z_ma - y_ma * z_pa))
    c13 = jnp.sum(masses * (x_ma * z_pa - x_pa * z_ma))
    c14 = jnp.sum(masses * (x_pa * y_ma - x_ma * y_pa))
    c22 = jnp.sum(masses * (x_ma**2 + y_pa**2 + z_pa**2))
    c23 = jnp.sum(masses * (x_ma * y_ma - x_pa * y_pa))
    c24 = jnp.sum(masses * (x_ma * z_ma - x_pa * z_pa))
    c33 = jnp.sum(masses * (x_pa**2 + y_ma**2 + z_pa**2))
    c34 = jnp.sum(masses * (y_ma * z_ma - y_pa * z_pa))
    c44 = jnp.sum(masses * (x_pa**2 + y_pa**2 + z_ma**2))

    c = jnp.array(
        [
            [c11, c12, c13, c14],
            [c12, c22, c23, c24],
            [c13, c23, c33, c34],
            [c14, c24, c34, c44],
        ]
    )

    e, v = eigh(c)
    quar = v[:, 0]

    rot_mat = jnp.array(
        [
            [
                quar[0] ** 2 + quar[1] ** 2 - quar[2] ** 2 - quar[3] ** 2,
                2 * (quar[1] * quar[2] + quar[0] * quar[3]),
                2 * (quar[1] * quar[3] - quar[0] * quar[2]),
            ],
            [
                2 * (quar[1] * quar[2] - quar[0] * quar[3]),
                quar[0] ** 2 - quar[1] ** 2 + quar[2] ** 2 - quar[3] ** 2,
                2 * (quar[2] * quar[3] + quar[0] * quar[1]),
            ],
            [
                2 * (quar[1] * quar[3] + quar[0] * quar[2]),
                2 * (quar[2] * quar[3] - quar[0] * quar[1]),
                quar[0] ** 2 - quar[1] ** 2 - quar[2] ** 2 + quar[3] ** 2,
            ],
        ]
    )
    return rot_mat




def com(masses: np.ndarray):
    """Wrapper function for `internal_to_cartesian` that computes the Cartesian coordinates
    of atoms from given internal coordinates and shifts them to the center of mass.

    Args:
        masses (np.ndarray): An array containing the masses of the atoms. The order of atoms
            in `masses` must match the order in the output of `internal_to_cartesian`.

    Returns:
        A function that first computes the Cartesian coordinates using `internal_to_cartesian`
        and then shifts them to the center of mass.
    """

    def wrapper(internal_to_cartesian):
        @functools.wraps(internal_to_cartesian)
        def wrapper_com(*args, **kwargs):
            xyz = internal_to_cartesian(*args, **kwargs)
            assert len(xyz) == len(masses), (
                "The number of elements in 'masses' must match the leading dimension of the array "
                "returned by the 'internal_to_cartesian' function"
            )
            masses_ = jnp.asarray(masses)
            com = masses_ @ xyz / jnp.sum(masses_)
            return xyz - com[None, :]

        return wrapper_com

    return wrapper


@functools.partial(jax.jit, static_argnums=(2,))
def jac_Gmat_vib(q, masses, internal_to_cartesian):
    nq = len(q)
    return jax.jacfwd(lambda *arg: Gmat(*arg)[:nq, :nq])(
        q, masses, internal_to_cartesian
    )
@functools.partial(jax.jit, static_argnums=(2,))
def jac_Gmat(q, masses, internal_to_cartesian):
    return jax.jacfwd(Gmat)(q, masses, internal_to_cartesian)
@functools.partial(jax.jit, static_argnums=(2,))
def det_gmat(q, masses, internal_to_cartesian):
    nq = len(q)
    return det(gmat(q, masses, internal_to_cartesian)[: nq + 3, : nq + 3])
    # return jnp.linalg.det(gmat(q, masses, internal_to_cartesian)[: nq + 3, : nq + 3])


@jax.jit
def det(a):
    # NOTE: defines determinant up to a sign
    #   because we lose access to permutation
    #   by calling jax.scipy.linalg.lu(a, permute_l=True)
    #   in jet_prim.lu_impl
    l, u = lu(a)
    ud = [u[i, i] for i in range(len(u))]
    return reduce(operator.mul, ud, 1)

@functools.partial(jax.jit, static_argnums=(2,))
def log_abs_det_gmat(q, masses, internal_to_cartesian):
    nq = len(q)
    return log_abs_det(gmat(q, masses, internal_to_cartesian)[: nq + 3, : nq + 3])

@jax.jit
def log_abs_det(a):
    l, u = lu(a)
    return jnp.sum(jnp.array([jnp.log(jnp.abs(u[i, i])) for i in range(len(u))]))


@functools.partial(jax.jit, static_argnums=(2,))
def jac_log_abs_det_gmat(q, masses, internal_to_cartesian):
    return jax.jacrev(log_abs_det_gmat)(q, masses, internal_to_cartesian)

@functools.partial(jax.jit, static_argnums=2)
def gmat(q, masses, internal_to_cartesian):
    # xyz_g = jax.jacfwd(internal_to_cartesian)(jnp.asarray(q))
    xyz_g = jax.jacrev(internal_to_cartesian)(jnp.asarray(q))
    tvib = xyz_g
    xyz = internal_to_cartesian(jnp.asarray(q))
    trot = jnp.transpose(EPS @ xyz.T, (2, 0, 1))
    ttra = jnp.array([jnp.eye(3, dtype=jnp.float64) for _ in range(len(xyz))])
    tvec = jnp.concatenate((tvib, trot, ttra), axis=2)
    masses_sq = jnp.sqrt(jnp.asarray(masses))
    tvec = tvec * masses_sq[:, None, None]
    tvec = jnp.reshape(tvec, (len(xyz) * 3, len(q) + 6))

    return tvec.T @ tvec

@functools.partial(jax.jit, static_argnums=2)
def Gmat(
        q: np.ndarray,
        masses: np.ndarray,
        internal_to_cartesian: Callable[[jnp.ndarray], jnp.ndarray],
):
    """Computes the kinetic energy G-matrix for a molecular system.

    Args:
        q (np.ndarray): An array of internal coordinates with shape (3N-6,),
            where N is the number of atoms. Bond lengths are given in Angstroms,
            and angles are in radians.
        masses (np.ndarray): A 1D array containing the atomic masses. The order of atoms
            in `masses` must match the order of atoms in the output of `internal_to_cartesian`.
        internal_to_cartesian (Callable): A function that converts internal coordinates `q`
            into Cartesian coordinates, returning an array of shape (number of atoms, 3).

    Returns:
        np.ndarray: A square matrix of shape (ncoo+3+3, ncoo+3+3), representing the elements
        of the kinetic energy G-matrix. The first `ncoo` rows and columns correspond to
        vibrational coordinates, followed by three rotational and three translational
        coordinates. The units of the G-matrix are inverse centimeters.
    """
    return inv(gmat(q, masses, internal_to_cartesian))

@functools.partial(jax.jit, static_argnums=(2,))
def hess_log_abs_det_gmat(q, masses, internal_to_cartesian):
    return jax.jacfwd(jax.jacrev(log_abs_det_gmat))(q, masses, internal_to_cartesian)

def _eckart_rotate(changed, reference, mass):
    """
    Rotates 'changed' to satisfy both Eckart Conditions exactly with respect to 'reference'.

    Uses the quaternion method from:
        Dymarsky, Kudin, J. Chem. Phys. 122, 124103 (2005)
        Coutsias, et al., J. Comput. Chem. 25, 1849 (2004)
        Kudin, Dymarsky, J. Chem. Phys. 122, 224105 (2005)

    Args:
        changed: Cartesian coordinates of the displaced geometry, shape (natom, 3)
        reference: Cartesian coordinates of the reference geometry, shape (natom, 3)
        mass: 1D array of atomic masses, shape (natom,)

    Returns:
        Rotated 'changed' geometry satisfying Eckart conditions, shape (natom, 3)
    """
    # Transpose to [3, natom] for the algorithm
    changed_T = changed.T    # (3, natom)
    reference_T = reference.T  # (3, natom)

    # Matrix A: A_ij = sum_a mass_a * changed_ia * reference_ja
    A = jnp.einsum('a, ia, ja -> ij', mass, changed_T, reference_T)

    F = jnp.array([
        [A[0,0] + A[1,1] + A[2,2],  A[1,2] - A[2,1],          A[2,0] - A[0,2],          A[0,1] - A[1,0]],
        [A[1,2] - A[2,1],           A[0,0] - A[1,1] - A[2,2],  A[0,1] + A[1,0],          A[0,2] + A[2,0]],
        [A[2,0] - A[0,2],           A[0,1] + A[1,0],          -A[0,0] + A[1,1] - A[2,2], A[1,2] + A[2,1]],
        [A[0,1] - A[1,0],           A[0,2] + A[2,0],           A[1,2] + A[2,1],         -A[0,0] - A[1,1] + A[2,2]],
    ])

    # The maximum eigenvalue and its eigenvector give the optimal rotation
    D_, V = eigh(F)

    # eigh returns eigenvalues in ascending order: D_[0] is smallest, D_[3] is largest
    # Pick the eigenvector corresponding to max |eigenvalue|
    q = jnp.where(-D_[0] > D_[3], V[:, 0], V[:, 3])
    sign = jnp.where(-D_[0] > D_[3], -1.0, 1.0)

    U = sign * jnp.array([
        [q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2,
         2 * (q[1]*q[2] - q[0]*q[3]),
         2 * (q[1]*q[3] + q[0]*q[2])],
        [2 * (q[1]*q[2] + q[0]*q[3]),
         q[0]**2 + q[2]**2 - q[1]**2 - q[3]**2,
         2 * (q[2]*q[3] - q[0]*q[1])],
        [2 * (q[1]*q[3] - q[0]*q[2]),
         2 * (q[2]*q[3] + q[0]*q[1]),
         q[0]**2 + q[3]**2 - q[1]**2 - q[2]**2],
    ])

    # Rotate: xyz_rot = U @ changed_T, then transpose back to (natom, 3)
    return (U @ changed_T).T


def eckart(q_ref, masses):
    """
    Decorator factory for Eckart frame embedding.

    Wraps an `internal_to_cartesian` function so that the returned Cartesian
    coordinates are rotated into the Eckart frame defined by the reference
    geometry.

    Uses the quaternion-based method from:
        Dymarsky, Kudin, J. Chem. Phys. 122, 124103 (2005)
        Coutsias, et al., J. Comput. Chem. 25, 1849 (2004)

    Usage::

        @eckart(q0, jnp.array(masses))
        def my_transform(q):
            ...
            return xyz  # shape (natom, 3)

    Args:
        q_ref: Reference internal coordinates (1D array).
        masses: 1D array of atomic masses.

    Returns:
        Decorator that wraps `internal_to_cartesian`.
    """
    masses_ = jnp.asarray(masses)

    def _wrapper(internal_to_cartesian):
        @functools.wraps(internal_to_cartesian)
        def wrapper_eckart(*args, **kwargs):
            xyz = internal_to_cartesian(*args, **kwargs)

            assert len(xyz) == len(masses_), (
                "The number of elements in 'masses' (i.e., number of atoms) must match "
                "the leading dimension of the Cartesian coordinates array returned by "
                "the 'internal_to_cartesian' function"
            )

            # Shift to center of mass
            com = masses_ @ xyz / jnp.sum(masses_)
            xyz = xyz - com

            # Reference geometry
            xyz_ref = internal_to_cartesian(q_ref, **kwargs)
            com_ref = masses_ @ xyz_ref / jnp.sum(masses_)
            xyz_ref = xyz_ref - com_ref

            # Apply Eckart rotation (quaternion method)
            return _eckart_rotate(xyz, xyz_ref, masses_)

        return wrapper_eckart

    return _wrapper

@functools.partial(jax.jit, static_argnums=(2,))
def pseudo(
        q: np.ndarray,
        masses: np.ndarray,
        internal_to_cartesian: Callable[[jnp.ndarray], jnp.ndarray],
):
    """Pseudopotential implementation according to Eq. (21)
    in Edit Mátyus, Gábor Czakó, and Attila G. Császár,
    J. Chem. Phys. 130, 134112 (2009)
    http://dx.doi.org/10.1063/1.3076742
    """
    nq = len(q)
    G = Gmat(q, masses, internal_to_cartesian)[:nq, :nq]
    dG = jac_Gmat_vib(q, masses, internal_to_cartesian)
    k = jnp.arange(nq)
    dG = dG[k, :, k]
    dlogdet = jac_log_abs_det_gmat(q, masses, internal_to_cartesian)
    hlogdet = hess_log_abs_det_gmat(q, masses, internal_to_cartesian)
    pseudo1 = dlogdet @ G @ dlogdet
    pseudo2 = jnp.sum(dG @ dlogdet) + jnp.sum(G * hlogdet)
    return (pseudo1 + 4 * pseudo2) / 32.0






def build_J_matrices(J, M=None):
    """
    构造在完全基底 |J, K, M> 下的角动量矩阵。
    基底顺序约定：K 为外层循环（慢），M 为内层循环（快）。
    总维度 dim_full = (2J+1)^2 x (2J+1)^2

    Returns:
        dict: 包含体定系 (x, y, z) 和空间系 (X, Y, Z) 的完整矩阵。
    """
    dim = int(2 * J + 1)
    vals = np.arange(-J, J + 1)

    # ==========================================
    # 1. 构造体定系 (Body-Fixed) 的基础矩阵 (仅作用于 K)
    # 遵循反常对易关系: [jx, jy] = -i jz
    # ==========================================
    jx_K = np.zeros((dim, dim), dtype=np.complex128)
    jy_K = np.zeros((dim, dim), dtype=np.complex128)
    jz_K = np.zeros((dim, dim), dtype=np.complex128)

    np.fill_diagonal(jz_K, vals)
    for i in range(dim):
        K = vals[i]
        if K < J:
            C_plus = 0.5 * np.sqrt(J * (J + 1) - K * (K + 1))
            jx_K[i + 1, i] = C_plus
            jy_K[i + 1, i] = -1j * C_plus  # 体定系反常负号
        if K > -J:
            C_minus = 0.5 * np.sqrt(J * (J + 1) - K * (K - 1))
            jx_K[i - 1, i] = C_minus
            jy_K[i - 1, i] = 1j * C_minus

    # ==========================================
    # 2. 构造空间系 (Space-Fixed) 的基础矩阵 (仅作用于 M)
    # 遵循正常对易关系: [JX, JY] = i JZ
    # ==========================================
    JX_M = np.zeros((dim, dim), dtype=np.complex128)
    JY_M = np.zeros((dim, dim), dtype=np.complex128)
    JZ_M = np.zeros((dim, dim), dtype=np.complex128)

    np.fill_diagonal(JZ_M, vals)
    for i in range(dim):
        M_quant = vals[i]
        if M_quant < J:
            C_plus = 0.5 * np.sqrt(J * (J + 1) - M_quant * (M_quant + 1))
            JX_M[i + 1, i] = C_plus
            JY_M[i + 1, i] = 1j * C_plus  # 空间系正常正号
        if M_quant > -J:
            C_minus = 0.5 * np.sqrt(J * (J + 1) - M_quant * (M_quant - 1))
            JX_M[i - 1, i] = C_minus
            JY_M[i - 1, i] = -1j * C_minus

    if M is not None:
        M = int(M)
        if M < -J or M > J:
            raise ValueError(f"M={M} is outside the allowed range [-J, J] for J={J}.")
        I_dim = np.eye(dim)
        return {
            "jx": jx_K,
            "jy": jy_K,
            "jz": jz_K,
            "JX": None,
            "JY": None,
            "JZ": M * I_dim,
        }

    # ==========================================
    # 3. 扩展到全空间 |J, K, M> = |K> ⊗ |M>
    # 维度扩大为 (2J+1)^2
    # ==========================================
    I_dim = np.eye(dim)

    # 体定系矩阵：作用于 K，对 M 相当于单位阵
    Jx_body_full = np.kron(jx_K, I_dim)
    Jy_body_full = np.kron(jy_K, I_dim)
    Jz_body_full = np.kron(jz_K, I_dim)

    # 空间系矩阵：作用于 M，对 K 相当于单位阵
    JX_space_full = np.kron(I_dim, JX_M)
    JY_space_full = np.kron(I_dim, JY_M)
    JZ_space_full = np.kron(I_dim, JZ_M)

    # 返回一个字典，方便按需提取
    return {
        "jx": Jx_body_full, "jy": Jy_body_full, "jz": Jz_body_full,
        "JX": JX_space_full, "JY": JY_space_full, "JZ": JZ_space_full
    }


def calculate_exact_keo(dvrs, masses, internal_to_cartesian, mode='vib', J_val=0, M_val=None, verbose=True):
    """
    Exact KEO Calculator (支持振动和转动)

    Args:
        dvrs (list): List of DVR objects (e.g., [dvr_r1, dvr_theta])
        masses (list): List of atomic masses
        internal_to_cartesian (function): Coordinate mapping function (JAX compatible)
        mode (str): 'G', 'vib', 'rot', 'cor', 'all'
        J_val (int): Total Angular Momentum
            J^2 |J,K,M> = J(J+1)|J,K,M>
        K : The projection of the total angular momentum onto the molecule's internal body-fixed z-axis.
            jz|J,K,M> = K|J,K,M>  (from -J to J)
        M : The projection of the total angular momentum onto the laboratory space-fixed Z-axis.
            JZ|J,K,M> = M|J,K,M>  (from -J to J)
        M_val (int or None): If given, exploit field-free Jz conservation by
            projecting to a fixed space-fixed M block.  The rotational dimension
            is then 2J+1 instead of (2J+1)^2.
        verbose (bool): Whether to print construction process information

    Returns:
        np.ndarray: 对应模式的哈密顿矩阵或G矩阵
    """
    if verbose: print(f"[KEO] Starting calculation with {len(dvrs)} dimensions...")

    grids = [d.x for d in dvrs]
    mesh = jnp.meshgrid(*grids, indexing='ij')

    q_batch = jnp.stack([m.flatten() for m in mesh], axis=1)
    n_tot = q_batch.shape[0]
    n_dim = len(dvrs)

    if M_val is not None and (M_val < -J_val or M_val > J_val):
        raise ValueError(f"M_val={M_val} is outside the allowed range [-J, J] for J={J_val}.")

    # Full |K,M> basis has (2J+1)^2 states.  For field-free dynamics, M is
    # conserved, so a fixed-M block only needs the K ladder.
    dim_rot = int(2 * J_val + 1) if M_val is not None else int((2 * J_val + 1) ** 2)

    if verbose: print(f"[KEO] Total grid points: {n_tot} (Shape: {q_batch.shape})")
    if verbose: print("[KEO] Computing exact G-matrix via JAX AD...")

    # 计算全局 G 矩阵和赝势
    batch_Gmat_fn = jax.vmap(Gmat, in_axes=(0, None, None))
    G_all = np.array(batch_Gmat_fn(q_batch, masses, internal_to_cartesian))

    batch_pseudo_fn = jax.vmap(pseudo, in_axes=(0, None, None))
    pseudo_all = np.array(batch_pseudo_fn(q_batch, masses, internal_to_cartesian))

    if mode == 'G':
        if verbose: print("[KEO] Returning G-matrix values only.")
        return G_all

    Ids = [np.eye(d.npts) for d in dvrs]
    D1s = [d.momentum() for d in dvrs]

    if J_val > 0:
        # 解析返回的字典，提取出动能算符所需的“体定系”角动量矩阵
        J_matrices_dict = build_J_matrices(J_val, M=M_val)
        Jx = J_matrices_dict["jx"]
        Jy = J_matrices_dict["jy"]
        Jz = J_matrices_dict["jz"]
        J_ops = [Jx, Jy, Jz]

    # 初始化矩阵容器
    T_vib = None
    T_rot = None
    T_cor = None

    if mode in ['vib', 'all']:
        if verbose: print("[KEO] Assembling Vibrational Hamiltonian matrix T_vib...")
        G_vib = G_all[:, :n_dim, :n_dim]
        T_vib = np.zeros((n_tot, n_tot), dtype=np.complex128)

        for i in range(n_dim):
            for j in range(n_dim):
                ops_i = [D1s[k] if k == i else Ids[k] for k in range(n_dim)]
                D_i_full = reduce(np.kron, ops_i)

                if i == j:
                    D_j_full = D_i_full
                else:
                    ops_j = [D1s[k] if k == j else Ids[k] for k in range(n_dim)]
                    D_j_full = reduce(np.kron, ops_j)

                g_diag_values = G_vib[:, i, j]
                G_op = np.diag(g_diag_values)
                # 0.5 * P_i^\dagger * G_ij * P_j
                term = 0.5 * (D_i_full.conj().T @ G_op @ D_j_full)
                T_vib += term

        T_vib += np.diag(pseudo_all)
        if mode == 'vib':
            return T_vib

    if J_val == 0 and mode == 'all':
        print("[KEO] J=0, returning pure vibrational T_vib.")
        return T_vib

    if mode in ['rot', 'all']:
        G_rot = G_all[:, n_dim:n_dim + 3, n_dim:n_dim + 3]
        T_rot = np.zeros((n_tot * dim_rot, n_tot * dim_rot), dtype=np.complex128)

        for a in range(3):
            for b in range(3):
                G_ab_diag = np.diag(G_rot[:, a, b])
                J_ab = J_ops[a] @ J_ops[b]

                T_rot += 0.5 * np.kron(G_ab_diag, J_ab)
                # T_rot += term

        if mode == 'rot':
            return T_rot

    if mode in ['cor', 'all']:
        if verbose: print(f"[KEO] Assembling Coriolis coupling terms for J={J_val}...")

        G_cor_T = G_all[:, n_dim:n_dim + 3, :n_dim]
        T_cor = np.zeros((n_tot * dim_rot, n_tot * dim_rot), dtype=np.complex128)

        for a in range(3):
            for i in range(n_dim):
                ops_i = [D1s[k] if k == i else Ids[k] for k in range(n_dim)]
                D_i_full = reduce(np.kron, ops_i)

                G_ai_diag = np.diag(G_cor_T[:, a, i])

                # 量子力学对称排序: 0.5 * (P_i^dagger * G + G * P_i)
                vib_op = 0.5 * (D_i_full.conj().T @ G_ai_diag + G_ai_diag @ D_i_full)
                J_a = J_ops[a]

                term = np.kron(vib_op, J_a)
                T_cor += term

        if mode == 'cor':
            return T_cor

    if mode == 'all':
        if verbose: print("[KEO] Combining Vib, Rot, and Coriolis matrices into T_all...")

        # 将振动块扩展到完整的 振动 ⊗ 旋转(K,M) 全空间
        I_rot = np.eye(dim_rot)
        T_vib_expanded = np.kron(T_vib, I_rot)

        T_all = T_vib_expanded + T_rot + T_cor

        if verbose: print(f"[KEO] Final T_all shape: {T_all.shape}")
        return T_all

    else:
        raise ValueError(f"Unknown mode: {mode}. Choose from 'vib', 'rot', 'cor', 'all', 'G'.")




import os
import tqdm
from concurrent.futures import ProcessPoolExecutor
import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from pyscf import gto, dft, lib
# from pyscf.geomopt.geometric_solver import optimize


# （确保已导入 SineDVR, eckart, calculate_exact_keo, Gmat, pseudo 等）

def calc_single_point(X_current):
    lib.num_threads(1)
    atom_str = (
        f"O {X_current[0, 0]} {X_current[0, 1]} {X_current[0, 2]}; "
        f"H {X_current[1, 0]} {X_current[1, 1]} {X_current[1, 2]}; "
        f"H {X_current[2, 0]} {X_current[2, 1]} {X_current[2, 2]}"
    )
    mol = gto.M(atom=atom_str, basis='6-31G**', unit='Bohr', verbose=0)
    mf = dft.RKS(mol)
    mf.xc = 'b3lyp'
    return mf.kernel()


if __name__ == "__main__":
    cm_inv = 219474.63
    MASS_O = 29156.946
    MASS_H = 1836.153
    masses = [MASS_O, MASS_H, MASS_H]

    print("  [PySCF] 正在寻找 B3LYP/6-31G** 的真实全局极小值...")
    mol = gto.M(
        atom="O 0.0 0.0 0.0; H 0.757 0.587 0.0; H -0.757 0.587 0.0",
        basis='6-31G**', unit='Angstrom', verbose=0
    )
    mf_init = dft.RKS(mol)
    mf_init.xc = 'b3lyp'

    # mol_opt = optimize(mf_init)
    E_opt_pyscf = mf_init.kernel()

    coords = mol.atom_coords()
    vec_O_H1 = coords[1] - coords[0]
    vec_O_H2 = coords[2] - coords[0]

    r_eq_dft = float(np.linalg.norm(vec_O_H1))
    cos_theta = np.dot(vec_O_H1, vec_O_H2) / (np.linalg.norm(vec_O_H1) * np.linalg.norm(vec_O_H2))
    theta_eq_dft = float(np.arccos(cos_theta))
    q0_dft = jnp.array([r_eq_dft, r_eq_dft, theta_eq_dft])


    # 【重要修复】：恢复 @eckart 装饰器，否则势能面和动能矩阵将出现灾难性偏差
    @eckart(q0_dft, jnp.array(masses))
    def h2o_3d_transform_dft(q_int):
        r1, r2, ang = q_int
        return jnp.array([
            [0.0, 0.0, 0.0],
            [0.0, r1 * jnp.sin(ang / 2), -r1 * jnp.cos(ang / 2)],
            [0.0, -r2 * jnp.sin(ang / 2), -r2 * jnp.cos(ang / 2)],
        ])


    # 降低网格数以应对高 J 维度的矩阵爆炸 (N=5，总振动网格点 125)
    N_R, N_THETA = 15,15
    R_MIN, R_MAX = r_eq_dft - 0.4, r_eq_dft + 0.5
    THETA_MIN, THETA_MAX = theta_eq_dft - 35 * (np.pi / 180.0), theta_eq_dft + 40 * (np.pi / 180.0)

    dvrs_int = [SineDVR(R_MIN, R_MAX, N_R), SineDVR(R_MIN, R_MAX, N_R), SineDVR(THETA_MIN, THETA_MAX, N_THETA)]
    grids_int = [d.x for d in dvrs_int]
    mesh_int = jnp.meshgrid(*grids_int, indexing='ij')
    q_batch_int = jnp.stack([m.flatten() for m in mesh_int], axis=1)

    print("  [Transform] Transform internal coordinates to Cartesian coordinates")
    X_batch_list = [np.array(h2o_3d_transform_dft(q_int)) for q_int in q_batch_int]

    # print(f"  [PySCF] 正在开启多进程扫描 {len(X_batch_list)} 个 DVR 网格点势能...")
    # with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
    #     energies_raw = list(tqdm.tqdm(
    #         executor.map(calc_single_point, X_batch_list),
    #         total=len(X_batch_list),
    #         desc="PES Parallel Scanning"
    #     ))

    # E_min_scan = np.min(energies_raw)
    # V_pyscf_vals = [E - E_min_scan for E in energies_raw]
    # V_diag_vib = np.diag(np.array(V_pyscf_vals))

    # 先提取 J=0 的绝对零点能作为绘图基准
    print("  [J=0] Computing ZPE...")
    # Gmat(q, masses, internal_to_cartesian)
    
    T_vib_J0 = calculate_exact_keo(dvrs_int, jnp.array(masses), \
                                   h2o_3d_transform_dft, mode='G', J_val=0, verbose=False)
    
    print(T_vib_J0.shape)
    
    # E_J0, _ = la.eigh(T_vib_J0 + V_diag_vib)
    # ZPE = E_J0[0] * cm_inv

    # 准备绘图数据
    J_max = 3
    plot_data = []

    # print("\n  [Rovibrational] 开始批量计算 J=0 到 J={J_max} 的全维振转能级...")
    # for J in range(J_max + 1):
    #     print(f"    --> 正在求解 J = {J} ...")
    #     dim_rot = (2 * J + 1) ** 2

    #     # 构建 J > 0 的全动能矩阵
    #     T_all = calculate_exact_keo(dvrs_int, jnp.array(masses), h2o_3d_transform_dft, mode='all', J_val=J, verbose=False)

    #     # 将纯振动势能扩展到全振转空间
    #     I_rot = np.eye(dim_rot)
    #     V_all = np.kron(V_diag_vib, I_rot)

    #     # 对角化全哈密顿量
    #     E_all, _ = la.eigh(T_all + V_all)
    #     E_all_cm = E_all * cm_inv

    #     # 记录前 150 个能级（或所有能级，取其小者）用于绘图
    #     n_levels = min(35, len(E_all_cm))
    #     rel_energies = E_all_cm[:n_levels] - ZPE
    #     print(rel_energies[:5])
    #     # 过滤掉高于 6500 cm-1 的能级，保持图面整洁
    #     rel_energies = rel_energies[rel_energies < 6500]

    #     plot_data.append(rel_energies)


    # plt.figure(figsize=(10, 8))
    # colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'y', 'black']

    # for J, energies in enumerate(plot_data):
    #     x_start = J - 0.35
    #     x_end = J + 0.35
    #     # 画每一条水平能级线
    #     for E in energies:
    #         plt.hlines(E, x_start, x_end, colors=colors[J], linewidth=1.0)

    # plt.xticks(range(J_max + 1), [str(j) for j in range(J_max + 1)], fontsize=14)
    # plt.xlabel('J =', fontsize=16)
    # plt.ylabel(r'$\Delta E$ / cm$^{-1}$', fontsize=16)
    # plt.title('Rovibrational Energy Levels ($\Delta E$ vs $J$)', fontsize=18)

    # # 限制 y 轴显示范围对齐文献图
    # plt.ylim(-100, 6500)
    # plt.xlim(-0.5, 6.5)

    # # 隐藏上方和右方的边框
    # ax = plt.gca()
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)

    # # 将 x 轴下移，模仿文献的刻度样式
    # ax.spines['bottom'].set_position(('data', 0))
    # ax.xaxis.set_label_coords(0.5, -0.05)

    # plt.tight_layout()
    # plt.savefig("h2o_rovibrational_levels.png", dpi=300)
    # plt.show()
