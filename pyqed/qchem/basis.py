#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 00:01:55 2024

@author: bingg
"""

import os
import re
import math
import ctypes
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import numpy as np
import pyqed
_CYTHON_COMPILED = False
try:
    _CYTHON_COMPILED = bool(getattr(__import__("cython"), "compiled", False))
except ImportError:  # pragma: no cover - optional
    pass
njit = None


PI = math.pi
PI_1P5 = PI ** 1.5
ERI_PREFAC = 2.0 * (PI ** 2.5)

_NATIVE_PAR_SIGNATURES = None
_NATIVE_PAR_PAIRS = None
_NATIVE_PAR_PAIR_BOUNDS = None
_NATIVE_PAR_SCREEN_TOL = 0.0
_NUMBA_AVAILABLE = njit is not None
_NUMBA_DENSE_ERI_ENABLED = False
_BASIS_ACCEL = None
try:
    from . import _basis_cy
except Exception:  # pragma: no cover - optional accelerator
    _basis_cy = None


def _load_basis_accel():
    global _BASIS_ACCEL
    if _BASIS_ACCEL is not None:
        return _BASIS_ACCEL

    here = os.path.dirname(__file__)
    candidates = [
        os.path.join(here, "_basis_accel.dylib"),
        os.path.join(here, "_basis_accel.so"),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            lib = ctypes.CDLL(path)
            func = lib.compute_dense_eri
            func.argtypes = [
                ctypes.c_int,
                ctypes.c_int,
                np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags=("C_CONTIGUOUS",)),
                np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags=("C_CONTIGUOUS",)),
                np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags=("C_CONTIGUOUS",)),
                np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags=("C_CONTIGUOUS",)),
                np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags=("C_CONTIGUOUS",)),
                np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags=("C_CONTIGUOUS",)),
                ctypes.c_double,
                np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags=("C_CONTIGUOUS", "WRITEABLE")),
                ctypes.POINTER(ctypes.c_longlong),
                ctypes.POINTER(ctypes.c_longlong),
            ]
            func.restype = ctypes.c_int
            _BASIS_ACCEL = lib
            return _BASIS_ACCEL
        except OSError:
            continue

    _BASIS_ACCEL = False
    return _BASIS_ACCEL

# @njit(float64(int64, int64, int64, float64, float64, float64), parallel=True)
@lru_cache(maxsize=65536)
def E(i: int, j:int, t:int, Qx:float, a:float, b:float):
    '''
    Recursive definition of Hermite Gaussian coefficients.

    Returns a float.
    a: orbital exponent on Gaussian 'a' (e.g. alpha in the text)
    b: orbital exponent on Gaussian 'b' (e.g. beta in the text)
    i,j: orbital angular momentum number on Gaussian 'a' and 'b'
    t: number nodes in Hermite (depends on type of integral,
    e.g. always zero for overlap integrals)
    Qx: distance between origins of Gaussian 'a' and 'b'

    Refs
        https://joshuagoings.com/assets/integrals.pdf
    '''
    p = a + b
    q = a*b/p
    if (t < 0) or (t > (i + j)):
        # out of bounds for t
        return 0.0
    elif i == j == t == 0:
        # base case
        return np.exp(-q*Qx*Qx) # K_AB
    elif j == 0:
        # decrement index i
        return (1/(2*p))*E(i-1,j,t-1,Qx,a,b) - \
        (q*Qx/a)*E(i-1,j,t,Qx,a,b) + \
        (t+1)*E(i-1,j,t+1,Qx,a,b)
    else:
        # decrement index j
        return (1/(2*p))*E(i,j-1,t-1,Qx,a,b) + \
        (q*Qx/b)*E(i,j-1,t,Qx,a,b) + \
        (t+1)*E(i,j-1,t+1,Qx,a,b)

def overlap(a,lmn1,A,b,lmn2,B):
    ''' Evaluates overlap integral between two Gaussians
    Returns a float.
    a: orbital exponent on Gaussian 'a' (e.g. alpha in the text)
    b: orbital exponent on Gaussian 'b' (e.g. beta in the text)
    lmn1: int tuple containing orbital angular momentum (e.g. (1,0,0))
    for Gaussian 'a'
    lmn2: int tuple containing orbital angular momentum for Gaussian 'b'
    A: list containing origin of Gaussian 'a', e.g. [1.0, 2.0, 0.0]
    B: list containing origin of Gaussian 'b'
    '''
    l1,m1,n1 = lmn1 # shell angular momentum on Gaussian 'a'
    l2,m2,n2 = lmn2 # shell angular momentum on Gaussian 'b'
    S1 = E(l1,l2,0,A[0]-B[0],a,b) # X
    S2 = E(m1,m2,0,A[1]-B[1],a,b) # Y
    S3 = E(n1,n2,0,A[2]-B[2],a,b) # Z
    return S1*S2*S3*((PI/(a+b))**1.5)

def S(a,b):
    '''Evaluates overlap between two contracted Gaussians
    Returns float.
    Arguments:
    a: contracted Gaussian 'a', BasisFunction object
    b: contracted Gaussian 'b', BasisFunction object
    '''
    s = 0.0
    for ia, wa in enumerate(a.prim_weights):
        for ib, wb in enumerate(b.prim_weights):
            s += wa*wb*\
                overlap(a.exps[ia],a.shell,a.origin,
                b.exps[ib],b.shell,b.origin)
    return s


from scipy.special import factorial2
from scipy.special import hyp1f1


# @jit(int64(int64))
def fact2(n: int):
    """
    double factorial n!!

    Parameters
    ----------
    n : int
        int.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if n >= 0:
        return factorial2(n)
    elif n % 2:
        return (-1)**(abs(n+1)//2) * 1/factorial2(abs(n+2))
    else:
        raise ValueError('Factorial2 is not defined for negative even number.')


# @jit
@lru_cache(maxsize=16384)
def boys(n,T):
    return hyp1f1(n+0.5,n+1.5,-T)/(2.0*n+1.0)

class ContractedGaussian(object):
    ''' A class that contains all contracted Gaussian basis function data
    Attributes:
    origin: array/list containing the coordinates of the Gaussian origin
    shell: tuple of angular momentum
    exps: list of primitive Gaussian exponents
    coefs: list of primitive Gaussian coefficients
    norm: list of normalization factors for Gaussian primitives
    '''
    def __init__(self,origin=[0.0,0.0,0.0],shell=(0,0,0),exps=[],coefs=[]):
        self.origin = np.asarray(origin)
        self.shell = shell
        self.exps = exps
        self.coefs = coefs
        self.norm = None
        self.prim_weights = None
        self.normalize()

    def normalize(self):
        ''' Routine to normalize the basis functions, in case they
        do not integrate to unity.
        '''
        l,m,n = self.shell
        L = l+m+n

        # self.norm is a list of length equal to number primitives
        # normalize primitives first (PGBFs)
        self.norm = np.sqrt(np.power(2,2*(l+m+n)+1.5)*np.power(self.exps,l+m+n+1.5)/fact2(2*l-1)/fact2(2*m-1)/fact2(2*n-1)/PI_1P5)
        # now normalize the contracted basis functions (CGBFs)
        # Eq. 1.44 of Valeev integral whitepaper
        prefactor = PI_1P5 * fact2(2*l - 1)*fact2(2*m-1)*fact2(2*n - 1)/np.power(2.0,L)

        N = 0.0
        num_exps = len(self.exps)

        for ia in range(num_exps):
            for ib in range(num_exps):
                N += self.norm[ia]*self.norm[ib]*self.coefs[ia]*self.coefs[ib]/np.power(self.exps[ia] + self.exps[ib],L+1.5)

        # print(prefactor, N)

        N = N * prefactor
        N = np.power(N,-0.5)
        for ia in range(num_exps):
            self.coefs[ia] *= N
        self.prim_weights = self.norm * self.coefs

def kinetic(a,lmn1,A,b,lmn2,B):
    ''' Evaluates kinetic energy integral between two Gaussians
    Returns a float.
    a: orbital exponent on Gaussian 'a' (e.g. alpha in the text)
    b: orbital exponent on Gaussian 'b' (e.g. beta in the text)
    lmn1: int tuple containing orbital angular momentum (e.g. (1,0,0))
    for Gaussian 'a'
    lmn2: int tuple containing orbital angular momentum for Gaussian 'b'
    A: list containing origin of Gaussian 'a', e.g. [1.0, 2.0, 0.0]
    B: list containing origin of Gaussian 'b'
    '''
    l1,m1,n1 = lmn1
    l2,m2,n2 = lmn2
    term0 = b*(2*(l2+m2+n2)+3)*\
    overlap(a,(l1,m1,n1),A,b,(l2,m2,n2),B)
    term1 = -2*np.power(b,2)*\
    (overlap(a,(l1,m1,n1),A,b,(l2+2,m2,n2),B) +
    overlap(a,(l1,m1,n1),A,b,(l2,m2+2,n2),B) +
    overlap(a,(l1,m1,n1),A,b,(l2,m2,n2+2),B))
    term2 = -0.5*(l2*(l2-1)*overlap(a,(l1,m1,n1),A,b,(l2-2,m2,n2),B) +
        m2*(m2-1)*overlap(a,(l1,m1,n1),A,b,(l2,m2-2,n2),B) +
        n2*(n2-1)*overlap(a,(l1,m1,n1),A,b,(l2,m2,n2-2),B))

    return term0+term1+term2

def T(a,b):
    '''Evaluates kinetic energy between two contracted Gaussians
    Returns float.
    Arguments:
    a: contracted Gaussian 'a', BasisFunction object
    b: contracted Gaussian 'b', BasisFunction object
    '''
    t = 0.0
    for ia, wa in enumerate(a.prim_weights):
        for ib, wb in enumerate(b.prim_weights):
            t += wa*wb*\
            kinetic(a.exps[ia],a.shell,a.origin,\
            b.exps[ib],b.shell,b.origin)
    return t

# @jit
@lru_cache(maxsize=131072)
def R(t,u,v,n,p,PCx,PCy,PCz,RPC):
    ''' Returns the Coulomb auxiliary Hermite integrals
    Returns a float.
    Arguments:
    t,u,v: order of Coulomb Hermite derivative in x,y,z
    (see defs in Helgaker and Taylor)
    n: order of Boys function
    PCx,y,z: Cartesian vector distance between Gaussian
    composite center P and nuclear center C
    RPC: Distance between P and C
    '''
    T = p*RPC*RPC
    val = 0.0
    if t == u == v == 0:
        val += np.power(-2*p,n)*boys(n,T)
    elif t == u == 0:
        if v > 1:
            val += (v-1)*R(t,u,v-2,n+1,p,PCx,PCy,PCz,RPC)
        val += PCz*R(t,u,v-1,n+1,p,PCx,PCy,PCz,RPC)
    elif t == 0:
        if u > 1:
            val += (u-1)*R(t,u-2,v,n+1,p,PCx,PCy,PCz,RPC)
        val += PCy*R(t,u-1,v,n+1,p,PCx,PCy,PCz,RPC)
    else:
        if t > 1:
            val += (t-1)*R(t-2,u,v,n+1,p,PCx,PCy,PCz,RPC)
        val += PCx*R(t-1,u,v,n+1,p,PCx,PCy,PCz,RPC)
    return val

def gaussian_product_center(a,A,b,B):
    return (a*A+b*B)/(a+b)


@lru_cache(maxsize=65536)
def _nuclear_attraction_cached(a,l1,m1,n1,Ax,Ay,Az,b,l2,m2,n2,Bx,By,Bz,Cx,Cy,Cz):
    p = a + b
    px = (a * Ax + b * Bx) / p
    py = (a * Ay + b * By) / p
    pz = (a * Az + b * Bz) / p
    dx = px - Cx
    dy = py - Cy
    dz = pz - Cz
    rpc = math.sqrt(dx * dx + dy * dy + dz * dz)

    val = 0.0
    abx = Ax - Bx
    aby = Ay - By
    abz = Az - Bz
    for t in range(l1 + l2 + 1):
        ex = E(l1, l2, t, abx, a, b)
        for u in range(m1 + m2 + 1):
            exy = ex * E(m1, m2, u, aby, a, b)
            for v in range(n1 + n2 + 1):
                val += exy * E(n1, n2, v, abz, a, b) * R(t, u, v, 0, p, dx, dy, dz, rpc)

    return val * (2.0 * PI / p)


def nuclear_attraction(a,lmn1,A,b,lmn2,B,C):
    ''' Evaluates kinetic energy integral between two Gaussians
    Returns a float.
    a: orbital exponent on Gaussian 'a' (e.g. alpha in the text)
    b: orbital exponent on Gaussian 'b' (e.g. beta in the text)
    lmn1: int tuple containing orbital angular momentum (e.g. (1,0,0))
    for Gaussian 'a'
    lmn2: int tuple containing orbital angular momentum for Gaussian 'b'
    A: list containing origin of Gaussian 'a', e.g. [1.0, 2.0, 0.0]
    B: list containing origin of Gaussian 'b'
    C: list containing origin of nuclear center 'C'
    '''
    l1,m1,n1 = lmn1
    l2,m2,n2 = lmn2
    return _nuclear_attraction_cached(
        a, l1, m1, n1, A[0], A[1], A[2],
        b, l2, m2, n2, B[0], B[1], B[2],
        C[0], C[1], C[2],
    )




def point_charge(a,b,C):
    '''Evaluates electron-nuclear attraction

    $%overlap between two contracted Gaussians

    Returns float.
    Arguments:
    a: contracted Gaussian 'a', BasisFunction object
    b: contracted Gaussian 'b', BasisFunction object
    C: center of nucleus
    '''
    v = 0.0
    for ia, wa in enumerate(a.prim_weights):
        for ib, wb in enumerate(b.prim_weights):
            v += wa*wb*\
                nuclear_attraction(a.exps[ia],a.shell,a.origin,
                b.exps[ib],b.shell,b.origin,C)
    return v


@lru_cache(maxsize=262144)
def _electron_repulsion_cached(
    a,l1,m1,n1,Ax,Ay,Az,
    b,l2,m2,n2,Bx,By,Bz,
    c,l3,m3,n3,Cx,Cy,Cz,
    d,l4,m4,n4,Dx,Dy,Dz,
):
    p = a + b
    q = c + d
    alpha = p * q / (p + q)

    px = (a * Ax + b * Bx) / p
    py = (a * Ay + b * By) / p
    pz = (a * Az + b * Bz) / p
    qx = (c * Cx + d * Dx) / q
    qy = (c * Cy + d * Dy) / q
    qz = (c * Cz + d * Dz) / q
    dx = px - qx
    dy = py - qy
    dz = pz - qz
    rpq = math.sqrt(dx * dx + dy * dy + dz * dz)

    abx = Ax - Bx
    aby = Ay - By
    abz = Az - Bz
    cdx = Cx - Dx
    cdy = Cy - Dy
    cdz = Cz - Dz

    val = 0.0
    for t in range(l1 + l2 + 1):
        ex_ab = E(l1, l2, t, abx, a, b)
        for u in range(m1 + m2 + 1):
            exy_ab = ex_ab * E(m1, m2, u, aby, a, b)
            for v in range(n1 + n2 + 1):
                xyz_ab = exy_ab * E(n1, n2, v, abz, a, b)
                for tau in range(l3 + l4 + 1):
                    ex_cd = E(l3, l4, tau, cdx, c, d)
                    for nu in range(m3 + m4 + 1):
                        exy_cd = ex_cd * E(m3, m4, nu, cdy, c, d)
                        for phi in range(n3 + n4 + 1):
                            sign = -1.0 if ((tau + nu + phi) & 1) else 1.0
                            val += (
                                xyz_ab
                                * exy_cd
                                * E(n3, n4, phi, cdz, c, d)
                                * sign
                                * R(t + tau, u + nu, v + phi, 0, alpha, dx, dy, dz, rpq)
                            )

    return val * (ERI_PREFAC / (p * q * math.sqrt(p + q)))


def electron_repulsion(a,lmn1,A,b,lmn2,B,c,lmn3,C,d,lmn4,D):
    ''' Evaluates kinetic energy integral between two Gaussians
    Returns a float.
    a,b,c,d: orbital exponent on Gaussian 'a','b','c','d'
    lmn1,lmn2
    lmn3,lmn4: int tuple containing orbital angular momentum
    for Gaussian 'a','b','c','d', respectively
    A,B,C,D: list containing origin of Gaussian 'a','b','c','d'
    '''
    l1,m1,n1 = lmn1
    l2,m2,n2 = lmn2
    l3,m3,n3 = lmn3
    l4,m4,n4 = lmn4
    return _electron_repulsion_cached(
        a, l1, m1, n1, A[0], A[1], A[2],
        b, l2, m2, n2, B[0], B[1], B[2],
        c, l3, m3, n3, C[0], C[1], C[2],
        d, l4, m4, n4, D[0], D[1], D[2],
    )

def ERI(a,b,c,d):
    '''Evaluates overlap between two contracted Gaussians
    Returns float.
    Arguments:
    a: contracted Gaussian 'a', BasisFunction object
    b: contracted Gaussian 'b', BasisFunction object
    c: contracted Gaussian 'b', BasisFunction object
    d: contracted Gaussian 'b', BasisFunction object
    '''
    eri = 0.0
    for ja, wa in enumerate(a.prim_weights):
        for jb, wb in enumerate(b.prim_weights):
            for jc, wc in enumerate(c.prim_weights):
                for jd, wd in enumerate(d.prim_weights):
                    eri += wa*wb*wc*wd*\
                    electron_repulsion(a.exps[ja],a.shell,a.origin,\
                    b.exps[jb],b.shell,b.origin,\
                    c.exps[jc],c.shell,c.origin,\
                    d.exps[jd],d.shell,d.origin)
    return eri


def _basis_signature(basis_fn):
    return (
        tuple(int(x) for x in basis_fn.shell),
        tuple(float(x) for x in basis_fn.origin),
        tuple(float(x) for x in basis_fn.exps),
        tuple(float(x) for x in basis_fn.prim_weights),
    )


def _canonical_quartet_signature(sig_a, sig_b, sig_c, sig_d):
    pair_ab = (sig_a, sig_b) if sig_a <= sig_b else (sig_b, sig_a)
    pair_cd = (sig_c, sig_d) if sig_c <= sig_d else (sig_d, sig_c)
    if pair_ab <= pair_cd:
        return pair_ab + pair_cd
    return pair_cd + pair_ab


@lru_cache(maxsize=262144)
def _contracted_eri_from_signatures_cached(sig_a, sig_b, sig_c, sig_d):
    shell_a, origin_a, exps_a, weights_a = sig_a
    shell_b, origin_b, exps_b, weights_b = sig_b
    shell_c, origin_c, exps_c, weights_c = sig_c
    shell_d, origin_d, exps_d, weights_d = sig_d

    eri = 0.0
    for ja, wa in enumerate(weights_a):
        for jb, wb in enumerate(weights_b):
            for jc, wc in enumerate(weights_c):
                for jd, wd in enumerate(weights_d):
                    eri += wa * wb * wc * wd * _electron_repulsion_cached(
                        exps_a[ja], shell_a[0], shell_a[1], shell_a[2], origin_a[0], origin_a[1], origin_a[2],
                        exps_b[jb], shell_b[0], shell_b[1], shell_b[2], origin_b[0], origin_b[1], origin_b[2],
                        exps_c[jc], shell_c[0], shell_c[1], shell_c[2], origin_c[0], origin_c[1], origin_c[2],
                        exps_d[jd], shell_d[0], shell_d[1], shell_d[2], origin_d[0], origin_d[1], origin_d[2],
                    )
    return eri


def _contracted_eri_from_signatures(sig_a, sig_b, sig_c, sig_d):
    return _contracted_eri_from_signatures_cached(
        *_canonical_quartet_signature(sig_a, sig_b, sig_c, sig_d)
    )


def _pack_signatures_for_numba(signatures):
    nsig = len(signatures)
    max_prim = max(len(sig[2]) for sig in signatures) if signatures else 0
    shells = np.zeros((nsig, 3), dtype=np.int64)
    origins = np.zeros((nsig, 3), dtype=np.float64)
    exps = np.zeros((nsig, max_prim), dtype=np.float64)
    weights = np.zeros((nsig, max_prim), dtype=np.float64)
    nprim = np.zeros((nsig,), dtype=np.int64)

    for idx, sig in enumerate(signatures):
        shell, origin, sig_exps, sig_weights = sig
        shells[idx, :] = np.asarray(shell, dtype=np.int64)
        origins[idx, :] = np.asarray(origin, dtype=np.float64)
        n = len(sig_exps)
        nprim[idx] = n
        exps[idx, :n] = np.asarray(sig_exps, dtype=np.float64)
        weights[idx, :n] = np.asarray(sig_weights, dtype=np.float64)

    return shells, origins, exps, weights, nprim


def _compute_dense_eri_serial_c(signatures, pair_bounds, screen_tol):
    lib = _load_basis_accel()
    if not lib:
        return None

    shells, origins, exps, weights, nprim = _pack_signatures_for_numba(signatures)
    nao = len(signatures)
    max_prim = exps.shape[1]
    eri = np.zeros((nao, nao, nao, nao), dtype=np.float64)
    computed = ctypes.c_longlong(0)
    skipped = ctypes.c_longlong(0)
    status = lib.compute_dense_eri(
        int(nao),
        int(max_prim),
        np.ascontiguousarray(shells.reshape(-1), dtype=np.int64),
        np.ascontiguousarray(origins.reshape(-1), dtype=np.float64),
        np.ascontiguousarray(exps.reshape(-1), dtype=np.float64),
        np.ascontiguousarray(weights.reshape(-1), dtype=np.float64),
        np.ascontiguousarray(nprim.reshape(-1), dtype=np.int64),
        np.ascontiguousarray(np.asarray(pair_bounds, dtype=np.float64).reshape(-1)),
        float(screen_tol),
        eri.reshape(-1),
        ctypes.byref(computed),
        ctypes.byref(skipped),
    )
    if status != 0:
        return None
    return eri, int(computed.value), int(skipped.value)


def _compute_dense_eri_serial_cython(signatures, pair_bounds, screen_tol):
    if _basis_cy is None:
        return None

    shells, origins, exps, weights, nprim = _pack_signatures_for_numba(signatures)
    try:
        eri, computed, skipped = _basis_cy.compute_dense_eri(
            np.ascontiguousarray(shells, dtype=np.int64),
            np.ascontiguousarray(origins, dtype=np.float64),
            np.ascontiguousarray(exps, dtype=np.float64),
            np.ascontiguousarray(weights, dtype=np.float64),
            np.ascontiguousarray(nprim, dtype=np.int64),
            np.ascontiguousarray(pair_bounds, dtype=np.float64),
            float(screen_tol),
        )
    except Exception:
        return None
    return np.asarray(eri, dtype=np.float64), int(computed), int(skipped)


def _compute_dense_eri_serial_cython_blocked(signatures, pair_bounds, screen_tol):
    if _basis_cy is None:
        return None

    shells, origins, exps, weights, nprim = _pack_signatures_for_numba(signatures)
    shell_blocks = _contiguous_shell_blocks_from_signatures(signatures)
    shell_starts = np.asarray([start for start, _ in shell_blocks], dtype=np.int64)
    shell_stops = np.asarray([stop for _, stop in shell_blocks], dtype=np.int64)
    try:
        eri, computed, skipped = _basis_cy.compute_dense_eri_blocked(
            np.ascontiguousarray(shells, dtype=np.int64),
            np.ascontiguousarray(origins, dtype=np.float64),
            np.ascontiguousarray(exps, dtype=np.float64),
            np.ascontiguousarray(weights, dtype=np.float64),
            np.ascontiguousarray(nprim, dtype=np.int64),
            np.ascontiguousarray(pair_bounds, dtype=np.float64),
            np.ascontiguousarray(shell_starts, dtype=np.int64),
            np.ascontiguousarray(shell_stops, dtype=np.int64),
            float(screen_tol),
        )
    except Exception:
        return None
    return np.asarray(eri, dtype=np.float64), int(computed), int(skipped)


def _compute_cartesian_shell_quartet_block_cython(signatures, shell_block):
    if _basis_cy is None:
        return None
    shells, origins, exps, weights, nprim = _pack_signatures_for_numba(signatures)
    p0, p1, q0, q1, r0, r1, s0, s1 = map(int, shell_block)
    try:
        block = _basis_cy.compute_cartesian_shell_quartet_block(
            np.ascontiguousarray(shells, dtype=np.int64),
            np.ascontiguousarray(origins, dtype=np.float64),
            np.ascontiguousarray(exps, dtype=np.float64),
            np.ascontiguousarray(weights, dtype=np.float64),
            np.ascontiguousarray(nprim, dtype=np.int64),
            p0, p1, q0, q1, r0, r1, s0, s1,
        )
    except Exception:
        return None
    return np.asarray(block, dtype=np.float64)


def _pivoted_cholesky_from_integral_oracle_cython(signatures, pair_bounds, tol=1e-8, max_rank=None, screen_tol=0.0):
    if _basis_cy is None:
        return None

    shells, origins, exps, weights, nprim = _pack_signatures_for_numba(signatures)
    try:
        factors_packed, pairs = _basis_cy.compute_pivoted_cholesky_factors(
            np.ascontiguousarray(shells, dtype=np.int64),
            np.ascontiguousarray(origins, dtype=np.float64),
            np.ascontiguousarray(exps, dtype=np.float64),
            np.ascontiguousarray(weights, dtype=np.float64),
            np.ascontiguousarray(nprim, dtype=np.int64),
            np.ascontiguousarray(pair_bounds, dtype=np.float64),
            float(tol),
            max_rank,
            float(screen_tol),
        )
    except Exception:
        return None

    factors_packed = np.asarray(factors_packed, dtype=np.float64)
    pairs = np.asarray(pairs, dtype=np.int64)
    return _unpack_packed_pair_factors(factors_packed, pairs, len(signatures))


def _pivoted_cholesky_from_integral_oracle_cython_blocked(signatures, pair_bounds, shell_starts, shell_stops, tol=1e-8, max_rank=None, screen_tol=0.0):
    if _basis_cy is None:
        return None

    shells, origins, exps, weights, nprim = _pack_signatures_for_numba(signatures)
    try:
        factors_packed, pairs = _basis_cy.compute_pivoted_cholesky_factors_blocked(
            np.ascontiguousarray(shells, dtype=np.int64),
            np.ascontiguousarray(origins, dtype=np.float64),
            np.ascontiguousarray(exps, dtype=np.float64),
            np.ascontiguousarray(weights, dtype=np.float64),
            np.ascontiguousarray(nprim, dtype=np.int64),
            np.ascontiguousarray(pair_bounds, dtype=np.float64),
            np.ascontiguousarray(shell_starts, dtype=np.int64),
            np.ascontiguousarray(shell_stops, dtype=np.int64),
            float(tol),
            max_rank,
            float(screen_tol),
        )
    except Exception:
        return None

    factors_packed = np.asarray(factors_packed, dtype=np.float64)
    pairs = np.asarray(pairs, dtype=np.int64)
    return _unpack_packed_pair_factors(factors_packed, pairs, len(signatures))


if _NUMBA_AVAILABLE:
    @njit(cache=True)
    def _boys_numba(n, T):
        if T < 1e-12:
            return 1.0 / (2.0 * n + 1.0)

        if T < 30.0:
            term = 1.0 / (2.0 * n + 1.0)
            value = term
            for k in range(1, 256):
                term *= -T / k
                add = term / (2.0 * n + 2.0 * k + 1.0)
                value += add
                if abs(add) < 1e-16:
                    break
            return value

        sqrt_T = math.sqrt(T)
        value = 0.5 * math.sqrt(math.pi / T) * math.erf(sqrt_T)
        if n == 0:
            return value

        exp_T = math.exp(-T)
        for m in range(n):
            value = ((2.0 * m + 1.0) * value - exp_T) / (2.0 * T)
        return value


    @njit(cache=True)
    def _E_numba(i, j, t, Qx, a, b, memo):
        if t < 0 or t > (i + j):
            return 0.0

        cached = memo[i, j, t]
        if not math.isnan(cached):
            return cached

        p = a + b
        q = a * b / p
        if i == 0 and j == 0 and t == 0:
            value = math.exp(-q * Qx * Qx)
        elif j == 0:
            value = (
                (1.0 / (2.0 * p)) * _E_numba(i - 1, j, t - 1, Qx, a, b, memo)
                - (q * Qx / a) * _E_numba(i - 1, j, t, Qx, a, b, memo)
                + (t + 1.0) * _E_numba(i - 1, j, t + 1, Qx, a, b, memo)
            )
        else:
            value = (
                (1.0 / (2.0 * p)) * _E_numba(i, j - 1, t - 1, Qx, a, b, memo)
                + (q * Qx / b) * _E_numba(i, j - 1, t, Qx, a, b, memo)
                + (t + 1.0) * _E_numba(i, j - 1, t + 1, Qx, a, b, memo)
            )

        memo[i, j, t] = value
        return value


    @njit(cache=True)
    def _R_numba(t, u, v, n, p, PCx, PCy, PCz, RPC, memo):
        if t < 0 or u < 0 or v < 0:
            return 0.0

        cached = memo[t, u, v, n]
        if not math.isnan(cached):
            return cached

        if t == 0 and u == 0 and v == 0:
            value = ((-2.0 * p) ** n) * _boys_numba(n, p * RPC * RPC)
        elif t == 0 and u == 0:
            value = 0.0
            if v > 1:
                value += (v - 1.0) * _R_numba(t, u, v - 2, n + 1, p, PCx, PCy, PCz, RPC, memo)
            value += PCz * _R_numba(t, u, v - 1, n + 1, p, PCx, PCy, PCz, RPC, memo)
        elif t == 0:
            value = 0.0
            if u > 1:
                value += (u - 1.0) * _R_numba(t, u - 2, v, n + 1, p, PCx, PCy, PCz, RPC, memo)
            value += PCy * _R_numba(t, u - 1, v, n + 1, p, PCx, PCy, PCz, RPC, memo)
        else:
            value = 0.0
            if t > 1:
                value += (t - 1.0) * _R_numba(t - 2, u, v, n + 1, p, PCx, PCy, PCz, RPC, memo)
            value += PCx * _R_numba(t - 1, u, v, n + 1, p, PCx, PCy, PCz, RPC, memo)

        memo[t, u, v, n] = value
        return value


    @njit(cache=True)
    def _primitive_eri_numba(
        a, l1, m1, n1, Ax, Ay, Az,
        b, l2, m2, n2, Bx, By, Bz,
        c, l3, m3, n3, Cx, Cy, Cz,
        d, l4, m4, n4, Dx, Dy, Dz,
    ):
        p = a + b
        q = c + d
        alpha = p * q / (p + q)

        px = (a * Ax + b * Bx) / p
        py = (a * Ay + b * By) / p
        pz = (a * Az + b * Bz) / p
        qx = (c * Cx + d * Dx) / q
        qy = (c * Cy + d * Dy) / q
        qz = (c * Cz + d * Dz) / q
        dx = px - qx
        dy = py - qy
        dz = pz - qz
        rpq = math.sqrt(dx * dx + dy * dy + dz * dz)

        abx = Ax - Bx
        aby = Ay - By
        abz = Az - Bz
        cdx = Cx - Dx
        cdy = Cy - Dy
        cdz = Cz - Dz

        memo_abx = np.full((l1 + 1, l2 + 1, l1 + l2 + 1), np.nan)
        memo_aby = np.full((m1 + 1, m2 + 1, m1 + m2 + 1), np.nan)
        memo_abz = np.full((n1 + 1, n2 + 1, n1 + n2 + 1), np.nan)
        memo_cdx = np.full((l3 + 1, l4 + 1, l3 + l4 + 1), np.nan)
        memo_cdy = np.full((m3 + 1, m4 + 1, m3 + m4 + 1), np.nan)
        memo_cdz = np.full((n3 + 1, n4 + 1, n3 + n4 + 1), np.nan)

        tx_max = l1 + l2 + l3 + l4
        uy_max = m1 + m2 + m3 + m4
        vz_max = n1 + n2 + n3 + n4
        nmax = tx_max + uy_max + vz_max + 1
        memo_r = np.full((tx_max + 1, uy_max + 1, vz_max + 1, nmax), np.nan)

        value = 0.0
        for t in range(l1 + l2 + 1):
            ex_ab = _E_numba(l1, l2, t, abx, a, b, memo_abx)
            for u in range(m1 + m2 + 1):
                exy_ab = ex_ab * _E_numba(m1, m2, u, aby, a, b, memo_aby)
                for v in range(n1 + n2 + 1):
                    xyz_ab = exy_ab * _E_numba(n1, n2, v, abz, a, b, memo_abz)
                    for tau in range(l3 + l4 + 1):
                        ex_cd = _E_numba(l3, l4, tau, cdx, c, d, memo_cdx)
                        for nu in range(m3 + m4 + 1):
                            exy_cd = ex_cd * _E_numba(m3, m4, nu, cdy, c, d, memo_cdy)
                            for phi in range(n3 + n4 + 1):
                                sign = -1.0 if ((tau + nu + phi) & 1) else 1.0
                                value += (
                                    xyz_ab
                                    * exy_cd
                                    * _E_numba(n3, n4, phi, cdz, c, d, memo_cdz)
                                    * sign
                                    * _R_numba(
                                        t + tau,
                                        u + nu,
                                        v + phi,
                                        0,
                                        alpha,
                                        dx,
                                        dy,
                                        dz,
                                        rpq,
                                        memo_r,
                                    )
                                )

        return value * (ERI_PREFAC / (p * q * math.sqrt(p + q)))


    @njit(cache=True)
    def _contracted_eri_numba(shells, origins, exps, weights, nprim, p, q, r, s):
        value = 0.0
        np_p = nprim[p]
        np_q = nprim[q]
        np_r = nprim[r]
        np_s = nprim[s]

        for ip in range(np_p):
            wp = weights[p, ip]
            ap = exps[p, ip]
            for iq in range(np_q):
                wq = weights[q, iq]
                aq = exps[q, iq]
                for ir in range(np_r):
                    wr = weights[r, ir]
                    ar = exps[r, ir]
                    for is_ in range(np_s):
                        ws = weights[s, is_]
                        a_s = exps[s, is_]
                        value += (
                            wp
                            * wq
                            * wr
                            * ws
                            * _primitive_eri_numba(
                                ap,
                                shells[p, 0], shells[p, 1], shells[p, 2],
                                origins[p, 0], origins[p, 1], origins[p, 2],
                                aq,
                                shells[q, 0], shells[q, 1], shells[q, 2],
                                origins[q, 0], origins[q, 1], origins[q, 2],
                                ar,
                                shells[r, 0], shells[r, 1], shells[r, 2],
                                origins[r, 0], origins[r, 1], origins[r, 2],
                                a_s,
                                shells[s, 0], shells[s, 1], shells[s, 2],
                                origins[s, 0], origins[s, 1], origins[s, 2],
                            )
                        )
        return value


    @njit(cache=True)
    def _compute_dense_eri_serial_numba(shells, origins, exps, weights, nprim, pair_bounds, screen_tol):
        nao = shells.shape[0]
        eri = np.zeros((nao, nao, nao, nao), dtype=np.float64)
        computed = 0
        skipped = 0

        for p in range(nao):
            for q in range(p + 1):
                bound_pq = pair_bounds[p, q]
                for r in range(p + 1):
                    s_max = q if r == p else r
                    for s in range(s_max + 1):
                        if screen_tol > 0.0 and bound_pq * pair_bounds[r, s] < screen_tol:
                            skipped += 1
                            continue

                        value = _contracted_eri_numba(shells, origins, exps, weights, nprim, p, q, r, s)
                        if screen_tol > 0.0 and abs(value) < screen_tol:
                            skipped += 1
                            continue

                        eri[p, q, r, s] = value
                        eri[q, p, r, s] = value
                        eri[p, q, s, r] = value
                        eri[q, p, s, r] = value
                        eri[r, s, p, q] = value
                        eri[s, r, p, q] = value
                        eri[r, s, q, p] = value
                        eri[s, r, q, p] = value
                        computed += 1

        return eri, computed, skipped


def _unique_ao_pairs(nao):
    npair = nao * (nao + 1) // 2
    pairs = np.empty((npair, 2), dtype=np.int32)
    idx = 0
    for p in range(nao):
        nfill = p + 1
        pairs[idx : idx + nfill, 0] = p
        pairs[idx : idx + nfill, 1] = np.arange(nfill, dtype=np.int32)
        idx += nfill
    return pairs


def _compute_pair_bounds(signatures):
    nao = len(signatures)
    bounds = np.zeros((nao, nao), dtype=float)
    for p in range(nao):
        sig_p = signatures[p]
        for q in range(p + 1):
            val = _contracted_eri_from_signatures(sig_p, signatures[q], sig_p, signatures[q])
            bound = math.sqrt(max(abs(float(np.real(val))), 0.0))
            bounds[p, q] = bounds[q, p] = bound
    return bounds


def _store_eri_eightfold(eri, p, q, r, s, value):
    eri[p, q, r, s] = value
    eri[q, p, r, s] = value
    eri[p, q, s, r] = value
    eri[q, p, s, r] = value
    eri[r, s, p, q] = value
    eri[s, r, p, q] = value
    eri[r, s, q, p] = value
    eri[s, r, q, p] = value


def _store_eri_eightfold_batch(eri, p, q, r, s, values):
    p, q, r, s, values = np.broadcast_arrays(
        np.asarray(p, dtype=np.intp),
        np.asarray(q, dtype=np.intp),
        np.asarray(r, dtype=np.intp),
        np.asarray(s, dtype=np.intp),
        np.asarray(values, dtype=float),
    )
    eri[p, q, r, s] = values
    eri[q, p, r, s] = values
    eri[p, q, s, r] = values
    eri[q, p, s, r] = values
    eri[r, s, p, q] = values
    eri[s, r, p, q] = values
    eri[r, s, q, p] = values
    eri[s, r, q, p] = values


def _contracted_shell_key_from_signature(sig):
    shell, origin, exps, weights = sig
    return (sum(shell), origin, exps, weights)


def _shell_blocks_from_signatures(signatures):
    blocks = {}
    for idx, sig in enumerate(signatures):
        key = _contracted_shell_key_from_signature(sig)
        if key not in blocks:
            blocks[key] = []
        blocks[key].append(idx)
    return [np.asarray(indices, dtype=np.int32) for indices in blocks.values()]


def _contiguous_shell_blocks_from_signatures(signatures):
    blocks = []
    start = 0
    nao = len(signatures)
    while start < nao:
        shell, origin, exps, _weights = signatures[start]
        l = sum(shell)
        ncart = (l + 1) * (l + 2) // 2
        stop = start + ncart
        if stop > nao:
            raise ValueError("Invalid Cartesian shell partition in signature list.")
        ref = (l, origin, exps)
        if any((sum(signatures[k][0]), signatures[k][1], signatures[k][2]) != ref for k in range(start, stop)):
            raise ValueError("Signature ordering is not contiguous by Cartesian shell.")
        blocks.append((start, stop))
        start = stop
    return blocks


def _shell_blocks_from_basis_cart(basis_cart):
    blocks = []
    start = 0
    nao = len(basis_cart)
    while start < nao:
        sig = _cart_shell_signature(basis_cart[start])
        l = sig[1]
        ncart = (l + 1) * (l + 2) // 2
        stop = start + ncart
        if stop > nao:
            raise ValueError("Invalid Cartesian shell partition in builtin basis.")
        if any(_cart_shell_signature(basis_cart[k]) != sig for k in range(start, stop)):
            raise ValueError("Builtin Cartesian shell ordering is not contiguous.")
        blocks.append((start, stop))
        start = stop
    return blocks


def _compute_one_electron_shellblocked(basis_cart, atcoords, atnums):
    nao = len(basis_cart)
    overlap_mat = np.eye(nao, dtype=float)
    kinetic_mat = np.zeros((nao, nao), dtype=float)
    vnuc_mat = np.zeros((nao, nao), dtype=float)

    shell_blocks = _shell_blocks_from_basis_cart(basis_cart)
    for bi, (istart, istop) in enumerate(shell_blocks):
        block_i = basis_cart[istart:istop]
        ni = istop - istart
        for bj in range(bi + 1):
            jstart, jstop = shell_blocks[bj]
            block_j = basis_cart[jstart:jstop]
            nj = jstop - jstart

            s_block = np.zeros((ni, nj), dtype=float)
            t_block = np.zeros((ni, nj), dtype=float)
            v_block = np.zeros((ni, nj), dtype=float)

            for ii, gto_i in enumerate(block_i):
                jj_stop = ii + 1 if bi == bj else nj
                for jj in range(jj_stop):
                    gto_j = block_j[jj]
                    s_ij = float(S(gto_i, gto_j))
                    t_ij = float(T(gto_i, gto_j))
                    v_ij = 0.0
                    for c in range(len(atnums)):
                        v_ij -= atnums[c] * float(point_charge(gto_i, gto_j, atcoords[c]))
                    s_block[ii, jj] = s_ij
                    t_block[ii, jj] = t_ij
                    v_block[ii, jj] = v_ij
                    if bi == bj and ii != jj:
                        s_block[jj, ii] = s_ij
                        t_block[jj, ii] = t_ij
                        v_block[jj, ii] = v_ij

            overlap_mat[istart:istop, jstart:jstop] = s_block
            kinetic_mat[istart:istop, jstart:jstop] = t_block
            vnuc_mat[istart:istop, jstart:jstop] = v_block
            if bi != bj:
                overlap_mat[jstart:jstop, istart:istop] = s_block.T
                kinetic_mat[jstart:jstop, istart:istop] = t_block.T
                vnuc_mat[jstart:jstop, istart:istop] = v_block.T

    return overlap_mat, kinetic_mat, vnuc_mat


def _compute_one_electron_shellblocked_cython(signatures, atcoords, atnums):
    if _basis_cy is None:
        return None
    shells, origins, exps, weights, nprim = _pack_signatures_for_numba(signatures)
    try:
        overlap, kinetic, vnuc = _basis_cy.compute_one_electron(
            np.ascontiguousarray(shells, dtype=np.int64),
            np.ascontiguousarray(origins, dtype=np.float64),
            np.ascontiguousarray(exps, dtype=np.float64),
            np.ascontiguousarray(weights, dtype=np.float64),
            np.ascontiguousarray(nprim, dtype=np.int64),
            np.ascontiguousarray(atcoords, dtype=np.float64),
            np.ascontiguousarray(atnums, dtype=np.float64),
        )
    except Exception:
        return None
    return (
        np.asarray(overlap, dtype=np.float64),
        np.asarray(kinetic, dtype=np.float64),
        np.asarray(vnuc, dtype=np.float64),
    )


def _shell_pair_blocks_from_signatures(signatures, pair_bounds):
    shell_blocks = _shell_blocks_from_signatures(signatures)
    pair_blocks = []
    pair_to_index = {(int(p), int(q)): idx for idx, (p, q) in enumerate(_unique_ao_pairs(len(signatures)))}

    for i, block_i in enumerate(shell_blocks):
        for j in range(i + 1):
            block_j = shell_blocks[j]
            p_idx = np.repeat(block_i, block_j.size).astype(np.int32, copy=False)
            q_idx = np.tile(block_j, block_i.size).astype(np.int32, copy=False)
            if i == j:
                mask = p_idx >= q_idx
                p_idx = p_idx[mask]
                q_idx = q_idx[mask]

            pair_block_bounds = pair_bounds[p_idx, q_idx]
            pair_indices = np.asarray(
                [pair_to_index[(int(p), int(q))] for p, q in zip(p_idx, q_idx)],
                dtype=np.int32,
            )
            pair_blocks.append(
                {
                    "p": p_idx,
                    "q": q_idx,
                    "pair_indices": pair_indices,
                    "bounds": pair_block_bounds,
                    "bound_max": float(pair_block_bounds.max()) if pair_block_bounds.size else 0.0,
                    "sig_p": tuple(signatures[int(p)] for p in p_idx),
                    "sig_q": tuple(signatures[int(q)] for q in q_idx),
                }
            )

    return pair_blocks


def _compute_dense_eri_serial_aopairs(signatures, pair_bounds, screen_tol):
    nao = len(signatures)
    eri = np.zeros((nao, nao, nao, nao), dtype=float)
    pairs = _unique_ao_pairs(nao)
    pair_bound_vec = pair_bounds[pairs[:, 0], pairs[:, 1]]
    skipped = 0
    computed = 0

    for pq_idx in range(len(pairs)):
        p, q = pairs[pq_idx]
        bound_pq = pair_bound_vec[pq_idx]
        sig_p = signatures[p]
        sig_q = signatures[q]
        r_keep = []
        s_keep = []
        value_keep = []

        for rs_idx in range(pq_idx + 1):
            r, s = pairs[rs_idx]
            if screen_tol > 0.0 and bound_pq * pair_bound_vec[rs_idx] < screen_tol:
                skipped += 1
                continue

            value = float(
                _contracted_eri_from_signatures(sig_p, sig_q, signatures[r], signatures[s])
            )
            if screen_tol > 0.0 and abs(value) < screen_tol:
                skipped += 1
                continue

            r_keep.append(r)
            s_keep.append(s)
            value_keep.append(value)
            computed += 1

        if value_keep:
            _store_eri_eightfold_batch(
                eri,
                p,
                q,
                np.asarray(r_keep, dtype=np.intp),
                np.asarray(s_keep, dtype=np.intp),
                np.asarray(value_keep, dtype=float),
            )

    return eri, computed, skipped


def _compute_dense_eri_serial_shellblocked(signatures, pair_bounds, screen_tol):
    nao = len(signatures)
    eri = np.zeros((nao, nao, nao, nao), dtype=float)
    pair_blocks = _shell_pair_blocks_from_signatures(signatures, pair_bounds)
    skipped = 0
    computed = 0

    for pq_block_idx, pq_block in enumerate(pair_blocks):
        pq_bound = pq_block["bound_max"]
        pq_bounds = pq_block["bounds"]
        pq_p = pq_block["p"]
        pq_q = pq_block["q"]
        pq_sig_p = pq_block["sig_p"]
        pq_sig_q = pq_block["sig_q"]
        npq = len(pq_p)

        for rs_block_idx in range(pq_block_idx + 1):
            rs_block = pair_blocks[rs_block_idx]
            nrs = len(rs_block["p"])
            if screen_tol > 0.0 and pq_bound * rs_block["bound_max"] < screen_tol:
                if pq_block_idx == rs_block_idx:
                    skipped += npq * (npq + 1) // 2
                else:
                    skipped += npq * nrs
                continue

            rs_bounds = rs_block["bounds"]
            rs_p = rs_block["p"]
            rs_q = rs_block["q"]
            rs_sig_p = rs_block["sig_p"]
            rs_sig_q = rs_block["sig_q"]

            p_keep = []
            q_keep = []
            r_keep = []
            s_keep = []
            values_keep = []

            for pq_local in range(npq):
                p = int(pq_p[pq_local])
                q = int(pq_q[pq_local])
                bound_pq = float(pq_bounds[pq_local])
                sig_p = pq_sig_p[pq_local]
                sig_q = pq_sig_q[pq_local]
                rs_stop = pq_local + 1 if pq_block_idx == rs_block_idx else nrs

                for rs_local in range(rs_stop):
                    if screen_tol > 0.0 and bound_pq * float(rs_bounds[rs_local]) < screen_tol:
                        skipped += 1
                        continue

                    value = float(
                        _contracted_eri_from_signatures(
                            sig_p, sig_q, rs_sig_p[rs_local], rs_sig_q[rs_local]
                        )
                    )
                    if screen_tol > 0.0 and abs(value) < screen_tol:
                        skipped += 1
                        continue

                    p_keep.append(p)
                    q_keep.append(q)
                    r_keep.append(int(rs_p[rs_local]))
                    s_keep.append(int(rs_q[rs_local]))
                    values_keep.append(value)
                    computed += 1

            if values_keep:
                _store_eri_eightfold_batch(
                    eri,
                    np.asarray(p_keep, dtype=np.intp),
                    np.asarray(q_keep, dtype=np.intp),
                    np.asarray(r_keep, dtype=np.intp),
                    np.asarray(s_keep, dtype=np.intp),
                    np.asarray(values_keep, dtype=float),
                )

    return eri, computed, skipped


def _compute_dense_eri_serial(signatures, pair_bounds, screen_tol):
    cy_result = _compute_dense_eri_serial_cython(signatures, pair_bounds, screen_tol)
    if cy_result is not None:
        return cy_result

    blocked_cy_result = _compute_dense_eri_serial_cython_blocked(signatures, pair_bounds, screen_tol)
    if blocked_cy_result is not None:
        return blocked_cy_result

    c_result = _compute_dense_eri_serial_c(signatures, pair_bounds, screen_tol)
    if c_result is not None:
        return c_result

    if _NUMBA_AVAILABLE and _NUMBA_DENSE_ERI_ENABLED:
        shells, origins, exps, weights, nprim = _pack_signatures_for_numba(signatures)
        eri, computed, skipped = _compute_dense_eri_serial_numba(
            shells, origins, exps, weights, nprim, np.asarray(pair_bounds, dtype=np.float64), float(screen_tol)
        )
        return eri, int(computed), int(skipped)

    return _compute_dense_eri_serial_shellblocked(signatures, pair_bounds, screen_tol)


def _init_builtin_eri_worker(signatures, pairs, pair_bounds, screen_tol):
    global _NATIVE_PAR_SIGNATURES, _NATIVE_PAR_PAIRS, _NATIVE_PAR_PAIR_BOUNDS, _NATIVE_PAR_PAIR_BOUND_VEC, _NATIVE_PAR_SCREEN_TOL
    _NATIVE_PAR_SIGNATURES = signatures
    _NATIVE_PAR_PAIRS = pairs
    _NATIVE_PAR_PAIR_BOUNDS = pair_bounds
    _NATIVE_PAR_PAIR_BOUND_VEC = pair_bounds[pairs[:, 0], pairs[:, 1]]
    _NATIVE_PAR_SCREEN_TOL = float(screen_tol)


def _eri_chunk_worker(start, stop):
    p_idx = []
    q_idx = []
    r_idx = []
    s_idx = []
    values = []
    skipped = 0

    for pq_idx in range(start, stop):
        p, q = _NATIVE_PAR_PAIRS[pq_idx]
        bound_pq = _NATIVE_PAR_PAIR_BOUND_VEC[pq_idx]
        sig_p = _NATIVE_PAR_SIGNATURES[p]
        sig_q = _NATIVE_PAR_SIGNATURES[q]
        for rs_idx in range(pq_idx + 1):
            r, s = _NATIVE_PAR_PAIRS[rs_idx]
            if _NATIVE_PAR_SCREEN_TOL > 0.0 and bound_pq * _NATIVE_PAR_PAIR_BOUND_VEC[rs_idx] < _NATIVE_PAR_SCREEN_TOL:
                skipped += 1
                continue

            value = float(
                _contracted_eri_from_signatures(sig_p, sig_q, _NATIVE_PAR_SIGNATURES[r], _NATIVE_PAR_SIGNATURES[s])
            )
            if _NATIVE_PAR_SCREEN_TOL > 0.0 and abs(value) < _NATIVE_PAR_SCREEN_TOL:
                skipped += 1
                continue

            p_idx.append(p)
            q_idx.append(q)
            r_idx.append(r)
            s_idx.append(s)
            values.append(value)

    return (
        np.asarray(p_idx, dtype=np.int32),
        np.asarray(q_idx, dtype=np.int32),
        np.asarray(r_idx, dtype=np.int32),
        np.asarray(s_idx, dtype=np.int32),
        np.asarray(values, dtype=float),
        skipped,
    )


def _compute_dense_eri_parallel(signatures, pair_bounds, screen_tol, workers):
    nao = len(signatures)
    eri = np.zeros((nao, nao, nao, nao), dtype=float)
    pairs = _unique_ao_pairs(nao)
    if len(pairs) == 0:
        return eri, 0, 0

    chunk = max(1, len(pairs) // max(workers * 4, 1))
    tasks = [(start, min(start + chunk, len(pairs))) for start in range(0, len(pairs), chunk)]
    skipped = 0
    computed = 0

    start_methods = mp.get_all_start_methods()
    ctx = mp.get_context("fork") if "fork" in start_methods else None
    executor_kwargs = {}
    if ctx is not None:
        executor_kwargs["mp_context"] = ctx

    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_builtin_eri_worker,
            initargs=(signatures, pairs, pair_bounds, screen_tol),
            **executor_kwargs,
        ) as pool:
            for p_idx, q_idx, r_idx, s_idx, values, chunk_skipped in pool.map(
                _eri_chunk_worker_from_task, tasks
            ):
                skipped += int(chunk_skipped)
                computed += int(values.size)
                if values.size:
                    _store_eri_eightfold_batch(eri, p_idx, q_idx, r_idx, s_idx, values)
    except (PermissionError, OSError):
        return _compute_dense_eri_serial(signatures, pair_bounds, screen_tol)

    return eri, computed, skipped


def _eri_chunk_worker_from_task(task):
    return _eri_chunk_worker(*task)


def _unpack_packed_pair_factors(factors_packed, pairs, nao):
    rank = factors_packed.shape[0]
    factors = np.zeros((rank, nao, nao), dtype=factors_packed.dtype)
    for pair_idx, (i, j) in enumerate(pairs):
        factors[:, i, j] = factors_packed[:, pair_idx]
        if i != j:
            factors[:, j, i] = factors_packed[:, pair_idx]
    return factors


def _pivoted_cholesky_from_integral_oracle(signatures, pair_bounds, tol=1e-8, max_rank=None, screen_tol=0.0):
    cy_result = _pivoted_cholesky_from_integral_oracle_cython(
        signatures,
        pair_bounds,
        tol=tol,
        max_rank=max_rank,
        screen_tol=screen_tol,
    )
    if cy_result is not None:
        return cy_result

    nao = len(signatures)
    pairs = _unique_ao_pairs(nao)
    npair = len(pairs)
    if max_rank is None:
        max_rank = npair
    max_rank = min(int(max_rank), npair)

    diag = np.array(
        [max(float(np.real(pair_bounds[i, j] * pair_bounds[i, j])), 0.0) for i, j in pairs],
        dtype=float,
    )
    chol = np.zeros((npair, max_rank), dtype=float)
    rank = 0
    pair_blocks = _shell_pair_blocks_from_signatures(signatures, pair_bounds)

    for _ in range(max_rank):
        pivot = int(np.argmax(diag))
        delta = float(diag[pivot])
        if delta <= tol:
            break

        pi, pj = pairs[pivot]
        pivot_bound = pair_bounds[pi, pj]
        col = np.zeros(npair, dtype=float)
        pivot_sig_i = signatures[pi]
        pivot_sig_j = signatures[pj]
        for pair_block in pair_blocks:
            if screen_tol > 0.0 and pivot_bound * pair_block["bound_max"] < screen_tol:
                continue

            pair_indices = pair_block["pair_indices"]
            values = np.zeros(pair_indices.size, dtype=float)
            for local_idx, pair_idx in enumerate(pair_indices):
                if screen_tol > 0.0 and pivot_bound * float(pair_block["bounds"][local_idx]) < screen_tol:
                    continue
                values[local_idx] = float(
                    _contracted_eri_from_signatures(
                        pair_block["sig_p"][local_idx],
                        pair_block["sig_q"][local_idx],
                        pivot_sig_i,
                        pivot_sig_j,
                    )
                )
            col[pair_indices] = values

        if rank > 0:
            col -= chol[:, :rank] @ chol[pivot, :rank].conj()

        delta = float(np.real(col[pivot]))
        if delta <= tol:
            diag[pivot] = 0.0
            continue

        chol[:, rank] = col / math.sqrt(delta)
        diag -= np.real(chol[:, rank] * chol[:, rank].conj())
        diag = np.maximum(diag, 0.0)
        rank += 1

    return _unpack_packed_pair_factors(chol[:, :rank].T, pairs, nao)

ALIAS = {
    '631g'       : '6-31g.1.gbs',
    'sto3g'      : "sto-3g.1.gbs",
    'sto6g'      : 'sto-6g.1.gbs',
    '631g**'     : "6-31g_st__st_.0.gbs",
    '6311g**'    : "6-311g_st__st_.0.gbs",
    '6311g'      : "6-311g.0.gbs",
    '631g++'     : "/6-31g++.gbs",
    '631g*'      : "6-31g_st_.0.gbs",
    'ccpvdz'     : 'cc-pvdz.0.gbs'    ,
    'ccpvtz'     : 'cc-pvtz.dat'    ,
    'ccpvqz'     : 'cc-pvqz.dat'    ,
    'ccpv5z'     : 'cc-pv5z.dat'    ,
    'ccpvdpdz'   : 'cc-pvdpdz.dat'  ,
    'augccpvdz'  : 'aug-cc-pvdz.dat',
    'augccpvtz'  : 'aug-cc-pvtz.dat',
    'augccpvqz'  : 'aug-cc-pvqz.dat',
    'augccpv5z'  : 'aug-cc-pv5z.dat',
    'augccpvdpdz': 'aug-cc-pvdpdz.dat',
    'ccpvdzdk'   : 'cc-pvdz-dk.dat' ,
    'ccpvtzdk'   : 'cc-pvtz-dk.dat' ,
    'ccpvqzdk'   : 'cc-pvqz-dk.dat' ,
    'ccpv5zdk'   : 'cc-pv5z-dk.dat' ,
    'ccpvdzdkh'  : 'cc-pvdz-dk.dat' ,
    'ccpvtzdkh'  : 'cc-pvtz-dk.dat' ,
    'ccpvqzdkh'  : 'cc-pvqz-dk.dat' ,
    'ccpv5zdkh'  : 'cc-pv5z-dk.dat' ,
}


def _normalize_basis_lookup_name(name):
    return str(name).replace('-', '').replace(' ', '').lower()


def _basis_path(basis_name):
    if not isinstance(basis_name, str):
        raise NotImplementedError('Customized basis not supported yet.')

    basis_dir = os.path.abspath(f'{pyqed.__file__}/../qchem/basis_set/')
    key = _normalize_basis_lookup_name(basis_name)
    if key in ALIAS:
        return os.path.join(basis_dir, ALIAS[key].lstrip('/'))

    candidates = []
    for entry in os.listdir(basis_dir):
        stem = re.sub(r'\.(?:[01]\.)?(?:gbs|dat)$', '', entry, flags=re.IGNORECASE)
        if _normalize_basis_lookup_name(stem) == key:
            candidates.append(entry)

    if not candidates:
        raise ValueError(
            f"Unsupported basis '{basis_name}' for the builtin integral driver."
        )

    def _basis_sort_key(name):
        lower = name.lower()
        if lower.endswith('.0.gbs'):
            return (0, lower)
        if lower.endswith('.1.gbs'):
            return (1, lower)
        return (2, lower)

    return os.path.join(basis_dir, sorted(candidates, key=_basis_sort_key)[0])


def _reset_builtin_integral_caches():
    """
    Clear recurrence caches before building a fresh builtin AO integral set.

    The memoized Hermite/Boys helpers remove a lot of duplicated scalar work
    inside a single build. Clearing them between builds keeps memory bounded.
    """
    E.cache_clear()
    boys.cache_clear()
    R.cache_clear()
    _nuclear_attraction_cached.cache_clear()
    _electron_repulsion_cached.cache_clear()
    _contracted_eri_from_signatures_cached.cache_clear()


def _builtin_worker_count(mol, nao):
    if not bool(getattr(mol, "builtin_parallel", getattr(mol, "native_parallel", False))):
        return 1
    min_nao = int(
        getattr(mol, "builtin_parallel_min_nao", getattr(mol, "native_parallel_min_nao", 12))
    )
    if nao < min_nao:
        return 1
    requested = getattr(mol, "builtin_eri_workers", getattr(mol, "native_eri_workers", None))
    if requested is None:
        requested = min(4, max(1, os.cpu_count() or 1))
    return max(1, int(requested))


def build_builtin(mol):
    """
    Build AO integrals with pyqed's builtin Gaussian integral engine.
    """
    _reset_builtin_integral_caches()
    atoms = mol.atom_symbols()
    atcoords = np.asarray(mol.atom_coords(), dtype=float)
    atnums = np.asarray(mol.atom_charges(), dtype=float)

    basis_dict = parse_gbs(_basis_path(mol.basis))
    basis_cart = make_contractions(basis_dict, atoms, atcoords, coord_types='c')
    nao_cart = len(basis_cart)
    signatures = tuple(_basis_signature(fn) for fn in basis_cart)
    one_electron_result = _compute_one_electron_shellblocked_cython(signatures, atcoords, atnums)
    if one_electron_result is not None:
        overlap_mat, kinetic_mat, vnuc_mat = one_electron_result
    else:
        overlap_mat, kinetic_mat, vnuc_mat = _compute_one_electron_shellblocked(
            basis_cart,
            atcoords,
            atnums,
        )

    screen_tol = float(
        getattr(mol, "builtin_eri_screen_tol", getattr(mol, "native_eri_screen_tol", 0.0)) or 0.0
    )
    pair_bounds = _compute_pair_bounds(signatures)
    eri_representation = getattr(
        mol,
        "builtin_eri_representation",
        getattr(mol, "native_eri_representation", "dense"),
    )
    if eri_representation not in {"dense", "dense+factors", "factors"}:
        raise ValueError(
            "builtin_eri_representation must be 'dense', 'dense+factors', or 'factors'."
        )

    coord_type = str(
        getattr(mol, "builtin_coord_type", getattr(mol, "native_coord_type", "spherical"))
    ).lower()
    if coord_type in ("p", "spherical"):
        coord_type = "spherical"
    elif coord_type in ("c", "cartesian"):
        coord_type = "cartesian"
    else:
        raise ValueError("builtin_coord_type/native_coord_type must be 'spherical' or 'cartesian'.")

    workers = _builtin_worker_count(mol, nao_cart)
    computed = 0
    skipped = 0
    eri = None
    factors = None
    dense_builder = None
    factor_builder = None

    if eri_representation in {"dense", "dense+factors"}:
        if workers > 1:
            eri, computed, skipped = _compute_dense_eri_parallel(
                signatures, pair_bounds, screen_tol, workers
            )
            dense_builder = "python-parallel"
        else:
            eri, computed, skipped = _compute_dense_eri_serial(
                signatures, pair_bounds, screen_tol
            )
            dense_builder = (
                "cython-kernel"
                if _basis_cy is not None
                else (
                "c-serial"
                if _load_basis_accel()
                else (
                "numba-serial"
                if (_NUMBA_AVAILABLE and _NUMBA_DENSE_ERI_ENABLED)
                else "python-serial"
                )
                )
            )

        if eri_representation == "dense+factors" or bool(
            getattr(mol, "builtin_build_factors", getattr(mol, "native_build_factors", False))
        ):
            from pyqed.qchem.hf.rhf import pivoted_cholesky_eri

            factors = pivoted_cholesky_eri(
                eri,
                tol=float(
                    getattr(mol, "builtin_low_rank_tol", getattr(mol, "native_low_rank_tol", 1e-8))
                ),
                max_rank=getattr(
                    mol,
                    "builtin_low_rank_max_rank",
                    getattr(mol, "native_low_rank_max_rank", None),
                ),
            )
    else:
        shell_blocks = _cart_shell_blocks(basis_cart)
        shell_starts = np.asarray([start for start, _stop, _l in shell_blocks], dtype=np.int64)
        shell_stops = np.asarray([stop for _start, stop, _l in shell_blocks], dtype=np.int64)
        factors = _pivoted_cholesky_from_integral_oracle_cython_blocked(
            signatures,
            pair_bounds,
            shell_starts=shell_starts,
            shell_stops=shell_stops,
            tol=float(
                getattr(mol, "builtin_low_rank_tol", getattr(mol, "native_low_rank_tol", 1e-8))
            ),
            max_rank=getattr(
                mol,
                "builtin_low_rank_max_rank",
                getattr(mol, "native_low_rank_max_rank", None),
            ),
            screen_tol=screen_tol,
        )
        if factors is not None:
            factor_builder = "cython-kernel-blocked"
        else:
            factors = _pivoted_cholesky_from_integral_oracle(
                signatures,
                pair_bounds,
                tol=float(
                    getattr(mol, "builtin_low_rank_tol", getattr(mol, "native_low_rank_tol", 1e-8))
                ),
                max_rank=getattr(
                    mol,
                    "builtin_low_rank_max_rank",
                    getattr(mol, "native_low_rank_max_rank", None),
                ),
                screen_tol=screen_tol,
            )
            factor_builder = "cython-kernel" if _basis_cy is not None else "python-oracle"

    transform = None
    basis_out = basis_cart
    if coord_type == "spherical":
        blocks = _cart_shell_blocks(basis_cart)
        nsph = sum(2 * l + 1 for _, _, l in blocks)
        transform = np.zeros((nao_cart, nsph), dtype=float)
        col = 0
        for start, stop, l in blocks:
            blk = _cart2sph_unit_block(l)
            ncols = blk.shape[1]
            transform[start:stop, col:col + ncols] = blk
            col += ncols

        overlap_mat = np.einsum('pi,pq,qj->ij', transform, overlap_mat, transform, optimize=True)
        hcore_mat = np.einsum('pi,pq,qj->ij', transform, kinetic_mat + vnuc_mat, transform, optimize=True)
        if eri is not None:
            eri = np.einsum('pa,qb,rc,sd,pqrs->abcd', transform, transform, transform, transform, eri, optimize=True)
        if factors is not None:
            factors = np.einsum('pa,rpq,qb->rab', transform, factors, transform, optimize=True)
        basis_out = basis_cart
        mol.cart = False
    else:
        hcore_mat = kinetic_mat + vnuc_mat
        mol.cart = True

    mol.nao = overlap_mat.shape[0]
    mol.overlap = overlap_mat
    mol.hcore = hcore_mat
    mol.eri = eri
    mol.eri_factors = factors
    mol._bas = basis_out
    mol._bas_cart = basis_cart if transform is not None else None
    mol._ao_cart2sph = transform
    if basis_out:
        for fn in basis_out:
            setattr(fn, "coord_type", coord_type)
    mol.nbas = mol.nao
    mol._builtin_build_info = {
        "coord_type": coord_type,
        "representation": eri_representation,
        "workers": workers,
        "screen_tol": screen_tol,
        "quartets_computed": int(computed),
        "quartets_screened": int(skipped),
        "factor_rank": None if factors is None else int(factors.shape[0]),
        "dense_builder": dense_builder,
        "factor_builder": factor_builder,
    }
    mol._native_build_info = mol._builtin_build_info
    return


def build_native(mol):
    """
    Backward-compatible alias for the builtin AO integral builder.
    """
    return build_builtin(mol)


def build(mol, pyscf=False):
    """
    Build AO integrals in the gbasis backend.
    """
    atoms = mol.atom_symbols()
    atcoords = mol.atom_coords()
    atnums = mol.atom_charges()

    try:
        from gbasis.integrals.electron_repulsion import electron_repulsion_integral
        from gbasis.integrals.kinetic_energy import kinetic_energy_integral
        from gbasis.integrals.nuclear_electron_attraction import \
            nuclear_electron_attraction_integral
        from gbasis.integrals.overlap import overlap_integral
        from gbasis.parsers import make_contractions as gbasis_make_contractions
        from gbasis.parsers import parse_gbs as gbasis_parse_gbs
    except ImportError as exc:
        raise ImportError(
            "gbasis is required for driver='gbasis'. Use driver='builtin' to avoid gbasis."
        ) from exc

    if not pyscf:
        basis_dict = gbasis_parse_gbs(_basis_path(mol.basis))
        basis = gbasis_make_contractions(basis_dict, atoms, atcoords, coord_types="p")
    else:
        from gbasis.wrappers import from_pyscf
        basis = from_pyscf(mol.topyscf())

    mol.overlap = overlap_integral(basis)
    mol.nao = mol.overlap.shape[0]
    k_int1e = kinetic_energy_integral(basis)
    nuc_int1e = nuclear_electron_attraction_integral(basis, atcoords, atnums)
    mol.hcore = k_int1e + nuc_int1e
    mol.eri = electron_repulsion_integral(basis, notation='chemist')
    mol._bas = basis
    mol.nbas = mol.nao
    return


@lru_cache(maxsize=None)
def parse_gbs(gbs_basis_file):
    """Parse Gaussian94 basis set file.

    Parameters
    ----------
    gbs_basis_file : str
        Path to the Gaussian94 basis set file.

    Returns
    -------
    basis_dict : dict of str to list of 3-tuple of (int, np.ndarray, np.ndarray)
        Dictionary of the element to the list of angular momentum, exponents, and contraction
        coefficients associated with each contraction at the given atom.

    Notes
    -----
    Angular momentum symbol is hard-coded into this function. This means that if the selected basis
    set has an angular momentum greater than "k", an error will be raised.

    Since Gaussian94 basis format does not explicitly state which contractions are generalized, we
    infer that subsequent contractions belong to the same generalized shell if they have the same
    exponents and angular momentum. If two contractions are not one after another or if they are
    associated with more than one angular momentum, they are treated to be segmented contractions.

    """
    # pylint: disable=R0914
    with open(gbs_basis_file) as basis_fh:
        gbs_basis = basis_fh.read()
    # splits file into 'element', 'basis stuff', 'element',' basis stuff'
    # e.g., ['H','stuff with exponents & coefficients\n', 'C', 'stuff with etc\n']
    data = re.split(r"\n\s*(\w[\w]?)\s+\w+\s*\n", gbs_basis)
    dict_angmom = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5, "i": 6, "k": 7}
    # remove first part
    if "\n" in data[0]:  # pragma: no branch
        data = data[1:]
    # atoms: stride of 2 get the ['H','C', etc]. basis: take strides of 2 to skip elements
    atoms = data[::2]
    basis = data[1::2]
    # trim out headers at the end
    output = {}
    for atom, shells in zip(atoms, basis):
        output.setdefault(atom, [])

        shells = re.split(r"\n?\s*(\w+)\s+\w+\s+\w+\.\w+\s*\n", shells)
        # remove the ends
        atom_basis = shells[1:]
        # get angular momentums
        angmom_shells = atom_basis[::2]
        # get exponents and coefficients
        exps_coeffs_shells = atom_basis[1::2]

        for angmom_seg, exp_coeffs in zip(angmom_shells, exps_coeffs_shells):
            angmom_seg = [dict_angmom[i.lower()] for i in angmom_seg]
            exps = []
            coeffs_seg = []
            exp_coeffs = exp_coeffs.split("\n")
            for line in exp_coeffs:
                test = re.search(
                    r"^\s*([0-9\.DE\+\-]+)\s+((?:(?:[0-9\.DE\+\-]+)\s+)*(?:[0-9\.DE\+\-]+))\s*$",
                    line,
                )
                try:
                    exp, coeff_seg = test.groups()
                    coeff_seg = re.split(r"\s+", coeff_seg)
                except AttributeError:
                    continue
                # clean up
                exp = float(exp.lower().replace("d", "e"))
                coeff_seg = [float(i.lower().replace("d", "e")) for i in coeff_seg if i is not None]
                exps.append(exp)
                coeffs_seg.append(coeff_seg)
            exps = np.array(exps)
            coeffs_seg = np.array(coeffs_seg)
            # if len(angmom_seg) == 1:
            #     coeffs_seg = coeffs_seg[:, None]
            for i, angmom in enumerate(angmom_seg):
                # ensure previous and current exps are same length before using np.allclose()
                if output[atom] and len(output[atom][-1][1]) == len(exps):
                    # check if current exp's should be added to previous generalized contraction
                    hstack = np.allclose(output[atom][-1][1], exps)
                else:
                    hstack = False
                if output[atom] and output[atom][-1][0] == angmom and hstack:
                    output[atom][-1] = (
                        angmom,
                        exps,
                        np.hstack([output[atom][-1][2], coeffs_seg[:, i : i + 1]]),
                    )
                else:
                    output[atom].append((angmom, exps, coeffs_seg[:, i : i + 1]))

    return output


def make_contractions(basis_dict, atoms, coords, coord_types):
    """Return the contractions that correspond to the given atoms for the given basis.

    Parameters
    ----------
    basis_dict : dict of str to list of 3-tuple of (int, np.ndarray, np.ndarray)
        Output of the parsers from gbasis.parsers.
    atoms : N-list/tuple of str
        Atoms at which the contractions are centered.
    coords : np.ndarray(N, 3)
        Coordinates of each atom.
    coord_types : {"cartesian"/"c", list/tuple of "cartesian"/"c" or "spherical"/"p", "spherical"/"p"}
        Types of the coordinate system for the contractions.
        If "cartesian" or "c", then all of the contractions are treated as Cartesian contractions.
        If "spherical" or "p", then all of the contractions are treated as spherical contractions.
        If list/tuple, then each entry must be a "cartesian" (or "c") or "spherical" (or "p") to specify the
        coordinate type of each `GeneralizedContractionShell` instance.
        Default value is "spherical".

    Returns
    -------
    basis : tuple of GeneralizedContractionShell
        Contractions for each atom.
        Contractions are ordered in the same order as in the values of `basis_dict`.

    Raises
    ------
    TypeError
        If `atoms` is not a list or tuple of strings.
        If `coords` is not a two-dimensional `numpy` array with 3 columns.
        If `tol` is not a float.
        If `ovr` is not boolean
    ValueError
        If the length of atoms is not equal to the number of rows of `coords`.

    """
    if not (isinstance(atoms, (list, tuple)) and all(isinstance(i, str) for i in atoms)):
        raise TypeError("Atoms must be provided as a list or tuple.")
    if not (isinstance(coords, np.ndarray) and coords.ndim == 2 and coords.shape[1] == 3):
        raise TypeError(
            "Coordinates must be provided as a two-dimensional `numpy` array with three columns."
        )

    if len(atoms) != coords.shape[0]:
        raise ValueError("Number of atoms must be equal to the number of rows in the coordinates.")

    basis = []
    # expected number of coordinates
    num_coord_types = sum([len(basis_dict[i]) for i in atoms])

    # check and assign coord_types
    if isinstance(coord_types, str):
        if coord_types not in ["c", "cartesian", "p", "spherical"]:
            raise ValueError(
                f"If coord_types is a string, it must be either 'spherical'/'p' or 'cartesian'/'c'."
                f"got {coord_types}"
            )
        coord_types = [coord_types] * num_coord_types

    if len(coord_types) != num_coord_types:
        raise ValueError(
            f"If coord_types is a list, it must be the same length as the total number of contractions."
            f"got {len(coord_types)}"
        )

    # make shells

    for icenter, (atom, coord) in enumerate(zip(atoms, coords)):
        for angmom, exps, coeffs in basis_dict[atom]:
            coeffs = np.asarray(coeffs, dtype=float)
            if coeffs.ndim == 1:
                coeffs = coeffs[:, None]
            for shell in _shell(angmom):
                for icontr in range(coeffs.shape[1]):
                    basis.append(
                        ContractedGaussian(
                            origin=coord,
                            shell=shell,
                            exps=np.asarray(exps, dtype=float),
                            coefs=coeffs[:, icontr].copy(),
                        )
                    )
    return tuple(basis)

def _shell(l):
    """
    Enumerate Cartesian angular-momentum components for shell ``l``.

    The ordering matches the usual Cartesian convention used by libcint/PySCF,
    e.g. for ``d``:
    ``xx, xy, xz, yy, yz, zz`` and for ``f``:
    ``xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz``.
    """
    if l < 0:
        raise ValueError(f"Angular momentum must be non-negative, got l={l}.")

    shells = []
    for lx in range(l, -1, -1):
        remaining = l - lx
        for ly in range(remaining, -1, -1):
            lz = remaining - ly
            shells.append((lx, ly, lz))
    return shells


def _cart_shell_signature(fn):
    return (
        tuple(np.round(np.asarray(fn.origin, dtype=float), 12).tolist()),
        int(sum(fn.shell)),
        tuple(np.round(np.asarray(fn.exps, dtype=float), 12).tolist()),
        tuple(np.round(np.asarray(fn.coefs, dtype=float), 12).tolist()),
    )


def _cart_shell_blocks(basis_cart):
    blocks = []
    start = 0
    nao_cart = len(basis_cart)
    while start < nao_cart:
        sig = _cart_shell_signature(basis_cart[start])
        l = sig[1]
        ncart = (l + 1) * (l + 2) // 2
        stop = start + ncart
        if stop > nao_cart:
            raise ValueError("Invalid Cartesian shell partition in builtin basis.")
        if any(_cart_shell_signature(basis_cart[k]) != sig for k in range(start, stop)):
            raise ValueError("Builtin Cartesian shell ordering is not contiguous.")
        blocks.append((start, stop, l))
        start = stop
    return blocks


def _cart2sph_unit_block(l):
    """
    Cartesian -> real-spherical transform for the builtin unit-normalized
    Cartesian AO convention.

    Rows follow `_shell(l)` ordering. Columns match the real-spherical AO
    ordering used by PySCF/libcint:
    - d:  xy, yz, z^2, xz, x2-y2
    - f: -3, -2, -1, 0, +1, +2, +3
    """
    if l == 0:
        return np.array([[1.0]], dtype=float)
    if l == 1:
        return np.eye(3, dtype=float)
    if l == 2:
        return np.array(
            [
                [0.0, 0.0, -0.5, 0.0, 0.86602540378443864676],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, -0.5, 0.0, -0.86602540378443864676],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
            ],
            dtype=float,
        )
    if l == 3:
        return np.array(
            [
                [0.0, 0.0, 0.0, 0.0, -0.61237243569579452455, 0.0, 0.79056941504209483299],
                [1.0606601717798212866, 0.0, -0.27386127875258305686, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -0.67082039324993690892, 0.0, 0.86602540378443864676, 0.0],
                [0.0, 0.0, 0.0, 0.0, -0.27386127875258305686, 0.0, -1.0606601717798212866],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0954451150103322269, 0.0, 0.0],
                [-0.79056941504209483299, 0.0, -0.61237243569579452455, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -0.67082039324993690892, 0.0, -0.86602540378443864676, 0.0],
                [0.0, 0.0, 1.0954451150103322269, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ],
            dtype=float,
        )
    if l == 4:
        return np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.39467353541831303197, 0.0, -0.58834840541455207145, 0.0, 0.79056941504209483299],
                [1.0606601717798212866, 0.0, -0.40824829046386301637, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, -0.89442719099991587856, 0.0, 0.79056941504209483299, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.18257418583505537115, 0.0, 0.0, 0.0, -1.0606601717798212866],
                [0.0, 1.1180339887498948482, 0.0, -0.40824829046386301637, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, -0.73029674334022148461, 0.0, 0.81649658092772603273, 0.0, 0.0],
                [-1.0606601717798212866, 0.0, -0.40824829046386301637, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, -0.40824829046386301637, 0.0, -1.1180339887498948482, 0.0],
                [0.0, 0.0, 1.154700538379251529, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.1180339887498948482, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.39467353541831303197, 0.0, 0.58834840541455207145, 0.0, 0.79056941504209483299],
                [0.0, -0.79056941504209483299, 0.0, -0.89442719099991587856, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, -0.73029674334022148461, 0.0, -0.81649658092772603273, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.1180339887498948482, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=float,
        )
    raise NotImplementedError(
        f"Builtin spherical AO transform is implemented only for l <= 4, got l={l}."
    )



if __name__=='__main__':

    # kin_e = np.trace(dm.dot(k_int1e))
    # print("Kinetic energy (Hartree):", kin_e)

    import time

    # Define atomic symbols and coordinates (i.e., basis function centers)
    atoms = ["F", "F"]
    atcoords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    basis_dir = os.path.abspath(f'{pyqed.__file__}/../qchem/basis_set/')

    basis_dict = parse_gbs(basis_dir+'/6-31g.1.gbs')
    basis = make_contractions(basis_dict, atoms, atcoords, 'c')

    # print([g.exps for g in basis])

    print(len(basis))

    # # To obtain the total number of AOs we compute the cartesian components for each angular momentum
    # total_ao = 0
    # print(f"Number of generalized shells: {len(basis)}") # Output 6
    # for shell in basis:
    #     total_ao += shell.angmom_components_cart.shape[0]

    # print("Total number of AOs: ", total_ao) # output 10


    # myOrigin = [0.0, 0.0, 0.0]
    # myShell = (0,0,0) # p‐orbitals would be (1,0,0) or (0,1,0) or (0,0,1), etc.
    # myExps = [3.42525091, 0.62391373, 0.16885540]
    # myCoefs = [0.15432897, 0.53532814, 0.44463454]
    # a = ContractedGaussian(origin=myOrigin,shell=myShell,exps=myExps,coefs=myCoefs)


    # H2 = [0.0, 0.0, 1.0]
    # myShell = (0,0,0) # p‐orbitals would be (1,0,0) or (0,1,0) or (0,0,1), etc.
    # myExps = [3.42525091, 0.62391373, 0.16885540]
    # myCoefs = [0.15432897, 0.53532814, 0.44463454]
    # b = ContractedGaussian(origin=H2,shell=myShell,exps=myExps,coefs=myCoefs)

    # basis = [a, b]

    def ao_ints(basis, coords):
        nao = len(basis)
        natom, _ = coords.shape

        s = np.eye(nao)
        for i in range(nao):
            for j in range(i):
                s[i,j] = S(basis[i], basis[j])
                s[j,i] = s[i,j]

        t = np.zeros((nao, nao))
        for i in range(nao):
            for j in range(i+1):
                t[i,j] = T(basis[i], basis[j])
                if i != j: t[j,i] = t[i,j]

        v = np.zeros((nao,nao))
        for i in range(nao):
            for j in range(i+1):
                for C in range(natom):
                    v[i,j] -= point_charge(basis[i], basis[j], coords[C])
                if i != j: v[j,i] = v[i,j]

        eri = np.zeros((nao, nao, nao, nao))
        for p in range(nao):
            for q in range(p):
                for r in range(nao):
                    for s in range(r):
                        eri[p,q,r,s] = ERI(basis[p], basis[q], basis[r], basis[s])

        return s, t, v, eri

    # print(basis[0].exps, basis[0].coefs)
    print(atcoords[1])
    print(point_charge(basis[0], basis[0], atcoords[1]))

    start_time = time.time()
    ao_ints(basis, atcoords)
    end_time = time.time()

    print(end_time-start_time)

    # s,t, v, eri = ao_ints(basis, atcoords)
    # print(t)
    # point_charge(a, a, myOrigin))
    # print(v)

    # from pyqed.qchem import Molecule
    # mol = Molecule(atom = [
    # ['H' , (0. , 0. , 0)],
    # ['H' , (0. , 0. , 1.)], ], basis='631g')
    # mol.build()

    # print(mol._bas[0].)

    # mol.build()
    # print(mol.eri)
