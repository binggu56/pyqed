'''
disspaton equation of motion
Author: Zi-Hao Chen
Date: 2023/07/12
Email: czh5@mail.ustc.edu.cn
Only for bosonic bath
'''


from numba import njit
from importlib import import_module
import numpy as np
import sympy as sp
import math
from tqdm import tqdm
from scipy.sparse import coo_matrix
from scipy import linalg as la
from scipy.optimize import least_squares
import os
from pyqed.integrate import (
    krylov,
    normalize_method,
)
# from cvxopt import solvers, matrix

_heom_cpp = None
_heom_cpp_import_attempted = False


def _get_heom_cpp():
    """Load the optional native HEOM backend on first use."""
    global _heom_cpp, _heom_cpp_import_attempted
    if not _heom_cpp_import_attempted:
        _heom_cpp_import_attempted = True
        try:
            _heom_cpp = import_module("pyqed.heom._heom_cpp")
        except Exception:
            _heom_cpp = None
    return _heom_cpp


def _distance_array(distances):
    arr = np.asarray(distances, dtype=float)
    scalar = arr.ndim == 0
    arr = np.atleast_1d(arr)
    if np.any(arr < -1.0e-14):
        raise ValueError("distances must be non-negative.")
    return np.maximum(arr, 0.0), scalar


@njit
def spectrum_exp(w, res, expn, etal, sigma=-1):
    for i in range(len(etal)):
        res += etal[i] / (expn[i] + sigma * 1.j * w)


# def numpy_to_cvxopt_matrix(A):
#     if A is None:
#         return A
#     if isinstance(A, np.ndarray):
#         if A.ndim == 1:
#             return matrix(A, (A.shape[0], 1), 'd')
#         return matrix(A, A.shape, 'd')
#     return A


def sort_symmetry(etal, expn, if_sqrt=True):
    expn_imag_sort = np.argsort(np.abs(np.imag(expn)))[::-1]
    expn_imag = np.sort(np.abs(np.imag(expn)))[::-1]
    expn = expn[expn_imag_sort]
    etal = etal[expn_imag_sort]
    etar = etal[expn_imag_sort]
    expn_val_cc = np.where(expn[expn_imag > 1e-10])[0]
    etaa = np.zeros(len(etal), dtype=float)
    for ii in range(0, len(expn_val_cc), 2):
        even_i = ii
        odd_i = ii + 1
        etar[even_i] = np.conj(etal[odd_i])
        etar[odd_i] = np.conj(etal[even_i])
        etaa[even_i] = np.abs(etal[even_i])
        etaa[odd_i] = np.abs(etal[odd_i])
    for ii in range(len(expn_val_cc), len(expn)):
        even_i = ii
        etar[even_i] = np.conj(etal[even_i])
        etaa[even_i] = np.abs(etal[even_i])
    if (if_sqrt):
        etaa = np.sqrt(etaa)
    return etal, etar, etaa, expn


@njit
def fit_t(t, res, expn, etal):
    for i in range(len(etal)):
        res += etal[i] * np.exp(-expn[i] * t)
    return res


def function_bose(x, pole, resi):
    '''
    Distribution function of bosonic bath
    '''
    return 1 / x + 0.5 + sum(2.0 * resi[i] * x / (x**2 + pole[i]**2)
                             for i in range(len(pole)))


def tseig(D, E):
    '''
    Tridiagonal matrix eigenvalue solver
    '''
    mat = np.diag(E, -1) + np.diag(D, 0) + np.diag(E, 1)
    return -np.sort(-np.linalg.eigvalsh(mat))


def matsubara_approximation_distribution(N, BoseFermi=1):
    '''
    Matsubara approximation of bosonic/fermionic distribution function
    Input:
        N: number of trem of Matsubara frequencies
        BoseFermi: 1 for Bose, 2 for Fermi
    Output:
        pole: poles of the distribution function
        resi: residues of the distribution function
    '''
    if BoseFermi == 1:
        pole = np.array([2 * (i + 1) * np.pi for i in range(N)])
        resi = np.ones(N, dtype=float)
        return pole, resi
    elif BoseFermi == 2:
        pole = np.array([(2 * i + 1) * np.pi for i in range(N)])
        resi = np.ones(N, dtype=float)
        return pole, resi


def pade_approximation_distribution(N, BoseFermi=1, pade=1):
    '''
    pade approximation of bosonic/fermionic distribution function
    Input:
        N: number of trem of Matsubara frequencies
        BoseFermi: 1 for Bose, 2 for Fermi
        pade: 0 for Matsubara approximation, 1 for Pade
    Output:
        pole: poles of the distribution function
        resi: residues of the distribution function
    '''
    if N < 0 or BoseFermi < 1 or BoseFermi > 2 or pade < 0 or pade > 3:
        raise ValueError("N or BoseFermi or pade has wrong value!")

    if pade == 0:
        return matsubara_approximation_distribution(N, BoseFermi)
    elif pade == 1 or pade == 2:
        pole, resi = [], []
        if N > 0:
            M = 2 * N + pade // 2
            temp = 3.0 if BoseFermi == 1 else 1.0
            diag = np.zeros(M, dtype=float)
            doff = np.array([
                1.0 / math.sqrt((temp + 2.0 * i) * (temp + 2.0 * (i + 1)))
                for i in range(M - 1)
            ])
            pole = 2.0 / tseig(diag, doff)[:N]
            pol2 = np.array([x * x for x in pole])
            M -= 1
            temp = 5.0 if BoseFermi == 1 else 3.0
            diag = np.zeros(M, dtype=float)
            doff = np.array([
                1.0 / math.sqrt((temp + 2.0 * i) * (temp + 2.0 * (i + 1)))
                for i in range(M - 1)
            ])
            M //= 2
            eig2 = np.power(2.0 / tseig(diag, doff)[:M], 2)
            scaling = 0.0
            if BoseFermi == 1:
                scaling = N*(2.0*N+3.0) if pade == 1 else 1.0 / \
                    (4.0*(N+1.0)*(2.0*N+3.0))
            elif BoseFermi == 2:
                scaling = N*(2.0*N+1.0) if pade == 1 else 1.0 / \
                    (4.0*(N+1.0)*(2.0*N+1.0))
            resi = np.zeros(N, dtype=float)
            for j in range(N):
                if pade == 2:
                    temp = 0.5 * scaling * (eig2[j] - pol2[j])
                elif pade == 1:
                    if j == N - 1:
                        temp = 0.5 * scaling
                    else:
                        temp = 0.5*scaling * \
                            (eig2[j]-pol2[j])/(pol2[N-1]-pol2[j])
                for k in range(M):
                    temp *= (eig2[k]-pol2[j]) / \
                        (pol2[k]-pol2[j]) if k != j else 1.0
                resi[j] = temp
        return pole, resi
    elif pade == 3:
        Np1 = N + 1
        temp = 3.0 if BoseFermi == 1 else 1.0
        d = np.empty(2 * Np1, dtype=float)
        d[0] = 0.25 / temp
        d[-1] = -4.0 * (N + 1.0) * (N + 1.0) * (temp + 2 * N) * (
            temp + 2 * N) * (temp + 4 * N + 2.0)
        for i in range(1, Np1):
            d[2*i-1] = -4.0*i*i*(temp+2.0*i-2.0) * \
                (temp+2.0*i-2.0)*(temp+4.0*i-2.0)
            d[2 * i] = -0.25 * (temp + 4.0 * i) / i / (i + 1) / (
                temp + 2.0 * i - 2.0) / (temp + 2.0 * i)
        sumd2 = np.empty(Np1, dtype=float)
        sumd2[0] = d[1]
        for i in range(1, Np1):
            sumd2[i] = sumd2[i - 1] + d[2 * i + 1]
        M = 2 * N + 1
        diag = np.zeros(M, dtype=float)
        doff = np.array(
            [1.0 / math.sqrt(d[i + 1] * d[i + 2]) for i in range(M - 1)])
        pole = 2.0 / tseig(diag, doff)[:N]
        resi = np.zeros(N, dtype=float)
        for j in range(N):
            scaling = pole[j] * pole[j]
            r0, t1 = 0.0, 0.25 / d[1]
            eta0, eta1, eta2 = 0.0, 0.5, 0.0
            for i in range(Np1):
                r1 = t1 if (i == j
                            or i == N) else t1 / (pole[i] * pole[i] - scaling)
                r2 = 2.0*math.sqrt(abs(r1)) if r1 > 0 else - \
                    2.0*math.sqrt(abs(r1))
                r1 = 2.0 * math.sqrt(abs(r1))
                eta2 = d[2 * i] * r1 * eta1 - 0.25 * r1 * r0 * scaling * eta0
                eta0 = eta1
                eta1 = eta2
                eta2 = d[2 * i +
                         1] * r2 * eta1 - 0.25 * r2 * r1 * scaling * eta0
                eta0 = eta1
                eta1 = eta2
                r0 = r2
                if i != N:
                    t1 = sumd2[i] / sumd2[i + 1]
            resi[j] = eta2
        return pole, resi


def return_qmds(qmd1a, qmd1c, mode, nsys, etaa, etal, etar):
    '''
    return the qmds for the given mode
    '''
    qmdta_l = np.zeros((len(mode), nsys, nsys), dtype=complex)
    qmdta_r = np.zeros((len(mode), nsys, nsys), dtype=complex)
    qmdtc_l = np.zeros((len(mode), nsys, nsys), dtype=complex)
    qmdtc_r = np.zeros((len(mode), nsys, nsys), dtype=complex)
    for i in range(len(mode)):
        i_mod = mode[i]
        qmdta_l[i, :, :] = qmd1a[i_mod, :, :] * np.sqrt(etaa[i])
        qmdta_r[i, :, :] = qmd1a[i_mod, :, :] * np.sqrt(etaa[i])
        qmdtc_l[i, :, :] = qmd1c[i_mod, :, :] * etal[i] / np.sqrt(etaa[i])
        qmdtc_r[i, :, :] = qmd1c[i_mod, :, :] * etar[i] / np.sqrt(etaa[i])
    return np.array([qmdtc_l, qmdtc_r, qmdta_l, qmdta_r])


def decompose_spectrum_pade(spe, w_sp, beta, npsd, pade=1, bose_fermi=1):
    '''
    decompose the spectrum into the pade approximation
    '''
    if (sp.cancel(spe).as_real_imag()[1] == 0):
        imag_part = sp.cancel(spe).as_real_imag()[0]
    else:
        imag_part = sp.cancel(spe).as_real_imag()[1]
    numer, denom = sp.cancel(sp.factor(imag_part)).as_numer_denom()

    poles = sp.nroots(denom)
    float(sp.re(poles[0]))

    expn, etal, etar, etaa = [], [], [], []
    poles_allplane = np.array([])
    for i in poles:
        i = complex(i)
        if i.imag < 0:
            expn.append(i * 1.J)
        poles_allplane = np.append(poles_allplane, i)

    expn = np.array(expn)

    expn_imag_sort = np.argsort(np.abs(np.imag(expn)))[::-1]
    expn_imag = np.sort(np.abs(np.imag(expn)))[::-1]

    expn_val_cc = expn[expn_imag_sort[expn_imag != 0]]
    expn_val_n_cc = expn[expn_imag_sort[expn_imag == 0]]

    expn = list(expn[expn_imag_sort])
    pole, resi = pade_approximation_distribution(npsd, bose_fermi, pade)
    temp = 1 / beta

    for ii in range(0, len(expn_val_cc), 2):
        etal.append(
            complex(
                sp.N((-2.j * numer /
                      np.multiply.reduce(w_sp - poles_allplane[np.abs(
                          poles_allplane + 1.J * expn_val_cc[ii]) > 1e-14])
                      ).subs({w_sp: -1.j * expn_val_cc[ii]}) *
                     function_bose(-1.J * expn_val_cc[ii] / temp, pole, resi))))

        etal.append(
            complex(
                sp.N((-2.j * numer /
                      np.multiply.reduce(w_sp - poles_allplane[np.abs(
                          poles_allplane + 1.J * expn_val_cc[ii + 1]) > 1e-14])
                      ).subs({w_sp: -1.j * expn_val_cc[ii + 1]}) *
                     function_bose(-1.J * expn_val_cc[ii + 1] / temp, pole, resi))))

        etar.append(np.conj(etal[-1]))
        etar.append(np.conj(etal[-2]))
        etaa.append(np.sqrt(np.abs(etal[-2]) * np.abs(etar[-2])))
        etaa.append(np.sqrt(np.abs(etal[-1]) * np.abs(etar[-1])))

    for ii in range(len(expn_val_n_cc)):
        etal.append(
            complex(
                sp.N((-2.j * numer /
                      np.multiply.reduce(w_sp - poles_allplane[np.abs(
                          poles_allplane + 1.J * expn_val_n_cc[ii]) > 1e-14])
                      ).subs({w_sp: -1.j * expn_val_n_cc[ii]}) *
                     function_bose(-1.J * expn_val_n_cc[ii] / temp, pole, resi))))
        etar.append(np.conj(etal[-1]))
        etaa.append(np.sqrt(np.abs(etal[-1]) * np.abs(etar[-1])))

    f = numer / np.multiply.reduce(w_sp - poles_allplane)
    f = sp.lambdify(w_sp, f)

    for inma in range(len(pole)):
        zomg = -1.J * pole[inma] * temp
        jsum = np.sum(f(zomg))
        expn.append(pole[inma] * temp)
        etal.append(-2.J * resi[inma] * temp * jsum)
        etar.append(np.conj(etal[-1]))
        etaa.append(np.abs(etal[-1]))

    etal = np.array(etal)
    etar = np.array(etar)
    etaa = np.array(etaa)
    expn = np.array(expn)
    return etal, etar, etaa, expn


def decompose_spectrum_pade_real(spe, w_sp):
    if (sp.cancel(spe).as_real_imag()[1] == 0):
        imag_part = sp.cancel(spe).as_real_imag()[0]
    else:
        imag_part = sp.cancel(spe).as_real_imag()[1]
    numer, denom = sp.cancel(sp.factor(imag_part)).as_numer_denom()
    poles = sp.nroots(denom)
    float(sp.re(poles[0]))

    expn, etal, etar, etaa = [], [], [], []
    poles_allplane = np.array([])
    for i in poles:
        i = complex(i)
        if i.imag < 0:
            expn.append(i * 1.J)
        poles_allplane = np.append(poles_allplane, i)

    expn = np.array(expn)
    expn_imag_sort = np.argsort(np.abs(np.imag(expn)))[::-1]
    expn_imag = np.sort(np.abs(np.imag(expn)))[::-1]
    expn_val_cc = expn[expn_imag_sort[expn_imag != 0]]
    expn_val_n_cc = expn[expn_imag_sort[expn_imag == 0]]

    for ii in range(0, len(expn_val_cc), 2):
        etal.append(
            complex(
                sp.N((-1.j * numer /
                      np.multiply.reduce(w_sp - poles_allplane[np.abs(
                          poles_allplane + 1.J * expn_val_cc[ii]) > 1e-14])
                      ).subs({w_sp: -1.j * expn_val_cc[ii]}))))

        etal.append(
            complex(
                sp.N((-1.j * numer /
                      np.multiply.reduce(w_sp - poles_allplane[np.abs(
                          poles_allplane + 1.J * expn_val_cc[ii + 1]) > 1e-14])
                      ).subs({w_sp: -1.j * expn_val_cc[ii + 1]}))))

        etar.append(np.conj(etal[-1]))
        etar.append(np.conj(etal[-2]))
        etaa.append(np.sqrt(np.abs(etal[-2]) * np.abs(etar[-2])))
        etaa.append(np.sqrt(np.abs(etal[-1]) * np.abs(etar[-1])))

    for ii in range(len(expn_val_n_cc)):
        etal.append(
            complex(
                sp.N((-1.j * numer /
                      np.multiply.reduce(w_sp - poles_allplane[np.abs(
                          poles_allplane -
                          -1.J * expn_val_n_cc[ii]) > 1e-14])).subs(
                              {w_sp: -1.j * expn_val_n_cc[ii]}))))
        etar.append(np.conj(etal[-1]))
        etaa.append(np.sqrt(np.abs(etal[-1]) * np.abs(etar[-1])))

    return np.array(etal), np.array(
        etar), np.array(etaa), np.array(expn)


def decompose_spectrum_pade_imag(spe, w_sp):
    if (sp.cancel(spe).as_real_imag()[1] == 0):
        imag_part = sp.cancel(spe).as_real_imag()[0]
    else:
        imag_part = sp.cancel(spe).as_real_imag()[1]
    numer, denom = sp.cancel(sp.factor(imag_part)).as_numer_denom()

    poles = sp.nroots(denom)
    float(sp.re(poles[0]))

    expn, etal, etar, etaa = [], [], [], []
    poles_allplane = np.array([])
    for i in poles:
        i = complex(i)
        if i.imag < 0:
            expn.append(i * 1.J)
        poles_allplane = np.append(poles_allplane, i)

    expn = np.array(expn)

    expn_imag_sort = np.argsort(np.abs(np.imag(expn)))[::-1]
    expn_imag = np.sort(np.abs(np.imag(expn)))[::-1]
    expn_val_cc = expn[expn_imag_sort[expn_imag != 0]]
    expn_val_n_cc = expn[expn_imag_sort[expn_imag == 0]]

    for ii in range(0, len(expn_val_cc), 2):
        etal.append(
            complex(
                sp.N((-1.j * numer /
                      np.multiply.reduce(w_sp - poles_allplane[np.abs(
                          poles_allplane + 1.J * expn_val_cc[ii]) > 1e-14])
                      ).subs({w_sp: -1.j * expn_val_cc[ii]}))))

        etal.append(
            complex(
                sp.N((-1.j * numer /
                      np.multiply.reduce(w_sp - poles_allplane[np.abs(
                          poles_allplane + 1.J * expn_val_cc[ii + 1]) > 1e-14])
                      ).subs({w_sp: -1.j * expn_val_cc[ii + 1]}))))

        etar.append(np.conj(etal[-1]))
        etar.append(np.conj(etal[-2]))
        etaa.append(np.sqrt(np.abs(etal[-2]) * np.abs(etar[-2])))
        etaa.append(np.sqrt(np.abs(etal[-1]) * np.abs(etar[-1])))

    for ii in range(len(expn_val_n_cc)):
        etal.append(
            complex(
                sp.N((-1.j * numer /
                      np.multiply.reduce(w_sp - poles_allplane[np.abs(
                          poles_allplane + 1.J * expn_val_n_cc[ii]) > 1e-14])
                      ).subs({w_sp: -1.j * expn_val_n_cc[ii]}))))
        etar.append(np.conj(etal[-1]))
        etaa.append(np.sqrt(np.abs(etal[-1]) * np.abs(etar[-1])))

    return np.array(etal), np.array(
        etar), np.array(etaa), np.array(expn)


def prony_find_gamma(h, n_sample, nind):
    mat_h = np.zeros((n_sample, n_sample))
    for i in range(n_sample):
        mat_h[i, :] = h[i:n_sample + i]
    sing_vs, Q = la.eig(mat_h)
    phase_mat = np.diag([np.exp(-1j * np.angle(sing_v) / 2.0)
                        for sing_v in sing_vs])
    vs = np.array([np.abs(sing_v) for sing_v in sing_vs])
    Qp = np.dot(Q, phase_mat)
    sort_array = np.argsort(vs)[::-1]
    vs = vs[sort_array]
    Qp = (Qp[:, sort_array])

    gamma = np.roots(Qp[:, nind][::-1])
    gamma_new = gamma[np.argsort(np.abs(gamma))[:nind]]
    return gamma_new


def prony_fitting(h, t, nind, scale, n, gamma_real=None, gamma_imag=None):
    '''
    h: sample
    t: list of time
    nind: number of poles. Can be a list of [nind_real, nind_imag], or a single number. If nind_real or nind_imag is 'a', then it will be using the analytical method.
    scale: range of the sample, for example, if the time is from 0 to 10, then the scale should be 10.
    n: number of sample
    gamma_real: the gamma for real part. If not given, it will be calculated.
    gamma_imag: the gamma for imag part. If not given, it will be calculated.
    '''
    if isinstance(nind, (list, tuple)):
        if (gamma_real is None):
            gamma_real = prony_find_gamma(np.real(h), n, nind[0])
        else:
            gamma_real = np.array(gamma_real)
        if (gamma_imag is None):
            gamma_imag = prony_find_gamma(np.imag(h), n, nind[1])
        else:
            gamma_imag = np.array(gamma_imag)
        gamma = np.append(gamma_real, gamma_imag)
        n_row = nind[0] + nind[1]
    else:
        gamma = prony_find_gamma(np.real(h), n, nind)
        n_row = nind

    t_new = 2*n*np.log(gamma)
    n_col = len(t)
    gamma_m = np.zeros((2 * n_col, 2 * n_row), dtype=float)
    for i in range(n_row):
        for j in range(n_col):
            gamma_m[j, i] = np.real(gamma[i]**j)
            gamma_m[n_col + j, n_row + i] = np.real(gamma[i]**j)
            gamma_m[j, n_row + i] = -np.imag(gamma[i]**j)
            gamma_m[n_col + j, i] = np.imag(gamma[i]**j)
    h_m = np.append(np.real(h), np.imag(h))

    # freq_m = np.zeros((2 * n_col, 2 * n_row), dtype=float)
    # C = numpy_to_cvxopt_matrix(gamma_m)
    # d = numpy_to_cvxopt_matrix(h_m)
    # A = numpy_to_cvxopt_matrix(-freq_m)
    # b = numpy_to_cvxopt_matrix(np.zeros(2 * n_col))
    # Q = C.T * C
    # q = - d.T * C
    # opts = {'show_progress': False, 'abstol': 1e-50,
    #         'reltol': 1e-50, 'feastol': 1e-50}
    # for k, v in opts.items():
    #     solvers.options[k] = v
    # sol = solvers.qp(Q, q.T, A, b, None, None, None, None)
    # omega_new_temp = np.array(sol['x']).reshape(2, n_row)

    omega_new_temp = la.lstsq(gamma_m, h_m)[0].reshape(2, n_row)
    #
    omega_new = omega_new_temp[0, :] + 1.j*omega_new_temp[1, :]

    etal_p = omega_new
    expn_p = -t_new / scale
    return sort_symmetry(etal_p, expn_p)


def fit_spectrum_prony(
    spe: sp.core.mul.Mul,
    w_sp: sp.core.symbol.Symbol,
    beta,
    nind: int | list | tuple | None = None,
    *,
    nexp: int | list | tuple | None = None,
    scale=250000,
    n=1250,
    npsd=10,
    bose_fermi=1,
    as_bath=False,
    mode=None,
    verbose=False,
    omega_grid=None,
    omega_min=1.0e-8,
    omega_max=30.0,
    n_omega=4096,
    fit="time",
    frequency_weight=0.1,
    refine_n_omega=240,
    refine_max_nfev=80,
):
    """Fit a bath spectrum with Prony exponentials.

    ``spe`` is a SymPy spectral-density expression in frequency symbol
    ``w_sp``.  ``nexp`` is the preferred name for the number of exponentials;
    ``nind`` remains accepted for older call sites.  For non-rational spectra,
    ``fit="hybrid"`` refines the sampled-correlation fit against a positive
    frequency grid as well as against the time-domain correlation.
    """
    if fit not in {"time", "hybrid"}:
        raise ValueError("fit must be 'time' or 'hybrid'.")
    if nind is not None and nexp is not None:
        raise ValueError("Specify only one of nind or nexp.")
    if nexp is not None:
        nind = nexp
    if nind is None:
        raise ValueError("Specify nind or nexp.")
    if isinstance(nind, tuple):
        nind = list(nind)
    elif isinstance(nind, list):
        nind = list(nind)

    t = np.linspace(0, 1, 2 * n + 1)
    sampled_correlation = False
    try:
        etal_pade, _, _, expn_pade = decompose_spectrum_pade(
            spe, w_sp, beta, npsd, bose_fermi=bose_fermi)
        res_t = np.zeros(len(t), dtype=complex)
        fit_t(scale * t, res_t, expn_pade, etal_pade)
    except Exception as exc:
        if bose_fermi != 1:
            raise
        if isinstance(nind, list) and ("a" in nind):
            raise ValueError("Analytical real/imaginary pole shortcuts require a rational spectrum.") from exc
        res_t = _sample_boson_correlation(
            spe,
            w_sp,
            beta,
            scale * t,
            omega_grid=omega_grid,
            omega_min=omega_min,
            omega_max=omega_max,
            n_omega=n_omega,
        )
        sampled_correlation = True

    if verbose:
        print("check the sample points")
        print(res_t[:10])
        print(res_t[-10:])
    if isinstance(nind, list):
        if nind[0] == 'a':
            _, _, _, expn_real = decompose_spectrum_pade_real(spe, w_sp)
            gamma_real = np.exp(- expn_real * scale / (2*n))
            nind[0] = len(gamma_real)
            if bose_fermi == 1:
                raise ValueError("For the bose case, C(t) has the analytical imaginary part.")
            etal, etar, etaa, expn = prony_fitting(
                res_t, t, nind, scale, n, gamma_real=gamma_real)
        elif nind[1] == 'a':
            _, _, _, expn_imag = decompose_spectrum_pade_imag(spe, w_sp)
            gamma_imag = np.exp(- expn_imag * scale / (2*n))
            nind[1] = len(gamma_imag)
            if bose_fermi == 2:
                raise ValueError("For the fermi case, C(t) has the analytical real part.")
            etal, etar, etaa, expn = prony_fitting(
                res_t, t, nind, scale, n, gamma_imag=gamma_imag)
        else:
            etal, etar, etaa, expn = prony_fitting(res_t, t, nind, scale, n)
    else:
        etal, etar, etaa, expn = prony_fitting(res_t, t, nind, scale, n)

    if sampled_correlation:
        stable = np.real(expn) > 1.0e-12
        if not np.any(stable):
            raise ValueError("Prony fit did not produce any decaying exponentials.")
        etal, etar, etaa, expn = etal[stable], etar[stable], etaa[stable], expn[stable]
        if fit == "hybrid":
            etal, etar, etaa, expn = _refine_sampled_prony_fit(
                spe,
                w_sp,
                beta,
                scale * t,
                res_t,
                etal,
                expn,
                omega_min=omega_min,
                omega_max=omega_max,
                n_omega=refine_n_omega,
                frequency_weight=frequency_weight,
                max_nfev=refine_max_nfev,
            )

    if as_bath:
        return Bath.from_exponential_terms(expn, etal, etar, etaa, mode=mode)
    return etal, etar, etaa, expn


prony = fit_spectrum_prony


def _sample_boson_correlation(
    spe,
    w_sp,
    beta,
    times,
    *,
    omega_grid=None,
    omega_min=1.0e-8,
    omega_max=30.0,
    n_omega=4096,
):
    """Sample a bosonic bath correlation from a spectral density."""
    times = np.asarray(times, dtype=float)
    if omega_grid is None:
        omega_grid = np.geomspace(float(omega_min), float(omega_max), int(n_omega))
    else:
        omega_grid = np.asarray(omega_grid, dtype=float)
    if np.any(omega_grid <= 0.0):
        raise ValueError("omega_grid must contain positive frequencies.")

    spectrum = sp.lambdify(w_sp, spe, "numpy")
    values = np.asarray(spectrum(omega_grid), dtype=np.complex128)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    thermal = 1.0 / np.tanh(0.5 * float(beta) * omega_grid)
    phase = np.outer(times, omega_grid)
    integrand = values[None, :] * (
        thermal[None, :] * np.cos(phase) - 1.j * np.sin(phase)
    )
    return np.trapezoid(integrand, omega_grid, axis=1)


def _sample_boson_correlation_spectrum(spe, w_sp, beta, omega):
    omega = np.asarray(omega, dtype=float)
    if np.any(omega <= 0.0):
        raise ValueError("omega must contain positive frequencies.")
    spectrum = sp.lambdify(w_sp, spe, "numpy")
    values = np.asarray(spectrum(omega), dtype=np.complex128)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return np.real(values / (1.0 - np.exp(-float(beta) * omega)))


def _real_lstsq_for_complex_coefficients(
    time_matrix,
    time_target,
    *,
    time_weight=1.0,
    freq_matrix=None,
    freq_target=None,
    freq_weight=1.0,
):
    ar = np.real(time_matrix)
    ai = np.imag(time_matrix)
    target_r = np.real(time_target)
    target_i = np.imag(time_target)
    time_weight = np.asarray(time_weight, dtype=float)

    lhs = [
        np.hstack([ar, -ai]) * time_weight[:, None],
        np.hstack([ai, ar]) * time_weight[:, None],
    ]
    rhs = [target_r * time_weight, target_i * time_weight]

    if freq_matrix is not None:
        fr = np.real(freq_matrix)
        fi = np.imag(freq_matrix)
        freq_weight = np.asarray(freq_weight, dtype=float)
        lhs.append(np.hstack([fr, -fi]) * freq_weight[:, None])
        rhs.append(np.asarray(freq_target, dtype=float) * freq_weight)

    solution = la.lstsq(np.vstack(lhs), np.concatenate(rhs))[0]
    n_terms = time_matrix.shape[1]
    return solution[:n_terms] + 1.0j * solution[n_terms:]


def _refine_sampled_prony_fit(
    spe,
    w_sp,
    beta,
    times,
    sampled_correlation,
    etal,
    expn,
    *,
    omega_min,
    omega_max,
    n_omega,
    frequency_weight,
    max_nfev,
):
    times = np.asarray(times, dtype=float)
    sampled_correlation = np.asarray(sampled_correlation, dtype=np.complex128)
    etal = np.asarray(etal, dtype=np.complex128)
    expn = np.asarray(expn, dtype=np.complex128)

    n_time = min(len(times), 320)
    time_index = np.unique(np.linspace(0, len(times) - 1, n_time, dtype=int))
    fit_times = times[time_index]
    fit_correlation = sampled_correlation[time_index]
    time_scale = max(np.max(np.abs(fit_correlation)), 1.0e-14)
    time_weight = np.full(len(fit_times), 1.0 / time_scale)

    omega_lo = max(float(omega_min), 1.0e-5)
    fit_omega = np.geomspace(omega_lo, float(omega_max), int(n_omega))
    fit_spectrum = _sample_boson_correlation_spectrum(spe, w_sp, beta, fit_omega)
    spectrum_scale = max(np.max(np.abs(fit_spectrum)), 1.0e-14)
    freq_weight = np.full(len(fit_omega), float(frequency_weight) / spectrum_scale)

    real0 = np.maximum(np.real(expn), 1.0e-10)
    imag0 = np.imag(expn)
    x0 = np.concatenate([np.log(real0), imag0])
    imag_bound = max(2.0 * float(omega_max), np.max(np.abs(imag0)) + 1.0)
    lower = np.concatenate([
        np.full(len(expn), np.log(1.0e-10)),
        np.full(len(expn), -imag_bound),
    ])
    upper = np.concatenate([
        np.full(len(expn), np.log(max(100.0 * float(omega_max), 1.0))),
        np.full(len(expn), imag_bound),
    ])

    def build_terms(params, use_full=False):
        n_terms = len(expn)
        gamma = np.exp(params[:n_terms]) + 1.0j * params[n_terms:]
        t_grid = times if use_full else fit_times
        time_matrix = np.exp(-np.outer(t_grid, gamma))
        freq_matrix = 1.0 / (gamma[None, :] - 1.0j * fit_omega[:, None]) / np.pi
        target = sampled_correlation if use_full else fit_correlation
        weights = np.full(len(t_grid), 1.0 / time_scale)
        coeff = _real_lstsq_for_complex_coefficients(
            time_matrix,
            target,
            time_weight=weights,
            freq_matrix=freq_matrix,
            freq_target=fit_spectrum,
            freq_weight=freq_weight,
        )
        return gamma, coeff, time_matrix, freq_matrix, target, weights

    def residual(params):
        _, coeff, time_matrix, freq_matrix, target, weights = build_terms(params)
        time_res = (time_matrix @ coeff - target) * weights
        freq_res = (np.real(freq_matrix @ coeff) - fit_spectrum) * freq_weight
        return np.concatenate([np.real(time_res), np.imag(time_res), freq_res])

    result = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
        max_nfev=int(max_nfev),
        xtol=1.0e-10,
        ftol=1.0e-10,
        gtol=1.0e-10,
    )
    gamma, coeff, _, _, _, _ = build_terms(result.x, use_full=True)
    stable = np.real(gamma) > 1.0e-12
    return sort_symmetry(coeff[stable], gamma[stable])


def single_oscillator(omega: float, w_sp: sp.core.symbol.Symbol, beta: float, nind: int):
    etal = np.array([1/(2 * (1-np.exp(-beta * omega))), -1 /
                    (2 * (1-np.exp(beta * omega)))], dtype=complex)
    etar = np.array([-1/(2 * (1-np.exp(beta * omega))), 1 /
                    (2 * (1-np.exp(-beta * omega)))], dtype=complex)
    etaa = np.sqrt(np.abs(etal + etar))
    expn = np.array([1.j * omega, -1.j * omega])
    return etal, etar, etaa, expn


@njit
def gen_hash_value(key, nexp, comb_list):
    '''
    generate the hash value for the key
    '''
    sum_ = 0
    hash_value = 0
    for i in range(nexp):
        sum_ += key[i]
        hash_value += comb_list[sum_ + i, i + 1]
    return hash_value


@njit
def key_plus(key, pos):
    '''
    generate the key for the \rho_{{n}^{+}_{k}}
    '''
    key1 = key.copy()
    key1[pos] += 1
    return key1


@njit
def key_minus(key, pos):
    '''
    generate the key for the \rho_{{n}^{-}_{k}}
    '''
    key1 = key.copy()
    key1[pos] -= 1
    return key1


@njit
def hash_plus(key, pos, nexp, comb_list):
    '''
    generate the hash value for the \rho_{{n}^{+}_{k}}
    '''
    key1 = key.copy()
    key1[pos] += 1
    return gen_hash_value(key1, nexp, comb_list)


@njit
def hash_minus(key, pos, nexp, comb_list):
    '''
    generate the hash value for the \rho_{{n}^{-}_{k}}
    '''
    key1 = key.copy()
    key1[pos] -= 1
    return gen_hash_value(key1, nexp, comb_list)


@njit
def gen_keys_element(keys, iado, comb_list, nexp, lmax):
    '''
    construct the keys for the iado-th element
    '''
    hash_val = gen_hash_value(keys[iado], nexp, comb_list)

    for mp in range(nexp):
        if (np.sum(keys[iado]) < lmax):
            hash_val = hash_plus(keys[iado], mp, nexp, comb_list)
            keys[hash_val] = key_plus(keys[iado], mp)
        if (keys[iado, mp] > 0):
            hash_val = hash_minus(keys[iado], mp, nexp, comb_list)
            keys[hash_val] = key_minus(keys[iado], mp)


@njit
def gen_keys(keys, lmax, nexp, comb_list):
    '''
    construct the keys for the whole system
    '''
    nddo = 1
    nddo_bef = 0
    for ii in range(lmax + 1):
        for iado in range(nddo_bef, nddo):
            gen_keys_element(keys, iado, comb_list, nexp, lmax)
        nddo_bef = nddo
        if (ii <= lmax):
            nddo = comb_list[nexp + ii, nexp]
        else:
            nddo = comb_list[nexp + lmax, nexp]


def smolyak_hierarchy_level(n):
    """Return the one-dimensional Smolyak level for an ADO occupation."""
    n = int(n)
    if n < 0:
        raise ValueError("Hierarchy occupations must be non-negative.")
    if n == 0:
        return 0
    return n.bit_length()


def smolyak_hierarchy_score(key, weights=None):
    """Return the Smolyak score of an ADO multi-index."""
    if weights is None:
        return sum(smolyak_hierarchy_level(n) for n in key)
    weights = _validate_smolyak_weights(weights, len(key))
    return sum(int(w) * smolyak_hierarchy_level(n) for w, n in zip(weights, key))


def _validate_smolyak_weights(weights, nexp):
    weights = np.asarray(weights, dtype=np.int64)
    if weights.shape != (int(nexp),):
        raise ValueError("Smolyak weights must have shape (nexp,).")
    if np.any(weights < 1):
        raise ValueError("Smolyak weights must be positive integers.")
    return weights


def _append_smolyak_keys(keys, current, pos, remaining_level, weights):
    if pos == len(current):
        keys.append(tuple(current))
        return
    max_level = remaining_level // int(weights[pos])
    max_occupation = 0 if max_level <= 0 else 2**max_level - 1
    for occupation in range(max_occupation + 1):
        level = int(weights[pos]) * smolyak_hierarchy_level(occupation)
        if level > remaining_level:
            continue
        current[pos] = occupation
        _append_smolyak_keys(keys, current, pos + 1, remaining_level - level, weights)
    current[pos] = 0


def generate_smolyak_keys(lmax, nexp, weights=None):
    """Generate ADO keys with weighted Smolyak score below ``lmax``."""
    if weights is None:
        weights = np.ones(int(nexp), dtype=np.int64)
    weights = _validate_smolyak_weights(weights, nexp)

    keys = []
    _append_smolyak_keys(keys, [0] * int(nexp), 0, int(lmax), weights)
    keys.sort(key=lambda key: (smolyak_hierarchy_score(key, weights), sum(key), key))
    return np.asarray(keys, dtype=np.int64)


def gamma_smolyak_weights(gamma):
    """Return anisotropic Smolyak weights from bath exponent rates."""
    gamma = np.asarray(gamma, dtype=np.complex128)
    rates = np.abs(np.real(gamma))
    positive = rates[rates > 1.0e-14]
    if len(positive) == 0:
        return np.ones(len(gamma), dtype=np.int64)
    scale = np.min(positive)
    return np.maximum(1, np.ceil(np.log2(1.0 + rates / scale))).astype(np.int64)


def generate_dot_element(dot_ddos, ddos, key, lmax, bath_list, mode, H, Q, comb_list, iado):
    '''
    generate the dot_ddos for the iado-th element
    '''
    # unpack coupling_t
    expn, etal, etar, etaa = bath_list
    nexp = len(expn)
    dot_ddos[iado] = - np.sum(key * expn) * ddos[iado]
    dot_ddos[iado] -= 1.j * (H @ ddos[iado] - ddos[iado] @ H)

    for mp in range(nexp):
        n = key[mp]
        if (n > 0):
            pos = hash_minus(key, mp, nexp, comb_list)
            m = mode[mp]
            dot_ddos[iado] -= 1.j * np.sqrt(n) / np.sqrt(etaa[mp]) * (
                etal[mp] * Q[m] @ ddos[pos] - etar[mp] * ddos[pos] @ Q[m])

        if (sum(key) < lmax):
            pos = hash_plus(key, mp, nexp, comb_list)
            m = mode[mp]
            dot_ddos[iado] -= 1.j * \
                np.sqrt(n + 1) * np.sqrt(etaa[mp]) * \
                (Q[m] @ ddos[pos] - ddos[pos] @ Q[m])


def generate_dot_element_by_index(dot_ddos, ddos, key, key_to_index, bath_list, mode, H, Q, iado):
    '''
    generate the dot_ddos for arbitrary hierarchy key sets
    '''
    expn, etal, etar, etaa = bath_list
    nexp = len(expn)
    dot_ddos[iado] = -np.sum(key * expn) * ddos[iado]
    dot_ddos[iado] -= 1.j * (H @ ddos[iado] - ddos[iado] @ H)

    for mp in range(nexp):
        n = int(key[mp])
        m = mode[mp]
        if n > 0:
            key1 = key.copy()
            key1[mp] -= 1
            pos = key_to_index.get(tuple(key1))
            if pos is not None:
                dot_ddos[iado] -= 1.j * np.sqrt(n) / np.sqrt(etaa[mp]) * (
                    etal[mp] * Q[m] @ ddos[pos] - etar[mp] * ddos[pos] @ Q[m])

        key1 = key.copy()
        key1[mp] += 1
        pos = key_to_index.get(tuple(key1))
        if pos is not None:
            dot_ddos[iado] -= 1.j * np.sqrt(n + 1) * np.sqrt(etaa[mp]) * (
                Q[m] @ ddos[pos] - ddos[pos] @ Q[m])


def build_neighbor_indices(keys):
    """Return minus/plus neighbor index tables for a hierarchy key set."""
    keys = np.asarray(keys, dtype=np.int64)
    nmax, nexp = keys.shape
    key_to_index = {tuple(key): i for i, key in enumerate(keys)}
    minus_index = np.full((nmax, nexp), -1, dtype=np.int64)
    plus_index = np.full((nmax, nexp), -1, dtype=np.int64)

    for iado, key in enumerate(keys):
        for mp in range(nexp):
            if key[mp] > 0:
                key1 = key.copy()
                key1[mp] -= 1
                minus_index[iado, mp] = key_to_index.get(tuple(key1), -1)

            key1 = key.copy()
            key1[mp] += 1
            plus_index[iado, mp] = key_to_index.get(tuple(key1), -1)

    return minus_index, plus_index


def build_native_edge_tables(keys, minus_index, plus_index, bath_list, mode):
    """Return CSR-style lower/upper hierarchy edge tables for native RHS."""
    keys = np.asarray(keys, dtype=np.int64)
    minus_index = np.asarray(minus_index, dtype=np.int64)
    plus_index = np.asarray(plus_index, dtype=np.int64)
    _, etal, etar, etaa = bath_list
    etal = np.asarray(etal, dtype=np.complex128)
    etar = np.asarray(etar, dtype=np.complex128)
    etaa = np.asarray(etaa, dtype=np.complex128)
    mode = np.asarray(mode, dtype=np.int64)
    nmax, nexp = keys.shape

    lower_mask = (keys > 0) & (minus_index >= 0)
    upper_mask = plus_index >= 0

    lower_offset = np.zeros(nmax + 1, dtype=np.int64)
    upper_offset = np.zeros(nmax + 1, dtype=np.int64)
    lower_offset[1:] = np.cumsum(np.count_nonzero(lower_mask, axis=1))
    upper_offset[1:] = np.cumsum(np.count_nonzero(upper_mask, axis=1))

    _, lower_mp = np.nonzero(lower_mask)
    lower_src = minus_index[lower_mask]
    lower_mode = mode[lower_mp]
    lower_occupation = keys[lower_mask].astype(np.float64)
    lower_scale = np.sqrt(lower_occupation) / np.sqrt(etaa[lower_mp])
    lower_left = -1.0j * lower_scale * etal[lower_mp]
    lower_right = 1.0j * lower_scale * etar[lower_mp]

    _, upper_mp = np.nonzero(upper_mask)
    upper_src = plus_index[upper_mask]
    upper_mode = mode[upper_mp]
    upper_occupation = keys[upper_mask].astype(np.float64)
    upper_scale = np.sqrt(upper_occupation + 1.0) * np.sqrt(etaa[upper_mp])
    upper_left = -1.0j * upper_scale
    upper_right = 1.0j * upper_scale

    return (
        np.ascontiguousarray(lower_offset, dtype=np.int64),
        np.ascontiguousarray(lower_src, dtype=np.int64),
        np.ascontiguousarray(lower_mode, dtype=np.int64),
        np.ascontiguousarray(lower_left, dtype=np.complex128),
        np.ascontiguousarray(lower_right, dtype=np.complex128),
        np.ascontiguousarray(upper_offset, dtype=np.int64),
        np.ascontiguousarray(upper_src, dtype=np.int64),
        np.ascontiguousarray(upper_mode, dtype=np.int64),
        np.ascontiguousarray(upper_left, dtype=np.complex128),
        np.ascontiguousarray(upper_right, dtype=np.complex128),
    )


def generate_total_keys(lmax, nexp):
    """Generate keys satisfying sum(key) <= lmax with the zero ADO first."""
    keys = []
    key = np.zeros(nexp, dtype=np.int64)

    def visit(pos, remaining):
        if pos == nexp:
            keys.append(key.copy())
            return
        for occupation in range(remaining + 1):
            key[pos] = occupation
            visit(pos + 1, remaining - occupation)
        key[pos] = 0

    visit(0, lmax)
    return np.ascontiguousarray(keys, dtype=np.int64)


def rem_cal(dot_ddos, ddos, keys, lmax, bath_list, mode, system_t, coupling_t, comb_list, nmax):
    '''
    calculate the dot_ddos for the whole system
    '''
    for iado in range(nmax):
        generate_dot_element(
            dot_ddos, ddos, keys[iado], lmax, bath_list, mode, system_t, coupling_t, comb_list, iado)


def rem_cal_by_index(dot_ddos, ddos, keys, key_to_index, bath_list, mode, system_t, coupling_t, nmax):
    '''
    calculate dot_ddos for arbitrary hierarchy key sets
    '''
    for iado in range(nmax):
        generate_dot_element_by_index(
            dot_ddos, ddos, keys[iado], key_to_index, bath_list, mode, system_t, coupling_t, iado)


def generate_time(system, system_dip, pulse_system_func, coupling, \
                  coupling_dip, pulse_coupling_func, t):
    '''
    generate the system hamiltonian and the coupling hamiltonian at the next time step
    '''
    system_t = system + system_dip * pulse_system_func(t)
    coupling_t = [coo_matrix(
        np.shape(coupling[0]), dtype=np.complex128)] * len(coupling)
    for i in range(np.shape(coupling)[0]):
        coupling_t[i] = coupling[i] + \
            coupling_dip[i] * pulse_coupling_func(t)
    return system_t, coupling_t


def generate_time_dense(system, system_dip, pulse_system_func, coupling,
                        coupling_dip, pulse_coupling_func, t):
    """Dense version of :func:`generate_time` for native kernels."""
    system_t = np.asarray(system + system_dip * pulse_system_func(t), dtype=np.complex128)
    coupling_t = np.asarray(coupling, dtype=np.complex128) + \
        np.asarray(coupling_dip, dtype=np.complex128) * pulse_coupling_func(t)
    return np.ascontiguousarray(system_t), np.ascontiguousarray(coupling_t)


def _as_operator_list(operators, name):
    if operators is None:
        return None
    array = np.asarray(operators, dtype=np.complex128)
    if array.ndim == 2:
        return [array]
    if array.ndim == 3:
        return [np.asarray(operator, dtype=np.complex128) for operator in array]
    if isinstance(operators, (list, tuple)):
        return [np.asarray(operator, dtype=np.complex128) for operator in operators]
    raise ValueError(f"{name} must be a matrix or a list of matrices.")


def native_rem_cal_by_neighbors(ddos, keys, minus_index, plus_index, bath_list, mode, system_t, coupling_t):
    """Evaluate the dense HEOM RHS with the optional C++ backend."""
    heom_cpp = _get_heom_cpp()
    if heom_cpp is None:
        raise RuntimeError("optional HEOM C++ extension is not available")
    expn, etal, etar, etaa = bath_list
    return heom_cpp.rhs_by_index(
        np.ascontiguousarray(ddos, dtype=np.complex128),
        np.ascontiguousarray(keys, dtype=np.int64),
        np.ascontiguousarray(minus_index, dtype=np.int64),
        np.ascontiguousarray(plus_index, dtype=np.int64),
        np.ascontiguousarray(expn, dtype=np.complex128),
        np.ascontiguousarray(etal, dtype=np.complex128),
        np.ascontiguousarray(etar, dtype=np.complex128),
        np.ascontiguousarray(etaa, dtype=np.complex128),
        np.ascontiguousarray(mode, dtype=np.int64),
        system_t,
        coupling_t,
    )


def rk4_native_total(ddos, keys, minus_index, plus_index, bath_list, mode, system,
                     system_dip, pulse_system_func, coupling, coupling_dip,
                     pulse_coupling_func, dt, t, workspace=None, decay_rates=None):
    """Runge-Kutta step for total/simplex hierarchies using the C++ backend."""
    heom_cpp = _get_heom_cpp()
    if heom_cpp is None:
        raise RuntimeError("optional HEOM C++ extension is not available")
    system_0, coupling_0 = generate_time_dense(
        system, system_dip, pulse_system_func, coupling, coupling_dip, pulse_coupling_func, t)
    system_half, coupling_half = generate_time_dense(
        system, system_dip, pulse_system_func, coupling, coupling_dip, pulse_coupling_func, t + dt / 2)
    system_1, coupling_1 = generate_time_dense(
        system, system_dip, pulse_system_func, coupling, coupling_dip, pulse_coupling_func, t + dt)

    expn, etal, etar, etaa = bath_list
    args = [
        ddos,
        np.ascontiguousarray(keys, dtype=np.int64),
        np.ascontiguousarray(minus_index, dtype=np.int64),
        np.ascontiguousarray(plus_index, dtype=np.int64),
        np.ascontiguousarray(expn, dtype=np.complex128),
        np.ascontiguousarray(etal, dtype=np.complex128),
        np.ascontiguousarray(etar, dtype=np.complex128),
        np.ascontiguousarray(etaa, dtype=np.complex128),
        np.ascontiguousarray(mode, dtype=np.int64),
        system_0,
        coupling_0,
        system_half,
        coupling_half,
        system_1,
        coupling_1,
        dt,
    ]
    if workspace is not None:
        args.extend(workspace)
        if decay_rates is not None:
            args.append(decay_rates)
    heom_cpp.rk4_step_by_index(*args)


def native_rhs_total(ddos, keys, minus_index, plus_index, bath_list, mode,
                     system_t, coupling_t):
    """Evaluate the dense total-hierarchy RHS using cached native arrays."""
    heom_cpp = _get_heom_cpp()
    if heom_cpp is None:
        raise RuntimeError("optional HEOM C++ extension is not available")
    expn, etal, etar, etaa = bath_list
    return heom_cpp.rhs_by_index(
        np.ascontiguousarray(ddos, dtype=np.complex128),
        keys,
        minus_index,
        plus_index,
        expn,
        etal,
        etar,
        etaa,
        mode,
        system_t,
        coupling_t,
    )


# def rk4(ddos, ddos1, ddos2, ddos3, keys, lmax, bath_list, mode, system, system_dip, pulse_system_func, coupling, coupling_dip, pulse_coupling_func, comb_list, nmax, dt, t):
#     '''
#     Runge-Kutta 4th order method
#     '''
#     system_t, coupling_t = generate_time(
#         system, system_dip, pulse_system_func, coupling, coupling_dip, pulse_coupling_func, t)
#     rem_cal(ddos1, ddos, keys, lmax, bath_list, mode, system_t,
#             coupling_t, comb_list, nmax)
#     for i in range(nmax):
#         ddos3[i] = ddos[i] + ddos1[i] * dt / 2

#     system_t, coupling_t = generate_time(
#         system, system_dip, pulse_system_func, coupling, coupling_dip, pulse_coupling_func, t + dt / 2)
#     rem_cal(ddos2, ddos3, keys, lmax, bath_list, mode, system_t,
#             coupling_t, comb_list, nmax)
#     for i in range(nmax):
#         ddos1[i] += ddos2[i] * 2
#         ddos3[i] = ddos[i] + ddos2[i] * dt / 2

#     system_t, coupling_t = generate_time(
#         system, system_dip, pulse_system_func, coupling, coupling_dip, pulse_coupling_func, t + dt / 2)
#     rem_cal(ddos2, ddos3, keys, lmax, bath_list, mode, system_t,
#             coupling_t, comb_list, nmax)
#     for i in range(nmax):
#         ddos1[i] += ddos2[i] * 2
#         ddos3[i] = ddos[i] + ddos2[i] * dt

#     system_t, coupling_t = generate_time(
#         system, system_dip, pulse_system_func, coupling, coupling_dip, pulse_coupling_func, t + dt)
#     rem_cal(ddos2, ddos3, keys, lmax, bath_list, mode, system_t,
#             coupling_t, comb_list, nmax)
#     for i in range(nmax):
#         ddos1[i] += ddos2[i]
#         ddos[i] += ddos1[i] * dt / 6

def rk4(ddos, ddos1, ddos2, ddos3, keys, lmax, bath_list, mode, system, \
        system_dip, pulse_system_func, coupling, coupling_dip, pulse_coupling_func, comb_list, nmax, dt, t):
    '''
    Runge-Kutta 4th order method
    '''
    system_t, coupling_t = generate_time(
        system, system_dip, pulse_system_func, coupling, coupling_dip, pulse_coupling_func, t)

    rem_cal(ddos1, ddos, keys, lmax, bath_list, mode, system_t,
            coupling_t, comb_list, nmax)

    for i in range(nmax):
        ddos3[i] = ddos[i] + ddos1[i] * dt / 2

    system_t, coupling_t = generate_time(
        system, system_dip, pulse_system_func, coupling, coupling_dip, pulse_coupling_func, t + dt / 2)

    rem_cal(ddos2, ddos3, keys, lmax, bath_list, mode, system_t,
            coupling_t, comb_list, nmax)
    for i in range(nmax):
        ddos1[i] += ddos2[i] * 2
        ddos3[i] = ddos[i] + ddos2[i] * dt / 2

    system_t, coupling_t = generate_time(
        system, system_dip, pulse_system_func, coupling, coupling_dip, pulse_coupling_func, t + dt / 2)

    rem_cal(ddos2, ddos3, keys, lmax, bath_list, mode, system_t,
            coupling_t, comb_list, nmax)

    for i in range(nmax):
        ddos1[i] += ddos2[i] * 2
        ddos3[i] = ddos[i] + ddos2[i] * dt

    system_t, coupling_t = generate_time(
        system, system_dip, pulse_system_func, coupling, coupling_dip, pulse_coupling_func, t + dt)

    rem_cal(ddos2, ddos3, keys, lmax, bath_list, mode, system_t,
            coupling_t, comb_list, nmax)

    for i in range(nmax):
        ddos1[i] += ddos2[i]
        ddos[i] += ddos1[i] * dt / 6


def rk4_by_index(ddos, ddos1, ddos2, ddos3, keys, key_to_index, bath_list, mode, system,
                 system_dip, pulse_system_func, coupling, coupling_dip, pulse_coupling_func,
                 nmax, dt, t):
    '''
    Runge-Kutta 4th order method for arbitrary hierarchy key sets.
    '''
    system_t, coupling_t = generate_time(
        system, system_dip, pulse_system_func, coupling, coupling_dip, pulse_coupling_func, t)

    rem_cal_by_index(ddos1, ddos, keys, key_to_index, bath_list, mode, system_t,
                     coupling_t, nmax)

    for i in range(nmax):
        ddos3[i] = ddos[i] + ddos1[i] * dt / 2

    system_t, coupling_t = generate_time(
        system, system_dip, pulse_system_func, coupling, coupling_dip, pulse_coupling_func, t + dt / 2)

    rem_cal_by_index(ddos2, ddos3, keys, key_to_index, bath_list, mode, system_t,
                     coupling_t, nmax)
    for i in range(nmax):
        ddos1[i] += ddos2[i] * 2
        ddos3[i] = ddos[i] + ddos2[i] * dt / 2

    system_t, coupling_t = generate_time(
        system, system_dip, pulse_system_func, coupling, coupling_dip, pulse_coupling_func, t + dt / 2)

    rem_cal_by_index(ddos2, ddos3, keys, key_to_index, bath_list, mode, system_t,
                     coupling_t, nmax)

    for i in range(nmax):
        ddos1[i] += ddos2[i] * 2
        ddos3[i] = ddos[i] + ddos2[i] * dt

    system_t, coupling_t = generate_time(
        system, system_dip, pulse_system_func, coupling, coupling_dip, pulse_coupling_func, t + dt)

    rem_cal_by_index(ddos2, ddos3, keys, key_to_index, bath_list, mode, system_t,
                     coupling_t, nmax)

    for i in range(nmax):
        ddos1[i] += ddos2[i]
        ddos[i] += ddos1[i] * dt / 6


@njit
def gen_index2(iado, i, j, nsys):
    return iado * nsys * nsys + i * nsys + j


def allcator_H(H, dot_ddos, iado, i, j):
    nsys = len(H)
    for k in range(nsys):
        index1 = gen_index2(iado, i, k, nsys)
        index2 = gen_index2(iado, j, k, nsys)
        dot_ddos[index1, index2] -= 1.j * H[i, j]
        index1 = gen_index2(iado, k, j, nsys)
        index2 = gen_index2(iado, k, i, nsys)
        dot_ddos[index1, index2] += 1.j * H[i, j]


def allcator_H_l(H, dot_ddos, iado, i, j):
    nsys = len(H)
    for k in range(nsys):
        index1 = gen_index2(iado, i, k, nsys)
        index2 = gen_index2(iado, j, k, nsys)
        dot_ddos[index1, index2] += H[i, j]


def allcator_H_r(H, dot_ddos, iado, i, j):
    nsys = len(H)
    for k in range(nsys):
        index1 = gen_index2(iado, k, j, nsys)
        index2 = gen_index2(iado, k, i, nsys)
        dot_ddos[index1, index2] += H[i, j]


def allcator_Q_m(Q, dot_ddos, etaa, etal, etar, n, iado, pos, i, j):
    nsys = len(Q)
    for k in range(nsys):
        index1 = gen_index2(iado, i, k, nsys)
        index2 = gen_index2(pos, j, k, nsys)
        dot_ddos[index1, index2] -= 1.j * np.sqrt(n) \
            / np.sqrt(etaa) * etal * Q[i, j]
        index1 = gen_index2(iado, k, j, nsys)
        index2 = gen_index2(pos, k, i, nsys)
        dot_ddos[index1, index2] += 1.j * np.sqrt(n) \
            / np.sqrt(etaa) * etar * Q[i, j]


def allcator_Q_p(Q, dot_ddos, etaa, n, iado, pos, i, j):
    nsys = len(Q)
    for k in range(nsys):
        index1 = gen_index2(iado, i, k, nsys)
        index2 = gen_index2(pos, j, k, nsys)
        dot_ddos[index1, index2] -= 1.j * np.sqrt(n + 1) \
            * np.sqrt(etaa) * Q[i, j]
        index1 = gen_index2(iado, k, j, nsys)
        index2 = gen_index2(pos, k, i, nsys)
        dot_ddos[index1, index2] += 1.j * np.sqrt(n + 1) \
            * np.sqrt(etaa) * Q[i, j]


def actions_element(actions, A, iado, lcr='l'):
    '''
    generate the dot_ddos for the iado-th element
    '''
    nsys = len(A)

    for i in range(nsys):
        for j in range(nsys):
            if np.abs(A[i, j]) > 1e-10:
                if (lcr == 'l') or (lcr == 'c'):
                    allcator_H_l(A, actions, iado, i, j)
                if (lcr == 'r') or (lcr == 'c'):
                    allcator_H_r(A, actions, iado, i, j)


def generate_actions(actions, A, nmax, lcr='l'):
    r'''
    calculate the actions, \mathcal{A}, for the whole system
    '''
    for iado in range(nmax):
        actions_element(actions, A, iado, lcr)


class Bath():
    '''
    bath class for DEOM
    '''

    @classmethod
    def from_exponential_terms(cls, expn, etal, etar=None, etaa=None, mode=None):
        r"""Build a bath directly from decomposed exponential terms.

        This is useful when the correlation kernel has already been fitted or
        chosen in the form

        $$
        C(t) = \sum_k \eta_k e^{-\gamma_k t}.
        $$

        The arrays are stored in the same DEOM normalization used by the
        existing Padé decomposition path.  If ``etar`` is omitted it is taken as
        ``conj(etal)``; if ``etaa`` is omitted it is set to
        ``sqrt(abs(etal) * abs(etar))``.
        """
        obj = cls.__new__(cls)
        obj.bath = None
        obj.w_sp = None
        obj.beta = None
        obj.npsd = None

        obj.expn = np.atleast_1d(np.asarray(expn, dtype=np.complex128))
        obj.etal = np.atleast_1d(np.asarray(etal, dtype=np.complex128))
        if obj.expn.shape != obj.etal.shape:
            raise ValueError("expn and etal must have the same shape.")

        if etar is None:
            obj.etar = np.conjugate(obj.etal)
        else:
            obj.etar = np.atleast_1d(np.asarray(etar, dtype=np.complex128))
        if obj.etar.shape != obj.etal.shape:
            raise ValueError("etar must have the same shape as etal.")

        if etaa is None:
            obj.etaa = np.sqrt(np.abs(obj.etal) * np.abs(obj.etar)).astype(np.complex128)
        else:
            obj.etaa = np.atleast_1d(np.asarray(etaa, dtype=np.complex128))
        if obj.etaa.shape != obj.etal.shape:
            raise ValueError("etaa must have the same shape as etal.")
        if np.any(np.abs(obj.etaa) <= 1.0e-15):
            raise ValueError("all exponential terms need nonzero etaa in the current HEOM normalization.")

        if mode is None:
            obj.mode = np.zeros(obj.expn.shape, dtype=np.int64)
        else:
            obj.mode = np.atleast_1d(np.asarray(mode, dtype=np.int64))
        if obj.mode.shape != obj.expn.shape:
            raise ValueError("mode must have the same shape as expn.")
        if np.any(obj.mode < 0):
            raise ValueError("mode indices must be non-negative.")
        return obj

    def __init__(self, spectrum_sp=None, w_sp=None, beta=None, npsd=None, mode=None, function=None):
        '''
        initialize the bath
        Input:
            spectrum_sp: the spectral density of the bath, a function of frequency. Must be a sympy expression.
            w_sp: the frequency of the bath, a sympy symbol.
            beta: the inverse temperature of the bath.
            npsd: the number of the poles of the spectral density.
        '''
        self.bath = spectrum_sp
        self.w_sp = w_sp
        self.beta = beta
        self.npsd = npsd
        self.mode = mode
        self.etal, self.etar, self.etaa, self.expn = [], [], [], []

        if function is None:
            function = [decompose_spectrum_pade]*len(self.bath)

        if ((type(self.bath) is list) and (type(self.beta) is list) and (type(self.npsd) is list)):
            if (len(self.bath) != len(self.beta)) or (len(self.bath) != len(self.npsd)):
                print("the length of bath, w_sp, beta, npsd is not equal!")
            for i in range(len(self.bath)):
                etal_p, etar_p, etaa_p, expn_p = function[i](
                    self.bath[i], self.w_sp, self.beta[i], self.npsd[i])
                self.etal = np.append(self.etal, etal_p)
                self.etar = np.append(self.etar, etar_p)
                self.etaa = np.append(self.etaa, etaa_p)
                self.expn = np.append(self.expn, expn_p)
            self.etal = np.array(self.etal, dtype=np.complex128)
            self.etar = np.array(self.etar, dtype=np.complex128)
            self.etaa = np.array(self.etaa, dtype=np.complex128)
            self.expn = np.array(self.expn, dtype=np.complex128)
            if self.mode is None:
                print("mode is not set!")
                raise ValueError
            if (len(self.mode) != len(self.expn)):
                print("the length of mode is not equal to the number of dissipatons!")
                raise ValueError
        else:
            self.etal, self.etar, self.etaa, self.expn = decompose_spectrum_pade(
                self.bath, self.w_sp, self.beta, self.npsd)
            self.mode = np.zeros_like(self.expn, dtype=np.int64)


@njit(fastmath=True)
def operator_action_ddos(operator, ddos, nmax):
    output_ddos = np.zeros_like(ddos)
    for i in range(nmax):
        output_ddos[i] = operator @ ddos[i]
    return output_ddos


class HEOM():
    '''
    General HEOM solver class using decomposed bath correlation functions.
    '''

    def __init__(self, system=None, bath=None, coupling=None, lmax=None,
                 system_dipole=None, coupling_dipole=None, pulse_system_func=None,
                 pulse_coupling_func=None, hierarchy_truncation="total",
                 hierarchy_weights=None, system_liouvillian=None):
        '''
        initialize the HEOM solver
        Input:
            system: the system hamiltonian, a numpy array.
            bath: the bath object.
            coupling: the coupling between the system and the bath, a matrix or list of matrices.
            lmax: the maximum hierarchy level.
            system_dipole: optional system dipole, defaults to zero.
            coupling_dipole: optional coupling dipole, defaults to zero for each coupling.
            pulse_system_func: the pulse function of the system, a function of time.
            pulse_coupling_func: the pulse function of the coupling, a function of time.
            hierarchy_truncation: 'total' for sum(n_k) <= lmax,
                'smolyak' for isotropic sum(bit_length(n_k)) <= lmax,
                or 'weighted-smolyak'/'gamma-smolyak' to derive anisotropic
                weights from bath exponent rates unless hierarchy_weights is given.
            hierarchy_weights: optional positive integer weights for Smolyak.
            system_liouvillian: optional Liouville-space base generator.  If
                omitted, the base block is ``-1j[system, .]``.
            mode: the mode of the solver, a numpy array.
        '''
        self.system = None if system is None else np.asarray(system, dtype=np.complex128)
        self.system_liouvillian = (
            None if system_liouvillian is None else np.asarray(system_liouvillian, dtype=np.complex128)
        )
        self.system_dipole = system_dipole
        self.coupling = _as_operator_list(coupling, "coupling")
        self.coupling_dipole = coupling_dipole
        self.pulse_system_func = (lambda t: 0.0) if pulse_system_func is None else pulse_system_func
        self.pulse_coupling_func = (lambda t: 0.0) if pulse_coupling_func is None else pulse_coupling_func
        self.lmax = lmax
        self.hierarchy_truncation = hierarchy_truncation
        self.hierarchy_weights = hierarchy_weights
        self.nsys = 0
        self.nmax = 1
        self.nexp = 0
        self.nmod = 0
        self.bath = bath
        self.comb_list = []
        self.key_to_index = None
        self.minus_index = None
        self.plus_index = None
        self._use_native_total = False
        self._native_keys = None
        self._native_minus_index = None
        self._native_plus_index = None
        self._native_bath_list = None
        self._native_mode = None
        self._native_workspace = None
        self._native_decay_rates = None
        self._native_edge_tables = None
        self.green_function = None
        self.Δ = None
        self.V = None
        self.V_inv = None
        self.actions = None
        self.hierarchy_liouvillian_matrix = None
        self.steady_state_hierarchy = None
        self.steady_state_residual = None
        self.steady_state_trace_error = None
        self.success = None
        self.message = ""
        self.nfev = 0
        self.n_steps = 0
        self.n_rejected = 0
        self.method = None
        self.threads = 1

    def set_hierarchy(self, lmax, truncation=None, weights=None):
        '''
        set the hierarchy level
        '''
        self.lmax = lmax
        if truncation is not None:
            self.hierarchy_truncation = truncation
        if weights is not None:
            self.hierarchy_weights = weights

    def set_system(self, system):
        '''
        set the system hamiltonian
        '''
        self.system = np.asarray(system, dtype=np.complex128)

    def set_system_liouvillian(self, system_liouvillian):
        """Set a Liouville-space base generator for the zero-tier block."""
        self.system_liouvillian = np.asarray(system_liouvillian, dtype=np.complex128)

    def set_system_dipole(self, system_dipole):
        '''
        set the system dipole
        '''
        self.system_dipole = np.asarray(system_dipole, dtype=np.complex128)

    def set_coupling(self, coupling):
        '''
        set the coupling between the system and the bath
        '''
        self.coupling = _as_operator_list(coupling, "coupling")

    def set_coupling_dipole(self, coupling_dipole):
        '''
        set the coupling dipole between the system and the bath
        '''
        self.coupling_dipole = _as_operator_list(coupling_dipole, "coupling_dipole")

    def set_pulse_system_func(self, pulse_system_func):
        '''
        set the pulse function of the system, a function of time.
        '''
        self.pulse_system_func = pulse_system_func

    def set_pulse_coupling_func(self, pulse_coupling_func):
        '''
        set the pulse function of the coupling, a function of time.
        '''
        self.pulse_coupling_func = pulse_coupling_func

    def check_(self):
        '''
        check if all the parameters are setted.
        '''
        if self.system is None:
            raise ValueError('System Hamiltonian is not set.')
        if self.bath is None:
            raise ValueError('Bath is not set.')
        if self.coupling is None:
            raise ValueError('system bath interaction operator is not set.')

        self.system = np.asarray(self.system, dtype=np.complex128)
        if self.system.ndim != 2 or self.system.shape[0] != self.system.shape[1]:
            raise ValueError("System Hamiltonian must be a square matrix.")
        self.nsys = self.system.shape[0]
        if self.system_liouvillian is not None:
            self.system_liouvillian = np.asarray(self.system_liouvillian, dtype=np.complex128)
            if self.system_liouvillian.shape != (self.nsys * self.nsys, self.nsys * self.nsys):
                raise ValueError("system_liouvillian must have shape (nsys**2, nsys**2).")

        self.coupling = _as_operator_list(self.coupling, "coupling")
        for operator in self.coupling:
            if operator.shape != (self.nsys, self.nsys):
                raise ValueError("Each coupling operator must match the system shape.")

        if self.system_dipole is None:
            self.system_dipole = np.zeros_like(self.system)
        else:
            self.system_dipole = np.asarray(self.system_dipole, dtype=np.complex128)
            if self.system_dipole.shape != self.system.shape:
                raise ValueError("system_dipole must match the system shape.")

        if self.coupling_dipole is None:
            self.coupling_dipole = [np.zeros_like(operator) for operator in self.coupling]
        else:
            self.coupling_dipole = _as_operator_list(self.coupling_dipole, "coupling_dipole")
            if len(self.coupling_dipole) != len(self.coupling):
                raise ValueError("coupling_dipole must have one operator per coupling operator.")
            for operator in self.coupling_dipole:
                if operator.shape != (self.nsys, self.nsys):
                    raise ValueError("Each coupling_dipole operator must match the system shape.")

        self.nexp = len(self.bath.expn)
        self.nmod = np.max(self.bath.mode) + 1
        if len(self.coupling) < self.nmod:
            raise ValueError("Not enough coupling operators for the bath mode indices.")

    def init_(self):
        truncation = self.hierarchy_truncation.lower().replace("_", "-")
        if truncation in ("smolyak", "weighted-smolyak", "gamma-smolyak"):
            if truncation in ("weighted-smolyak", "gamma-smolyak") and self.hierarchy_weights is None:
                weights = gamma_smolyak_weights(self.bath.expn)
            elif self.hierarchy_weights is None:
                weights = np.ones(self.nexp, dtype=np.int64)
            elif isinstance(self.hierarchy_weights, str):
                weights_key = self.hierarchy_weights.lower().replace("_", "-")
                if weights_key not in ("gamma", "gammas"):
                    raise ValueError("String hierarchy_weights must be 'gamma'.")
                weights = gamma_smolyak_weights(self.bath.expn)
            else:
                weights = _validate_smolyak_weights(self.hierarchy_weights, self.nexp)
            self.hierarchy_weights = weights
            self.keys = generate_smolyak_keys(self.lmax, self.nexp, weights)
            self.nmax = len(self.keys)
            self.key_to_index = {tuple(key): i for i, key in enumerate(self.keys)}
            self.minus_index = None
            self.plus_index = None
            self._use_native_total = False
            self._native_keys = None
            self._native_minus_index = None
            self._native_plus_index = None
            self._native_bath_list = None
            self._native_mode = None
            self._native_workspace = None
            self._native_decay_rates = None
            self._native_edge_tables = None
            self.comb_list = None
            return
        if truncation not in ("total", "simplex"):
            raise ValueError(
                "hierarchy_truncation must be 'total', 'smolyak', "
                "'weighted-smolyak', or 'gamma-smolyak'."
            )

        # note the maximum number of int64 is  2, 147, 483, 647 which is nearly comb(40, 10)
        combmax = self.nexp + self.lmax + 1
        self.comb_list = np.zeros((combmax, combmax), dtype=np.int64)
        for j in range(combmax):
            self.comb_list[0, j] = 0
        self.comb_list[0, 0] = 1
        for i in range(1, combmax):
            for j in range(1, combmax):
                self.comb_list[i, j] = self.comb_list[i - 1, j] \
                    + self.comb_list[i - 1, j - 1]
            self.comb_list[i, 0] = 1
        self.nmax = self.comb_list[self.lmax+self.nexp, self.lmax]

        self.key_to_index = None
        if _get_heom_cpp() is not None:
            self.keys = generate_total_keys(self.lmax, self.nexp)
            self.nmax = len(self.keys)
            self.minus_index, self.plus_index = build_neighbor_indices(self.keys)
            self._use_native_total = True
            self._native_keys = np.ascontiguousarray(self.keys, dtype=np.int64)
            self._native_minus_index = np.ascontiguousarray(self.minus_index, dtype=np.int64)
            self._native_plus_index = np.ascontiguousarray(self.plus_index, dtype=np.int64)
            self._native_bath_list = (
                np.ascontiguousarray(self.bath.expn, dtype=np.complex128),
                np.ascontiguousarray(self.bath.etal, dtype=np.complex128),
                np.ascontiguousarray(self.bath.etar, dtype=np.complex128),
                np.ascontiguousarray(self.bath.etaa, dtype=np.complex128),
            )
            self._native_mode = np.ascontiguousarray(self.bath.mode, dtype=np.int64)
            self._native_decay_rates = np.ascontiguousarray(
                self._native_keys @ self._native_bath_list[0],
                dtype=np.complex128,
            )
            self._native_edge_tables = build_native_edge_tables(
                self._native_keys,
                self._native_minus_index,
                self._native_plus_index,
                self._native_bath_list,
                self._native_mode,
            )
        else:
            # The Python total-hierarchy kernels use hash positions tied to
            # gen_keys ordering, so keep that path unchanged.
            self.keys = np.zeros((self.nmax, self.nexp), dtype=np.int64)
            gen_keys(self.keys, self.lmax, self.nexp, self.comb_list)
            self.minus_index = None
            self.plus_index = None
            self._use_native_total = False
            self._native_keys = None
            self._native_minus_index = None
            self._native_plus_index = None
            self._native_bath_list = None
            self._native_mode = None
            self._native_workspace = None
            self._native_decay_rates = None
            self._native_edge_tables = None

    def rk4(self, dt, t):
        '''
        the Runge-Kutta wrapper
        '''
        if self._use_native_total:
            return rk4_native_total(self.ddos, self._native_keys, self._native_minus_index, self._native_plus_index, self._native_bath_list, self._native_mode, self.system, self.system_dipole, self.pulse_system_func, self.coupling, self.coupling_dipole, self.pulse_coupling_func, dt, t, self._native_workspace, self._native_decay_rates)
        if self.key_to_index is not None:
            return rk4_by_index(self.ddos, self.ddos1, self.ddos2, self.ddos3, self.keys, self.key_to_index, [self.bath.expn, self.bath.etal, self.bath.etar, self.bath.etaa], self.bath.mode, self.system, self.system_dipole, self.pulse_system_func, self.coupling, self.coupling_dipole, self.pulse_coupling_func, self.nmax, dt, t)
        return rk4(self.ddos, self.ddos1, self.ddos2, self.ddos3, self.keys, self.lmax, [self.bath.expn, self.bath.etal, self.bath.etar, self.bath.etaa],  self.bath.mode, self.system, self.system_dipole, self.pulse_system_func, self.coupling, self.coupling_dipole, self.pulse_coupling_func, self.comb_list, self.nmax, dt, t)

    def _native_time_independent_operators(self, dt, t_final):
        system_0, coupling_0 = generate_time_dense(
            self.system,
            self.system_dipole,
            self.pulse_system_func,
            self.coupling,
            self.coupling_dipole,
            self.pulse_coupling_func,
            0.0,
        )
        for sample_t in (dt, t_final):
            system_t, coupling_t = generate_time_dense(
                self.system,
                self.system_dipole,
                self.pulse_system_func,
                self.coupling,
                self.coupling_dipole,
                self.pulse_coupling_func,
                sample_t,
            )
            if not (np.allclose(system_t, system_0) and np.allclose(coupling_t, coupling_0)):
                raise ValueError("method='krylov' requires time-independent system and coupling operators")
        return system_0, coupling_0

    def _run_native_dop853(self, rho0, dt, nt, p1, rtol, atol, threads,
                           t_eval=None, t_span=None):
        self.ddos[0] = rho0
        heom_cpp = _get_heom_cpp()
        if heom_cpp is None or not hasattr(heom_cpp, "dop853_by_index"):
            raise RuntimeError("the compiled HEOM backend does not provide native DOP853")

        adaptive_output = t_eval is None and t_span is not None
        if adaptive_output:
            if not hasattr(heom_cpp, "dop853_adaptive_by_index"):
                raise RuntimeError("the compiled HEOM backend does not provide adaptive-step DOP853 output")
            t_span = np.asarray(t_span, dtype=np.float64)
            if t_span.shape != (2,) or not np.all(np.isfinite(t_span)) or t_span[1] <= t_span[0]:
                raise ValueError("t_span must be a finite increasing pair (t0, tf)")
        else:
            if t_eval is None:
                if dt is None or nt is None:
                    raise ValueError("method='dop853' requires t_span, t_eval, or both dt and nt")
                nt = int(nt)
                if nt < 0:
                    raise ValueError("nt must be non-negative")
                dt = float(dt)
                if not np.isfinite(dt) or dt < 0.0:
                    raise ValueError("dt must be a finite non-negative value")
                t_save = np.linspace(0.0, nt * dt, nt + 1)
            else:
                t_save = np.asarray(t_eval, dtype=np.float64)
                if t_save.ndim != 1 or len(t_save) == 0:
                    raise ValueError("t_eval must be a non-empty one-dimensional array")
                if not np.all(np.isfinite(t_save)) or np.any(np.diff(t_save) <= 0.0):
                    raise ValueError("t_eval must be finite and strictly increasing")

        system_dipole = np.ascontiguousarray(self.system_dipole, dtype=np.complex128)
        coupling = np.ascontiguousarray(self.coupling, dtype=np.complex128)
        coupling_dipole = np.ascontiguousarray(self.coupling_dipole, dtype=np.complex128)
        pulse_system = self.pulse_system_func if np.any(system_dipole) else None
        pulse_coupling = self.pulse_coupling_func if np.any(coupling_dipole) else None
        expn, etal, etar, etaa = self._native_bath_list

        args = [
            self.ddos,
            self._native_keys,
            self._native_minus_index,
            self._native_plus_index,
            expn,
            etal,
            etar,
            etaa,
            self._native_mode,
            np.ascontiguousarray(self.system, dtype=np.complex128),
            system_dipole,
            coupling,
            coupling_dipole,
            pulse_system,
            pulse_coupling,
        ]
        if adaptive_output:
            t_save, rho_t, nfev, n_steps, n_rejected = heom_cpp.dop853_adaptive_by_index(
                *args,
                float(t_span[0]),
                float(t_span[1]),
                rtol,
                atol,
                threads,
                *self._native_edge_tables,
            )
        else:
            rho_t, nfev, n_steps, n_rejected = heom_cpp.dop853_by_index(
                *args,
                t_save,
                rtol,
                atol,
                threads,
                *self._native_edge_tables,
            )
        self.success = True
        self.message = "success"
        self.nfev = int(nfev)
        self.n_steps = int(n_steps)
        self.n_rejected = int(n_rejected)
        if p1 is None:
            ddos_save = rho_t
        else:
            p1 = np.asarray(p1, dtype=np.complex128)
            ddos_save = np.einsum("ij,tji->t", p1, rho_t)
        return t_save, ddos_save

    def _run_native_krylov(self, rho0, dt, nt, p1, krylov_dim, krylov_tol, progress=False):
        self.ddos[0] = rho0
        t_save = np.linspace(0.0, nt * dt, nt + 1)
        if p1 is not None:
            p1 = np.asarray(p1, dtype=np.complex128)

        system_t, coupling_t = self._native_time_independent_operators(dt, nt * dt)
        shape = (self.nmax, self.nsys, self.nsys)

        def matvec(y):
            ddos = np.ascontiguousarray(y.reshape(shape))
            return native_rhs_total(
                ddos,
                self._native_keys,
                self._native_minus_index,
                self._native_plus_index,
                self._native_bath_list,
                self._native_mode,
                system_t,
                coupling_t,
            ).reshape(-1)

        y = np.ascontiguousarray(self.ddos.reshape(-1))

        def observe(_t, state):
            rho = state.reshape(shape)[0]
            if p1 is None:
                return rho.copy()
            return (p1 @ rho).trace()

        sol = krylov(
            matvec,
            y,
            t_save,
            krylov_dim=krylov_dim,
            tol=krylov_tol,
            observer=observe,
            save=False,
            progress=tqdm if progress else None,
        )
        self.ddos[:] = sol.y_final.reshape(shape)
        self.success = sol.success
        self.message = sol.message
        self.nfev = sol.nfev
        self.n_steps = nt
        self.n_rejected = 0
        if p1 is None:
            ddos_save = sol.observations
        else:
            ddos_save = np.asarray(sol.observations, dtype=np.complex128)
        return t_save, ddos_save

    def run(self, rho0, dt=None, nt=None, p1=None, method="dop853", rtol=1.0e-8,
            atol=1.0e-10, krylov_dim=30, krylov_tol=1.0e-12,
            progress=False, threads=1, t_eval=None, t_span=None):
        """Propagate the hierarchy and return times and reduced-state data.

        ``method="dop853"`` uses native adaptive DOP853 with the precomputed
        edge-table RHS.  Use ``t_span=(t0, tf)`` for accepted adaptive-step
        output, ``t_eval`` for requested output times, or ``dt`` and ``nt`` as
        a convenience fixed output grid.  RK4 uses ``dt`` as the actual fixed
        step and requires ``nt``. ``threads`` controls the native DOP853 RHS
        only; use ``threads=0`` to request all hardware threads.
        """
        self.check_()
        self.init_()
        threads = 1 if threads is None else int(threads)
        if threads < 0:
            raise ValueError("threads must be non-negative")
        if threads == 0:
            threads = os.cpu_count() or 1
        self.threads = threads

        method_key = normalize_method(method)
        if method_key not in ("rk4", "dop853", "krylov"):
            raise ValueError("method must be 'rk4', 'dop853', or 'krylov'")
        if method_key != "rk4" and not self._use_native_total:
            raise NotImplementedError(
                "method='dop853' and method='krylov' require the native total hierarchy"
            )
        self.method = method_key

        # allocate memory for ddos
        if self._use_native_total:
            self.ddos = np.zeros((self.nmax, self.nsys, self.nsys), dtype=np.complex128)
            self.ddos1 = None
            self.ddos2 = None
            self.ddos3 = None
            self._native_workspace = (
                tuple(np.empty_like(self.ddos) for _ in range(5))
                if method_key == "rk4"
                else None
            )
        elif self.key_to_index is None:
            self.ddos = [coo_matrix((self.nsys, self.nsys,),
                                    dtype=np.complex128)] * self.nmax
            self.ddos1 = [coo_matrix((self.nsys, self.nsys,),
                                     dtype=np.complex128)] * self.nmax
            self.ddos2 = [coo_matrix((self.nsys, self.nsys,),
                                     dtype=np.complex128)] * self.nmax
            self.ddos3 = [coo_matrix((self.nsys, self.nsys,),
                                     dtype=np.complex128)] * self.nmax
        else:
            self.ddos = [np.zeros((self.nsys, self.nsys), dtype=np.complex128)
                         for _ in range(self.nmax)]
            self.ddos1 = [np.zeros((self.nsys, self.nsys), dtype=np.complex128)
                          for _ in range(self.nmax)]
            self.ddos2 = [np.zeros((self.nsys, self.nsys), dtype=np.complex128)
                          for _ in range(self.nmax)]
            self.ddos3 = [np.zeros((self.nsys, self.nsys), dtype=np.complex128)
                          for _ in range(self.nmax)]

        if method_key != "rk4":
            if method_key == "dop853":
                return self._run_native_dop853(
                    rho0, dt, nt, p1, rtol, atol, threads,
                    t_eval=t_eval, t_span=t_span,
                )
            if method_key == "krylov":
                if dt is None or nt is None:
                    raise ValueError("method='krylov' requires dt and nt")
                return self._run_native_krylov(rho0, dt, nt, p1, krylov_dim, krylov_tol, progress=progress)

        if dt is None or nt is None:
            raise ValueError("method='rk4' requires dt and nt")
        nt = int(nt)
        if nt < 0:
            raise ValueError("nt must be non-negative")
        dt = float(dt)
        if not np.isfinite(dt) or dt < 0.0:
            raise ValueError("dt must be a finite non-negative value")
        self.ddos[0] = rho0

        t_save = np.zeros(nt + 1, dtype=np.float64)
        if p1 is None and self._use_native_total:
            ddos_save = [np.zeros((self.nsys, self.nsys), dtype=np.complex128)
                         for _ in range(nt + 1)]
        elif p1 is None:
            ddos_save = [coo_matrix((self.nsys, self.nsys,),
                                    dtype=np.complex128)] * (nt + 1)
        else:
            p1 = np.asarray(p1, dtype=np.complex128)
            ddos_save = np.zeros(nt + 1, dtype=np.complex128)

        t_save[0] = 0
        if p1 is None:
            ddos_save[0] = self.ddos[0].copy()
        else:
            ddos_save[0] = (p1 @ self.ddos[0]).trace()

        steps = tqdm(range(nt)) if progress else range(nt)
        for i in steps:
            self.rk4(dt, i * dt)
            t_save[i + 1] = (i + 1) * dt
            if p1 is None:
                ddos_save[i + 1] = self.ddos[0].copy()
            else:
                ddos_save[i + 1] = (p1 @ self.ddos[0]).trace()
        self.success = True
        self.message = "success"
        self.nfev = 4 * nt
        self.n_steps = nt
        self.n_rejected = 0
        return t_save, ddos_save

    def hierarchy_liouvillian(self):
        """Return a dense HEOM hierarchy generator.

        The default diagonal block is the Hamiltonian Liouvillian
        ``-1j[H, .]``.  If ``self.system_liouvillian`` is set, that stored
        Liouville-space generator is used as the zero-tier/base block.
        """
        self.check_()
        self.init_()

        nsys = int(self.nsys)
        nsys2 = nsys * nsys
        eye_sys = np.eye(nsys, dtype=np.complex128)
        eye_liouville = np.eye(nsys2, dtype=np.complex128)

        if self.system_liouvillian is None:
            base = np.kron(self.system, eye_sys) - np.kron(eye_sys, self.system.T)
            base = -1.j * base
        else:
            base = self.system_liouvillian

        keys = np.asarray(self.keys, dtype=np.int64)
        key_to_index = {tuple(key): index for index, key in enumerate(keys)}
        expn = np.asarray(self.bath.expn, dtype=np.complex128)
        etal = np.asarray(self.bath.etal, dtype=np.complex128)
        etar = np.asarray(self.bath.etar, dtype=np.complex128)
        etaa = np.asarray(self.bath.etaa, dtype=np.complex128)
        mode = np.asarray(self.bath.mode, dtype=np.int64)

        coupling_left = []
        coupling_right = []
        for operator in self.coupling:
            q = np.asarray(operator, dtype=np.complex128)
            coupling_left.append(np.kron(q, eye_sys))
            coupling_right.append(np.kron(eye_sys, q.T))

        size = int(self.nmax) * nsys2
        generator = np.zeros((size, size), dtype=np.complex128)

        def block(index):
            start = int(index) * nsys2
            return slice(start, start + nsys2)

        for iado, key in enumerate(keys):
            row = block(iado)
            generator[row, row] += base - np.sum(key * expn) * eye_liouville

            for mp in range(len(expn)):
                n = int(key[mp])
                m = int(mode[mp])
                coupling_commutator = coupling_left[m] - coupling_right[m]

                if n > 0:
                    lower_key = key.copy()
                    lower_key[mp] -= 1
                    pos = key_to_index.get(tuple(lower_key))
                    if pos is not None:
                        lower = etal[mp] * coupling_left[m] - etar[mp] * coupling_right[m]
                        generator[row, block(pos)] += -1.j * np.sqrt(n) / np.sqrt(etaa[mp]) * lower

                upper_key = key.copy()
                upper_key[mp] += 1
                pos = key_to_index.get(tuple(upper_key))
                if pos is not None:
                    generator[row, block(pos)] += (
                        -1.j * np.sqrt(n + 1) * np.sqrt(etaa[mp]) * coupling_commutator
                    )

        self.hierarchy_liouvillian_matrix = generator
        return generator

    def steady_state(
        self,
        *,
        trace=1.0,
        method="replace",
        replace_row=0,
        rcond=None,
        return_vector=False,
    ):
        r"""Return the normalized stationary HEOM hierarchy.

        The stationary hierarchy solves

        $$
        \mathcal{L}_{\rm HEOM}\varrho_\ast = 0,\qquad
        \mathrm{Tr}\,\varrho_{\ast,0} = \mathrm{trace}.
        $$

        The default method replaces one row of the dense hierarchy Liouvillian
        by the zero-tier trace constraint and solves the resulting linear
        system.  ``method="lstsq"`` solves the overdetermined system
        ``[L; trace_row] rho = [0; trace]``.
        """
        liouvillian = self.hierarchy_liouvillian()
        dim = int(self.nsys)
        dim2 = dim * dim
        size = liouvillian.shape[0]
        trace_row = np.zeros(size, dtype=np.complex128)
        trace_row[:dim2] = np.eye(dim, dtype=np.complex128).reshape(-1)
        rhs_trace = complex(trace)
        method_key = method.lower().replace("-", "_")

        if method_key in ("replace", "row_replace", "trace_replace"):
            row = int(replace_row)
            if row < 0 or row >= size:
                raise ValueError("replace_row must select a row of the hierarchy Liouvillian.")
            matrix = np.array(liouvillian, copy=True)
            rhs = np.zeros(size, dtype=np.complex128)
            matrix[row, :] = trace_row
            rhs[row] = rhs_trace
            vector = np.linalg.solve(matrix, rhs)
        elif method_key in ("lstsq", "least_squares", "least-squares"):
            matrix = np.vstack([liouvillian, trace_row])
            rhs = np.zeros(size + 1, dtype=np.complex128)
            rhs[-1] = rhs_trace
            vector = np.linalg.lstsq(matrix, rhs, rcond=rcond)[0]
        elif method_key in ("eig", "eigen", "null"):
            values, vectors = la.eig(liouvillian)
            index = int(np.argmin(np.abs(values)))
            vector = vectors[:, index]
            norm = trace_row @ vector
            if abs(norm) <= 1.0e-14:
                raise FloatingPointError("stationary null vector has nearly zero zero-tier trace.")
            vector = vector * (rhs_trace / norm)
        else:
            raise ValueError("method must be 'replace', 'lstsq', or 'eig'.")

        trace_value = trace_row @ vector
        if abs(trace_value) <= 1.0e-14:
            raise FloatingPointError("steady-state hierarchy has nearly zero zero-tier trace.")
        vector = vector * (rhs_trace / trace_value)
        hierarchy = vector.reshape((int(self.nmax), dim, dim))

        self.steady_state_hierarchy = hierarchy
        self.steady_state_residual = float(np.linalg.norm(liouvillian @ vector))
        self.steady_state_trace_error = float(abs((trace_row @ vector) - rhs_trace))
        self.success = True
        self.message = "success"
        return vector if return_vector else hierarchy

    def hierarchy_contract(
        self,
        initial_hierarchy,
        final_operator,
        distances,
        *,
        method="eig",
    ):
        r"""Contract a full-hierarchy HEOM two-point object.

        This evaluates

        $$
        \mathrm{Tr}_0[O_f e^{x\mathcal{L}_{\rm HEOM}}\varrho(0)]
        $$

        where ``initial_hierarchy`` may be shaped as ``(nado, nsys, nsys)`` or
        as a flattened hierarchy vector.
        """
        liouvillian = self.hierarchy_liouvillian()
        dim = int(self.nsys)
        dim2 = dim * dim
        initial = np.asarray(initial_hierarchy, dtype=np.complex128)
        if initial.shape == (int(self.nmax), dim, dim):
            vector0 = initial.reshape(-1)
        elif initial.shape == (liouvillian.shape[0],):
            vector0 = initial
        else:
            raise ValueError("initial_hierarchy must have shape (nado, nsys, nsys) or a flat hierarchy shape.")

        final = np.asarray(final_operator, dtype=np.complex128)
        if final.shape != (dim, dim):
            raise ValueError("final_operator must have shape (nsys, nsys).")

        trace_row = np.zeros(liouvillian.shape[0], dtype=np.complex128)
        trace_row[:dim2] = final.T.reshape(-1)
        xlist, scalar = _distance_array(distances)
        values = []
        method_key = method

        if method_key == "eig":
            evals, evecs = la.eig(liouvillian)
            try:
                coeff = np.linalg.solve(evecs, vector0)
            except np.linalg.LinAlgError:
                method_key = "expm"
            else:
                for distance in xlist:
                    values.append(trace_row @ (evecs @ (np.exp(evals * float(distance)) * coeff)))
        if method_key == "expm":
            for distance in xlist:
                values.append(trace_row @ (la.expm(liouvillian * float(distance)) @ vector0))
        elif method_key != "eig":
            raise ValueError("method must be 'eig' or 'expm'.")

        out = np.real_if_close(np.asarray(values))
        return out[0] if scalar else out

    def zero_tier_contract(
        self,
        initial_matrix,
        final_operator,
        distances,
        *,
        method="eig",
    ):
        r"""Contract a zero-tier HEOM two-point object.

        This evaluates

        $$
        \mathrm{Tr}_0[O_f e^{x\mathcal{L}_{\rm HEOM}}(X_0)]
        $$

        with ``X_0`` embedded in the zero ADO.
        """
        self.check_()
        self.init_()
        dim = int(self.nsys)
        initial = np.asarray(initial_matrix, dtype=np.complex128)
        final = np.asarray(final_operator, dtype=np.complex128)
        if initial.shape != (dim, dim):
            raise ValueError("initial_matrix must have shape (nsys, nsys).")
        if final.shape != (dim, dim):
            raise ValueError("final_operator must have shape (nsys, nsys).")

        hierarchy = np.zeros((int(self.nmax), dim, dim), dtype=np.complex128)
        hierarchy[0] = initial
        return self.hierarchy_contract(
            hierarchy,
            final,
            distances,
            method=method,
        )

    def correlation_2p_1t(self, rho0, operators, distances, *, method="eig"):
        r"""Return the two-point HEOM correlation ``<A(x) B(0)>``.

        The operator convention follows the older QME helper:
        ``operators=[A, B]`` computes

        $$
        \mathrm{Tr}_0[A e^{x\mathcal{L}_{\rm HEOM}}(B\rho_0)].
        $$
        """
        if len(operators) != 2:
            raise ValueError("operators must be [A, B].")
        final_operator, initial_operator = operators
        self.check_()
        self.init_()
        dim = int(self.nsys)
        initial_operator = np.asarray(initial_operator, dtype=np.complex128)
        rho = np.asarray(rho0, dtype=np.complex128)
        if rho.shape == (dim, dim):
            return self.zero_tier_contract(
                initial_operator @ rho,
                final_operator,
                distances,
                method=method,
            )
        if rho.shape == (int(self.nmax), dim, dim):
            hierarchy = np.einsum("ij,njk->nik", initial_operator, rho)
        elif rho.shape == (int(self.nmax) * dim * dim,):
            hierarchy = np.einsum("ij,njk->nik", initial_operator, rho.reshape(int(self.nmax), dim, dim))
        else:
            raise ValueError("rho0 must have shape (nsys, nsys), (nado, nsys, nsys), or a flat hierarchy shape.")
        return self.hierarchy_contract(
            hierarchy,
            final_operator,
            distances,
            method=method,
        )

    def correlation_2op_1t(
        self,
        rho0,
        a_op,
        b_op,
        distances,
        *,
        method="eig",
    ):
        """Return ``<A(x)B(0)>`` using explicit ``A`` and ``B`` arguments."""
        return self.correlation_2p_1t(
            rho0,
            [a_op, b_op],
            distances,
            method=method,
        )

    def correlation_4op_3t(self, operator_a, operator_b, operator_c, operator_d, rho0, T, w_x, w_y, if_full=True, cut_off_min=0.5, cut_off_max=1.1, if_load=False, if_save=False, lcr='llll'):
        if self.hierarchy_liouvillian_matrix is None:
            print('hierarchy Liouvillian is not generated, generating now...')
            self.hierarchy_liouvillian()
            print('hierarchy Liouvillian is generated.')

        if if_load and os.path.exists('correlation_4op_3t.npz'):
            print('loading correlation_4op_3t from file...')
            data = np.load('correlation_4op_3t.npz')
            self.Δ = data['Δ']
            self.V = data['V']
            self.V_inv = data['V_inv']
            print('correlation_4op_3t is loaded.')

        if self.Δ is None:
            print('eigenvalues and eigenvectors are not generated, generating now...')
            self.Δ, self.V = la.eig(self.hierarchy_liouvillian_matrix)
            self.V_inv = la.pinv(self.V)
            print('eigenvalues and eigenvectors are generated.')

        if if_save:
            print('saving correlation_4op_3t to file...')
            np.savez('correlation_4op_3t.npz', Δ=self.Δ,
                     V=self.V, V_inv=self.V_inv)
            print('correlation_4op_3t is saved.')

        self.actions1 = np.zeros((self.nmax * self.nsys * self.nsys, self.nmax *
                                 self.nsys * self.nsys), dtype=np.complex128)
        self.actions2 = np.zeros((self.nmax * self.nsys * self.nsys, self.nmax *
                                 self.nsys * self.nsys), dtype=np.complex128)
        self.actions3 = np.zeros((self.nmax * self.nsys * self.nsys, self.nmax *
                                 self.nsys * self.nsys), dtype=np.complex128)
        self.actions4 = np.zeros((self.nmax * self.nsys * self.nsys, self.nmax *
                                 self.nsys * self.nsys), dtype=np.complex128)

        generate_actions(self.actions1, operator_d, self.nmax, lcr=lcr[3])
        generate_actions(self.actions2, operator_c, self.nmax, lcr=lcr[2])
        generate_actions(self.actions3, operator_b, self.nmax, lcr=lcr[1])
        generate_actions(self.actions4, operator_a, self.nmax, lcr=lcr[0])

        rho = np.zeros((self.nmax * self.nsys * self.nsys, 1),
                       dtype=np.complex128)
        rho[:self.nsys*self.nsys, 0] = (rho0).flatten()

        if if_full:  # do not cut off the small eigenvalues
            print('calculating auxiliary matrix...')
            actions1_V = self.actions1 @ self.V
            V_actions2_V = self.V_inv @ self.actions2 @ self.V
            V_actions3_V = self.V_inv @ self.actions3 @ self.V
            V_actions4 = self.V_inv @ (self.actions4 @ rho)
            c_w = np.zeros((len(w_x), len(w_y)), dtype=np.complex128)
            V_actions2_V_G_V_actions3_V = V_actions2_V @ (
                np.diag(np.exp(self.Δ * T)) @ V_actions3_V)

            print('calculating correlation_4op_3t...')
            for i in tqdm(range(len(w_x))):
                for j in range(len(w_y)):
                    c_w[i, j] = np.trace(((actions1_V @ ((1 / (-self.Δ - 1.j * w_x[i])).reshape(len(self.Δ), 1) * (V_actions2_V_G_V_actions3_V @ (
                        (1 / (-self.Δ - 1.j * w_y[j])).reshape(len(self.Δ), 1) * V_actions4))))[:self.nsys*self.nsys, 0]).reshape(self.nsys, self.nsys))
        else:
            print('calculating auxiliary matrix...')
            cut_off_min = np.min(np.real(self.Δ)) * cut_off_min
            cut_off_max = np.max(np.real(self.Δ)) * cut_off_max
            V1 = self.V[:, (np.real(self.Δ) > cut_off_min) &
                        (np.real(self.Δ) < cut_off_max)]
            V1_inv = self.V_inv[(np.real(self.Δ) > cut_off_min) & (
                np.real(self.Δ) < cut_off_max), :]
            Δ1 = self.Δ[(np.real(self.Δ) > cut_off_min) &
                        (np.real(self.Δ) < cut_off_max)]

            print('calculating correlation_4op_3t...')
            actions1_V1 = self.actions1 @ V1
            V1_actions2_V1 = V1_inv @ self.actions2 @ V1
            V1_actions3_V1 = V1_inv @ self.actions3 @ V1
            V1_actions4 = V1_inv @ self.actions4 @ rho
            c_w = np.zeros((len(w_x), len(w_y)), dtype=np.complex128)
            V1_actions2_V1_G_V1_actions3_V1 = V1_actions2_V1 @ (
                np.diag(np.exp(Δ1 * T)) @ V1_actions3_V1)

            for i in tqdm(range(len(w_x))):
                for j in range(len(w_y)):
                    c_w[i, j] = np.trace(((actions1_V1 @ ((1 / (-Δ1 - 1.j * w_x[i])).reshape(len(Δ1), 1) * (V1_actions2_V1_G_V1_actions3_V1 @ (
                        (1 / (-Δ1 - 1.j * w_y[j])).reshape(len(Δ1), 1) * V1_actions4))))[:self.nsys*self.nsys, 0]).reshape(self.nsys, self.nsys))
        return c_w
