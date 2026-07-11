'''
disspaton equation of motion
Author: Zi-Hao Chen
Date: 2023/07/12
Email: czh5@mail.ustc.edu.cn
Only for bosonic bath
'''


from numba import njit
import numpy as np
import sympy as sp
import math
from tqdm import tqdm
from scipy.sparse import coo_matrix
from scipy import linalg as la
import os
# from cvxopt import solvers, matrix


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

    for i in [nind]:
        print(i)
        gamma = np.roots(Qp[:, i][::-1])
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
    if type(nind) is list:
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

    omega_new_temp = (la.inv(gamma_m.T @ gamma_m) @
                      gamma_m.T @ h_m).reshape(2, n_row)
    #
    omega_new = omega_new_temp[0, :] + 1.j*omega_new_temp[1, :]

    etal_p = omega_new
    expn_p = -t_new / scale
    return sort_symmetry(etal_p, expn_p)


def decompose_spectrum_prony(spe: sp.core.mul.Mul, w_sp: sp.core.symbol.Symbol, beta, nind: int or list, scale=250000, n=1250, npsd=10, bose_fermi=1):
    '''
    decompose the spectrum with prony method
    input
    spe: the spectrum, a sp.core.mul.Mul object (sympy expression)
    w_sp: the frequency symbol
    nind: int or list. If int, find gamma using the real part. If list then the number of real poles and imaginary poles
    '''
    etal_pade, _, _, expn_pade = decompose_spectrum_pade(
        spe, w_sp, beta, npsd, bose_fermi=bose_fermi)

    t = np.linspace(0, 1, 2 * n + 1)
    res_t = np.zeros(len(t), dtype=complex)
    fit_t(scale * t, res_t, expn_pade, etal_pade)

    print("check the sample points")
    print(res_t[:10])
    print(res_t[-10:])
    if type(nind) is list:
        if nind[0] == 'a':
            _, _, _, expn_real = decompose_spectrum_pade_real(spe, w_sp)
            gamma_real = np.exp(- expn_real * scale / (2*n))
            nind[0] = len(gamma_real)
            if bose_fermi == 1:
                print("For the bose case, C(t) have the analytical imag part")
                exit()
            return prony_fitting(res_t, t, nind, scale, n, gamma_real=gamma_real)
        elif nind[1] == 'a':
            _, _, _, expn_imag = decompose_spectrum_pade_imag(spe, w_sp)
            gamma_imag = np.exp(- expn_imag * scale / (2*n))
            nind[1] = len(gamma_imag)
            if bose_fermi == 2:
                print("For the fermi case, C(t) have the analytical real part")
                exit()
            return prony_fitting(res_t, t, nind, scale, n, gamma_imag=gamma_imag)
    return prony_fitting(res_t, t, nind, scale, n)


def single_oscillator(omega: float, w_sp: sp.core.symbol.Symbol, beta: float, nind: int):
    etal = np.array([1/(2 * (1-np.exp(-beta * omega))), -1 /
                    (2 * (1-np.exp(beta * omega)))], dtype=complex)
    etar = np.array([-1/(2 * (1-np.exp(beta * omega))), 1 /
                    (2 * (1-np.exp(-beta * omega)))], dtype=complex)
    etaa = np.sqrt(np.abs(etal + etar))
    expn = np.array([1.j * omega, -1.j * omega])
    return etal, etar, etaa, expn


@njit
def gen_hash_value(key, nind, comb_list):
    '''
    generate the hash value for the key
    '''
    sum_ = 0
    hash_value = 0
    for i in range(nind):
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
def hash_plus(key, pos, nind, comb_list):
    '''
    generate the hash value for the \rho_{{n}^{+}_{k}}
    '''
    key1 = key.copy()
    key1[pos] += 1
    return gen_hash_value(key1, nind, comb_list)


@njit
def hash_minus(key, pos, nind, comb_list):
    '''
    generate the hash value for the \rho_{{n}^{-}_{k}}
    '''
    key1 = key.copy()
    key1[pos] -= 1
    return gen_hash_value(key1, nind, comb_list)


@njit
def gen_keys_element(keys, iado, comb_list, nind, lmax):
    '''
    construct the keys for the iado-th element
    '''
    hash_val = gen_hash_value(keys[iado], nind, comb_list)

    for mp in range(nind):
        if (np.sum(keys[iado]) < lmax):
            hash_val = hash_plus(keys[iado], mp, nind, comb_list)
            keys[hash_val] = key_plus(keys[iado], mp)
        if (keys[iado, mp] > 0):
            hash_val = hash_minus(keys[iado], mp, nind, comb_list)
            keys[hash_val] = key_minus(keys[iado], mp)


@njit
def gen_keys(keys, lmax, nind, comb_list):
    '''
    construct the keys for the whole system
    '''
    nddo = 1
    nddo_bef = 0
    for ii in range(lmax + 1):
        for iado in range(nddo_bef, nddo):
            gen_keys_element(keys, iado, comb_list, nind, lmax)
        nddo_bef = nddo
        if (ii <= lmax):
            nddo = comb_list[nind + ii, nind]
        else:
            nddo = comb_list[nind + lmax, nind]


def generate_dot_element(dot_ddos, ddos, key, lmax, bath_list, mode, H, Q, comb_list, iado):
    '''
    generate the dot_ddos for the iado-th element
    '''
    # unpack coupling_t
    expn, etal, etar, etaa = bath_list
    nind = len(expn)
    dot_ddos[iado] = - np.sum(key * expn) * ddos[iado]
    dot_ddos[iado] -= 1.j * (H @ ddos[iado] - ddos[iado] @ H)

    for mp in range(nind):
        n = key[mp]
        if (n > 0):
            pos = hash_minus(key, mp, nind, comb_list)
            m = mode[mp]
            dot_ddos[iado] -= 1.j * np.sqrt(n) / np.sqrt(etaa[mp]) * (
                etal[mp] * Q[m] @ ddos[pos] - etar[mp] * ddos[pos] @ Q[m])

        if (sum(key) < lmax):
            pos = hash_plus(key, mp, nind, comb_list)
            m = mode[mp]
            dot_ddos[iado] -= 1.j * \
                np.sqrt(n + 1) * np.sqrt(etaa[mp]) * \
                (Q[m] @ ddos[pos] - ddos[pos] @ Q[m])


def rem_cal(dot_ddos, ddos, keys, lmax, bath_list, mode, system_t, coupling_t, comb_list, nmax):
    '''
    calculate the dot_ddos for the whole system
    '''
    for iado in range(nmax):
        generate_dot_element(
            dot_ddos, ddos, keys[iado], lmax, bath_list, mode, system_t, coupling_t, comb_list, iado)


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


def propgator_element(propgator, key, lmax, bath_list, mode, H, Q, comb_list, iado):
    '''
    generate the dot_ddos for the iado-th element
    '''
    # unpack coupling_t
    expn, etal, etar, etaa = bath_list
    nind = len(expn)
    nsys = len(H)
    for i in range(nsys):
        for j in range(nsys):
            index = gen_index2(iado, i, j, nsys)
            propgator[index, index] -= np.sum(key * expn)

    for i in range(nsys):
        for j in range(nsys):
            if np.abs(H[i, j]) > 1e-10:
                allcator_H(H, propgator, iado, i, j)

    for mp in range(nind):
        n = key[mp]
        m = mode[mp]
        # if np.abs(Q[m][i, j]) > 1e-10:
        if (n > 0):
            for i in range(nsys):
                for j in range(nsys):
                    pos = hash_minus(key, mp, nind, comb_list)
                    allcator_Q_m(Q[m], propgator, etaa[mp], etal[mp],
                                 etar[mp], n, iado, pos, i, j)
        if (sum(key) < lmax):
            for i in range(nsys):
                for j in range(nsys):
                    pos = hash_plus(key, mp, nind, comb_list)
                    allcator_Q_p(Q[m], propgator, etaa[mp],
                                 n, iado, pos, i, j)


def generate_propgator(propgator, keys, lmax, bath_list, mode, system_t, coupling_t, comb_list, nmax):
    '''
    calculate the liouville operator, i \mathcal{L}, for the whole system
    '''
    for iado in range(nmax):
        propgator_element(
            propgator, keys[iado], lmax, bath_list, mode, system_t, coupling_t, comb_list, iado)


def generate_actions(actions, A, nmax, lcr='l'):
    '''
    calculate the actions, \mathcal{A}, for the whole system
    '''
    for iado in range(nmax):
        actions_element(actions, A, iado, lcr)


class Bath():
    '''
    bath class for DEOM
    '''

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


class DEOMSolver():
    '''
    DEOM solver class
    '''

    def __init__(self, system=None, system_dipole=None, bath=None, coupling=None, coupling_dipole=None, pulse_system_func=None, pulse_coupling_func=None, lmax=None):
        '''
        initialize the DEOM solver
        Input:
            system: the system hamiltonian, a numpy array.
            system_dipole: the system dipole, a numpy array.
            bath: the bath, a DEOMBath object.
            coupling: the coupling between the system and the bath, a numpy array.
            coupling_dipole: the coupling dipole, a numpy array.
            pulse_system_func: the pulse function of the system, a function of time.
            pulse_coupling_func: the pulse function of the coupling, a function of time.
            lmax: the maximum hierarchy level.
            mode: the mode of the solver, a numpy array.
        '''
        self.system = system
        self.system_dipole = system_dipole
        self.coupling = coupling
        self.coupling_dipole = coupling_dipole
        self.pulse_system_func = pulse_system_func
        self.pulse_coupling_func = pulse_coupling_func
        self.lmax = lmax
        self.nsys = 0
        self.nmax = 1
        self.nind = 0
        self.nmod = 0
        self.bath = bath
        self.comb_list = []
        self.green_function = None
        self.Δ = None
        self.V = None
        self.V_inv = None
        self.actions = None
        self.propgator = None

    def set_hierarchy(self, lmax):
        '''
        set the hierarchy level
        '''
        self.lmax = lmax

    def set_system(self, system):
        '''
        set the system hamiltonian
        '''
        self.system = np.array(system, dtype=np.complex128)

    def set_system_dipole(self, system_dipole):
        '''
        set the system dipole
        '''
        self.system_dipole = np.array(system_dipole, dtype=np.complex128)

    def set_coupling(self, coupling):
        '''
        set the coupling between the system and the bath
        '''
        self.coupling = np.array(coupling, dtype=np.complex128)

    def set_coupling_dipole(self, coupling_dipole):
        '''
        set the coupling dipole between the system and the bath
        '''
        self.coupling_dipole = np.array(
            coupling_dipole, dtype=np.complex128)

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
        self.nsys = np.shape(self.system)[0]
        self.nind = len(self.bath.expn)
        self.nmod = np.max(self.bath.mode) + 1

        if self.system is None:
            raise ValueError('System Hamiltonian is not set.')
        if self.coupling is None:
            raise ValueError('system bath interaction operator is not set.')

    def init_(self):
        # note the maximum number of int64 is  2, 147, 483, 647 which is nearly comb(40, 10)
        combmax = self.nind + self.lmax + 1
        self.comb_list = np.zeros((combmax, combmax), dtype=np.int64)
        for j in range(combmax):
            self.comb_list[0, j] = 0
        self.comb_list[0, 0] = 1
        for i in range(1, combmax):
            for j in range(1, combmax):
                self.comb_list[i, j] = self.comb_list[i - 1, j] \
                    + self.comb_list[i - 1, j - 1]
            self.comb_list[i, 0] = 1
        self.nmax = self.comb_list[self.lmax+self.nind, self.lmax]

        # allocate memory for keys
        self.keys = np.zeros((self.nmax, self.nind), dtype=np.int64)
        gen_keys(self.keys, self.lmax, self.nind, self.comb_list)

    def rk4(self, dt, t):
        '''
        the Runge-Kutta wrapper
        '''
        return rk4(self.ddos, self.ddos1, self.ddos2, self.ddos3, self.keys, self.lmax, [self.bath.expn, self.bath.etal, self.bath.etar, self.bath.etaa],  self.bath.mode, self.system, self.system_dipole, self.pulse_system_func, self.coupling, self.coupling_dipole, self.pulse_coupling_func, self.comb_list, self.nmax, dt, t)

    def run(self, rho0, dt, nt, p1=None):
        '''
        solve the DEOM, return the time and density matrix of the system.
        Input:
            rho0: the initial density matrix of the system.
            dt: the time step.
            nt: the number of time steps.
        '''
        self.check_()
        self.init_()

        # allocate memory for ddos
        self.ddos = [coo_matrix((self.nsys, self.nsys,),
                                dtype=np.complex128)] * self.nmax
        self.ddos1 = [coo_matrix((self.nsys, self.nsys,),
                                 dtype=np.complex128)] * self.nmax
        self.ddos2 = [coo_matrix((self.nsys, self.nsys,),
                                 dtype=np.complex128)] * self.nmax
        self.ddos3 = [coo_matrix((self.nsys, self.nsys,),
                                 dtype=np.complex128)] * self.nmax
        self.ddos[0] = rho0

        t_save = np.zeros(nt + 1, dtype=np.float64)
        if p1 is None:
            ddos_save = [coo_matrix((self.nsys, self.nsys,),
                                    dtype=np.complex128)] * (nt + 1)
        else:
            ddos_save = np.zeros(nt + 1, dtype=np.complex128)

        t_save[0] = 0
        if p1 is None:
            ddos_save[0] = self.ddos[0].copy()
        else:
            ddos_save[0] = (p1 @ self.ddos[0]).trace()

        for i in tqdm(range(nt)):
            self.rk4(dt, i * dt)
            t_save[i + 1] = (i + 1) * dt
            if p1 is None:
                ddos_save[i + 1] = self.ddos[0].copy()
            else:
                ddos_save[i + 1] = (p1 @ self.ddos[0]).trace()
        return t_save, ddos_save

    def gen_generate_propgator(self):
        '''
        generate the green function of the system.
        '''
        self.check_()
        self.init_()
        self.propgator = np.zeros((self.nmax * self.nsys * self.nsys, self.nmax *
                                   self.nsys * self.nsys), dtype=np.complex128)
        generate_propgator(self.propgator, self.keys, self.lmax, [
            self.bath.expn, self.bath.etal, self.bath.etar, self.bath.etaa], self.bath.mode, self.system, self.coupling, self.comb_list, self.nmax)

    def correlation_4op_3t(self, operator_a, operator_b, operator_c, operator_d, rho0, T, w_x, w_y, if_full=True, cut_off_min=0.5, cut_off_max=1.1, if_load=False, if_save=False, lcr='llll'):
        if self.propgator is None:
            print('propgator is not generated, generating now...')
            self.gen_generate_propgator()
            print('propgator is generated.')

        if if_load and os.path.exists('correlation_4op_3t.npz'):
            print('loading correlation_4op_3t from file...')
            data = np.load('correlation_4op_3t.npz')
            self.Δ = data['Δ']
            self.V = data['V']
            self.V_inv = data['V_inv']
            print('correlation_4op_3t is loaded.')

        if self.Δ is None:
            print('eigenvalues and eigenvectors are not generated, generating now...')
            self.Δ, self.V = la.eig(self.propgator)
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