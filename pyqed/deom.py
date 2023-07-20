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


def decompose_specturm_pade(spe, w_sp, beta, npsd, pade=1):
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
    pole, resi = pade_approximation_distribution(npsd, 1, pade)
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


def key_plus(key, pos):
    '''
    generate the key for the \rho_{{n}^{+}_{k}}
    '''
    key1 = key.copy()
    key1[pos] += 1
    return key1


def key_minus(key, pos):
    '''
    generate the key for the \rho_{{n}^{-}_{k}}
    '''
    key1 = key.copy()
    key1[pos] -= 1
    return key1


def hash_plus(key, pos, nind, comb_list):
    '''
    generate the hash value for the \rho_{{n}^{+}_{k}}
    '''
    key1 = key.copy()
    key1[pos] += 1
    return gen_hash_value(key1, nind, comb_list)


def hash_minus(key, pos, nind, comb_list):
    '''
    generate the hash value for the \rho_{{n}^{-}_{k}}
    '''
    key1 = key.copy()
    key1[pos] -= 1
    return gen_hash_value(key1, nind, comb_list)


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


def generate_time(system, system_dip, pulse_system_func, coupling, coupling_dip, pulse_coupling_func, t):
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


def rk4(ddos, ddos1, ddos2, ddos3, keys, lmax, bath_list, mode, system, system_dip, pulse_system_func, coupling, coupling_dip, pulse_coupling_func, comb_list, nmax, dt, t):
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


class Bath():
    '''
    bath class for DEOM
    '''

    def __init__(self, specturm_sp=None, w_sp=None, beta=None, npsd=None, mode=None):
        '''
        initialize the bath
        Input:
            specturm_sp: the spectral density of the bath, a function of frequency. Must be a sympy expression.
            w_sp: the frequency of the bath, a sympy symbol.
            beta: the inverse temperature of the bath.
            npsd: the number of the poles of the spectral density.
        '''
        self.bath = specturm_sp
        self.w_sp = w_sp
        self.beta = beta
        self.npsd = npsd
        self.mode = mode
        self.etal, self.etar, self.etaa, self.expn = [], [], [], []
        if ((type(self.bath) is list) and (type(self.beta) is list) and (type(self.npsd) is list)):
            if (len(self.bath) != len(self.beta)) or (len(self.bath) != len(self.npsd)):
                print("the length of bath, w_sp, beta, npsd is not equal!")
            for i in range(len(self.bath)):
                etal_p, etar_p, etaa_p, expn_p = decompose_specturm_pade(
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
            self.etal, self.etar, self.etaa, self.expn = decompose_specturm_pade(
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

    def gen_comb_list(self):
        '''
        generate the combination list
        '''
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

    def init_(self):
        # allocate memory for ddos and key
        self.ddos = [coo_matrix((self.nsys, self.nsys,),
                                dtype=np.complex128)] * self.nmax
        self.ddos1 = [coo_matrix((self.nsys, self.nsys,),
                                 dtype=np.complex128)] * self.nmax
        self.ddos2 = [coo_matrix((self.nsys, self.nsys,),
                                 dtype=np.complex128)] * self.nmax
        self.ddos3 = [coo_matrix((self.nsys, self.nsys,),
                                 dtype=np.complex128)] * self.nmax

        self.keys = np.zeros((self.nmax, self.nind), dtype=np.int64)
        gen_keys(self.keys, self.lmax, self.nind, self.comb_list)

    def rk4(self, dt, t):
        '''
        the Runge-Kutta wrapper
        '''
        return rk4(self.ddos, self.ddos1, self.ddos2, self.ddos3, self.keys, self.lmax, [self.bath.expn, self.bath.etal, self.bath.etar, self.bath.etaa],  self.bath.mode, self.system, self.system_dipole, self.pulse_system_func, self.coupling, self.coupling_dipole, self.pulse_coupling_func, self.comb_list, self.nmax, dt, t)

    def run(self, rho0, dt, nt):
        '''
        solve the DEOM, return the time and density matrix of the system.
        Input:
            rho0: the initial density matrix of the system.
            dt: the time step.
            nt: the number of time steps.
        '''
        self.check_()
        self.gen_comb_list()
        self.init_()
        self.ddos[0] = rho0

        t_save = np.zeros(nt + 1, dtype=np.float64)
        ddos_save = [coo_matrix((self.nsys, self.nsys,),
                                dtype=np.complex128)] * (nt + 1)
        t_save[0] = 0
        ddos_save[0] = self.ddos[0]

        for i in tqdm(range(nt)):
            self.rk4(dt, i * dt)
            t_save[i + 1] = i * dt
            ddos_save[i + 1] = self.ddos[0]
        return t_save, ddos_save

    # def correlation_4op_3t(self, operator_a, operator_b, operator_c, operator_d, rho0, dt, nt):
    #     self.check_()
    #     self.gen_comb_list()
    #     self.init_()
    #     self.ddos[0] = rho0
    #     operator_a, operator_b, operator_c, operator_d = np.array(
    #         operator_a, dtype=np.complex128), np.array(operator_b, dtype=np.complex128), np.array(operator_c, dtype=np.complex128), np.array(operator_d, dtype=np.complex128)
    #     # for i in tqdm(range(nt)):
    #     #     rk4(self.ddos, self.ddos1, self.ddos2, self.ddos3, self.keys, self.lmax, self.bath.expn, self.system, self.system_dipole, self.pulse_system_func,
    #     #         self.coupling, self.coupling_dipole, self.pulse_coupling_func, self.comb_list, self.nmax, dt, i * dt)
    #     ddos_bk = np.copy(self.ddos)

    #     corr_save = np.zeros(
    #         (nt, nt, nt), dtype=np.complex128)
    #     for i in range(nt):
    #         for j in range(nt):
    #             for k in range(nt):
    #                 self.ddos = np.copy(ddos_bk)
    #                 self.ddos = operator_action_ddos(
    #                     operator_a, self.ddos, self.nmax)
    #                 for ti in range(i):
    #                     self.rk4(dt, ti * dt)
    #                 self.ddos = operator_action_ddos(
    #                     operator_b, self.ddos, self.nmax)
    #                 for tj in range(j):
    #                     self.rk4(dt, (i + tj) * dt)
    #                 self.ddos = operator_action_ddos(
    #                     operator_c, self.ddos, self.nmax)
    #                 for tk in range(k):
    #                     self.rk4(dt, (i + j + tk) * dt)
    #                 corr_save[i, j, k] = np.trace(operator_d @  self.ddos[0])
    #     return corr_save
