
# -*- coding: utf-8 -*-
"""

Spin-orbital G0W0

Created on Sat Jan  5 23:55:43 2019
@author: Bing Gu

@description:
    The theory for the code can be found in [Fabien Bruneval, J. Chem. Phys. 2012, 136, 194107]

Based on the PySCF implementation
"""

import numpy as np
import scipy.linalg
from scipy.optimize import newton

from pyscf.lib import logger
import pyscf.ao2mo
import pyscf
from pyscf import dft
from functools import reduce

from pyqed import is_positive_def

class GW(object):
    def __init__(self, mf, ao2mofn=pyscf.ao2mo.outcore.general_iofree,
                 screening='TDH', eta=1e-2):
        
        assert screening in ('TDH', 'TDHF', 'TDDFT')
        
        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        self.nocc = self.mol.nelectron//2
        try:
            # DFT
            mf.xc = mf.xc
            v_mf = mf.get_veff() - mf.get_j()

        except AttributeError:
            # HF
            v_mf = -mf.get_k()

        if mf.mo_occ[0] == 2:
            # RHF, convert to spin-orbitals
#            nso = 2*len(mf.mo_energy)
#            self.nso = nso
#            self.e_mf = np.zeros(nso)
#            self.e_mf[0::2] = self.e_mf[1::2] = mf.mo_energy
#            b = np.zeros((nso//2,nso))
#            b[:,0::2] = b[:,1::2] = mf.mo_coeff
#            self.v_mf = 0.5 * reduce(np.dot, (b.T, v_mf, b))
#            self.v_mf[::2,1::2] = self.v_mf[1::2,::2] = 0
#            eri = ao2mofn(mf.mol, (b,b,b,b),
#                          compact=False).reshape(nso,nso,nso,nso)
#            eri[::2,1::2] = eri[1::2,::2] = eri[:,:,::2,1::2] = eri[:,:,1::2,::2] = 0
#            # Integrals are in "chemist's notation"
#            # eri[i,j,k,l] = (ij|kl) = \int i(1) j(1) 1/r12 k(r2) l(r2)
#            print("Imag part of ERIs =", np.linalg.norm(eri.imag))
#            self.eri = eri.real

            nso = len(mf.mo_energy)
            self.nso = nso
            self.e_mf = mf.mo_energy
            b = mf.mo_coeff
            self.v_mf = reduce(np.dot, (b.T, v_mf, b))
            eri = ao2mofn(mf.mol, (b,b,b,b),
                          compact=False).reshape(nso,nso,nso,nso)
            self.eri = eri

        else:
            # ROHF or UHF, these are already spin-orbitals
            print("\n*** Only supporting restricted calculations right now! ***\n")
            raise NotImplementedError
            nso = len(mf.mo_energy)
            self.nso = nso
            self.e_mf = mf.mo_energy
            b = mf.mo_coeff
            self.v_mf = reduce(np.dot, (b.T, v_mf, b))
            eri = ao2mofn(mf.mol, (b,b,b,b),
                          compact=False).reshape(nso,nso,nso,nso)
            self.eri = eri

        print("There are %d spin-orbitals"%(self.nso))

        self.screening = screening
        self.eta = eta
        self._M = None

        self.egw = None
        self.e_rpa = None

    def kernel(self, mo_energy=None, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = self._scf.mo_coeff
        if mo_energy is None:
            mo_energy = self._scf.mo_energy

        self.egw = kernel(self, mo_energy, mo_coeff, verbose=self.verbose)
        logger.log(self, 'GW bandgap = %.15g', self.egw[self.nocc//2]-self.egw[self.nocc//2-1])
        return self.egw

    def sigma(self, p, q, omegas, e_rpa, t_rpa, vir_sgn=1):
        return sigma(self, p, q, omegas, e_rpa, t_rpa, vir_sgn)

    def g0(self, omega):
        return g0(self, omega)

    def get_m_rpa(self, e_rpa, t_rpa):
        return get_m_rpa(self, e_rpa, t_rpa)

    def rpa(self, using_tda=False, using_casida=True, method='TDH'):
        self.e_rpa, t = rpa(self, using_tda, using_casida, method)
        return self.e_rpa, t

    def rpa_AB_matrices(self, method='TDH'):
        return rpa_AB_matrices(self, method)

    def bse(self, using_tda=False, using_casida=True):
        return bse(self, using_tda, using_casida)




def g0(gw, omega):
    '''
    Return the 0th order GF matrix [G0]_{pq} in the basis of single-particle
    orbitals (MF eigenvectors).
    '''
    g0 = np.zeros((gw.nso,gw.nso), dtype=np.complex128)
    for p in range(gw.nso):
        if p < gw.nocc: sgn = -1
        else: sgn = +1
        g0[p,p] = 1.0/(omega - gw.e_mf[p] + 1j*sgn*gw.eta)
    return g0

def rpa_AB_matrices(gw, method='TDHF'):
    '''Get the RPA A and B matrices, using TDH, TDHF, or TDDFT.
    '''
    assert method in ('TDH','TDHF','TDDFT')
    nso = gw.nso
    nocc = gw.nocc
    nvir = nso - nocc

    dim_rpa = nocc*nvir
    A = np.zeros((dim_rpa,dim_rpa))
    B = np.zeros((dim_rpa,dim_rpa))

    ia = 0
    for i in range(nocc):
        for a in range(nocc,nso):
            A[ia, ia] = gw.e_mf[a] - gw.e_mf[i]
            jb = 0
            for j in range(nocc):
                for b in range(nocc,nso):
                    A[ia,jb] += 2.*gw.eri[a,i,b,j]
                    B[ia,jb] += 2.*gw.eri[i,a,j,b]
                    
                    if method == 'TDHF':
                        A[ia,jb] -= gw.eri[a,b,i,j]
                        B[ia,jb] -= gw.eri[a,j,i,b]
                    jb += 1
            ia += 1

    assert np.allclose(A, A.transpose())
    assert np.allclose(B, B.transpose())

    return A, B

def rpa(gw, using_tda=False, using_casida=True, method='TDHF'):
    '''Get the RPA eigenvalues and eigenvectors.

    Q^\dagger = \sum_{ia} X_{ia} a^+ i - Y_{ia} i^+ a
    Leads to the RPA eigenvalue equations:
      [ A  B ][X] = omega [ 1  0 ][X]
      [ B  A ][Y]         [ 0 -1 ][Y]
    which is equivalent to
      [ A  B ][X] = omega [ 1  0 ][X]
      [-B -A ][Y] =       [ 0  1 ][Y]

    See, e.g. Stratmann, Scuseria, and Frisch,
              J. Chem. Phys., 109, 8218 (1998)
    '''
    A, B = rpa_AB_matrices(gw, method=method)

    if using_tda:
        ham_rpa = A
        e, x = eig(ham_rpa)
        return e, x
    else:
        if not using_casida:
            ham_rpa = np.array(np.bmat([[A,B],[-B,-A]]))
            assert is_positive_def(ham_rpa)
            e, xy = eig_asymm(ham_rpa)
            return e, xy
        else:
            assert is_positive_def(A-B)
            sqrt_A_minus_B = scipy.linalg.sqrtm(A-B)
            ham_rpa = np.dot(sqrt_A_minus_B, np.dot((A+B),sqrt_A_minus_B))
            esq, t = eig(ham_rpa)
            return np.sqrt(esq), t


def get_m_rpa(gw, e_rpa, t_rpa):
    '''
    Get the (intermediate) M_{pq,L} tensor.
    The M (or w) is needed to construct the screened Coulomb interaction W
    
    .. math::
        
        M_{pq,L} = \sum_{ia} ( (eps_a-eps_i)/erpa_L )^{1/2} T_{ai,L} (ai|pq)
    '''
    nso = gw.nso
    nocc = gw.nocc
    nvir = nso - nocc
    t_by_e = t_rpa.copy()
    for L in range(len(e_rpa)):
        t_by_e[:,L] /= np.sqrt(e_rpa[L])
    sqrt_eps = np.zeros(nocc*nvir)
    eri_product = np.zeros((nocc*nvir, nso, nso))
    ai = 0
    for i in range(nocc):
        for a in range(nocc,nso):
            sqrt_eps[ai] = np.sqrt(gw.e_mf[a]-gw.e_mf[i])
            eri_product[ai,:,:] = gw.eri[a,i,:,:]
            ai += 1
    M = np.einsum('a,al,apq->pql', sqrt_eps, t_by_e, eri_product)
    return M



def sigma(gw, p, q, omegas, e_rpa, t_rpa, vir_sgn=1):
    '''
    self energy 
    .. math::
        
        \Sigma_{pq} = i [GW]_{pq}
    '''
    if not isinstance(omegas, (list,tuple,np.ndarray)):
        single_point = True
        omegas = [omegas]
    else:
        single_point = False

    # This usually takes the longest:
    if gw._M is None:
        gw._M = get_m_rpa(gw, e_rpa, t_rpa)

    nso = gw.nso
    nocc = gw.nocc

    sigma_c = []
    sigma_x = []
    for omega in omegas:
        sigma_cw = 0.
        sigma_xw = 0.
        for L in range(len(e_rpa)):
            for i in range(nocc):
                sigma_cw += gw._M[i,q,L]*gw._M[i,p,L]/(
                            omega - gw.e_mf[i] + e_rpa[L] - 1j*gw.eta )
            for a in range(nocc, nso):
                sigma_cw += gw._M[a,q,L]*gw._M[a,p,L]/(
                            omega - gw.e_mf[a] - e_rpa[L] + vir_sgn*1j*gw.eta )
        for i in range(nocc):
            sigma_xw += -gw.eri[p,i,i,q]

        sigma_c.append(sigma_cw)
        sigma_x.append(sigma_xw)

    if single_point:
        return sigma_c[0], sigma_x[0]
    else:
        return sigma_c, sigma_x

#def sigma(gw, p, q, omegas, e_rpa, t_rpa, vir_sgn=1):
#    '''
#    self energy sigma_{pq} = i [GW]_{pq}
#    '''
#    if not isinstance(omegas, (list,tuple,np.ndarray)):
#        single_point = True
#        omegas = [omegas]
#    else:
#        single_point = False
#
#    # This usually takes the longest:
#    if gw._M is None:
#        gw._M = get_m_rpa(gw, e_rpa, t_rpa)
#
#    nso = gw.nso
#    nocc = gw.nocc
#
#    sigma_c = []
#    sigma_x = []
#    for omega in omegas:
#        sigma_cw = 0.
#        sigma_xw = 0.
#        for L in range(len(e_rpa)):
#            for i in range(nocc):
#                sigma_cw += gw._M[i,q,L]*gw._M[i,p,L]/(
#                            omega - gw.e_mf[i] + e_rpa[L] - 1j*gw.eta )
#            for a in range(nocc, nso):
#                sigma_cw += gw._M[a,q,L]*gw._M[a,p,L]/(
#                            omega - gw.e_mf[a] - e_rpa[L] + vir_sgn*1j*gw.eta )
#        for i in range(nocc):
#            sigma_xw += -gw.eri[p,i,i,q]
#
#        sigma_c.append(sigma_cw)
#        sigma_x.append(sigma_xw)
#
#    if single_point:
#        return sigma_c[0], sigma_x[0]
#    else:
#        return sigma_c, sigma_x

def kernel(gw, so_energy, so_coeff, verbose=logger.NOTE):
    '''Get the GW-corrected spatial orbital energies.

    Note: Works in spin-orbitals but returns energies for spatial orbitals.

    Args:
        gw : instance of :class:`GW`
        so_energy : (nso,) ndarray
        so_coeff : (nso,nso) ndarray

    Returns:
        egw : (nso/2,) ndarray
            The GW-corrected spatial orbital energies.
    '''
    print("# --- Performing RPA calculation ...")
    e_rpa, t_rpa = rpa(gw, method=gw.screening)

    print('RPA eigenvalues = ', e_rpa)


    # store the RPA eigvalues for BSE calculations
    gw.e_rpa = e_rpa

    print("done.")
    print("# --- Calculating GW QP corrections ...")
    egw = np.zeros(int(gw.nso))
    for p in range(0,gw.nso):
        def quasiparticle(omega):
            sigma_c_ppw, sigma_x_ppw = sigma(gw, p, p, omega, e_rpa, t_rpa)
            sigma_ppw = sigma_c_ppw + sigma_x_ppw
            return omega - gw.e_mf[p] - (2.*sigma_ppw.real - gw.v_mf[p,p])
        try:
            egw[p] = newton(quasiparticle, gw.e_mf[p], tol=1e-6, maxiter=100)
        except RuntimeError:
            print("Newton-Raphson unconverged, setting GW eval to MF eval.")
            egw[p] = gw.e_mf[p]
        print(egw[p])
    print("done.")

    return egw

def bse_AB_matrices(gw):
    '''Get the BSE A and B matrices, using the quasiparticle energies computed
    from GW, and screened interaction W computed from RPA
    '''
    method = gw.screening
    assert method in ('TDH','TDHF','TDDFT')

    # restricted calculations only
    nso = gw.nso
    nocc = gw.nocc
    nvir = nso - nocc

    dim_rpa = nocc*nvir
    A = np.zeros((dim_rpa,dim_rpa))
    B = np.zeros((dim_rpa,dim_rpa))

    ia = 0
    for i in range(nocc):
        for a in range(nocc,nso):
            # with GW corrected quasiparticle energy
            A[ia, ia] = gw.egw[a] - gw.egw[i]
            # with HF orbital energy
            #A[ia, ia] = gw._scf.mo_energy[a] - gw._scf.mo_energy[i]

            jb = 0
            for j in range(nocc):
                for b in range(nocc,nso):
                    A[ia,jb] += 2.*gw.eri[a,i,b,j] - gw.eri[a,b,i,j]
                    B[ia,jb] += 2.*gw.eri[a,i,j,b] - gw.eri[a,j,i,b]
                    #if method == 'TDHF':
                    #    A[ia,jb] -= gw.eri[a,b,i,j]
                    #    B[ia,jb] -= gw.eri[a,j,i,b]
                    for L in range(len(gw.e_rpa)):
                        A[ia, jb] -= 2.*gw._M[i,j,L] * gw._M[a,b,L]/ gw.e_rpa[L]
                        B[ia, jb] -= 2.*gw._M[i,b,L] * gw._M[a,j,L]/ gw.e_rpa[L]

                    jb += 1

            ia += 1

    #assert np.allclose(A, A.transpose())
    #assert np.allclose(B, B.transpose())

    return A, B

def bse(gw, using_tda=False, using_casida=True):
    '''Get the RPA eigenvalues and eigenvectors.

    Q^\dagger = \sum_{ia} X_{ia} a^+ i - Y_{ia} i^+ a
    Leads to the RPA eigenvalue equations:
      [ A  B ][X] = omega [ 1  0 ][X]
      [ B  A ][Y]         [ 0 -1 ][Y]
    which is equivalent to
      [ A  B ][X] = omega [ 1  0 ][X]
      [-B -A ][Y] =       [ 0  1 ][Y]

    See, e.g. Stratmann, Scuseria, and Frisch,
              J. Chem. Phys., 109, 8218 (1998)
    '''

    A, B = bse_AB_matrices(gw)

    if using_tda:
        ham_rpa = A
        e, x = eig(ham_rpa)
        return e, x
    else:
        if not using_casida:
            ham_rpa = np.array(np.bmat([[A,B],[-B,-A]]))
            assert is_positive_def(ham_rpa)
            e, xy = eig_asymm(ham_rpa)
            return e, xy
        else:
            assert is_positive_def(A-B)
            sqrt_A_minus_B = scipy.linalg.sqrtm(A-B)
            ham_rpa = np.dot(sqrt_A_minus_B, np.dot((A+B),sqrt_A_minus_B))
            esq, t = eig(ham_rpa)
            return np.sqrt(esq), t


def eig(h, s=None):
    e, c = scipy.linalg.eigh(h,s)
    return e, c


def eig_asymm(h):
    '''Diagonalize a real, *asymmetrix* matrix and return sorted results.

    Return the eigenvalues and eigenvectors (column matrix)
    sorted from lowest to highest eigenvalue.
    '''
    e, c = np.linalg.eig(h)
    if np.allclose(e.imag, 0*e.imag):
        e = np.real(e)
    else:
        print("WARNING: Eigenvalues are complex, will be returned as such.")

    idx = e.argsort()
    e = e[idx]
    c = c[:,idx]

    return e, c


# def is_positive_def(A):
#     try:
#         np.linalg.cholesky(A)
#         return True
#     except np.linalg.LinAlgError:
#         return False




def pes():

    mol = gto.Mole()
    mol.verbose = 2
    #mol.atom = [['Ne' , (0., 0., 0.)]]
    #mol.basis = {'Ne': '3-21G'}
    # This is from G2/97 i.e. MP2/6-31G*

    R = np.linspace(0.2, 4, 10)
    ex_energy = np.zeros(len(R))

    f = open('excite_energy.dat','w')
    for i in range(len(R)):

        mol.atom = [['H' , (0.,  0., 0.)],
                    ['H' , (R[i], 0., 0.)]]
        mol.basis = '321g'
        mol.build()
        mf = scf.RHF(mol)

        mf.kernel()

        gw = GW(mf, screening='TDHF')
        egw = gw.kernel()

        print('HF    vs.   GW ')
        for emf, eqp in zip(mf.mo_energy, egw):
            print("%0.6f %0.6f"%(emf, eqp))

        nocc = mol.nelectron//2
        ehomo = egw[nocc-1]
        print("GW -IP = GW HOMO =", ehomo, "au =", ehomo*27.211, "eV")


        #print('GW  spacial orbital energies (eV) = ', gw.egw*27.211)

        ex = bse(gw, using_tda=True, using_casida=False)[0]
        ex_energy[i] = ex[0]

        print("BSE Excitation energy =", ex ) #* au2ev)
        print("RPA eigenvalue = ", gw.e_rpa[0])
        f.write('{} {} {} \n'.format(R[i], ex_energy[i], gw.e_rpa[0]))

    f.close()

    return


if __name__ == '__main__':

    from pyscf import scf, gto
    from pyqed import au2ev

    mol = gto.Mole()
    mol.verbose = 2
    mol_name = 'H2'

#    if mol_name == 'H2':
##    #mol.atom = [['Ne' , (0., 0., 0.)]]
##    #mol.basis = {'Ne': '3-21G'}
##    # This is from G2/97 i.e. MP2/6-31G*
#        mol.atom = [['H' , (0.,  0., 0.)],
#                ['H' , (1.6, 0., 0.)]]
#    elif mol_name == 'benzene':
#        mol.atom = '''
#       C        0.70021        1.21284        0.00000
#       C        1.40045       -0.00002       -0.00000
#       C        0.70024       -1.21282        0.00000
#       C       -0.70024       -1.21282        0.00000
#       C       -1.40045       -0.00002        0.00000
#       C       -0.70021        1.21284       -0.00000
#       H       -1.24315        2.15309        0.00000
#       H        1.24315        2.15309        0.00000
#       H        2.48621        0.00006        0.00000
#       H        1.24305       -2.15315       -0.00000
#       H       -1.24305       -2.15315        0.00000
#       H       -2.48621        0.00006        0.00000
#       '''
#    else:
#        raise NotImplementedError

    # mol.atom = [
    #     [8 , (0. , 0.     , 0.)],
    #     [1 , (0. , -0.757 , 0.587)],
    #     [1 , (0. , 0.757  , 0.587)]]
    
    #!/usr/bin/env python


    mol = gto.M(
        atom = 'H 0 0 0; F 0 0 1.1',
        basis = '631g')
    
    nocc = mol.nelectron//2
    
    # By default, GW is done with analytic continuation
    # gw = gw.GW(mf)
    # same as gw = gw.GW(mf, freq_int='ac')
    # gw.kernel(orbs=range(nocc-3,nocc+3))

    # mol.basis = 'cc-pvdz'
#    mol.atom = '''
#    H   0.000000   0.934473    -0.588078
#    H   0.000000   -0.934473   -0.588078
#    C   0.000000   0.000000    0.000000
#    O   0.000000   0.000000    1.221104
#    '''
    # mol.basis = 'sto3g'
    mol.build()

    mf = scf.RHF(mol)
    #print(mf.scf())
    mf.kernel()

    gw = GW(mf, screening='TDH')
    egw = gw.kernel()
    print('HF    vs.   GW ')
    for emf, eqp in zip(mf.mo_energy, egw):
        print("%0.6f %0.6f"%(emf, eqp))

    nocc = mol.nelectron//2
    ehomo = egw[nocc-1]
    elumo = egw[nocc]
    print("GW -IP = GW HOMO =", ehomo, "au =", ehomo*27.211, "eV")
    print("GW EA = GW LUMO =", elumo, "au =", elumo*27.211, "eV")

#
#
    # print('GW  spacial orbital energies (eV) = ', gw.egw*27.211)

    #excite = bse(gw, using_tda=True, using_casida=False)
    #print("BSE Excitation energy =", excite[0] ) #* au2ev)

    # pes()

    import matplotlib.pyplot as plt

    R,E, e_rpa = np.genfromtxt('excite_energy.dat', dtype=float, unpack=True)
    #print(R,E)
    #print(R, e_rpa)
    # R /= 0.529177
    plt.plot(R, E)
    # plt.plot(R, e_rpa)