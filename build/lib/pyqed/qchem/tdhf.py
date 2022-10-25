#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 16:29:52 2022

@author: bing
"""
import numpy as np
from scipy.linalg import eigh, sqrtm
import scipy
import pyscf

from pyqed import is_positive_def
from pyqed.units import au2ev

from functools import reduce
import logging


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


def rpa(gw, using_tda=False, using_casida=True, method='TDH'):
    '''Get the RPA eigenvalues and eigenvectors.

    The RPA computation is required to construct the dielectric function, i.e. screened
    Coloumb interaction.

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
    A, B = get_ab(gw, method=method)

    if using_tda:
        ham_rpa = A
        e, x = eigh(ham_rpa)
        return e, x
    else:
        if not using_casida:
            ham_rpa = np.array(np.bmat([[A,B],[-B,-A]]))
            assert is_positive_def(ham_rpa)
            e, xy = eig_asymm(ham_rpa)
            return e, xy
        else:
            assert is_positive_def(A-B)
            sqrt_A_minus_B = sqrtm(A-B)
            ham_rpa = np.dot(sqrt_A_minus_B, np.dot((A+B),sqrt_A_minus_B))
            esq, t = eigh(ham_rpa)
            return np.sqrt(esq), t


def get_ab(gw, method='TDH'):
    '''Compute the RPA A and B matrices, using TDH, TDHF, or TDDFT.
    '''
    assert method in ('TDH','TDHF','TDDFT')
    nso = gw.nso
    nocc = gw.nocc
    nvir = nso - nocc

    dim_rpa = nocc*nvir
    logging.info('dim of AB matrices = {}'.format(dim_rpa))

    A = np.zeros((dim_rpa, dim_rpa))
    B = np.zeros((dim_rpa, dim_rpa))

    ai = 0
    for i in range(nocc):
        for a in range(nocc,nso):
            A[ai,ai] = gw.e_mf[a] - gw.e_mf[i]
            bj = 0
            for j in range(nocc):
                for b in range(nocc,nso):
                    A[ai,bj] += gw.eri[a,i,j,b]
                    B[ai,bj] += gw.eri[a,i,b,j]
                    if method == 'TDHF':
                        A[ai,bj] -= gw.eri[a,b,j,i]
                        B[ai,bj] -= gw.eri[a,j,b,i]
                    bj += 1
            ai += 1

    assert np.allclose(A, A.transpose())
    assert np.allclose(B, B.transpose())

    return A, B

class TDH:
    '''
    Time-dependent Hartree 
    '''
    def __init__(self):
        pass
    
class TDHF:
    def __init__(self, mf):

        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory
        self.spin = 0

        self.nocc = self.mol.nelectron # spin-polarized

        self._a = None
        self._b = None

        # HF
        v_mf = -mf.get_k()

        if mf.mo_occ[0] == 2:
            # RHF, convert to spin-orbitals
            nso = 2*len(mf.mo_energy)
            self.nso = nso
            self.e_mf = np.zeros(nso)
            self.e_mf[0::2] = self.e_mf[1::2] = mf.mo_energy
            b = np.zeros((nso//2, nso)) # nao, nso
            b[:,0::2] = b[:,1::2] = mf.mo_coeff
            self.v_mf = 0.5 * reduce(np.dot, (b.T, v_mf, b))
            self.v_mf[::2,1::2] = self.v_mf[1::2,::2] = 0

            # electron repulsion integral
            ao2mofn = pyscf.ao2mo.outcore.general_iofree
            eri = ao2mofn(mf.mol, (b,b,b,b),
                          compact=False).reshape(nso,nso,nso,nso)

            eri[::2,1::2] = eri[1::2,::2] = eri[:,:,::2,1::2] = eri[:,:,1::2,::2] = 0
            # Integrals are in "chemist's notation"
            # eri[i,j,k,l] = (ij|kl) = \int i(1) j(1) 1/r12 k(r2) l(r2)
            print("Imag part of ERIs =", np.linalg.norm(eri.imag))
            self.eri = eri.real

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

        # self.eta = eta
        self._M = None

    # def run(self):
    #     # if mo_coeff is None:
    #     #     mo_coeff = self._scf.mo_coeff
    #     # if mo_energy is None:
    #     #     mo_energy = self._scf.mo_energy

    #     # self.egw = kernel(self, mo_energy, mo_coeff, verbose=self.verbose)
    #     # logger.log(self, 'GW bandgap = %.15g', self.egw[self.nocc//2]-self.egw[self.nocc//2-1])
    #     # return self.egw
    #     return rpa(self, using_tda=True, method='TDHF')

    # def sigma(self, p, q, omegas, e_rpa, t_rpa, vir_sgn=1):
    #     return sigma(self, p, q, omegas, e_rpa, t_rpa, vir_sgn)

    # def g0(self, omega):
    #     return g0(self, omega)

    # def get_m_rpa(self, e_rpa, t_rpa):
    #     return get_m_rpa(self, e_rpa, t_rpa)

    def run(self, nstates=None, using_tda=False, using_casida=True, method='TDHF'):
        '''Get the RPA eigenvalues and eigenvectors.

        The RPA computation is required to construct the dielectric function, i.e. screened
        Coloumb interaction.

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
        A, B = self.get_ab(method=method)

        if using_tda:

            logging.info('Using TDA approximation')
            ham_rpa = A
            e, x = eig(ham_rpa)
            # if isinstance(nstates, int):
            #     e, x = scipy.sparse.linalg.eigsh(ham_rpa, k=nstates)
            # else:
            #     e, x = eigh(ham_rpa)
            # return e, x
        else:
            if using_casida:
                assert is_positive_def(A-B)
                sqrt_A_minus_B = sqrtm(A-B)
                ham_rpa = np.dot(sqrt_A_minus_B, np.dot((A+B),sqrt_A_minus_B))

                esq, t = eig(ham_rpa, k=nstates, which='SM')
                e = np.sqrt(esq)
                print('Roots (eV) = ', e*au2ev)
                return e, t

            else:
                ham_rpa = np.array(np.bmat([[A,B],[-B,-A]]))
                assert is_positive_def(ham_rpa)
                e, xy = eig_asymm(ham_rpa)
                return e, xy




    def get_ab(self, method='TDHF'):
        a, b = get_ab(self, method=method)
        self._a = a
        self._b = b
        return a, b

def eig(a, k=None, **kwargs):
    '''
    customized eigenvalue function for Hermitian matrix

    Parameters
    ----------
    a : TYPE
        DESCRIPTION.
    k : TYPE, optional
        number of required eigenstates. If None, do the full calculation. The default is None.
    **kwargs : TYPE
        kwargs for scipy.sparse.linalg.eigsh()

    Returns
    -------
    e : TYPE
        DESCRIPTION.
    x : TYPE
        DESCRIPTION.

    '''
    if isinstance(k, int):
        e, x = scipy.sparse.linalg.eigsh(a, k=k, **kwargs)
    else:
        e, x = eigh(a)
    return e, x


if __name__ == '__main__':
    from pyscf import gto, scf, tddft

    mol = gto.Mole()
    mol.atom = [
        ['H' , (0. , 0. , .917)],
        ['H' , (0. , 0. , 0.)], ]
    mol.basis = '631g*'
    mol.build()

    #
    # RHF/RKS-TDDFT
    #
    def diagonalize(a, b, nroots=5):
        nocc, nvir = a.shape[:2]
        a = a.reshape(nocc*nvir,nocc*nvir)
        b = b.reshape(nocc*nvir,nocc*nvir)
        e = np.linalg.eig(np.bmat([[a        , b       ],
                                   [-b.conj(),-a.conj()]]))[0]
        lowest_e = np.sort(e[e > 0])[:nroots]
        return lowest_e

    mf = scf.RHF(mol).run()

    print(mf.mo_energy*au2ev)

    # a, b = tddft.TDHF(mf).get_ab()
    # print('Direct diagoanlization:', diagonalize(a, b))
    td = tddft.TDHF(mf)
    td.singlet=False
    # td.verbose=6
    td.kernel(nstates=10)[0]
    td.analyze()
    # print(a.shape)

    tdhf = TDHF(mf)
    print('occ orbs = ', tdhf.nocc)
    tdhf.run(nstates=10)
    print(tdhf._a.shape)
