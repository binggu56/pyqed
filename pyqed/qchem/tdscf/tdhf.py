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
from pyscf import ao2mo, scf

from functools import reduce
import logging

au2ev = 27.211386245988


def is_positive_def(a):
    vals = np.linalg.eigvalsh(np.asarray(a))
    return np.all(vals > 0)


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


def _ov_blocks(gw):
    mo_energy = np.asarray(gw._scf.mo_energy)
    mo_coeff = np.asarray(gw._scf.mo_coeff)
    mo_occ = np.asarray(gw._scf.mo_occ)

    occidx = np.where(mo_occ > 0)[0]
    viridx = np.where(mo_occ == 0)[0]

    orbo = mo_coeff[:, occidx]
    orbv = mo_coeff[:, viridx]
    return mo_energy, occidx, viridx, orbo, orbv


def get_ab(gw, method='TDH', singlet=True):
    '''Compute restricted RHF A/B matrices in the occupied/virtual response space.'''
    assert method in ('TDH', 'TDHF', 'TDDFT')
    if method == 'TDDFT':
        raise NotImplementedError('TDDFT is not implemented in this legacy TDHF module.')

    mo_energy, occidx, viridx, orbo, orbv = _ov_blocks(gw)
    nocc = len(occidx)
    nvir = len(viridx)
    dim_rpa = nocc * nvir
    logging.info('dim of AB matrices = {}'.format(dim_rpa))

    e_ia = mo_energy[viridx] - mo_energy[occidx, None]
    a = np.diag(e_ia.ravel()).reshape(nocc, nvir, nocc, nvir)
    b = np.zeros_like(a)

    # Coulomb block J_{ia,jb} = (ia|jb)
    eri_iajb = ao2mo.general(
        gw.mol,
        (orbo, orbv, orbo, orbv),
        compact=False,
    ).reshape(nocc, nvir, nocc, nvir)

    # Exchange blocks written in occupied/virtual order.
    eri_ijab = ao2mo.general(
        gw.mol,
        (orbo, orbo, orbv, orbv),
        compact=False,
    ).reshape(nocc, nocc, nvir, nvir)
    k_a = np.transpose(eri_ijab, (0, 2, 1, 3))

    eri_jaib = ao2mo.general(
        gw.mol,
        (orbo, orbv, orbo, orbv),
        compact=False,
    ).reshape(nocc, nvir, nocc, nvir)
    k_b = np.transpose(eri_jaib, (2, 1, 0, 3))

    if singlet:
        a += 2.0 * eri_iajb
        b += 2.0 * eri_iajb

    if method == 'TDHF':
        a -= k_a
        b -= k_b

    a = a.reshape(dim_rpa, dim_rpa)
    b = b.reshape(dim_rpa, dim_rpa)
    assert np.allclose(a, a.transpose())
    assert np.allclose(b, b.transpose())
    return a, b

class TDH:
    '''
    Time-dependent Hartree 
    '''
    def __init__(self):
        pass
    
class TDHF:
    def __init__(self, mf):

        self.mol = mf.mol
        self._scf  = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory
        self.spin = 0
        self.singlet = True

        self._a = None
        self._b = None
        self.e = None
        self.xy = None

        if isinstance(mf, scf.rhf.RHF):
            self.e_mf = np.asarray(mf.mo_energy)
            self.nocc = self.mol.nelectron // 2
            self.nso = len(self.e_mf)
        else:
            raise NotImplementedError("\n*** Only supporting restricted calculations right now! ***\n")
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

    def run(self, nstates=None, using_tda=False, using_casida=True, method='TDHF', singlet=None):
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
        if singlet is None:
            singlet = self.singlet

        A, B = self.get_ab(method=method, singlet=singlet)

        if using_tda:
            logging.info('Using TDA approximation')
            e, x = eig(A, k=nstates, which='SA' if isinstance(nstates, int) else None)
            self.e = e
            self.xy = x
            return e, x
        
        else:
            if using_casida:
                assert is_positive_def(A-B)
                sqrt_A_minus_B = sqrtm(A-B)
                ham_rpa = np.dot(sqrt_A_minus_B, np.dot((A+B),sqrt_A_minus_B))

                if nstates is not None:
                    esq, t = eig(ham_rpa, k=nstates, which='SA')
                else:
                    esq, t = eigh(ham_rpa)
                e = np.sqrt(esq)
                self.e = e
                self.xy = t
                return e, t

            else:
                ham_rpa = np.array(np.bmat([[A,B],[-B,-A]]))
                e, xy = eig_asymm(ham_rpa)
                e = e[e > 1e-8]
                if nstates is not None:
                    e = e[:nstates]
                    xy = xy[:, :nstates]
                self.e = e
                self.xy = xy
                return e, xy




    def get_ab(self, method='TDHF', singlet=None):
        if singlet is None:
            singlet = self.singlet
        a, b = get_ab(self, method=method, singlet=singlet)
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
    # td = tddft.TDHF(mf)
    # td.singlet=True
    # # td.verbose=6
    # td.kernel(nstates=10)[0]
    # td.analyze()


    tdhf = TDHF(mf)
    print('occ orbs = ', tdhf.nocc)
    tdhf.run(nstates=10)
    print(tdhf._a.shape)
