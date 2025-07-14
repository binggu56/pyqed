#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 14:08:29 2025


core functions for spinful fermion chains

@author: Bing Gu (gubing@westlake.edu.cn)

"""

from scipy.sparse.linalg import eigsh
from scipy.sparse import eye

from scipy.linalg import ishermitian

from pyqed import tensor, dag, isherm
from pyqed.phys import eigh
from pyqed.qchem.jordan_wigner.spinful import create, annihilate
import numpy as np

from opt_einsum import contract






def Is(l):
    """
    list of identity matrices

    Parameters
    ----------
    l : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if l > 0:
        return [eye(4), ] * l
    else:
        return []

class SpinHalfFermionChain:

    """
    exact diagonalization of spin-half open fermion chain with long-range interactions

    by Jordan-Wigner transformation

    .. math::

        H = \sum_{<rs>} (c_r^\dagger c_s + c†scr−γ(c†rc†s+cscr))−2λ \sum_r c^\dagger_r c_r,

    where r and s indicate neighbors on the chain.

    Electron interactions can be included in the Hamiltonian easily.

    """
    def __init__(self, h1e, eri, nelec=None):
        # if L is None:
        L = h1e.shape[-1]
        self.L = self.nsites = L

        self.h1e = h1e
        self.eri = eri
        self.d = 4 # local dimension of each site
        # self.filling = filling
        self.nelec = nelec


        self.H = None
        self.e_tot = None
        self.X = None # eigenstates
        self.operators = None # basic operators for a chain


        self.Cu = None
        self.Cd = None
        self.Cdd = None
        self.Cdu = None # C^\dag_\uparrow
        self.Nu_tot = None
        self.Nd_tot = None
        self.Ntot = None
        self.Sz = None
        self.Sx = None
        self.Sy = None
        self.Sp = None
        self.S2 = None

    def brute_force(self, nstates=1):

        if self.H is None:
            self.jordan_wigner()

        E, X = eigsh(self.H, k=nstates, which='SA')

        self.e_tot = E

        Nu = np.diag(dag(X) @ self.Nu_tot @ X)
        Nd = np.diag(dag(X) @ self.Nd_tot @ X)


        spin = np.real(np.diag(dag(X) @ self.S2 @ X))

        # spin = contract('ia, ij, ja -> a', X.conj(), self.S2, X)

        print('\n   Energy     Nu     Nd     S')
        for i in range(nstates):
            print('{:12.6f}  {:4.2f}   {:4.2f}  {:4.2f}'.format(E[i], Nu[i], Nd[i], spin[i]))

        return E, X


    def run(self, nstates=1):

        from pyqed.mps.abelian import ConservedSite

        # # single electron part
        # Ca = mf.mo_coeff[:, :self.ncas]
        # hcore_mo = contract('ia, ij, jb -> ab', Ca.conj(), mf.hcore, Ca)


        # eri = self.mf.eri
        # eri_mo = contract('ip, iq, ij, jr, js -> pqrs', Ca.conj(), Ca, eri, Ca.conj(), Ca)

        # # eri_mo = contract('ip, jq, ij, ir, js', mo.conj(), mo.conj(), eri, mo, mo)

        # self.hcore_mo = hcore_mo


        # Construc the full H and then truncate. This is NOT efficient.
        #TODO: Should apply truncation during the build
        if self.H is None:
            self.jordan_wigner()

        H = self.H

        if self.nelec is None:
            # build all possible quantum numbers

            # a = ConservedSite()
            s = ConservedSite()
            for n in range(int(np.log2(self.L))):

                s += s

            # H = self.jordan_wigner()


            e = []
            u = []
            for ne in s.qn:

                h = s.block(ne, H)

                _e, _u = eigh(h, k=nstates, which='SA')

                e.append(_e.copy())
                u.append(_u.copy())

                print('# electrons = {}, e = {}'.format(ne, _e))

            self.e_tot = _e
            self.X = u

        elif isinstance(self.nelec, (int, np.int16, np.int32, np.int64)):

            ###
            # H = model.H

            # print(H.toarray())

            # print(H[np.ix_(idx, idx)])
            s = ConservedSite()
            for n in range(int(np.log2(self.L))):
                s += s
            # e = []
            # u = []
            # for ne in s.qn:

            #     i = s.qn.index(self.nelec)


            H = s.block(self.nelec, self.H)

            _e, _u = eigsh(H, k=nstates, which='SA')

            # e.append(_e.copy())
            # u.append(_u.copy())

            print('e = {}'.format(_e))

            self.e_tot = _e
            self.X = _u

        elif isinstance(self.nelec, (list, tuple)):

            print('build state index with na, nb symmetry')
            na, nb = self.nelec

            sa = ConservedSite(qn=[0, 1], degeneracy=[2, 2], state_index=[[0, 2], [1, 3]],\
                               qmax=na)
            sb = ConservedSite(qn=[0, 1], degeneracy=[2, 2], state_index=[[0, 1], [2, 3]],\
                               qmax=nb)

            for i in range(int(np.log2(self.L))):

                print('sss')

                sa += sa
                sb += sb

            idxa = sa.ravel_index(na)
            idxb = sb.ravel_index(nb)

            idx = np.intersect1d(idxa, idxb)

            print('number of states = ', len(idx))

            H = self.H[np.ix_(idx, idx)]

            E, X = eigh(H, k=nstates, which='SA')

            self.e_tot = E
            self.X = X

        else:
            raise ValueError('nelec must be an interger.', self.nelec)



# print('Energies = ', E)

        return self

    def jordan_wigner(self, forward=True, aosym='8'):
        """
        MOs based on Restricted HF calculations

        Returns
        -------
        H : TYPE
            DESCRIPTION.
        aosym: int, AO symmetry
            8: eight-fold symmetry

        """
        h1e = self.h1e
        v = self.eri

        # an inefficient implementation without consdiering any syemmetry
        # can be used to compute triplet states

        nelec = self.nelec

        norb = h1e.shape[-1]
        nmo = L = norb # does not necesarrily have to MOs

        Cu = annihilate(norb, spin='up', forward=forward)
        Cd = annihilate(norb, spin='down', forward=forward)
        Cdu = create(norb, spin='up', forward=forward)
        Cdd = create(norb, spin='down', forward=forward)


        self.Cu = Cu
        self.Cd = Cd
        self.Cdu = Cdu
        self.Cdd = Cdd

        Sz = 0
        Sy = 0
        Sx = 0
        Sp = 0
        for p in range(nmo):
            Sz += 0.5 * (Cdu[p] @ Cu[p] - Cdd[p] @ Cd[p])
            Sx += 0.5 * (Cdu[p] @ Cd[p] + Cdd[p] @ Cu[p])
            Sy += -0.5j * (Cdu[p] @ Cd[p] - Cdd[p] @ Cu[p])

            Sp += Cdu[p] @ Cd[p]

        # print(ishermitian(Sy.toarray()))

        self.Sx = Sx
        self.Sy = Sy
        self.Sz = Sz
        self.Sp = Sp # S^+

        self.S2 = Sx @ Sx + Sy @ Sy + Sz @ Sz

        H = 0
        # for p in range(nmo):
        #     for q in range(p+1):
                # H += jordan_wigner_one_body(q, p, hcore_mo[q, p], hc=True)
        for p in range(nmo):
            for q in range(nmo):
                H += h1e[p, q] * (Cdu[p] @ Cu[q] + Cdd[p] @ Cd[q])

        # build total number operator
        # number_operator = 0
        Na = 0
        Nb = 0
        for p in range(L):
            Na += Cdu[p] @ Cu[p]
            Nb += Cdd[p] @ Cd[p]

        self.Nu_tot = Na
        self.Nd_tot = Nb


        # poor man's implementation of JWT for 2e operators wihtout exploiting any symmetry
        for p in range(nmo):
            for q in range(nmo):
                for r in range(nmo):
                    for s in range(nmo):
                        H += 0.5 * v[p, q, r, s] * (\
                            Cdu[p] @ Cdu[r] @ Cu[s] @ Cu[q] +\
                            Cdu[p] @ Cdd[r] @ Cd[s] @ Cu[q] +\
                            Cdd[p] @ Cdu[r] @ Cu[s] @ Cd[q] +
                            Cdd[p] @ Cdd[r] @ Cd[s] @ Cd[q])
                        # H += jordan_wigner_two_body(p, q, s, r, )

        # digonal elements for p = q, r = s
        self.H = H

        return H

    # def jordan_wigner(self):
    #     """
    #     MOs based on Restricted HF calculations

    #     Returns
    #     -------
    #     H : TYPE
    #         DESCRIPTION.

    #     """
    #     # an inefficient implementation without consdiering any syemmetry

    #     from pyqed.qchem.jordan_wigner.spinful import jordan_wigner_one_body, annihilate, \
    #         create, Is #, jordan_wigner_two_body

    #     nelec = self.nelec
    #     h1e = self.h1e
    #     v = self.eri


    #     norb = h1e.shape[-1]
    #     nmo = L = norb # does not necesarrily have to MOs


    #     Cu = annihilate(norb, spin='up')
    #     Cd = annihilate(norb, spin='down')
    #     Cdu = create(norb, spin='up')
    #     Cdd = create(norb, spin='down')

    #     self.Cu = Cu
    #     self.Cd = Cd
    #     self.Cdu = Cdu
    #     self.Cdd = Cdd

    #     Sz = 0
    #     Sy = 0
    #     Sx = 0
    #     Sp = 0
    #     for p in range(nmo):
    #         Sz += 0.5 * (Cdu[p] @ Cu[p] - Cdd[p] @ Cd[p])
    #         Sx += 0.5 * (Cdu[p] @ Cd[p] + Cdd[p] @ Cu[p])
    #         Sy += -0.5j * (Cdu[p] @ Cd[p] - Cdd[p] @ Cu[p])

    #         Sp += Cdu[p] @ Cd[p]

    #     # print(ishermitian(Sy.toarray()))

    #     self.Sx = Sx
    #     self.Sy = Sy
    #     self.Sz = Sz
    #     self.Sp = Sp # S^+

    #     self.S2 = Sx @ Sx + Sy @ Sy + Sz @ Sz

    #     H = 0
    #     # for p in range(nmo):
    #     #     for q in range(p+1):
    #             # H += jordan_wigner_one_body(q, p, hcore_mo[q, p], hc=True)
    #     for p in range(nmo):
    #         for q in range(nmo):
    #             H += h1e[p, q] * (Cdu[p] @ Cu[q] + Cdd[p] @ Cd[q])

    #     # build total number operator
    #     # number_operator = 0
    #     Na = 0
    #     Nb = 0
    #     for p in range(L):
    #         Na += Cdu[p] @ Cu[p]
    #         Nb += Cdd[p] @ Cd[p]
    #     Ntot = Na + Nb

    #     self.Nu_tot = Na
    #     self.Nd_tot = Nb
    #     self.Ntot = Ntot

    #     # poor man's implementation of JWT for 2e operators wihtout exploiting any symmetry
    #     for p in range(nmo):
    #         for q in range(nmo):
    #             for r in range(nmo):
    #                 for s in range(nmo):
    #                     H += 0.5 * v[p, q, r, s] * (\
    #                         Cdu[p] @ Cdu[r] @ Cu[s] @ Cu[q] +\
    #                         Cdu[p] @ Cdd[r] @ Cd[s] @ Cu[q] +\
    #                         Cdd[p] @ Cdu[r] @ Cu[s] @ Cd[q] +
    #                         Cdd[p] @ Cdd[r] @ Cd[s] @ Cd[q])
    #                     # H += jordan_wigner_two_body(p, q, s, r, )

    #     # digonal elements for p = q, r = s
    #     # if self.nelec is not None:
    #     #     I = tensor(Is(L))

    #     #     H += 0.2* ((Na - nelec/2 * I) @ (Na - self.nelec/2 * I) + \
    #     #         (Nb - self.nelec/2 * I) @ (Nb - self.nelec/2 * I))
    #     self.H = H

    #     self.operators = {"H": H,
    #            "Cd": Cd,
    #            "Cu": Cu,
    #            "Cdd": Cdd,
    #            "Cdu": Cdu,
    #            "Nu" : Na,
    #            "Nd" : Nb,
    #            "Ntot": Ntot
    #            }

    #     return H

    def fix_nelec(self, nelec=None, s=1):

        if self.H is None:
            self.build()

        I = tensor(Is(self.L))

        Na = self.Nu_tot
        Nb = self.Nd_tot

        if nelec is None:
            nelec = self.nelec

        self.H += s * (Na - nelec/2 * I) @ (Na - nelec/2 * I) + \
                s * (Nb - self.nelec/2 * I) @ (Nb - self.nelec/2 * I)
        return

    def DMRG(self):
        # build the MPO of H and then apply the DMRG algorithm
        pass
        # return DMRG(H, D)


    def gen_mps(self):
        pass


if __name__=='__main__':
    from pyscf import gto, scf, dft, tddft, ao2mo
    from pyqed.qchem import get_hcore_mo, get_eri_mo
    # from pyqed.qchem.gto.rhf import RHF
    from pyqed.qchem.mol import atomic_chain

    # mol = gto.Mole()
    # mol.atom = [
    #     ['H' , (0. , 0. , .917)],
    #     ['H' , (0. , 0. , 0.)], ]
    # mol.basis = '6311g'
    # mol.build()

    natom = 8
    z = np.linspace(-3, 3, natom)
    mol = atomic_chain(natom, z)
    mol.basis = 'sto6g'
    mol.build()
    # mf = scf.RHF(mol).run()

    print(type(mol.nelec))

    mf = mol.RHF().run()

    print('number of electrons', mol.nelec)
    print('number of orbs = ', mol.nao)

    # e, fcivec = pyscf.fci.FCI(mf).kernel(verbose=4)
    # print(e)
    # Ca = mf.mo_coeff[0ArithmeticError
    # n = Ca.shape[-1]

    # mo_coeff = mf.mo_coeff
    # get the two-electron integrals as a numpy array
    # eri = get_eri_mo(mol, mo_coeff)

    # n = mol.nao
    # Ca = mo_coeff

    # h1e = get_hcore_mo(mf)
    # eri = get_eri_mo(mf)

    # print(mol.nelec)
    # model = SpinHalfFermionChain(h1e, eri).run(3)


    h1e = mf.get_hcore_mo()
    eri = mf.get_eri_mo()


    model = SpinHalfFermionChain(h1e, eri, [mol.nelec//2, mol.nelec//2])

    model.run(10)



    print(model.e_tot + mol.energy_nuc())