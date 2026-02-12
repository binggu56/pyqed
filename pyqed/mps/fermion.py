#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 14:08:29 2025


core functions for spinful fermion chains

@author: Bing Gu (gubing@westlake.edu.cn)

"""

from scipy.sparse.linalg import eigsh
from scipy.sparse import kron, eye, csr_matrix, issparse

from scipy.linalg import ishermitian

from pyqed import tensor, dag, isherm, SpinHalfFermionOperators, sort

from pyqed.mps.abelian import ConservedSite
from pyqed.phys import eigh
from pyqed.qchem.jordan_wigner.spinful import create, annihilate
import numpy as np

from opt_einsum import contract
import logging



class SpinOrbital(ConservedSite):
    """A site for spin-half electronic models

    You use this site for models where the single sites are electron
    sites. The Hilbert space is ordered such as:

    - the first state, labelled 0,  is the empty site,
    - the second, labelled 1, is spin down,
    - the third, labelled 2, is spin up, and
    - the fourth, labelled 3, is double occupancy.

    Notes
    -----
    Postcond: The site has already built-in the spin operators for:

    - c_up : destroys an spin up electron,
    - c_up_dag, creates an spin up electron,
    - c_down, destroys an spin down electron,
    - c_down_dag, creates an spin down electron,
    - s_z, component z of spin,
    - s_p, raises the component z of spin,
    - s_m, lowers the component z of spin,
    - n_up, number of electrons with spin up,
    - n_down, number of electrons with spin down,
    - n, number of electrons, i.e. n_up+n_down, and
    - u, number of double occupancies, i.e. n_up*n_down.

    """
    def __init__(self, H=None):
        super(SpinOrbital, self).__init__()

        self.operators = SpinHalfFermionOperators()
        self.H = H

        # 	# add the operators
        # self.add_operator("c_up")
        # self.add_operator("c_up_dag")
        # self.add_operator("c_down")
        # self.add_operator("c_down_dag")
        # self.add_operator("s_z")
        # self.add_operator("s_p")
        # self.add_operator("s_m")
        # self.add_operator("n_up")
        # self.add_operator("n_down")
        # self.add_operator("n")
        # self.add_operator("u")

        # 	# for clarity
        # c_up = self.operators["c_up"]
        # c_up_dag = self.operators["c_up_dag"]
        # c_down = self.operators["c_down"]
        # c_down_dag = self.operators["c_down_dag"]
        # s_z = self.operators["s_z"]
        # s_p = self.operators["s_p"]
        # s_m = self.operators["s_m"]
        # n_up = self.operators["n_up"]
        # n_down = self.operators["n_down"]
        # n = self.operators["n"]
        # u = self.operators["u"]
        # 	# set the matrix elements different from zero to the right values
        # 	# TODO: missing s_p, s_m
        # c_up[0,2] = 1.0
        # c_up[1,3] = 1.0
        # c_up_dag[2,0] = 1.0
        # c_up_dag[3,1] = 1.0
        # c_down[0,1] = 1.0
        # c_down[2,3] = 1.0
        # c_down_dag[1,0] = 1.0
        # c_down_dag[3,2] = 1.0
        # s_z[1,1] = -1.0
        # s_z[2,2] = 1.0
        # n_up[2,2] = 1.0
        # n_up[3,3] = 1.0
        # n_down[1,1] = 1.0
        # n_down[3,3] = 1.0
        # n[1,1] = 1.0
        # n[2,2] = 1.0
        # n[3,3] = 2.0
        # u[3,3] = 1.0

    def __add__(self):
        # build the qn

        # block the Hamiltonian
        pass

    def add_coupling(self):
        pass

def Is(l, d=4):
    """
    list of identity matrices of dimension d

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
        return [eye(d), ] * l
    else:
        return []



def concatenate(list_of_array):
#    assemble an array of subblock matrices into a total one
    pass

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

        self.block = None # ConservedSite

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

    def full_diagonalization(self, nstates=1):
        return self.brute_force(nstates)

    def brute_force(self, nstates=1):

        if self.H is None:
            self.jordan_wigner()

        E, X = eigsh(self.H, k=nstates, which='SA')

        self.e_tot = E

        self.X = X

        Nu = np.diag(dag(X) @ self.Nu_tot @ X)
        Nd = np.diag(dag(X) @ self.Nd_tot @ X)


        spin = np.real(np.diag(dag(X) @ self.S2 @ X))

        # spin = contract('ia, ij, ja -> a', X.conj(), self.S2, X)

        print('\n   Energy     Nu     Nd     SS')
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

            self.e_tot = e
            self.X = u

            # create a block
            self.block = s

            self.block.e_tot = e
            self.block.operators = self.operators

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

            print('\nExact diagonalization with Sz U(1) symmetry')
            na, nb = self.nelec

            sa = ConservedSite(qn=[0, 1], degeneracy=[2, 2], state_index=[[0, 2], [1, 3]],\
                               qmax=na)
            sb = ConservedSite(qn=[0, 1], degeneracy=[2, 2], state_index=[[0, 1], [2, 3]],\
                               qmax=nb)

            for i in range(int(np.log2(self.L))):

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

        self.operators = {'Cu': Cu, 'Cd': Cd, 'Cdu': Cdu, 'Cdd': Cdd}

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


class NARG(ConservedSite):
    def __init__(self, h1e, eri, D):

        super(NARG, self).__init__()

        self.h1e = h1e
        self.eri = eri

        self.L = h1e.shape[-1]
        self.D = D

        self.H = None
        self.e_tot = None
        self.X = None # eigenstates
        self.operators = None # basic operators for a chain

        self.site = None # ConservedSite

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

    def initialize(self, n0=3):
        """
        initiate the NARG calculation by an exact diagonalization within a small number of
        orbitals

        D retained adiabatic eigenstates

        Parameters
        ----------
        n0 : TYPE, optional
            DESCRIPTION. The default is 3.

        Returns
        -------
        None.

        """

        nstart = n0
        h1e = self.h1e[:nstart, :nstart]
        eri = self.eri[:nstart, :nstart, :nstart, :nstart]

        # initial block containing the n0 lowest energy MOs
        block = SpinHalfFermionChain(h1e, eri)
        block.jordan_wigner(forward=False)
        block.run(self.D)

        self.block = block.block

        return self.block

    def truncate(self, k=None):

        # truncate the states by energy

        # if self.site is None:
        #     raise ValueError('This is not a Site yet.')
        # s = self.site

        if k is None:
            k = self.D


        e_sort = []
        for e in self.e_tot:
            e_sort += list(e)

        # print(e_sort.shape)
        e_sort.sort()

        cutoff = e_sort[k]

        print('Truncate states with energy higher than', cutoff)

        degeneracy = []
        energy = []
        qn = []
        state_index = []

        assert(isinstance(self.e_tot, list))

        for i, _e in enumerate(self.e_tot):

            _e_truncated = _e[_e < cutoff]

            _d = len(_e_truncated)

            if _d == len(_e):
                print('No truncation for electron number {} block. Suggest increasing the  corresponding D.'.format(self.qn[i]))
                #TODO: Increase D and redo the computation for i-th block

            if _d > 0:

                n = sum(degeneracy)
                state_index.append(list(range(n, n + _d)))

                energy += [_e_truncated]
                degeneracy += [_d]
                qn += [self.qn[i]]

        # self.site = ConservedSite(qn=qn, degeneracy=degeneracy, state_index=state_index)


        # update the block information
        self.block.qn = qn
        self.block.degeneracy = degeneracy
        self.block.state_index = state_index
        self.block.e_tot = energy

        # rotate all operators to this truncated space

        return self




    def grow(self, site=None):
        # grow by one site and update the Hamiltonian
        pass

    def single_site(self, n):


        return


    def run(self, n0=4):

        # diagonalization of the initial block
        nstart = n0
        h1e = self.h1e
        eri = self.eri


        # initial block containing the n0 lowest energy MOs
        # model = SpinHalfFermionChain(self.h1e[:nstart, :nstart], self.eri[:nstart, :nstart, :nstart, :nstart])
        # model.jordan_wigner(forward=False)
        # model.run(self.D)

        # block = model.block
        block = self.initialize(n0)


        Cdu = block.operators['Cdu']
        Cdd = block.operators['Cdd']
        Cu = block.operators['Cu']
        Cd = block.operators['Cd']

        ops = SpinHalfFermionOperators()
        cd = ops['Cd']
        cu = ops['Cu']
        cdu = ops['Cdu']
        cdd = ops['Cdd']
        JW = ops['JW']
        Ntot = ops['Ntot']
        Nu = ops['Nu']
        Nd = ops['Nd']

        p = n0

        # add the pth site

        def single_site_hamiltonian(n):
            return h1e[n,n] * (cdu @ cu + cdd @ cd) + eri[n, n, n, n] * Nu @ Nd

        h = single_site_hamiltonian(p)
        so = SpinOrbital(h)

        print(block.__str__())

        block = truncate(block, self.D)

        print(block.__str__())

        block = block + so # add the p-th orbital

        print(block.__str__())


        # # the adaibatic states at |\uparrow>

        # # Cdu = model.Cdu
        # # Cdd = model.Cdd
        # # Cu = model.Cu
        # # Cd = model.Cd

        # nu = 1
        # nd = 0

        # ### add all interaction between previous sites (0,1,...n-1) and the new site (n)

        # # two-operator \sum_{i, j < p} v[i,j,p,p] - v[i, p, p, j] * (nu + nd)
        # H = H0.copy()
        # for i in range(nstart):
        #     for j in range(nstart):
        #         H += v[i,j,p,p] * (nu + nd) * (Cdu[i] @ Cu[j] + Cdd[i] @  Cd[j])
        #         H -= v[i, p, p, j] * (nu * Cdu[i] @ Cu[j] + nd * Cdd[i] @  Cd[j])

        # E1, U1 = eigh(H, k=D)
        # # print(E1)

        # # the adaibatic states at |\downarrow>
        # nd = 1
        # nu = 0

        # H2 = H0.copy()
        # for i in range(nstart):
        #     for j in range(nstart):
        #         H2 += v[i,j,p,p] * (nu + nd) * (Cdu[i] @ Cu[j] + Cdd[i] @  Cd[j])
        #         H2 -= v[i, p, p, j] * (nu * Cdu[i] @ Cu[j] + nd * Cdd[i] @  Cd[j])

        # E2, U2 = eigh(H2, k=D)
        # # print(E2)

        # # the adaibatic states at |\uparrow \downarrow>
        # nu = 1
        # nd = 1

        # H3 = H0.copy()
        # for i in range(nstart):
        #     for j in range(nstart):
        #         H3 += v[i,j,p,p] * (nu + nd) * (Cdu[i] @ Cu[j] + Cdd[i] @  Cd[j])
        #         H3 -= v[i, p, p, j] * (nu * Cdu[i] @ Cu[j] + nd * Cdd[i] @  Cd[j])

        # E3, U3 = eigh(H3, k=D)
        # # print(E3)


        # d = 4 # local dim
        # E = np.zeros((d, min(D, d**nstart)))
        # U = np.zeros((d**nstart, min(D, d**nstart), d))

        # E[0, :] = E0 + h[0, 0]
        # E[1, :] = E1 + h[1, 1]
        # E[2, :] = E2 + h[2, 2]
        # E[3, :] = E3 + h[3, 3]

        # # print('E = ', E)

        # U[:, :, 0] = U0
        # U[:, :, 1] = U1
        # U[:, :, 2] = U2
        # U[:, :, 3] = U3

        # Es = [E] # adiabatic energies
        # Cs = [U] # core tensors

        # # build total Hamiltonian for 123 + 4

        # # adiabatic H + diagonal part of h4
        # # S = contract('ibm,  ian -> mbna', U.conj(), U)

        # # residual interactions including a_p, a_p^dag a_p a_p

        # Htot = np.diag(E.reshape((D * d)))

        # # c_p V1, V2, V3
        # v1u = 0
        # v1d = 0


        # for i in range(nstart):

        #     v1u = v1u + h1e[i, p] * Cdu[i]
        #     v1d = v1d + h1e[i, p] * Cdd[i]

        # # print('v1u', v1u, v1d)

        # for i in range(nstart):
        #     for j in range(nstart):
        #         for k in range(nstart):
        #             v1u += eri[k,p,j,i] * Cdu[k] @ (Cdu[j] @ Cu[i] + Cdd[j] @ Cd[i])
        #             v1d += eri[k,p,j,i] * Cdd[k] @ (Cdu[j] @ Cu[i] + Cdd[j] @ Cd[i])

        # # jw_string = tensor([JW, ] * n0)

        # V1u =  contract('ibm, ij, jan -> mbna', U.conj(), v1u.toarray() , U)
        # V1d =  contract('ibm, ij, jan -> mbna', U.conj(), v1d.toarray(), U)

        # # Cdu_p = create(p, spin='up')[-1]
        # # Cdd_p = create(p, spin='down')[-1]

        # # print(cu, cd)

        # V1 = contract('mbna, mn -> mbna', V1u, JW @ cu).reshape((d*D, d*D)) + \
        #     contract('mbna, mn -> mbna', V1d, JW @ cd).reshape((d*D, d*D))

        # # print('V1', V1)

        # Htot += V1 + dag(V1) # this is not correct? I have to consider the JW string for Cp!

        # # V2 term
        # v2a = 0
        # for i in range(nstart):
        #     for j in range(nstart):
        #         v2a += -eri[i, p, p, j] * Cdd[i] @ Cu[j]

        # v2b = 0
        # for i in range(n0):
        #     for j in range(n0):
        #         v2b += 0.5 * eri[p,i,p,j] * (Cd[i] @ Cu[j] - Cu[i] @ Cd[j])

        # # print(dag(U) @ (Cdd+ Cd) @ U)

        # V2 = contract('ibm, ij, jan -> mbna', U.conj(), v2a.toarray(), U)
        # H2a = contract('mbna, mn -> mbna', V2, cdu @ cd).reshape((d*D, d*D))

        # V2b = contract('ibm, ij, jan -> mbna', U.conj(), v2b.toarray(), U)
        # H2b = contract('mbna, mn -> mbna', V2b, cdu @ cdd).reshape((d*D, d*D))

        # # print('V2', H2a, H2b)
        # Htot += H2a + dag(H2a) + H2b + dag(H2b)


        # ## V3 (V3 can be combined with V1)
        # v3u = 0
        # v3d = 0
        # for i in range(n0):
        #     v3u += eri[i, p, p, p] * Cdu[i]
        #     v3d += eri[i, p, p, p] * Cdd[i]

        # V3u =  contract('ibm, ij, jan -> mbna', U.conj(), v3u.toarray(), U)
        # V3d =  contract('ibm, ij, jan -> mbna', U.conj(), v3d.toarray(), U)

        # # Cu_p = annihilate(p, spin='up')[-1]
        # # Cd_p = annihilate(p, spin='down')[-1]


        # H3 = contract('mbna, mn -> mbna', V3u, JW @ Nd @ cu).reshape((d*D, d*D)) + \
        #     contract('mbna, mn -> mbna', V3d, JW @ Nu @ cd).reshape((d*D, d*D))

        # Htot += H3 + dag(H3)

def rotate(A, B, U):
    """
    rotate :math:`V = A \otimes B` to the adiabatic representation
    :math:`|\phi_{\alpha_l n_{l+1}} \rangle \otimes | n_{l+1} \rangle

    Parameters
    ----------
    A : TYPE
        operator in the first l sites
    B : TYPE
        operator in the l+1 site
    U : ndarray [primitive basis index, adiabatic state index, (l+1)th site state index]
        DESCRIPTION.

    Returns
    -------
    v : TYPE
        DESCRIPTION.

    """
    n, D, d = U.shape

    assert(n == A.shape[0])
    assert(d == B.shape[1])

    if issparse(A):
        A = A.toarray()

    _v = contract('ibm, ij, jan -> mbna', U.conj(), A, U)
    v = contract('mbna, mn -> mbna', _v, B).reshape((d*D, d*D))
    return v


def truncate(block, k):

    # truncate the states by energy

    # if self.site is None:
    #     raise ValueError('This is not a Site yet.')
    # s = self.site

    e_tot = block.e_tot

    e_sort = []
    for e in e_tot:
        e_sort += list(e)

    # print(e_sort.shape)
    e_sort.sort()

    cutoff = e_sort[k]

    print('Truncate states with energy higher than', cutoff)

    degeneracy = []
    energy = []
    qn = []
    state_index = []

    assert(isinstance(e_tot, list))

    for i, _e in enumerate(e_tot):

        _e_truncated = _e[_e < cutoff]

        _d = len(_e_truncated)

        if _d == len(_e):
            print('No truncation for electron number {} block. Suggest increasing the  corresponding D.'.format(block.qn[i]))
            #TODO: Increase D and redo the computation for i-th block

        if _d > 0:

            n = sum(degeneracy)
            state_index.append(list(range(n, n + _d)))

            energy += [_e_truncated]
            degeneracy += [_d]
            qn += [block.qn[i]]

    # self.site = ConservedSite(qn=qn, degeneracy=degeneracy, state_index=state_index)


    # update the block information
    block.qn = qn
    block.degeneracy = degeneracy
    block.state_index = state_index
    block.e_tot = energy

    # rotate all operators to this truncated space

    return block

def kernel(h1e, eri, D=20, n0=4, nstates=1):
    """
    NARG for interacting electrons WIHTOUT exploiting any symmetry

    Parameters
    ----------
    h1e : TYPE
        DESCRIPTION.
    eri : TYPE
        DESCRIPTION.
    D : TYPE, optional
        DESCRIPTION. The default is 20.
    n0 : TYPE, optional
        DESCRIPTION. The default is 4.
    nstates : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    L = h1e.shape[-1]

    v = eri

    ops = SpinHalfFermionOperators()
    cd = ops['Cd']
    cu = ops['Cu']
    cdu = ops['Cdu']
    cdd = ops['Cdd']
    JW = ops['JW']
    Ntot = ops['Ntot']
    Nu = ops['Nu']
    Nd = ops['Nd']


    # initiate the block with l0 Spin-Orbitals
    nstart = n0
    model = SpinHalfFermionChain(h1e[:nstart, :nstart], eri[:nstart, :nstart, :nstart, :nstart],
                                 nelec=mol.nelec)

    model.jordan_wigner(forward=False)

    E0, U0 = model.brute_force(nstates=D)

    print('Initial block energy = ', E0)

    H0 = model.H



    def single_site_hamiltonian(n):
        """
        Hamiltonian for a single spin-orbital

        Parameters
        ----------
        n : TYPE
            orbital ID. Starting from zero.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        return h1e[n,n] * (cdu @ cu + cdd @ cd) + eri[n, n, n, n] * Nu @ Nd



    p = nstart
    # add the pth site
    h = single_site_hamiltonian(p)
    # assert(isdiag(h))
    print('site', p, ' H = ', np.diag(h))

    # the adaibatic states at |\uparrow>
    # psi = np.array([0, 1., 0, 0])

    # nu = obs(psi, Nu) # expect 1
    # nd = obs(psi, Nd) # expect 0

    Cdu = model.Cdu
    Cdd = model.Cdd
    Cu = model.Cu
    Cd = model.Cd

    nu = 1
    nd = 0

    ### add all interaction between previous sites (0,1,...n-1) and the new site (n)

    # two-operator \sum_{i, j < p} v[i,j,p,p] - v[i, p, p, j] * (nu + nd)
    H = H0.copy()
    for i in range(nstart):
        for j in range(nstart):
            H += v[i,j,p,p] * (nu + nd) * (Cdu[i] @ Cu[j] + Cdd[i] @  Cd[j])
            H -= v[i, p, p, j] * (nu * Cdu[i] @ Cu[j] + nd * Cdd[i] @  Cd[j])

    E1, U1 = eigh(H, k=D)
    # print(E1)

    # the adaibatic states at |\downarrow>
    nd = 1
    nu = 0

    H2 = H0.copy()
    for i in range(nstart):
        for j in range(nstart):
            H2 += v[i,j,p,p] * (nu + nd) * (Cdu[i] @ Cu[j] + Cdd[i] @  Cd[j])
            H2 -= v[i, p, p, j] * (nu * Cdu[i] @ Cu[j] + nd * Cdd[i] @  Cd[j])

    E2, U2 = eigh(H2, k=D)
    # print(E2)

    # the adaibatic states at |\uparrow \downarrow>
    nu = 1
    nd = 1

    H3 = H0.copy()
    for i in range(nstart):
        for j in range(nstart):
            H3 += v[i,j,p,p] * (nu + nd) * (Cdu[i] @ Cu[j] + Cdd[i] @  Cd[j])
            H3 -= v[i, p, p, j] * (nu * Cdu[i] @ Cu[j] + nd * Cdd[i] @  Cd[j])

    E3, U3 = eigh(H3, k=D)
    # print(E3)


    d = 4 # local dim
    E = np.zeros((d, min(D, d**nstart)))
    U = np.zeros((d**nstart, min(D, d**nstart), d))

    E[0, :] = E0 + h[0, 0]
    E[1, :] = E1 + h[1, 1]
    E[2, :] = E2 + h[2, 2]
    E[3, :] = E3 + h[3, 3]

    # print('E = ', E)

    U[:, :, 0] = U0
    U[:, :, 1] = U1
    U[:, :, 2] = U2
    U[:, :, 3] = U3

    Es = [E] # adiabatic energies
    Cs = [U] # core tensors

    # build total Hamiltonian for 123 + 4

    # adiabatic H + diagonal part of h4
    # S = contract('ibm,  ian -> mbna', U.conj(), U)

    # residual interactions including a_p, a_p^dag a_p a_p

    Htot = np.diag(E.reshape((D * d)))

    # c_p V1, V2, V3
    v1u = 0
    v1d = 0


    for i in range(nstart):

        v1u = v1u + h1e[i, p] * Cdu[i]
        v1d = v1d + h1e[i, p] * Cdd[i]

    # print('v1u', v1u, v1d)

    for i in range(nstart):
        for j in range(nstart):
            for k in range(nstart):
                v1u += eri[k,p,j,i] * Cdu[k] @ (Cdu[j] @ Cu[i] + Cdd[j] @ Cd[i])
                v1d += eri[k,p,j,i] * Cdd[k] @ (Cdu[j] @ Cu[i] + Cdd[j] @ Cd[i])

    # jw_string = tensor([JW, ] * n0)


    # V1u =  contract('ibm, ij, jan -> mbna', U.conj(), (v1u @ jw_string).toarray() , U)
    # V1d =  contract('ibm, ij, jan -> mbna', U.conj(), (v1d @ jw_string).toarray(), U)

    V1u =  contract('ibm, ij, jan -> mbna', U.conj(), v1u.toarray() , U)
    V1d =  contract('ibm, ij, jan -> mbna', U.conj(), v1d.toarray(), U)

    # Cdu_p = create(p, spin='up')[-1]
    # Cdd_p = create(p, spin='down')[-1]

    # print(cu, cd)

    V1 = contract('mbna, mn -> mbna', V1u, JW @ cu).reshape((d*D, d*D)) + \
        contract('mbna, mn -> mbna', V1d, JW @ cd).reshape((d*D, d*D))

    # print('V1', V1)

    Htot += V1 + dag(V1) # this is not correct? I have to consider the JW string for Cp!

    # V2 term
    v2a = 0
    for i in range(nstart):
        for j in range(nstart):
            v2a += -eri[i, p, p, j] * Cdd[i] @ Cu[j]

    v2b = 0
    for i in range(n0):
        for j in range(n0):
            v2b += 0.5 * eri[p,i,p,j] * (Cd[i] @ Cu[j] - Cu[i] @ Cd[j])

    # print(dag(U) @ (Cdd+ Cd) @ U)

    V2 = contract('ibm, ij, jan -> mbna', U.conj(), v2a.toarray(), U)
    H2a = contract('mbna, mn -> mbna', V2, cdu @ cd).reshape((d*D, d*D))

    V2b = contract('ibm, ij, jan -> mbna', U.conj(), v2b.toarray(), U)
    H2b = contract('mbna, mn -> mbna', V2b, cdu @ cdd).reshape((d*D, d*D))

    # print('V2', H2a, H2b)
    Htot += H2a + dag(H2a) + H2b + dag(H2b)


    ## V3 (V3 can be combined with V1)
    v3u = 0
    v3d = 0
    for i in range(n0):
        v3u += eri[i, p, p, p] * Cdu[i]
        v3d += eri[i, p, p, p] * Cdd[i]

    V3u =  contract('ibm, ij, jan -> mbna', U.conj(), v3u.toarray(), U)
    V3d =  contract('ibm, ij, jan -> mbna', U.conj(), v3d.toarray(), U)

    # Cu_p = annihilate(p, spin='up')[-1]
    # Cd_p = annihilate(p, spin='down')[-1]


    H3 = contract('mbna, mn -> mbna', V3u, JW @ Nd @ cu).reshape((d*D, d*D)) + \
        contract('mbna, mn -> mbna', V3d, JW @ Nu @ cd).reshape((d*D, d*D))

    Htot += H3 + dag(H3)

    if p < L-1:

        E0, U0 = eigsh(Htot, k=D, which='SA')

        logging.info('\nTotal energy for {} orbitals = {}'.format(p+1, E0))

        p += 1 # site id for the new orbital
        print('\n--- adding the {}th orbital ---'.format(p+1))
        print('p = ', p)

        # the annihilation are operators \sigma_i Z_{i+1}......Z_l

        H0 = Htot.copy()

        # Iblock = eye(d*D) # block identity
        Iblock = eye(Cu[0].shape[-1])
        Isite = eye(d) # site identity


        Cu = [rotate(op, JW, U) for op in Cu] + [rotate(Iblock,  cu, U)]
        Cd = [rotate(op, JW, U) for op in Cd] + [rotate(Iblock, cd, U)]
        Cdu = [rotate(op, JW, U) for op in Cdu] + [rotate(Iblock, cdu, U)]
        Cdd = [rotate(op, JW, U) for op in Cdd] + [rotate(Iblock, cdd, U)]


        # print('Cu', Cu)
        ### add all interaction between previous sites (0,1,...p-1) and the new site (p)

        nu = 1
        nd = 0

        # two-operator \sum_{i, j < p} v[i,j,p,p] - v[i, p, p, j] * (nu + nd)
        H1 = H0.copy()
        for i in range(p):
            for j in range(p):
                H1 += v[i,j,p,p] * (nu + nd) * (Cdu[i] @ Cu[j] + Cdd[i] @  Cd[j])
                H1 -= v[i, p, p, j] * (nu * Cdu[i] @ Cu[j] + nd * Cdd[i] @  Cd[j])

        E1, U1 = eigh(H1, k=D)
        # print(E1)

        # the adaibatic states at |\downarrow>
        nu = 0
        nd = 1

        H2 = H0.copy()
        for i in range(p):
            for j in range(p):
                H2 += v[i,j,p,p] * (nu + nd) * (Cdu[i] @ Cu[j] + Cdd[i] @  Cd[j])
                H2 -= v[i, p, p, j] * (nu * Cdu[i] @ Cu[j] + nd * Cdd[i] @  Cd[j])

        E2, U2 = eigh(H2, k=D)
        # print(r'adiabatic states corresponding to |\uparrow> = \n', E2)

        # the adaibatic states at |\uparrow \downarrow>
        nu = 1
        nd = 1

        H3 = H0.copy()
        for i in range(p):
            for j in range(p):
                H3 += v[i,j,p,p] * (nu + nd) * (Cdu[i] @ Cu[j] + Cdd[i] @  Cd[j])
                H3 -= v[i, p, p, j] * (nu * Cdu[i] @ Cu[j] + nd * Cdd[i] @  Cd[j])

        E3, U3 = eigh(H3, k=D)
        # print(E3)

        #########################
        # build the total H for the superblock of l_0 + 1 + 1 sites
        #########################
        # nstates = min(D, d**l)


        E = np.zeros((d, D))
        U = np.zeros((D * d, D, d))

        h = single_site_hamiltonian(p)
        logging.info('site', p, 'H = ', np.diag(h))

        E[0, :] = E0 + h[0, 0]
        E[1, :] = E1 + h[1, 1]
        E[2, :] = E2 + h[2, 2]
        E[3, :] = E3 + h[3, 3]

        U[:, :, 0] = U0
        U[:, :, 1] = U1
        U[:, :, 2] = U2
        U[:, :, 3] = U3

        Es.append(E)
        Cs.append(U.copy().reshape(D, d, D, d)) # indices a_n, j_n, a_{n+1}, j_{n+1}

        # add residual interactions including a_p, a_p^dag a_p a_p

        Htot = np.diag(E.reshape((D * d)))

        # c_p V1
        v1u = 0
        v1d = 0

        for i in range(p):
            v1u += h1e[i, p] * Cdu[i]
            v1d += h1e[i, p] * Cdd[i]


        for i in range(p):
            for j in range(p):
                for k in range(p):
                    v1u += eri[k,p,j,i] * Cdu[k] @ (Cdu[j] @ Cu[i] + Cdd[j] @ Cd[i])
                    v1d += eri[k,p,j,i] * Cdd[k] @ (Cdu[j] @ Cu[i] + Cdd[j] @ Cd[i])

        # jw_string = tensor([JW, ] * n0)

        # V1u =  contract('ibm, ij, jan -> mbna', U.conj(), v1u.toarray() , U)
        # V1d =  contract('ibm, ij, jan -> mbna', U.conj(), v1d.toarray(), U)

        V1 = rotate(v1u, JW @ cu, U) + rotate(v1d, JW @ cd, U)


        Htot += V1 + dag(V1)

        v2a = 0
        for i in range(p):
            for j in range(p):
                v2a += -eri[i, p, p, j] * Cdd[i] @ Cu[j]

        v2b = 0
        for i in range(p):
            for j in range(p):
                v2b += 0.5 * eri[p,i,p,j] * (Cd[i] @ Cu[j] - Cu[i] @ Cd[j])


        # V2 = contract('ibm, ij, jan -> mbna', U.conj(), v2a.toarray(), U)
        # H2a = contract('mbna, mn -> mbna', V2, cdu @ cd).reshape((d*D, d*D))
        V2a = rotate(v2a, cdu @ cd, U)
        # V2b = contract('ibm, ij, jan -> mbna', U.conj(), v2b.toarray(), U)
        # H2b = contract('mbna, mn -> mbna', V2b, cdu @ cdd).reshape((d*D, d*D))
        V2b = rotate(v2b, cdu @ cdd, U)

        V2 = V2a + V2b

        Htot += V2 + dag(V2)


        ## V3 (V3 can be combined with V1)
        v3u = 0
        v3d = 0
        for i in range(p):
            v3u += eri[i, p, p, p] * Cdu[i]
            v3d += eri[i, p, p, p] * Cdd[i]

        # V3u =  contract('ibm, ij, jan -> mbna', U.conj(), v3u.toarray(), U)
        # V3d =  contract('ibm, ij, jan -> mbna', U.conj(), v3d.toarray(), U)

        # H3 = contract('mbna, mn -> mbna', V3u, JW @ Nd @ cu).reshape((d*D, d*D)) + \
        #     contract('mbna, mn -> mbna', V3d, JW @ Nu @ cd).reshape((d*D, d*D))

        V3 = rotate(v3u, JW @ Nd @ cu, U) + rotate(v3d, JW @ Nu @ cd, U)
        Htot += V3 + dag(V3)

    ###############################


    ### Final diagonalization


    # nroots = 20
    E, X = eigsh(Htot, k=nstates, which='SA')

    print('NARG energy = ', E + mol.energy_nuc())

    return Es, Cs


class LETTA:
    def __init__(self, cores):
        self.cores = cores

    def expect(self):
        pass

    def corr(self):
        pass

    def time_corr(self):
        pass

    def compress(self):
        # ??? how can we compress a LETTA
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

    natom = 5
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


    # model = SpinHalfFermionChain(h1e, eri, [mol.nelec//2, mol.nelec//2])
    # model = SpinHalfFermionChain(h1e, eri)

    narg = NARG(h1e, eri, D=20)

    # block = model.initialize()
    narg.run()

    # print(block.X.shape)

    # C0 = np.concatenate(model.X, axis=0)

    # print(C0.shape)

    # model.truncate(10)


    # print(model.site.energy)
    # print(model.site.energy + mol.energy_nuc())