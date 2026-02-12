#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 10:47:12 2025

@author: Bing Gu (gubing@westlake.edu.cn)
"""

from pyqed.mps.mps import Site
# from pyqed import

from itertools import product
import numpy as np

# class Block(ConservedSite):
#     def __init__(self, *arg):
#         self.dims =

def str_to_int(s):

    return [int(i) for i in s.split(',')]


class ConservedSite(Site):
    def __init__(self, qn=None, degeneracy=None, state_index=None, d=None, \
                 qmin=None, qmax=None, operators=None):



        if qn is None:
            self.qn = [0, 1, 2] # quantum numbers of number operator
            self.degeneracy = [1, 2, 1] # degeneracy corresponding to the quantum numbers
            self.state_index = [[0], [1, 2], [3]]


        else:
            self.qn = qn
            self.degeneracy = degeneracy
            self.state_index = state_index

        if d is not None:
            self.d = d
            assert(d == sum(degeneracy))
        else:
            self.d = sum(self.degeneracy) # total size

        # super().__init__(self.d)

        self.dims = [self.d] # for composite sites

        self.qmin = qmin
        self.qmax = qmax

        self.energy = None
        
        ###
        self.H = None
        self.e_tot = None


    # def __add__(self, other):
    #     qn = []
    #     degeneracy = []
    #     state_index = []

    #     for n in range(len(self.qn)):
    #         for m in range(len(other.qn)):

    #             c = self.qn[n] + other.qn[m]

    #             # if qmax is not None and c > qmax: continue
    #             # if qmin is not None and c < qmin: continue

    #             idx = composite_index(self.state_index[n], other.state_index[m])

    #             d = self.degeneracy[n] * other.degeneracy[m]


    #             if c not in qn: # if the charge does not exist yet
    #                 qn.append(c)
    #                 degeneracy += [d]
    #                 state_index.append(idx)

    #             else:
    #                 i = qn.index(c)
    #                 degeneracy[i] += d
    #                 state_index[i] += idx

    #     block = ConservedSite(qn, degeneracy, state_index, d=self.d * other.d)
    #     block.dims = self.dims + other.dims

    #     return block

    def __add__(self, other):
        qn = []
        degeneracy = []
        state_index = []

        for n in range(len(self.qn)):
            for m in range(len(other.qn)):

                c = self.qn[n] + other.qn[m]

                if self.qmax is not None and c > self.qmax:
                    continue
                if self.qmin is not None and c < self.qmin:
                    continue

                idx = composite_index(self.state_index[n], other.state_index[m])

                d = self.degeneracy[n] * other.degeneracy[m]


                if c not in qn: # if the charge does not exist yet
                    qn.append(c)
                    degeneracy += [d]
                    state_index.append(idx)

                else:
                    i = qn.index(c)
                    degeneracy[i] += d
                    state_index[i] += idx

        block = ConservedSite(qn, degeneracy, state_index, qmin=self.qmin, qmax = self.qmax)
        block.dims = self.dims + other.dims

        return block

    def remove(self, qn):
        """
        truncate the quantum numbers to the range [qmin, qmax]

        Parameters
        ----------
        qmin : TYPE
            DESCRIPTION.
        qmax : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # if qmin is None:
        #     qmin = 0

        # for i, q in enumerate(self.qn):
        #     if q > qmax or q < qmin:

        i = self.qn.index(qn)

        self.qn.pop(i)
        self.degeneracy.pop(i)
        self.state_index.pop(i)

        if self.energy is not None:
            self.energy.pop(i)

        return self

    def truncate(self, qmin, qmax):
        """
        truncate the quantum numbers to the range [qmin, qmax]

        Parameters
        ----------
        qmin : TYPE
            DESCRIPTION.
        qmax : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        retain = list(filter(lambda x:   qmax > x > qmin, self.qn))

        idx = [self.qn.index(i) for i in retain]

        self.qn = [self.qn[i] for i in idx]
        self.degeneracy = [self.degeneracy[i] for i in idx]
        self.state_index = [self.state_index[i] for i in idx]

        if self.energy is not None:
            self.energy = [self.energy[i] for i in idx]

        return self

    def truncate_by_energy(self, cutoff):

        # truncate the states by an energy cutoff
        assert(self.energy is not None)

        degeneracy = []
        energy = []
        qn = []
        state_index = []
        for i, _e in enumerate(self.energy):

            _e_truncated = _e[_e < cutoff]

            _d = len(_e_truncated)

            if _d == len(_e):
                raise Warning('No truncation for electron number {} block. Suggest increasing the  corresponding D.'.format(self.qn[i]))
                #TODO: Increase D and redo the computation for i-th block

            if _d > 0:

                n = sum(degeneracy)
                state_index.append(list(range(n, n + _d)))

                energy += [_e]
                degeneracy += [_d]
                qn += [self.qn[i]]

        block = ConservedSite(qn=qn, degeneracy=degeneracy, state_index=state_index)
        block.energy = energy
        return block

    def ravel_index(self, qn):

        i = self.qn.index(qn)
        idx = self.state_index[i]

        idx = [str_to_int(item) for item in idx]

        # np.ravel_multi_index(arr, (7,6))

        return np.ravel_multi_index(np.array(idx).T, dims = self.dims)




    def block(self, qn, H):
        """
        find the subblock corresponding to a given quantum number
        (e.g. the block with 4 electrons)

        Parameters
        ----------
        qn : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # i = self.qn.index(qn)
        idx = self.ravel_index(qn)

        return H[np.ix_(idx, idx)]


    def __str__(self):
        print('Quantum number', self.qn)
        print('Degeneracy', self.degeneracy)
        print('Total number of states', self.d)
        print('state indices', self.state_index)










class NonabelianSite(ConservedSite):
    def __init__(self):
        self.j = [1/2]
        self.degeneracy = [1]
        self.state_index = [[0, 1]] # corresponding to m = +1, -1
        self.multiplicity = [int(2 * j + 1) for j in self.j]
        self.m = [np.linspace(-j, j, int(2 * j + 1)) for j in self.j]

        print(self.m)

    def __add__(self, other):

        # we need Clebsh-Gordon coefficients for addition of angular momentum
        j1 = self.j
        j2 = other.j

        j = range(abs(j1 - j2), j1 + j2)
        m = np.linspace(-j, j, int(2 * j + 1))

        # transformation matrix






def composite_index(lista, listb):
    s = [str(a) + ',' + str(b) for a, b in product(lista,listb)]
    # s = []
    # for a in l:
    #     s.append([int(i) for i in a.split(',')])
    return s

if __name__=='__main__':
    # lista = ['0']
    # listb = ['1', '2']
    # print(composite_state_index(lista, listb))
    import numpy as np

    site = NonabelianSite()

    sa = ConservedSite(qn=[0, 1], degeneracy=[2, 2], state_index=[[0, 2], [1, 3]])

    sb = ConservedSite(qn=[0, 1], degeneracy=[2, 2], state_index=[[0, 1], [2, 3]])



    for i in range(2):
        sa += sa
        sa.truncate(qmin = 0, qmax = 2)
        sb += sb

    print(sa.qn, sb.qn)

    # idx = s.state_index[5]
    # idxa = sa.ravel_index(2)


    # idxb = sb.ravel_index(2)

    # idx = np.intersect1d(idxa, idxb)




    # idx = s.ravel_index(1)


    # from pyqed.dmrg.dmrg import SpinHalfFermionChain
    from pyqed.mps import SpinHalfFermionChain
    from pyqed.qchem.mol import atomic_chain
    from pyqed.phys import eigh





    natom = 4
    z = np.linspace(-3, 3, natom)
    mol = atomic_chain(natom, z)
    mol.basis = 'sto6g'

    mol.build()

    mf = mol.RHF()
    mf.run()

    h1e = mf.get_hcore_mo()
    v = mf.get_eri_mo()


    # initiate the block with l0 Spin-Orbitals
    nstart = mol.nao
    model = SpinHalfFermionChain(h1e[:nstart, :nstart], v[:nstart, :nstart, :nstart, :nstart],
                                 nelec=mol.nelec)
    # H = model.jordan_wigner()

    model.run(10)
    # print('Exact = ', model.e_tot)

    # h = model.H[np.ix_(idx, idx)]

    # E, U = eigh(h, k=10)
    # print(E)

    ###
    # H = model.H

    # print(H.toarray())

    # print(H[np.ix_(idx, idx)])
    # D = 3
    # e = []
    # u = []
    # for ne in s.qn:

    #     h = s.block(ne, H)

    #     _e, _u = eigh(h, k=D, which='SA')

    #     e.append(_e.copy())
    #     u.append(_u.copy())

    #     print('# electrons = {}, e = {}'.format(ne, _e))

    # print(len([]))

    # truncate the states by an energy cutoff
    cutoff = -3.9

    degeneracy = []
    energy = []
    qn = []
    state_index = []
    for i, _e in enumerate(e):

        _e_truncated = _e[_e < cutoff]

        _d = len(_e_truncated)

        if _d == len(_e):
            raise Warning('No truncation for electron number {} block. Suggest increasing the  corresponding D.'.format(s.qn[i]))
            #TODO: Increase D and redo the computation for i-th block

        if _d > 0:

            n = sum(degeneracy)
            state_index.append(list(range(n, n + _d)))

            energy += [_e]
            degeneracy += [_d]
            qn += [s.qn[i]]

    a = ConservedSite()

    block = ConservedSite(qn=qn, degeneracy=degeneracy, state_index=state_index)
    block += a

    block.info()



    # print(qn, energy, degenaracy)
    # print(state_index)

    # def truncate_by_energy(e, u):

    #     cutoff = -3.9


    #     return TruncatedSite(qn, degeneracy, states, e)

    def truncate_by_quantum_number():
        pass



    # print(idx[0])

    # int_list = list(map(int, idx[0].split()))
    # print(int_list)

    # pos = [int(i) for i in idx[0]]
    # print(pos)

    # print(H[pos])


    # print( H[int(i) for i in idx[3]])