#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 19:56:20 2025

@author: Bing Gu

@email: gubing@westlake.edu.cn

"""

# import psi4
import scipy
import numpy as np
from pyqed import au2ev, dag
from opt_einsum import contract
from scipy.sparse.linalg import eigsh


class CI:
    def __init__(self, mf, frozen=None, max_cycle=50):

#        assert(isinstance(mf, (scf.rhf.RHF, RHF)))

        self.mf = mf
        self.mol = mf.mol
        self.mo_energy = mf.mo_energy
        self.mo_coeff = mf.mo_coeff

        # self.nstates = nstates

        self.spin = mf.mol.spin

        self.nao = mf.mol.nao
        self.nmo = self.nao

        self.nocc = mf.mol.nelec//2
        self.nvir = self.nmo - self.nocc

        self.max_cycle = max_cycle

        self.nso = self.nmo * 2


        # self.mo_energy = np.zeros(self.nso)
        # self.mo_energy[0::2] = self.mo_energy[1::2] = mf.mo_energy


        self.binary = None
        self.H = None



class CIS(CI):

    def run(self, nroots=1):
        e_tot, ci = kernel(self.mf, nroots)

        self.e_tot = e_tot
        self.ci = [ci[:, n] for n in range(nroots)]

        return self

    def vec_to_amplitudes(self, ci):

        return ci.reshape(self.mf.nocc, self.mf.nvir)


    def make_rdm1(self, state_id=0, ao_repr=True):
        """

        spin-traced 1e RDM

        .. math::

            D_{qp} =  < p^dagger q > = \sqrt{2} c_{pq} n_p (1 - n_q) (wrong)

        Parameters
        ----------
        state_id : TYPE, optional
            DESCRIPTION. The default is 0.
            Note 0 means the first excited state.

        Returns
        -------
        None.

        """
        assert state_id < len(self.ci)

        # mo_coeff = self.mf.mo_coeff
        c = self.vec_to_amplitudes(self.ci[state_id])

        D = np.zeros((self.nmo, self.nmo))
        nocc = self.nocc
        nvir = self.nvir

        oo = np.eye(nocc) - c @ c.T
        ov = np.zeros((nocc, nvir))
        vo = np.zeros((nvir, nocc))
        vv = c.T @ c
        D = np.block([[oo, ov], [vo, vv]])

        if not ao_repr:
            return 2*D
        else:
            return  self.mo_coeff @ (2*D) @ dag(self.mo_coeff)

    def make_natural_orbitals(self, state_id, ao_repr=True):
        """
        return natural orbitals in MO/AO repr

        .. math::

            S(D_α + D_β)SC = SCn

        where S is the overlap matrix, Dα and Dβ are the reduced first
        order density matrices for the α and β spins, respectively, the
        columns of C contain the coefficients of the natural orbitals,
        and the diagonal matrix n holds the occupation numbers.

        Refs
        ----
        J. Chem. Phys. 142, 244104 (2015)


        Parameters
        ----------
        state_id : TYPE
            DESCRIPTION.
        ao_repr : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        noons : TYPE
            DESCRIPTION.
        natorbs : TYPE
            DESCRIPTION.

        """
        D = self.make_rdm1(state_id)

        if not ao_repr: # in MO repr

            # Diagonalize the DM in AO
            # A = reduce(numpy.dot, (S, Dm, S))
            w, v = np.linalg.eigh(D)


            # Flip NOONs (and NOs) since they're in increasing order
            noons = np.flip(w) # nat orb occ number
            natorbs = np.flip(v, axis=1)

        else:

            # Diagonalize the DM in AO
            S = self.mol.get_overlap()
            A =  S @ D @ S
            w, v = np.linalg.eigh(A, b=S)

            # Flip NOONs (and NOs) since they're in increasing order
            noons = np.flip(w)
            natorbs = np.flip(v, axis=1)


        return noons, natorbs

    def state_average_rdm1(self, nstates=None, weights=None):
        """
        state averaged (SA)-CIS first order reduced density matrix


        Refs
        ----
        J. Chem. Phys. 142, 024102 (2015)


        Parameters
        ----------
        nstates : TYPE, optional
            DESCRIPTION. The default is None.
        weights : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        D : TYPE
            DESCRIPTION.

        """

        if nstates is None:
            nstates = len(self.ci) + 1

        if weights is None:
            weights = np.ones(nstates)/nstates

        # ground state RDM
        D = self.mf.make_rdm1() * weights[0]

        for n in range(nstates-1): # loop over excited states
            D += self.make_rdm1(n, ao_repr=True) * weights[n+1]

        return D


# set molecule
# mol = psi4.geometry("""
# o
# h 1 1.0
# h 1 1.0 2 104.5
# symmetry c1
# """)

# # set options
# psi4.set_options({'basis': 'sto-3g',
#                   'scf_type': 'pk',
#                   'e_convergence': 1e-8,
#                   'd_convergence': 1e-8})

# # compute the Hartree-Fock energy and wave function
# scf_e, wfn = psi4.energy('SCF', return_wfn=True)

def make_natural_orbitals(D, S):
    """
    return natural orbitals in MO/AO repr

    .. math::

        S(D_α + D_β)SC = SCn

    where S is the overlap matrix, Dα and Dβ are the reduced first
    order density matrices for the α and β spins, respectively, the
    columns of C contain the coefficients of the natural orbitals,
    and the diagonal matrix n holds the occupation numbers.

    Refs
    ----
    J. Chem. Phys. 142, 244104 (2015)


    Parameters
    ----------
    D : TYPE
        spin-traced rdm.
    S : TYPE, optional
        AO overlap

    Returns
    -------
    noons : TYPE
        DESCRIPTION.
    natorbs : TYPE
        DESCRIPTION.

    """

    A =  S @ D @ S
    w, v = scipy.linalg.eigh(A, b=S)

    # Flip NOONs (and NOs) since they're in increasing order
    noons = np.flip(w)
    natorbs = np.flip(v, axis=1)

    return noons, natorbs

# Grab data from wavfunction

def kernel(mf, nroots=1):

    spin = mf.mol.spin

    # number of doubly occupied orbitals
    # nocc   = wfn.nalpha()
    nocc = mf.nocc

    # total number of orbitals
    # nmo     = wfn.nmo()
    nmo = mf.nmo

    # number of virtual orbitals
    nvir   = nmo - nocc

    # orbital energies
    eps     = mf.mo_energy # np.asarray(wfn.epsilon_a())

    # occupied orbitals:
    # Co = wfn.Ca_subset("AO", "OCC")
    Co = mf.mo_coeff[:, :nocc]

    # virtual orbitals:
    # Cv = wfn.Ca_subset("AO", "VIR")
    Cv = mf.mo_coeff[:, nocc:]

    # use Psi4's MintsHelper to generate ERIs
    # mints = psi4.core.MintsHelper(wfn.basisset())

    eri = mf.mol.eri

    # build the (ov|ov) integrals:
    # ovov = np.asarray(eri(Co, Cv, Co, Cv))
    ovov = contract('pqrs, pi, qa, rj, sb -> iajb', eri, Co, Cv, Co, Cv)


    # build the (oo|vv) integrals:
    oovv = contract('pqrs, pi, qj, ra, sb -> ijab', eri, Co, Co, Cv, Cv)

    # strip out occupied orbital energies, eps_o spans 0..ndocc-1
    eps_o = eps[:nocc]

    # strip out virtual orbital energies, eps_v spans 0..nvirt-1
    eps_v = eps[nocc:]

    # CIS Hamiltonian
    H = np.zeros((nocc*nvir, nocc*nvir))

    if spin == 0:
        # build singlet hamiltonian
        for i in range(0,nocc):
            for a in range(0,nvir):
                ia = i * nvir + a
                for j in range(0,nocc):
                    for b in range(0,nvir):
                        jb = j * nvir + b
                        H[ia][jb] = 2.0 * ovov[i][a][j][b] - oovv[i][j][a][b]
                H[ia][ia] += eps_v[a] - eps_o[i]

        # diagonalize Hamiltonian
        eig, ci = eigsh(H, k=nroots, which='SA')

        print("")
        print("    ==> CIS singlet excitation energies (eV) <==")
        print("")
        for ia in range(0,nroots):
                print("    %5i %10.6f" % (ia,eig[ia]*au2ev))
        print("")

    elif spin == 1:



        # build triplet hamiltonian
        for i in range(0,nocc):
            for a in range(0,nvir):
                ia = i * nvir + a
                for j in range(0,nocc):
                    for b in range(0, nvir):
                        jb = j * nvir + b
                        H[ia][jb] = - oovv[i][j][a][b]
                H[ia][ia] += eps_v[a] - eps_o[i]

        # diagonalize Hamiltonian
        eig, ci = eigsh(H, k=nroots, which='SA')

        print("")
        print("    ==> CIS triplet excitation energies (eV) <==")
        print("")
        for ia in range(nroots):
            print("    %5i %10.4f" % (ia,eig[ia]*au2ev))
        print("")

    return eig, ci


if __name__=='__main__':

    from pyqed.qchem import Molecule


#     coords = np.array([
#     [0.00000000, 0.00000000, 0.66796400],
#     [0.92288300, 0.00000000, 1.24294900],
#     [-0.92288300, 0.00000000, 1.24294900],
#     [0.00000000, 0.00000000, -0.66796400],
#     [0.54030916, 0.92288300, -0.86462045],
#     [0.54030916, -0.92288300, -0.86462045],
# ])
    input_file = 'geometry_theta_000.npz'
    data = np.load(input_file)
    coords = data['geometry']

    mol = Molecule(atom = [
    ("C", coords[0]),
    ("H", coords[1]),
    ("H", coords[2]),
    ("C", coords[3]),
    ("H", coords[4]),
    ("H", coords[5])
    ], basis = '6-31g', unit = 'a')

    # mol = Molecule(atom = [
    #     ['H' , (0. , 0. , 0)],
    #     ['Li' , (0. , 0. , 2.2)], ])

    # set molecule
    # atom = """
    # o
    # h 1 1.0
    # h 1 1.0 2 104.5
    # """

    # mol = gto.M(atom=atom)

    # mol = Molecule(mol._atom)

    # mol = mol.topyscf()
    # mf = mol.RHF()
    # hcore = mf.get_ovlp()

    mol.build()

    mf = mol.RHF().run()


    ci = CIS(mf)
    ci.run(nroots=5)

    D = ci.state_average_rdm1()

    S = mol.overlap

    noons, natorbs = make_natural_orbitals(D, S)

    print(noons)
    ncore = sum(1 for x in noons if x > 1.98)
    ncas = sum(1 for x in noons if 1.98 > x > 0.02)

    print(ncore, ncas)
    nelecas = mol.nelec - ncore * 2
    print(nelecas)

    from pyqed.qchem.ci.casci import CASCI

    print('CIS-NO-CASCI')
    nstates = 3
    mc = CASCI(mf, ncas, nelecas)
    mc.run(nstates, mo_coeff = natorbs[:,:ncas+ncore])


    print('CMO-CASCI')
    mc.run(nstates)

    ###pyscf benchmark

    # print('\n*********PYSCF***********')
    # from pyscf import fci, mcscf, scf, gto

    # # mol2 = gto.M(atom = [
    # #     ['H' , (0. , 0. , 0)],
    # #     ['Li' , (0. , 0. , 2.2)]], unit='b', basis='sto3g', symmetry=0)

    # mol2 = mol.topyscf()

    # hf = scf.RHF(mol2).run()

    # # mc =  hf.CASSCF(ncas, nelecas)

    # mc = mcscf.CASSCF(hf, ncas, nelecas)


    # # mc.fcisolver = fci.direct_spin0.FCI(mol)
    # # mc.fcisolver.nroots = 4
    # mc.state_average_(np.ones(nstates)/nstates)
    # mc.kernel()