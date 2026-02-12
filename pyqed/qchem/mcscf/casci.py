#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 14:59:07 2024

complete active space configuration interaction

@author: Bing Gu (gubing@westlake.edu.cn)
"""

import logging
from functools import reduce
import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

import sys
from opt_einsum import contract

from pyqed import tensor
from itertools import combinations
import itertools
import warnings

from pyqed.qchem import get_veff
from pyqed.qchem.ci.fci import givenΛgetB, SpinOuterProduct, get_fci_combos, SlaterCondon, CI_H
from pyqed.qchem.jordan_wigner.spinful import jordan_wigner_one_body, annihilate, \
            create, Is #, jordan_wigner_two_body


from pyqed.qchem.hf.rhf import ao2mo

def cistring(norb, nelec, sz=0):
    """
    Return all the possible :math:`n`-length binary
    where :math:`k` of :math:`n` digitals are set to 1.

    Parameters
    ----------
    norb: int
        number of orbs. Binary length :math:`n`.
    nelec: int or list of length 2
        number of electrons

    Returns
    -------
    res: list of int-lists
        A list of list containing the binary digitals.

    """
    n = norb
    if isinstance(nelec, (list, tuple)):
        na, nb = nelec
    else:
        if np.mod(nelec, 2):
            raise ValueError('odd number of electrons results in ambiguity. \
                             Please set alpha and beta electrons seperately.')
        na = nb = nelec//2

    if n == 0:
        return [[0]]

    res = []
    for bits in itertools.combinations(list(range(n)), na):
        sa = [0] * n
        for bit in bits:
            sa[bit] = 1
        res.append(sa)

    if na != nb:

        res_b = []
        for bits in itertools.combinations(list(range(n)), nb):

            sb = [0] * n
            for bit in bits:
                sb[bit] = 1

            res_b.append(sb)


    return list(itertools.product(res,res_b))


def get_combos(mo_occ, space='fci', ncore=None, ncas=None, nvir=None):

    space = space.replace(" ", "").lower()
    nmo = len(mo_occ)

    if space == 'fci': # FCI, CAS

        O_sp = np.asarray(mo_occ, dtype=np.int8)
        # number of electrons for each spin
        N_s = np.einsum("sp -> s", O_sp)



        N = O_sp.shape[1]

        Λ_α = np.asarray( list(combinations( np.arange(0, N, 1, dtype=np.int8) , N_s[0] ) ) )
        Λ_β = np.asarray( list(combinations( np.arange(0, N, 1, dtype=np.int8) , N_s[1] ) ) )
        ΛA, ΛB = SpinOuterProduct(Λ_α, Λ_β)
        Binary = givenΛgetB(ΛA, ΛB, N)

        return Binary

    elif space == 'cisd':
        pass
    elif space == 'cas+s': # MR-CIS

        if nvir is None:
            nvir = nmo - ncore - ncas


        # if isinstance(mf, (scf.rhf.RHF, RHF)):
        # if mo_occ is None:
        #     mo_occ = [mf.mo_occ//2, ] * 2
        # else:
        #     mo_occ = mf.mo_occ



# def ao2mo(mf, mo_coeff=None, spin_flip=False, H1=None, H2=None):
#     """
#     Given a rhf object get Spin-Orbit Matrices

#     SF: bool
#         spin-flip

#     Returns
#     -------
#     H1: list of [h1e_a, h1e_b]
#     H2: list of ERIs [[ERI_aa, ERI_ab], [ERI_ba, ERI_bb]]
#     """
#     # from pyscf import ao2mo

#     # mf = self.mf

#     if mo_coeff is None:
#         mo_coeff = mf.mo_coeff

#     # molecular orbitals
#     Ca, Cb = mo_coeff

#     H, energy_core = h1e_for_cas(mf, mo_coeff=mo_core)

#     self.e_core = energy_core


#     # S = (uhf_pyscf.mol).intor("int1e_ovlp")
#     # eig, v = np.linalg.eigh(S)
#     # A = (v) @ np.diag(eig**(-0.5)) @ np.linalg.inv(v)

#     # H1e in AO
#     # H = mf.get_hcore()
#     # H = dag(Ca) @ H @ Ca

#     # nmo = Ca.shape[1] # n

#     eri = mf.eri  # (pq||rs) 1^* 1 2^* 2

#     ### compute SO ERIs (MO)
#     eri_aa = contract('ip, jq, ijkl, kr, ls -> pqrs', Ca.conj(), Ca, eri, Ca.conj(), Ca)

#     # physicts notation <pq|rs>
#     # eri_aa = contract('ip, jq, ij, ir, js -> pqrs', Ca.conj(), Ca.conj(), eri, Ca, Ca)

#     eri_aa -= eri_aa.swapaxes(1,3)

#     eri_bb = eri_aa.copy()

#     eri_ab = contract('ip, jq, ijkl, kr, ls -> pqrs', Ca.conj(), Ca, eri, Cb.conj(), Cb)
#     eri_ba = contract('ip, jq, ijkl, kr, ls -> pqrs', Cb.conj(), Cb, eri, Ca.conj(), Ca)


#     H2 = np.stack(( np.stack((eri_aa, eri_ab)), np.stack((eri_ba, eri_bb)) ))

#     # H1 = np.asarray([np.einsum("AB, Ap, Bq -> pq", H, Ca, Ca),
#                      # np.einsum("AB, Ap, Bq -> pq", H, Cb, Cb)])
#     H1 = [H, H]

#     if spin_flip:
#         raise NotImplementedError('Spin-flip matrix elements not implemented yet')
#     #     eri_abab = (ao2mo.general( (uhf_pyscf)._eri , (Ca, Cb, Ca, Cb),
#     #     compact=False)).reshape((n,n,n,n), order="C")
#     #     eri_abba = (ao2mo.general( (uhf_pyscf)._eri , (Ca, Cb, Cb, Ca),
#     #     compact=False)).reshape((n,n,n,n), order="C")
#     #     eri_baab = (ao2mo.general( (uhf_pyscf)._eri , (Cb, Ca, Ca, Cb),
#     #     compact=False)).reshape((n,n,n,n), order="C")
#     #     eri_baba = (ao2mo.general( (uhf_pyscf)._eri , (Cb, Ca, Cb, Ca),
#     #     compact=False)).reshape((n,n,n,n), order="C")
#     #     H2_SF = np.stack(( np.stack((eri_abab, eri_abba)), np.stack((eri_baab, eri_baba)) ))
#     #     return H1, H2, H2_SF
#     # else:
#     #     return H1, H2
#     return H1, H2

def h1e_for_cas(self, mo_coeff=None):

    '''CAS space effective one-electron hamiltonian

    Args:
        casci : a RHF object

    Returns:
        A tuple, the first is the effective one-electron hamiltonian defined in CAS space,
        the second is the electronic energy from core.
    '''
    mf, ncas, ncore = self.mf, self.ncas, self.ncore
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    # if ncas is None: ncas = .ncas
    # if ncore is None: ncore = casci.ncore
    mo_core = mo_coeff[:,:ncore]
    mo_cas = mo_coeff[:,ncore:ncore+ncas]

    # hcore = mf.get_hcore()
    hcore = self.get_hcore()
    energy_core = mf.energy_nuc()
    if mo_core.size == 0:
        corevhf = 0
    else:
        core_dm = np.dot(mo_core, mo_core.conj().T) * 2
        corevhf = get_veff(mf.mol, core_dm)
        energy_core += np.einsum('ij,ji', core_dm, hcore).real
        energy_core += np.einsum('ij,ji', core_dm, corevhf).real * .5

    h1eff = reduce(np.dot, (mo_cas.conj().T, hcore+corevhf, mo_cas))
    return h1eff, energy_core


def add_electric_field(mol, electric_field=np.array([0,0,0])):
    """add external electric field interaction term
    math::
    V_{int} = - \hat{\mu} \cdot E
    where mu is electric dipole, E is electric field

    Parameters
    ----------
    mol : _type_
        _description_
    electric_field : electric_field=np.array([Ex, Ey, Ez])
        _description_, by default np.array([0,0,0])

    Returns
    -------
    _type_
        _description_
    """    
    hcore_with_field =  np.einsum('ijk,i -> jk', mol.ao_dip, electric_field)

    return hcore_with_field

def add_vector_potential(mol, vector_potential=np.array([0,0,0])):

    A = vector_potential
    print('A^2', np.dot(A, A))
    hcore_with_vector_potential = -np.einsum('ijk,i -> jk', mol.ao_moment, vector_potential).astype(complex) \
        + 0.5 * np.dot(A, A) * mol.overlap


    return hcore_with_vector_potential

class CASCI:
    def __init__(self, mf, ncas, nelecas, ncore=None, spin=None):
        """
        Exact diagonalization (FCI) on the complete active space (CAS) by FCI or
        Jordan-Wigner transformation

        .. math::
            H = h_{ij}c_i^\dagger c_j + \frac{1}{2} v_{pqrs} c_p^\dagger c_q^\dagger c_s c_r\
                -\mu \sum_\sigma c_{i\sigma}^\dag c_{i\sigma}


        From Pyscf: Hartree-Fock orbitals are often poor for systems with significant static correlation.
        In such cases, orbitals from density functional calculations often
        yield better starting points for CAS calculations.

        Parameters
        ----------
        mf : TYPE
            A DFT/HF object.
        nstates : TYPE, optional
            number of excited states. The default is 3.
        ncas : TYPE, optional
            DESCRIPTION. The default is None.
        nelecas : TYPE, optional
            DESCRIPTION. The default is None.

        mu: float
            chemical pontential. The default is None.

        Returns
        -------
        None.

        """
        self.ncas = ncas # number of MOs in active space
        self.nelecas = nelecas

        ncore = mf.nelec//2 - self.nelecas//2 # core orbs
        assert(ncore >= 0)

        self.ncore = ncore

        if ncas > 10:
            warnings.warn('Active space with {} orbitals is probably too big.'.format(ncas))

        self.nstates = None
        # if nelecas is None:
        #     nelecas = mf.mol.nelec

        # if nelecas <= 2:
        #     print('Electrons < 2. Use CIS or CISD instead.')


        self.mo_core = None
        self.mo_cas = None

        if spin is None:
            spin = mf.mol.spin
        self.spin = spin
        self.ss = None
        self.shift = None
        self.spin_purification = False


        self.mf = mf
        # self.chemical_potential = mu

        self.mol = mf.mol

        ###
        self.e_tot = None
        self.e_core = None # core energy
        self.ci = None # CI coefficients
        self.H = None


        self.hcore = self.h1e_cas = None # effective 1e CAS Hamiltonian including the influence of frozen orbitals
        self.Nu = None
        self.Nd = None
        self.binary = None
        self.SC1 = None # SlaterCondon rule 1
        self.eri_so = self.h2e_cas = None # spin-orbital ERI in the active space


        # effective CAS Hamiltonian
        self.h1e = None
        self.h2e = None

        self.electric_dipole = None
        self.magnetic_dipole = None

        hcore = mf.get_hcore()
        self.dtype = hcore.dtype
        print('hcore', hcore.dtype)
        if hcore.dtype == complex:
            self.dtype = np.complex128
        else:
            self.dtype = np.float64

        self.v_solvent = None   # AO PCM potential

        # print('self.add_electric',self.add_electric)
        # print('self.electric_field',self.electric_field)

    def get_hcore(self):
        return self.mf.get_hcore()


    def get_SO_matrix(self, spin_flip=False, H1=None, H2=None):
        """
        Given a rhf object get Spin-Orbit Matrices

        SF: bool
            spin-flip

        Returns
        -------
        H1: list of [h1e_a, h1e_b]
        H2: list of ERIs [[ERI_aa, ERI_ab], [ERI_ba, ERI_bb]]
        """
        # from pyscf import ao2mo

        mf = self.mf

        # molecular orbitals
        Ca, Cb = [self.mo_cas, ] * 2

        H, energy_core = self.h1e_for_cas()

        self.e_core = energy_core


        # S = (uhf_pyscf.mol).intor("int1e_ovlp")
        # eig, v = np.linalg.eigh(S)
        # A = (v) @ np.diag(eig**(-0.5)) @ np.linalg.inv(v)

        # H1e in AO
        # H = mf.get_hcore()
        # H = dag(Ca) @ H @ Ca

        # nmo = Ca.shape[1] # n

        eri = mf.eri  # (pq||rs) = (pq|rs) - (ps|qr) 1^* 1 2^* 2

        ### compute SO antisymmetrized ERIs (MO)
        eri_aa = contract('ip, jq, ijkl, kr, ls -> pqrs', Ca.conj(), Ca, eri, Ca.conj(), Ca)
        # eri_aa -= eri_aa.swapaxes(1,3)

        eri_bb = eri_aa.copy()

        eri_ab = contract('ip, jq, ijkl, kr, ls -> pqrs', Ca.conj(), Ca, eri, Cb.conj(), Cb)
        eri_ba = contract('ip, jq, ijkl, kr, ls -> pqrs', Cb.conj(), Cb, eri, Ca.conj(), Ca)




        # eri_aa = (ao2mo.general( (uhf_pyscf)._eri , (Ca, Ca, Ca, Ca),
        #                         compact=False)).reshape((n,n,n,n), order="C")
        # eri_aa -= eri_aa.swapaxes(1,3)

        # eri_bb = (ao2mo.general( (uhf_pyscf)._eri , (Cb, Cb, Cb, Cb),
        # compact=False)).reshape((n,n,n,n), order="C")
        # eri_bb -= eri_bb.swapaxes(1,3)

        # eri_ab = (ao2mo.general( (uhf_pyscf)._eri , (Ca, Ca, Cb, Cb),
        # compact=False)).reshape((n,n,n,n), order="C")
        # #eri_ba = (1.*eri_ab).swapaxes(0,3).swapaxes(1,2) ## !! caution depends on symmetry

        # eri_ba = (ao2mo.general( (uhf_pyscf)._eri , (Cb, Cb, Ca, Ca),
        # compact=False)).reshape((n,n,n,n), order="C")

        H2 = np.stack(( np.stack((eri_aa, eri_ab)), np.stack((eri_ba, eri_bb)) ))

        # H1 = np.asarray([np.einsum("AB, Ap, Bq -> pq", H, Ca, Ca),
                         # np.einsum("AB, Ap, Bq -> pq", H, Cb, Cb)])
        H1 = [H, H]

        if spin_flip:
            raise NotImplementedError('Spin-flip matrix elements not implemented yet')
        #     eri_abab = (ao2mo.general( (uhf_pyscf)._eri , (Ca, Cb, Ca, Cb),
        #     compact=False)).reshape((n,n,n,n), order="C")
        #     eri_abba = (ao2mo.general( (uhf_pyscf)._eri , (Ca, Cb, Cb, Ca),
        #     compact=False)).reshape((n,n,n,n), order="C")
        #     eri_baab = (ao2mo.general( (uhf_pyscf)._eri , (Cb, Ca, Ca, Cb),
        #     compact=False)).reshape((n,n,n,n), order="C")
        #     eri_baba = (ao2mo.general( (uhf_pyscf)._eri , (Cb, Ca, Cb, Ca),
        #     compact=False)).reshape((n,n,n,n), order="C")
        #     H2_SF = np.stack(( np.stack((eri_abab, eri_abba)), np.stack((eri_baab, eri_baba)) ))
        #     return H1, H2, H2_SF
        # else:
        #     return H1, H2
        return H1, H2



    def natural_orbitals(self, dm, nco=None):
        natural_orb_occ, natural_orb_coeff = np.linalg.eigh(dm)

        return natural_orb_occ, natural_orb_coeff

    def size(self, basis='sd', S=0):

        return size_of_cas(self.ncas, self.nelecas)

    def qubitization(self, orb='mo'):

        if orb == 'mo':

            # transform the Hamiltonian in DVR set to (truncated) MOs
            # nmo = self.ncas
            mf = self.mf

            # single electron part
            Ca = mf.mo_coeff[:, self.ncore:self.ncore + self.ncas]
            # hcore_mo = contract('ia, ij, jb -> ab', Ca.conj(), mf.hcore, Ca)

            h1eff, e_core = self.h1e_for_cas()

            self.e_core = e_core

            eri = self.mf.eri
            eri_mo = contract('ip, iq, ij, jr, js -> pqrs', Ca.conj(), Ca,
                              eri, Ca.conj(), Ca)

            # eri_mo = contract('ip, jq, ij, ir, js', mo.conj(), mo.conj(), eri, mo, mo)

            # self.hcore_mo = hcore_mo

            return self.jordan_wigner(h1eff, eri_mo)


        elif orb == 'natural':
            raise NotImplementedError('Nartural orbitals qubitization not implemented.')


    def fix_nelec(self, shift=0.1):
        """
        fix the electron number by energy penalty.
        This is only needed for JW solver without symmetry.

        Parameters
        ----------
        shift : TYPE, optional
            DESCRIPTION. The default is 0.1.

        Returns
        -------
        None.

        """

        Na = self.Nu
        Nb = self.Nd

        I = tensor(Is(self.ncas))

        self.H += shift * ((Na - self.nelecas/2 * I) @ (Na - self.nelecas/2 * I) + \
            (Nb - self.nelecas/2 * I) @ (Nb - self.nelecas/2 * I))

    def jordan_wigner(self, h1e, v):
        """
        MOs based on Restricted HF calculations

        Returns
        -------
        H : TYPE
            DESCRIPTION.

        """
        # an inefficient implementation without consdiering any syemmetry


        norb = h1e.shape[-1]
        nmo = L = norb # does not necesarrily have to MOs


        Cu = annihilate(norb, spin='up')
        Cd = annihilate(norb, spin='down')
        Cdu = create(norb, spin='up')
        Cdd = create(norb, spin='down')

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

        self.Nu = Na
        self.Nd = Nb


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

    def fix_spin(self, s=None, ss=0, shift=0.2):
        """
        fix the spin by energy penalty

        .. math::

            H = H + \mu (\hat{S}^2 - S(S+1))

        Parameters
        ----------
        s : TYPE, optional
            DESCRIPTION. The default is None.
        ss : TYPE, optional
            DESCRIPTION. The default is None.
        shift : TYPE, optional
            DESCRIPTION. The default is 0.2.

        Returns
        -------
        None.

        """
        if s is None:
            s = (np.sqrt(4*ss+1)-1)/2
            if not np.isclose(2*s, round(2*s)):
                raise Warning("s = {} inconsistant spin value".format(s))
        else:
            if ss is None:
                ss = s * (s+1)
            else:
                # assert ss == s * (s+1)

                raise ValueError('s and ss cannot be specified simultaneously.')

        if ss == 0:
            # first-order spin penalty J. Phys. Chem. A 2022, 126, 12, 2050–2060
            # H' = H + J \hat{S}^2
            # norb = h1e[0].shape[0]

            # ncas = self.ncas

            # h1e = [h + 3./4 * shift * np.eye(ncas) for h in h1e]

            # for p in range(ncas):
            #     for q in range(ncas):
            #         h2e[:, :, p, q, q, p] -= 0.5 * shift
            #         h2e[:, :, p, p, q, q] -= 1./4 * shift

            self.spin_purification = True
            self.ss = 0
            self.shift = shift

            return self


        else:
            # second-order spin penalty
            raise NotImplementedError('Second-order spin panelty not implemented.')

    def h1e_for_cas(self):
        print('self h1e for cas')
        return h1e_for_cas(self, self.mo_coeff)

    def run(self, nstates=1, mo_coeff=None, method='ci', ci0=None):
        """
        solve the full CI in the active space, more efficient than the JW solver

        Parameters
        ----------
        nstates : TYPE, optional
            DESCRIPTION. The default is 3.
        mo : CAS MOs
            Default is canonical MOs.
        method : TYPE, optional
            choose which solver to use.
            'ci' is the standard CI solver.
            'jw' is the exact diagonalizaion by Jordan-Wigner transformation.
            The default is 'ci'.

        TODO: spin

        Returns
        -------
        TYPE
            DESCRIPTION.
        X : TYPE
            DESCRIPTION.

        """
        # print('------------------------------')
        # print("             CASCI              ")
        # print('------------------------------\n')
        self.nstates = nstates

        # if method == 'ci':

        ncore = self.ncore
        ncas = self.ncas

        # define the core and active space orbitals
        if mo_coeff is None:
            self.mo_coeff = self.mf.mo_coeff # use HF MOs
            # self.mo_core = self.mo_coeff[:, :ncore]
            # self.mo_cas = self.mo_coeff[:, ncore:ncore+ncas]

        else:
            self.mo_coeff = mo_coeff

        self.mo_core = self.mo_coeff[:, :ncore]
        self.mo_cas = self.mo_coeff[:, ncore:ncore+ncas]


        if self.binary is None:
            mo_occ = [self.mf.mo_occ[ncore: ncore+ncas]//2, ] * 2
            binary = get_fci_combos(mo_occ = mo_occ)
            self.binary = binary
        else:
            binary = self.binary

        # print('Number of determinants', binary.shape[0])

        # effective hamiltonian in the CAS
        h1e, h2e = self.get_SO_matrix()

        if self.spin_purification:

            logging.info('Purify spin by energy penalty')

            # if self.shift is not None:
            # H1, H2 = self.fix_spin(H1, H2, ss=ss, shift=shift)
            shift = self.shift

            norb = self.ncas
            h1e = [h + 3./4 * shift * np.eye(norb) for h in h1e]

            for p in range(norb):
                for q in range(norb):
                    h2e[:, :, p, q, q, p] -=  0.5 * shift * 2
                    # h2e[1, 1, p, q, q, p] -=  0.5 * shift
                    # h2e[0, 1, p, q, q, p] -=  0.5 * shift
                    # h2e[1, 0, p, q, q, p] -=  0.5 * shift

                    # h2e[0, 0, p, p, q, q] -= 0.25 * shift
                    # h2e[1, 1, p, p, q, q] -= 0.25 * shift

                    # h2e[0, 1, p, p, q, q] -= 0.25 * shift
                    # h2e[1, 0, p, p, q, q] -= 0.25 * shift


                    h2e[:, :, p, p, q, q] -= 0.25 * shift * 2

        h2e[0,0] -= h2e[0,0].swapaxes(1,3)
        h2e[1,1] -= h2e[1,1].swapaxes(1,3)




        self.hcore = h1e
        self.eri_so = h2e


        SC1, SC2 = SlaterCondon(binary)
        self.SC1 = SC1
        self.SC2 = SC2

        I_A, J_A, a_t , a, I_B, J_B, b_t , b, ca, cb = SC1


        H_CI = CI_H(binary, h1e, h2e, SC1, SC2)
        E, X = eigsh(H_CI, k=nstates, which='SA')

        sort_idx = np.argsort(E)
        E = E[sort_idx]
        X = X[:, sort_idx]
        # print('E = ', E)


        # nuclear repulsion energy is included in Ecore
        self.e_tot = E + self.e_core
        self.ci = [X[:, n] for n in range(nstates)]

        for i in range(nstates):
            ss = spin_square(*self.make_rdm12(i))
            print("CASCI Root {}  E = {:.10f}  S^2 = {:.6f}".format(i, self.e_tot[i], ss))

        return self

    def exact_diag(self, nstates=1, mo_coeff=None, ci0=None):
        # elif method == 'jw':

        # exact diagonalization by JW transform

        H = self.qubitization()
        E, X = eigsh(H, k=nstates, which='SA')

    # else:
    #     raise ValueError("There is no {} solver for CASCI. Use 'ci' or 'jw'".format(method))
        return E, X

    def make_rdm1_contract(self, state_id, h1e=None, representation='ao'):
        """
        spin-traced 1e reduced density matrix
        .. math::

            \gamma[p,q] = <q_alpha^\dagger p_alpha> + <q_beta^\dagger p_beta>


        Returns
        -------
        None.

        """

        ci = self.ci[state_id]
        if representation.lower() == 'ao':
            C = self.mf.mo_coeff
            h1e = ao2mo(h1e, C)

        ncore = self.ncore
        ncas = self.ncas

        if ncore > 0:
            c_core = 2 * np.trace(h1e[:ncore,:ncore])
        else:
            c_core = 0

        h1e = h1e[ncore:ncas+ncore, ncore:ncas+ncore]

        c_cas = contract_with_rdm1(ci, self.binary, self.SC1, h1e=h1e)

        return c_core + c_cas

    def make_rdm1(self, state_id, with_core=False, with_vir=False, representation='mo'):
        """
        spin-traced 1e reduced density matrix
        .. math::

            \gamma[p,q] = <q_alpha^\dagger p_alpha> + <q_beta^\dagger p_beta>

        Parameters
        ----------
        representation: str
            indicate which representation RDM is defined. Default 'mo'.

        Returns
        -------
        None.

        """

        ci = self.ci[state_id]
        # print('rdm1 ci dtype', ci.dtype)
        # if representation.lower() == 'ao':
        #     C = self.mf.mo_coeff
        #     h1e = ao2mo(h1e, C)

        ncore = self.ncore
        ncas = self.ncas
        nmo = self.mf.nmo

        # if ncore > 0:
        #     c_core = 2 * np.trace(h1e[:ncore,:ncore])
        # else:
        #     c_core = 0
        if with_core and not with_vir:

            norb = ncas + ncore
            D = np.zeros((norb, norb), dtype=self.dtype)

            if ncore > 0: 
                for i in range(ncore): 
                    D[i, i] = 2
            D[ncore:ncore+ncas, ncore:ncore+ncas] = make_rdm1(ci, self.binary, self.SC1)

            return D

        if with_core and with_vir:

            D = np.zeros((nmo, nmo), dtype=self.dtype)
            if ncore > 0: 
                for i in range(ncore): 
                    D[i, i] = 2
            D[ncore:ncore+ncas, ncore:ncore+ncas] = make_rdm1(ci, self.binary, self.SC1)

            return D
        else:
            return make_rdm1(ci, self.binary, self.SC1)


    def make_rdm1s(self, state_id):
        """
        spin-polarized 1e reduced density matrix
        .. math::

            \gamma_s[p,q] = <q_s^\dagger p_s>


        Returns
        -------
        None.

        """

        raise NotImplementedError()

    def make_rdm2(self, state_id=0, with_core=False, with_vir=False):
        """
        2-e reduced density matrix

        The definition follows the PySCF convention.
        .. math::

            \Gamma[p,q,r,s] = \sum_{sigma,tau} <p_sigma^\dagger r_tau^\dagger s_tau q_sigma>

        with this convention, the energy is computed as

        E = einsum('pqrs,pqrs', eri, rdm2)/2

        Returns
        -------
        None.

        """
        ci = self.ci[state_id]


        if with_core: # we probably never need this!

            ncore = self.ncore
            ncas = self.ncas
            # nmo = self.mf.nmo
            nmo = ncore + ncas

            D = np.zeros((nmo, nmo, nmo, nmo), dtype=self.dtype)

            assert ncore > 0

            # cccc block
            I = np.eye(ncore)
            D[:ncore, :ncore, :ncore, :ncore] = 4 * contract('ij, kl -> ijkl', I, I) - 2 * contract('ps, rq -> pqrs', I, I)

            # ccaa block
            dm1 = self.make_rdm1(state_id)

            for i in range(ncore):
                D[i, i, ncore:ncore+ncas, ncore:ncore+ncas] = 2*dm1
                D[ncore:ncore+ncas, ncore:ncore+ncas, i, i] = 2*dm1
                D[i, ncore:ncore+ncas, i, ncore:ncore+ncas] = -dm1
                D[ncore:ncore+ncas, i, ncore:ncore+ncas, i] = -dm1

            D[ncore:ncore+ncas, ncore:ncore+ncas, ncore:ncore+ncas, ncore:ncore+ncas]=\
                make_rdm2(ci, self.binary, self.SC1, self.SC2)

            return D

        else: #active space DM

            return make_rdm2(ci, self.binary, self.SC1, self.SC2)


    def contract_with_rdm2(self, h2e, state_id=0):

        if h2e.ndim == 4: # spin-free operator
            h2e = np.einsum('IJ, pqrs -> IJpqrs', np.ones((2,2)), h2e)

        return contract_with_rdm2(self.ci[state_id], h2e, self.binary, self.SC1, self.SC2)



    def make_rdm12(self, state_id, with_core=False):
        dm1 = self.make_rdm1(state_id, with_core=with_core)
        dm2 = self.make_rdm2(state_id, with_core=with_core)

        return dm1, dm2

    def spin_square(self, state_id=0):

        return spin_square(*self.make_rdm12(state_id))



    def dump(self, fname):
        import pickle

        with open(fname, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
        return

    def overlap(self, other):
        return overlap(self, other)

    def contract_with_tdm1(self, bra_id, ket_id=0, h1e=None, representation='mo'):
        """
        spin-traced 1e transition density matrix

        .. math::

            \gamma_{pq}^{\beta \alpha} = <\Psi_\beta | \hat{E}_{qp} | \Psi_\alpha >

        E_{qp} = q_alpha^\dagger p_alpha + q_beta^\dagger p_beta

        Parameters
        ----------
        bra_id : TYPE
            DESCRIPTION.
        ket_id : TYPE, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        None.

        """

        if bra_id == ket_id:

            print("CI ket and bra are the same. Computing 1e RDM instead.")
            return self.make_rdm1(ket_id, h1e)

        else:

            if representation.lower() == 'ao':
                C = self.mf.mo_coeff
                h1e = ao2mo(h1e, C)

            ncore = self.ncore
            ncas = self.ncas

            if ncore > 0:
                c_core = 2 * np.trace(h1e[:ncore,:ncore])
            else:
                c_core = 0

            h1e = h1e[ncore:ncas+ncore, ncore:ncas+ncore]


            c_cas = make_tdm1(self.ci[bra_id], self.ci[ket_id], self.binary, self.SC1, h1e)

        return c_cas + c_core

    def make_tdm1(self, bra_id, ket_id=0):
        """
        TDM

        Parameters
        ----------
        bra_id : TYPE
            DESCRIPTION.
        ket_id : TYPE, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        None.

        """
        cibra = self.ci[bra_id]
        ciket = self.ci[ket_id]

        return make_tdm1(cibra, ciket, self.binary, self.SC1)

    # def make_tdm1_tmp(self, bra_id, ket_id=0, with_core=False, with_vir=False):
    #     """
    #     TDM

    #     Parameters
    #     ----------
    #     bra_id : TYPE
    #         DESCRIPTION.
    #     ket_id : TYPE, optional
    #         DESCRIPTION. The default is 0.

    #     Returns
    #     -------
    #     None.

    #     """
    #     cibra = self.ci[bra_id]
    #     ciket = self.ci[ket_id]
    #     ncore = self.ncore
    #     ncas = self.ncas
    #     nmo = self.mf.nmo

    #     # if ncore > 0:
    #     #     c_core = 2 * np.trace(h1e[:ncore,:ncore])
    #     # else:
    #     #     c_core = 0
    #     if with_core and not with_vir:

    #         norb = ncas + ncore
    #         D = np.zeros((norb, norb), dtype=float)

    #         if ncore > 0: 
    #             # for i in range(ncore): 
    #             #     D[i, i] = 2
    #             if bra_id == ket_id:
    #                 D = self.make_rdm1(bra_id, with_core)
    #             else:
    #                 D[ncore:ncore+ncas, ncore:ncore+ncas] = make_tdm1(cibra, ciket, self.binary, self.SC1)

    #         return D

    #     if with_core and with_vir:

    #         D = np.zeros((nmo, nmo), dtype=float)
    #         if ncore > 0: 
    #             for i in range(ncore): 
    #                 D[i, i] = 2
    #         D[ncore:ncore+ncas, ncore:ncore+ncas] = make_tdm1(cibra, ciket, self.binary, self.SC1)

    #         return D
    #     else:
    #         return make_tdm1(cibra, ciket, self.binary, self.SC1)

    def make_tdm2(self, bra_id, ket_id=0):
        """
        spin-traced 1e transition density matrix in MO

        .. math::

            \gamma_{pq}^{\beta \alpha} = <\Psi_\beta | \hat{E}_{qp} | \Psi_\alpha >

        E_{qp} = q_alpha^\dagger p_alpha + q_beta^\dagger p_beta
        """
        raise NotImplementedError('TDM not implemented')

    def get_electric_dip(self, initial_state, final_state, unit, **kwargs):


        tdm1 = self.make_tdm1(final_state, initial_state)
        # print('tdm',tdm1)
        # tdm1 = self.make_tdm1(final_state, initial_state)

        mo_coeff = self.mo_coeff[:, self.ncore:self.ncas + self.ncore]



        dip = electric_dipole(self.mol, tdm1, mo_coeff, unit)

        self.electric_dipole = dip
        print('initial state : {}; final state {}; transitoion electric dipole : {} {}'.format(initial_state, final_state, dip, unit))



        # D = 2*np.eye(self.ncore)
        # print(D)
        # print('*'*100)
        # orbcas = self.mo_coeff[:, 0:self.ncore]
        # t_dm1_ao = reduce(np.dot, (orbcas, D, orbcas.T))
        # ao_dip = mol2.ao_dip
        # print("$$$", np.einsum('xij,ji->x', ao_dip, t_dm1_ao))

        return dip

    def get_magnetic_dip(self, initial_state, final_state, **kwargs):

        tdm1 = self.make_tdm1(final_state, initial_state)
        mo_coeff = self.mo_coeff[:, self.ncore:self.ncas + self.ncore]
        mag_dip = orbital_magnetic_dipole(self.mol, tdm1, mo_coeff)

        self.magnetic_dipole = mag_dip
        print('initial state : {}; final state {}; transitoion magnetic dipole : {}'.format(initial_state, final_state, mag_dip))
        return mag_dip

def electric_dipole(mol, tdm1, mo_coeff, unit='Debye', **kwargs):

    # from pyscf.gto.moleintor import getints
    """dipole moment calculation

    math:
        x^hat = \sum_{p,q} x_{p,q} a_p^dagger a_q
        y^hat = \sum_{p,q}yx_{p,q} a_p^dagger a_q
        z^hat = \sum_{p,q} z_{p,q} a_p^dagger a_q

        dip_x = -\sum_{p,q} x_{p,q} dm1_pq + \sum_A Q_A X_A
        dip_y = -\sum_{p,q} y_{p,q} dm1_pq + \sum_A Q_A Y_A
        dip_z = -\sum_{p,q} z_{p,q} dm1_pq + \sum_A Q_A Z_A
        
    Parameters
    ----------
    mol : _type_
        _description_
    tdm1 : _type_
        one body transition density matrix 
    ao_dip : _type_
        x_{p,q}
    """    

    # charges = mol.atom_charges()
    # coords = mol.atom_coords()


    # print('tdm', tdm1)
    # print('coeff',mo_coeff)
    
    # nuc_charge_center = np.einsum('z,zx->x', charges, coords) / charges.sum()
    # mol.build

    
    ao_dip = mol.ao_dip
    # ao_dip = getints('int1e_r', atm= , bas= , comp=3, env=nuc_charge_center)
    # print('ao_dip', ao_dip)
    # print('ao_dip shape', np.shape(ao_dip))
    mo_dip = np.einsum('ik, xkl, lj -> xij', mo_coeff.T, ao_dip, mo_coeff)
    # print('mo_dip shape', np.shape(mo_dip))

    el_dip = np.einsum('xji, ij -> x', mo_dip, tdm1).real

    # nucl_dip = np.einsum('i, ix -> x', charges, coords)
    # nucl_dip = 0
    # mol_dip = nucl_dip - el_dip




    if unit == 'Debye':
        el_dip *= 2.541746231
        logging.info('Dipole moment(X, Y, Z, Debye): %8.5f, %8.5f, %8.5f', *el_dip)
        # print('Dipole moment(X, Y, Z, Debye): %8.5f, %8.5f, %8.5f', *mol_dip)
    else:
        logging.info('Dipole moment(X, Y, Z, a.u.): %8.5f, %8.5f, %8.5f', *el_dip)
        # print('Dipole moment(X, Y, Z, a.u.): %8.5f, %8.5f, %8.5f', *mol_dip)
    
    return el_dip

def magnetic_dipole():

    pass

def spin_magnetic_dipole():
    """spin magnetic dipole calculation

    math:

    """    
    pass

def orbital_magnetic_dipole(mol, tdm1, mo_coeff, **kwargs):
    """orbital magnetic dipole calculation

    math:
        mu = - e / 2m * L
        L = r × p
        magnetic_dip = -0.5 * \sum_{i,j} (r × p)_{i,j} dm1_{i,j}
    """    
    ao_mag_dip = mol.ao_magnetic_dip
    mo_mag_dip = np.einsum('ik, xkl, lj -> xij', mo_coeff.T, ao_mag_dip, mo_coeff)
    mol_mag_dip = -0.5 * np.einsum('xji, ij -> x', mo_mag_dip, tdm1).real
    logging.info('magnetic dipole moment(X, Y, Z): %8.5f, %8.5f, %8.5f', *mol_mag_dip)
    
    return mo_mag_dip

# def get_SO_matrix(mo_coeff, eri, spin_flip=False, H1=None, H2=None):
#     """
#     Given a rhf object get Spin-Orbit Matrices

#     SF: bool
#         spin-flip
#     """

#     Ca, Cb = mo_coeff

#     H, energy_core = h1e_for_cas(mf, ncas=ncas, ncore=ncore, \
#                                  mo_coeff=mo_coeff)

#     # self.e_core = energy_core



#     eri = mf.eri  # (pq||rs) 1^* 1 2^* 2

#     ### compute SO ERIs (MO)
#     eri_aa = contract('ip, jq, ijkl, kr, ls -> pqrs', Ca.conj(), Ca, eri, Ca.conj(), Ca)

#     # physicts notation <pq|rs>
#     # eri_aa = contract('ip, jq, ij, ir, js -> pqrs', Ca.conj(), Ca.conj(), eri, Ca, Ca)

#     eri_aa -= eri_aa.swapaxes(1,3)

#     eri_bb = eri_aa.copy()

#     eri_ab = contract('ip, jq, ijkl, kr, ls -> pqrs', Ca.conj(), Ca, eri, Cb.conj(), Cb)
#     eri_ba = contract('ip, jq, ijkl, kr, ls -> pqrs', Cb.conj(), Cb, eri, Ca.conj(), Ca)


#     H2 = np.stack(( np.stack((eri_aa, eri_ab)), np.stack((eri_ba, eri_bb)) ))

#     H1 = [H, H]

#     if spin_flip:
#         raise NotImplementedError('Spin-flip matrix elements not implemented yet')

#     return H1, H2

# def fcisolver(mo_occ):
#     # mo_occ = [self.mf.mo_occ[ncore: ncore+ncas]//2, ] * 2
#     binary = get_fci_combos(mo_occ = mo_occ)
#     # self.binary = binary

#     print('Number of determinants', binary.shape[0])

#     H1, H2 = get_SO_matrix()

#     SC1, SC2 = SlaterCondon(binary)


#     I_A, J_A, a_t , a, I_B, J_B, b_t , b, ca, cb = SC1

#     H_CI = CI_H(binary, H1, H2, SC1, SC2)

#     E, X = eigsh(H_CI, k=nstates, which='SA')

#     return E, X

def size_of_cas(norb, nelec, basis='sd', S=0):
    """
    size of CAS

    Eq. 91, 92 Chem Rev 2012, 112, 108

    Parameters
    ----------
    norb : TYPE
        DESCRIPTION.
    nelec : TYPE
        DESCRIPTION.
    basis : TYPE, optional
        DESCRIPTION. The default is 'sd'.
    S : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    from math import comb
    # if isinstance(norb, int): norb = [norb, ] * 2
    if isinstance(nelec, int): nelec = [nelec, ] * 2


    # norb_a, norb_b = norb
    na, nb = nelec
    if basis == 'sd':
        return comb(norb, na) * comb(norb, nb)
    elif basis == 'csf':
        N = na + nb
        return (2*S+1)/(norb + 1) * comb(norb+1, N//2 - S) * comb(norb+1, N//2+S+1)

def spin_square(dm1, dm2):
    """

    Compute the total spin S^2, require 2e RDM

    Ref:
        J. Chem. Theory Comput. 2021, 17, 5684−5703


    For a single SO,
    .. math::

            S_i^2 = \frac{3}{4} (E_ii - e_{ii,ii})
            S_i \cdot S_j = -frac{1}{2} (e_{ij,ji} + \frac{1}{2} e_{ii, jj}), j \ne i

    where E_{ij}, e_{ijkl} are 1 and 2e RDMs.

    Parameters
    ----------
    dm1 : TYPE
        DESCRIPTION.
    dm2 : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    spin_square = (0.75*np.einsum("ii", dm1)
               - 0.5*np.einsum("ijji", dm2)
               - 0.25*np.einsum("iijj", dm2))

    return spin_square

# def ss_matrix():

#     ss = CI_H(binary, H1, H2, SC1, SC2)

def contract_with_tdm1(cibra, ciket, binary, SC1, h1e):
    """

    1e transition DM contracted with 1e operators

    .. math::

        \langle \Phi_I  O_{pq} p^\dagger q | \Phi_J \rangle = O_{pq} A^{IJ}_{qp}}

    Parameters
    ----------
    ci : TYPE
        DESCRIPTION.
    h1e : TYPE, optional
        One electron operator in MO. The default is None.

    Returns
    -------
    D : TYPE
        DESCRIPTION.

    SC1 (1-body Slater-Condon Rules)
    SC2 (2-body Slater-Condon Rules)

    Return
    ======
    HCI: CI Hamiltonian
    """

    if isinstance(h1e, np.ndarray): # spin-independent 1e operator
        h1e = [h1e, h1e]

    I_A, J_A, a_t , a, I_B, J_B, b_t , b, ca, cb = SC1

    # sum of MO energies
    H = np.einsum("Spp, ISp -> I", h1e, binary, optimize=True)
    H = np.diag(H)

    ## Rule 1
    H[I_A, J_A] -= np.einsum("pq, Kp, Kq -> K", h1e[0], a_t, a, optimize=True)
    H[I_B , J_B ] -= np.einsum("pq, Kp, Kq -> K", h1e[1], b_t, b, optimize=True)


    return np.einsum('I, IJ, J -> ', cibra.conj(), H, ciket)

def contract_with_rdm1(ci, binary, SC1, h1e):
    """

    make 1e RDM contracted with 1e operators without returning RDM

    .. math::
        \Tr{ O D} = O_{pq} D_{qp}} = O_{pq} \hat{E}_{pq}

    Parameters
    ----------
    ci : TYPE
        DESCRIPTION.
    h1e : TYPE, optional
        One electron operator in MO. The default is None.

    Returns
    -------
    D : TYPE
        DESCRIPTION.

    SC1 (1-body Slater-Condon Rules)
    SC2 (2-body Slater-Condon Rules)

    Return
    ======
    HCI: CI Hamiltonian
    """

    if isinstance(h1e, np.ndarray): # spin-independent 1e operator
        h1e = [h1e, h1e]

    I_A, J_A, a_t , a, I_B, J_B, b_t , b, ca, cb = SC1


    # sum of MO energies
    H = np.einsum("Spp, ISp -> I", h1e, binary, optimize=True)
    H = np.diag(H)

    ## Rule 1
    H[I_A, J_A] -= np.einsum("pq, Kp, Kq -> K", h1e[0], a_t, a, optimize=True)
    H[I_B , J_B ] -= np.einsum("pq, Kp, Kq -> K", h1e[1], b_t, b, optimize=True)


    return np.einsum('I, IJ, J -> ', ci.conj(), H, ci)

def make_rdm1(ci, binary, SC1):
    """

    make spin-traced 1e RDM E_{pq}

    .. math::

        \hat{E}_{pq}

    Parameters
    ----------
    ci : TYPE
        DESCRIPTION.
    h1e : TYPE, optional
        One electron operator in MO. The default is None.

    Returns
    -------
    D : TYPE
        DESCRIPTION.

    SC1 (1-body Slater-Condon Rules)
    SC2 (2-body Slater-Condon Rules)

    Return
    ======
    HCI: CI Hamiltonian
    """


    I_A, J_A, a_t , a, I_B, J_B, b_t , b, ca, cb = SC1


    # sum of MO energies
    # H = np.einsum("ISp -> Ip", binary, optimize=True)
    # H = binary[:, 0, :] + binary[:, 1, :]

    # H = np.diag(H)

    nsd, _, nmo = binary.shape
    H = np.zeros((nsd, nsd, nmo, nmo))
    for I in range(nsd):
        for p in range(nmo):
            H[I, I, p, p] = sum(binary[I, :, p])

    ## Rule 1
    H[I_A, J_A] -= np.einsum("Kp, Kq -> Kpq", a_t, a, optimize=True)
    H[I_B, J_B] -= np.einsum("Kp, Kq -> Kpq", b_t, b, optimize=True)


    return np.einsum('I, IJpq, J -> pq', ci.conj(), H, ci).T

def make_tdm1(cibra, ciket, binary, SC1):
    """

    make spin-traced 1e TDM E_{pq}

    .. math::

        \braket{I|\hat{E}_{pq}|J}

    Parameters
    ----------
    ci : TYPE
        DESCRIPTION.
    h1e : TYPE, optional
        One electron operator in MO. The default is None.

    Returns
    -------
    D : TYPE
        DESCRIPTION.

    SC1 (1-body Slater-Condon Rules)
    SC2 (2-body Slater-Condon Rules)

    Return
    ======
    HCI: CI Hamiltonian
    """


    I_A, J_A, a_t , a, I_B, J_B, b_t , b, ca, cb = SC1


    # sum of MO energies
    # H = np.einsum("ISp -> Ip", binary, optimize=True)
    # H = binary[:, 0, :] + binary[:, 1, :]

    # H = np.diag(H)

    nsd, _, nmo = binary.shape
    H = np.zeros((nsd, nsd, nmo, nmo))
    for I in range(nsd):
        for p in range(nmo):
            H[I, I, p, p] = sum(binary[I, :, p])

    ## Rule 1
    H[I_A, J_A] -= np.einsum("Kp, Kq -> Kpq", a_t, a, optimize=True)
    H[I_B, J_B] -= np.einsum("Kp, Kq -> Kpq", b_t, b, optimize=True)


    return np.einsum('I, IJpq, J -> pq', cibra.conj(), H, ciket)





def contract_with_rdm2(ci, H2, Binary, SC1, SC2):
    I_A, J_A, a_t , a, I_B, J_B, b_t , b, ca, cb = SC1
    I_AA, J_AA, aa_t, aa, I_BB, J_BB, bb_t, bb, I_AB, J_AB, ab_t, ab, ba_t, ba = SC2

    # # sum of MO energies I: configuration index, S: spin index, p: MO index
    # H_CI = np.einsum("Spp, ISp -> I", H1, Binary, optimize=True)

    # ERI
    H_CI = np.einsum("STppqq, ISp, ITq -> I", H2, Binary, Binary, optimize=True)/2
    H_CI = np.diag(H_CI)

    ## Rule 1
    # H_CI[I_A , J_A ] -= np.einsum("pq, Kp, Kq -> K", H1[0], a_t, a, optimize=True)
    H_CI[I_A , J_A ] -= np.einsum("pqrr, Kp, Kq, Kr -> K", H2[0,0], a_t, a, ca, optimize=True)
    H_CI[I_A , J_A ] -= np.einsum("pqrr, Kp, Kq, Kr -> K", H2[0,1], a_t, a, Binary[I_A,1],
    optimize=True)

    # H_CI[I_B , J_B ] -= np.einsum("pq, Kp, Kq -> K", H1[1], b_t, b, optimize=True)
    H_CI[I_B , J_B ] -= np.einsum("pqrr, Kp, Kq, Kr -> K", H2[1,1], b_t, b, cb, optimize=True)
    H_CI[I_B , J_B ] -= np.einsum("pqrr, Kp, Kq, Kr -> K", H2[1,0], b_t, b, Binary[I_B,0],
    optimize=True)

    if len(I_AA) > 0:
    ## Rule 2
        H_CI[I_AA, J_AA] = np.einsum("pqrs, Kp, Kq, Kr, Ks -> K", H2[0,0], aa_t[0], aa[0],
        aa_t[1], aa[1], optimize=True)

    if len(I_BB) > 0:
        H_CI[I_BB, J_BB] = np.einsum("pqrs, Kp, Kq, Kr, Ks -> K", H2[1,1], bb_t[0], bb[0],
        bb_t[1], bb[1], optimize=True)

    H_CI[I_AB, J_AB] = np.einsum("pqrs, Kp, Kq, Kr, Ks -> K", H2[0,1], ab_t, ab, ba_t, ba,
        optimize=True)

    return np.einsum('I, IJ, J -> ', ci.conj(), H_CI, ci)

def make_rdm2(ci, Binary, SC1, SC2):
    """
    build the spin-traced 2-particle operator with the 2e RDM

    .. math::

        \Gamma_{pqrs} = \sum_{\sigma, \tau} p^\dagger_\sigma r^\dagger_\tau s_\tau q_\sigma

    TODO: fix it

    Params
    ------
    Binary: binary string (I, s, p)
        I: configuration index, S: spin index, p: MO index

    Refs
    ----
    J. Chem. Theory Comput. 2022, 18, 6690−6699

    """
    I_A, J_A, a_t , a, I_B, J_B, b_t , b, ca, cb = SC1
    I_AA, J_AA, aa_t, aa, I_BB, J_BB, bb_t, bb, I_AB, J_AB, ab_t, ab, ba_t, ba = SC2

    nsd, _, nmo = Binary.shape
    I = np.eye(nmo)

    H_CI = np.zeros((nsd, nsd, nmo, nmo, nmo, nmo)) # slow implementation

    # diagonal elements
    D = np.einsum("I, ISp, ITr, pq, rs -> pqrs", np.abs(ci)**2, Binary, Binary, I, I, optimize=True).astype(ci.dtype)
    D -= np.einsum("I, ISp, ISr, ps, rq -> pqrs", np.abs(ci)**2, Binary, Binary, I, I, optimize=True)

    ## Rule 1
    H_CI[I_A , J_A ] = -2 * np.einsum("Kp, Kq, Kr, rs -> Kpqrs",  a_t, a, ca, I, optimize=True)
    H_CI[I_A , J_A ] -= np.einsum("Kp, Kq, Kr, rs -> Kpqrs", a_t, a, Binary[I_A,1], I, optimize=True)

    H_CI[I_B , J_B ] -= 2 * np.einsum("Kp, Kq, Kr, rs -> Kpqrs", b_t, b, cb, I, optimize=True)
    H_CI[I_B , J_B ] -= np.einsum("Kp, Kq, Kr, rs -> Kpqrs", b_t, b, Binary[I_B,0], I, optimize=True)

    ## Rule 2
    if len(I_AA) > 0:

        H_CI[I_AA, J_AA] = 2 * np.einsum("Kp, Kq, Kr, Ks -> Kpqrs", aa_t[0], aa[0],
        aa_t[1], aa[1], optimize=True)

    if len(I_BB) > 0:
        H_CI[I_BB, J_BB] = 2 * np.einsum("Kp, Kq, Kr, Ks -> Kpqrs", bb_t[0], bb[0],
        bb_t[1], bb[1], optimize=True)

    H_CI[I_AB, J_AB] = 2 * np.einsum("Kp, Kq, Kr, Ks -> Kpqrs", ab_t, ab, ba_t, ba,
        optimize=True)

    D += contract('I, IJpqrs, J -> pqrs', ci.conj(), H_CI, ci)

    return D

def overlap(cibra, ciket, s=None):
    """
    CASCI electronic overlap matrix

    The MO overlap is a block matrix

    for Restricted calculation only! (spin unpolarized.)

    TODO: unrestricted HF.

    S = [S_CC, S_CA]
        [S_AC, S_AA]



    Compute the overlap between Slater determinants first
    and contract with CI coefficients

    Parameters
    ----------
    cibra : TYPE
        DESCRIPTION.
    binary1 : TYPE
        DESCRIPTION.
    ciket : TYPE
        DESCRIPTION.
    binary2 : TYPE
        DESCRIPTION.
    s : TYPE
        AO overlap.

    Returns
    -------
    None.

    """
    # nstates = len(cibra) + 1

    # overlap matrix between MOs at different geometries
    if s is None:

        from gbasis.integrals.overlap_asymm import overlap_integral_asymmetric

        s = overlap_integral_asymmetric(cibra.mol._bas, ciket.mol._bas)
        s = reduce(np.dot, (cibra.mf.mo_coeff.T, s, ciket.mf.mo_coeff))


    nsd_bra = cibra.binary.shape[0]
    nsd_ket = ciket.binary.shape[0]
    S = np.zeros((nsd_bra, nsd_ket)) # overlap between determinants

    ncore_bra = cibra.ncore
    ncore_ket = ciket.ncore

    scc = s[:ncore_bra, :ncore_ket]
    sca = s[:ncore_bra, ncore_ket:]
    sac = s[ncore_bra:, :ncore_ket]
    saa = s[ncore_bra:, ncore_ket:]

    scc_det = np.linalg.det(scc)
    scc_inv = np.linalg.inv(scc)

    for I in range(nsd_bra):
        occidx1_a  = [i for i, char in enumerate(cibra.binary[I, 0]) if char == 1]
        occidx1_b  = [i for i, char in enumerate(cibra.binary[I, 1]) if char == 1]

        for J in range(nsd_ket):
            occidx2_a  =  [i for i, char in enumerate(ciket.binary[J, 0]) if char == 1]
            occidx2_b  =  [i for i, char in enumerate(ciket.binary[J, 1]) if char == 1]

            # print('b', occidx2_a, occidx2_b)
            # print(ciket.binary[J])

    # TODO: the overlap matrix can be efficiently computed for CAS factoring out the core-electron overlap.
            saa_occ_a = saa[np.ix_(occidx1_a, occidx2_a)]
            sca_occ_a = sca[:, occidx2_a]
            sac_occ_a = sac[occidx1_a, :]

            saa_occ_b = saa[np.ix_(occidx1_b, occidx2_b)]
            sca_occ_b = sca[:, occidx2_b]
            sac_occ_b = sac[occidx1_b, :]


            S[I, J] = scc_det**2 * np.linalg.det(saa_occ_a - sac_occ_a @ scc_inv @ sca_occ_a)*\
                np.linalg.det(saa_occ_b - sac_occ_b @ scc_inv @ sca_occ_b)



    # core_bra = list(range(cibra.ncore))
    # core_ket = list(range(ciket.ncore))




    # for I in range(nsd_bra):
    #     occidx1_a  = core_bra + [i + ncore_bra for i, char in enumerate(cibra.binary[I, 0]) if char == 1]
    #     occidx1_b  = core_bra + [i + ncore_bra for i, char in enumerate(cibra.binary[I, 1]) if char == 1]

    #     for J in range(nsd_ket):
    #         occidx2_a  = core_ket + [i + ncore_ket for i, char in enumerate(ciket.binary[J, 0]) if char == 1]
    #         occidx2_b  = core_ket + [i + ncore_ket for i, char in enumerate(ciket.binary[J, 1]) if char == 1]

    #         # print('b', occidx2_a, occidx2_b)
    #         # print(ciket.binary[J])

    # # TODO: the overlap matrix can be efficiently computed for CAS factoring out the core-electron overlap.

    #         S[I, J] = np.linalg.det(s[np.ix_(occidx1_a, occidx2_a)]) * \
    #                   np.linalg.det(s[np.ix_(occidx1_b, occidx2_b)])


    return contract('BI, IJ, AJ -> BA', np.array(cibra.ci).conj(), S, np.array(ciket.ci))

if __name__ == "__main__":
    from pyqed import Molecule
    from pyqed.qchem.ci.cisd import overlap


    # mol = Molecule(atom = [
    # ['Li' , (0. , 0. , 0)],
    # ['F' , (0. , 0. , 1)], ])

    # mol.basis = '631g'
    # mol.charge = 0

    # mol.molecular_frame()
    # print(mol.atom_coords())

    # nstates = 3
    # Rs = np.linspace(1,4,4)
    # E = np.zeros((nstates, len(Rs)))

    # for R in Rs:

    #     atom = [
    #     ['Li' , (0. , 0. , 0)],
    #     ['F' , (0. , 0. , R)]]

    #     mol = Molecule(atom, basis='631g')

    #     mol.build()

    #     mf = mol.RHF()
    #     mf.run()

    #     ncas, nelecas = (4,2)
    #     casci = CASCI(mf, ncas, nelecas)

    #     casci.run(nstates)

    #     casci.e_tot

    #### test overlap
    # mol2 = Molecule(atom = [
    # ['Li' , (0. , 0. , 0)],
    # ['H' , (0. , 0. , 1.4)], ])
    # mol2.basis = 'sto6g'

    mol = Molecule(atom='Li 0 0 0; H 0 0 1.4', unit='b', basis='sto3g')
    # mol.unit = 'b'
    mol.build()
    mol.molecular_frame()
    nstates = 2

    mf = mol.RHF().run()

    ncas, nelecas = (2,2)
    mc = CASCI(mf, ncas, nelecas)
    mc.run(nstates)

    rdm1 = mc.make_rdm1(state_id=0, with_core=True)
    print('rdm1', rdm1.shape)

    C = mc.mo_coeff[:, :mc.ncore+ mc.ncas]
    print('c', C.shape)

    print('e_tot', mc.e_tot[0])

    # C0 = mf.mo_coeff

    hcore0 = mf.get_hcore()
    # print('hcore ===',hcore0.all() == mol2.hcore.all())
    # print('hcore0', hcore0)



    # add electric field or vector potential
    E_vec=np.array([0,0,0.1])
    # A_vec=np.array([0.0,0,0.01])
    hcore_electric_field = add_electric_field(mol, electric_field=E_vec)
    # hcore_vector_potential = add_vector_potential(mol2, vector_potential=A_vec)
    # print('hcore_vector_potential', hcore_vector_potential)

    # mol2.hcore = hcore0 - hcore_electric_field + hcore_vector_potential
    mol.hcore = hcore0 - hcore_electric_field
    mf2 = mol.RHF().run()
    # CA = mf2.mo_coeff



    # s0 = mol2.overlap
    # coords = -mol2.ao_dip # <mu | r | nu>
    # A_dot_r = np.einsum('i,imn->mn', A_vec, coords)
    # # S_exp = s0 - 1j * A_dot_r - 0.5 * (A_dot_r @ np.linalg.inv(s0) @ A_dot_r)
    # # S_exp = s0 - 1j * A_dot_r - 0.5 * (A_dot_r @ np.linalg.inv(s0) @ A_dot_r) + 1j/6 * (A_dot_r @ np.linalg.inv(s0) @ A_dot_r @ np.linalg.inv(s0) @ A_dot_r)

    # s, U = eigh(mol2.overlap)
    # # building transformation matrix S^{-1/2}
    # X = U.dot(np.diagflat(s**(-0.5)).dot(U.conj().T))
    # C0_p = np.linalg.inv(X) @ C0
    # CA_p = np.linalg.inv(X) @ CA
    # overlap = C0.conj().T @ CA
    # phase = s0 - 1j * A_dot_r - 0.5 * (A_dot_r @ s0 @ A_dot_r)
    # print('-'*100)
    # print('hcore+A', mf2.get_hcore())
    # print('fock', mf2.get_fock())
    # h0000 = mf2.get_fock()

    # print('CA',CA)
    # print('exp(-iAr)C0', S_exp @ C0)
    # print('phase', CA[:,0] - S_exp @ C0[:,0])

    # S_prime = C0.conj().T @ S_exp @ CA
    # print('s0',s0)
    # print('S_prime', S_prime)
    # print('overlap', np.einsum('ii->', S_prime)- np.sum(S_prime.diagonal())) 

    # print(f"{'MO Index':<10} | {'|Overlap|':<12} ")
    # print("-" * 30)
    # for i in range(mol2.nelec // 2):
    #     overlap = S_prime[i, i]
    #     print(f"{i:<10} | {np.abs(overlap):<12.6f} ")

    # # CASCI 
    # wavefunction_2 = mf2.mo_coeff
    # ncas, nelecas = (2,2)
    # nstates = 4
    # mc = CASCI(mf2, ncas, nelecas)
    # # mc.run(nstates=nstates, method='ci')

    # print('Fix spin by penalty')

    # # mc = CASCI(mf2, ncas, nelecas)
    # mc.fix_spin(ss=0, shift=0.2)
    # mc.run(nstates=nstates)
    # for state_id in range(nstates):
    #     mc.get_electric_dip(initial_state=0, final_state=state_id, unit='au')









    # print("dip",mol2.ao_dip)
    # print('pyqed casci electric dip', mc.electric_dipole)

    # mc.get_magnetic_dip(initial_state=0, final_state=1)
    # print('pyqed casci magnetic dip', mc.magnetic_dipole)

    # casci.run()
    # S = overlap(casci, casci2)
    # print(S)

    ### pyscf
    # from pyscf import gto, mp, mcscf
    # # mol = gto.M(
    # #     atom = 'O 0 0 0; O 0 0 1.2',
    # #     basis = 'ccpvdz',
    # #     spin = 2)

    # mol = gto.M(atom = [
    # ['H' , (0. , 0. , 0)],
    # ['Li' , (0. , 0. , 1)], ], unit='b')
    # mol.basis = 'sto3g'
    # myhf = mol.RHF().run()
    # # Use MP2 natural orbitals to define the active space for the single-point CAS-CI calculation
    # # mymp = mp.UMP2(myhf).run()

    # # noons, natorbs = mcscf.addons.make_natural_orbitals(mymp)

    # mycas = mcscf.CASCI(myhf, ncas, nelecas)
    # mycas.nroots = 4
    # mycas.run()
