# From PySCF
#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Overlap of two CISD wave functions (they can be obtained from different
geometries).
'''

from functools import reduce
import numpy as np


from gbasis.integrals.overlap_asymm import overlap_integral_asymmetric



# def wf_overlap(Xm, Xn, Cm, Cn, S):
#     """
#     CIS wavefunction overlap matrix

#     Parameters
#     ----------
#     Xm : TYPE
#         DESCRIPTION.
#     Xn : TYPE
#         DESCRIPTION.
#     Cm : TYPE
#         DESCRIPTION.
#     Cn : TYPE
#         DESCRIPTION.
#     S : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     ovlp_00 : TYPE
#         DESCRIPTION.

#     """
#     # restricted case has same orbitals for alpha and beta electrons
#     has_m = True if isinstance(Xm, np.ndarray) else False
#     has_n = True if isinstance(Xn, np.ndarray) else False
#     has_y = True if (isinstance(Ym, np.ndarray) and isinstance(Yn, np.ndarray)) else False

#     nroots, no, nv = Xm.shape

#     smo = np.einsum('mp,mn,nq->pq', Cm, S, Cn)
#     #print_matrix('smo:', smo, 5, 1)
#     smo_oo = np.copy(smo[:no,:no])
#     dot_0 = np.linalg.det(smo_oo)

#     ovlp = np.zeros((nroots, nroots))

#     ovlp[0, 0] = dot_0**2


    # TODO

    # if not (has_m or has_n):

def overlap_ao(mol, mol2):
    """
    compute overlap of two different basis sets


    Parameters
    ----------
    mol : TYPE
        DESCRIPTION.
    mol2 : TYPE
        DESCRIPTION.

    Returns
    -------
    s : TYPE
        DESCRIPTION.

    """
    s = overlap_integral_asymmetric(mol._bas, mol2._bas)
    return s

def overlap_mo(mf, mf2, s=None):
    """
    compute the overlap between MOs of two different configurations

    Parameters
    ----------
    mf : RHF or RKS object
        DESCRIPTION.
    mf2 : TYPE
        DESCRIPTION.
    s : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    2darray
        DESCRIPTION.

    """
    if s is None:
        s = overlap_ao(mf.mol, mf2.mol)

    return mf.mo_coeff.conj().T @ s @ mf2.mo_coeff



def wavefunction_overlap():
    #
    # RCISD wavefunction overlap
    #
    myhf1 = gto.M(atom='H 0 0 0; Li 0 0 1', basis='sto3g', verbose=0, unit='au').apply(scf.RHF).run()
    ci1 = ci.CISD(myhf1).run(nstates=3)
    print('CISD energy of mol1', ci1.e_tot)
    # ci1.nstates = 3
    # ci1.run()


    myhf2 = gto.M(atom='H 0 0 0; Li 0 0 1.1', basis='sto3g', verbose=0, unit='au').apply(scf.RHF).run()
    ci2 = ci.CISD(myhf2).run(nstates=3)
    print('CISD energy of mol2', ci2.e_tot)


    # overlap matrix between MOs at different geometries
    s12 = gto.intor_cross('cint1e_ovlp_sph', myhf1.mol, myhf2.mol)
    s12 = reduce(np.dot, (myhf1.mo_coeff.T, s12, myhf2.mo_coeff))



    nmo = myhf2.mo_energy.size
    nocc = myhf2.mol.nelectron // 2
    print('<CISD-mol1|CISD-mol2> = ', ci.cisd.overlap(ci1.ci[0], ci2.ci[0], nmo, nocc, s12))


def nonadiabatic_coupling(mol, mode_id):
    """
    Compute NACs by finite difference along nth mode

    Parameters
    ----------
    mol : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
# print('<CISD-mol1|CISD-mol2> = ', ci.cisd.overlap(ci1.ci[1], ci1.ci[0], nmo, nocc, s12))


if __name__=='__main__':

    from gbasis.parsers import parse_gbs, make_contractions
    from gbasis.integrals.overlap_asymm import overlap_integral_asymmetric

    from pyqed import Molecule
    from pyscf import gto, scf, ci

    print(wavefunction_overlap())

    mol = gto.M(atom = [
        ['H' , (0. , 0. , 0)],
        ['Li' , (0. , 0. , 1)], ])
    mol.basis = 'sto3g'
    mol.charge = 0
    mol.unit = 'b'
    mol.build()
    hf = mol.HF().run()

    mol2 = gto.M(atom = [
        ['H' , (0. , 0. , 0)],
        ['Li' , (0. , 0. , 1.1)], ])
    mol2.basis = 'sto3g'
    mol2.charge = 0
    mol2.unit = 'b'
    mol2.build()
    hf2 = mol2.HF().run()

    ### PySCF AO overlap
    s = gto.intor_cross('cint1e_ovlp_sph', mol, mol2)
    # s = reduce(np.dot, (hf.mo_coeff.T, s, hf2.mo_coeff))

    print(s)



    mol = Molecule(atom = [
        ['H' , (0. , 0. , 0)],
        ['Li' , (0. , 0. , 1)], ])
    mol.basis = 'sto3g'
    mol.charge = 0
    mol.unit = 'b'
    mol.build()
    hf = mol.RHF().run()

    mol2 = Molecule(atom = [
        ['H' , (0. , 0. , 0)],
        ['Li' , (0. , 0. , 1.1)], ])
    mol2.basis = 'sto3g'
    mol2.charge = 0
    mol2.unit = 'b'
    mol2.build()

    hf2 = mol2.RHF().run()




    # basis_dict = parse_gbs("6-311g.0.gbs")
    # ao_basis_new = make_contractions(basis_dict, ["H", "He"], mol.atom_coords(), "p")


    # # create an 6-311G basis set for the helium hydride ion in spherical coordinates
    # basis_dict = parse_gbs("6-311g.0.gbs")
    # ao_basis_new = make_contractions(basis_dict, ["H", "He"], mol.atom_coords(), "p")

    # print(f"Number of shells in 6-311G basis: {len(ao_basis_new)}")
    # print(f"Number of shells in 6-31G basis: {len(ao_basis)}", end="\n\n")

    # compute overlap of two different basis sets
    int1e_overlap_basis = overlap_integral_asymmetric(mol._bas, mol2._bas)
    # s = reduce(np.dot, (hf.mo_coeff.T, int1e_overlap_basis, hf2.mo_coeff))

    print(f"Shape of overlap matrix: {int1e_overlap_basis.shape}")
    print("Overlap matrix (S) of atomic orbitals between old and new basis:")
    print(int1e_overlap_basis)