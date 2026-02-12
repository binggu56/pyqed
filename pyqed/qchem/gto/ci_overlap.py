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
import numpy
from pyscf import gto, scf, ci

def wavefunction_overlap(geometry1, geometry2, basis='6-31g'):
    """
    Compute the overlap of two CISD wavefunctions

    Parameters
    ----------
    geometry1 : str
        The geometry of the first molecule
    geometry2 : str
        The geometry of the second molecule
    """
    myhf1 = gto.M(atom=geometry1, basis=basis, verbose=0, unit='au').apply(scf.RHF).run()
    ci1 = ci.CISD(myhf1).run()
    print('CISD energy of mol1', ci1.e_tot) 
    ci1.nstates = 1
    ci1.run()
    
    
    myhf2 = gto.M(atom=geometry2, basis=basis, verbose=0, unit='au').apply(scf.RHF).run()
    ci2 = ci.CISD(myhf2).run()
    print('CISD energy of mol2', ci2.e_tot)
    
    
    # overlap matrix between MOs at different geometries
    s12 = gto.intor_cross('cint1e_ovlp_sph', myhf1.mol, myhf2.mol)
    s12 = reduce(numpy.dot, (myhf1.mo_coeff.T, s12, myhf2.mo_coeff))
    
    
    
    nmo = myhf2.mo_energy.size
    nocc = myhf2.mol.nelectron // 2
    print('<CISD-mol1|CISD-mol2> = ', ci.cisd.overlap(ci1.ci, ci2.ci, nmo, nocc, s12))


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

if __name__ == '__main__':

    geometry1 = 'Na 0 0 0; F 0 0 10'
    geometry2 = 'Na 0 0 0; F 0 0 10.02'
    wavefunction_overlap(geometry1, geometry2)