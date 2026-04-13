'''main code for pcm solvent'''

import numpy as np
from pyqed import Molecule
from pyqed.qchem import solvent, mcscf
from pyqed.qchem.mcscf.casci import CASCI
# from pyqed.qchem.mcscf.direct_ci import CASCI
from pyqed.qchem.mcscf.casscf import CASSCF


# from pyqed_solvent.pyqed import Molecule
# from pyqed_solvent.pyqed.qchem import solvent, mcscf
# from pyqed_solvent.pyqed.qchem.mcscf.casci import CASCI
# from pyqed_solvent.pyqed.qchem.mcscf.casscf import CASSCF

if __name__ == '__main__':

    # mol = Molecule(atom='Li 0 0 0; H 0 0 1.4', unit='b', basis='sto3g')
    # mol.build()
    # mf = mol.RHF()

    # # RHF with pcm
    # mf = solvent.PCM(mf)
    # # mf.with_solvent.eps = 32.613   # methanol
    # # mf.with_solvent.eps = 24.852   # ethanol
    # # mf.with_solvent.eps = 78.3553   # water
    # mf.with_solvent.eps = 2.3653   # 1,2,4-TriMethylBenzene

    # # These dielectric constants are obtained from https://gaussian.com/scrf/.
    # # More dataset can be found in Minnesota Solvent Descriptor Database 
    # # (https://comp.chem.umn.edu/solvation)

    # mf.run()

    # nstates = 3

    # # C_pcm = mf.mo_coeff.copy()
    # # v_hf  = mf.with_solvent.v.copy()
    # # e_hf  = mf.with_solvent.e  



    # # CASCI with pcm
    # mc = CASCI(mf, ncas=4, nelecas=4)
    # mc = solvent.PCM(mc)

    # # mc.with_solvent.eps = 32.613   # methanol
    # # mc.with_solvent.eps = 24.852   # ethanol
    # # mc.with_solvent.eps = 78.3553   # water
    # mc.with_solvent.eps = 2.3653   # 1,2,4-TriMethylBenzene

    # # mc.with_solvent.v = v_hf
    # # mc.with_solvent.e = e_hf   
    # mc.run(nstates=nstates)



    # # # CASSCF with pcm
    # # mc2 = CASSCF(mf, ncas=2, nelecas=2, max_cycles=50)
    # # mc2.state_average(weights = np.ones(nstates)/nstates)
    # # mc2.fix_spin(ss=0, shift=0.2)
    # # mc2 = solvent.PCM(mc2)

    # # # mc.with_solvent.eps = 32.613   # methanol
    # # # mc.with_solvent.eps = 24.852   # ethanol
    # # # mc.with_solvent.eps = 78.3553   # water
    # # mc2.with_solvent.eps = 2.3653   # 1,2,4-TriMethylBenzene
    # # mc2.with_solvent.v = v_hf
    # # mc2.with_solvent.e = e_hf   
    # # mc2.run(nstates=nstates)
    # # # print('casscf solll ener', mc2.with_solvent.e)
    # # # print('e_tot', mc2.e_tot)


    # print('---------- compare with pyscf ----------')

    # import pyscf
    # from pyscf import gto, scf, solvent, mcscf


    # mol = gto.Mole()
    # mol.atom = 'Li 0 0 0; H 0 0 1.4' 
    # mol.basis = 'sto-3g' 
    # mol.unit = 'Bohr'
    # # mol.verbose = 5
    # mol.build()

    # # RHF
    # mf = scf.RHF(mol)
    # mf = solvent.PCM(mf)
    # # mf.with_solvent.eps = 32.613   # methanol
    # # mf.with_solvent.eps = 24.852   # ethanol
    # # mf.with_solvent.eps = 78.3553   # water
    # mf.with_solvent.eps = 2.3653   # 1,2,4-TriMethylBenzene
    # mf.kernel()
    # # dm2 = mf.make_rdm1()
    # # print('pyscf hf dm', dm2)
    # # print('compare hf dm', dm1.all() == dm2.all())


    # # casci
    # mc = mcscf.CASCI(mf, 4,4)
    # mc.fcisolver.nstates = 3
    # mc = solvent.PCM(mc)
    # mc.with_solvent.frozen = False
    # mc.kernel()
    # print('solvent energy', mc.with_solvent.e)



    # # # casscf
    # # mc = mcscf.CASSCF(mf, 2,2)
    # # mc.fcisolver.nstates = 2
    # # mc = mc.state_average([0.5, 0.5])
    # # mc = solvent.PCM(mc)
    # # mc.kernel()
    # # print('solvent energy', mc.with_solvent.e)

    # # print('*'*50)


    """calculation in gas phase"""

    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='a', basis='sto3g')
    mol.build(driver='gbasis')
    print('coords', mol.atom_coords())
    mf = mol.RHF()
    mf.run()
    print(mf.e_nuc)
    print('hcore', mf.get_hcore())
    print('dm', mf.dm)

    # nstates = 2

    # # # CASCI
    # # mc = CASCI(mf, ncas=2, nelecas=2)
    # # mc.fix_spin(ss=0, shift=0.2)
    # # mc.run(nstates=nstates)




    print('---------- compare with pyscf ----------')

    import pyscf
    from pyscf import gto, scf, solvent, mcscf


    mol = gto.Mole()
    mol.atom = 'H 0 0 0; H 0 0 1.4' 
    mol.basis = 'sto3g' 
    mol.unit = 'a'
    # mol.verbose = 5
    mol.build()
    print('coords', mol.atom_coords())

    # RHF
    mf = scf.RHF(mol)
    mf.kernel()
    print(mf.energy_nuc())
    print('hcore', mf.get_hcore())
    print('dm', mf.make_rdm1())
 

    # nstates = 2
    # # # casci
    # # mc = mcscf.CASCI(mf, 6,6)
    # # mc.fcisolver.nroots = nstates
    # # mc.fix_spin_(ss=0.0)
    # # mc.kernel()

    # # casscf
    # mc = mcscf.CASSCF(mf, 2,2)
    # mc.fcisolver.nstates = nstates
    # mc = mc.state_average(np.ones(nstates)/nstates)
    # mc.fix_spin_(ss=0.0)
    # mc.kernel()