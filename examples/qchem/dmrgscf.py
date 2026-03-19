from pyqed import Molecule
from pyqed.qchem.dmrg.dmrgscf import DMRGSCF

mol = Molecule(atom='Li 0 0 0; F 0 0 1.4', unit='b', basis='6311g')
mol.build(driver='pyscf')

mf = mol.RHF().run()

mc = DMRGSCF(mf, ncas=2, nelecas=2, D=60, max_cycles=50)

mc.fix_spin(ss=0, shift=0.2)
mc.run(
    nstates=2,
    symmetry_list=['charge', 'sz'], 
    initial_guess='cid'
)

# energy logs for you to use
print(mc.e_tot[0]) #ground state energy
print(mc.e_tot[1]) #fitst excited state
print([list(h) for h in mc.e_history]) #whole energy log in list
print(mc.e_history) #whole energy log in array