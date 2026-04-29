from pyqed import Molecule
from pyqed.qchem.dmrg import DMRGSCF, DMRG

from timeit import time
mol = Molecule(atom='Li 0 0 0; F 0 0 1.4', unit='b', basis='631g')
mol.build(driver='pyscf')

mf = mol.RHF().run()

print('E(HF) = ', mf.e_tot)

dmrg = DMRG(mf, ncas=8, nelecas=8, D=40, site='spatial', verbose=2)

# dmrg = DMRG(mf, ncas=8, nelecas=8, D=40, site='spin', verbose=1, init_guess='cid')


# dmrg = DMRGSCF(mf, ncas=8, nelecas=8, D=40, verbose=1)

# mc.fix_spin(ss=0, shift=0.2)
dmrg.run(
    nstates=1,
    symmetry_list=['charge', 'sz'],
    # initial_guess='cid',
)

# energy logs for you to use
print(dmrg.e_tot) #ground state energy
# print(mc.e_tot[1]) #fitst excited state
# print(mc.e_history) #whole energy log in array