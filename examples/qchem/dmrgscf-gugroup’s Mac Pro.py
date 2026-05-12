from pyqed import Molecule
from pyqed.qchem.dmrg import DMRGSCF, DMRG
from pyqed.qchem import CASCI

from timeit import time

mol = Molecule(atom='Li 0 0 0; F 0 0 1.4', unit='b', basis='631g')
mol.build(driver='builtin', eri='dense')

mf = mol.RHF().run()

print('E(HF) = ', mf.e_tot)

# dmrg = DMRGSCF(mf, ncas=6, nelecas=6, D=20, site='spatial', symmetry='su2',
#             spin=0, verbose=2).run(nstates=2, state_average_backend="sweep")


dmrg = DMRG(
    mf, ncas=4, nelecas=4, D=10,
    site="spatial", symmetry="su2", spin=0, verbose=1,
).run(nstates=2, nsweeps=8)
    # state_average_backend="sweep",
    # state_average_reference_dense=True,


# print(dmrg.dmrg.history[-1]["bond_objectives"][-1]["effective_local_problem"])
# print(dmrg.dmrg.history[-1]["dense_reference_energy_errors"])

# mc = CASCI(mf, 8, 8).run()

# dmrg = DMRG(mf, ncas=8, nelecas=8, D=40, site='spin', verbose=1, init_guess='cid')


# dmrg = DMRGSCF(mf, ncas=8, nelecas=8, D=40, verbose=1)

# mc.fix_spin(ss=0, shift=0.2)
# dmrg.run()    # initial_guess='cid'

