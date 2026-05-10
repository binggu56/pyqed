# from pyqed import Molecule
# from pyqed.qchem.dmrg import DMRGSCF, DMRG
# from pyqed.qchem import CASCI

# from timeit import time

# mol = Molecule(atom='Li 0 0 0; F 0 0 1.4', unit='b', basis='631g')
# mol.build(driver='builtin', eri='dense')

# mf = mol.RHF().run()

# print('E(HF) = ', mf.e_tot)

# dmrg = DMRGSCF(mf, ncas=4, nelecas=4, D=20, site='spatial', symmetry='su2',
#             spin=0, verbose=1).run(nstates=2)

from pyqed.qchem import Molecule
from pyqed.qchem import RHF, CASSCF
from pyqed.qchem.dmrg import DMRGSCF, DMRG

mol = Molecule(
    atom="Li 0 0 0; F 0 0 1.4",
    unit="bohr",
    basis="sto-3g",
)
mol.build(driver="builtin", eri='dense') # auxbasis='cc-pvdz-jkfit')

mf = RHF(mol).run()


mc = DMRG(
    mf,
    ncas=4,
    nelecas=4,
    D=10,
    symmetry="su2",
    init_guess="hf",
    verbose=1, site='spatial',low_rank_mpo=True).run(nstates=2, nsweeps=8)

print("backend:", mc.backend)
print("SA-SU2-DMRGSCF energies:", mc.e_tot)
print("optimized MO coeff shape:", mc.converged)


# mc = CASSCF(mf, ncas=6, nelecas=6).fix_spin(ss=0, shift=0.2).run(nstates=2)
# print(mc.e_tot)
# [-104.09794657 -103.85656769]

# mc = CASCI(mf, 8, 8).run()

# dmrg = DMRG(mf, ncas=8, nelecas=8, D=40, site='spin', verbose=1, init_guess='cid')


# dmrg = DMRGSCF(mf, ncas=8, nelecas=8, D=40, verbose=1)

# mc.fix_spin(ss=0, shift=0.2)
# dmrg.run()    # initial_guess='cid'

# energy logs for you to use
# print(mc.e_tot[1]) #fitst excited state
# print(mc.e_history) #whole energy log in array