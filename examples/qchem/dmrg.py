from pyqed.qchem.dmrg.dmrg import QCDMRG
import numpy as np
from pyqed import Molecule
np.set_printoptions(precision=10, suppress=True, threshold=10000, linewidth=300)


atoms = ['N', 'C', 'C', 'N', 'C', 'C', 'H', 'H', 'H', 'H']

coords = np.array([
    [    0.0000000000,     0.0000000000,     1.5648929680],
    [   -0.0000000000,     1.0505512966,     0.7130576060],
    [    0.0000000000,     1.0505512966,    -0.7130576060],
    [    0.0000000000,     0.0000000000,    -1.5648929680],
    [    0.0000000000,    -1.0505512966,    -0.7130576060],
    [    0.0000000000,    -1.0505512966,     0.7130576060],
    [   -0.0000000000,     2.0619681303,     1.1514435560],
    [    0.0000000000,     2.0619681303,    -1.1514435560],
    [   -0.0000000000,    -2.0619681303,    -1.1514435560],
    [    0.0000000000,    -2.0619681303,     1.1514435560],
])

ncas = 24
nelecas = (5, 5)        
n_states = 3            

BOHR = 1.8897261246257702
coords_bohr = coords * BOHR
atom_list = [[a, *r] for a, r in zip(atoms, coords_bohr)]

mol = Molecule(atom=atom_list, unit='b', basis='631g')
mol.build(driver='pyscf')

mf = mol.RHF().run()


dmrg = QCDMRG(mf, ncas=6, nelecas=6, D=100) 
dmrg.build().run(symmetry_list=['charge','sz'], initial_guess='cid')
