"""Minimal quantum-chemistry DMRG calculation for H2."""

from pyqed.qchem import Molecule
from pyqed.qchem.dmrg.dmrg import DMRG
from pyqed.qchem.hf import RHF


mol = Molecule(
    atom="H 0 0 0; H 0 0 1.4",
    unit="bohr",
    basis="sto-3g",
)
mol.build(driver="builtin")

mf = RHF(mol).run()

dmrg = DMRG(
    mf,
    ncas=2,
    nelecas=2,
    D=8,
    symmetry="su2",
    init_guess="hf",
    verbose=1,
)
dmrg.run(nsweeps=4)

print(f"E(DMRG) = {dmrg.e_tot:.12f} Ha")
