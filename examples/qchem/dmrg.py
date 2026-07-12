#!/usr/bin/env python3
"""Run a compact native DMRG calculation for an H4 chain.

This is the fixed-orbital counterpart to ``dmrgscf.py``.  The small bond
dimension and active space keep the example suitable for a local smoke run.
"""

from pyqed.qchem import Molecule
from pyqed.qchem.dmrg import DMRG
from pyqed.qchem.hf import RHF


mol = Molecule(
    atom="H 0 0 0; H 0 0 1.6; H 0 0 3.2; H 0 0 4.8",
    unit="bohr",
    basis="sto-3g",
)
mol.build(driver="gbasis")
mf = RHF(mol).run()

solver = DMRG(
    mf,
    ncas=4,
    nelecas=4,
    D=8,
    verbose=1,
)
solver.run(nstates=1, nsweeps=4, symmetry_list=["charge", "sz"])

print("DMRG total energy / Eh:", solver.e_tot)
