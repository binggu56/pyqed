"""Five-minute PyQED quickstart used by the website, documentation, and CI."""

from pyqed.qchem import Molecule


mol = Molecule(
    atom="H 0 0 0; H 0 0 0.74",
    unit="angstrom",
    basis="sto-3g",
)
mol.build(eri="auto")

mf = mol.RHF().run()
if not mf.converged:
    raise RuntimeError("The quickstart RHF calculation did not converge.")

print(f"RHF energy: {mf.e_tot:.12f} Eh")
