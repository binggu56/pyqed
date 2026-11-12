"""H2 photon LDR with equal-weight two-state SA-CASSCF at each DVR point."""

from functools import partial

from pyqed.dvr import DVR
from pyqed.polariton.cavity import Cavity, Polariton
from pyqed.qchem import Molecule
from pyqed.qchem.mcscf.casscf import CASSCF
from pyqed.units import eV


if __name__ == "__main__":
    mol = Molecule(atom="H 0 0 -0.7; H 0 0 0.7", basis="6-31g", unit="bohr")
    mol.build(eri="s8")
    solver = partial(CASSCF, ncas=2, nelecas=2, verbose=0)
    dvr = DVR([(-6, 6)], [7], mass=1)
    system = Polariton(mol, Cavity(6 * eV))
    ldr = system.LDR(
        dvr, solver=solver, coupling=0.05, nstates=2, weights=(0.5, 0.5)
    )
    energies, _ = ldr.eigenstates(roots=3)
    print("Lowest three polaritonic energies (Ha)", energies)
