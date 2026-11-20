"""LiF photon sine DVR with ten CI roots on field-dressed SA4-CASSCF orbitals."""

from functools import partial
from math import sqrt

from pyscf import gto

from pyqed import num_workers
from pyqed.dvr import DVR
from pyqed.polariton.cavity import Cavity, Polariton
from pyqed.qchem import Molecule
from pyqed.qchem.mcscf.casscf import CASSCF
from pyqed.units import eV


if __name__ == "__main__":
    # Fix the Li–F bond length at 1.55 Å and build density-fitted (RI) integrals.
    basis = {atom: gto.basis.load("aug-cc-pVQZ", atom) for atom in ("Li", "F")}
    mol = Molecule(atom="Li 0 0 0; F 0 0 1.55", unit="angstrom", basis=basis)
    mol.build(eri="ri", auxbasis="def2-universal-jkfit", workers=num_workers())

    # Start from dressed RHF at each photon coordinate. The orbitals are optimized
    # independently, without the pilot's orbital rotation constraints.
    solver = partial(
        CASSCF,
        ncas=13,                  # 13 active spatial orbitals
        nelecas=6,                # 6 active electrons
        multiplicity=1,           # Singlet sector
        ci_method="direct_spin0_symm",
        use_cholesky=True,        # Use the RI factors built above
        weights=(0.25,) * 4,      # Four equally weighted roots optimize the orbitals
        nroots=10,                # Ten final CI roots on those fixed orbitals
    )

    omega = 6.487590402594753 * eV
    G = 2 * eV                    # Field strength in Hartree per atomic unit of dipole
    system = Polariton(mol, Cavity(omega))

    # Place 33 sine-DVR points inside the box −4.25 < √ω Q < 4.25.
    qmax = 4.25 / sqrt(omega)
    dvr = DVR([(-qmax, qmax)], [33], mass=1)

    # Convert G to the length-gauge coupling. LDR combines the local potentials
    # with the photon kinetic energy using electronic overlaps between grid points.
    ldr = system.LDR(dvr, solver=solver, coupling=-G / sqrt(omega))

    # Extract the lowest six joint states from the 33 × 10 basis.
    energies, _ = ldr.eigenstates(roots=6)
    print("Lowest six polaritonic energies (Ha)", energies)
