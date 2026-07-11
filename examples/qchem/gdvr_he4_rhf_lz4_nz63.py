#!/usr/bin/env python3
"""Simple He4 GDVR RHF example with the default helium STO-6G transverse basis."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyqed.qchem.gdvr.rhf import AtomicChain


def main():
    spacing = 1.4  # bohr
    coords = [
        [0.0, 0.0, -1.5 * spacing],
        [0.0, 0.0, -0.5 * spacing],
        [0.0, 0.0, 0.5 * spacing],
        [0.0, 0.0, 1.5 * spacing],
    ]

    mol = AtomicChain(
        elements=["He", "He", "He", "He"],
        coords=coords,
    )

    mol.build(
        Lz=4.0,
        Nz=63,
        M=1,
        max_offset=None,
        auto_cut=False,
        cut_eps=1e-8,
        verbose=False,
        dvr_method="sine",
    )

    mf = mol.RHF().run(conv=1e-8, max_iter=200, verbose=False)
    nocc = mol.nelec // 2

    print(f"He4 spacing = {spacing:.3f} bohr")
    print("coordinates (bohr) =")
    for i, xyz in enumerate(coords, start=1):
        print(f"  He{i}: {xyz}")
    print("Lz = 4.0 bohr")
    print("Nz = 63")
    print("transverse basis = default helium STO-6G s exponents")
    print(f"nelec = {mol.nelec}")
    print(f"GDVR RHF total energy before Newton = {mf.e_tot:.12f} Ha")
    print(f"GDVR RHF HOMO energy before Newton = {mf.mo_energy[nocc - 1]:.12f} Ha")
    print(f"SCF iterations = {mf.info['iter']}")

    mf.newton(tol=1e-8, scf_conv=1e-8, scf_max_iter=200, verbose=False)
    print(f"GDVR RHF total energy after Newton = {mf.e_tot:.12f} Ha")
    print(f"GDVR RHF HOMO energy after Newton = {mf.mo_energy[nocc - 1]:.12f} Ha")
    print(f"Newton cycles = {mf.info['newton_cycles']}")
    print(f"Grid size = {mol.shapes['Nz']}  dz = {mol.dz:.6f} bohr")
    print(f"Density matrix shape = {mf.dm.shape}")


if __name__ == "__main__":
    main()
