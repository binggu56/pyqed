#!/usr/bin/env python3
"""H4 example for GDVR RHF with Lz=4 and Nz=63, followed by Newton refinement."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyqed.qchem.gdvr.rhf import AtomicChain


def main():
    spacing = 1.4  # bohr
    z_coords = [-1.5 * spacing, -0.5 * spacing, 0.5 * spacing, 1.5 * spacing]
    coords = [[0.0, 0.0, z] for z in z_coords]
    mol = AtomicChain(
        elements=["H", "H", "H", "H"],
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

    mf = mol.RHF().run(
        conv=1e-8,
        max_iter=100,
        verbose=False,
    )
    e_scf = mf.e_tot
    homo_scf = mf.mo_energy[mol.nelec // 2 - 1]
    iter_scf = mf.info["iter"]

    mf.newton(
        tol=1e-8,
        sweep_iterations=10,
        ridge=0.5,
        trust_step=1.0,
        trust_radius=2.0,
        scf_conv=1e-8,
        scf_max_iter=100,
        verbose=False,
    )

    print(f"H4 spacing = {spacing:.3f} bohr")
    print("coordinates (bohr) =")
    for i, xyz in enumerate(coords):
        print(f"  H{i + 1}: {xyz}")
    print(f"Lz = 4.0 bohr")
    print(f"Nz = 63")
    print("transverse basis = default hydrogen STO-6G s exponents")
    print(f"GDVR RHF total energy before Newton = {e_scf:.12f} Ha")
    print(f"GDVR RHF HOMO energy before Newton = {homo_scf:.12f} Ha")
    print(f"SCF iterations = {iter_scf}")
    print(f"GDVR RHF total energy after Newton = {mf.e_tot:.12f} Ha")
    print(f"GDVR RHF HOMO energy after Newton = {mf.mo_energy[mol.nelec // 2 - 1]:.12f} Ha")
    print(f"Newton cycles = {mf.info['newton_cycles']}")
    print(f"Grid size = {mol.shapes['Nz']}  dz = {mol.dz:.6f} bohr")
    print(f"Density matrix shape = {mf.make_rdm1().shape}")


if __name__ == "__main__":
    main()
