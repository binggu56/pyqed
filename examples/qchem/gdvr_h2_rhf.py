#!/usr/bin/env python3
"""H2 example for GDVR RHF followed by slice-local Newton sweep refinement."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyqed.qchem.gdvr.rhf import AtomicChain


def main():
    bond_length = 1.4  # bohr
    mol = AtomicChain(
        elements=["H", "H"],
        coords=[
            [0.0, 0.0, -bond_length / 2.0],
            [0.0, 0.0, bond_length / 2.0],
        ],
    )

    mol.build(
        Lz=6.0,
        Nz=127,
        M=1,
        max_offset=None,
        auto_cut=False,
        cut_eps=1e-8,
        verbose=False,
        dvr_method="sine",
    )

    mf = mol.RHF().run(
        newton=False,
        conv=1e-8,
        max_iter=100,
        verbose=False,
    )
    e_scf = mf.e_tot
    eps0_scf = mf.mo_energy[0]
    iter_scf = mf.info["iter"]

    mf.newton(
        tol=1e-8,
        sweeps=3,
        ridge=0.5,
        trust_step=1.0,
        trust_radius=2.0,
        scf_conv=1e-8,
        scf_max_iter=100,
        verbose=False,
    )

    print(f"H2 bond length = {bond_length:.3f} bohr")
    print(f"GDVR RHF energy before Newton = {e_scf:.12f} Ha")
    print(f"Lowest orbital energy before Newton = {eps0_scf:.12f} Ha")
    print(f"SCF iterations = {iter_scf}")
    print(f"GDVR RHF energy after Newton = {mf.e_tot:.12f} Ha")
    print(f"Lowest orbital energy after Newton = {mf.mo_energy[0]:.12f} Ha")
    print(f"Newton cycles = {mf.info['newton_cycles']}")
    print(f"Grid size = {mol.shapes['Nz']}  dz = {mol.dz:.6f} bohr")
    print(f"Density matrix shape = {mf.make_rdm1().shape}")


if __name__ == "__main__":
    main()
