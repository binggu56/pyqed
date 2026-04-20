#!/usr/bin/env python3
"""Example: project CASCI BO-Hamiltonian derivatives onto a normal mode.

This example gets normal modes from an RHF Cartesian Hessian, then projects the
analytic CASCI Born-Oppenheimer Hamiltonian derivatives onto one selected mode.
"""

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyqed.qchem import CASCI, Molecule, bo_hamiltonian_derivatives
from pyqed.qchem.dft.hessian import analyze_cartesian_hessian


def parse_args():
    parser = argparse.ArgumentParser(
        description="Project CASCI BO-Hamiltonian derivatives onto a selected normal mode."
    )
    parser.add_argument(
        "--mode-id",
        type=int,
        default=0,
        help="Vibrational mode index after translation/rotation removal.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    mol.build(driver="builtin", options={"eri_representation": "factors"})

    mf = mol.RHF().run()
    mc = CASCI(mf, ncas=2, nelecas=2).run(nstates=2)

    # Use the PySCF RHF Hessian to obtain a normal mode at the same geometry.
    pyscf_mol = mol.topyscf()
    pyscf_mol.build()
    pyscf_mf = pyscf_mol.RHF().run()
    cart_hess_4d = pyscf_mf.Hessian().kernel()

    natom = mol.natom
    cart_hess = np.asarray(cart_hess_4d).transpose(0, 2, 1, 3).reshape(3 * natom, 3 * natom)
    vib = analyze_cartesian_hessian(
        cart_hess,
        mol.atom_coords(),
        mol.atom_mass_list(),
        remove_translation_rotation=True,
    )

    frequencies_cm1 = vib["freq_cm1"]
    normal_modes = vib["modes"]  # shape (nmodes, natom, 3)

    mode_id = args.mode_id
    if mode_id < 0 or mode_id >= len(frequencies_cm1):
        raise ValueError(
            f"mode_id={mode_id} is out of range for {len(frequencies_cm1)} vibrational mode(s)."
        )
    mode_vectors = normal_modes[[mode_id]]

    terms = bo_hamiltonian_derivatives(
        mc,
        state_ids=[0, 1],
        mode_vectors=mode_vectors,
    )

    print("RHF energy:", mf.e_tot)
    print("CASCI energies:", mc.e_tot)
    print()
    print("Vibrational frequencies (cm^-1):")
    print(frequencies_cm1)
    print()
    print(f"Selected mode {mode_id} (Cartesian components):")
    print(mode_vectors[0])
    print()
    print("Projected first-order derivatives F_projected[k, beta, alpha]:")
    print(terms.F_projected.real)
    print()
    print("Projected second-order derivatives G_projected[k, l, beta, alpha]:")
    print(terms.G_projected.real)


if __name__ == "__main__":
    main()
