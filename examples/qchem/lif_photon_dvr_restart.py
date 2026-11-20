"""Restart LiF SA4-CASSCF(6,12) on a photon sine DVR from PyQED checkpoints."""

import argparse
from pathlib import Path

import numpy as np
from pyscf import gto, lib

from pyqed import num_workers
from pyqed.dvr import DVR
from pyqed.ldr.overlap import full
from pyqed.polariton.cavity import Cavity, Polariton
from pyqed.qchem import Molecule
from pyqed.qchem.mcscf.casscf import CASSCF
from pyqed.units import eV


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nodes", type=int, default=25)
    parser.add_argument("--box", type=float, default=3.25, help="Half-width in sqrt(omega) Q")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.output.resolve() == args.checkpoints.resolve():
        parser.error("Choose a separate output directory for the new wavefunctions.")
    lib.num_threads(1)
    omega = 6.487590402594753 * eV
    coupling = -2 * eV / np.sqrt(omega)
    basis = {atom: gto.basis.load("aug-cc-pVQZ", atom) for atom in ("Li", "F")}
    mol = Molecule(atom="Li 0 0 0; F 0 0 1.55", basis=basis, unit="angstrom")
    mol.build(eri="ri", auxbasis="def2-universal-jkfit", workers=num_workers())
    electronic = Polariton(mol, Cavity(omega)).electronic_hamiltonian(coupling=coupling)
    limit = args.box / np.sqrt(omega)
    dvr = DVR([(-limit, limit)], [args.nodes], mass=1)
    with np.load(args.checkpoints / "scan.npz") as previous:
        old_x = previous["x"]
    frames, energies, potentials = [], [], []
    for i, q in enumerate(dvr.x[0]):
        dressed, offset, photon = electronic(q)
        nearest = np.argmin(abs(old_x - np.sqrt(omega) * q))
        with np.load(args.checkpoints / f"state_{nearest:02d}.npz") as checkpoint:
            orbitals, ci = checkpoint["mo_coeff"], list(checkpoint["ci"])
        np.testing.assert_allclose(orbitals.T @ mol.overlap() @ orbitals, np.eye(mol.nao),
                                   atol=1e-7, err_msg="Checkpoint orbitals do not match the AO basis")
        mf = dressed.RHF(verbose=0)
        mf.mo_coeff = orbitals
        mf.mo_occ = np.r_[np.full(6, 2.), np.zeros(mol.nao - 6)]
        mf.e_nuc = mol.energy_nuc()
        mf.eri_factors = dressed.eri_factors
        mf.cholesky_jk = True
        mc = CASSCF(mf, ncas=12, nelecas=6, multiplicity=1, ci_method="direct_spin0_symm",
                    use_cholesky=True, max_cycle=120, conv_tol=1e-9, conv_tol_grad=1e-5,
                    conv_tol_grad_relaxed=1e-5, step_size=.08, max_step=.025,
                    level_shift=.02, verbose=0)
        mc.state_average([.25] * 4)
        mc.run(nstates=4, mo_coeff=orbitals, ci0=ci)
        if not mc.converged:
            raise RuntimeError(f"SA4 did not converge at node {i}, Q={q:g}")
        energies.append(mc.e_tot + offset)
        potentials.append(photon)
        frames.append(mc.casci.frame())
        np.savez(args.output / f"state_{i:02d}.npz", mo_coeff=mc.mo_coeff, ci=mc.ci)
    overlaps = full(frames, lambda left, right: left.overlap(right), nstates=4)
    np.savez(args.output / "scan.npz", x=dvr.x[0] * np.sqrt(omega))
    np.savez(args.output / "ldr_inputs.npz", Q=dvr.x[0], energies=energies,
             photon_potential=potentials, overlaps=overlaps,
             kinetic=dvr.kinetic().toarray(), domains=dvr.domains)


if __name__ == "__main__":
    main()
