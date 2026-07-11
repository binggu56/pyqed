"""Compare PyQED IR spectrum plumbing with a PySCF reference Hessian."""

from pathlib import Path
import sys

import numpy as np
from pyscf import gto, scf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyqed.qchem import IR
from pyqed.qchem.dft.hessian import analyze_cartesian_hessian


ATOM = [("H", (0.0, 0.0, 0.0)), ("F", (0.0, 0.0, 1.7329))]
BASIS = "sto-3g"


def make_mf(coords):
    mol = gto.M(
        atom=[(sym, tuple(coord)) for (sym, _), coord in zip(ATOM, coords)],
        unit="Bohr",
        basis=BASIS,
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.conv_tol = 1.0e-12
    mf.kernel()
    return mol, mf


def main():
    coords0 = np.array([coord for _, coord in ATOM], dtype=float)
    mol, mf = make_mf(coords0)

    hess_4d = mf.Hessian().kernel()
    natom = mol.natm
    hess = np.asarray(hess_4d).transpose(0, 2, 1, 3).reshape(3 * natom, 3 * natom)
    vib = analyze_cartesian_hessian(
        hess,
        coords0,
        mol.atom_mass_list(isotope_avg=True),
        remove_translation_rotation=True,
    )
    frequencies = np.asarray(vib["freq_cm1"]).real
    modes = np.asarray(vib["modes"], dtype=float)

    def dipole_fn(coords):
        _, step_mf = make_mf(coords)
        return np.asarray(step_mf.dip_moment(unit="AU", verbose=0), dtype=float)

    dipole_derivatives = IR.finite_difference_dipole_derivatives(
        dipole_fn,
        coords0,
        modes,
        step=2.0e-3,
    )
    ir = IR.from_harmonic_analysis(
        vib,
        dipole_derivatives=dipole_derivatives,
    ).run()

    manual_intensity = np.einsum("kx,kx->k", dipole_derivatives, dipole_derivatives)

    print("PyQED harmonic-analysis frequencies / cm^-1:", frequencies)
    print("PyQED IR frequencies / cm^-1:", ir.frequencies)
    print("frequency max abs diff:", np.max(np.abs(ir.frequencies - frequencies)))
    print("dipole derivatives / au:", ir.dipole_derivatives)
    print("PyQED IR intensities / au:", ir.intensities)
    print("manual intensities / au:", manual_intensity)
    print("intensity max abs diff:", np.max(np.abs(ir.intensities - manual_intensity)))


if __name__ == "__main__":
    main()
