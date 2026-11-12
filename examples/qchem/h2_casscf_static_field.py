"""Verify native PyQED GTO CASSCF(2,2) for H2 in a static electric field."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pyscf import gto, mcscf, scf

from pyqed.qchem import Molecule
from pyqed.qchem.mcscf.casscf import CASSCF


GEOMETRY = "H 0 0 -0.7; H 0 0 0.7"
BASIS = "6-31g"


def pyqed_energy(field):
    mol = Molecule(atom=GEOMETRY, basis=BASIS, unit="bohr")
    mol.build(eri="s8")
    mol.hcore = mol.hcore + field * mol.position_integral()[2]
    mf = mol.RHF(verbose=0).run(tol=1e-11)
    mc = CASSCF(
        mf, ncas=2, nelecas=2, verbose=0,
        conv_tol=1e-10, conv_tol_grad=1e-7,
    ).run()
    if not mc.converged:
        raise RuntimeError(f"PyQED CASSCF did not converge at field {field:g}")
    nuclear_field = -field * (mol.atom_charges() @ mol.atom_coords()[:, 2])
    return float(np.ravel(mc.e_tot)[0] + nuclear_field)


def pyscf_energy(field):
    mol = gto.M(atom=GEOMETRY, basis=BASIS, unit="bohr", verbose=0)
    mf = scf.RHF(mol)
    hcore = mf.get_hcore() + field * mol.intor_symmetric("int1e_r", comp=3)[2]
    mf.get_hcore = lambda mol=None: hcore
    mf.conv_tol = 1e-11
    mf.kernel()
    mc = mcscf.CASSCF(mf, 2, 2)
    mc.conv_tol = 1e-10
    mc.conv_tol_grad = 1e-6
    mc.kernel()
    if not mf.converged or not mc.converged:
        raise RuntimeError(f"PySCF did not converge at field {field:g}")
    nuclear_field = -field * (mol.atom_charges() @ mol.atom_coords()[:, 2])
    return mc.e_tot + nuclear_field


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("h2_casscf_static_field.png"))
    args = parser.parse_args()

    fields = np.linspace(-0.02, 0.02, 5)
    pyqed = np.array([pyqed_energy(field) for field in fields])
    pyscf = np.array([pyscf_energy(field) for field in fields])
    zero = len(fields) // 2
    shift = pyqed - pyqed[zero]
    np.testing.assert_allclose(pyqed, pyscf, atol=1e-8, rtol=0)
    np.testing.assert_allclose(pyqed, pyqed[::-1], atol=1e-9, rtol=0)
    if np.any(shift[fields != 0] >= -1e-6):
        raise AssertionError("Expected a negative Stark shift at nonzero field")

    print("Field (a.u.)       PyQED (Ha)          PySCF (Ha)       Difference (Ha)")
    for field, energy, reference in zip(fields, pyqed, pyscf):
        print(f"{field:+.3f}          {energy:.12f}     {reference:.12f}    {energy-reference:+.2e}")
    step = fields[zero + 1]
    alpha = -(shift[zero - 1] + shift[zero + 1]) / step**2
    print(f"Central-difference polarizability at step {step:g} a.u.: {alpha:.6f} a.u.")

    fig, ax = plt.subplots(figsize=(6, 3.7), layout="constrained")
    ax.plot(fields, 1e3 * shift, "o-", label="PyQED")
    ax.plot(fields, 1e3 * (pyscf - pyscf[zero]), "x", label="PySCF")
    ax.set(xlabel="Axial electric field (a.u.)", ylabel="Energy shift (mHa)",
           title="H₂ CASSCF(2,2)/6-31G at 1.4 bohr")
    ax.legend()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(f"Figure: {args.output.resolve()}")


if __name__ == "__main__":
    main()
