"""Benchmark PyQED IR(RKS) against a PySCF RKS reference for H2O.

The PySCF build used in some developer environments does not ship the
``pyscf.prop.infrared`` helper.  To keep the comparison portable, this script
uses PySCF for the RKS Hessian and displaced dipoles, then evaluates the same
IR stick intensity definition used by :class:`pyqed.qchem.IR`.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

import numpy as np
from pyscf import dft, gto

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyqed.qchem import IR, Molecule
from pyqed.qchem.dft import AOGrid, RKS
from pyqed.qchem.dft.hessian import analyze_cartesian_hessian


ATOM = (
    ("O", (0.0000000000, 0.0000000000, 0.0000000000)),
    ("H", (0.0000000000, -1.4323367300, 1.1071526600)),
    ("H", (0.0000000000, 1.4323367300, 1.1071526600)),
)


def _atom_with_coords(coords):
    return [(sym, tuple(coord)) for (sym, _), coord in zip(ATOM, coords)]


def build_pyqed_rks(*, basis, xc, n_radial, n_angular, conv_tol):
    mol = Molecule(atom=list(ATOM), unit="bohr", basis=basis)
    mol.build(driver="gbasis")
    grid = AOGrid.atom_centered(
        mol,
        n_radial=n_radial,
        n_angular=n_angular,
        with_grad=True,
    )
    mf = RKS(mol, grid=grid, xc=xc)
    mf.max_cycle = 80
    mf.conv_tol = conv_tol
    mf.run()
    return mf


def build_pyscf_rks(coords, *, basis, xc, n_radial, n_angular, conv_tol):
    mol = gto.M(
        atom=_atom_with_coords(coords),
        unit="Bohr",
        basis=basis,
        verbose=0,
    )
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.grids.atom_grid = {sym: (n_radial, n_angular) for sym, _ in ATOM}
    mf.conv_tol = conv_tol
    mf.kernel()
    return mf


def pyscf_harmonic_analysis(mf):
    hess_4d = mf.Hessian().kernel()
    natom = mf.mol.natm
    hess = np.asarray(hess_4d).transpose(0, 2, 1, 3).reshape(3 * natom, 3 * natom)
    return analyze_cartesian_hessian(
        hess,
        mf.mol.atom_coords(),
        mf.mol.atom_mass_list(isotope_avg=True),
        remove_translation_rotation=True,
    )


def pyscf_ir_reference(mf, *, basis, xc, n_radial, n_angular, conv_tol, dipole_step):
    vib = pyscf_harmonic_analysis(mf)
    coords0 = np.asarray(mf.mol.atom_coords(), dtype=float)

    def dipole_fn(coords):
        displaced = build_pyscf_rks(
            coords,
            basis=basis,
            xc=xc,
            n_radial=n_radial,
            n_angular=n_angular,
            conv_tol=conv_tol,
        )
        return np.asarray(displaced.dip_moment(unit="AU", verbose=0), dtype=float)

    dipole_derivatives = IR.finite_difference_dipole_derivatives(
        dipole_fn,
        coords0,
        vib["modes"],
        step=dipole_step,
    )
    return IR.from_harmonic_analysis(vib, dipole_derivatives=dipole_derivatives).run()


def _print_table(pyqed_ir, pyscf_ir):
    header = (
        "mode  pyqed_freq/cm^-1  pyscf_freq/cm^-1  dfreq      "
        "pyqed_I/au      pyscf_I/au      dI"
    )
    print(header)
    print("-" * len(header))
    for idx, (fq, fp, iq, ip) in enumerate(
        zip(
            pyqed_ir.frequencies,
            pyscf_ir.frequencies,
            pyqed_ir.intensities,
            pyscf_ir.intensities,
        ),
        start=1,
    ):
        print(
            f"{idx:4d}  {fq:16.8f}  {fp:16.8f}  {fq - fp:9.2e}  "
            f"{iq:14.8e}  {ip:14.8e}  {iq - ip:9.2e}"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--xc", default="svwn")
    parser.add_argument("--n-radial", type=int, default=10)
    parser.add_argument("--n-angular", type=int, default=50)
    parser.add_argument("--hessian-step", type=float, default=2.0e-3)
    parser.add_argument("--dipole-step", type=float, default=2.0e-3)
    parser.add_argument("--conv-tol", type=float, default=1.0e-9)
    args = parser.parse_args(argv)

    pyqed_mf = build_pyqed_rks(
        basis=args.basis,
        xc=args.xc,
        n_radial=args.n_radial,
        n_angular=args.n_angular,
        conv_tol=args.conv_tol,
    )
    pyqed_ir = IR.from_method(
        pyqed_mf,
        hessian_step=args.hessian_step,
        dipole_step=args.dipole_step,
    ).run()

    coords0 = np.array([coord for _, coord in ATOM], dtype=float)
    pyscf_mf = build_pyscf_rks(
        coords0,
        basis=args.basis,
        xc=args.xc,
        n_radial=args.n_radial,
        n_angular=args.n_angular,
        conv_tol=args.conv_tol,
    )
    pyscf_ir = pyscf_ir_reference(
        pyscf_mf,
        basis=args.basis,
        xc=args.xc,
        n_radial=args.n_radial,
        n_angular=args.n_angular,
        conv_tol=args.conv_tol,
        dipole_step=args.dipole_step,
    )

    _print_table(pyqed_ir, pyscf_ir)
    print()
    print(f"PyQED E(RKS) = {pyqed_mf.e_tot:.12f} Eh")
    print(f"PySCF E(RKS) = {pyscf_mf.e_tot:.12f} Eh")
    print(f"max |dfreq|  = {np.max(np.abs(pyqed_ir.frequencies - pyscf_ir.frequencies)):.6e} cm^-1")
    print(f"max |dI|     = {np.max(np.abs(pyqed_ir.intensities - pyscf_ir.intensities)):.6e} au")


if __name__ == "__main__":
    main()
