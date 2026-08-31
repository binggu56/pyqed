"""Benchmark fixed-orbital PyQED CASCI NACs against a PySCF CASCI reference.

PySCF does not expose analytic NACs for ``mcscf.CASCI``; its
``pyscf.nac.sacasscf.NonAdiabaticCouplings`` expects CASSCF orbital-response
metadata.  This script therefore builds a PySCF finite-difference derivative-H
CASCI reference with fixed reference orbitals and compares it to
``pyqed.qchem.nac.sacasscf.casci_nac(..., moving_basis=False)``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed.qchem import Molecule
from pyqed.qchem.mcscf.direct_ci import CASCI
from pyqed.qchem.nac.sacasscf import casci_nac


@dataclass(frozen=True)
class Case:
    name: str
    atom: str
    basis: str
    ncas: int
    nelecas: int | tuple[int, int]
    nroots: int
    pair: tuple[int, int]
    charge: int = 0
    spin: int = 0


CASES = {
    "h2": Case(
        name="H2 / STO-3G / CASCI(2,2) / pair 0,1",
        atom="H 0 0 0; H 0.15 0.05 1.40",
        basis="sto-3g",
        ncas=2,
        nelecas=2,
        nroots=2,
        pair=(0, 1),
    ),
    "lih": Case(
        name="LiH / STO-3G / CASCI(2,2) / pair 0,1",
        atom="Li 0 0 0; H 0.20 0.10 3.00",
        basis="sto-3g",
        ncas=2,
        nelecas=2,
        nroots=2,
        pair=(0, 1),
    ),
    "h2o": Case(
        name="H2O asym / STO-3G / CASCI(4,4) / pair 0,2",
        atom="O 0 0 0; H 0 1.45 1.05; H 0 -1.25 1.2",
        basis="sto-3g",
        ncas=4,
        nelecas=4,
        nroots=3,
        pair=(0, 2),
    ),
}


def best_error(vec, ref):
    vec = np.asarray(vec, dtype=float).reshape(-1)
    ref = np.asarray(ref, dtype=float).reshape(-1)
    sign = 1 if np.linalg.norm(vec - ref) <= np.linalg.norm(vec + ref) else -1
    diff = vec - sign * ref
    return sign, float(np.linalg.norm(diff)), float(np.max(np.abs(diff)))


def atom_with_coords(symbols, coords):
    return "; ".join(
        f"{symbol} {x:.14f} {y:.14f} {z:.14f}"
        for symbol, (x, y, z) in zip(symbols, np.asarray(coords).reshape(-1, 3), strict=True)
    )


def build_pyqed(case):
    mol = Molecule(
        atom=case.atom,
        basis=case.basis,
        charge=case.charge,
        spin=case.spin,
        unit="bohr",
    )
    mol.build(eri="dense")
    mf = mol.RHF(verbose=0).run(max_cycle=100)
    mc = CASCI(mf, ncas=case.ncas, nelecas=case.nelecas, verbose=0).run(
        nstates=case.nroots,
        method="direct_ci",
    )
    return mol, mf, mc


def build_pyscf(case, atom=None):
    from pyscf import gto, mcscf, scf

    mol = gto.M(
        atom=case.atom if atom is None else atom,
        basis=case.basis,
        charge=case.charge,
        spin=case.spin,
        unit="bohr",
        verbose=0,
    )
    mf = scf.RHF(mol).run(verbose=0)
    mc = mcscf.CASCI(mf, case.ncas, case.nelecas)
    mc.fcisolver.nroots = case.nroots
    mc.kernel()
    return mol, mf, mc


def pyscf_active_integrals(mc, mo_coeff):
    from pyscf import ao2mo

    h1, ecore = mc.get_h1eff(mo_coeff=mo_coeff)
    h2 = ao2mo.restore(1, mc.get_h2eff(mo_coeff=mo_coeff), mc.ncas)
    return np.asarray(h1, dtype=float), np.asarray(h2, dtype=float), float(ecore)


def pyscf_h_matrix(mc, h1, h2):
    from pyscf import fci

    out = np.zeros((len(mc.ci), len(mc.ci)), dtype=float)
    for bra in range(len(mc.ci)):
        for ket in range(len(mc.ci)):
            dm1, dm2 = fci.direct_spin1.trans_rdm12(
                mc.ci[bra],
                mc.ci[ket],
                mc.ncas,
                mc.nelecas,
            )
            out[bra, ket] = (
                np.einsum("pq,qp->", h1, dm1, optimize=True)
                + 0.5 * np.einsum("pqrs,pqrs->", h2, dm2, optimize=True)
            )
    return out


def pyscf_fd_casci_nac(case, step):
    ref_mol, ref_mf, ref_mc = build_pyscf(case)
    symbols = [atom[0] for atom in ref_mol._atom]
    coords = np.asarray(ref_mol.atom_coords(), dtype=float)
    mo_coeff = np.asarray(ref_mc.mo_coeff, dtype=float)
    energies = np.asarray(ref_mc.e_tot[: case.nroots], dtype=float)
    ncoord = coords.size
    h_derivs = np.zeros((case.nroots, case.nroots, ncoord), dtype=float)

    for coord in range(ncoord):
        delta = np.zeros_like(coords).reshape(-1)
        delta[coord] = step
        plus_atom = atom_with_coords(symbols, coords.reshape(-1) + delta)
        minus_atom = atom_with_coords(symbols, coords.reshape(-1) - delta)
        _, _, plus_mc = build_pyscf(case, atom=plus_atom)
        _, _, minus_mc = build_pyscf(case, atom=minus_atom)
        h1p, h2p, _ = pyscf_active_integrals(plus_mc, mo_coeff)
        h1m, h2m, _ = pyscf_active_integrals(minus_mc, mo_coeff)
        hp = pyscf_h_matrix(ref_mc, h1p, h2p)
        hm = pyscf_h_matrix(ref_mc, h1m, h2m)
        h_derivs[:, :, coord] = (hp - hm) / (2.0 * step)

    beta, alpha = case.pair
    gap = energies[alpha] - energies[beta]
    nac = h_derivs[beta, alpha] / gap
    return ref_mc, nac


def print_case(case, step):
    _, _, pyqed_mc = build_pyqed(case)
    pyqed = casci_nac(pyqed_mc, state_pairs=[case.pair], moving_basis=False).nac[case.pair]
    pyscf_mc, pyscf = pyscf_fd_casci_nac(case, step)
    sign, err_norm, err_max = best_error(pyqed, pyscf)

    print(f"\n{case.name}")
    print(f"  pair nac[{case.pair[0]},{case.pair[1]}], step={step:.1e} bohr")
    print("  PyQED energies:", np.array2string(np.asarray(pyqed_mc.e_tot[: case.nroots]), precision=10))
    print("  PySCF energies:", np.array2string(np.asarray(pyscf_mc.e_tot[: case.nroots]), precision=10))
    print(f"  max |dE|: {np.max(np.abs(pyqed_mc.e_tot[: case.nroots] - pyscf_mc.e_tot[: case.nroots])):.3e} Eh")
    print(f"  best sign: {sign:+d}")
    print(f"  diff norm/max: {err_norm:.3e} / {err_max:.3e}")
    print("  pyqed:", np.array2string(np.asarray(pyqed).reshape(-1), precision=8, suppress_small=True))
    print("  pyscf:", np.array2string(sign * np.asarray(pyscf).reshape(-1), precision=8, suppress_small=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[*CASES.keys(), "all"], default="all")
    parser.add_argument("--step", type=float, default=2.0e-4)
    args = parser.parse_args()

    keys = CASES.keys() if args.case == "all" else [args.case]
    for key in keys:
        print_case(CASES[key], args.step)


if __name__ == "__main__":
    main()
