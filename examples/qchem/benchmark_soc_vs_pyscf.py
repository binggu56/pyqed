#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark pyqed singlet-triplet SOC matrix elements against a PySCF reference.

The comparison is phase-invariant: CI/state phases are arbitrary, so the
benchmark reports |<S|H_SO|T>| rather than the raw complex sign.
"""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from pyscf import ao2mo, fci, gto, mcscf, scf

from pyqed.qchem import Molecule
from pyqed.qchem.mcscf.direct_ci import CASCI as PyqedCASCI
from pyqed.qchem.soc import get_soc_1e_mo, soc_1e_prefactor, spatial_soc_to_spin_orbital


AU2CM = 219474.6313705


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    atom: str
    basis: str
    ncas: int
    nelecas: int
    ncore: int


CASES = [
    BenchmarkCase(
        name="CH2",
        atom="""
        C  0.000000  0.000000  0.000000
        H  0.000000  1.030000  1.110000
        H  0.000000 -1.030000  1.110000
        """,
        basis="sto-3g",
        ncas=2,
        nelecas=2,
        ncore=3,
    ),
    BenchmarkCase(
        name="CH2S",
        atom="""
        C  0.000000  0.000000  0.000000
        S  0.000000  0.000000  1.610000
        H  0.940000  0.000000 -0.540000
        H -0.940000  0.000000 -0.540000
        """,
        basis="sto-3g",
        ncas=4,
        nelecas=4,
        ncore=10,
    ),
    BenchmarkCase(
        name="SO2",
        atom="""
        S  0.000000  0.000000  0.000000
        O  0.000000  1.432000  1.108000
        O  0.000000 -1.432000  1.108000
        """,
        basis="sto-3g",
        ncas=4,
        nelecas=4,
        ncore=14,
    ),
]


def one_center_soc_mo_pyscf(mol, mo_cas):
    hso_ao = np.zeros((3, mol.nao_nr(), mol.nao_nr()), dtype=float)
    aoslices = mol.aoslice_by_atom()
    for ia in range(mol.natm):
        p0, p1 = aoslices[ia, 2], aoslices[ia, 3]
        with mol.with_rinv_as_nucleus(ia):
            w = mol.intor("int1e_prinvxp", comp=3)
        hso_ao[:, p0:p1, p0:p1] += (-mol.atom_charge(ia)) * w[:, p0:p1, p0:p1]
    hso_ao *= soc_1e_prefactor()
    return np.einsum("xpq,pi,qj->xij", hso_ao, mo_cas.conj(), mo_cas)


def ci_to_state(ci, norb, na, nb):
    alpha = list(combinations(range(norb), na))
    beta = list(combinations(range(norb), nb))
    coeffs = {}
    for ia, occ_a in enumerate(alpha):
        for ib, occ_b in enumerate(beta):
            coeff = ci[ia, ib]
            if abs(coeff) > 1e-14:
                coeffs[tuple(list(occ_a) + [norb + x for x in occ_b])] = coeff
    return coeffs


def apply_a(det, q):
    det = list(det)
    if q not in det:
        return None
    pos = det.index(q)
    sign = -1 if pos % 2 else 1
    det.pop(pos)
    return sign, tuple(det)


def apply_adag(det, p):
    det = list(det)
    if p in det:
        return None
    pos = sum(1 for x in det if x < p)
    sign = -1 if pos % 2 else 1
    det.insert(pos, p)
    return sign, tuple(det)


def one_body_me(h1, bra, ket):
    value = 0j
    for det_k, ck in ket.items():
        for q in range(h1.shape[0]):
            ann = apply_a(det_k, q)
            if ann is None:
                continue
            sign_ann, det1 = ann
            for p in range(h1.shape[0]):
                hpq = h1[p, q]
                if abs(hpq) < 1e-14:
                    continue
                cre = apply_adag(det1, p)
                if cre is None:
                    continue
                sign_cre, det_b = cre
                cb = bra.get(det_b)
                if cb is not None:
                    value += np.conj(cb) * ck * sign_ann * sign_cre * hpq
    return value


def pyscf_reference(case):
    mol = gto.M(atom=case.atom, basis=case.basis, unit="angstrom", spin=0, verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    mo = mf.mo_coeff
    mo_cas = mo[:, case.ncore:case.ncore + case.ncas]

    h1, ecore = mcscf.casci.h1e_for_cas(mf, mo, case.ncas, case.ncore)
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mo_cas), case.ncas)

    na = case.nelecas // 2
    nb = case.nelecas - na
    solver = fci.direct_spin1.FCI(mol)
    e_s, c_s = solver.kernel(h1, eri, case.ncas, (na, nb), nroots=1, ecore=ecore)
    e_t, c_t = solver.kernel(h1, eri, case.ncas, (na + 1, nb - 1), nroots=1, ecore=ecore)
    if not np.isscalar(e_s):
        e_s, c_s = e_s[0], c_s[0]
    if not np.isscalar(e_t):
        e_t, c_t = e_t[0], c_t[0]

    hso = spatial_soc_to_spin_orbital(one_center_soc_mo_pyscf(mol, mo_cas), order="grouped")
    soc = one_body_me(
        hso,
        ci_to_state(c_s, case.ncas, na, nb),
        ci_to_state(c_t, case.ncas, na + 1, nb - 1),
    )
    return float(e_s), float(e_t), soc


def pyqed_workflow(case):
    mol = Molecule(atom=case.atom, unit="angstrom", basis=case.basis)
    mol.build(driver="gbasis-pyscf")

    mf = mol.RHF().run()
    mc_s = PyqedCASCI(
        mf,
        ncas=case.ncas,
        nelecas=case.nelecas,
        ncore=case.ncore,
        spin=0,
    ).run(nstates=1, method="direct_ci")
    mc_t = PyqedCASCI(
        mf,
        ncas=case.ncas,
        nelecas=case.nelecas,
        ncore=case.ncore,
        spin=2,
    ).run(nstates=1, method="direct_ci")

    hso = spatial_soc_to_spin_orbital(
        get_soc_1e_mo(mf, mo_coeff=mc_s.mo_cas, one_center=True),
        order="grouped",
    )
    soc = mc_s.soc_matrix_element(0, other=mc_t, hso=hso, order="grouped")
    return float(mc_s.e_tot[0]), float(mc_t.e_tot[0]), soc


def main():
    header = (
        f"{'Molecule':<8} {'|SOC| PySCF (cm^-1)':>20} {'|SOC| pyqed (cm^-1)':>20} "
        f"{'RelErr(|SOC|)':>15} {'dE_S (Ha)':>12} {'dE_T (Ha)':>12}"
    )
    print(header)
    print("-" * len(header))

    for case in CASES:
        e_s_ref, e_t_ref, soc_ref = pyscf_reference(case)
        e_s_py, e_t_py, soc_py = pyqed_workflow(case)

        soc_ref_abs = abs(soc_ref)
        soc_py_abs = abs(soc_py)
        rel = abs(soc_py_abs - soc_ref_abs) / max(soc_ref_abs, 1e-16)

        print(
            f"{case.name:<8} "
            f"{soc_ref_abs * AU2CM:20.6f} "
            f"{soc_py_abs * AU2CM:20.6f} "
            f"{rel:15.6e} "
            f"{abs(e_s_ref - e_s_py):12.3e} "
            f"{abs(e_t_ref - e_t_py):12.3e}"
        )


if __name__ == "__main__":
    main()
