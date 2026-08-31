#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark pyqed singlet-triplet SOC matrix elements against a PySCF reference.

The comparison is phase-invariant: CI/state phases are arbitrary, so the
benchmark reports |<S|H_SO|T>| rather than the raw complex sign.
"""

from dataclasses import dataclass

import numpy as np
from pyscf import ao2mo, fci, gto, mcscf, scf

from pyqed.qchem import Molecule, st_soc
from pyqed.qchem.soc import (
    soc_1e_prefactor,
    spatial_soc_to_spin_orbital,
)
from pyqed.units import au2wavenumber


AU2CM = au2wavenumber


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


def one_center_soc_ao_pyscf(mol):
    hso_ao = np.zeros((3, mol.nao_nr(), mol.nao_nr()), dtype=float)
    aoslices = mol.aoslice_by_atom()
    for ia in range(mol.natm):
        p0, p1 = aoslices[ia, 2], aoslices[ia, 3]
        with mol.with_rinv_as_nucleus(ia):
            w = mol.intor("int1e_prinvxp", comp=3)
        hso_ao[:, p0:p1, p0:p1] += (-mol.atom_charge(ia)) * w[:, p0:p1, p0:p1]
    hso_ao *= soc_1e_prefactor()
    return hso_ao


def one_center_soc_mo_pyscf(mol, mo_cas):
    hso_ao = one_center_soc_ao_pyscf(mol)
    return np.einsum("xpq,pi,qj->xij", hso_ao, mo_cas.conj(), mo_cas)


def somf_soc_mo_pyscf(mf, mo_cas):
    mol = mf.mol
    dm = mf.make_rdm1()
    g = mol.intor("int2e_p1vxp1", comp=3)
    term1 = np.einsum("xpqrs,rs->xpq", g, dm, optimize=True)
    term2 = np.einsum("xprsq,rs->xpq", g, dm, optimize=True)
    term3 = np.einsum("xsqpr,rs->xpq", g, dm, optimize=True)
    hso_ao = one_center_soc_ao_pyscf(mol)
    hso_ao += soc_1e_prefactor() * (term1 - 1.5 * term2 - 1.5 * term3)
    return np.einsum("xpq,pi,qj->xij", hso_ao, mo_cas.conj(), mo_cas)


def pyscf_string_occupations(norb, nelec):
    """Return PySCF FCI string occupations in PySCF's packed-string order."""
    strings = fci.cistring.gen_strings4orblist(range(norb), nelec)
    return [
        tuple(orb for orb in range(norb) if int(string) & (1 << orb))
        for string in strings
    ]


def ci_to_state(ci, norb, na, nb):
    alpha = pyscf_string_occupations(norb, na)
    beta = pyscf_string_occupations(norb, nb)
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


def determinant_count(norb, nelec):
    from math import comb

    na, nb = nelec
    return comb(norb, na) * comb(norb, nb)


def fci_root_by_s2(solver, h1, eri, norb, nelec, target_s2, root=0, ecore=0.0):
    ndet = determinant_count(norb, nelec)
    nroots = min(ndet, max(root + 1, root + 8))
    while True:
        e, c = solver.kernel(h1, eri, norb, nelec, nroots=nroots, ecore=ecore)
        if np.isscalar(e):
            energies = np.asarray([e], dtype=float)
            coeffs = [c]
        else:
            energies = np.asarray(e, dtype=float)
            coeffs = list(c)

        s2 = np.asarray(
            [fci.spin_op.spin_square0(ci, norb, nelec)[0] for ci in coeffs],
            dtype=float,
        )
        selected = [
            i for i in np.argsort(energies)
            if abs(s2[i] - target_s2) <= 1.0e-5
        ]
        if len(selected) > root:
            idx = selected[root]
            return float(energies[idx]), coeffs[idx], float(s2[idx])
        if nroots >= ndet:
            detail = ", ".join(
                f"root {i}: S^2={s2[i]:.6g}" for i in range(min(6, len(s2)))
            )
            raise RuntimeError(
                f"Could not find PySCF root {root} with target S^2={target_s2}; {detail}"
            )
        nroots = min(ndet, max(nroots + 4, 2 * nroots))


def pyscf_reference(case, soc_model="1e"):
    mol = gto.M(atom=case.atom, basis=case.basis, unit="angstrom", spin=0, verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    mo = mf.mo_coeff
    mo_cas = mo[:, case.ncore:case.ncore + case.ncas]

    h1, ecore = mcscf.casci.h1e_for_cas(mf, mo, case.ncas, case.ncore)
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mo_cas), case.ncas)

    na = case.nelecas // 2
    nb = case.nelecas - na
    solver = fci.direct_spin1.FCI(mol)
    e_s, c_s, _ = fci_root_by_s2(
        solver, h1, eri, case.ncas, (na, nb), target_s2=0.0, ecore=ecore
    )
    triplet_nelec = {
        -1: (na - 1, nb + 1),
        0: (na, nb),
        1: (na + 1, nb - 1),
    }
    triplets = {
        ms: fci_root_by_s2(
            solver, h1, eri, case.ncas, nelec, target_s2=2.0, ecore=ecore
        )
        for ms, nelec in triplet_nelec.items()
    }

    if soc_model == "1e":
        hso_spatial = one_center_soc_mo_pyscf(mol, mo_cas)
    elif soc_model == "somf":
        hso_spatial = somf_soc_mo_pyscf(mf, mo_cas)
    else:
        raise ValueError("soc_model must be '1e' or 'somf'.")

    hso = spatial_soc_to_spin_orbital(hso_spatial, order="grouped")
    singlet_state = ci_to_state(c_s, case.ncas, na, nb)
    components = {
        ms: one_body_me(
            hso,
            singlet_state,
            ci_to_state(ci, case.ncas, *triplet_nelec[ms]),
        )
        for ms, (_, ci, _) in triplets.items()
    }
    norm = float(np.sqrt(sum(abs(value) ** 2 for value in components.values())))
    return float(e_s), {ms: data[0] for ms, data in triplets.items()}, components, norm


def pyqed_workflow(case, soc_model="1e"):
    mol = Molecule(atom=case.atom, unit="angstrom", basis=case.basis)
    mol.build()

    # SOC matrix elements are sensitive to small occupied/virtual rotations.
    mf = mol.RHF().run(tol=1e-12, conv_tol_dm=1e-10)
    result = st_soc(
        mf,
        ncas=case.ncas,
        nelecas=case.nelecas,
        ncore=case.ncore,
        model=soc_model,
        dm=mf.make_rdm1() if soc_model == "somf" else None,
        method="direct_ci",
    )
    return (
        float(result.singlet.e_tot[result.singlet_root]),
        {ms: float(mc.e_tot[result.triplet_root]) for ms, mc in result.triplets.items()},
        result.components,
        result.norm,
    )


def main():
    header = (
        f"{'Molecule':<8} {'Model':<6} {'Ms':>4} "
        f"{'|SOC| PySCF (cm^-1)':>20} {'|SOC| pyqed (cm^-1)':>20} "
        f"{'RelErr(|SOC|)':>15} {'dE_S (Ha)':>12} {'dE_T (Ha)':>12}"
    )
    print(header)
    print("-" * len(header))

    for case in CASES:
        for soc_model in ("1e", "somf"):
            e_s_ref, e_t_ref, soc_ref, norm_ref = pyscf_reference(case, soc_model=soc_model)
            e_s_py, e_t_py, soc_py, norm_py = pyqed_workflow(case, soc_model=soc_model)

            for ms in (-1, 0, 1):
                soc_ref_abs = abs(soc_ref[ms])
                soc_py_abs = abs(soc_py[ms])
                rel = abs(soc_py_abs - soc_ref_abs) / max(soc_ref_abs, 1e-16)

                print(
                    f"{case.name:<8} "
                    f"{soc_model:<6} "
                    f"{ms:4d} "
                    f"{soc_ref_abs * AU2CM:20.6f} "
                    f"{soc_py_abs * AU2CM:20.6f} "
                    f"{rel:15.6e} "
                    f"{abs(e_s_ref - e_s_py):12.3e} "
                    f"{abs(e_t_ref[ms] - e_t_py[ms]):12.3e}"
                )

            norm_rel = abs(norm_py - norm_ref) / max(norm_ref, 1e-16)
            print(
                f"{case.name:<8} "
                f"{soc_model:<6} "
                f"{'norm':>4} "
                f"{norm_ref * AU2CM:20.6f} "
                f"{norm_py * AU2CM:20.6f} "
                f"{norm_rel:15.6e} "
                f"{abs(e_s_ref - e_s_py):12.3e} "
                f"{max(abs(e_t_ref[ms] - e_t_py[ms]) for ms in (-1, 0, 1)):12.3e}"
            )


if __name__ == "__main__":
    main()
