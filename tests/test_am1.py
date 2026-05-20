import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from pyqed.qchem import Molecule
from pyqed.qchem.semiempirical.am1 import HARTREE2EV, RAM1, UAM1


def _am1_mrci_integrals(mf):
    h1_spatial = mf.get_hcore_mo()
    eri_spatial = mf.get_eri_mo()
    eri_aa = eri_spatial - eri_spatial.swapaxes(1, 3)
    return (
        np.asarray([h1_spatial, h1_spatial]),
        np.stack(
            (
                np.stack((eri_aa, eri_spatial)),
                np.stack((eri_spatial, eri_aa)),
            )
        ),
    )


def _apply_one_spin(occ, create, annihilate):
    occ = np.asarray(occ, dtype=np.int8).copy()
    if occ[annihilate] == 0:
        return None, 0
    sign = -1 if int(np.sum(occ[:annihilate])) % 2 else 1
    occ[annihilate] = 0
    if occ[create] == 1:
        return None, 0
    sign *= -1 if int(np.sum(occ[:create])) % 2 else 1
    occ[create] = 1
    return occ, sign


def _annihilate_one(occ, orbital):
    if occ[orbital] == 0:
        return None, 0
    sign = -1 if int(np.sum(occ[:orbital])) % 2 else 1
    occ = occ.copy()
    occ[orbital] = 0
    return occ, sign


def _create_one(occ, orbital):
    if occ[orbital] == 1:
        return None, 0
    sign = -1 if int(np.sum(occ[:orbital])) % 2 else 1
    occ = occ.copy()
    occ[orbital] = 1
    return occ, sign


def _apply_same_spin_pair(occ, create1, annihilate1, create2, annihilate2):
    """Apply a^+_create1 a^+_create2 a_annihilate2 a_annihilate1."""
    occ, phase1 = _annihilate_one(occ, annihilate1)
    if phase1 == 0:
        return None, 0
    occ, phase2 = _annihilate_one(occ, annihilate2)
    if phase2 == 0:
        return None, 0
    occ, phase3 = _create_one(occ, create2)
    if phase3 == 0:
        return None, 0
    occ, phase4 = _create_one(occ, create1)
    if phase4 == 0:
        return None, 0
    return occ, phase1 * phase2 * phase3 * phase4


def _bruteforce_ci_hamiltonian(binary, h1, h2):
    """Independent CI Hamiltonian in the alpha/beta product determinant basis."""
    index = {
        (tuple(det[0].tolist()), tuple(det[1].tolist())): i
        for i, det in enumerate(binary)
    }
    h = np.zeros((len(binary), len(binary)))
    nspin, _nspin2, nmo, _nmo2, _nmo3, _nmo4 = h2.shape
    for j, det in enumerate(binary):
        for spin in range(nspin):
            for p in range(nmo):
                for q in range(nmo):
                    new_occ, phase = _apply_one_spin(det[spin], p, q)
                    if phase == 0:
                        continue
                    bra = [det[0].copy(), det[1].copy()]
                    bra[spin] = new_occ
                    i = index.get((tuple(bra[0].tolist()), tuple(bra[1].tolist())))
                    if i is not None:
                        h[i, j] += phase * h1[spin, p, q]

        for spin1 in range(nspin):
            for spin2 in range(nspin):
                for p in range(nmo):
                    for q in range(nmo):
                        for r in range(nmo):
                            for s in range(nmo):
                                bra = [det[0].copy(), det[1].copy()]
                                if spin1 == spin2:
                                    occ, phase = _apply_same_spin_pair(
                                        det[spin1], p, q, r, s
                                    )
                                    if phase == 0:
                                        continue
                                    bra[spin1] = occ
                                else:
                                    occ2, phase2 = _apply_one_spin(det[spin2], r, s)
                                    if phase2 == 0:
                                        continue
                                    occ1, phase1 = _apply_one_spin(det[spin1], p, q)
                                    if phase1 == 0:
                                        continue
                                    bra[spin2] = occ2
                                    bra[spin1] = occ1
                                    phase = phase1 * phase2
                                i = index.get(
                                    (tuple(bra[0].tolist()), tuple(bra[1].tolist()))
                                )
                                if i is not None:
                                    h[i, j] += 0.5 * phase * h2[spin1, spin2, p, q, r, s]
    return h


def _mopac_h2_meci_relative_energies():
    mopac = shutil.which("mopac")
    if mopac is None:
        pytest.skip("MOPAC executable not found")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "h2_meci.mop"
        path.write_text(
            "\n".join(
                (
                    "AM1 1SCF MECI C.I.=2 SINGLET",
                    "H2 AM1 MECI",
                    "",
                    "H 0.0 0 0.0 0 0.0 0",
                    "H 0.0 0 0.0 0 0.74 0",
                    "",
                )
            ),
            encoding="utf-8",
        )
        subprocess.run(
            [mopac, str(path)],
            cwd=tmp,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        out = path.with_suffix(".out").read_text(encoding="utf-8")

    rel = []
    for line in out.splitlines():
        match = re.match(r"\s*\d+\+?\s+[-+]?\d+\.\d+\s+([-+]?\d+\.\d+)\s+", line)
        if match:
            rel.append(float(match.group(1)))
    if len(rel) < 4:
        raise AssertionError("Could not parse four MOPAC AM1/MECI roots for H2.")
    return np.asarray(rel[:4])


def _run_heat(atom):
    mol = Molecule(atom=atom, unit="Angstrom")
    mf = RAM1(mol).run(conv_tol=1.0e-8, verbose=0)
    assert mf.converged
    return float(mf.e_heat_formation), float(mf.e_tot), mf


def test_am1_hf_is_atom_order_invariant():
    heat_hf, e_hf, mf_hf = _run_heat("H 0 0 0; F 0.9168 0 0")
    heat_fh, e_fh, mf_fh = _run_heat("F 0.9168 0 0; H 0 0 0")

    np.testing.assert_allclose(e_hf, e_fh, atol=1.0e-10)
    np.testing.assert_allclose(heat_hf, heat_fh, atol=1.0e-8)

    pop_hf = [
        np.trace(mf_hf.make_rdm1()[p0:p1, p0:p1])
        for p0, p1 in mf_hf._mindo_mol.aoslice_by_atom()[:, 2:]
    ]
    pop_fh = [
        np.trace(mf_fh.make_rdm1()[p0:p1, p0:p1])
        for p0, p1 in mf_fh._mindo_mol.aoslice_by_atom()[:, 2:]
    ]
    np.testing.assert_allclose(pop_hf, pop_fh[::-1], atol=1.0e-8)


def test_am1_acetylene_is_atom_order_invariant():
    heat_hcch, e_hcch, _ = _run_heat(
        "H -1.6615 0 0; C -0.6015 0 0; C 0.6015 0 0; H 1.6615 0 0"
    )
    heat_cchh, e_cchh, _ = _run_heat(
        "C -0.6015 0 0; C 0.6015 0 0; H -1.6615 0 0; H 1.6615 0 0"
    )

    np.testing.assert_allclose(e_hcch, e_cchh, atol=1.0e-10)
    np.testing.assert_allclose(heat_hcch, heat_cchh, atol=1.0e-8)


def test_am1_uses_am1_atomic_reference_energy():
    heat, _e_tot, _mf = _run_heat(
        "O 0.0 0.0 0.0; H 0.75695 0.0 0.58588; H -0.75695 0.0 0.58588"
    )
    np.testing.assert_allclose(heat, -59.26702070792, atol=1.0e-8)


def test_am1_eri_tensor_reproduces_scf_jk():
    _heat, _e_tot, mf = _run_heat(
        "O 0.0 0.0 0.0; H 0.75695 0.0 0.58588; H -0.75695 0.0 0.58588"
    )
    eri = mf.get_eri_ao()
    dm = mf.make_rdm1()
    vj, vk = mf.get_jk(dm=dm)

    vj_from_eri = np.einsum("pqrs,rs->pq", eri, dm, optimize=True)
    vk_from_eri = np.einsum("prqs,rs->pq", eri, dm, optimize=True)

    np.testing.assert_allclose(vj, vj_from_eri, atol=1.0e-12)
    np.testing.assert_allclose(vk, vk_from_eri, atol=1.0e-12)


def test_am1_mrci_runs_on_ram1_reference():
    from pyqed.qchem.semiempirical import MECI, MRCI

    mol = Molecule(atom="H 0 0 0; H 0 0 0.74", unit="Angstrom")
    mf = RAM1(mol).run(conv_tol=1.0e-8, verbose=0)

    mrci = MRCI(mf, nstates=2, full=True).run()
    meci = MECI(mf, nstates=2, ncas=2).run()
    via_method = mf.MECI(nstates=2, ncas=2).run()

    assert mrci.e.shape == (2,)
    assert mrci.ci.shape == (4, 2)
    assert mrci.determinants.shape == (4, 2, 2)
    np.testing.assert_allclose(mrci.e, meci.e, atol=1.0e-12)
    np.testing.assert_allclose(meci.e, via_method.e, atol=1.0e-12)


def test_am1_meci_computes_spin_square_for_closed_shell_roots():
    mol = Molecule(atom="H 0 0 0; H 0 0 0.74", unit="Angstrom")
    mf = RAM1(mol).run(conv_tol=1.0e-8, verbose=0)

    fci = mf.MECI(nstates=4, ncas=2).run()

    assert fci.s2.shape == (4,)
    np.testing.assert_allclose(fci.spin_square(), fci.s2)
    np.testing.assert_allclose(sorted(np.round(fci.s2, 8)), [0.0, 0.0, 0.0, 2.0])


def test_am1_meci_spin_penalty_can_select_triplet_root():
    mol = Molecule(atom="H 0 0 0; H 0 0 0.74", unit="Angstrom")
    mf = RAM1(mol).run(conv_tol=1.0e-8, verbose=0)

    singlet = mf.MECI(nstates=1, ncas=2).run()
    triplet = mf.MECI(
        nstates=1,
        ncas=2,
        spin_penalty=10.0,
        target_spin=1.0,
    ).run()

    np.testing.assert_allclose(singlet.s2[0], 0.0, atol=1.0e-10)
    np.testing.assert_allclose(triplet.s2[0], 2.0, atol=1.0e-10)
    assert triplet.e_penalized is not None
    assert triplet.e[0] > singlet.e[0]


def test_am1_mrci_hamiltonian_matches_independent_second_quantization():
    from pyqed.qchem.semiempirical import MRCI

    mol = Molecule(atom="H 0 0 0; H 0 0 0.74", unit="Angstrom")
    mf = RAM1(mol).run(conv_tol=1.0e-8, verbose=0)
    driver = MRCI(mf, nstates=4, full=True)

    h_ci = mf.build_mrci_hamiltonian(driver)
    h1, h2 = _am1_mrci_integrals(mf)
    h_ref = _bruteforce_ci_hamiltonian(driver.determinants, h1, h2)

    np.testing.assert_allclose(h_ci, h_ref, atol=1.0e-12)


def test_am1_mrci_is_variational_with_larger_ci_spaces():
    mol = Molecule(
        atom="O 0 0 0; H 0.75695 0 0.58588; H -0.75695 0 0.58588",
        unit="Angstrom",
    )
    mf = RAM1(mol).run(conv_tol=1.0e-8, verbose=0)

    ref = mf.MRCI(nstates=1, singles=False, doubles=False).run()
    cis = mf.MRCI(nstates=1, singles=True, doubles=False).run()
    cisd = mf.MRCI(nstates=1, singles=True, doubles=True).run()
    fci = mf.MECI(nstates=1, ncas=mf.nao).run()

    assert fci.e[0] <= cisd.e[0] + 1.0e-12
    assert cisd.e[0] <= cis.e[0] + 1.0e-12
    assert cis.e[0] <= ref.e[0] + 1.0e-12
    assert ref.e[0] <= mf.e_tot + 1.0e-12


def test_am1_full_ci_is_invariant_to_mo_rotation():
    mol = Molecule(atom="H 0 0 0; H 0 0 0.74", unit="Angstrom")
    mf = RAM1(mol).run(conv_tol=1.0e-8, verbose=0)
    fci = mf.MECI(nstates=4, ncas=2).run()

    angle = 0.31
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    mf_rot = RAM1(mol)
    mf_rot.build()
    mf_rot.mo_coeff = mf.mo_coeff @ rotation
    mf_rot.mo_occ = mf.mo_occ.copy()
    mf_rot.mo_energy = mf.mo_energy.copy()
    mf_rot.e_tot = mf.e_tot
    mf_rot.converged = mf.converged

    fci_rot = mf_rot.MECI(nstates=4, ncas=2).run()

    np.testing.assert_allclose(fci.e, fci_rot.e, atol=1.0e-12)


def test_am1_full_ci_h2_matches_mopac_meci_relative_energies():
    mol = Molecule(atom="H 0 0 0; H 0 0 0.74", unit="Angstrom")
    mf = RAM1(mol).run(conv_tol=1.0e-10, verbose=0)
    fci = mf.MECI(nstates=4, ncas=2).run()

    pyqed_rel = (fci.e - fci.e[0]) * HARTREE2EV
    mopac_rel = _mopac_h2_meci_relative_energies()

    np.testing.assert_allclose(pyqed_rel, mopac_rel, atol=2.0e-3)


def test_am1_meci_wavefunction_overlap_uses_mo_transport():
    theta = np.pi / 3.0
    atom_a = (
        "H 1.6 0 0; "
        "H 0 0 0; "
        f"H {1.6 * np.cos(theta)} {1.6 * np.sin(theta)} 0"
    )
    atom_b = (
        "H 1.7 0 0; "
        "H 0 0 0; "
        f"H {1.6 * np.cos(theta)} {1.6 * np.sin(theta)} 0"
    )
    mf_a = RAM1(Molecule(atom=atom_a, charge=1, spin=0, unit="bohr")).run(
        conv_tol=1.0e-10,
        verbose=0,
    )
    mf_a2 = RAM1(Molecule(atom=atom_a, charge=1, spin=0, unit="bohr")).run(
        conv_tol=1.0e-10,
        verbose=0,
    )
    mf_b = RAM1(Molecule(atom=atom_b, charge=1, spin=0, unit="bohr")).run(
        conv_tol=1.0e-10,
        verbose=0,
    )

    np.testing.assert_allclose(
        mf_a.get_mo_cross_overlap(mf_a2),
        np.eye(mf_a.nao),
        atol=1.0e-10,
    )

    meci_a = mf_a.MECI(nstates=3, ncas=3).run()
    meci_a2 = mf_a2.MECI(nstates=3, ncas=3).run()
    meci_b = mf_b.MECI(nstates=3, ncas=3).run()

    np.testing.assert_allclose(meci_a.wavefunction_overlap(meci_a), np.eye(3), atol=1.0e-8)
    same_geometry = meci_a.wavefunction_overlap(meci_a2)
    np.testing.assert_allclose(same_geometry.T @ same_geometry, np.eye(3), atol=1.0e-8)
    np.testing.assert_allclose(abs(same_geometry[0, 0]), 1.0, atol=1.0e-8)
    transported = meci_a.wavefunction_overlap(meci_b)
    pseudo = meci_a.ci.T @ meci_b.ci
    assert transported.shape == (3, 3)
    assert not np.allclose(transported, pseudo, atol=1.0e-6)


def test_uam1_runs_open_shell_no2_doublet():
    mol = Molecule(
        atom="N 0 0 0; O 1.2 0 0; O -0.6 1.0392304845 0",
        charge=0,
        spin=1,
        unit="Angstrom",
    )

    mf = UAM1(mol).run(conv_tol=1.0e-7, max_cycle=100, damping=0.35, verbose=0)

    assert mf.converged
    assert mf.nelec_alpha_beta == (9, 8)
    assert mf.mo_coeff.shape == (2, mf.nao, mf.nao)
    assert mf.mo_energy.shape == (2, mf.nao)
    np.testing.assert_allclose(mf.mo_occ.sum(), mf.nelec)
    np.testing.assert_allclose(np.trace(mf.dm[0]), 9.0, atol=1.0e-10)
    np.testing.assert_allclose(np.trace(mf.dm[1]), 8.0, atol=1.0e-10)
    assert np.linalg.norm(mf.dm[0] - mf.dm[1]) > 1.0e-6


def test_uam1_meci_runs_on_open_shell_no2():
    mol = Molecule(
        atom="N 0 0 0; O 1.2 0 0; O -0.6 1.0392304845 0",
        charge=0,
        spin=1,
        unit="Angstrom",
    )
    mf = UAM1(mol).run(conv_tol=1.0e-7, max_cycle=100, damping=0.35, verbose=0)

    meci = mf.MECI(nstates=3, ncas=3).run()

    assert meci.e.shape == (3,)
    assert meci.ci.shape == (9, 3)
    assert meci.determinants.shape == (9, 2, mf.nao)
    assert meci.active_orbitals == (7, 8, 9)
    np.testing.assert_allclose(meci.determinants[:, 0, :].sum(axis=1), 9)
    np.testing.assert_allclose(meci.determinants[:, 1, :].sum(axis=1), 8)
    assert np.all(np.isfinite(meci.e))
    assert meci.s2.shape == (3,)
    assert np.all(np.isfinite(meci.s2))
    assert np.all(meci.s2 >= 0.75 - 1.0e-8)


def test_uam1_meci_spin_penalty_reduces_doublet_contamination():
    mol = Molecule(
        atom="N 0 0 0; O 1.2 0 0; O -0.6 1.0392304845 0",
        charge=0,
        spin=1,
        unit="Angstrom",
    )
    mf = UAM1(mol).run(conv_tol=1.0e-7, max_cycle=100, damping=0.35, verbose=0)

    unfiltered = mf.MECI(nstates=3, ncas=4).run()
    filtered = mf.MECI(
        nstates=3,
        ncas=4,
        spin_penalty=5.0,
        target_spin=0.5,
    ).run()

    assert filtered.e_penalized is not None
    assert np.max(np.abs(filtered.s2 - 0.75)) < np.max(np.abs(unfiltered.s2 - 0.75))


def test_uam1_meci_overlap_uses_spin_resolved_mo_transport():
    atom_a = "N 0 0 0; O 1.2 0 0; O -0.6 1.0392304845 0"
    atom_b = "N 0 0 0; O 1.22 0 0; O -0.61 1.0565510 0"
    mf_a = UAM1(
        Molecule(atom=atom_a, charge=0, spin=1, unit="Angstrom")
    ).run(conv_tol=1.0e-7, max_cycle=100, damping=0.35, verbose=0)
    mf_a2 = UAM1(
        Molecule(atom=atom_a, charge=0, spin=1, unit="Angstrom")
    ).run(conv_tol=1.0e-7, max_cycle=100, damping=0.35, verbose=0)
    mf_b = UAM1(
        Molecule(atom=atom_b, charge=0, spin=1, unit="Angstrom")
    ).run(conv_tol=1.0e-7, max_cycle=100, damping=0.35, verbose=0)

    same_mo_overlap = mf_a.get_mo_cross_overlap(mf_a2)
    np.testing.assert_allclose(same_mo_overlap, np.repeat(np.eye(mf_a.nao)[None], 2, axis=0), atol=1.0e-8)

    meci_a = mf_a.MECI(nstates=3, ncas=3).run()
    meci_a2 = mf_a2.MECI(nstates=3, ncas=3).run()
    meci_b = mf_b.MECI(nstates=3, ncas=3).run()

    same_geometry = meci_a.wavefunction_overlap(meci_a2)
    np.testing.assert_allclose(same_geometry.T @ same_geometry, np.eye(3), atol=1.0e-7)
    transported = meci_a.wavefunction_overlap(meci_b)
    pseudo = meci_a.ci.T @ meci_b.ci
    assert transported.shape == (3, 3)
    assert not np.allclose(transported, pseudo, atol=1.0e-6)


def test_am1_meci_chooses_frontier_active_space():
    mol = Molecule(
        atom="O 0 0 0; H 0.75695 0 0.58588; H -0.75695 0 0.58588",
        unit="Angstrom",
    )
    mf = RAM1(mol).run(conv_tol=1.0e-8, verbose=0)

    meci = mf.MECI(nstates=4).run()

    assert meci.determinants.shape == (4, 2, mf.nao)
    assert meci.active_orbitals == (3, 4)
    np.testing.assert_allclose(meci.determinants[:, :, :3], 1)
    np.testing.assert_allclose(meci.determinants[:, :, 5:], 0)


def test_am1_meci_accepts_explicit_active_orbitals():
    mol = Molecule(
        atom="O 0 0 0; H 0.75695 0 0.58588; H -0.75695 0 0.58588",
        unit="Angstrom",
    )
    mf = RAM1(mol).run(conv_tol=1.0e-8, verbose=0)

    by_ncas = mf.MECI(nstates=4, ncas=2).run()
    explicit = mf.MECI(nstates=4, active_orbitals=[3, 4]).run()

    assert explicit.active_orbitals == (3, 4)
    np.testing.assert_allclose(explicit.e, by_ncas.e, atol=1.0e-12)


def test_am1_meci_odd_ncas_biases_frontier_space_to_occupied_side():
    mol = Molecule(
        atom="O 0 0 0; H 0.75695 0 0.58588; H -0.75695 0 0.58588",
        unit="Angstrom",
    )
    mf = RAM1(mol).run(conv_tol=1.0e-8, verbose=0)

    meci = mf.MECI(nstates=6, ncas=3).run()

    assert meci.determinants.shape == (9, 2, mf.nao)
    assert meci.active_orbitals == (2, 3, 4)
    np.testing.assert_allclose(meci.determinants[:, :, :2], 1)
    np.testing.assert_allclose(meci.determinants[:, :, 5:], 0)
