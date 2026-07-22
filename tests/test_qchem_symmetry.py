import numpy as np
import pytest

from pyqed.qchem import Molecule
from pyqed.qchem.hf import RHF
from pyqed.qchem.mcscf.casci import CASCI
from pyqed.qchem.symmetry import (
    determinant_irrep_labels,
    determinant_linear_momenta,
    get_point_group,
    linear_irrep_momentum,
    linear_irrep_symb2id,
)


def test_c2v_ao_irreps_for_linear_hf_gbasis():
    mol = Molecule(
        atom="H 0 0 0; F 0 0 0.9",
        unit="angstrom",
        basis="cc-pvdz",
    )
    mol.build(driver="gbasis", symmetry="c2v")

    assert mol.groupname == "C2v"
    assert mol.irrep_names == ("A1", "A2", "B1", "B2")

    labels = dict(zip(mol.ao_labels(), mol.ao_irrep_labels))
    assert labels["1 F 2px"] == "B1"
    assert labels["1 F 2py"] == "B2"
    assert labels["1 F 2pz"] == "A1"
    assert labels["1 F 3dxy"] == "A2"
    assert labels["1 F 3dyz"] == "B2"
    assert labels["1 F 3dz2"] == "A1"
    assert labels["1 F 3dxz"] == "B1"
    assert labels["1 F 3dx2-y2"] == "A1"


def test_coov_ao_irreps_for_linear_hf_gbasis():
    mol = Molecule(
        atom="H 0 0 0; F 0 0 0.9",
        unit="angstrom",
        basis="cc-pvdz",
    )
    mol.build(driver="gbasis", symmetry="coov")

    assert mol.groupname == "Coov"
    assert set(mol.irrep_names) >= {"A1", "A2", "E1x", "E1y", "E2x", "E2y"}

    labels = dict(zip(mol.ao_labels(), mol.ao_irrep_labels))
    assert labels["1 F 2px"] == "E1x"
    assert labels["1 F 2py"] == "E1y"
    assert labels["1 F 2pz"] == "A1"
    assert labels["1 F 3dxy"] == "E2y"
    assert labels["1 F 3dyz"] == "E1y"
    assert labels["1 F 3dz2"] == "A1"
    assert labels["1 F 3dxz"] == "E1x"
    assert labels["1 F 3dx2-y2"] == "E2x"


def test_c2v_ao_irreps_for_builtin_cartesian_basis():
    mol = Molecule(
        atom="H 0 0 0; F 0 0 0.9",
        unit="angstrom",
        basis="sto-3g",
    )
    mol.build(driver="builtin", eri="dense", symmetry="c2v")

    labels = dict(zip(mol.ao_labels(), mol.ao_irrep_labels))
    assert labels["1 F 2px"] == "B1"
    assert labels["1 F 2py"] == "B2"
    assert labels["1 F 2pz"] == "A1"


def test_c2v_symmetry_rejects_off_axis_geometry():
    mol = Molecule(
        atom="H 0.1 0 0; F 0 0 0.9",
        unit="angstrom",
        basis="sto-3g",
    )

    with pytest.raises(ValueError, match="not invariant"):
        mol.build(driver="gbasis", symmetry="c2v")


def test_rhf_assigns_native_mo_irreps():
    mol = Molecule(
        atom="H 0 0 0; F 0 0 0.9",
        unit="angstrom",
        basis="sto-3g",
    )
    mol.build(driver="gbasis", symmetry="c2v")

    mf = RHF(mol).run(verbose=0, max_cycle=50)

    assert mf.groupname == "C2v"
    assert len(mf.orb_irrep_labels) == mol.nao
    assert len(mf.orb_sym) == mol.nao
    assert set(mf.orb_irrep_labels) >= {"A1", "B1", "B2"}
    assert set(mf.orb_irrep_labels) <= set(mol.irrep_names)


def test_c2v_determinant_irrep_labels():
    group = get_point_group("c2v")
    orbital_ids = [
        group.irrep_id("A1"),
        group.irrep_id("B1"),
        group.irrep_id("B2"),
    ]
    binary = np.asarray(
        [
            [[1, 0, 0], [1, 0, 0]],
            [[1, 0, 0], [0, 1, 0]],
            [[0, 1, 0], [0, 1, 0]],
            [[0, 1, 0], [0, 0, 1]],
        ],
        dtype=np.int8,
    )

    assert determinant_irrep_labels(binary, orbital_ids, group) == (
        "A1",
        "B1",
        "A1",
        "A2",
    )


def test_coov_determinant_labels_carry_axial_momentum():
    group = get_point_group("coov")
    orbital_ids = [
        linear_irrep_symb2id("coov", "A1"),
        linear_irrep_symb2id("coov", "E1x"),
        linear_irrep_symb2id("coov", "E1y"),
    ]
    binary = np.asarray(
        [
            [[1, 0, 0], [1, 0, 0]],
            [[1, 0, 0], [0, 1, 0]],
            [[1, 0, 0], [0, 0, 1]],
            [[0, 1, 0], [0, 0, 1]],
            [[0, 1, 0], [0, 1, 0]],
            [[0, 0, 1], [0, 0, 1]],
        ],
        dtype=np.int8,
    )

    assert determinant_irrep_labels(binary, orbital_ids, group) == (
        "A1",
        "E1x",
        "E1y",
        "A2",
        "E2x",
        "E2y",
    )
    assert determinant_linear_momenta(binary, orbital_ids, group) == (0, 1, -1, 0, 2, -2)
    assert linear_irrep_momentum(linear_irrep_symb2id("coov", "E2y")) == -2


def test_casci_attaches_active_symmetry_metadata():
    mol = Molecule(
        atom="H 0 0 0; F 0 0 0.9",
        unit="angstrom",
        basis="sto-3g",
    )
    mol.build(driver="gbasis", symmetry="c2v")
    mf = RHF(mol).run(verbose=0, max_cycle=50)

    mc = CASCI(mf, ncas=2, nelecas=2).run(nstates=1, method="direct_ci")

    assert mc.active_symmetry.groupname == "C2v"
    assert len(mc.active_orb_irrep_labels) == 2
    assert len(mc.active_orb_sym) == 2
    assert len(mc.det_irrep_labels) == mc.binary.shape[0]
    assert len(mc.det_irrep_ids) == mc.binary.shape[0]
    assert sum(mc.det_irrep_counts.values()) == mc.binary.shape[0]
    assert set(mc.det_irrep_labels) <= set(mol.irrep_names)


def test_casci_can_filter_determinants_by_wfnsym():
    mol = Molecule(
        atom="H 0 0 0; F 0 0 0.9",
        unit="angstrom",
        basis="sto-3g",
    )
    mol.build(driver="gbasis", symmetry="c2v")
    mf = RHF(mol).run(verbose=0, max_cycle=50)

    full = CASCI(mf, ncas=2, nelecas=2).run(nstates=1, method="direct_ci")
    a1 = CASCI(mf, ncas=2, nelecas=2).run(nstates=1, method="direct_ci", wfnsym="A1")
    b2 = CASCI(mf, ncas=2, nelecas=2).run(nstates=1, method="direct_ci", target_irrep="B2")

    assert full.binary.shape[0] == 4
    assert a1.wfnsym == "A1"
    assert b2.wfnsym == "B2"
    assert a1.binary.shape[0] == full.det_irrep_counts["A1"]
    assert b2.binary.shape[0] == full.det_irrep_counts["B2"]
    assert set(a1.det_irrep_labels) == {"A1"}
    assert set(b2.det_irrep_labels) == {"B2"}


def test_casci_can_filter_coov_wfnsym():
    mol = Molecule(
        atom="H 0 0 0; F 0 0 0.9",
        unit="angstrom",
        basis="sto-3g",
    )
    mol.build(driver="gbasis", symmetry="coov")
    mf = RHF(mol).run(verbose=0, max_cycle=50)

    full = CASCI(mf, ncas=2, nelecas=2).run(nstates=1, method="direct_ci")
    sigma = CASCI(mf, ncas=2, nelecas=2).run(nstates=1, method="direct_ci", wfnsym="A1")
    pi_y = CASCI(mf, ncas=2, nelecas=2).run(nstates=1, method="direct_ci", wfnsym="E1y")

    assert full.active_symmetry.groupname == "Coov"
    assert full.active_orb_irrep_labels == ("E1y", "A1")
    assert full.active_symmetry.orbital_momentum == (-1, 0)
    assert sigma.wfnsym == "A1"
    assert pi_y.wfnsym == "E1y"
    assert set(sigma.det_irrep_labels) == {"A1"}
    assert set(pi_y.det_irrep_labels) == {"E1y"}
    assert set(sigma.active_symmetry.determinant_momentum) == {0}
    assert set(pi_y.active_symmetry.determinant_momentum) == {-1}


def test_native_mo_irreps_match_pyscf_c2v():
    pyscf_gto = pytest.importorskip("pyscf.gto")
    pyscf_scf = pytest.importorskip("pyscf.scf")
    pyscf_symm = pytest.importorskip("pyscf.symm")

    cases = [
        (
            "H 0 0 0; F 0 0 0.9",
            "sto-3g",
            "HF",
        ),
        (
            "H 0 0 0; F 0 0 0.9",
            "cc-pvdz",
            "HF-degenerate-pi",
        ),
        (
            "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
            "sto-3g",
            "H2O",
        ),
    ]

    for atom, basis, _name in cases:
        pmol = pyscf_gto.M(atom=atom, unit="Angstrom", basis=basis, symmetry="C2v", verbose=0)
        pmf = pyscf_scf.RHF(pmol).run(verbose=0)
        pyscf_labels = tuple(
            pyscf_symm.label_orb_symm(pmol, pmol.irrep_name, pmol.symm_orb, pmf.mo_coeff)
        )

        mol = Molecule(atom=atom, unit="angstrom", basis=basis)
        mol.build(driver="gbasis", symmetry="c2v")
        mf = RHF(mol).run(verbose=0, max_cycle=80)

        assert mf.orb_irrep_labels == pyscf_labels
        np.testing.assert_allclose(mf.e_tot, pmf.e_tot, atol=1.0e-6)


def test_native_mo_irreps_match_pyscf_coov():
    pyscf_gto = pytest.importorskip("pyscf.gto")
    pyscf_scf = pytest.importorskip("pyscf.scf")
    pyscf_symm = pytest.importorskip("pyscf.symm")

    atom = "H 0 0 0; F 0 0 0.9"
    basis = "cc-pvdz"
    pmol = pyscf_gto.M(atom=atom, unit="Angstrom", basis=basis, symmetry="Coov", verbose=0)
    pmf = pyscf_scf.RHF(pmol).run(verbose=0)
    pyscf_labels = tuple(
        str(label)
        for label in pyscf_symm.label_orb_symm(pmol, pmol.irrep_name, pmol.symm_orb, pmf.mo_coeff)
    )

    mol = Molecule(atom=atom, unit="angstrom", basis=basis)
    mol.build(driver="gbasis", symmetry="coov")
    mf = RHF(mol).run(verbose=0, max_cycle=80)

    assert mf.orb_irrep_labels == pyscf_labels
    np.testing.assert_allclose(mf.e_tot, pmf.e_tot, atol=1.0e-6)


def test_dooh_labels_for_homonuclear_h2_sigma_orbitals():
    mol = Molecule(
        atom="H 0 0 -0.37; H 0 0 0.37",
        unit="angstrom",
        basis="sto-3g",
    )
    mol.build(driver="gbasis", symmetry="dooh")
    mf = RHF(mol).run(verbose=0, max_cycle=50)

    assert mol.groupname == "Dooh"
    assert mf.orb_irrep_labels == ("A1g", "A1u")
    assert mf.orb_sym == (
        linear_irrep_symb2id("dooh", "A1g"),
        linear_irrep_symb2id("dooh", "A1u"),
    )


def test_native_casci_wfnsym_singlets_match_pyscf_spin0_symm():
    pyscf_gto = pytest.importorskip("pyscf.gto")
    pyscf_scf = pytest.importorskip("pyscf.scf")
    pyscf_mcscf = pytest.importorskip("pyscf.mcscf")
    pyscf_fci = pytest.importorskip("pyscf.fci")

    atom = "H 0 0 0; F 0 0 0.9"
    basis = "sto-3g"
    pmol = pyscf_gto.M(atom=atom, unit="Angstrom", basis=basis, symmetry="C2v", verbose=0)
    pmf = pyscf_scf.RHF(pmol).run(verbose=0)

    mol = Molecule(atom=atom, unit="angstrom", basis=basis)
    mol.build(driver="gbasis", symmetry="c2v")
    mf = RHF(mol).run(verbose=0, max_cycle=80)

    pyscf_refs = {}
    for wfnsym, nroots in (("A1", 2), ("B2", 1)):
        pmc = pyscf_mcscf.CASCI(pmf, 2, 2)
        pmc.fcisolver = pyscf_fci.direct_spin0_symm.FCI(pmol)
        pmc.fcisolver.nroots = nroots
        pmc.fcisolver.wfnsym = wfnsym
        pmc.kernel(pmf.mo_coeff)
        pyscf_e = np.atleast_1d(pmc.e_tot)
        pyscf_refs[wfnsym] = pyscf_e

        mc = CASCI(mf, ncas=2, nelecas=2).run(
            nstates=nroots,
            method="direct_spin0_symm",
            wfnsym=wfnsym,
        )

        assert mc.solver_backend == "direct_spin0_symm_dense"
        assert all(abs(mc.spin_square(idx)) < 1.0e-8 for idx in range(len(mc.e_tot)))
        np.testing.assert_allclose(mc.e_tot[:nroots], pyscf_e[:nroots], atol=1.0e-6)

    from pyqed.qchem.mcscf.direct_ci import CASCI as DirectCASCI

    for wfnsym, nroots in (("A1", 2), ("B2", 1)):
        mc = DirectCASCI(mf, ncas=2, nelecas=2)
        mc.direct_spin0_symm_dense_fallback_nconfigs = 0
        mc.run(nstates=nroots, method="direct_spin0_symm", wfnsym=wfnsym)

        assert mc.solver_backend.startswith("direct_spin0_symm_davidson")
        assert all(abs(mc.spin_square(idx)) < 1.0e-8 for idx in range(len(mc.e_tot)))
        np.testing.assert_allclose(mc.e_tot[:nroots], pyscf_refs[wfnsym][:nroots], atol=1.0e-6)
