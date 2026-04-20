import numpy as np

from pyqed.mps.decompose import decompose
from pyqed.mps.mps import MPS
from pyqed.qchem import Molecule
from pyqed.qchem.hf import RHF
from pyqed.qchem.mcscf.casci import CASCI
from pyqed.qchem.dmrg import DMRG
from pyqed.qchem.dmrg.overlap import (
    _coefficients_from_dense_states,
    _dense_exact_fock_operator,
    _orbital_transform_mpo,
    _unitary_rotation_mpo,
)
from pyqed.qchem.mcscf.casci import get_combos


def _interleaved_to_grouped_sign(det):
    alpha_occ = np.asarray(det[0], dtype=np.int8)
    beta_occ = np.asarray(det[1], dtype=np.int8)
    occupied_alpha = np.flatnonzero(alpha_occ)
    occupied_beta = np.flatnonzero(beta_occ)
    n_cross = 0
    for p in occupied_beta:
        n_cross += np.count_nonzero(occupied_alpha > p)
    return -1.0 if (n_cross % 2) else 1.0


def _run_h2_solver(atom):
    mol = Molecule(atom=atom, unit="bohr", basis="sto-3g")
    mol.build(driver="gbasis")
    mf = RHF(mol).run()
    casci = CASCI(mf, ncas=2, nelecas=2).run(nstates=1)
    dmrg = DMRG(mf, ncas=2, nelecas=2, D=8, init_guess="cid").build()
    dmrg.run(nstates=1, nsweeps=8, symmetry_list=["charge", "sz"])
    return casci, dmrg


def _basis_mps_from_det(det):
    tensor = np.zeros((2, 2, 2, 2), dtype=complex)
    idx = (int(det[0, 0]), int(det[1, 0]), int(det[0, 1]), int(det[1, 1]))
    tensor[idx] = 1.0
    factors = decompose(tensor, rank=[1, 4, 4, 4, 1])
    return MPS(factors)


def test_dmrg_overlap_matches_casci_for_displaced_h2():
    cas1, dmrg1 = _run_h2_solver("H 0 0 0; H 0 0 1.4")
    cas2, dmrg2 = _run_h2_solver("H 0 0 0; H 0 0 1.5")

    cas_overlap = cas1.overlap(cas2)
    dmrg_overlap = dmrg1.overlap(dmrg2)
    mixed_overlap = dmrg1.overlap(cas2)

    np.testing.assert_allclose(dmrg1.overlap(dmrg1), np.array([[1.0]]), atol=1e-8)
    np.testing.assert_allclose(dmrg_overlap, cas_overlap, atol=1e-6)
    np.testing.assert_allclose(np.abs(mixed_overlap), np.abs(cas_overlap), atol=1e-6)


def test_dmrg_unitary_overlap_matches_exact_bridge_for_active_rotation():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    mol.build(driver="gbasis")
    mf = RHF(mol).run()

    dmrg_ref = DMRG(mf, ncas=2, nelecas=2, D=8, init_guess="cid").build()
    dmrg_ref.run(nstates=1, nsweeps=8, symmetry_list=["charge", "sz"])

    theta = 0.37
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ],
        dtype=float,
    )
    mo_rot = mf.mo_coeff.copy()
    mo_rot[:, :2] = mo_rot[:, :2] @ rotation

    dmrg_rot = DMRG(mf, ncas=2, nelecas=2, D=8, init_guess="cid").build(mo_coeff=mo_rot)
    dmrg_rot.run(nstates=1, nsweeps=8, symmetry_list=["charge", "sz"])

    via_unitary = dmrg_ref.overlap_unitary(dmrg_rot)
    via_explicit = dmrg_ref.overlap_unitary(dmrg_rot, orbital_transform=rotation)

    np.testing.assert_allclose(via_unitary, via_explicit, atol=1e-6)


def test_unitary_rotation_mpo_matches_exact_fock_operator_on_basis():
    theta = 0.37
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ],
        dtype=complex,
    )

    mo_occ = np.array([[1, 0], [1, 0]], dtype=np.int8)
    binary = np.asarray(get_combos(mo_occ, space="fci"), dtype=np.int8)
    basis_indices = []
    for det in binary:
        bits = (int(det[0, 0]), int(det[1, 0]), int(det[0, 1]), int(det[1, 1]))
        basis_indices.append(np.ravel_multi_index(bits, (2, 2, 2, 2)))

    exact_dense = _dense_exact_fock_operator(rotation)
    exact_block = exact_dense[np.ix_(basis_indices, basis_indices)]
    signs = np.array([_interleaved_to_grouped_sign(det) for det in binary], dtype=complex)
    exact_block = np.diag(signs) @ exact_block

    mpo = _unitary_rotation_mpo(rotation, mpo_bond_dim=64, order=8, scale=2)
    cols = []
    for det in binary:
        state = _basis_mps_from_det(det)
        transformed = (mpo @ state).compress(64).normalize()
        coeff = _coefficients_from_dense_states([transformed], binary)[0]
        cols.append(coeff)
    mpo_block = np.column_stack(cols)

    np.testing.assert_allclose(mpo_block, exact_block, atol=1e-12)


def test_factorized_orbital_transform_fallback_matches_exact_fock_operator_on_basis():
    transform = np.array(
        [
            [1.03, 0.07],
            [-0.04, 0.96],
        ],
        dtype=complex,
    )

    mo_occ = np.array([[1, 0], [1, 0]], dtype=np.int8)
    binary = np.asarray(get_combos(mo_occ, space="fci"), dtype=np.int8)
    basis_indices = []
    for det in binary:
        bits = (int(det[0, 0]), int(det[1, 0]), int(det[0, 1]), int(det[1, 1]))
        basis_indices.append(np.ravel_multi_index(bits, (2, 2, 2, 2)))

    exact_dense = _dense_exact_fock_operator(transform)
    exact_block = exact_dense[np.ix_(basis_indices, basis_indices)]
    signs = np.array([_interleaved_to_grouped_sign(det) for det in binary], dtype=complex)
    exact_block = np.diag(signs) @ exact_block

    mpo = _orbital_transform_mpo(
        transform,
        mpo_bond_dim=128,
        dense_exact_max_spin_orbitals=0,
    )
    cols = []
    for det in binary:
        state = _basis_mps_from_det(det)
        transformed = mpo @ state
        coeff = _coefficients_from_dense_states([transformed], binary)[0]
        cols.append(coeff)
    mpo_block = np.column_stack(cols)

    np.testing.assert_allclose(mpo_block, exact_block, atol=1e-12)


def test_dmrg_biorthogonal_overlap_tracks_exact_bridge_for_displaced_h2():
    _, dmrg1 = _run_h2_solver("H 0 0 0; H 0 0 1.4")
    _, dmrg2 = _run_h2_solver("H 0 0 0; H 0 0 1.5")

    exact = dmrg1.overlap(dmrg2)
    structured_biorth = dmrg1.overlap_biorthogonal(dmrg2)
    mpo_biorth = dmrg1.overlap_biorthogonal(dmrg2, backend="mpo")

    np.testing.assert_allclose(dmrg1.overlap_biorthogonal(dmrg1), np.array([[1.0]]), atol=1e-8)
    np.testing.assert_allclose(structured_biorth, exact, atol=1e-6)
    np.testing.assert_allclose(mpo_biorth, exact, atol=1e-6)


def test_dmrg_biorthogonal_overlap_diagnostics_identify_mpo_error():
    _, dmrg1 = _run_h2_solver("H 0 0 0; H 0 0 1.4")
    _, dmrg2 = _run_h2_solver("H 0 0 0; H 0 0 1.5")

    diag = dmrg1.overlap_biorthogonal_diagnostics(dmrg2)

    np.testing.assert_allclose(diag["exact_overlap"], diag["exact_bridge_overlap"], atol=1e-6)
    np.testing.assert_allclose(diag["structured_overlap"], diag["exact_bridge_overlap"], atol=1e-6)
    np.testing.assert_allclose(diag["mpo_overlap_from_ci"], diag["mpo_overlap_direct"], atol=1e-6)
    np.testing.assert_allclose(diag["mpo_overlap_direct"], diag["exact_bridge_overlap"], atol=1e-6)
    assert diag["left_mpo_method"] == "dense_exact"
    assert diag["right_mpo_method"] == "dense_exact"


def test_dmrg_auto_overlap_uses_polar_mode_for_displaced_h2():
    _, dmrg1 = _run_h2_solver("H 0 0 0; H 0 0 1.4")
    _, dmrg2 = _run_h2_solver("H 0 0 0; H 0 0 1.5")

    auto, info = dmrg1.overlap_auto(dmrg2, return_info=True)
    polar = dmrg1.overlap_unitary(dmrg2, use_polar=True)

    assert info["mode"] == "polar"
    assert info["active_unitarity_error"] > info["unitary_tol"]
    np.testing.assert_allclose(auto, polar, atol=1e-8)
    assert 0.0 <= np.abs(auto[0, 0]) <= 1.0
