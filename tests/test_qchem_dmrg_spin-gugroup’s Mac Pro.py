import numpy as np
import pytest

from pyqed.qchem import Molecule
from pyqed.qchem.hf import RHF
from pyqed.qchem.dmrg.dmrg import (
    DMRG,
    _build_spatial_active_hamiltonian_matrix,
    _build_spatial_fermion_operators,
    _build_spin_orbital_dense_hamiltonian_tensor_mpo,
    _group_spin_orbital_mpo_pairs,
    _build_spatial_hamiltonian_tensor_mpo,
    _build_spatial_s2_matrix,
    _build_s2_term_map,
    _build_spin_purification_term_map,
)
from pyqed.mps.mps import _mpo_to_dense_operator


def _kron_all(ops):
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def _spin_orbital_annihilation(site, nsites):
    ident = np.eye(2)
    z = np.diag([1.0, -1.0])
    a = np.array([[0.0, 1.0], [0.0, 0.0]])
    ops = [z] * site + [a] + [ident] * (nsites - site - 1)
    return _kron_all(ops)


def _spin_orbital_creation(site, nsites):
    return _spin_orbital_annihilation(site, nsites).T


def _dense_from_symbolic_term_map(term_map, nsites):
    ident = np.eye(2)
    loc = {
        "a": np.array([[0.0, 1.0], [0.0, 0.0]]),
        r"a^\dagger": np.array([[0.0, 0.0], [1.0, 0.0]]),
        "n": np.diag([0.0, 1.0]),
        "sigma_z": np.diag([1.0, -1.0]),
    }
    out = np.zeros((2**nsites, 2**nsites), dtype=complex)
    for (symbol, dofs), factor in term_map.items():
        pieces = symbol.split()
        ops = [ident.copy() for _ in range(nsites)]
        for site, piece in zip(dofs, pieces):
            ops[site] = loc[piece]
        out += factor * _kron_all(ops)
    return out


def _dense_spin_square(ncas):
    nsites = 2 * ncas
    sp = np.zeros((2**nsites, 2**nsites), dtype=complex)
    sm = np.zeros_like(sp)
    sz = np.zeros_like(sp)
    for p in range(ncas):
        cup_dag = _spin_orbital_creation(2 * p, nsites)
        cup = _spin_orbital_annihilation(2 * p, nsites)
        cdn_dag = _spin_orbital_creation(2 * p + 1, nsites)
        cdn = _spin_orbital_annihilation(2 * p + 1, nsites)
        sp += cup_dag @ cdn
        sm += cdn_dag @ cup
        sz += 0.5 * (cup_dag @ cup - cdn_dag @ cdn)
    return sz @ sz + 0.5 * (sp @ sm + sm @ sp)


def test_dmrg_s2_term_map_matches_exact_dense_spin_square():
    ncas = 2
    nsites = 2 * ncas
    exact = _dense_spin_square(ncas)
    symbolic = _dense_from_symbolic_term_map(_build_s2_term_map(ncas), nsites)
    np.testing.assert_allclose(symbolic, exact, atol=1e-12)


def test_spin_purification_term_map_is_scaled_s2():
    ncas = 3
    shift = 0.37
    ref = _build_s2_term_map(ncas, scale=shift)
    got = _build_spin_purification_term_map(ncas, shift)
    assert set(ref) == set(got)
    for key in ref:
        assert np.allclose(got[key], ref[key], atol=1e-12)


def test_spatial_site_spin_square_resolves_singlet_and_triplet():
    ops = _build_spatial_fermion_operators(2)
    s2 = _build_spatial_s2_matrix(ops)
    singlet = np.zeros(16, dtype=complex)
    triplet = np.zeros(16, dtype=complex)
    singlet[1 * 4 + 2] = 1.0 / np.sqrt(2.0)
    singlet[2 * 4 + 1] = -1.0 / np.sqrt(2.0)
    triplet[1 * 4 + 2] = 1.0 / np.sqrt(2.0)
    triplet[2 * 4 + 1] = 1.0 / np.sqrt(2.0)

    assert np.vdot(singlet, s2 @ singlet).real == pytest.approx(0.0, abs=1e-12)
    assert np.vdot(triplet, s2 @ triplet).real == pytest.approx(2.0, abs=1e-12)


def test_spatial_symbolic_mpo_matches_dense_reference():
    h1 = np.array([[0.2, 0.03], [0.03, -0.1]])
    eri_aa = np.zeros((2, 2, 2, 2))
    eri_aa[0, 0, 0, 0] = 0.7
    eri_aa[1, 1, 1, 1] = 0.5
    eri_aa[0, 0, 1, 1] = 0.2
    eri_aa[1, 1, 0, 0] = 0.2
    h2 = np.stack((np.stack((eri_aa, eri_aa.copy())), np.stack((eri_aa.copy(), eri_aa.copy()))))

    dense_ref, _ = _build_spatial_active_hamiltonian_matrix([h1, h1], h2)
    mpo, _, _ = _build_spatial_hamiltonian_tensor_mpo([h1, h1], h2)

    np.testing.assert_allclose(_mpo_to_dense_operator(mpo), dense_ref, atol=1e-12)


def test_grouped_spin_orbital_mpo_matches_spatial_dense_reference():
    h1 = np.array([[0.2, 0.03], [0.03, -0.1]])
    eri_aa = np.zeros((2, 2, 2, 2))
    eri_aa[0, 0, 0, 0] = 0.7
    eri_aa[1, 1, 1, 1] = 0.5
    eri_aa[0, 0, 1, 1] = 0.2
    eri_aa[1, 1, 0, 0] = 0.2
    h2 = np.stack((np.stack((eri_aa, eri_aa.copy())), np.stack((eri_aa.copy(), eri_aa.copy()))))

    dense_ref, _ = _build_spatial_active_hamiltonian_matrix([h1, h1], h2)
    spin_mpo, _, _ = _build_spin_orbital_dense_hamiltonian_tensor_mpo([h1, h1], h2, 2)
    grouped = _group_spin_orbital_mpo_pairs(spin_mpo)

    np.testing.assert_allclose(_mpo_to_dense_operator(grouped), dense_ref, atol=1e-12)


def test_spatial_dmrg_supports_abelian_charge_sz_symmetry():
    mol = Molecule(atom="H 0 0 0; H 0 0 0.74", unit="angstrom", basis="sto-3g")
    mol.build(driver="pyscf")
    mf = mol.RHF(verbose=0).run()

    dense = DMRG(mf, ncas=2, nelecas=2, D=8, init_guess="hf", site="spatial", verbose=0)
    dense.run(nsweeps=4, symmetry_list=None)
    sym = DMRG(mf, ncas=2, nelecas=2, D=8, init_guess="hf", site="spatial", verbose=0)
    sym.run(nsweeps=4, symmetry_list=["charge", "sz"])

    assert hasattr(sym.dmrg.ground_state.Bs[0], "qns")
    assert sym.e_tot == pytest.approx(dense.e_tot, abs=1e-8)


def test_spatial_dmrg_routes_su2_to_nonabelian_backend():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    mol.build(driver="gbasis")
    mf = RHF(mol).run()

    dense = DMRG(mf, ncas=2, nelecas=2, D=8, init_guess="hf", site="spatial", verbose=0)
    dense.run(nsweeps=4, symmetry_list=None)
    su2 = DMRG(mf, ncas=2, nelecas=2, D=8, init_guess="hf", site="spatial", verbose=0)
    su2.run(nsweeps=4, symmetry_list=["charge", "su2"])

    assert su2.dmrg.backend == "nonabelian"
    assert su2.e_tot == pytest.approx(dense.e_tot, abs=1e-7)


def test_spatial_su2_dmrg_auto_recoupling_preserves_small_chain_reference():
    mol = Molecule(
        atom="H 0 0 0; H 0 0 1.6; H 0 0 3.2; H 0 0 4.8",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(driver="gbasis")
    mf = RHF(mol).run()

    legacy = DMRG(mf, ncas=4, nelecas=4, D=16, init_guess="cid", site="spatial", verbose=0)
    legacy.run(
        nsweeps=4,
        conv_tol=None,
        symmetry_list=["charge", "su2"],
        local_solver_kwargs={"recoupled_reduced": False, "itermax": 40},
    )
    auto = DMRG(mf, ncas=4, nelecas=4, D=16, init_guess="cid", site="spatial", verbose=0)
    auto.run(
        nsweeps=4,
        conv_tol=None,
        symmetry_list=["charge", "su2"],
        local_solver_kwargs={"recoupled_reduced": "auto", "itermax": 40},
    )

    assert auto.dmrg.backend == "nonabelian"
    assert auto.e_tot == pytest.approx(legacy.e_tot, abs=1e-8)
    assert all(
        "recoupled_generalized" not in history.get("local_problem_counts", {})
        for history in auto.dmrg.history
    )


def test_spatial_su2_dmrg_supports_state_average_roots():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    mol.build(driver="gbasis")
    mf = RHF(mol).run()

    su2 = DMRG(mf, ncas=2, nelecas=2, D=8, init_guess="hf", site="spatial", verbose=0)
    su2.run(nstates=2, weights=[0.5, 0.5], nsweeps=2, symmetry_list=["charge", "su2"])

    assert su2.dmrg.backend == "nonabelian"
    assert len(su2.dmrg.states) == 2
    assert np.asarray(su2.e_tot).shape == (2,)
    assert su2.e_tot[0] < su2.e_tot[1]
    assert su2.e_tot[1] == pytest.approx(-0.169291740911, abs=1e-7)
    assert "state_energies" in su2.dmrg.history[-1]["bond_objectives"][-1]
    assert su2.dmrg.history[-1]["bond_objectives"][-1]["target_irrep_filtered"] is True


def test_spatial_su2_state_average_supports_multisite_singlet_roots():
    mol = Molecule(
        atom="H 0 0 0; H 0 0 1.6; H 0 0 3.2; H 0 0 4.8",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(driver="gbasis")
    mf = RHF(mol).run()

    dmrg = DMRG(mf, ncas=4, nelecas=4, D=16, init_guess="cid", site="spatial", verbose=0)
    dmrg.run(nstates=2, weights=[0.5, 0.5], nsweeps=2, symmetry_list=["charge", "su2"])

    assert len(dmrg.dmrg.states) == 2
    np.testing.assert_allclose(
        dmrg.e_tot,
        [-2.177899321294, -1.557192705150],
        atol=1e-8,
    )
    np.testing.assert_allclose(dmrg.dmrg.history[-1]["state_s2"], [0.0, 0.0], atol=1e-8)


def test_two_site_abelian_state_average_returns_root_mps_lists():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    mol.build(driver="gbasis")
    mf = RHF(mol).run()

    dmrg = DMRG(mf, ncas=2, nelecas=2, D=8, init_guess="hf", site="spatial", verbose=0)
    dmrg.run(nstates=2, weights=[0.5, 0.5], nsweeps=2, symmetry_list=["charge", "sz"])

    assert len(dmrg.dmrg.states) == 2
    assert np.asarray(dmrg.e_tot).shape == (2,)
    assert dmrg.dmrg.states[0].L == 2
    assert dmrg.dmrg.states[1].L == 2


def test_dmrg_fix_spin_accepts_non_singlet_targets_and_warns_for_linear_penalty():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    mol.build(driver="gbasis")
    mf = RHF(mol).run()
    dmrg = DMRG(mf, ncas=2, nelecas=2, D=4, init_guess="hf")

    with pytest.warns(RuntimeWarning, match="linear \\+shift\\*S\\^2 penalty"):
        dmrg.fix_spin(ss=2, shift=0.3)

    assert dmrg.spin_purification is True
    assert dmrg.ss == pytest.approx(2.0)
    assert dmrg.shift == pytest.approx(0.3)
