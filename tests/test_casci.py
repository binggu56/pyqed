import numpy as np
import pytest
from itertools import combinations
from types import SimpleNamespace

from pyqed.qchem import Molecule, bo_hamiltonian_derivatives
from pyqed.qchem.geometric import (
    BOHamiltonianDerivatives,
    GeometricFGTerms,
    _build_cbasis_from_reference,
    _contract_ao_operator_with_state_model,
)
from pyqed.qchem.hf import RHF, UHF
from pyqed.qchem.mcscf.casci import (
    CASCI,
    _biorthogonal_ci_overlap_candidate,
    _factorized_ci_overlap,
    _occupation_lists,
    _overlap_slow_from_mo_overlap,
    _reconstruct_string_overlap_from_svd,
    _string_overlap_matrix,
    _string_singular_weights,
    _string_transform_matrix,
    transform_spatial_eri_to_mo,
)
from pyqed.qchem.mcscf import direct_ci


def test_casci_accepts_closed_shell_uhf_reference():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='gbasis')

    rhf = RHF(mol).run()
    uhf = UHF(mol).run()

    mc_rhf = CASCI(rhf, ncas=2, nelecas=2).run(nstates=1)
    mc_uhf = CASCI(uhf, ncas=2, nelecas=2).run(nstates=1)

    np.testing.assert_allclose(mc_uhf.e_tot[0], mc_rhf.e_tot[0], atol=1e-8)
    np.testing.assert_allclose(
        mc_uhf.make_rdm1(0),
        mc_rhf.make_rdm1(0),
        atol=1e-8,
    )


def test_casci_is_exported_from_pyqed_qchem():
    from pyqed.qchem import CASCI as ExportedCASCI

    assert ExportedCASCI is CASCI


def test_casci_accepts_open_shell_uhf_reference():
    mol = Molecule(atom='H 0 0 0', unit='bohr', basis='sto-3g', spin=1)
    mol.build(driver='gbasis')

    uhf = UHF(mol).run()
    mc = CASCI(uhf, ncas=1, nelecas=(1, 0)).run(nstates=1)

    np.testing.assert_allclose(mc.e_tot[0], uhf.e_tot, atol=1e-10)
    np.testing.assert_allclose(mc.make_rdm1(0), np.array([[1.0]]), atol=1e-10)
    assert str(mc.solver_backend).startswith('direct_ci')


def test_casci_make_rdm1s_matches_pyscf_for_open_shell_li():
    pyscf = pytest.importorskip('pyscf')
    mcscf = pytest.importorskip('pyscf.mcscf')

    mol = Molecule(atom='Li 0 0 0', unit='bohr', basis='sto-3g', spin=1)
    mol.build(driver='gbasis')

    mf = UHF(mol).run()
    mc = CASCI(mf, ncas=2, nelecas=(2, 1)).run(nstates=1)

    pmf = pyscf.scf.UHF(mol.topyscf())
    pmf.conv_tol = 1e-10
    pmf.kernel()
    pmc = mcscf.UCASCI(pmf, 2, (2, 1))
    pmc.kernel()

    dm1a, dm1b = mc.make_rdm1s(0)
    pdm1a, pdm1b = pmc.fcisolver.make_rdm1s(pmc.ci, 2, (2, 1))

    np.testing.assert_allclose(dm1a, np.array(pdm1a), atol=1e-8)
    np.testing.assert_allclose(dm1b, np.array(pdm1b), atol=1e-8)
    np.testing.assert_allclose(dm1a + dm1b, mc.make_rdm1(0), atol=1e-8)


def test_direct_ci_casci_accepts_open_shell_uhf_reference():
    mol = Molecule(atom='Li 0 0 0', unit='bohr', basis='sto-3g', spin=1)
    mol.build(driver='gbasis')

    mf = UHF(mol).run()
    mc = direct_ci.CASCI(mf, ncas=2, nelecas=(2, 1)).run(nstates=1, method='direct_ci')

    dm1a, dm1b = mc.make_rdm1s(0)
    np.testing.assert_allclose(dm1a + dm1b, mc.make_rdm1(0), atol=1e-8)
    assert np.isfinite(mc.e_tot[0])
    assert str(mc.solver_backend).startswith('direct_ci')


def test_direct_ci_casci_uhf_supports_factor_backend():
    mol = Molecule(atom='Li 0 0 0', unit='bohr', basis='sto-3g', spin=1)
    mol.build(driver='gbasis')

    mf = UHF(mol).run()
    mc_dense = direct_ci.CASCI(mf, ncas=2, nelecas=(2, 1)).run(nstates=1, method='direct_ci')
    mc_factor = direct_ci.CASCI(mf, ncas=2, nelecas=(2, 1)).run(
        nstates=1,
        method='direct_ci',
        use_cholesky=True,
    )

    np.testing.assert_allclose(mc_factor.e_tot[0], mc_dense.e_tot[0], atol=1e-8)
    assert str(mc_factor.solver_backend).startswith('direct_ci_factor_conn')


def test_casci_make_tdm1s_sum_to_spin_traced_tdm():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = RHF(mol).run()
    mc = CASCI(mf, ncas=2, nelecas=2).run(nstates=2)

    tdm1a, tdm1b = mc.make_tdm1s(1, 0)

    np.testing.assert_allclose(tdm1a + tdm1b, mc.make_tdm1(1, 0), atol=1e-10)
    np.testing.assert_allclose(sum(mc.make_rdm1s(0)), mc.make_rdm1(0), atol=1e-10)


def test_bo_hamiltonian_derivatives_match_manual_tdm_contractions():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = RHF(mol).run()
    mc = CASCI(mf, ncas=2, nelecas=2).run(nstates=2)

    terms = bo_hamiltonian_derivatives(mc, state_ids=[0, 1])

    cart = 2
    manual_offdiag_f = _contract_ao_operator_with_state_model(mc, 1, 0, terms.h1_ao_cartesian[cart])
    manual_diag_f = (
        _contract_ao_operator_with_state_model(mc, 0, 0, terms.h1_ao_cartesian[cart])
        + terms.vnn_gradient_cartesian[cart]
    )

    np.testing.assert_allclose(terms.F_cartesian[cart, 1, 0], manual_offdiag_f, atol=1e-10)
    np.testing.assert_allclose(terms.F_cartesian[cart, 0, 0], manual_diag_f, atol=1e-10)

    manual_offdiag_g = _contract_ao_operator_with_state_model(
        mc,
        1,
        0,
        terms.h2_ao_cartesian[cart, cart],
    )
    manual_diag_g = (
        _contract_ao_operator_with_state_model(mc, 0, 0, terms.h2_ao_cartesian[cart, cart])
        + terms.vnn_hessian_cartesian[cart, cart]
    )

    np.testing.assert_allclose(terms.G_cartesian[cart, cart, 1, 0], manual_offdiag_g, atol=1e-10)
    np.testing.assert_allclose(terms.G_cartesian[cart, cart, 0, 0], manual_diag_g, atol=1e-10)


def test_bo_hamiltonian_derivatives_projection_matches_cartesian_contraction():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = RHF(mol).run()
    mc = CASCI(mf, ncas=2, nelecas=2).run(nstates=2)

    mode = np.zeros((1, mol.natom, 3))
    mode[0, 0, 2] = -1.0
    mode[0, 1, 2] = 1.0

    terms = bo_hamiltonian_derivatives(mc, state_ids=[0, 1], mode_vectors=mode)
    mode_flat = mode.reshape(1, -1)

    expected_f = np.einsum('ka,aij->kij', mode_flat, terms.F_cartesian, optimize=True)
    expected_g = np.einsum(
        'ka,lb,abij->klij',
        mode_flat,
        mode_flat,
        terms.G_cartesian,
        optimize=True,
    )

    np.testing.assert_allclose(terms.F_projected, expected_f, atol=1e-10)
    np.testing.assert_allclose(terms.G_projected, expected_g, atol=1e-10)


def test_bo_hamiltonian_derivatives_match_fixed_basis_finite_difference():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = RHF(mol).run()
    mc = CASCI(mf, ncas=2, nelecas=2).run(nstates=2)
    terms = bo_hamiltonian_derivatives(mc, state_ids=[0, 1])

    cbas = _build_cbasis_from_reference(mol)
    coords = np.asarray(mol.atom_coords(), dtype=float)
    charges = np.asarray(mol.atom_charges(), dtype=float)

    def electron_nuclear_operator(test_coords):
        v = np.zeros((cbas.nbfn, cbas.nbfn), dtype=float)
        for atom_id, charge in enumerate(charges):
            v -= charge * cbas.int1e(
                'int1e_rinv',
                inv_origin=np.asarray(test_coords[atom_id], dtype=float),
            )
        return v

    atom_id = 1
    axis = 2
    cart = 3 * atom_id + axis
    delta = 1e-4

    coords_plus = coords.copy()
    coords_minus = coords.copy()
    coords_plus[atom_id, axis] += delta
    coords_minus[atom_id, axis] -= delta

    v0 = electron_nuclear_operator(coords)
    v_plus = electron_nuclear_operator(coords_plus)
    v_minus = electron_nuclear_operator(coords_minus)

    fd_first = (v_plus - v_minus) / (2.0 * delta)
    fd_second = (v_plus - 2.0 * v0 + v_minus) / (delta ** 2)

    np.testing.assert_allclose(
        terms.h1_ao_cartesian[cart].real,
        fd_first,
        atol=1e-6,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        terms.h2_ao_cartesian[cart, cart].real,
        fd_second,
        atol=2e-4,
        rtol=2e-4,
    )
    assert np.isfinite(terms.F_cartesian).all()
    assert np.isfinite(terms.G_cartesian).all()


def test_bo_hamiltonian_derivatives_work_with_builtin_factor_only_reference():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='builtin', options={'eri_representation': 'factors'})

    mf = RHF(mol).run()
    mc = CASCI(mf, ncas=2, nelecas=2).run(nstates=2)
    terms = bo_hamiltonian_derivatives(mc, state_ids=[0, 1])

    assert terms.h1_ao_cartesian.shape == (3 * mol.natom, mol.nao, mol.nao)
    assert terms.h2_ao_cartesian.shape == (3 * mol.natom, 3 * mol.natom, mol.nao, mol.nao)
    assert np.isfinite(terms.F_cartesian).all()
    assert np.isfinite(terms.G_cartesian).all()


def test_bo_hamiltonian_derivatives_backward_aliases_remain_available():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = RHF(mol).run()
    mc = CASCI(mf, ncas=2, nelecas=2).run(nstates=2)

    terms_new = bo_hamiltonian_derivatives(mc, state_ids=[0, 1])
    from pyqed.qchem.geometric import build_casci_geometric_fg_terms
    terms_old = build_casci_geometric_fg_terms(mc, state_ids=[0, 1])

    assert isinstance(terms_new, BOHamiltonianDerivatives)
    assert isinstance(terms_old, GeometricFGTerms)
    np.testing.assert_allclose(terms_new.F_cartesian, terms_old.F_cartesian, atol=1e-10)
    np.testing.assert_allclose(terms_new.G_cartesian, terms_old.G_cartesian, atol=1e-10)


def test_public_casci_defaults_to_direct_ci():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = RHF(mol).run()
    mc = CASCI(mf, ncas=4, nelecas=4).run(nstates=2)

    assert mc.solver_backend in {'ci_dense_fallback', 'direct_ci_compact_conn', 'direct_ci_factor_conn'}


def test_direct_ci_falls_back_to_dense_solver_for_small_spaces():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = RHF(mol).run()

    mc_direct = direct_ci.CASCI(mf, ncas=4, nelecas=4).run(nstates=2, method='direct_ci')
    mc_dense = CASCI(mf, ncas=4, nelecas=4).run(nstates=2, method='ci')

    assert mc_direct.solver_backend == 'ci_dense_fallback'
    np.testing.assert_allclose(mc_direct.e_tot, mc_dense.e_tot, atol=1e-10)


def test_direct_ci_on_the_fly_matches_dense_solver():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = RHF(mol).run()

    mc_direct = direct_ci.CASCI(mf, ncas=4, nelecas=4)
    mc_direct.direct_ci_dense_fallback_ndets = 1
    mc_direct.run(nstates=2, method='direct_ci')

    mc_dense = CASCI(mf, ncas=4, nelecas=4).run(nstates=2, method='ci')

    assert mc_direct.solver_backend == 'direct_ci_compact_conn'
    np.testing.assert_allclose(mc_direct.e_tot, mc_dense.e_tot, atol=1e-10)


def test_direct_ci_davidson_matches_eigsh():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = RHF(mol).run()

    mc_davidson = direct_ci.CASCI(mf, ncas=4, nelecas=4)
    mc_davidson.direct_ci_dense_fallback_ndets = 1
    mc_davidson.direct_ci_eigensolver = 'davidson'
    mc_davidson.run(nstates=2, method='direct_ci')

    mc_eigsh = direct_ci.CASCI(mf, ncas=4, nelecas=4)
    mc_eigsh.direct_ci_dense_fallback_ndets = 1
    mc_eigsh.direct_ci_eigensolver = 'eigsh'
    mc_eigsh.run(nstates=2, method='direct_ci')

    np.testing.assert_allclose(mc_davidson.e_tot, mc_eigsh.e_tot, atol=1e-10)


def test_direct_ci_defaults_to_davidson():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = RHF(mol).run()
    mc = direct_ci.CASCI(mf, ncas=4, nelecas=4)

    assert mc.direct_ci_eigensolver == 'davidson'
    assert mc.direct_ci_root_cushion == 2


def test_direct_ci_auto_eigensolver_uses_eigsh_for_medium_spaces():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = RHF(mol).run()

    mc_auto = direct_ci.CASCI(mf, ncas=4, nelecas=4)
    mc_auto.direct_ci_dense_fallback_ndets = 1
    mc_auto.direct_ci_eigensolver = 'auto'
    mc_auto.direct_ci_auto_eigsh_ndets = 1000
    mc_auto.run(nstates=2, method='direct_ci')

    mc_eigsh = direct_ci.CASCI(mf, ncas=4, nelecas=4)
    mc_eigsh.direct_ci_dense_fallback_ndets = 1
    mc_eigsh.direct_ci_eigensolver = 'eigsh'
    mc_eigsh.run(nstates=2, method='direct_ci')

    assert mc_auto.direct_ci_eigensolver == 'auto'
    np.testing.assert_allclose(mc_auto.e_tot, mc_eigsh.e_tot, atol=1e-10)


def test_direct_ci_root_cushion_recovers_lowest_h8_roots():
    distance = 1.8
    atom = '\n'.join(f'H 0 0 {i * distance:.10f}' for i in range(8))

    mol = Molecule(atom=atom, unit='bohr', basis='sto-6g', spin=0)
    mol.build(driver='gbasis')

    mf = RHF(mol).run()

    mc_cushion = direct_ci.CASCI(mf, ncas=8, nelecas=8)
    mc_cushion.direct_ci_dense_fallback_ndets = 1
    mc_cushion.direct_ci_eigensolver = 'davidson'
    mc_cushion.direct_ci_root_cushion = 2
    mc_cushion.run(nstates=10, method='direct_ci')

    mc_reference = direct_ci.CASCI(mf, ncas=8, nelecas=8)
    mc_reference.direct_ci_dense_fallback_ndets = 1
    mc_reference.direct_ci_eigensolver = 'davidson'
    mc_reference.direct_ci_root_cushion = 0
    mc_reference.run(nstates=12, method='direct_ci')

    np.testing.assert_allclose(mc_cushion.e_tot, mc_reference.e_tot[:10], atol=1e-10)


def test_direct_ci_spin_square_matches_rdm_definition():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = RHF(mol).run()

    mc_direct = direct_ci.CASCI(mf, ncas=2, nelecas=2)
    mc_direct.direct_ci_dense_fallback_ndets = 1
    mc_direct.run(nstates=2, method='direct_ci')

    for root in range(2):
        dm1, dm2 = mc_direct.make_rdm12(root)
        np.testing.assert_allclose(
            mc_direct.spin_square(root),
            direct_ci.spin_square_from_rdm(dm1, dm2),
            atol=1e-10,
        )


def test_cholesky_active_space_eri_transform_matches_dense():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis-pyscf')

    mf = RHF(mol).run(cholesky_jk=True, cholesky_tol=1e-10)
    mo_cas = mf.mo_coeff[:, 1:5]

    eri_dense = transform_spatial_eri_to_mo(mf, mo_cas, use_cholesky=False)
    eri_cd = transform_spatial_eri_to_mo(mf, mo_cas, use_cholesky=True)

    np.testing.assert_allclose(eri_cd, eri_dense, atol=1e-8)


def test_casci_use_cholesky_matches_dense_energy():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis-pyscf')

    mf = RHF(mol).run(cholesky_jk=True, cholesky_tol=1e-10)

    mc_dense = CASCI(mf, ncas=4, nelecas=4).run(nstates=2, method='ci')
    mc_cd = CASCI(mf, ncas=4, nelecas=4).run(nstates=2, use_cholesky=True)

    np.testing.assert_allclose(mc_cd.e_tot, mc_dense.e_tot, atol=1e-8)


def test_casci_auto_uses_cholesky_from_rhf_label():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis-pyscf')

    mf = RHF(mol).run(cholesky_jk=True, cholesky_tol=1e-10)

    mc_auto = CASCI(mf, ncas=4, nelecas=4).run(nstates=2)
    mc_cd = CASCI(mf, ncas=4, nelecas=4).run(nstates=2, use_cholesky=True)

    assert mc_auto.use_cholesky_integrals
    np.testing.assert_allclose(mc_auto.e_tot, mc_cd.e_tot, atol=1e-8)


def test_direct_ci_use_cholesky_matches_dense_energy():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis-pyscf')

    mf = RHF(mol).run(cholesky_jk=True, cholesky_tol=1e-10)

    mc_direct = direct_ci.CASCI(mf, ncas=4, nelecas=4)
    mc_direct.direct_ci_dense_fallback_ndets = 1
    mc_direct.run(nstates=2, method='direct_ci', use_cholesky=True)

    mc_dense = CASCI(mf, ncas=4, nelecas=4).run(nstates=2, method='ci')

    np.testing.assert_allclose(mc_direct.e_tot, mc_dense.e_tot, atol=1e-8)


def test_public_casci_direct_ci_method_dispatches_to_direct_backend():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis-pyscf')

    mf = RHF(mol).run(cholesky_jk=True, cholesky_tol=1e-10)

    mc_public = CASCI(mf, ncas=4, nelecas=4).run(nstates=2, method='direct_ci')

    mc_direct = direct_ci.CASCI(mf, ncas=4, nelecas=4)
    mc_direct.run(nstates=2, method='direct_ci')

    assert mc_public.solver_backend == mc_direct.solver_backend
    np.testing.assert_allclose(mc_public.e_tot, mc_direct.e_tot, atol=1e-10)


def test_direct_ci_auto_uses_cholesky_from_rhf_label():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis-pyscf')

    mf = RHF(mol).run(cholesky_jk=True, cholesky_tol=1e-10)

    mc_auto = direct_ci.CASCI(mf, ncas=4, nelecas=4)
    mc_auto.direct_ci_dense_fallback_ndets = 1
    mc_auto.run(nstates=2, method='direct_ci')

    mc_cd = direct_ci.CASCI(mf, ncas=4, nelecas=4)
    mc_cd.direct_ci_dense_fallback_ndets = 1
    mc_cd.run(nstates=2, method='direct_ci', use_cholesky=True)

    assert mc_auto.use_cholesky_integrals
    np.testing.assert_allclose(mc_auto.e_tot, mc_cd.e_tot, atol=1e-8)


def _all_string_basis(norb, nelec):
    basis = np.zeros((0, norb), dtype=np.int8)
    if nelec < 0 or nelec > norb:
        return basis
    rows = []
    for occ in combinations(range(norb), nelec):
        row = np.zeros(norb, dtype=np.int8)
        row[list(occ)] = 1
        rows.append(row)
    return np.asarray(rows, dtype=np.int8)


def _dummy_ci_state(alpha_strings, beta_strings, ci):
    binary = []
    for alpha in alpha_strings:
        for beta in beta_strings:
            binary.append(np.stack((alpha, beta), axis=0))
    return SimpleNamespace(
        binary=np.asarray(binary, dtype=np.int8),
        ci=np.asarray(ci),
        ncore=1,
        ncas=alpha_strings.shape[1],
    )


def test_string_space_svd_reconstruction_matches_exact_minors():
    rng = np.random.default_rng(123)
    norb = 4
    nelec = 2

    q_left, _ = np.linalg.qr(rng.normal(size=(norb, norb)))
    q_right, _ = np.linalg.qr(rng.normal(size=(norb, norb)))
    sigma = np.array([1.7, 1.2, 0.8, 0.4], dtype=float)
    s = q_left @ np.diag(sigma) @ q_right.T

    strings = _all_string_basis(norb, nelec)
    exact = _string_overlap_matrix(
        s,
        _occupation_lists(strings),
        _occupation_lists(strings),
        s.dtype,
    )
    reconstructed = _reconstruct_string_overlap_from_svd(
        q_left,
        sigma,
        q_right.T,
        strings,
        s.dtype,
    )

    np.testing.assert_allclose(reconstructed, exact, atol=1e-10)


def test_string_space_right_factor_uses_vh_orientation():
    rng = np.random.default_rng(456)
    norb = 4
    nelec = 2

    q_left, _ = np.linalg.qr(rng.normal(size=(norb, norb)))
    q_right, _ = np.linalg.qr(rng.normal(size=(norb, norb)))
    sigma = np.array([1.9, 1.1, 0.7, 0.3], dtype=float)
    s = q_left @ np.diag(sigma) @ q_right.T

    strings = _all_string_basis(norb, nelec)
    exact = _string_overlap_matrix(
        s,
        _occupation_lists(strings),
        _occupation_lists(strings),
        s.dtype,
    )

    correct = _reconstruct_string_overlap_from_svd(
        q_left,
        sigma,
        q_right.T,
        strings,
        s.dtype,
    )

    wrong = (
        _string_transform_matrix(q_left, strings, s.dtype)
        @ np.diag(_string_singular_weights(sigma, strings))
        @ _string_transform_matrix(q_right, strings, s.dtype)
    )

    np.testing.assert_allclose(correct, exact, atol=1e-10)
    assert np.max(np.abs(wrong - exact)) > 1e-3


def test_string_space_svd_reconstruction_matches_exact_minors_complex():
    rng = np.random.default_rng(789)
    norb = 4
    nelec = 2

    left_real = rng.normal(size=(norb, norb))
    left_imag = rng.normal(size=(norb, norb))
    right_real = rng.normal(size=(norb, norb))
    right_imag = rng.normal(size=(norb, norb))
    q_left, _ = np.linalg.qr(left_real + 1j * left_imag)
    q_right, _ = np.linalg.qr(right_real + 1j * right_imag)
    sigma = np.array([1.6, 1.0, 0.75, 0.35], dtype=float)
    s = q_left @ np.diag(sigma) @ q_right.conj().T

    strings = _all_string_basis(norb, nelec)
    exact = _string_overlap_matrix(
        s,
        _occupation_lists(strings),
        _occupation_lists(strings),
        s.dtype,
    )
    reconstructed = _reconstruct_string_overlap_from_svd(
        q_left,
        sigma,
        q_right.conj().T,
        strings,
        s.dtype,
    )

    np.testing.assert_allclose(reconstructed, exact, atol=1e-10)


def test_factorized_overlap_ignores_external_mo_block():
    alpha = _all_string_basis(2, 1)
    beta = _all_string_basis(2, 1)
    ci = np.array([0.6, -0.2, 0.3, 0.7], dtype=float)
    cibra = _dummy_ci_state(alpha, beta, ci)
    ciket = _dummy_ci_state(alpha, beta, ci[::-1])

    base = np.array(
        [
            [1.2, 0.1, -0.2, 0.4, -0.1],
            [0.3, 0.8, 0.2, 0.5, 0.7],
            [-0.4, 0.1, 1.1, -0.3, 0.6],
            [0.2, -0.6, 0.9, 1.4, -0.8],
            [0.5, 0.3, -0.7, 0.2, 0.9],
        ],
        dtype=float,
    )
    modified = base.copy()
    modified[:, 3:] = np.array(
        [
            [5.0, -4.0],
            [-3.5, 2.5],
            [7.2, -6.1],
            [0.4, 1.7],
            [-2.2, 3.3],
        ]
    )
    modified[3:, :] = np.array(
        [
            [4.1, -5.2, 6.3, 0.4, 1.7],
            [-1.9, 2.8, -3.7, -2.2, 3.3],
        ]
    )

    exact_base = _factorized_ci_overlap(cibra, ciket, s=base)
    exact_modified = _factorized_ci_overlap(cibra, ciket, s=modified)

    np.testing.assert_allclose(exact_modified, exact_base, atol=1e-12)


def test_biorthogonal_ci_overlap_candidate_matches_exact_overlap():
    alpha = _all_string_basis(2, 1)
    beta = _all_string_basis(2, 1)
    cibra = _dummy_ci_state(alpha, beta, np.array([0.45, -0.3, 0.25, 0.8], dtype=float))
    ciket = _dummy_ci_state(alpha, beta, np.array([-0.1, 0.7, 0.55, -0.4], dtype=float))

    s = np.array(
        [
            [1.15, 0.12, -0.08, 0.5, -0.3],
            [0.18, 0.95, 0.14, -0.6, 0.2],
            [-0.11, 0.21, 1.08, 0.7, -0.5],
            [0.4, -0.3, 0.2, 1.0, 0.0],
            [-0.2, 0.1, -0.4, 0.0, 1.0],
        ],
        dtype=float,
    )

    exact = _factorized_ci_overlap(cibra, ciket, s=s)
    candidate = _biorthogonal_ci_overlap_candidate(cibra, ciket, s=s)

    np.testing.assert_allclose(candidate, exact, atol=1e-10)


def test_public_overlap_matches_slow_reference_for_displaced_casci_pair():
    mol1 = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol1.build(driver='gbasis')
    mf1 = RHF(mol1).run()
    mc1 = CASCI(mf1, ncas=2, nelecas=2).run(nstates=2)

    mol2 = Molecule(atom='H 0 0 0; H 0 0 1.5', unit='bohr', basis='sto-3g')
    mol2.build(driver='gbasis')
    mf2 = RHF(mol2).run()
    mc2 = CASCI(mf2, ncas=2, nelecas=2).run(nstates=2)

    exact = _overlap_slow_from_mo_overlap(mc1, mc2, s=None)
    fast = mc1.overlap(mc2)

    np.testing.assert_allclose(fast, exact, atol=1e-10)
