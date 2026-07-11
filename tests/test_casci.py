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
from pyqed.qchem.mcscf.orbopt import (
    embed_rdm2,
    generalized_fock,
    orbital_gradient,
    pack_nonredundant,
)
from pyqed.qchem.mcscf.reduced_ci import (
    ReducedCISubspace,
    ci_rotation_gradient,
    ci_rotation_hessian,
    orbital_ci_coupling,
)


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


def test_casci_ms2_multiplicity_selects_triplet_m0_root():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = RHF(mol).run()
    mc_singlet = CASCI(mf, ncas=2, nelecas=2, ms2=0, multiplicity=1).run(
        nstates=1, method='direct_ci'
    )
    mc_triplet_m0 = CASCI(mf, ncas=2, nelecas=2, ms2=0, multiplicity=3).run(
        nstates=1, method='direct_ci'
    )
    mc_triplet_p1 = CASCI(mf, ncas=2, nelecas=2, ms2=2, multiplicity=3).run(
        nstates=1, method='direct_ci'
    )

    np.testing.assert_allclose(mc_singlet.spin_square(0), 0.0, atol=1e-10)
    np.testing.assert_allclose(mc_triplet_m0.spin_square(0), 2.0, atol=1e-10)
    np.testing.assert_allclose(mc_triplet_p1.spin_square(0), 2.0, atol=1e-10)
    np.testing.assert_allclose(mc_triplet_m0.e_tot[0], mc_triplet_p1.e_tot[0], atol=1e-10)
    assert mc_triplet_m0.ms2 == 0
    assert mc_triplet_m0.spin == 0
    assert mc_triplet_m0.multiplicity == 3


def test_casci_spin_alias_and_multiplicity_validation():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='gbasis')
    mf = RHF(mol).run()

    mc = CASCI(mf, ncas=2, nelecas=2, spin=2)
    assert mc.ms2 == 2
    assert mc.spin == 2
    assert mc.nelecas_spin == (2, 0)

    with pytest.raises(ValueError, match='spin and ms2'):
        CASCI(mf, ncas=2, nelecas=2, spin=2, ms2=0)

    with pytest.raises(ValueError, match='incompatible'):
        CASCI(mf, ncas=2, nelecas=2, ms2=2, multiplicity=1)


def test_casci_make_rdm1_ao_representation_matches_manual_transform():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = RHF(mol).run()
    mc = CASCI(mf, ncas=2, nelecas=2).run(nstates=1)

    dm_mo = mc.make_rdm1(0, with_core=True, with_vir=True, representation='mo')
    dm_mo_no_vir = mc.make_rdm1(0, with_core=True, representation='mo')
    dm_ao = mc.make_rdm1(0, with_core=True, with_vir=True, representation='ao')
    dm_ao_no_vir = mc.make_rdm1(0, with_core=True, representation='ao')
    dm_ao_alias = mc.make_rdm1(0, with_core=True, with_vir=True, repr='ao')

    expected = mf.mo_coeff @ dm_mo @ mf.mo_coeff.conj().T
    expected_no_vir = mf.mo_coeff[:, :dm_mo_no_vir.shape[0]] @ dm_mo_no_vir @ mf.mo_coeff[:, :dm_mo_no_vir.shape[0]].conj().T
    np.testing.assert_allclose(dm_ao, expected, atol=1e-12)
    np.testing.assert_allclose(dm_ao_no_vir, expected_no_vir, atol=1e-12)
    np.testing.assert_allclose(dm_ao_no_vir, expected, atol=1e-12)
    np.testing.assert_allclose(dm_ao_alias, expected, atol=1e-12)


def test_casci_make_tdm1_ao_representation_matches_manual_transform():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = RHF(mol).run()
    mc = CASCI(mf, ncas=2, nelecas=2).run(nstates=2)

    tdm_active = mc.make_tdm1(0, 1)
    tdm_ao = mc.make_tdm1(0, 1, representation='ao')
    tdm_ao_with_core = mc.make_tdm1(0, 1, with_core=True, representation='ao')
    tdm_ao_alias = mc.make_tdm1(0, 1, repr='ao')

    active_coeff = mf.mo_coeff[:, mc.ncore:mc.ncore + mc.ncas]
    expected = active_coeff @ tdm_active @ active_coeff.conj().T
    np.testing.assert_allclose(tdm_ao, expected, atol=1e-12)
    np.testing.assert_allclose(tdm_ao_with_core, expected, atol=1e-12)
    np.testing.assert_allclose(tdm_ao_alias, expected, atol=1e-12)


def test_casci_transition_dipole_moment_matches_ao_tdm_contraction():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = RHF(mol).run()
    center = np.zeros(3)
    dipole_ao = mf.dipole(center=center, basis='ao')

    dense = CASCI(mf, ncas=4, nelecas=2).run(nstates=4, method='ci')
    direct = CASCI(mf, ncas=4, nelecas=2).run(nstates=4, method='direct_ci')
    raw_direct = direct_ci.CASCI(mf, ncas=4, nelecas=2).run(nstates=4, method='direct_ci')

    tdm_ao = dense.make_tdm1(2, 0, representation='ao')
    expected = np.einsum('xij,ij->x', dipole_ao, tdm_ao, optimize=True)

    np.testing.assert_allclose(
        dense.transition_dipole_moment(2, 0, center=center),
        expected,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.abs(direct.transition_dipole_moment(2, 0, center=center)),
        np.abs(expected),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.abs(raw_direct.transition_dipole_moment(2, 0, center=center)),
        np.abs(expected),
        atol=1e-10,
    )

    all_dipoles = dense.transition_dipole_moment(center=center)
    assert all_dipoles.shape == (3, 3)
    np.testing.assert_allclose(all_dipoles[1], expected, atol=1e-10)
    np.testing.assert_allclose(dense.transition_dipole(center=center), all_dipoles)


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


def test_casci_make_tdm2_diagonal_matches_rdm2():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = RHF(mol).run()
    mc = CASCI(mf, ncas=2, nelecas=2).run(nstates=2)
    mc_direct = direct_ci.CASCI(mf, ncas=2, nelecas=2).run(
        nstates=2,
        method='direct_ci',
    )

    np.testing.assert_allclose(mc.make_tdm2(0, 0), mc.make_rdm2(0), atol=1e-10)
    np.testing.assert_allclose(mc.make_tdm2(1, 1), mc.make_rdm2(1), atol=1e-10)
    np.testing.assert_allclose(
        mc_direct.make_tdm2(0, 0),
        mc_direct.make_rdm2(0),
        atol=1e-10,
    )


def test_casci_rdm2_reconstructs_active_space_energy():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = RHF(mol).run()

    for cls, method in ((CASCI, 'ci'), (direct_ci.CASCI, 'direct_ci')):
        mc = cls(mf, ncas=4, nelecas=4).run(nstates=1, method=method)
        dm1, dm2 = mc.make_rdm12(0)
        hcore = np.asarray(mc.hcore)
        h1e = hcore[0] if hcore.ndim == 3 else hcore
        eri = getattr(mc, 'h2e_cas', None)
        if eri is None:
            ncore, ncas = mc.ncore, mc.ncas
            active = slice(ncore, ncore + ncas)
            eri = mf.get_eri_mo(mf.mo_coeff, notation='chem')[
                active, active, active, active
            ]
        eri = np.asarray(eri)
        eri = eri[0, 0] if eri.ndim == 6 else eri

        e_reconstructed = (
            mc.e_core
            + np.einsum('pq,pq', h1e, dm1)
            + 0.5 * np.einsum('pqrs,pqrs', eri, dm2)
        )

        np.testing.assert_allclose(e_reconstructed, mc.e_tot[0], atol=1e-10)


def test_reduced_ci_subspace_projects_casci_roots():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = RHF(mol).run()
    mc = direct_ci.CASCI(mf, ncas=2, nelecas=2).run(nstates=2, method='direct_ci')

    subspace = ReducedCISubspace.from_casci(mc, root_ids=[0, 1])
    energies, ci = subspace.diagonalize(nroots=2)
    residuals = subspace.residuals(mc, ci, energies)

    assert subspace.nvec == 2
    np.testing.assert_allclose(energies, mc.e_tot[:2], atol=1e-10)
    np.testing.assert_allclose(subspace.basis.T @ subspace.basis, np.eye(2), atol=1e-10)
    np.testing.assert_allclose(residuals, np.zeros_like(residuals), atol=1e-8)


def test_direct_ci_sigma_matches_ci_roots():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = RHF(mol).run()
    mc = direct_ci.CASCI(mf, ncas=2, nelecas=2).run(nstates=2, method='direct_ci')

    for root, energy in zip(mc.ci, mc.e_tot):
        sigma = mc.ci_sigma(root)
        np.testing.assert_allclose(sigma, (energy - mc.e_core) * root, atol=1e-8)
    hdiag = mc.ci_diagonal()
    assert hdiag.shape == (len(mc.ci[0]),)
    assert np.all(np.isfinite(hdiag))


def test_factorized_direct_ci_sigma_matches_ci_root():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = RHF(mol).run()
    mc = direct_ci.CASCI(mf, ncas=2, nelecas=2).run(
        nstates=1,
        method='direct_ci',
        use_cholesky=True,
    )

    sigma = mc.ci_sigma(mc.ci[0])
    np.testing.assert_allclose(sigma, (mc.e_tot[0] - mc.e_core) * mc.ci[0], atol=1e-8)


def test_reduced_ci_full_subspace_reproduces_dense_casci():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = RHF(mol).run()
    mc = direct_ci.CASCI(mf, ncas=2, nelecas=2).run(nstates=2, method='direct_ci')

    identity = np.eye(len(mc.ci[0]))
    subspace = ReducedCISubspace.from_casci(
        mc,
        root_ids=[],
        extra_vectors=identity,
    )
    energies, ci = subspace.diagonalize(nroots=2)
    residuals = subspace.residuals(mc, ci, energies)

    assert subspace.nvec == identity.shape[0]
    np.testing.assert_allclose(energies, mc.e_tot[:2], atol=1e-10)
    np.testing.assert_allclose(residuals, np.zeros_like(residuals), atol=1e-8)


def test_reduced_ci_rotation_block_for_root_plus_external_vector():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = RHF(mol).run()
    mc = direct_ci.CASCI(mf, ncas=2, nelecas=2).run(nstates=1, method='direct_ci')

    extra = np.eye(len(mc.ci[0]))[:, 0]
    subspace = ReducedCISubspace.from_casci(mc, root_ids=[0], extra_vectors=extra)
    grad, grad_pairs = subspace.rotation_gradient(nstates=1)
    hess, hess_pairs = subspace.rotation_hessian(nstates=1)
    grad_func, grad_pairs_func = ci_rotation_gradient(subspace.hamiltonian, nstates=1)
    hess_func, hess_pairs_func = ci_rotation_hessian(subspace.hamiltonian, nstates=1)

    assert grad_pairs == hess_pairs
    assert grad_pairs == grad_pairs_func
    assert hess_pairs == hess_pairs_func
    assert grad.shape == (subspace.nvec - 1,)
    assert hess.shape == (subspace.nvec - 1, subspace.nvec - 1)
    np.testing.assert_allclose(grad, grad_func, atol=1e-12)
    np.testing.assert_allclose(hess, hess_func, atol=1e-12)
    np.testing.assert_allclose(grad, np.zeros_like(grad), atol=1e-8)
    np.testing.assert_allclose(hess, hess.T, atol=1e-12)

    rotated = subspace.rotated_state_vectors(np.zeros(len(grad_pairs)), grad_pairs)
    np.testing.assert_allclose(rotated[:, 0], subspace.basis[:, 0], atol=1e-12)

    step = np.zeros(len(grad_pairs))
    step[0] = 0.1
    rotated = subspace.rotated_state_vectors(step, grad_pairs)
    assert rotated.shape == (subspace.ndet, 1)
    np.testing.assert_allclose(rotated.T @ rotated, np.eye(1), atol=1e-12)


def test_reduced_ci_expands_with_preconditioned_residual():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = RHF(mol).run()
    mc = direct_ci.CASCI(mf, ncas=4, nelecas=4).run(nstates=1, method='direct_ci')

    subspace = ReducedCISubspace.from_casci(mc, root_ids=[0])
    trial = mc.ci[0].copy()
    trial += 0.1 * np.eye(len(trial))[:, 0]
    trial /= np.linalg.norm(trial)
    energy = subspace.rayleigh_energies(mc, trial)[0]
    expanded, n_added = subspace.expand_with_residuals(
        mc,
        trial,
        np.array([energy]),
        max_vectors=1,
    )

    assert n_added == 1
    assert expanded.nvec == subspace.nvec + 1
    np.testing.assert_allclose(expanded.basis.T @ expanded.basis, np.eye(expanded.nvec), atol=1e-10)


def test_orbital_ci_coupling_matches_ci_finite_difference():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = RHF(mol).run()
    mc = direct_ci.CASCI(mf, ncas=2, nelecas=2).run(nstates=1, method='direct_ci')

    extra = np.eye(len(mc.ci[0]))[:, 0]
    subspace = ReducedCISubspace.from_casci(mc, root_ids=[0], extra_vectors=extra)
    coupling, pairs = subspace.orbital_coupling(
        mc,
        mf.get_hcore_mo(mf.mo_coeff),
        mf.get_eri_mo(mf.mo_coeff, notation='chem'),
        nstates=1,
        nmo=mf.nmo,
    )
    coupling_func, pairs_func = orbital_ci_coupling(
        mc,
        subspace,
        mf.get_hcore_mo(mf.mo_coeff),
        mf.get_eri_mo(mf.mo_coeff, notation='chem'),
        nstates=1,
        nmo=mf.nmo,
    )
    assert coupling.shape[1] == len(pairs)
    assert pairs == pairs_func
    np.testing.assert_allclose(coupling, coupling_func, atol=1e-12)

    p, m = pairs[0]
    eps = 1.0e-5
    c0 = subspace.basis[:, m]
    cp = subspace.basis[:, p]
    c_plus = c0 + eps * cp
    c_plus /= np.linalg.norm(c_plus)
    c_minus = c0 - eps * cp
    c_minus /= np.linalg.norm(c_minus)
    ci_saved = mc.ci
    try:
        mc.ci = [c_plus]
        dm1_plus = mc.make_rdm1(0, with_core=True, with_vir=True)
        dm2_plus = embed_rdm2(mc.make_rdm2(0, with_core=True), mf.nmo)
        grad_plus = orbital_gradient(
            generalized_fock(
                mf.get_hcore_mo(mf.mo_coeff),
                mf.get_eri_mo(mf.mo_coeff, notation='chem'),
                dm1_plus,
                dm2_plus,
            )
        )

        mc.ci = [c_minus]
        dm1_minus = mc.make_rdm1(0, with_core=True, with_vir=True)
        dm2_minus = embed_rdm2(mc.make_rdm2(0, with_core=True), mf.nmo)
        grad_minus = orbital_gradient(
            generalized_fock(
                mf.get_hcore_mo(mf.mo_coeff),
                mf.get_eri_mo(mf.mo_coeff, notation='chem'),
                dm1_minus,
                dm2_minus,
            )
        )
    finally:
        mc.ci = ci_saved

    fd_col = pack_nonredundant(
        (grad_plus - grad_minus) / (2.0 * eps),
        mc.ncore,
        mc.ncas,
        mf.nmo,
    )
    np.testing.assert_allclose(coupling[:, 0], fd_col, atol=1e-5, rtol=1e-5)


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


def test_casci_vibronic_couplings_return_projected_f_and_g():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = RHF(mol).run()
    mc = CASCI(mf, ncas=2, nelecas=2).run(nstates=2)

    mode = np.zeros((1, mol.natom, 3))
    mode[0, 0, 2] = -1.0
    mode[0, 1, 2] = 1.0

    f, g, terms = mc.vibronic_couplings(
        state_ids=[0, 1],
        modes=mode,
        return_terms=True,
    )

    assert f.shape == (2, 2, 1)
    assert g.shape == (2, 2, 1, 1)
    np.testing.assert_allclose(f, np.moveaxis(terms.F_projected, 0, -1), atol=1e-10)
    np.testing.assert_allclose(
        g,
        np.moveaxis(terms.G_projected, (0, 1), (-2, -1)),
        atol=1e-10,
    )


def test_casci_vibronic_couplings_modes_are_cartesian_displacement_coefficients():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = RHF(mol).run()
    mc = CASCI(mf, ncas=2, nelecas=2).run(nstates=2)

    mass_weighted = np.zeros((1, mol.natom, 3))
    mass_weighted[0, 0, 2] = -1.0
    mass_weighted[0, 1, 2] = 1.0
    masses = np.asarray(mol.atom_mass_list(), dtype=float)
    cartesian_modes = mass_weighted / np.sqrt(masses)[:, None]

    f, _ = mc.vibronic_couplings(state_ids=[0, 1], modes=cartesian_modes)
    terms = bo_hamiltonian_derivatives(mc, state_ids=[0, 1])

    expected = np.einsum(
        "mAx,Axab->abm",
        cartesian_modes,
        terms.F_cartesian.reshape(mol.natom, 3, 2, 2),
        optimize=True,
    )
    wrong_mass_weighted = np.einsum(
        "mAx,Axab->abm",
        mass_weighted,
        terms.F_cartesian.reshape(mol.natom, 3, 2, 2),
        optimize=True,
    )

    np.testing.assert_allclose(f, expected, atol=1e-10)
    assert not np.allclose(f, wrong_mass_weighted, atol=1e-8)


def test_casci_vibronic_couplings_return_cartesian_f_and_g_without_modes():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = RHF(mol).run()
    mc = CASCI(mf, ncas=2, nelecas=2).run(nstates=2)

    f, g, terms = mc.vibronic_couplings(state_ids=[0, 1], return_terms=True)

    assert f.shape == (2, 2, mol.natom, 3)
    assert g.shape == (2, 2, mol.natom, 3, mol.natom, 3)
    expected_f = np.moveaxis(
        terms.F_cartesian.reshape(mol.natom, 3, 2, 2),
        (0, 1),
        (-2, -1),
    )
    expected_g = np.moveaxis(
        terms.G_cartesian.reshape(mol.natom, 3, mol.natom, 3, 2, 2),
        (0, 1, 2, 3),
        (-4, -3, -2, -1),
    )
    np.testing.assert_allclose(f, expected_f, atol=1e-10)
    np.testing.assert_allclose(g, expected_g, atol=1e-10)


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

    assert mc.solver_backend in {
        'ci_dense_fallback',
        'direct_ci_spin_string',
        'direct_ci_compact_conn',
        'direct_ci_factor_conn',
    }


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

    assert mc_direct.solver_backend == 'direct_ci_spin_string'
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


def test_builtin_s8_dense_eri_transform_unpacks_compressed_cache():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='builtin', aosym='s8', eri='dense')

    mf = mol.RHF().run()
    mo_cas = mf.mo_coeff[:, :2]

    assert mf.eri is None
    assert mf.eri_s8 is not None
    eri_active = transform_spatial_eri_to_mo(mf, mo_cas, use_cholesky=False)

    np.testing.assert_allclose(eri_active, mf.get_eri_mo()[0:2, 0:2, 0:2, 0:2])


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
