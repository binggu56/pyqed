import pytest
import numpy as np

from pyqed import optimize as optimize_module
from pyqed.qchem import CASSCF, COCASCI, Molecule, soc_state_interaction
from pyqed.qchem.mcscf import casci as casci_module
from pyqed.qchem.mcscf import cocasci as cocasci_module
from pyqed.qchem.mcscf import direct_ci as direct_ci_module
from pyqed.qchem.mcscf.casci import (
    _get_mf_cholesky_factors,
    transform_eri_factors_to_mo_pair,
)
from pyqed.qchem.mcscf.direct_ci import CASCI
from pyqed.qchem.mcscf.orbopt import (
    augmented_hessian_direction,
    davidson_augmented_hessian_direction,
    diagonal_hessian,
    diagonal_preconditioned_vector,
    embed_rdm2,
    generalized_fock,
    generalized_fock_from_factors,
    limit_step_norm,
    orbital_hessian_action_from_integrals,
    orbital_step,
    pack_nonredundant,
    quadratic_model_change,
    rotate_orbitals,
    unpack_nonredundant,
)
from pyqed.qchem.soc import (
    get_soc_1e_ao,
    get_soc_2e_somf_ao,
    get_soc_somf_spin_orbital,
    soc_1e_prefactor,
)


def _grouped_spin_orbital_occ(binary):
    return np.concatenate((binary[0], binary[1])).astype(np.int8)


def _bruteforce_tdm1_spin_orbital(mc_bra, bra_id, mc_ket, ket_id):
    bra_occ = [_grouped_spin_orbital_occ(det) for det in mc_bra.binary]
    ket_occ = [_grouped_spin_orbital_occ(det) for det in mc_ket.binary]
    bra_lookup = {occ.tobytes(): idx for idx, occ in enumerate(bra_occ)}

    cibra = np.asarray(mc_bra.ci[bra_id])
    ciket = np.asarray(mc_ket.ci[ket_id])
    nso = 2 * mc_bra.ncas
    tdm = np.zeros((nso, nso), dtype=complex)

    for j, occ in enumerate(ket_occ):
        occupied = np.flatnonzero(occ)
        for v in occupied:
            sign_ann = -1 if int(np.sum(occ[:v])) % 2 else 1
            occ_after_ann = occ.copy()
            occ_after_ann[v] = 0
            unoccupied = np.flatnonzero(1 - occ_after_ann)
            for u in unoccupied:
                sign_cre = -1 if int(np.sum(occ_after_ann[:u])) % 2 else 1
                occ_final = occ_after_ann.copy()
                occ_final[u] = 1
                i = bra_lookup.get(occ_final.tobytes())
                if i is None:
                    continue
                tdm[u, v] += cibra[i].conj() * ciket[j] * (sign_ann * sign_cre)

    return tdm


def test_casscf_lih_lowers_the_initial_casci_energy():
    """Exercise the native U-matrix orbital optimizer on a nontrivial case."""
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = mol.RHF().run()

    mc0 = CASCI(mf, ncas=2, nelecas=2).run(nstates=1, method='ci')
    mc = COCASCI(mf, ncas=2, nelecas=2, max_cycles=20).run()

    assert mc.e_tot[0] < mc0.e_tot[0] - 1e-6
    assert np.isfinite(mc.e_tot[0])


def test_casscf_lih_lbfgs_matches_rcg_energy():
    """The alternative orbital optimizer should reach the same LiH minimum."""
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = mol.RHF().run()

    mc_rcg = COCASCI(mf, ncas=2, nelecas=2, max_cycles=20, optimizer='RCG').run()
    mc_lbfgs = COCASCI(mf, ncas=2, nelecas=2, max_cycles=20, optimizer='LBFGS').run()

    np.testing.assert_allclose(mc_lbfgs.e_tot, mc_rcg.e_tot, atol=1e-6)


def test_casscf_lih_diis_matches_non_diis_energy():
    """DIIS should accelerate the U updates without changing the LiH minimum."""
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = mol.RHF().run()

    mc_plain = COCASCI(
        mf, ncas=2, nelecas=2, max_cycles=20, optimizer='RCG', diis=False,
    ).run()
    mc_diis = COCASCI(
        mf, ncas=2, nelecas=2, max_cycles=20, optimizer='RCG', diis=True,
    ).run()

    np.testing.assert_allclose(mc_diis.e_tot, mc_plain.e_tot, atol=1e-6)


def test_stiefel_minimize_respects_max_iterations_cap():
    """The inner Stiefel optimizer should support a hard iteration cap."""
    U0 = np.array([[1.0], [0.0]])
    h1e = np.array([[0.0, 0.0], [0.0, -1.0]])
    eri = np.zeros((2, 2, 2, 2))
    dm1 = np.array([[1.0]])
    dm2 = np.zeros((1, 1, 1, 1))

    X, value = optimize_module.minimize(
        optimize_module.energy,
        U0,
        args=(h1e, eri, dm1, dm2),
        max_iterations=0,
    )

    np.testing.assert_allclose(X, U0)
    np.testing.assert_allclose(value, optimize_module.energy(U0, h1e, eri, dm1, dm2))


def test_stiefel_minimize_clips_large_tangent_steps():
    """The inner Stiefel optimizer should respect a hard tangent-step cap."""
    U0 = np.array([[1.0], [0.0]])
    h1e = np.array([[0.0, -1.0], [-1.0, 0.0]])
    eri = np.zeros((2, 2, 2, 2))
    dm1 = np.array([[1.0]])
    dm2 = np.zeros((1, 1, 1, 1))

    X, _ = optimize_module.minimize(
        optimize_module.energy,
        U0,
        args=(h1e, eri, dm1, dm2),
        tau=10.0,
        max_iterations=1,
        max_step_norm=0.25,
    )

    G = optimize_module.gradient(U0, h1e, eri, dm1, dm2)
    df = optimize_module.grad(U0, G)
    expected_step = 0.25 / optimize_module.norm(df)
    expected = optimize_module.retract(U0, -expected_step * df)

    np.testing.assert_allclose(X, expected)


def test_cocasci_accepts_inner_optimizer_tolerance():
    """COCASCI should expose configurable inner manifold optimizer controls."""
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = mol.RHF().run()
    mc = COCASCI(
        mf,
        ncas=2,
        nelecas=2,
        optimizer_tol=5.0e-4,
        optimizer_max_steps=12,
        optimizer_max_step_norm=0.3,
    )

    assert mc.optimizer_tol == pytest.approx(5.0e-4)
    assert mc.optimizer_max_steps == 12
    assert mc.optimizer_max_step_norm == pytest.approx(0.3)


def test_first_order_casscf_lih_lowers_initial_casci_energy():
    """The public CASSCF class should improve on the initial CASCI reference."""
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = mol.RHF().run()

    mc0 = CASCI(mf, ncas=2, nelecas=2).run(nstates=1, method='direct_ci')
    mc = CASSCF(
        mf,
        ncas=2,
        nelecas=2,
        max_cycle=40,
        ci_method='direct_ci',
    ).run()

    assert mc.e_tot[0] < mc0.e_tot[0] - 1e-6
    assert mc.converged
    assert np.isfinite(mc.e_tot[0])


def test_first_order_casscf_accepts_max_cycles_alias():
    """CASSCF should accept the legacy max_cycles keyword."""
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = mol.RHF().run()

    mc = CASSCF(
        mf,
        ncas=2,
        nelecas=2,
        max_cycles=17,
        ci_method='direct_ci',
    )

    assert mc.max_cycle == 17
    assert mc.max_cycles == 17


def test_first_order_casscf_repeated_run_resets_history():
    """Reusing the same CASSCF object should start from a clean history."""
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = mol.RHF().run()
    mc = CASSCF(
        mf,
        ncas=2,
        nelecas=2,
        max_cycle=40,
        ci_method='direct_ci',
    )

    mc.run()
    history_len = len(mc.history)
    assert history_len > 0
    assert "step_norm" in mc.history[-1]

    mc.run()
    assert len(mc.history) == history_len
    assert mc.converged


def test_cocasci_lih_4e4o_matches_first_order_casscf():
    """Regression test for the LiH (4e,4o) COCASCI energy consistency bug."""
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='pyscf')

    mf = mol.RHF().run()

    mc_ref = CASSCF(
        mf,
        ncas=4,
        nelecas=4,
        max_cycle=60,
        ci_method='direct_ci',
    ).run()
    mc_opt = COCASCI(
        mf,
        ncas=4,
        nelecas=4,
        max_cycles=20,
        ci_method='direct_ci',
    ).run()

    np.testing.assert_allclose(mc_opt.e_tot, mc_ref.e_tot, atol=1e-6)


def test_first_order_casscf_state_average_two_roots():
    """State-averaged first-order CASSCF should support 2-root optimization."""
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='pyscf')

    mf = mol.RHF().run()

    mc = CASSCF(
        mf,
        ncas=4,
        nelecas=4,
        max_cycle=60,
        ci_method='direct_ci',
    )
    mc.state_average([0.5, 0.5]).run(nstates=2)

    assert mc.converged
    assert len(mc.e_tot) == 2
    assert np.all(np.isfinite(mc.e_tot))
    assert mc.e_tot[0] < mc.e_tot[1]


def test_first_order_casscf_cholesky_matches_pyscf():
    """Public CASSCF should route a Cholesky RHF reference into factorized CASCI."""
    pyscf_mcscf = pytest.importorskip('pyscf.mcscf')
    pyscf_scf = pytest.importorskip('pyscf.scf')
    pyscf_gto = pytest.importorskip('pyscf.gto')

    atom = 'Li 0 0 0; H 0 0 1.6'
    basis = 'sto-3g'

    mol = Molecule(atom=atom, unit='angstrom', basis=basis)
    mol.build(driver='gbasis-pyscf')

    mf = mol.RHF().run(cholesky_jk=True, cholesky_tol=1e-10)
    mc = CASSCF(
        mf,
        ncas=4,
        nelecas=4,
        max_cycle=60,
        ci_method='direct_ci',
    ).run()

    assert mc.converged
    assert mc.use_cholesky_integrals
    assert mc.casci.use_cholesky_integrals

    pmol = pyscf_gto.M(atom=atom, basis=basis, unit='angstrom', verbose=0)
    pmf = pyscf_scf.RHF(pmol)
    pmf.conv_tol = 1e-10
    pmf.kernel()

    pmc = pyscf_mcscf.CASSCF(pmf, 4, 4)
    pmc.conv_tol = 1e-9
    e_ref, _, _, _, _ = pmc.kernel()

    np.testing.assert_allclose(mc.e_tot[0], e_ref, atol=5e-6)


def test_first_order_casscf_factorized_fock_matches_dense():
    """Factorized orbital gradients should reproduce the dense generalized Fock."""
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis-pyscf')

    mf = mol.RHF().run(cholesky_jk=True, cholesky_tol=1e-10)
    mc = CASCI(mf, ncas=4, nelecas=4).run(nstates=1, method='direct_ci', use_cholesky=True)

    h1_mo = mf.get_hcore_mo(mf.mo_coeff)
    eri_mo = mf.get_eri_mo(mf.mo_coeff, notation='chem')
    dm1_full = mc.make_rdm1(0, with_core=True, with_vir=True, representation='mo')
    dm2_full = embed_rdm2(mc.make_rdm2(0, with_core=True), mf.nmo)
    fock_dense = generalized_fock(h1_mo, eri_mo, dm1_full, dm2_full)

    occ_mo = mf.mo_coeff[:, :mc.ncore + mc.ncas]
    pair_factors = transform_eri_factors_to_mo_pair(
        _get_mf_cholesky_factors(mf),
        mf.mo_coeff,
        occ_mo,
    )
    fock_factor = generalized_fock_from_factors(
        h1_mo,
        pair_factors,
        mc.make_rdm1(0, with_core=True),
        mc.make_rdm2(0, with_core=True),
    )

    np.testing.assert_allclose(fock_factor, fock_dense, atol=1e-8)


def test_first_order_casscf_factorized_evaluate_avoids_dense_mo_eri(monkeypatch):
    """The factorized CASSCF orbital step should not request dense MO ERIs."""
    mol = Molecule(atom='H 0 0 0; H 0 0 0.8; H 0 0 1.6; H 0 0 2.4', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis-pyscf')

    mf = mol.RHF().run(cholesky_jk=True, cholesky_tol=1e-10)
    casscf = CASSCF(mf, ncas=4, nelecas=4, max_cycle=20, ci_method='direct_ci')
    casscf.use_cholesky_integrals = casscf._resolve_use_cholesky(True)

    def fail_get_eri_mo(*args, **kwargs):
        raise AssertionError("dense MO ERIs should not be built in factorized CASSCF")

    monkeypatch.setattr(mf, "get_eri_mo", fail_get_eri_mo)

    mc, fock, grad = casscf._evaluate(mf.mo_coeff, 1, 0)
    assert np.isfinite(mc.e_tot[0])
    assert fock.shape == (mf.nmo, mf.nmo)
    assert grad.shape == (mf.nmo, mf.nmo)


def test_first_order_casscf_reuses_direct_ci_setup_cache(monkeypatch):
    """Repeated trial CASCI solves should reuse determinant setup across line-search calls."""
    mol = Molecule(atom='H 0 0 0; H 0 0 0.8; H 0 0 1.6; H 0 0 2.4', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis-pyscf')

    mf = mol.RHF().run(cholesky_jk=True, cholesky_tol=1e-10)
    casscf = CASSCF(mf, ncas=4, nelecas=4, max_cycle=20, ci_method='direct_ci')
    casscf.use_cholesky_integrals = casscf._resolve_use_cholesky(True)

    mc0 = casscf._make_casci(mf.mo_coeff, 1)
    casscf._effective_rdms_occ(mc0, 0)

    def fail_build_direct_connectivity(*args, **kwargs):
        raise AssertionError("direct connectivity should have been reused from cache")

    def fail_slater_condon(*args, **kwargs):
        raise AssertionError("Slater-Condon tables should have been reused from cache")

    monkeypatch.setattr(direct_ci_module, "build_direct_connectivity", fail_build_direct_connectivity)
    monkeypatch.setattr(direct_ci_module, "SlaterCondon", fail_slater_condon)

    mc1 = casscf._make_casci(mf.mo_coeff, 1, ci0=casscf._copy_ci_guess(mc0.ci))
    dm1, dm2 = casscf._effective_rdms_occ(mc1, 0)

    assert np.isfinite(mc1.e_tot[0])
    assert dm1.shape[0] == mc1.ncore + mc1.ncas
    assert dm2.shape[0] == mc1.ncore + mc1.ncas


def test_cocas_fresh_casci_reuses_direct_ci_setup_cache(monkeypatch):
    """Fresh COCAS-style CASCI clones should reuse determinant setup tables."""
    mol = Molecule(atom='H 0 0 0; H 0 0 0.8; H 0 0 1.6; H 0 0 2.4', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis-pyscf')

    mf = mol.RHF().run(cholesky_jk=True, cholesky_tol=1e-10)
    mc0 = CASCI(mf, ncas=4, nelecas=4)
    mc0.run(nstates=1, method='direct_ci', use_cholesky=True)
    mc0.make_rdm12(0, with_core=True)

    def fail_build_direct_connectivity(*args, **kwargs):
        raise AssertionError("direct connectivity should have been reused from cache")

    def fail_slater_condon(*args, **kwargs):
        raise AssertionError("Slater-Condon tables should have been reused from cache")

    monkeypatch.setattr(direct_ci_module, "build_direct_connectivity", fail_build_direct_connectivity)
    monkeypatch.setattr(direct_ci_module, "SlaterCondon", fail_slater_condon)

    mc1 = cocasci_module._fresh_casci_like(mc0)
    mc1.run(
        nstates=1,
        mo_coeff=mf.mo_coeff,
        method='direct_ci',
        ci0=[np.array(vec, copy=True) for vec in mc0.ci],
        use_cholesky=True,
    )
    dm1, dm2 = mc1.make_rdm12(0, with_core=True)

    assert np.isfinite(mc1.e_tot[0])
    assert dm1.shape[0] == mc1.ncore + mc1.ncas
    assert dm2.shape[0] == mc1.ncore + mc1.ncas


def test_cocas_factorized_objective_and_gradient_match_dense():
    """COCAS orbital objective should agree between dense and factorized ERIs."""
    mol = Molecule(atom='H 0 0 0; H 0 0 0.8; H 0 0 1.6; H 0 0 2.4', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis-pyscf')

    mf = mol.RHF().run(cholesky_jk=True, cholesky_tol=1e-10)
    mc = CASCI(mf, ncas=4, nelecas=4)
    mc.run(nstates=1, method='direct_ci', use_cholesky=True)

    h1e = mf.get_hcore_mo()
    eri_dense = mf.get_eri_mo()
    eri_factors = transform_eri_factors_to_mo_pair(_get_mf_cholesky_factors(mf), mf.mo_coeff)
    dm1, dm2 = mc.make_rdm12(0, with_core=True)

    nocc_like = mc.ncore + mc.ncas
    U = np.eye(mf.nmo, nocc_like)

    e_dense = cocasci_module.energy(U, h1e, eri_dense, dm1, dm2)
    e_factor = cocasci_module.energy(U, h1e, eri_factors, dm1, dm2)
    np.testing.assert_allclose(e_factor, e_dense, atol=1e-10)

    g_dense = optimize_module.gradient(U, h1e, eri_dense, dm1, dm2)
    g_factor = optimize_module.gradient(U, h1e, eri_factors, dm1, dm2)
    np.testing.assert_allclose(g_factor, g_dense, atol=1e-10)


def test_casci_overlap_self_is_identity():
    """The determinant-overlap contraction should preserve orthonormal roots."""
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='pyscf')

    mf = mol.RHF().run()
    mc = CASCI(mf, ncas=4, nelecas=4).run(nstates=2, method='direct_ci')

    smo = mf.mo_coeff.T @ mf.get_ovlp() @ mf.mo_coeff
    ref = np.eye(2)

    np.testing.assert_allclose(direct_ci_module.overlap(mc, mc, s=smo), ref, atol=1e-10)
    np.testing.assert_allclose(casci_module.overlap(mc, mc, s=smo), ref, atol=1e-10)
    np.testing.assert_allclose(direct_ci_module.overlap(mc, mc), ref, atol=1e-10)
    np.testing.assert_allclose(casci_module.overlap(mc, mc), ref, atol=1e-10)


def test_casci_spin_orbital_tdm_matches_spin_blocks():
    """Spin-orbital TDMs should reduce to the alpha/beta RDM blocks on the diagonal."""
    mol = Molecule(atom='H 0 0 0; H 0 0 0.74', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = mol.RHF().run()
    mc = CASCI(mf, ncas=2, nelecas=2, spin=0).run(nstates=1, method='direct_ci')

    dm1a, dm1b = mc.make_rdm1s(0)
    dm1so = mc.make_tdm1_spin_orbital(0, order='grouped')
    ncas = mc.ncas

    np.testing.assert_allclose(dm1so[:ncas, :ncas], dm1a, atol=1e-10)
    np.testing.assert_allclose(dm1so[ncas:, ncas:], dm1b, atol=1e-10)
    np.testing.assert_allclose(dm1so[:ncas, ncas:], 0.0, atol=1e-10)
    np.testing.assert_allclose(dm1so[ncas:, :ncas], 0.0, atol=1e-10)


def test_casci_spin_orbital_tdm_matches_bruteforce_between_spin_sectors():
    """Spin-flip blocks should agree with an explicit determinant-space contraction."""
    mol = Molecule(atom='H 0 0 0; H 0 0 0.74', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = mol.RHF().run()
    mc_singlet = CASCI(mf, ncas=2, nelecas=2, spin=0).run(nstates=1, method='direct_ci')
    mc_triplet = CASCI(mf, ncas=2, nelecas=2, spin=2).run(nstates=1, method='direct_ci')

    tdm_ref = _bruteforce_tdm1_spin_orbital(mc_singlet, 0, mc_triplet, 0)
    tdm = mc_singlet.make_tdm1_spin_orbital(0, other=mc_triplet, order='grouped')
    np.testing.assert_allclose(tdm, tdm_ref, atol=1e-10)

    hso = np.zeros((2 * mc_singlet.ncas, 2 * mc_singlet.ncas), dtype=complex)
    hso[0, 2] = 0.37 + 0.11j
    hso[1, 3] = -0.19 + 0.23j
    value = mc_singlet.contract_with_tdm1_spin_orbital(0, other=mc_triplet, h1e=hso)
    value_ref = np.einsum('uv,uv->', hso, tdm_ref)
    np.testing.assert_allclose(value, value_ref, atol=1e-10)


def test_soc_state_interaction_builds_hermitian_matrix():
    """The SOC SI helper should build a Hermitian total Hamiltonian."""
    mol = Molecule(atom='H 0 0 0; H 0 0 0.74', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = mol.RHF().run()
    mc_singlet = CASCI(mf, ncas=2, nelecas=2, spin=0).run(nstates=1, method='direct_ci')
    mc_triplet = CASCI(mf, ncas=2, nelecas=2, spin=2).run(nstates=1, method='direct_ci')

    hso = np.zeros((2 * mc_singlet.ncas, 2 * mc_singlet.ncas), dtype=complex)
    hso[0, 2] = 0.37 + 0.11j
    hso[1, 3] = -0.19 + 0.23j
    hso[2, 0] = hso[0, 2].conjugate()
    hso[3, 1] = hso[1, 3].conjugate()

    result = soc_state_interaction([(mc_singlet, 0), (mc_triplet, 0)], hso=hso)

    np.testing.assert_allclose(result.h_soc, result.h_soc.conjugate().T, atol=1e-10)
    np.testing.assert_allclose(result.h_total, result.h_total.conjugate().T, atol=1e-10)
    np.testing.assert_allclose(
        result.h_soc[0, 1],
        mc_singlet.soc_matrix_element(0, other=mc_triplet, hso=hso),
        atol=1e-10,
    )
    assert result.eigenvalues.shape == (2,)
    assert result.eigenvectors.shape == (2, 2)


@pytest.mark.parametrize(
    ("atom", "basis"),
    [
        (
            '''
            Li 0 0 0
            H  0 0 1.6
            ''',
            'sto-3g',
        ),
        (
            '''
            S  0.000000  0.000000  0.000000
            H  0.000000  1.229000  0.958000
            H  0.000000 -1.229000  0.958000
            ''',
            'sto-3g',
        ),
    ],
)
def test_gbasis_pyscf_soc_ao_matches_direct_pyscf_operator(atom, basis):
    """The gbasis-pyscf SOC AO build should reproduce the PySCF one-center operator."""
    from pyscf import gto

    mol = Molecule(atom=atom, unit='angstrom', basis=basis)
    mol.build(driver='gbasis-pyscf')
    hso = get_soc_1e_ao(mol, one_center=True)

    pmol = gto.M(atom=atom, basis=basis, unit='angstrom', verbose=0)
    hso_ref = np.zeros_like(hso)
    aoslices = pmol.aoslice_by_atom()
    for ia in range(pmol.natm):
        p0, p1 = aoslices[ia, 2], aoslices[ia, 3]
        with pmol.with_rinv_as_nucleus(ia):
            w = pmol.intor('int1e_prinvxp', comp=3)
        hso_ref[:, p0:p1, p0:p1] += (-pmol.atom_charge(ia)) * w[:, p0:p1, p0:p1]
    hso_ref *= soc_1e_prefactor()

    np.testing.assert_allclose(hso, hso_ref, atol=1e-11)


def test_gbasis_pyscf_somf_ao_matches_manual_contraction():
    """The SOMF AO builder should match a direct PySCF int2e_p1vxp1 contraction."""
    from pyscf import gto

    atom = '''
        S  0.000000  0.000000  0.000000
        H  0.000000  1.229000  0.958000
        H  0.000000 -1.229000  0.958000
    '''
    basis = 'sto-3g'

    mol = Molecule(atom=atom, unit='angstrom', basis=basis)
    mol.build(driver='gbasis-pyscf')
    mf = mol.RHF().run()
    hso = get_soc_2e_somf_ao(mf)

    pmol = gto.M(atom=atom, basis=basis, unit='angstrom', verbose=0)
    dm = mf.make_rdm1()
    g = pmol.intor('int2e_p1vxp1', comp=3)
    hso_ref = (
        np.einsum('xpqrs,rs->xpq', g, dm, optimize=True)
        - 1.5 * np.einsum('xprsq,rs->xpq', g, dm, optimize=True)
        - 1.5 * np.einsum('xsqpr,rs->xpq', g, dm, optimize=True)
    ) * soc_1e_prefactor()

    np.testing.assert_allclose(hso, hso_ref, atol=1e-11)


def test_soc_state_interaction_somf_matches_explicit_operator():
    """The SOMF SI helper should use the same state-averaged operator as the explicit path."""
    mol = Molecule(
        atom='''
            S  0.000000  0.000000  0.000000
            H  0.000000  1.229000  0.958000
            H  0.000000 -1.229000  0.958000
        ''',
        unit='angstrom',
        basis='sto-3g',
    )
    mol.build(driver='gbasis-pyscf')

    mf = mol.RHF().run()
    mc_singlet = CASCI(mf, ncas=2, nelecas=2, spin=0).run(nstates=1, method='direct_ci')
    mc_triplet = CASCI(mf, ncas=2, nelecas=2, spin=2).run(nstates=1, method='direct_ci')
    states = [(mc_singlet, 0), (mc_triplet, 0)]

    hso = get_soc_somf_spin_orbital(
        mf,
        representation='mo',
        mo_coeff=mc_singlet.mo_cas,
        states=states,
        order='grouped',
    )
    result = soc_state_interaction(states, soc_model='somf')

    np.testing.assert_allclose(result.h_soc, result.h_soc.conjugate().T, atol=1e-10)
    np.testing.assert_allclose(
        result.h_soc[0, 1],
        mc_singlet.soc_matrix_element(0, other=mc_triplet, hso=hso),
        atol=1e-10,
    )


def test_first_order_casscf_line_search_failure_raises(monkeypatch):
    """A rejected orbital step should surface as a RuntimeError."""
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = mol.RHF().run()
    mc = CASSCF(mf, ncas=2, nelecas=2, max_cycle=4, ci_method='direct_ci')

    def fail_line_search(*args, **kwargs):
        return False, args[0], float("inf"), 0.0, None

    monkeypatch.setattr(mc, "_line_search", fail_line_search)

    with pytest.raises(RuntimeError, match="line search failed"):
        mc.run()


def test_first_order_casscf_stall_message_includes_diagnostics():
    """Non-convergence diagnostics should report the last and best macro cycles."""
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='pyscf')

    mf = mol.RHF().run()
    mc = CASSCF(mf, ncas=2, nelecas=2, ci_method='direct_ci')
    mc.state_average([0.5, 0.5])
    mc.history = [
        {"cycle": 1, "energy": -7.80, "gradient_norm": 1.0e-2, "step_norm": None},
        {"cycle": 2, "energy": -7.81, "gradient_norm": 2.0e-3, "step_norm": 4.0e-2},
    ]

    message = mc._format_stall_message("Max macro steps reached before the CASSCF optimizer converged.")

    assert "Last cycle: 2" in message
    assert "Best cycle: 2" in message
    assert "Inspect mc.history" in message
    assert "casscf_compare_vs_pyscf.py" in message


def test_first_order_casscf_line_search_fallback_generates_smaller_steps():
    """Rejected orbital steps should produce smaller fallback candidates."""
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = mol.RHF().run()
    mc = CASSCF(mf, ncas=2, nelecas=2, ci_method='direct_ci')

    step_vec = np.array([0.05, -0.03, 0.01])
    grad_vec = np.array([1.2, -0.8, 0.4])
    fallback = mc._fallback_step_vectors(step_vec, grad_vec)

    assert len(fallback) >= 3
    assert all(vec.shape == step_vec.shape for vec in fallback)
    assert all(np.max(np.abs(vec)) <= 0.02 + 1e-12 for vec in fallback[:2])
    assert any(np.max(np.abs(vec)) <= 0.01 + 1e-12 for vec in fallback)


def test_augmented_hessian_direction_is_a_descent_step():
    """The diagonal AH model should return a downhill packed orbital step."""
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = mol.RHF().run()
    mc = CASSCF(mf, ncas=2, nelecas=2, ci_method='direct_ci')
    casci, fock, grad = mc._evaluate(mf.mo_coeff, 1, 0)
    grad_vec = pack_nonredundant(grad, casci.ncore, casci.ncas, mf.nmo)
    diag_step = diagonal_preconditioned_vector(
        grad,
        fock,
        casci.ncore,
        casci.ncas,
        level_shift=mc.level_shift,
    )
    hess_diag = diagonal_hessian(
        fock,
        casci.ncore,
        casci.ncas,
        level_shift=mc.level_shift,
    )
    step_vec = augmented_hessian_direction(
        grad_vec,
        hess_diag,
        max_step=mc.max_step,
        regularization=mc.level_shift,
        fallback_step=diag_step,
    )

    assert step_vec.shape == grad_vec.shape
    assert np.max(np.abs(step_vec)) <= mc.max_step + 1e-12
    assert np.dot(step_vec, grad_vec) < 0.0
    diag_step_clipped = limit_step_norm(diag_step, mc.max_step)
    assert (
        quadratic_model_change(step_vec, grad_vec, np.maximum(np.abs(hess_diag), mc.level_shift))
        <= quadratic_model_change(
            diag_step_clipped,
            grad_vec,
            np.maximum(np.abs(hess_diag), mc.level_shift),
        )
        + 1e-12
    )


def test_davidson_augmented_hessian_matches_diagonal_model():
    """Matrix-free AH should recover the diagonal-model step on a diagonal test problem."""
    grad_vec = np.array([0.4, -0.2, 0.1, -0.05])
    hess_diag = np.array([2.0, 3.0, 4.0, 5.0])
    fallback = -grad_vec / hess_diag

    step_vec = davidson_augmented_hessian_direction(
        grad_vec,
        hess_diag,
        matvec=lambda vec: hess_diag * vec,
        max_step=0.5,
        regularization=1.0e-8,
        max_cycle=4,
        max_subspace=6,
        fallback_step=fallback,
    )

    assert step_vec.shape == grad_vec.shape
    assert np.dot(step_vec, grad_vec) < 0.0
    assert np.max(np.abs(step_vec)) <= 0.5 + 1e-12
    assert quadratic_model_change(step_vec, grad_vec, hess_diag) < 0.0
    np.testing.assert_allclose(step_vec, limit_step_norm(fallback, 0.5), atol=1.0e-2)


def test_orbital_hessian_action_from_integrals_matches_finite_difference():
    """Analytic orbital-only Hessian action should match a frozen-RDM finite difference."""
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = mol.RHF().run()
    mc = CASCI(mf, ncas=2, nelecas=2).run(nstates=1, method='direct_ci')

    dm1 = mc.make_rdm1(0, with_core=True, with_vir=True, representation='mo')
    dm2 = embed_rdm2(mc.make_rdm2(0, with_core=True), mf.nmo)
    h1_mo = mf.get_hcore_mo(mf.mo_coeff)
    eri_mo = mf.get_eri_mo(mf.mo_coeff, notation='chem')
    fock = generalized_fock(h1_mo, eri_mo, dm1, dm2)
    grad0 = orbital_step(
        fock,
        mc.ncore,
        mc.ncas,
        step_size=1.0,
        level_shift=1.0e-3,
        max_step=0.25,
    )[1]

    step_vec = np.array([0.03, -0.02, 0.01], dtype=float)
    kappa = unpack_nonredundant(step_vec, mc.ncore, mc.ncas, mf.nmo)
    analytic = orbital_hessian_action_from_integrals(h1_mo, eri_mo, dm1, dm2, kappa)

    eps = 1.0e-5
    trial_coeff = rotate_orbitals(mf.mo_coeff, eps * kappa)
    h1_t = mf.get_hcore_mo(trial_coeff)
    eri_t = mf.get_eri_mo(trial_coeff, notation='chem')
    fock_t = generalized_fock(h1_t, eri_t, dm1, dm2)
    grad_t = orbital_step(
        fock_t,
        mc.ncore,
        mc.ncas,
        step_size=1.0,
        level_shift=1.0e-3,
        max_step=0.25,
    )[1]
    fd = (grad_t - grad0) / eps

    np.testing.assert_allclose(analytic, fd, atol=1e-5, rtol=1e-4)


def test_analytic_hessian_action_reuses_reference_cache(monkeypatch):
    """Repeated analytic AH matvecs should reuse the same MO-integral/RDM bundle."""
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = mol.RHF().run()
    solver = CASSCF(mf, ncas=2, nelecas=2, optimizer='AH', ah_hessian='analytic')
    mc = CASCI(mf, ncas=2, nelecas=2).run(nstates=1, method='direct_ci')

    calls = {"rdms": 0, "integrals": 0}
    orig_rdms = solver._effective_rdms
    orig_get_integrals = solver._get_integrals

    def counting_rdms(current_mc, state_id):
        calls["rdms"] += 1
        return orig_rdms(current_mc, state_id)

    def counting_integrals(mo_coeff):
        calls["integrals"] += 1
        return orig_get_integrals(mo_coeff)

    monkeypatch.setattr(solver, "_effective_rdms", counting_rdms)
    monkeypatch.setattr(solver, "_get_integrals", counting_integrals)

    grad_vec = pack_nonredundant(np.zeros((mf.nmo, mf.nmo)), mc.ncore, mc.ncas, mf.nmo)
    solver._orbital_hessian_action(
        mf.mo_coeff,
        mc,
        grad_vec,
        np.array([0.02, -0.01, 0.005], dtype=float),
    )
    solver._orbital_hessian_action(
        mf.mo_coeff,
        mc,
        grad_vec,
        np.array([-0.01, 0.015, -0.004], dtype=float),
    )

    assert calls == {"rdms": 1, "integrals": 1}

    solver._invalidate_ah_reference_cache()
    solver._orbital_hessian_action(
        mf.mo_coeff,
        mc,
        grad_vec,
        np.array([0.01, 0.0, -0.003], dtype=float),
    )

    assert calls == {"rdms": 2, "integrals": 2}


def test_evaluate_populates_analytic_hessian_cache_for_state_average(monkeypatch):
    """The current-point SA bundle from `_evaluate` should feed AH matvecs directly."""
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = mol.RHF().run()
    solver = CASSCF(mf, ncas=2, nelecas=2, optimizer='AH', ah_hessian='analytic')
    solver.state_average(np.array([0.6, 0.4]))
    solver.nstates = 2
    solver.state_id = 0

    mc, _, grad = solver._evaluate(mf.mo_coeff, nstates=2, state_id=0)
    grad_vec = pack_nonredundant(grad, mc.ncore, mc.ncas, mf.nmo)

    calls = {"rdms": 0, "integrals": 0}
    orig_rdms = solver._effective_rdms
    orig_get_integrals = solver._get_integrals

    def counting_rdms(current_mc, state_id):
        calls["rdms"] += 1
        return orig_rdms(current_mc, state_id)

    def counting_integrals(mo_coeff):
        calls["integrals"] += 1
        return orig_get_integrals(mo_coeff)

    monkeypatch.setattr(solver, "_effective_rdms", counting_rdms)
    monkeypatch.setattr(solver, "_get_integrals", counting_integrals)

    solver._orbital_hessian_action(
        mf.mo_coeff,
        mc,
        grad_vec,
        np.array([0.02, -0.01, 0.005], dtype=float),
    )

    assert calls == {"rdms": 0, "integrals": 0}
