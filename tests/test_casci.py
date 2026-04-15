import numpy as np
import pytest

from pyqed.qchem import Molecule
from pyqed.qchem.hf import RHF, UHF
from pyqed.qchem.mcscf.casci import CASCI, transform_spatial_eri_to_mo
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


def test_casci_accepts_open_shell_uhf_reference():
    mol = Molecule(atom='H 0 0 0', unit='bohr', basis='sto-3g', spin=1)
    mol.build(driver='gbasis')

    uhf = UHF(mol).run()
    mc = CASCI(uhf, ncas=1, nelecas=(1, 0)).run(nstates=1)

    np.testing.assert_allclose(mc.e_tot[0], uhf.e_tot, atol=1e-10)
    np.testing.assert_allclose(mc.make_rdm1(0), np.array([[1.0]]), atol=1e-10)


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


def test_casci_make_tdm1s_sum_to_spin_traced_tdm():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = RHF(mol).run()
    mc = CASCI(mf, ncas=2, nelecas=2).run(nstates=2)

    tdm1a, tdm1b = mc.make_tdm1s(1, 0)

    np.testing.assert_allclose(tdm1a + tdm1b, mc.make_tdm1(1, 0), atol=1e-10)
    np.testing.assert_allclose(sum(mc.make_rdm1s(0)), mc.make_rdm1(0), atol=1e-10)


def test_direct_ci_falls_back_to_dense_solver_for_small_spaces():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = RHF(mol).run()

    mc_direct = direct_ci.CASCI(mf, ncas=4, nelecas=4).run(nstates=2, method='direct_ci')
    mc_dense = CASCI(mf, ncas=4, nelecas=4).run(nstates=2)

    assert mc_direct.solver_backend == 'ci_dense_fallback'
    np.testing.assert_allclose(mc_direct.e_tot, mc_dense.e_tot, atol=1e-10)


def test_direct_ci_on_the_fly_matches_dense_solver():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = RHF(mol).run()

    mc_direct = direct_ci.CASCI(mf, ncas=4, nelecas=4)
    mc_direct.direct_ci_dense_fallback_ndets = 1
    mc_direct.run(nstates=2, method='direct_ci')

    mc_dense = CASCI(mf, ncas=4, nelecas=4).run(nstates=2)

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

    mc_dense = CASCI(mf, ncas=4, nelecas=4).run(nstates=2)
    mc_cd = CASCI(mf, ncas=4, nelecas=4).run(nstates=2, use_cholesky=True)

    np.testing.assert_allclose(mc_cd.e_tot, mc_dense.e_tot, atol=1e-8)


def test_direct_ci_use_cholesky_matches_dense_energy():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis-pyscf')

    mf = RHF(mol).run(cholesky_jk=True, cholesky_tol=1e-10)

    mc_direct = direct_ci.CASCI(mf, ncas=4, nelecas=4)
    mc_direct.direct_ci_dense_fallback_ndets = 1
    mc_direct.run(nstates=2, method='direct_ci', use_cholesky=True)

    mc_dense = CASCI(mf, ncas=4, nelecas=4).run(nstates=2)

    np.testing.assert_allclose(mc_direct.e_tot, mc_dense.e_tot, atol=1e-8)
