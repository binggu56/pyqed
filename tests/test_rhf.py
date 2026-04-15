import numpy as np
import pytest

from pyqed.qchem import Molecule
from pyqed.qchem.hf import RHF
from pyqed.qchem.hf.rhf import get_or_build_low_rank_eri_factors, get_jk


def test_rhf_density_fit_matches_conventional_energy_and_jk():
    scf = pytest.importorskip('pyscf.scf')

    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis-pyscf')

    mf_df = RHF(mol).run(density_fit=True)
    mf_direct = RHF(mol).run()

    pmol = mol.topyscf()
    pmol.build(verbose=0)
    pmf_df = scf.RHF(pmol).density_fit()
    pmf_df.conv_tol = 1e-8
    pmf_df.kernel()

    np.testing.assert_allclose(mf_df.e_tot, pmf_df.e_tot, atol=1e-10)
    np.testing.assert_allclose(mf_df.mo_energy, pmf_df.mo_energy, atol=1e-6)
    np.testing.assert_allclose(mf_df.dm, pmf_df.make_rdm1(), atol=1e-5)
    assert abs(mf_df.e_tot - mf_direct.e_tot) < 1e-3

    vj_df, vk_df = mf_df.get_jk()
    vj_ref, vk_ref = pmf_df.get_jk(dm=mf_df.dm)
    np.testing.assert_allclose(vj_df, vj_ref, atol=1e-10)
    np.testing.assert_allclose(vk_df, vk_ref, atol=1e-10)

    assert mf_df.density_fit
    assert mf_df._pyscf_mf is not None


def test_cholesky_jk_factors_reproduce_dense_jk_for_tight_tolerance():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis-pyscf')

    mf = RHF(mol).run()
    factors = get_or_build_low_rank_eri_factors(mol, tol=1e-12)

    vj_dense, vk_dense = get_jk(mol, mf.dm)
    vj_lr, vk_lr = get_jk(mol, mf.dm, eri_factors=factors)

    np.testing.assert_allclose(vj_lr, vj_dense, atol=1e-8)
    np.testing.assert_allclose(vk_lr, vk_dense, atol=1e-8)


def test_rhf_cholesky_jk_matches_direct_energy():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis-pyscf')

    mf_direct = RHF(mol).run()
    mf_lr = RHF(mol).run(cholesky_jk=True, cholesky_tol=1e-10)

    np.testing.assert_allclose(mf_lr.e_tot, mf_direct.e_tot, atol=1e-7)
    np.testing.assert_allclose(mf_lr.dm, mf_direct.dm, atol=1e-6)

    assert mf_lr.cholesky_jk
    assert mf_lr.cholesky_tol == 1e-10
    assert mf_lr.low_rank_jk
    assert mf_lr.eri_factors is not None
    assert mf_lr.eri_factors.shape[0] <= mol.nao * mol.nao


def test_low_rank_aliases_match_cholesky_options():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis-pyscf')

    mf_alias = RHF(mol).run(low_rank_jk=True, low_rank_tol=1e-10)

    assert mf_alias.cholesky_jk
    assert mf_alias.low_rank_jk
    assert mf_alias.cholesky_tol == mf_alias.low_rank_tol == 1e-10


def test_low_rank_factor_cache_reuses_exact_geometry_after_rebuild():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis-pyscf')

    factors_first = get_or_build_low_rank_eri_factors(mol, tol=1e-10)
    assert mol._low_rank_eri_last_info['mode'] == 'cold'

    mol.build(driver='gbasis-pyscf')
    factors_second = get_or_build_low_rank_eri_factors(mol, tol=1e-10)

    assert factors_second is factors_first
    assert mol._low_rank_eri_last_info['mode'] == 'exact'


def test_low_rank_factor_cache_warm_starts_after_geometry_change():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis-pyscf')

    factors_first = get_or_build_low_rank_eri_factors(mol, tol=1e-10)
    assert mol._low_rank_eri_last_info['mode'] == 'cold'

    coords = mol.atom_coords().copy()
    coords[1, 2] += 0.1 / 0.52917721092
    mol.set_geom(coords)
    mol.build(driver='gbasis-pyscf')

    factors_second = get_or_build_low_rank_eri_factors(mol, tol=1e-10)
    assert factors_second is not factors_first
    assert mol._low_rank_eri_last_info['mode'] == 'warm'

    mf = RHF(mol).run()
    vj_dense, vk_dense = get_jk(mol, mf.dm)
    vj_lr, vk_lr = get_jk(mol, mf.dm, eri_factors=factors_second)
    np.testing.assert_allclose(vj_lr, vj_dense, atol=1e-7)
    np.testing.assert_allclose(vk_lr, vk_dense, atol=1e-7)


def test_low_rank_scanner_reuses_density_and_factor_history():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis-pyscf')

    mf0 = RHF(mol).run(cholesky_jk=True, cholesky_tol=1e-10)
    scanner = mf0.as_scanner()

    coords = mol.atom_coords().copy()
    coords[1, 2] += 0.05 / 0.52917721092
    e_scan = scanner(coords)

    mol_ref = Molecule(atom='Li 0 0 0; H 0 0 1.65', unit='angstrom', basis='sto-3g')
    mol_ref.build(driver='gbasis-pyscf')
    e_ref = RHF(mol_ref).run(cholesky_jk=True, cholesky_tol=1e-10).e_tot

    np.testing.assert_allclose(e_scan, e_ref, atol=1e-8)
    assert scanner.mf.dm is not None
    assert scanner.mf.eri_factors is not None
    assert mol._low_rank_eri_last_info['mode'] in {'warm', 'exact'}
