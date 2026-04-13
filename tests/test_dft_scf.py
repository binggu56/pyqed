import numpy as np
import pytest

from pyqed.qchem import Molecule
from pyqed.qchem.dft import AOGrid, RKS
from pyqed.qchem.dft.scf import ks_energy, run_rks
from pyqed.qchem.dft.xc import eval_xc
from pyqed.qchem._libxc import has_libxc_backend


def test_ks_energy_uses_direct_ks_expression():
    dm = np.array([[1.2, 0.1], [0.1, 0.8]])
    hcore = np.array([[-1.1, -0.2], [-0.2, -0.7]])
    j = np.array([[0.9, 0.05], [0.05, 0.6]])
    vxc = np.array([[-0.4, -0.02], [-0.02, -0.3]])
    exc = -0.55
    e_nuc = 0.72

    expected = (
        np.einsum('ij,ji->', hcore, dm).real
        + 0.5 * np.einsum('ij,ji->', j, dm).real
        + exc
        + e_nuc
    )

    energy = ks_energy(dm, hcore, j, exc, vxc, e_nuc=e_nuc)
    np.testing.assert_allclose(energy, expected)


def test_xc_aliases_match_pyscf_style_conventions():
    rho = np.array([0.05, 0.2, 0.8])

    eps_lda, v_lda = eval_xc(rho, xc='lda')
    eps_x, v_x = eval_xc(rho, xc='lda_x')
    np.testing.assert_allclose(eps_lda, eps_x)
    np.testing.assert_allclose(v_lda, v_x)

    eps_svwn, v_svwn = eval_xc(rho, xc='svwn')
    eps_ldavwn, v_ldavwn = eval_xc(rho, xc='lda,vwn')
    np.testing.assert_allclose(eps_svwn, eps_ldavwn)
    np.testing.assert_allclose(v_svwn, v_ldavwn)


def test_rks_builds_default_atom_centered_grid():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = RKS(mol, xc='lda')
    mf.max_cycle = 80
    mf.conv_tol = 1e-9
    mf.run()

    assert mf.grid is not None
    assert mf.grid.ngrids > 0
    assert mf.converged


@pytest.mark.skipif(not has_libxc_backend(), reason='libxc backend is unavailable')
def test_rks_b3lyp_smoke():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = RKS(mol, xc='b3lyp')
    mf.max_cycle = 80
    mf.conv_tol = 1e-9
    mf.run()

    assert mf.converged
    assert mf.k is not None
    assert mf.grid.ao_grad is not None
    assert np.isfinite(mf.e_tot)
    assert abs(np.einsum('ij,ji->', mol.overlap, mf.dm).real - mol.nelec) < 1e-4


@pytest.mark.skipif(not has_libxc_backend(), reason='libxc backend is unavailable')
def test_rks_b3lyp_rebuilds_default_grid_when_custom_grid_lacks_coords():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='gbasis')

    bare_grid = AOGrid(ao=np.eye(mol.nao), weights=np.ones(mol.nao))
    mf = RKS(mol, grid=bare_grid, xc='b3lyp')

    assert mf.grid is not bare_grid
    assert mf.grid.coords is not None
    assert mf.grid.ao_grad is not None


@pytest.mark.skipif(not has_libxc_backend(), reason='libxc backend is unavailable')
def test_run_rks_b3lyp_rebuilds_default_grid_when_custom_grid_lacks_coords():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='gbasis')

    bare_grid = AOGrid(ao=np.eye(mol.nao), weights=np.ones(mol.nao))
    out = run_rks(mol, bare_grid, xc='b3lyp', max_cycle=40, conv_tol=1e-8)

    assert out['converged']
    assert out['grid'] is not bare_grid
    assert out['grid'].coords is not None
    assert out['grid'].ao_grad is not None
