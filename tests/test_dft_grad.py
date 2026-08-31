import numpy as np
import pytest

from pyqed.qchem import Molecule
from pyqed.qchem.dft import AOGrid, RKS, cartesian_box_grid
from pyqed.qchem._libxc import has_libxc_backend


def _build_h2_frozen_grid(bond_length, xc='svwn', ngrid=9):
    half_bond = bond_length / 2.0
    mol = Molecule(
        atom=f'H 0 0 {-half_bond}; H 0 0 {half_bond}',
        unit='bohr',
        basis='sto-3g',
    )
    mol.build()

    coords, weights = cartesian_box_grid((-4, 4), (-4, 4), (-4, 4), ngrid)
    grid = AOGrid.from_molecule(mol, coords, weights, with_grad=True)

    mf = RKS(mol, grid=grid, xc=xc)
    mf.max_cycle = 80
    mf.conv_tol = 1e-10
    mf.run()
    return mf


def _build_h2_atom_centered_grid(bond_length, xc='svwn', n_radial=8, n_angular=14):
    half_bond = bond_length / 2.0
    mol = Molecule(
        atom=f'H 0 0 {-half_bond}; H 0 0 {half_bond}',
        unit='bohr',
        basis='sto-3g',
    )
    mol.build()

    grid = AOGrid.atom_centered(
        mol,
        n_radial=n_radial,
        n_angular=n_angular,
        with_grad=True,
    )

    mf = RKS(mol, grid=grid, xc=xc)
    mf.max_cycle = 80
    mf.conv_tol = 1e-10
    mf.run()
    return mf


def test_rks_svwn_frozen_grid_gradient_matches_finite_difference():
    pytest.importorskip('pyscf')

    bond = 1.3983973321781458  # 0.74 angstrom in bohr
    step = 1e-3

    mf = _build_h2_frozen_grid(bond, xc='svwn')
    grad = mf.nuc_grad_method().run()

    mf_plus = _build_h2_frozen_grid(bond + step, xc='svwn')
    mf_minus = _build_h2_frozen_grid(bond - step, xc='svwn')
    fd = (mf_plus.e_tot - mf_minus.e_tot) / (2.0 * step)

    assert mf.converged
    projected = 0.5 * (grad[1, 2] - grad[0, 2])

    np.testing.assert_allclose(grad.sum(axis=0), 0.0, atol=1e-9)
    np.testing.assert_allclose(projected, fd, atol=1e-6)


def test_rks_svwn_atom_centered_gradient_matches_finite_difference():
    pytest.importorskip('pyscf')

    bond = 1.3983973321781458  # 0.74 angstrom in bohr
    step = 1e-3

    mf = _build_h2_atom_centered_grid(bond, xc='svwn')
    grad = mf.nuc_grad_method().run()

    mf_plus = _build_h2_atom_centered_grid(bond + step, xc='svwn')
    mf_minus = _build_h2_atom_centered_grid(bond - step, xc='svwn')
    fd = (mf_plus.e_tot - mf_minus.e_tot) / (2.0 * step)

    assert mf.converged
    projected = 0.5 * (grad[1, 2] - grad[0, 2])

    np.testing.assert_allclose(grad.sum(axis=0), 0.0, atol=1e-8)
    np.testing.assert_allclose(projected, fd, atol=2e-6)


def test_rks_svwn_atom_centered_matches_pyscf():
    pytest.importorskip('pyscf')
    from pyscf import dft

    bond = 1.3983973321781458  # 0.74 angstrom in bohr
    mf = _build_h2_atom_centered_grid(bond, xc='svwn', n_radial=50, n_angular=110)
    grad = mf.nuc_grad_method().run()

    pmf = dft.RKS(mf.mol.topyscf())
    pmf.xc = 'svwn'
    pmf.conv_tol = 1e-10
    pmf.grids.atom_grid = {'H': (50, 110)}
    pmf.kernel()
    pgrad = pmf.nuc_grad_method().kernel()

    assert mf.converged
    assert pmf.converged
    np.testing.assert_allclose(mf.e_tot, pmf.e_tot, atol=1e-6)
    np.testing.assert_allclose(grad, pgrad, atol=1e-5)


@pytest.mark.skipif(not has_libxc_backend(), reason='libxc backend is unavailable')
def test_rks_pbe_frozen_grid_gradient_matches_finite_difference():
    pytest.importorskip('pyscf')

    bond = 1.3983973321781458  # 0.74 angstrom in bohr
    step = 1e-3

    mf = _build_h2_frozen_grid(bond, xc='pbe')
    grad = mf.nuc_grad_method().run()

    mf_plus = _build_h2_frozen_grid(bond + step, xc='pbe')
    mf_minus = _build_h2_frozen_grid(bond - step, xc='pbe')

    fd = (mf_plus.e_tot - mf_minus.e_tot) / (2.0 * step)
    projected = 0.5 * (grad[1, 2] - grad[0, 2])

    assert mf.converged
    np.testing.assert_allclose(grad.sum(axis=0), 0.0, atol=1e-8)
    np.testing.assert_allclose(projected, fd, atol=2e-6)


@pytest.mark.skipif(not has_libxc_backend(), reason='libxc backend is unavailable')
def test_rks_pbe_atom_centered_gradient_matches_finite_difference():
    pytest.importorskip('pyscf')

    bond = 1.3983973321781458  # 0.74 angstrom in bohr
    step = 1e-3

    mf = _build_h2_atom_centered_grid(bond, xc='pbe')
    grad = mf.nuc_grad_method().run()

    mf_plus = _build_h2_atom_centered_grid(bond + step, xc='pbe')
    mf_minus = _build_h2_atom_centered_grid(bond - step, xc='pbe')

    fd = (mf_plus.e_tot - mf_minus.e_tot) / (2.0 * step)
    projected = 0.5 * (grad[1, 2] - grad[0, 2])

    assert mf.converged
    np.testing.assert_allclose(grad.sum(axis=0), 0.0, atol=1e-7)
    np.testing.assert_allclose(projected, fd, atol=5e-6)
