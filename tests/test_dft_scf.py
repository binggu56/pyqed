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


def test_topyscf_preserves_geometry_for_angstrom_input():
    pytest.importorskip('pyscf')

    mol = Molecule(
        atom='H 0 0 0; H 0 0 0.74',
        unit='angstrom',
        basis='sto-3g',
    )
    pmol = mol.topyscf()

    np.testing.assert_allclose(pmol.atom_coords(), mol.atom_coords(), atol=1e-12)


def test_rks_builds_default_atom_centered_grid():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build()

    mf = RKS(mol, xc='lda')
    mf.max_cycle = 80
    mf.conv_tol = 1e-9
    mf.run()

    assert mf.grid is not None
    assert mf.grid.ngrids > 0
    assert mf.converged


def test_native_aogrid_matches_pyscf_for_values_gradients_and_hessians():
    pyscf_numint = pytest.importorskip('pyscf.dft.numint')

    atom = 'O 0 0 0; H 0 1.4 0; H 0 -1.4 0'
    coords = np.array(
        [
            [0.1, 0.2, 0.3],
            [1.0, -0.4, 0.2],
            [0.0, 0.0, 0.7],
            [-0.3, 0.5, -0.2],
        ],
        dtype=float,
    )
    weights = np.ones(coords.shape[0])

    native_mol = Molecule(atom=atom, unit='bohr', basis='sto-3g')
    native_mol.build()
    native_grid = AOGrid.from_molecule(
        native_mol,
        coords,
        weights,
        with_grad=True,
        with_hess=True,
    )

    pyscf_mol = native_mol.topyscf().build()
    values = pyscf_numint.eval_ao(pyscf_mol, coords, deriv=2)
    pyscf_hess = np.empty_like(native_grid.ao_hess)
    for component, (i, j) in enumerate(
        ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)), start=4
    ):
        pyscf_hess[i, j] = values[component]
        pyscf_hess[j, i] = values[component]

    np.testing.assert_allclose(native_grid.ao, values[0], rtol=2e-6, atol=1e-10)
    np.testing.assert_allclose(native_grid.ao_grad, values[1:4], rtol=2e-6, atol=1e-10)
    np.testing.assert_allclose(native_grid.ao_hess, pyscf_hess, rtol=2e-6, atol=1e-10)


@pytest.mark.skipif(not has_libxc_backend(), reason='libxc backend is unavailable')
def test_rks_pbe_smoke():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build()

    mf = RKS(mol, xc='pbe')
    mf.max_cycle = 80
    mf.conv_tol = 1e-9
    mf.run()

    assert mf.converged
    assert mf.grid.ao_grad is not None
    assert np.isfinite(mf.e_tot)
    assert abs(np.einsum('ij,ji->', mol.overlap, mf.dm).real - mol.nelec) < 1e-4


@pytest.mark.skipif(not has_libxc_backend(), reason='libxc backend is unavailable')
def test_rks_b3lyp_smoke():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build()

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
    mol.build()

    bare_grid = AOGrid(ao=np.eye(mol.nao), weights=np.ones(mol.nao))
    mf = RKS(mol, grid=bare_grid, xc='b3lyp')

    assert mf.grid is not bare_grid
    assert mf.grid.coords is not None
    assert mf.grid.ao_grad is not None


@pytest.mark.skipif(not has_libxc_backend(), reason='libxc backend is unavailable')
def test_run_rks_b3lyp_rebuilds_default_grid_when_custom_grid_lacks_coords():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build()

    bare_grid = AOGrid(ao=np.eye(mol.nao), weights=np.ones(mol.nao))
    out = run_rks(mol, bare_grid, xc='b3lyp', max_cycle=40, conv_tol=1e-8)

    assert out['converged']
    assert out['grid'] is not bare_grid
    assert out['grid'].coords is not None
    assert out['grid'].ao_grad is not None


def test_rks_geometry_optimization_lowers_h2_energy():
    mol = Molecule(atom='H 0 0 -1.1; H 0 0 1.1', unit='bohr', basis='sto-3g')
    mol.build()

    grid = AOGrid.atom_centered(mol, n_radial=8, n_angular=14, with_grad=False)
    mf = RKS(mol, grid=grid, xc='svwn')
    mf.max_cycle = 80
    mf.conv_tol = 1e-9
    mf.run()
    e0 = mf.e_tot
    r0 = np.linalg.norm(mol.atom_coords()[1] - mol.atom_coords()[0])

    opt = mf.optimize_geometry(maxiter=20, gtol=1e-3)
    r1 = np.linalg.norm(opt.coords[1] - opt.coords[0])

    assert np.isfinite(opt.energy)
    assert opt.energy < e0
    assert r1 < r0
    assert np.linalg.norm(opt.gradient) < 1e-2
    assert opt.hessian().shape == (6, 6)
    assert opt.hessian(inverse=True).shape == (6, 6)
    vib = opt.vibrational_analysis()
    assert vib['freq_cm1'].shape == (1,)
    assert vib['modes'].shape == (1, 2, 3)
    assert opt.frequencies().shape == (1,)


def test_rks_hessian_runs_and_exposes_frequencies():
    mol = Molecule(atom='H 0 0 -0.8; H 0 0 0.8', unit='bohr', basis='sto-3g')
    mol.build()

    grid = AOGrid.atom_centered(mol, n_radial=8, n_angular=14, with_grad=False)
    mf = RKS(mol, grid=grid, xc='svwn')
    mf.max_cycle = 80
    mf.conv_tol = 1e-9
    mf.run()

    hobj = mf.Hessian()
    hess = hobj.run(step=2e-3)
    vib = hobj.vibrational_analysis()

    assert hess.shape == (6, 6)
    np.testing.assert_allclose(hess, hess.T, atol=1e-8)
    assert vib['freq_cm1'].shape == (1,)
    assert vib['modes'].shape == (1, 2, 3)
    assert hobj.frequencies().shape == (1,)


def test_rks_geometry_optimization_rejects_unknown_backend():
    mol = Molecule(atom='H 0 0 -1.1; H 0 0 1.1', unit='bohr', basis='sto-3g')
    mol.build()

    mf = RKS(mol, xc='svwn')
    with pytest.raises(ValueError, match='backend must be either'):
        mf.optimize_geometry(backend='nope')


def test_rks_geometry_optimization_geometric_requires_dependency():
    pytest.importorskip('pyscf')

    mol = Molecule(atom='H 0 0 -1.1; H 0 0 1.1', unit='bohr', basis='sto-3g')
    mol.build()

    mf = RKS(mol, xc='svwn')

    try:
        import geometric  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match='geomeTRIC is not installed'):
            mf.optimize_geometry(backend='geometric', maxiter=3)
    else:
        opt = mf.optimize_geometry(backend='geometric', maxiter=10)
        assert np.isfinite(opt.energy)
        assert opt.backend == 'geometric'
        assert opt.hessian().shape == (6, 6)
