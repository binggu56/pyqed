import numpy as np

import pyqed.narg as narg
from pyqed.mps import iDMRG
from pyqed.narg import InfiniteNARG, iNARG, inarg_nearest_neighbor


def _spin_half_heisenberg_bond():
    sx = 0.5 * np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sy = 0.5 * np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    sz = 0.5 * np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    return np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz)


def _open_heisenberg_hamiltonian(nsites):
    sx = 0.5 * np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sy = 0.5 * np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    sz = 0.5 * np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    eye = np.eye(2, dtype=complex)
    hamiltonian = np.zeros((2**nsites, 2**nsites), dtype=complex)
    for site in range(nsites - 1):
        for op in (sx, sy, sz):
            factors = [eye] * nsites
            factors[site] = op
            factors[site + 1] = op
            term = factors[0]
            for factor in factors[1:]:
                term = np.kron(term, factor)
            hamiltonian += term
    return 0.5 * (hamiltonian + hamiltonian.T.conj())


def test_inarg_uses_lowercase_infinite_prefix_without_legacy_alias():
    assert hasattr(narg, "iNARG")
    assert hasattr(narg, "InfiniteNARG")
    assert not hasattr(narg, "INARG")


def test_inarg_solves_product_projector_fixed_point():
    hamiltonian = -np.diag([1.0, 0.0, 0.0, 0.0])
    solver = iNARG(hamiltonian, bond_dim=1, maxiter=5, tol=1.0e-12)

    result = solver.run()

    assert result is solver
    assert isinstance(result, InfiniteNARG)
    assert not isinstance(result, iDMRG)
    assert result.algorithm == "inarg"
    assert result.success is True
    assert result.fixed_layer.shape == (2, 1, 2)
    assert result.transition_tensor.shape == (1, 2, 1, 2)
    np.testing.assert_allclose(result.energy_density, -1.0, atol=1.0e-12)
    np.testing.assert_allclose(result.history[-1].growth_energy, -1.0, atol=1.0e-12)


def test_inarg_nearest_neighbor_returns_uniform_fixed_layer_for_heisenberg():
    hamiltonian = _spin_half_heisenberg_bond()

    result = inarg_nearest_neighbor(
        hamiltonian,
        bond_dim=4,
        maxiter=8,
        tol=1.0e-9,
    )

    assert isinstance(result, iNARG)
    assert result.fixed_layer.shape == (8, 4, 2)
    assert result.transition_tensor.shape == (4, 2, 4, 2)
    assert result.energy_density < -0.3
    assert result.history[-1].kept_dim == 4


def test_inarg_full_rank_growth_matches_exact_open_heisenberg_chain():
    hamiltonian = _spin_half_heisenberg_bond()
    result = iNARG(hamiltonian, bond_dim=32, maxiter=5, tol=0.0).run()

    for step in result.history:
        exact = np.linalg.eigvalsh(_open_heisenberg_hamiltonian(step.length))[0]
        np.testing.assert_allclose(step.energy, exact, atol=1.0e-10)


def test_two_site_inarg_full_rank_growth_matches_exact_open_heisenberg_chain():
    hamiltonian = _spin_half_heisenberg_bond()
    result = iNARG(hamiltonian, bond_dim=32, growth_sites=2, maxiter=3, tol=0.0).run()

    assert result.fixed_layer.shape == (32, 32, 2, 2)
    assert result.transition_tensor.shape == (16, 2, 32, 2, 2)
    for step in result.history:
        exact = np.linalg.eigvalsh(_open_heisenberg_hamiltonian(step.length))[0]
        np.testing.assert_allclose(step.energy, exact, atol=1.0e-10)
