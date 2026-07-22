import numpy as np

import pyqed.mps as mps
from pyqed.mps import (
    InfiniteDMRG,
    UniformMPS,
    factorize_nearest_neighbor_hamiltonian,
    idmrg_nearest_neighbor,
    iDMRG,
)


def _spin_half_heisenberg_bond():
    sx = 0.5 * np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sy = 0.5 * np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    sz = 0.5 * np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    return np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz)


def test_idmrg_uses_lowercase_infinite_prefix_without_legacy_alias():
    assert hasattr(mps, "iDMRG")
    assert hasattr(mps, "InfiniteDMRG")
    assert not hasattr(mps, "IDMRG")


def test_nearest_neighbor_channel_factorization_reconstructs_operator():
    rng = np.random.default_rng(8)
    h = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    h = 0.5 * (h + h.conj().T)

    terms = factorize_nearest_neighbor_hamiltonian(h)

    np.testing.assert_allclose(terms.reconstruct().reshape(4, 4), h, atol=1.0e-12)
    assert terms.nterms > 0


def test_idmrg_solves_exact_product_projector_fixed_point():
    h = -np.diag([1.0, 0.0, 0.0, 0.0])

    result = idmrg_nearest_neighbor(h, bond_dim=1, maxiter=4, tol=1.0e-12, solver="dense")

    assert isinstance(result, iDMRG)
    assert isinstance(result, InfiniteDMRG)
    assert result.success is True
    assert result.state is not None
    assert result.state.algorithm == "idmrg"
    assert result.state.unit_cell_size == 2
    np.testing.assert_allclose(result.center_bond_energy, -1.0, atol=1.0e-12)
    np.testing.assert_allclose(result.energy_density, -1.0, atol=1.0e-12)
    np.testing.assert_allclose(result.state.energy, -1.0, atol=1.0e-12)


def test_idmrg_keeps_persistent_renormalized_blocks_and_history():
    h = -np.diag([1.0, 0.0, 0.0, 0.0])
    solver = iDMRG(h, bond_dim=2, maxiter=3, solver="dense")

    result = solver.run()

    assert result is solver
    assert result.left_block.length == 1 + len(result.history)
    assert result.right_block.length == 1 + len(result.history)
    assert result.left_block.hamiltonian.shape == result.right_block.hamiltonian.shape
    assert len(result.left_block.edge_ops) == solver.terms.nterms
    assert all(step.kept_dim <= 2 for step in result.history)


def test_idmrg_returns_nontrivial_uniform_mps_for_antiferromagnetic_heisenberg():
    h = _spin_half_heisenberg_bond()

    result = idmrg_nearest_neighbor(h, bond_dim=4, maxiter=6, tol=1.0e-9, solver="dense")

    assert result.state is None
    assert result.metadata["state_export"] == "omitted_energy_mismatch"
    assert np.isfinite(result.metadata["candidate_uniform_state_energy_per_site"])
    assert result.metadata["state_energy_mismatch"] > result.metadata["state_energy_tol"]
    assert result.energy_density < -0.42
    assert result.history[-1].truncation_error < 2.0e-3


def test_idmrg_exports_state_when_uniform_candidate_matches_growth_energy():
    h = -np.diag([1.0, 0.0, 0.0, 0.0])

    result = iDMRG(h, bond_dim=1, maxiter=4, tol=1.0e-12, solver="dense").run()

    assert isinstance(result.state, UniformMPS)
    assert result.metadata["state_export"] == "raw"
    np.testing.assert_allclose(result.metadata["uniform_state_energy_per_site"], result.state.energy, atol=1.0e-12)
