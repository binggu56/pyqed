import numpy as np

from pyqed.letta import ULETTA, UniformLETTA
from pyqed.mps import UniformMPS


def _spin_half_heisenberg_bond():
    sx = 0.5 * np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sy = 0.5 * np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    sz = 0.5 * np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    return np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz)


def test_uniform_letta_export_alias():
    assert ULETTA is UniformLETTA


def test_uniform_letta_pair_product_ferromagnetic_energy():
    h = _spin_half_heisenberg_bond()
    weights = np.array([[1.0, 0.0], [0.0, 0.0]])

    psi = UniformLETTA.pair_product(weights)

    np.testing.assert_allclose(psi.one_site_density_matrix(), np.diag([1.0, 0.0]), atol=1.0e-12)
    np.testing.assert_allclose(psi.energy_density(h), 0.25, atol=1.0e-12)


def test_uniform_letta_pair_product_neel_energy():
    h = _spin_half_heisenberg_bond()
    weights = np.array([[0.0, 1.0], [1.0, 0.0]])

    psi = UniformLETTA.pair_product(weights)

    np.testing.assert_allclose(psi.one_site_density_matrix(), 0.5 * np.eye(2), atol=1.0e-12)
    np.testing.assert_allclose(
        psi.two_site_density_matrix().reshape(4, 4),
        np.diag([0.0, 0.5, 0.5, 0.0]),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(psi.energy_density(h), -0.25, atol=1.0e-12)


def test_uniform_letta_effective_transfer_matches_uniform_mps():
    h = _spin_half_heisenberg_bond()
    letta = UniformLETTA.random(physical_dim=2, bond_dim=2, seed=7, dtype=float)
    mps = UniformMPS(letta.effective_tensors()[0]).normalize_transfer()

    np.testing.assert_allclose(letta.energy_density(h), mps.energy_density(h), atol=1.0e-12)
    np.testing.assert_allclose(letta.one_site_density_matrix(), mps.one_site_density_matrix(), atol=1.0e-12)
    np.testing.assert_allclose(letta.two_site_density_matrix(), mps.two_site_density_matrix(), atol=1.0e-12)


def test_uniform_letta_embeds_one_site_uniform_mps_exactly():
    h = _spin_half_heisenberg_bond()
    mps = UniformMPS.random(physical_dim=2, bond_dim=3, seed=13, canonicalize=None).normalize_transfer()

    letta = UniformLETTA.from_uniform_mps(mps)

    assert letta.bond_dim == mps.bond_dim
    assert letta.unit_cell_size == mps.unit_cell_size
    np.testing.assert_allclose(letta.energy_density(h), mps.energy_density(h), atol=1.0e-12)
    np.testing.assert_allclose(letta.one_site_density_matrix(), mps.one_site_density_matrix(), atol=1.0e-12)
    np.testing.assert_allclose(letta.two_site_density_matrix(), mps.two_site_density_matrix(), atol=1.0e-12)


def test_uniform_letta_embeds_unit_cell_uniform_mps_exactly():
    h = _spin_half_heisenberg_bond()
    mps = UniformMPS.random(
        physical_dim=2,
        bond_dim=2,
        unit_cell=2,
        seed=14,
        canonicalize=None,
    ).normalize_transfer()

    letta = UniformLETTA.from_uniform_mps(mps)

    assert letta.bond_dim == mps.bond_dim
    assert letta.unit_cell_size == mps.unit_cell_size
    np.testing.assert_allclose(letta.energy_density(h), mps.energy_density(h), atol=1.0e-12)
    np.testing.assert_allclose(letta.energy_density(h, site=0), mps.energy_density(h, site=0), atol=1.0e-12)
    np.testing.assert_allclose(letta.energy_density(h, site=1), mps.energy_density(h, site=1), atol=1.0e-12)


def test_uniform_letta_optimizer_accepts_uniform_mps_seed_and_keeps_baseline():
    h = _spin_half_heisenberg_bond()
    seed = UniformMPS.random(
        physical_dim=2,
        bond_dim=2,
        unit_cell=2,
        seed=15,
        canonicalize=None,
    ).normalize_transfer()
    seed_energy = float(np.real(seed.energy_density(h)))

    optimized = UniformLETTA.optimize_nearest_neighbor(
        h,
        initial=seed,
        real=True,
        maxiter=0,
        gtol=1.0e-6,
    )

    assert optimized.unit_cell_size == seed.unit_cell_size
    assert optimized.bond_dim == seed.bond_dim
    assert optimized.energy <= seed_energy + 1.0e-12


def test_uniform_letta_direct_optimizer_lowers_heisenberg_energy():
    h = _spin_half_heisenberg_bond()

    optimized = UniformLETTA.optimize_nearest_neighbor(
        h,
        bond_dim=2,
        seed=80,
        restarts=1,
        real=True,
        maxiter=120,
        gtol=1.0e-7,
    )

    assert isinstance(optimized, UniformLETTA)
    assert optimized.algorithm == "dense-bfgs-uletta"
    assert optimized.energy < -0.39
    assert optimized.nfev > 0
