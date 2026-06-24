import numpy as np

from pyqed.mps import UMPS, UniformMPS


def _spin_half_heisenberg_bond():
    sx = 0.5 * np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sy = 0.5 * np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    sz = 0.5 * np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    return np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz)


def test_uniform_mps_product_state_one_and_two_site_expectations():
    psi = UniformMPS.product_state([1.0, 0.0])

    z = np.array([[1.0, 0.0], [0.0, -1.0]])
    x = np.array([[0.0, 1.0], [1.0, 0.0]])

    np.testing.assert_allclose(psi.one_site_density_matrix(), np.diag([1.0, 0.0]))
    np.testing.assert_allclose(psi.expectation_one_site(z), 1.0)
    np.testing.assert_allclose(psi.expectation_one_site(x), 0.0)
    np.testing.assert_allclose(psi.expectation_two_site(np.kron(z, z)), 1.0)
    assert psi.entanglement_entropy() == 0.0
    assert psi.correlation_length() == 0.0


def test_uniform_mps_product_state_two_site_operator_ordering():
    vector = np.array([0.6, 0.8j], dtype=complex)
    psi = UniformMPS.product_state(vector)

    rng = np.random.default_rng(4)
    h = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    h = 0.5 * (h + h.conj().T)

    pair = np.kron(vector / np.linalg.norm(vector), vector / np.linalg.norm(vector))
    expected = np.vdot(pair, h @ pair)
    np.testing.assert_allclose(psi.expectation_two_site(h), expected, atol=1.0e-12)


def test_uniform_mps_random_left_and_right_canonical_gauges():
    raw = UniformMPS.random(physical_dim=2, bond_dim=3, seed=3, canonicalize=None)

    left = raw.left_canonical()
    right = raw.right_canonical()

    assert left.canonical_errors()["left"] < 1.0e-10
    assert right.canonical_errors()["right"] < 1.0e-10
    np.testing.assert_allclose(abs(left.dominant_transfer_eigenvalue()), 1.0, atol=1.0e-10)
    np.testing.assert_allclose(abs(right.dominant_transfer_eigenvalue()), 1.0, atol=1.0e-10)


def test_uniform_mps_mixed_canonical_consistency_and_density_normalization():
    psi = UniformMPS.random(physical_dim=2, bond_dim=3, seed=9)
    canonical = psi.mixed_canonical()

    assert canonical.center_error() < 1.0e-10
    np.testing.assert_allclose(np.linalg.norm(canonical.singular_values()), 1.0)
    np.testing.assert_allclose(np.trace(psi.one_site_density_matrix()), 1.0, atol=1.0e-12)

    rho2 = psi.two_site_density_matrix().reshape(4, 4)
    np.testing.assert_allclose(np.trace(rho2), 1.0, atol=1.0e-12)
    np.testing.assert_allclose(psi.expectation_one_site(np.eye(2)), 1.0, atol=1.0e-12)
    np.testing.assert_allclose(psi.expectation_two_site(np.eye(4)), 1.0, atol=1.0e-12)


def test_uniform_mps_export_alias():
    assert UMPS is UniformMPS


def test_uniform_mps_heisenberg_product_and_alternating_energies():
    h = _spin_half_heisenberg_bond()

    ferromagnet = UniformMPS.product_state([1.0, 0.0])
    np.testing.assert_allclose(ferromagnet.energy_density(h), 0.25, atol=1.0e-12)

    alternating_tensor = np.zeros((2, 2, 2), dtype=complex)
    alternating_tensor[0] = np.array([[0.0, 1.0], [0.0, 0.0]])
    alternating_tensor[1] = np.array([[0.0, 0.0], [1.0, 0.0]])
    alternating = UniformMPS(alternating_tensor).normalize_transfer()

    np.testing.assert_allclose(alternating.energy_density(h), -0.25, atol=1.0e-12)
    np.testing.assert_allclose(
        alternating.two_site_density_matrix().reshape(4, 4),
        np.diag([0.0, 0.5, 0.5, 0.0]),
        atol=1.0e-12,
    )


def test_uniform_mps_two_site_unit_cell_product_densities_and_energy():
    h = _spin_half_heisenberg_bond()
    neel = UniformMPS.product_state_unit_cell([[1.0, 0.0], [0.0, 1.0]])

    assert neel.unit_cell_size == 2
    np.testing.assert_allclose(neel.one_site_density_matrix(site=0), np.diag([1.0, 0.0]))
    np.testing.assert_allclose(neel.one_site_density_matrix(site=1), np.diag([0.0, 1.0]))
    np.testing.assert_allclose(neel.one_site_density_matrix(), 0.5 * np.eye(2))
    np.testing.assert_allclose(neel.energy_density(h, site=0), -0.25, atol=1.0e-12)
    np.testing.assert_allclose(neel.energy_density(h, site=1), -0.25, atol=1.0e-12)
    np.testing.assert_allclose(neel.energy_density(h), -0.25, atol=1.0e-12)
    np.testing.assert_allclose(np.trace(neel.two_site_density_matrix().reshape(4, 4)), 1.0, atol=1.0e-12)


def test_uniform_mps_two_site_unit_cell_optimizer_lowers_heisenberg_energy():
    h = _spin_half_heisenberg_bond()

    optimized = UniformMPS.optimize_nearest_neighbor_unit_cell(
        h,
        unit_cell=2,
        bond_dim=2,
        seed=80,
        restarts=1,
        real=True,
        maxiter=160,
        gtol=1.0e-7,
    )

    assert isinstance(optimized, UniformMPS)
    assert optimized.unit_cell_size == 2
    assert optimized.algorithm == "dense-bfgs-unit-cell"
    assert optimized.energy < -0.4
    assert optimized.nfev > 0


def test_uniform_mps_optimizes_heisenberg_below_alternating_energy():
    h = _spin_half_heisenberg_bond()

    optimized = UniformMPS.optimize_nearest_neighbor(
        h,
        bond_dim=3,
        seed=31,
        restarts=1,
        real=True,
        maxiter=160,
        gtol=1.0e-7,
    )

    assert isinstance(optimized, UniformMPS)
    assert optimized.energy < -0.34
    assert optimized.nfev > 0
    assert optimized.canonical_errors()["left"] < 1.0e-10


def test_uniform_mps_vumps_solves_ferromagnetic_heisenberg_product_ground_state():
    h = -_spin_half_heisenberg_bond()

    psi = UniformMPS.vumps_nearest_neighbor(
        h,
        bond_dim=1,
        seed=11,
        maxiter=20,
        tol=1.0e-10,
        real=True,
    )

    assert isinstance(psi, UniformMPS)
    assert psi.algorithm == "vumps"
    assert psi.success is True
    np.testing.assert_allclose(psi.energy, -0.25, atol=1.0e-12)
    assert psi.gradient_norm < 1.0e-10
