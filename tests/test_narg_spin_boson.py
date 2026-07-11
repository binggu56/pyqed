import numpy as np

from pyqed.narg import (
    SpinBosonWilsonChain,
    SpinBosonWilsonNARG,
    boson_displaced_dvr_operators,
    boson_dvr_operators,
    extract_spin_boson_fpes_observable,
    fit_order_parameter_exponent,
    log_discretized_spin_boson_wilson_chain,
    narg_rescaled_spectrum_flow,
    scan_spin_boson_alpha,
    scan_spin_boson_fixed_point,
    scan_spin_boson_fixed_point_flows,
    scan_spin_boson_fpes_alpha,
    scan_spin_boson_fpes_observables,
    scan_spin_boson_gap_thresholds,
    sine_dvr_boson_operators,
    spin_boson_mode_pes,
    spin_boson_wilson_exact,
    spin_boson_wilson_exact_magnetization,
    star_to_wilson_chain,
)


def test_star_to_wilson_chain_tridiagonalizes_star_bath():
    frequencies = np.array([1.0, 0.5, 0.25, 0.125])
    couplings = np.array([0.4, 0.2, 0.1, 0.05])

    onsite, hopping, t0, transform = star_to_wilson_chain(frequencies, couplings)

    assert t0 > 0.0
    np.testing.assert_allclose(transform @ transform.T, np.eye(4), atol=1e-12)
    tridiagonal = transform @ np.diag(frequencies) @ transform.T
    np.testing.assert_allclose(np.diag(tridiagonal), onsite, atol=1e-12)
    np.testing.assert_allclose(np.diag(tridiagonal, 1), hopping, atol=1e-12)
    np.testing.assert_allclose(tridiagonal, np.diag(np.diag(tridiagonal)) + np.diag(hopping, 1) + np.diag(hopping, -1), atol=1e-12)


def test_oscillator_dvr_is_unitary_coordinate_basis():
    identity, b, bdag, number, grid, transform = boson_dvr_operators(5)

    np.testing.assert_allclose(transform.T.conj() @ transform, np.eye(5), atol=1e-12)
    np.testing.assert_allclose(identity, np.eye(5), atol=1e-12)
    np.testing.assert_allclose(bdag, b.T.conj(), atol=1e-12)
    np.testing.assert_allclose(number, bdag @ b, atol=1e-12)
    assert np.all(np.diff(grid) > 0.0)


def test_displaced_oscillator_dvr_builds_projected_coordinate_basis():
    identity, b, bdag, number, grid, transform = boson_displaced_dvr_operators(
        4,
        displacement=1.0,
        parent_dim=10,
    )

    np.testing.assert_allclose(transform.T.conj() @ transform, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(identity, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(bdag, b.T.conj(), atol=1e-12)
    np.testing.assert_allclose(number, number.T.conj(), atol=1e-12)
    assert np.all(np.diff(grid) > 0.0)


def test_sine_dvr_boson_operators_are_hermitian_coordinate_grid():
    identity, b, bdag, number, grid, kinetic, momentum = sine_dvr_boson_operators(
        6,
        qmax=5.0,
        center=0.5,
    )

    np.testing.assert_allclose(identity, np.eye(6), atol=1e-12)
    np.testing.assert_allclose(bdag, b.T.conj(), atol=1e-12)
    np.testing.assert_allclose(number, number.T.conj(), atol=1e-12)
    np.testing.assert_allclose(kinetic, kinetic.T.conj(), atol=1e-12)
    np.testing.assert_allclose(momentum, momentum.T.conj(), atol=1e-12)
    assert grid[0] > -4.5
    assert grid[-1] < 5.5
    assert np.all(np.diff(grid) > 0.0)


def test_spin_boson_wilson_narg_matches_exact_without_truncation():
    chain = log_discretized_spin_boson_wilson_chain(
        3,
        alpha=0.04,
        Lambda=2.0,
        s=1.0,
        omegac=1.0,
        epsilon=0.02,
        delta=0.1,
    )
    exact, _ = spin_boson_wilson_exact(chain, nboson=3, nroots=4, basis="dvr")
    result = SpinBosonWilsonNARG(chain, nboson=3, bond_dim=256, basis="dvr").run(nroots=4)

    np.testing.assert_allclose(result.energies, exact, atol=1e-10)
    assert result.steps[-1].product_dim == 54
    assert result.steps[-1].kept == 54
    assert result.steps[-1].effective_hamiltonian is not None
    assert result.steps[-1].boundary_annihilation is not None


def test_spin_boson_wilson_narg_magnetization_matches_exact_without_truncation():
    chain = log_discretized_spin_boson_wilson_chain(
        3,
        alpha=0.04,
        Lambda=2.0,
        s=1.0,
        omegac=1.0,
        epsilon=0.02,
        delta=0.1,
    )
    _, exact_mag = spin_boson_wilson_exact_magnetization(chain, nboson=3, nroots=3, basis="dvr")
    result = SpinBosonWilsonNARG(chain, nboson=3, bond_dim=256, basis="dvr").run(nroots=3)

    np.testing.assert_allclose(result.magnetizations, exact_mag, atol=1e-10)


def test_spin_boson_wilson_narg_truncated_ground_energy_is_variational():
    chain = log_discretized_spin_boson_wilson_chain(
        4,
        alpha=0.05,
        Lambda=2.0,
        s=1.0,
        omegac=1.0,
        epsilon=0.0,
        delta=0.1,
    )
    exact, _ = spin_boson_wilson_exact(chain, nboson=3, nroots=1, basis="dvr")
    result = SpinBosonWilsonNARG(chain, nboson=3, bond_dim=8, basis="dvr").run(nroots=1)

    assert result.energies[0] >= exact[0] - 1e-10
    assert result.steps[-1].kept <= 8


def test_spin_boson_wilson_chain_can_be_built_from_discretized_sbm_like_object():
    class DummySBM:
        onsite = np.array([0.5, 0.25])
        hopping = np.array([0.1])
        t0 = 0.3
        epsilon = 0.02
        delta = 0.1
        xi = np.array([0.6, 0.2])
        g = np.array([0.2, 0.1])

    chain = SpinBosonWilsonChain.from_sbm(DummySBM())

    np.testing.assert_allclose(chain.onsite, [0.5, 0.25])
    np.testing.assert_allclose(chain.hopping, [0.1])
    assert chain.impurity_coupling == 0.3
    assert chain.nmodes == 2


def test_spin_boson_alpha_scan_and_power_law_fit_are_well_formed():
    scan = scan_spin_boson_alpha(
        [0.02, 0.05, 0.08],
        nmodes=3,
        nboson=3,
        bond_dim=16,
        Lambda=2.0,
        s=0.5,
        epsilon=1e-4,
        delta=0.1,
        nroots=2,
        basis="dvr",
    )

    assert scan.energies.shape == (3, 2)
    assert scan.gaps.shape == (3,)
    assert scan.magnetizations.shape == (3,)
    assert scan.susceptibilities.shape == (3,)
    assert scan.pseudo_critical_alpha in scan.alphas

    fit = fit_order_parameter_exponent(
        np.array([0.2, 0.3, 0.4]),
        np.array([0.1, 0.2, 0.3]),
        alpha_c=0.1,
    )
    assert np.isfinite(fit.exponent)
    assert fit.r2 > 0.0


def test_spin_boson_gap_threshold_scan_is_well_formed():
    scan = scan_spin_boson_gap_thresholds(
        [2, 3],
        [0.02, 0.04, 0.06],
        nboson=3,
        bond_dim=[16, 20],
        gap_threshold=1e-8,
        Lambda=2.0,
        s=0.5,
        delta=0.1,
        basis="dvr",
    )

    assert scan.gaps.shape == (2, 3)
    assert scan.threshold_alphas.shape == (2,)
    assert scan.minimum_gap_alphas.shape == (2,)
    assert scan.minimum_gaps.shape == (2,)
    assert np.all(np.isin(scan.minimum_gap_alphas, scan.alphas))
    np.testing.assert_allclose(scan.minimum_gaps, np.nanmin(scan.gaps, axis=1))


def test_spin_boson_fixed_point_scan_is_well_formed():
    chain = log_discretized_spin_boson_wilson_chain(
        3,
        alpha=0.04,
        Lambda=2.0,
        s=0.5,
        epsilon=0.0,
        delta=0.1,
    )
    result = SpinBosonWilsonNARG(chain, nboson=3, bond_dim=20, basis="dvr").run(nroots=4)
    flow = narg_rescaled_spectrum_flow(result, Lambda=2.0, nlevels=4)

    assert flow.shape == (3, 3)
    assert np.all(np.isfinite(flow))

    scan = scan_spin_boson_fixed_point(
        [0.02, 0.04],
        nmodes=3,
        nboson=3,
        bond_dim=20,
        Lambda=2.0,
        s=0.5,
        delta=0.1,
        nlevels=3,
        late_steps=2,
        basis="dvr",
    )

    assert scan.spectra.shape == (2, 3, 2)
    assert scan.rescaled_gaps.shape == (2, 2)
    assert scan.best_alpha in scan.alphas
    assert np.all(np.isfinite(scan.scores))

    flows = scan_spin_boson_fixed_point_flows(
        [0.02, 0.04],
        nmodes=3,
        nboson=3,
        bond_dim=20,
        Lambda=2.0,
        s=0.5,
        delta=0.1,
        nlevels=3,
        late_steps=2,
        basis="dvr",
    )

    assert flows.spectra.shape == (2, 3, 2)
    assert flows.endpoint_spectra.shape == (2, 2)
    assert flows.drift_scores.shape == (2,)
    assert flows.endpoint_changes.shape == (1,)
    assert flows.crossover_alpha == 0.03


def test_spin_boson_mode_pes_is_well_formed():
    chain = log_discretized_spin_boson_wilson_chain(
        3,
        alpha=0.04,
        Lambda=2.0,
        s=0.5,
        epsilon=0.0,
        delta=0.1,
    )
    q = np.linspace(-2.0, 2.0, 9)
    pes0 = spin_boson_mode_pes(
        chain,
        0,
        q,
        nboson=3,
        bond_dim=20,
        nlevels=2,
        basis="dvr",
    )
    pes2 = spin_boson_mode_pes(
        chain,
        2,
        q,
        nboson=3,
        bond_dim=20,
        nlevels=3,
        basis="dvr",
    )

    assert pes0.surfaces.shape == (9, 2)
    assert pes2.surfaces.shape == (9, 3)
    assert pes0.onsite_frequency > 0.0
    assert pes2.coupling_norm > 0.0
    assert np.all(np.isfinite(pes2.surfaces))

    obs = extract_spin_boson_fpes_observable(pes2)
    assert obs.well_separation > 0.0
    assert obs.q0 > 0.0
    assert obs.barrier_height >= 0.0
    assert np.isfinite(obs.curvature)

    scan = scan_spin_boson_fpes_observables(
        chain,
        [0, 2],
        q,
        nboson=3,
        bond_dim=20,
        basis="dvr",
    )
    assert scan.sites.shape == (2,)
    assert scan.q0.shape == (2,)
    assert np.all(scan.energy_scales > 0.0)

    alpha_scan = scan_spin_boson_fpes_alpha(
        [0.01, 0.04],
        [0, 2],
        q,
        nmodes=3,
        nboson=3,
        bond_dim=20,
        Lambda=2.0,
        s=0.5,
        delta=0.1,
        q0_threshold=0.2,
        basis="dvr",
    )
    assert alpha_scan.q0.shape == (2, 2)
    assert alpha_scan.barrier_heights.shape == (2, 2)
    assert alpha_scan.endpoint_q0.shape == (2,)
    if alpha_scan.pseudo_critical_alpha is not None:
        assert alpha_scan.alphas[0] <= alpha_scan.pseudo_critical_alpha <= alpha_scan.alphas[-1]


def test_spin_boson_exact_spectrum_is_same_in_fock_and_dvr_basis():
    chain = log_discretized_spin_boson_wilson_chain(
        2,
        alpha=0.03,
        Lambda=2.0,
        s=0.5,
        epsilon=0.01,
        delta=0.1,
    )
    fock, _ = spin_boson_wilson_exact(chain, nboson=4, nroots=5, basis="fock")
    dvr, _ = spin_boson_wilson_exact(chain, nboson=4, nroots=5, basis="dvr")
    gh_dvr, _ = spin_boson_wilson_exact(chain, nboson=4, nroots=5, basis="gh-dvr")

    np.testing.assert_allclose(dvr, fock, atol=1e-10)
    np.testing.assert_allclose(gh_dvr, fock, atol=1e-10)


def test_spin_boson_displaced_dvr_narg_matches_displaced_exact_without_truncation():
    chain = log_discretized_spin_boson_wilson_chain(
        2,
        alpha=0.08,
        Lambda=2.0,
        s=0.5,
        epsilon=0.01,
        delta=0.1,
    )
    shifts = chain.estimate_displacements()
    assert shifts.shape == (2,)
    assert np.all(shifts >= 0.0)

    exact, _ = spin_boson_wilson_exact(
        chain,
        nboson=4,
        nroots=4,
        basis="displaced-dvr",
        displacements="auto",
        parent_dim=10,
    )
    result = SpinBosonWilsonNARG(
        chain,
        nboson=4,
        bond_dim=128,
        basis="displaced-dvr",
        displacements="auto",
        parent_dim=10,
    ).run(nroots=4)

    np.testing.assert_allclose(result.energies, exact, atol=1e-10)


def test_spin_boson_sine_dvr_narg_matches_sine_exact_without_truncation():
    chain = log_discretized_spin_boson_wilson_chain(
        2,
        alpha=0.04,
        Lambda=2.0,
        s=0.5,
        epsilon=0.01,
        delta=0.1,
    )
    exact, _ = spin_boson_wilson_exact(
        chain,
        nboson=5,
        nroots=4,
        basis="sine-dvr",
        displacements="auto",
        dvr_qmax=6.0,
    )
    result = SpinBosonWilsonNARG(
        chain,
        nboson=5,
        bond_dim=128,
        basis="sine-dvr",
        displacements="auto",
        dvr_qmax=6.0,
    ).run(nroots=4)

    np.testing.assert_allclose(result.energies, exact, atol=1e-10)
