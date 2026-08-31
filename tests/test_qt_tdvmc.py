import numpy as np

from pyqed.qt import ComplexJastrowTDVMC1D, split_operator_step


def test_complex_jastrow_spatial_derivatives():
    model = ComplexJastrowTDVMC1D(ngrid=401)
    rng = np.random.default_rng(3)
    theta = rng.normal(scale=0.1, size=model.nparams) + 1j * rng.normal(
        scale=0.1, size=model.nparams
    )
    x = np.linspace(-2.2, 2.2, 31)
    step = 2.0e-5
    value, first, second = model.log_derivatives(x, theta)
    value_plus = model.log_derivatives(x + step, theta)[0]
    value_minus = model.log_derivatives(x - step, theta)[0]
    first_fd = (value_plus - value_minus) / (2.0 * step)
    second_fd = (value_plus - 2.0 * value + value_minus) / step**2

    np.testing.assert_allclose(first, first_fd, rtol=2.0e-8, atol=2.0e-8)
    np.testing.assert_allclose(second, second_fd, rtol=2.0e-5, atol=2.0e-5)


def test_continuity_lift_cancels_density_time_derivative():
    model = ComplexJastrowTDVMC1D(ngrid=1201)
    theta = model.initial_parameters()
    theta[2:] = 0.03j * np.sin(np.arange(model.nparams - 2))
    theta_dot, _ = model.tdvp_velocity(theta)
    state = model.continuity_state(theta, theta_dot)
    flux_derivative = np.gradient(
        state["rho"] * state["tangent_velocity"], model.dx, edge_order=2
    )
    residual = state["rho_dot"] + flux_derivative
    central = state["rho"] > np.max(state["rho"]) * 1.0e-7

    assert np.sqrt(np.mean(residual[central] ** 2)) < 2.0e-4


def test_split_operator_preserves_norm():
    model = ComplexJastrowTDVMC1D(ngrid=401)
    psi = model.wavefunction(model.initial_parameters())
    propagated = split_operator_step(
        psi, model.potential_grid, model.dx, 0.01, mass=model.mass, hbar=model.hbar
    )

    np.testing.assert_allclose(
        np.trapezoid(np.abs(propagated) ** 2, model.grid), 1.0, atol=2.0e-12
    )


def test_fixed_quantiles_estimate_tdvp_velocity():
    model = ComplexJastrowTDVMC1D(
        ngrid=1201,
        centers=(-2.4, -1.2, 0.0, 1.2, 2.4),
        metric_shift=1.0e-3,
    )
    theta = model.initial_parameters()
    quantiles = (np.arange(512) + 0.5) / 512
    grid_velocity, _ = model.tdvp_velocity(theta)
    sampled_velocity, _ = model.tdvp_velocity(
        theta, model.quantile_trajectories(theta, quantiles)
    )

    relative_error = np.linalg.norm(sampled_velocity - grid_velocity) / np.linalg.norm(
        grid_velocity
    )
    assert relative_error < 0.1
