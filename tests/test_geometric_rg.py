import math

import numpy as np

from pyqed.narg.geometric_rg import (
    ExponentialRegulator,
    GaussianRegulator,
    Phi4CovariantFRG,
    Phi4FRG,
    Phi4ContinuousQGRF,
    Phi4FunctionalRegulatedQGRF,
    Phi4FunctionalQGRG,
    Phi4GaussianCouplings,
    Phi4GaussianShell,
    Phi4RegulatedQGRF,
    Phi4SmoothQGRF,
    Phi4VariationalQGRG,
    Phi4WegnerHoughtonLPA,
    gaussian_quantum_metric,
)


def test_phi4_gaussian_beta_matches_potential_derivatives():
    shell = Phi4GaussianShell(spatial_dimension=1)
    couplings = Phi4GaussianCouplings(0.03, 0.2, 0.35, 0.8)
    field = np.linspace(-0.08, 0.08, 81)
    values = shell.beta_potential(field, couplings)
    coefficients = np.polynomial.polynomial.polyfit(field, values, 8)
    fitted = np.array(
        [coefficients[n] * math.factorial(n) for n in range(1, 5)]
    )

    np.testing.assert_allclose(fitted, shell.beta(couplings).asarray(), rtol=2e-7)


def test_phi4_gaussian_flow_preserves_z2_symmetry():
    shell = Phi4GaussianShell(spatial_dimension=1)
    couplings = Phi4GaussianCouplings(mass2=0.15, quartic=0.9)
    beta = shell.beta(couplings)

    assert beta.source == 0.0
    assert beta.cubic == 0.0
    expected_mass2 = 2.0 * couplings.mass2 + couplings.quartic / (
        4.0 * np.pi * np.sqrt(1.0 + couplings.mass2)
    )
    expected_quartic = 2.0 * couplings.quartic - 3.0 * couplings.quartic**2 / (
        8.0 * np.pi * (1.0 + couplings.mass2) ** 1.5
    )
    np.testing.assert_allclose(beta.mass2, expected_mass2, atol=1e-14)
    np.testing.assert_allclose(beta.quartic, expected_quartic, atol=1e-14)


def test_global_wegner_houghton_fixed_point_and_stability():
    flow = Phi4WegnerHoughtonLPA(spacetime_dimension=3).solve_fixed_point(
        field_maxima=[0.5, 0.6, 0.8],
        mesh_points=101,
        tolerance=2.0e-6,
    )
    eigenvalues = flow.stability_spectrum(points=40)

    assert flow.success
    np.testing.assert_allclose(flow.fixed_curvature, -0.461534, rtol=2.0e-4)
    np.testing.assert_allclose(flow.correlation_exponent, 0.689459, rtol=3.0e-4)
    assert np.count_nonzero(eigenvalues.real > 1.0e-7) == 1


def test_global_wegner_houghton_reports_d2_lpa_degeneracy():
    flow = Phi4WegnerHoughtonLPA(spacetime_dimension=2)

    with np.testing.assert_raises_regex(ValueError, "field dimension is zero"):
        flow.solve_fixed_point()


def test_displaced_gaussian_metric_matches_fidelity():
    mean_derivative = 0.7
    covariance = 0.4
    covariance_derivative = -0.12
    metric = gaussian_quantum_metric(
        [[mean_derivative]], [[covariance]], [[[covariance_derivative]]]
    )[0, 0]

    def overlap(parameter):
        mean0 = 0.0
        mean1 = mean_derivative * parameter
        covariance1 = covariance + covariance_derivative * parameter
        prefactor = (
            (2.0 * covariance) ** 0.25
            * (2.0 * covariance1) ** 0.25
            / np.sqrt(covariance + covariance1)
        )
        exponent = -(mean0 - mean1) ** 2 / (
            4.0 * (covariance + covariance1)
        )
        return prefactor * np.exp(exponent)

    step = 2.0e-4
    fidelity_metric = (1.0 - abs(overlap(step)) ** 2) / step**2
    np.testing.assert_allclose(fidelity_metric, metric, rtol=4e-4)


def test_metric_and_weighted_metric_have_expected_signs():
    shell = Phi4GaussianShell(spatial_dimension=1)
    couplings = Phi4GaussianCouplings(mass2=0.2, cubic=0.3, quartic=0.8)

    metric = shell.metric_rate(0.25, couplings)
    weighted = shell.weighted_metric_rate(0.25, couplings, energy=0.0)

    assert metric > 0.0
    assert weighted < 0.0


def test_level2_response_zero_momentum_matches_local_gaussian_geometry():
    shell = Phi4GaussianShell(spatial_dimension=1)
    couplings = Phi4GaussianCouplings(mass2=0.2, quartic=0.8)
    field = 0.35

    response = shell.level2_response(field, couplings, [0.0])

    np.testing.assert_allclose(
        response["kernel_response"][0],
        couplings.quartic * field / (2.0 * response["frequency"]),
        atol=1e-14,
    )
    np.testing.assert_allclose(
        response["overlap_metric"][0],
        shell.metric_rate(field, couplings),
        atol=1e-14,
    )
    np.testing.assert_allclose(
        response["temporal_rate"],
        shell.external_kinetic_rates(field, couplings)[0],
        atol=1e-14,
    )


def test_level2_response_small_momentum_recovers_spatial_rate():
    shell = Phi4GaussianShell(spatial_dimension=1)
    couplings = Phi4GaussianCouplings(mass2=0.2, quartic=0.8)
    step = 2.0e-3
    response = shell.level2_response(
        0.35, couplings, [-step, 0.0, step]
    )
    values = response["two_point_rate"]
    numerical = (values[0] - 2.0 * values[1] + values[2]) / (2.0 * step**2)

    np.testing.assert_allclose(
        numerical, response["spatial_rate"], rtol=2e-5, atol=1e-12
    )


def test_level2_response_vanishes_at_symmetric_phi4_origin():
    shell = Phi4GaussianShell(spatial_dimension=1)
    couplings = Phi4GaussianCouplings(mass2=0.2, quartic=0.8)

    response = shell.level2_response(0.0, couplings, [0.0, 0.1])

    np.testing.assert_allclose(response["kernel_response"], 0.0, atol=0.0)
    np.testing.assert_allclose(response["two_point_rate"], 0.0, atol=0.0)
    np.testing.assert_allclose(response["overlap_metric"], 0.0, atol=0.0)
    np.testing.assert_allclose(response["spatial_rate"], 0.0, atol=0.0)


def test_z2_inertia_flow_reduces_to_gaussian_flow_at_zero_inertia2():
    shell = Phi4GaussianShell(spatial_dimension=1)
    couplings = Phi4GaussianCouplings(mass2=0.2, quartic=0.8)
    ordinary = shell.beta(couplings)
    dressed, inertia2_beta = shell.beta_z2(couplings, inertia2=0.0)

    np.testing.assert_allclose(dressed.asarray(), ordinary.asarray(), atol=1e-14)
    expected = couplings.quartic**2 / (
        16.0 * np.pi * (1.0 + couplings.mass2) ** 2.5
    )
    np.testing.assert_allclose(inertia2_beta, expected, atol=1e-14)


def test_inertia_feedback_metric_matches_analytic_z2_origin_curvature():
    shell = Phi4GaussianShell(spatial_dimension=1)
    couplings = Phi4GaussianCouplings(mass2=0.2, quartic=0.8)
    z2 = 0.15
    step = 1.0e-4

    def beta_z(field):
        inertia = 1.0 + 0.5 * z2 * field**2
        return shell.inertia_beta(
            field,
            couplings,
            inertia=inertia,
            inertia_derivative=z2 * field,
        )

    numerical = (beta_z(step) - 2.0 * beta_z(0.0) + beta_z(-step)) / step**2
    _, analytic = shell.beta_z2(couplings, inertia2=z2)
    np.testing.assert_allclose(numerical, analytic, rtol=2e-7)


def test_d2_geometric_z2_closure_has_no_positive_quartic_fixed_point():
    shell = Phi4GaussianShell(spatial_dimension=1)
    for mass2 in (-0.5, 0.0, 0.5):
        for quartic in (0.1, 1.0, 10.0):
            inertia2 = -quartic / (1.0 + mass2)
            beta, beta_z2 = shell.beta_z2(
                Phi4GaussianCouplings(mass2=mass2, quartic=quartic),
                inertia2,
            )
            np.testing.assert_allclose(beta_z2, 0.0, atol=1e-14)
            assert beta.quartic > 0.0


def test_minimum_anomalous_dimension_matches_weighted_metric_projection():
    shell = Phi4GaussianShell(spatial_dimension=1)
    couplings = Phi4GaussianCouplings(mass2=-0.3, quartic=6.0)
    minimum = np.sqrt(-6.0 * couplings.mass2 / couplings.quartic)
    metric = shell.metric_rate(minimum, couplings)
    frequency = np.sqrt(1.0 + shell.curvature(minimum, couplings))

    np.testing.assert_allclose(
        shell.anomalous_dimension(couplings), metric / frequency, atol=1e-14
    )


def test_lpa_prime_feeds_anomalous_dimension_into_scaling_terms():
    shell = Phi4GaussianShell(spatial_dimension=1)
    couplings = Phi4GaussianCouplings(mass2=-0.3, quartic=6.0)
    ordinary = shell.beta(couplings)
    dressed, eta = shell.beta_lpa_prime(couplings)

    np.testing.assert_allclose(
        dressed.mass2, ordinary.mass2 - eta * couplings.mass2, atol=1e-14
    )
    np.testing.assert_allclose(
        dressed.quartic,
        ordinary.quartic - 2.0 * eta * couplings.quartic,
        atol=1e-14,
    )
    assert eta > 0.0


def test_functional_qgrg_reduces_to_quartic_shell_flow():
    field = np.linspace(-0.7, 0.7, 81)
    functional = Phi4FunctionalQGRG(field, spatial_dimension=1)
    shell = Phi4GaussianShell(spatial_dimension=1)
    couplings = Phi4GaussianCouplings(mass2=0.2, quartic=0.8)
    potential = shell.potential(field, couplings)

    potential_rate, temporal_rate, spatial_rate = functional.rates(potential)
    expected_potential = shell.beta_potential(field, couplings)
    expected_kinetic = np.array(
        [shell.external_kinetic_rates(value, couplings) for value in field]
    )

    np.testing.assert_allclose(potential_rate, expected_potential, atol=2e-10)
    np.testing.assert_allclose(temporal_rate, expected_kinetic[:, 0], atol=2e-10)
    np.testing.assert_allclose(spatial_rate, expected_kinetic[:, 1], atol=2e-10)


def test_functional_qgrg_retains_full_kinetic_feedback():
    field = np.linspace(-0.6, 0.6, 71)
    functional = Phi4FunctionalQGRG(field, spatial_dimension=1)
    shell = Phi4GaussianShell(spatial_dimension=1)
    couplings = Phi4GaussianCouplings(mass2=0.25, quartic=0.9)
    potential = shell.potential(field, couplings)
    inertia = 1.0 + 0.08 * field**2 + 0.03 * field**4
    stiffness = 1.0 + 0.04 * field**2

    potential_rate, inertia_rate, stiffness_rate = functional.rates(
        potential, inertia=inertia, stiffness=stiffness
    )
    expected_potential = shell.beta_potential(
        field, couplings, inertia=inertia, stiffness=stiffness
    )
    expected_inertia = shell.inertia_beta(
        field,
        couplings,
        inertia=inertia,
        inertia_derivative=0.16 * field + 0.12 * field**3,
        stiffness=stiffness,
        stiffness_derivative=0.08 * field,
    )

    np.testing.assert_allclose(potential_rate, expected_potential, atol=3e-9)
    np.testing.assert_allclose(inertia_rate, expected_inertia, atol=3e-9)
    np.testing.assert_allclose(inertia_rate, inertia_rate[::-1], atol=2e-10)
    np.testing.assert_allclose(stiffness_rate, stiffness_rate[::-1], atol=2e-10)


def test_variational_qgrg_hartree_gap_is_stationary():
    field = np.linspace(-0.6, 0.6, 41)
    shell = Phi4GaussianShell(spatial_dimension=1)
    potential = shell.potential(
        field, Phi4GaussianCouplings(mass2=0.2, quartic=0.8)
    )
    flow = Phi4VariationalQGRG(
        field, log_width=0.2, quadrature_order=10
    )
    frame = flow.variational_frame(potential)
    index = field.size // 2
    base = frame["curvature"][index] + flow._momenta**2
    quartic = frame["quartic"][index]
    hartree = frame["hartree"][index]

    def gaussian_energy(shift):
        frequency = np.sqrt(base + shift)
        variance = np.sum(flow._momentum_weights / (2.0 * frequency))
        return np.sum(
            flow._momentum_weights
            * (0.25 * frequency + base / (4.0 * frequency))
        ) + quartic * variance**2 / 8.0

    step = 1.0e-5
    derivative = (
        gaussian_energy(hartree + step) - gaussian_energy(hartree - step)
    ) / (2.0 * step)
    np.testing.assert_allclose(
        hartree, 0.5 * quartic * frame["variance"][index], atol=2e-13
    )
    np.testing.assert_allclose(derivative, 0.0, atol=2e-9)


def test_variational_qgrg_retains_normal_ordered_feshbach_residual():
    field = np.linspace(-0.5, 0.5, 31)
    shell = Phi4GaussianShell(spatial_dimension=1)
    potential = shell.potential(
        field, Phi4GaussianCouplings(mass2=0.2, quartic=0.8)
    )
    flow = Phi4VariationalQGRG(
        field, log_width=0.2, quadrature_order=8
    )
    flow.rates(potential)

    np.testing.assert_allclose(flow.feshbach["three_boson"], 0.0, atol=0.0)
    assert np.all(flow.feshbach["four_boson"] < 0.0)
    np.testing.assert_allclose(
        flow.feshbach["total"], flow.feshbach["four_boson"], atol=0.0
    )


def test_variational_qgrg_converges_to_infinitesimal_gaussian_flow():
    field = np.linspace(-0.6, 0.6, 41)
    shell = Phi4GaussianShell(spatial_dimension=1)
    potential = shell.potential(
        field, Phi4GaussianCouplings(mass2=0.2, quartic=0.8)
    )
    gaussian = Phi4FunctionalQGRG(field).rates(potential)
    variational = Phi4VariationalQGRG(
        field, log_width=1.0e-3, quadrature_order=8
    ).rates(potential)

    for actual, expected in zip(variational, gaussian):
        np.testing.assert_allclose(actual, expected, rtol=1.4e-3, atol=2e-6)


def test_continuous_qgrf_free_limit_matches_standard_shell_flow():
    field = np.linspace(-0.4, 0.4, 9)
    couplings = Phi4GaussianCouplings(mass2=0.2)
    standard = Phi4GaussianShell(spatial_dimension=1).beta_potential(
        field, couplings
    )
    potential_rate, temporal_rate = Phi4ContinuousQGRF(
        quadrature_order=32, derivative_step=1.0e-3
    ).rates(field, couplings)

    np.testing.assert_allclose(potential_rate, standard, atol=6e-7)
    np.testing.assert_allclose(temporal_rate, 0.0, atol=0.0)


def test_continuous_qgrf_retains_single_shell_feshbach_response():
    field = np.array([0.0, 0.3])
    couplings = Phi4GaussianCouplings(mass2=0.2, quartic=0.8)
    flow = Phi4ContinuousQGRF(
        quadrature_order=32, derivative_step=1.0e-3
    )
    _, temporal_rate = flow.rates(field, couplings)

    assert temporal_rate[0] > 0.0
    assert flow.components["triplet_temporal_rate"][0] > 0.0
    assert flow.components["four_boson_energy_rate"][0] < 0.0
    np.testing.assert_allclose(
        temporal_rate,
        flow.components["pair_temporal_rate"]
        + flow.components["triplet_temporal_rate"],
        atol=2e-14,
    )


def test_continuous_qgrf_log_derivative_is_converged():
    field = np.array([0.0, 0.3])
    couplings = Phi4GaussianCouplings(mass2=0.2, quartic=0.8)
    coarse = Phi4ContinuousQGRF(
        quadrature_order=32, derivative_step=2.0e-3
    ).rates(field, couplings)
    fine = Phi4ContinuousQGRF(
        quadrature_order=32, derivative_step=1.0e-3
    ).rates(field, couplings)

    np.testing.assert_allclose(coarse[0], fine[0], rtol=2e-6)
    np.testing.assert_allclose(coarse[1], fine[1], rtol=2e-6)


def test_smooth_qgrf_free_limit_has_no_kinetic_dressing():
    field = np.array([0.0, 0.3])
    couplings = Phi4GaussianCouplings(mass2=0.2)
    potential_rate, temporal_rate, spatial_rate = Phi4SmoothQGRF(
        quadrature_order=24
    ).rates(field, couplings)

    assert np.all(np.isfinite(potential_rate))
    np.testing.assert_allclose(temporal_rate, 0.0, atol=0.0)
    np.testing.assert_allclose(spatial_rate, 0.0, atol=0.0)


def test_smooth_qgrf_resolves_broken_phase_hartree_frame():
    couplings = Phi4GaussianCouplings(mass2=-0.7, quartic=2.0)
    coarse = Phi4SmoothQGRF(quadrature_order=24)
    fine = Phi4SmoothQGRF(quadrature_order=32)
    coarse_mass = coarse.variational_frame(0.0, couplings)["mass2"]
    fine_mass = fine.variational_frame(0.0, couplings)["mass2"]

    assert coarse_mass > 1.0e-5
    np.testing.assert_allclose(coarse_mass, fine_mass, rtol=3.0e-4)


def test_smooth_qgrf_external_momentum_projection_is_converged():
    field = np.array([0.0, 0.3])
    couplings = Phi4GaussianCouplings(mass2=-0.3, quartic=6.0)
    coarse = Phi4SmoothQGRF(quadrature_order=24)
    fine = Phi4SmoothQGRF(quadrature_order=32)
    coarse_rates = coarse.rates(field, couplings)
    fine_rates = fine.rates(field, couplings)

    np.testing.assert_allclose(coarse_rates[0], fine_rates[0], rtol=2.0e-8)
    np.testing.assert_allclose(coarse_rates[1], fine_rates[1], rtol=3.0e-6)
    np.testing.assert_allclose(coarse_rates[2], fine_rates[2], rtol=3.2e-3)
    np.testing.assert_allclose(
        fine_rates[1],
        fine.components["pair_temporal_rate"]
        + fine.components["triplet_temporal_rate"],
        atol=2.0e-14,
    )


def test_smooth_level2_response_is_even_and_regular_at_zero_momentum():
    flow = Phi4SmoothQGRF(quadrature_order=32)
    couplings = Phi4GaussianCouplings(mass2=0.2, quartic=0.8)
    momenta = np.array([-0.02, -0.01, 0.0, 0.01, 0.02])

    response = flow.level2_response(0.35, couplings, momenta)

    np.testing.assert_allclose(
        response["two_point_rate"], response["two_point_rate"][::-1], atol=2e-14
    )
    np.testing.assert_allclose(
        response["overlap_metric"], response["overlap_metric"][::-1], atol=2e-14
    )
    assert np.all(np.isfinite(response["kernel_response"]))
    assert np.isfinite(response["spatial_rate"])


def test_smooth_level2_response_small_momentum_recovers_spatial_rate():
    flow = Phi4SmoothQGRF(quadrature_order=40)
    couplings = Phi4GaussianCouplings(mass2=0.2, quartic=0.8)
    momenta = np.array([0.0, 0.006, 0.009, 0.013, 0.018, 0.025])

    response = flow.level2_response(0.35, couplings, momenta)
    slopes = (
        response["two_point_rate"][1:] - response["two_point_rate"][0]
    ) / momenta[1:] ** 2
    extrapolated = np.polynomial.polynomial.polyfit(
        momenta[1:] ** 2, slopes, 2
    )[0]

    np.testing.assert_allclose(
        extrapolated, response["spatial_rate"], rtol=3e-7, atol=1e-12
    )


def test_smooth_level2_response_vanishes_at_symmetric_origin():
    flow = Phi4SmoothQGRF(quadrature_order=24)
    couplings = Phi4GaussianCouplings(mass2=0.2, quartic=0.8)

    response = flow.level2_response(0.0, couplings, [0.0, 0.1])

    np.testing.assert_allclose(response["kernel_response"], 0.0, atol=0.0)
    np.testing.assert_allclose(response["two_point_rate"], 0.0, atol=0.0)
    np.testing.assert_allclose(response["overlap_metric"], 0.0, atol=0.0)
    np.testing.assert_allclose(response["temporal_rate"], 0.0, atol=0.0)
    np.testing.assert_allclose(response["spatial_rate"], 0.0, atol=0.0)


def test_regulated_qgrf_free_limit_has_no_kinetic_dressing():
    couplings = Phi4GaussianCouplings(mass2=0.2)
    flow = Phi4RegulatedQGRF(quadrature_order=20)
    potential_rate, temporal_rate, spatial_rate = flow.rates(
        np.array([0.0, 0.3]), couplings
    )
    beta, eta_t, eta_x, dynamic_exponent = flow.beta(couplings)

    assert np.all(np.isfinite(potential_rate))
    np.testing.assert_allclose(temporal_rate, 0.0, atol=0.0)
    np.testing.assert_allclose(spatial_rate, 0.0, atol=0.0)
    np.testing.assert_allclose(beta.mass2, 0.4, atol=2.0e-12)
    np.testing.assert_allclose(beta.quartic, 0.0, atol=2.0e-8)
    np.testing.assert_allclose([eta_t, eta_x], 0.0, atol=0.0)
    np.testing.assert_allclose(dynamic_exponent, 1.0, atol=0.0)


def test_regulated_qgrf_gaussian_flow_matches_threshold_derivatives():
    couplings = Phi4GaussianCouplings(mass2=0.2, quartic=0.8)
    flow = Phi4RegulatedQGRF(
        quadrature_order=24, include_feshbach=False
    )
    beta, eta_t, eta_x, _ = flow.beta(couplings, radius=0.15)
    momentum2 = flow._momentum**2
    regulator = flow._regulator_value(flow._momentum, flow.cutoff)
    scale_derivative = flow.cutoff**2 * flow.regulator.scale_derivative(
        momentum2 / flow.cutoff**2
    )
    frequency = np.sqrt(momentum2 + regulator + couplings.mass2)
    measure = flow._weights / (2.0 * np.pi)
    expected_mass2 = 2.0 * couplings.mass2
    expected_mass2 += couplings.quartic * np.sum(
        measure * scale_derivative / frequency**3
    ) / 8.0
    expected_quartic = 2.0 * couplings.quartic
    expected_quartic -= 9.0 * couplings.quartic**2 * np.sum(
        measure * scale_derivative / frequency**5
    ) / 16.0

    np.testing.assert_allclose(beta.mass2, expected_mass2, rtol=2.0e-8)
    np.testing.assert_allclose(beta.quartic, expected_quartic, rtol=3.0e-7)
    np.testing.assert_allclose([eta_t, eta_x], 0.0, atol=0.0)


def test_regulated_feshbach_energy_rates_match_cutoff_difference():
    flow = Phi4RegulatedQGRF(quadrature_order=12)
    frame = {
        "curvature": -0.2,
        "gap2": 0.8,
        "curvature_derivative": 0.7,
        "inertia": 1.3,
        "stiffness": 0.8,
    }
    cubic2 = frame["curvature_derivative"] ** 2
    quartic = 3.5
    step = 1.0e-5
    upper = flow.cutoff * np.exp(step)
    lower = flow.cutoff * np.exp(-step)
    triplet_difference = -(
        -flow._triplet_moment(
            frame, cubic2, upper, denominator_power=1
        )
        + flow._triplet_moment(
            frame, cubic2, lower, denominator_power=1
        )
    ) / (2.0 * step)
    quartic_difference = -(
        flow._quartic_energy(frame, quartic, upper)
        - flow._quartic_energy(frame, quartic, lower)
    ) / (2.0 * step)

    np.testing.assert_allclose(
        flow._triplet_energy_rate(frame, cubic2, flow.cutoff),
        triplet_difference,
        rtol=2.0e-9,
    )
    np.testing.assert_allclose(
        flow._quartic_energy_rate(frame, quartic, flow.cutoff),
        quartic_difference,
        rtol=2.0e-9,
    )


def test_regulated_kinetic_rates_match_cutoff_difference():
    flow = Phi4RegulatedQGRF(quadrature_order=12)
    frame = {
        "curvature": -0.2,
        "gap2": 0.8,
        "curvature_derivative": 0.7,
        "inertia": 1.3,
        "stiffness": 0.8,
    }
    couplings = Phi4GaussianCouplings(quartic=3.5)
    step = 1.0e-5
    upper = flow.cutoff * np.exp(step)
    lower = flow.cutoff * np.exp(-step)
    pair_difference = -(
        flow._pair_temporal(frame, upper)
        - flow._pair_temporal(frame, lower)
    ) / (2.0 * step)
    triplet_difference = -2.0 * (
        flow._triplet_moment(
            frame, couplings.quartic**2, upper, denominator_power=3
        )
        - flow._triplet_moment(
            frame, couplings.quartic**2, lower, denominator_power=3
        )
    ) / (2.0 * step)
    momentum = 0.013
    static_difference = -(
        flow._static_response(frame, couplings, upper, momentum)
        - flow._static_response(frame, couplings, lower, momentum)
    ) / (2.0 * step)

    np.testing.assert_allclose(
        flow._pair_temporal_rate(frame, flow.cutoff),
        pair_difference,
        rtol=2.0e-9,
    )
    np.testing.assert_allclose(
        2.0
        * flow._triplet_moment_rate(
            frame,
            couplings.quartic**2,
            flow.cutoff,
            denominator_power=3,
        ),
        triplet_difference,
        rtol=2.0e-9,
    )
    np.testing.assert_allclose(
        flow._static_response_rate(
            frame, couplings, flow.cutoff, momentum
        ),
        static_difference,
        rtol=2.0e-9,
    )


def test_regulated_qgrf_feshbach_closure_retains_interacting_fixed_point():
    flow = Phi4RegulatedQGRF(quadrature_order=20)
    couplings = Phi4GaussianCouplings(
        mass2=-0.6060885976110706, quartic=4.715445357881669
    )
    beta, eta_t, eta_x, dynamic_exponent = flow.beta(couplings)

    np.testing.assert_allclose(
        [beta.mass2, beta.quartic], 0.0, atol=2.0e-7
    )
    assert eta_t > eta_x > 0.0
    assert dynamic_exponent > 1.0
    np.testing.assert_allclose(
        flow.components["regulated_gap2"],
        1.0 - 2.0 * couplings.mass2,
        rtol=2.0e-14,
    )


def test_functional_regulated_gaussian_sources_match_local_closure():
    field = np.linspace(-0.4, 0.4, 17)
    center = field.size // 2
    couplings = Phi4GaussianCouplings(mass2=-0.2, quartic=3.5)
    potential = Phi4GaussianShell.potential(field, couplings)
    functional = Phi4FunctionalRegulatedQGRF(
        field, stencil_size=7, quadrature_order=12, include_feshbach=False
    )
    local = Phi4RegulatedQGRF(
        quadrature_order=12, include_feshbach=False
    )
    sources = functional.sources(potential)
    local_rates = local.rates(np.array([0.0]), couplings)

    np.testing.assert_allclose(
        [values[center] for values in sources],
        [values[0] for values in local_rates],
        rtol=2.0e-8,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        functional.geometry["curvature"][center], couplings.mass2, atol=2e-14
    )
    np.testing.assert_allclose(
        functional.geometry["quartic"][center], couplings.quartic, atol=2e-12
    )


def test_functional_regulated_flow_normalizes_kinetics_at_origin():
    field = np.linspace(-0.4, 0.4, 17)
    couplings = Phi4GaussianCouplings(mass2=-0.2, quartic=3.5)
    potential = Phi4GaussianShell.potential(field, couplings)
    flow = Phi4FunctionalRegulatedQGRF(
        field, stencil_size=7, quadrature_order=12
    )
    _, temporal_rate, spatial_rate = flow.rates(potential)

    np.testing.assert_allclose(
        temporal_rate[flow.normalize_index], 0.0, atol=2.0e-15
    )
    np.testing.assert_allclose(
        spatial_rate[flow.normalize_index], 0.0, atol=2.0e-15
    )
    assert flow.geometry["eta_t"] > 0.0
    assert flow.geometry["eta_x"] > 0.0


def test_functional_regulated_full_grid_gaussian_fixed_point():
    field = np.linspace(-0.6, 0.6, 17)
    flow = Phi4FunctionalRegulatedQGRF(
        field, stencil_size=7, quadrature_order=12, include_feshbach=False
    )
    initial = Phi4GaussianShell.potential(
        field, Phi4GaussianCouplings(mass2=-0.2, quartic=3.5)
    )
    flow.solve_fixed_point(initial, tolerance=1.0e-8)

    assert flow.success
    np.testing.assert_allclose(flow.fixed_beta, 0.0, atol=1.0e-7)
    np.testing.assert_allclose(
        flow.fixed_potential, flow.fixed_potential[::-1], atol=1.0e-12
    )
    assert np.ptp(flow.fixed_potential) > 1.0e-3


def test_functional_regulated_self_energy_low_rank_fit():
    field = np.linspace(-0.4, 0.4, 9)
    potential = Phi4GaussianShell.potential(
        field, Phi4GaussianCouplings(mass2=-0.2, quartic=3.5)
    )
    flow = Phi4FunctionalRegulatedQGRF(
        field, stencil_size=7, quadrature_order=12
    )
    momenta = np.linspace(-0.08, 0.08, 7)
    fitted = flow.fit_self_energy(potential, momenta, rank=3)

    assert fitted["values"].shape == (field.size, momenta.size)
    assert fitted["rank"] == 3
    assert fitted["relative_error"] < 2.0e-4
    np.testing.assert_allclose(
        fitted["values"], fitted["values"][:, ::-1], rtol=2.0e-12
    )


def test_functional_regulated_kinetic_functions_feed_back_into_frame():
    field = np.linspace(-0.4, 0.4, 9)
    potential = Phi4GaussianShell.potential(
        field, Phi4GaussianCouplings(mass2=-0.2, quartic=3.5)
    )
    flow = Phi4FunctionalRegulatedQGRF(
        field, stencil_size=7, quadrature_order=12
    )
    unit = flow.sources(potential)[0]
    dressed = flow.sources(
        potential,
        inertia=np.full(field.size, 1.2),
        stiffness=np.full(field.size, 0.8),
    )[0]

    assert np.max(np.abs(unit - dressed)) > 1.0e-3


def test_functional_regulated_source_retains_feshbach_energy_terms():
    field = np.linspace(-0.4, 0.4, 9)
    potential = Phi4GaussianShell.potential(
        field, Phi4GaussianCouplings(mass2=-0.2, quartic=3.5)
    )
    flow = Phi4FunctionalRegulatedQGRF(
        field, stencil_size=7, quadrature_order=12
    )
    source = flow.sources(potential, kinetic=False)[0]
    center = flow.normalize_index
    frame = {
        "curvature": flow.geometry["curvature"][center],
        "gap2": 1.0 + flow.geometry["curvature"][center],
        "curvature_derivative": flow.geometry["cubic"][center],
        "inertia": 1.0,
        "stiffness": 1.0,
    }
    quartic = flow.geometry["quartic"][center]
    expected = flow.kernel._gaussian_loop(frame)
    expected += flow.kernel.feshbach_strength * flow.kernel._triplet_energy_rate(
        frame, frame["curvature_derivative"] ** 2, flow.kernel.cutoff
    )
    expected += flow.kernel.feshbach_strength * flow.kernel._quartic_energy_rate(
        frame, quartic, flow.kernel.cutoff
    )

    np.testing.assert_allclose(source[center], expected, rtol=2.0e-13)


def test_functional_regulated_feshbach_homotopy_is_linear():
    field = np.linspace(-0.4, 0.4, 9)
    potential = Phi4GaussianShell.potential(
        field, Phi4GaussianCouplings(mass2=-0.2, quartic=3.5)
    )
    gaussian = Phi4FunctionalRegulatedQGRF(
        field, stencil_size=7, quadrature_order=12, include_feshbach=False
    ).sources(potential)[0]
    full = Phi4FunctionalRegulatedQGRF(
        field, stencil_size=7, quadrature_order=12, feshbach_strength=1.0
    ).sources(potential)[0]
    half = Phi4FunctionalRegulatedQGRF(
        field, stencil_size=7, quadrature_order=12, feshbach_strength=0.5
    ).sources(potential)[0]

    np.testing.assert_allclose(
        half - gaussian, 0.5 * (full - gaussian), rtol=2.0e-13
    )


def test_functional_regulated_coupled_free_fixed_point():
    field = np.linspace(-0.4, 0.4, 9)
    flow = Phi4FunctionalRegulatedQGRF(
        field, stencil_size=7, quadrature_order=12, include_feshbach=False
    )
    source = flow.kernel._gaussian_loop(
        {"curvature": 0.0, "inertia": 1.0, "stiffness": 1.0}
    )
    initial = np.full(field.size, -0.5 * source)
    flow.solve_coupled_fixed_point(initial, tolerance=1.0e-9)

    assert flow.success
    np.testing.assert_allclose(flow.fixed_beta, 0.0, atol=1.0e-8)
    np.testing.assert_allclose(flow.fixed_inertia, 1.0, atol=1.0e-10)
    np.testing.assert_allclose(flow.fixed_stiffness, 1.0, atol=1.0e-10)


def test_functional_regulated_spectral_free_fixed_point():
    field = np.linspace(-0.4, 0.4, 9)
    flow = Phi4FunctionalRegulatedQGRF(
        field, stencil_size=7, quadrature_order=12, include_feshbach=False
    )
    source = flow.kernel._gaussian_loop(
        {"curvature": 0.0, "inertia": 1.0, "stiffness": 1.0}
    )
    initial = np.full(field.size, -0.5 * source)
    flow.solve_spectral_fixed_point(
        initial, modes=3, tolerance=1.0e-9, max_evaluations=20
    )

    assert flow.success
    np.testing.assert_allclose(flow.fixed_beta, 0.0, atol=1.0e-8)
    np.testing.assert_allclose(flow.fixed_inertia, 1.0, atol=1.0e-10)
    np.testing.assert_allclose(flow.fixed_stiffness, 1.0, atol=1.0e-10)


def test_functional_regulated_spectral_solver_retains_interacting_branch():
    field = np.linspace(-0.8, 0.8, 17)
    flow = Phi4FunctionalRegulatedQGRF(
        field,
        stencil_size=7,
        quadrature_order=12,
        feshbach_strength=1.0,
        kinetic_strength=0.0,
    )
    initial = Phi4GaussianShell.potential(
        field,
        Phi4GaussianCouplings(mass2=-0.628094322, quartic=4.30942319),
    )
    flow.solve_spectral_potential_fixed_point(
        initial, modes=3, tolerance=1.0e-9, max_evaluations=300
    )

    center = flow.normalize_index
    assert flow.success
    assert flow.derivative(flow.fixed_potential, 2)[center] < -0.4
    assert flow.derivative(flow.fixed_potential, 4)[center] > 4.0
    np.testing.assert_allclose(flow.fixed_beta, 0.0, atol=1.0e-8)


def test_functional_regulated_mode_homotopy_retains_interacting_branch():
    field = np.linspace(-0.8, 0.8, 17)
    flow = Phi4FunctionalRegulatedQGRF(
        field,
        stencil_size=7,
        quadrature_order=12,
        feshbach_strength=1.0,
        kinetic_strength=0.0,
    )
    initial = Phi4GaussianShell.potential(
        field,
        Phi4GaussianCouplings(mass2=-0.628094322, quartic=4.30942319),
    )
    flow.continue_potential_modes(
        initial,
        [3, 4],
        homotopy_steps=8,
        tolerance=2.0e-7,
        max_evaluations=600,
    )

    quartic = flow.derivative(flow.fixed_potential, 4)[flow.normalize_index]
    assert flow.success
    assert quartic > 3.0
    assert flow.mode_continuation[-1]["homotopy"] == 1.0
    assert flow.mode_continuation[-1]["accepted"]
    np.testing.assert_allclose(flow.fixed_beta, 0.0, atol=1.0e-7)


def test_functional_regulated_feshbach_continuation_retains_interacting_branch():
    field = np.linspace(-0.8, 0.8, 17)
    flow = Phi4FunctionalRegulatedQGRF(
        field,
        stencil_size=7,
        quadrature_order=12,
        feshbach_strength=1.0,
        kinetic_strength=0.0,
    )
    initial = Phi4GaussianShell.potential(
        field,
        Phi4GaussianCouplings(mass2=-0.628094322, quartic=4.30942319),
    )
    flow.continue_feshbach_potential_fixed_point(
        initial,
        [1.0, 0.5, 0.0],
        modes=3,
        tolerance=2.0e-7,
        max_evaluations=400,
    )

    assert flow.success
    assert all(point["success"] for point in flow.feshbach_continuation)
    assert flow.feshbach_continuation[-1]["quartic"] > 5.0
    np.testing.assert_allclose(flow.fixed_beta, 0.0, atol=1.0e-7)


def test_functional_regulated_stability_audits_field_rescaling_before_projection():
    field = np.linspace(-0.4, 0.4, 9)
    flow = Phi4FunctionalRegulatedQGRF(
        field, stencil_size=7, quadrature_order=12
    )
    flow.fixed_potential = 0.1 * field**2 + 0.2 * field**4
    flow.fixed_inertia = np.exp(0.1 * field**2)
    flow.fixed_stiffness = np.exp(0.2 * field**2)

    full = flow.stability_spectrum(
        modes=3, step=1.0e-5, project_redundant=False
    )
    audited = flow.stability_spectrum(
        modes=3, step=1.0e-5, project_redundant=True
    )

    assert full.shape == (6,)
    assert audited.shape == (6,)
    assert flow.redundancy_diagnostics["vacuum_energy_gauge_fixed"]
    assert flow.redundancy_diagnostics[
        "field_rescaling_projection_requested"
    ]
    assert not flow.redundancy_diagnostics["field_rescaling_is_invariant"]
    assert not flow.redundancy_diagnostics["field_rescaling_projected"]
    assert flow.redundancy_diagnostics["projection_warning"] is not None
    assert flow.redundancy_diagnostics["dominant_full_mode_overlap"] > 0.0
    assert len(flow.stability_mode_diagnostics) == 6
    for mode in flow.stability_mode_diagnostics:
        np.testing.assert_allclose(
            mode["potential_fraction"]
            + mode["temporal_metric_fraction"]
            + mode["spatial_metric_fraction"],
            1.0,
        )


def test_functional_regulated_mode_extension_probe_reaches_free_endpoint():
    field = np.linspace(-0.6, 0.6, 13)
    flow = Phi4FunctionalRegulatedQGRF(
        field, stencil_size=9, quadrature_order=12, include_feshbach=False
    )
    potential = np.zeros_like(field)
    flow.rates = lambda potential, inertia, stiffness: (
        np.zeros_like(potential),
        np.zeros_like(inertia),
        np.zeros_like(stiffness),
    )
    flow.probe_coupled_mode_extension(
        potential,
        3,
        4,
        initial_step=1.0,
        minimum_step=0.1,
        tolerance=1.0e-7,
        max_evaluations=30,
    )

    assert flow.success
    assert flow.mode_extension_diagnostics["endpoint_reached"]
    assert not flow.mode_extension_diagnostics["near_singular_fold"]
    np.testing.assert_allclose(flow.fixed_beta, 0.0, atol=1.0e-7)


def test_functional_regulated_continuation_and_stability_are_available():
    field = np.linspace(-0.4, 0.4, 9)
    flow = Phi4FunctionalRegulatedQGRF(
        field, stencil_size=7, quadrature_order=12
    )
    source = flow.kernel._gaussian_loop(
        {"curvature": 0.0, "inertia": 1.0, "stiffness": 1.0}
    )
    initial = np.full(field.size, -0.5 * source)
    flow.continue_spectral_fixed_point(
        initial,
        [0.0, 0.5, 1.0],
        modes=3,
        tolerance=1.0e-9,
        max_evaluations=20,
    )
    eigenvalues = flow.stability_spectrum(modes=2, step=1.0e-5)

    assert len(flow.continuation) == 3
    assert all(point["success"] for point in flow.continuation)
    assert eigenvalues.shape == (3,)
    assert np.all(np.isfinite(eigenvalues))
    assert np.isfinite(flow.correlation_exponent)


def test_external_momentum_projection_matches_zero_step_limit():
    shell = Phi4GaussianShell(spatial_dimension=1)
    couplings = Phi4GaussianCouplings(mass2=-0.3, quartic=6.0)
    minimum = np.sqrt(-6.0 * couplings.mass2 / couplings.quartic)
    analytic = shell.external_kinetic_rates(minimum, couplings)
    finite = shell.external_kinetic_rates(
        minimum, couplings, momentum_step=2.0e-3
    )

    np.testing.assert_allclose(finite, analytic, rtol=2.0e-5)
    assert analytic[0] > 0.0
    assert analytic[1] < 0.0


def test_exponential_regulator_is_regular_at_zero():
    regulator = ExponentialRegulator()

    np.testing.assert_allclose(regulator.value(0.0), 1.0, atol=1e-15)
    np.testing.assert_allclose(
        regulator.scale_derivative(0.0, 0.1), 1.9, atol=1e-15
    )


def test_gaussian_regulator_is_regular_and_decays():
    regulator = GaussianRegulator()

    np.testing.assert_allclose(regulator.value(0.0), 1.0, atol=1e-15)
    np.testing.assert_allclose(
        regulator.scale_derivative(0.0, 0.1), 1.9, atol=1e-15
    )
    assert regulator.value(10.0) < 5.0e-5


def test_regulator_momentum_derivatives_match_finite_differences():
    points = np.array([0.02, 0.2, 1.0, 4.0])
    first_step = 2.0e-5
    second_step = 2.0e-4
    for regulator in (ExponentialRegulator(), GaussianRegulator()):
        finite_first = (
            regulator.value(points + first_step)
            - regulator.value(points - first_step)
        ) / (2.0 * first_step)
        finite_second = (
            regulator.value(points + second_step)
            - 2.0 * regulator.value(points)
            + regulator.value(points - second_step)
        ) / second_step**2
        np.testing.assert_allclose(
            regulator.first_derivative(points), finite_first, rtol=2.0e-8
        )
        np.testing.assert_allclose(
            regulator.second_derivative(points), finite_second, rtol=2.0e-6
        )


def test_covariant_frg_external_projection_is_isotropic():
    flow = Phi4CovariantFRG(radial_order=80, angular_order=64)
    couplings = Phi4GaussianCouplings(mass2=-0.2, quartic=3.5)
    minimum = np.sqrt(-6.0 * couplings.mass2 / couplings.quartic)
    steps = [0.008, 0.012, 0.018]
    rate0 = flow.kinetic_rate(
        minimum, couplings, axis=0, momentum_steps=steps
    )
    rate1 = flow.kinetic_rate(
        minimum, couplings, axis=1, momentum_steps=steps
    )

    np.testing.assert_allclose(rate0, rate1, rtol=2e-10, atol=1e-12)
    assert rate0 > 0.0


def test_covariant_frg_quartic_fixed_point_residual():
    flow = Phi4CovariantFRG(radial_order=100, angular_order=64)
    couplings = Phi4GaussianCouplings(
        mass2=-0.19425961, quartic=3.60744398
    )
    beta, eta = flow.beta(couplings)

    np.testing.assert_allclose(
        [beta.mass2, beta.quartic], [0.0, 0.0], atol=2e-7
    )
    np.testing.assert_allclose(eta, 0.04793343, rtol=2e-6)


def test_covariant_frg_beta_matches_potential_derivatives():
    flow = Phi4CovariantFRG(radial_order=80, angular_order=48)
    couplings = Phi4GaussianCouplings(mass2=-0.2, quartic=3.5)
    field = np.linspace(-0.05, 0.05, 61)
    coefficients = np.polynomial.polynomial.polyfit(
        field, flow.beta_potential(field, couplings), 8
    )
    fitted = np.array([2.0 * coefficients[2], 24.0 * coefficients[4]])
    beta, _ = flow.beta(couplings)

    np.testing.assert_allclose(
        fitted, [beta.mass2, beta.quartic], rtol=5e-6
    )


def test_standard_frg_lpa_fixed_point_and_stability():
    flow = Phi4FRG(
        order=4,
        approximation="lpa",
        spacetime_dimension=3,
        radial_order=40,
        angular_order=24,
    ).solve_fixed_point(tolerance=1.0e-8)

    assert flow.success
    np.testing.assert_allclose(flow.fixed_beta, 0.0, atol=2.0e-8)
    np.testing.assert_allclose(flow.fixed_eta, 0.0, atol=1.0e-15)
    assert np.count_nonzero(flow.stability_eigenvalues.real > 1.0e-7) == 1
    np.testing.assert_allclose(flow.correlation_exponent, 0.66365, rtol=3.0e-3)


def test_standard_frg_lpa_prime_and_trajectory_workflow():
    flow = Phi4FRG(
        order=4,
        approximation="lpa_prime",
        spacetime_dimension=3,
        radial_order=40,
        angular_order=24,
    ).solve_fixed_point(tolerance=1.0e-7)

    assert flow.success
    assert flow.fixed_eta > 0.0
    np.testing.assert_allclose(flow.correlation_exponent, 0.635, rtol=2.0e-2)
    fixed = flow.fixed_state.copy()
    ell = np.linspace(0.0, 0.25, 5)
    flow.run(fixed, ell)

    assert flow.success
    np.testing.assert_allclose(
        flow.history, np.tile(fixed, (ell.size, 1)), rtol=2.0e-8, atol=2.0e-8
    )
    assert flow.eta_history.shape == ell.shape


def test_covariant_frg_external_projection_supports_three_dimensions():
    flow = Phi4CovariantFRG(
        spacetime_dimension=3, radial_order=40, angular_order=24
    )
    eta = flow.local_anomalous_dimension(curvature=0.5, cubic=2.0)

    assert np.isfinite(eta)
    assert eta > 0.0


def test_covariant_frg_analytic_kinetic_projection_matches_finite_momentum():
    flow = Phi4CovariantFRG(
        spacetime_dimension=3, radial_order=80, angular_order=64
    )
    couplings = Phi4GaussianCouplings(mass2=-0.2, quartic=3.5)
    minimum = np.sqrt(-6.0 * couplings.mass2 / couplings.quartic)
    finite = flow.kinetic_rate(
        minimum,
        couplings,
        momentum_steps=np.array([0.004, 0.006, 0.009, 0.013, 0.018]),
    )
    analytic = flow.local_kinetic_rate(
        Phi4GaussianShell.curvature(minimum, couplings),
        Phi4GaussianShell.third_derivative(minimum, couplings),
    )

    np.testing.assert_allclose(analytic, finite, rtol=2.0e-8)


def test_de2_zero_wavefunction_slope_reproduces_lpa_prime_potential_flow():
    settings = {
        "order": 4,
        "spacetime_dimension": 3,
        "radial_order": 40,
        "angular_order": 24,
    }
    lpa_prime = Phi4FRG(approximation="lpa_prime", **settings)
    de2 = Phi4FRG(approximation="de2", wavefunction_order=1, **settings)
    potential_state = np.array([0.04, 7.0, 50.0, 150.0])
    lpa_rates, lpa_eta = lpa_prime.beta(potential_state)
    de2_rates, de2_eta = de2.beta(np.append(potential_state, 0.0))

    np.testing.assert_allclose(de2_rates[:4], lpa_rates, rtol=2.0e-12, atol=1e-12)
    np.testing.assert_allclose(de2_eta, lpa_eta, rtol=2.0e-12)


def test_de2_fixed_point_normalization_and_trajectory():
    flow = Phi4FRG(
        order=4,
        approximation="de2",
        wavefunction_order=1,
        spacetime_dimension=3,
        radial_order=40,
        angular_order=24,
    ).solve_fixed_point(tolerance=1.0e-7)

    assert flow.success
    np.testing.assert_allclose(flow.fixed_beta, 0.0, atol=2.0e-8)
    np.testing.assert_allclose(flow.fixed_eta, 0.0986, rtol=2.0e-2)
    np.testing.assert_allclose(flow.correlation_exponent, 0.5824, rtol=2.0e-2)
    assert np.count_nonzero(flow.stability_eigenvalues.real > 1.0e-7) == 1
    minimum = np.sqrt(2.0 * flow.fixed_state[0])
    np.testing.assert_allclose(flow.wavefunction(minimum), 1.0, atol=1.0e-14)
    fixed = flow.fixed_state.copy()
    ell = np.linspace(0.0, 0.1, 3)
    flow.run(fixed, ell)

    assert flow.success
    np.testing.assert_allclose(
        flow.history, np.tile(fixed, (ell.size, 1)), rtol=2.0e-8, atol=2.0e-8
    )


def test_thin_1d_shell_has_no_three_boson_residual():
    shell = Phi4GaussianShell(spatial_dimension=1)
    couplings = Phi4GaussianCouplings(mass2=0.2, cubic=0.4, quartic=0.8)
    correction = shell.residual_corrections(
        0.2, couplings, log_width=0.1, quadrature_order=20
    )

    assert correction["three_boson"] == 0.0
    assert correction["four_boson"] < 0.0
    np.testing.assert_allclose(
        correction["total"], correction["four_boson"], atol=1e-18
    )

    wider = shell.residual_corrections(
        0.2, couplings, log_width=0.2, quadrature_order=20
    )
    ratio = wider["four_boson"] / correction["four_boson"]
    assert 7.0 < ratio < 9.5
