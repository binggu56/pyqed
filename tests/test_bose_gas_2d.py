import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import expm

from pyqed.mps.bose_gas_2d import (
    D2M1HierarchicalCLETTA2D,
    D2M1NestedCLETTA2D,
    D2TripletHNC2D,
    DiluteBoseGas2D,
    FunctionalD2HNC2D,
    GaussianPotentialBoseGas2D,
    HNCELBoseGas2D,
    HierarchicalShellContraction,
    JastrowHNC2D,
    RankOneDensityTransferChannel2D,
    fixed_density_gns_nested_hletta_state,
    fixed_density_nested_hletta_state,
    optimize_condensate_gns_hletta_fixed_density,
    optimize_condensate_nested_hletta_fixed_density,
    optimize_d2_triplet_hnc,
    optimize_gaussian_jastrow_hnc,
    optimize_nested_hletta_fixed_density,
)


def test_dilute_2d_bose_gas_energy_terms_are_separated():
    model = DiluteBoseGas2D(density=1.0, scattering_length=1.0e-3)
    y_value = 1.0 / abs(np.log(1.0e-6))
    leading = 4.0 * np.pi * y_value
    constant = 2.0 * 0.5772156649015329 + 0.5 + np.log(np.pi)

    np.testing.assert_allclose(model.expansion_parameter, y_value, atol=0.0)
    np.testing.assert_allclose(model.leading_energy_density, leading, atol=1.0e-14)
    np.testing.assert_allclose(
        model.logarithmic_energy_density,
        leading * (1.0 - y_value * abs(np.log(y_value))),
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        model.energy_density,
        leading
        * (
            1.0
            - y_value * abs(np.log(y_value))
            + constant * y_value
        ),
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        model.constant_order_energy_correction,
        leading * constant * y_value,
        atol=1.0e-14,
    )


def test_gaussian_2d_bose_gas_bogoliubov_relations():
    model = GaussianPotentialBoseGas2D(
        density=0.2,
        interaction_strength=1.5,
        interaction_range=0.7,
    )
    momenta = np.array([0.1, 0.4, 1.2])
    epsilon = model.free_dispersion(momenta)
    dispersion = model.quasiparticle_dispersion(momenta)

    np.testing.assert_allclose(
        model.static_structure_factor(momenta),
        epsilon / dispersion,
        atol=1.0e-14,
    )
    assert model.bogoliubov_energy_correction_density() < 0.0
    assert model.bogoliubov_energy_density < model.mean_field_energy_density
    assert model.depletion_density() > 0.0


def test_gaussian_real_space_potential_has_requested_fourier_weight():
    model = GaussianPotentialBoseGas2D(
        density=0.3,
        interaction_strength=0.8,
        interaction_range=0.9,
    )
    nodes, weights = leggauss(80)
    cutoff = 9.0 * model.interaction_range
    radii = 0.5 * cutoff * (nodes + 1.0)
    radial_weights = 0.5 * cutoff * weights
    integrated = (
        2.0
        * np.pi
        * radial_weights
        @ (radii * model.interaction_real_space(radii))
    )

    np.testing.assert_allclose(
        integrated,
        model.interaction_strength,
        rtol=2.0e-14,
        atol=2.0e-14,
    )


def test_hnc_zero_jastrow_is_the_uniform_condensate():
    model = GaussianPotentialBoseGas2D(
        density=0.4,
        interaction_strength=0.7,
        interaction_range=0.8,
    )
    state = JastrowHNC2D.gaussian(
        model,
        jastrow_amplitude=0.0,
        jastrow_range=1.0,
        quadrature_points=64,
        tolerance=1.0e-11,
    ).solve()

    assert state.success
    np.testing.assert_allclose(state.pair_distribution, 1.0, atol=0.0)
    np.testing.assert_allclose(state.kinetic_energy_density, 0.0, atol=0.0)
    np.testing.assert_allclose(
        state.potential_energy_density,
        model.mean_field_energy_density,
        rtol=2.0e-13,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(state.structure_factor, 1.0, atol=0.0)


def test_hnc_gaussian_jastrow_is_direct_and_quadrature_converged():
    model = GaussianPotentialBoseGas2D(
        density=1.0,
        interaction_strength=0.5,
        interaction_range=0.7,
    )

    def solve(points):
        return JastrowHNC2D.gaussian(
            model,
            jastrow_amplitude=0.05,
            jastrow_range=1.5,
            quadrature_points=points,
            tolerance=1.0e-10,
        ).solve()

    coarse = solve(64)
    fine = solve(96)

    assert coarse.success
    assert fine.success
    assert fine.pair_distribution[0] < 1.0
    assert np.min(fine.structure_factor) > 0.0
    assert fine.kinetic_energy_density > 0.0
    assert fine.potential_energy_density < model.mean_field_energy_density
    assert fine.energy_density < model.mean_field_energy_density
    np.testing.assert_allclose(
        coarse.energy_density,
        fine.energy_density,
        rtol=5.0e-9,
        atol=2.0e-10,
    )


def test_hnc_gaussian_jastrow_optimization_lowers_mean_field_energy():
    model = GaussianPotentialBoseGas2D(
        density=1.0,
        interaction_strength=0.5,
        interaction_range=0.7,
    )
    state = optimize_gaussian_jastrow_hnc(
        model,
        optimization_quadrature_points=64,
        quadrature_points=96,
        tolerance=1.0e-8,
        maxiter=50,
    )

    assert state.success
    assert state.optimization_success
    assert state.energy_density < model.mean_field_energy_density
    assert state.optimized_parameters["jastrow_amplitude"] > 0.0
    assert state.optimized_parameters["jastrow_range"] > 0.0


def test_hncel_free_function_generates_the_phonon_tail():
    model = GaussianPotentialBoseGas2D(
        density=1.0,
        interaction_strength=0.5,
        interaction_range=0.7,
    )
    coarse = HNCELBoseGas2D(
        model,
        quadrature_points=128,
        tolerance=1.0e-9,
    ).solve()
    fine = HNCELBoseGas2D(
        model,
        quadrature_points=160,
        tolerance=1.0e-9,
    ).solve()

    assert coarse.success
    assert fine.success
    assert coarse.structure_factor[0] < 0.01
    assert abs(coarse.infrared_exponent - 1.0) < 0.02
    assert coarse.infrared_slope_relative_spread < 0.02
    assert coarse.jastrow_tail_amplitude > 0.0
    assert coarse.energy_density < model.mean_field_energy_density
    np.testing.assert_allclose(
        coarse.energy_density,
        fine.energy_density,
        rtol=1.0e-5,
        atol=3.0e-7,
    )


def test_free_d2_triplet_function_exposes_hnc3_boundary_runaway():
    model = GaussianPotentialBoseGas2D(
        density=1.0,
        interaction_strength=0.5,
        interaction_range=0.7,
    )
    state = FunctionalD2HNC2D(
        model,
        quadrature_points=64,
        angular_points=8,
        structure_basis_size=16,
        transverse_basis_size=4,
        transverse_coefficient_bound=0.01,
        initial_transverse_amplitude=0.005,
    ).optimize(maxiter=80, gtol=1.0e-6)

    assert state.scalar_reference.success
    assert state.transverse_boundary_limited
    assert not state.controlled_d2_stationary_point
    assert not state.success
    assert np.max(np.abs(state.transverse_coefficients)) >= 0.00999


def test_d2_triplet_kernel_is_the_connected_virtual_cumulant():
    model = GaussianPotentialBoseGas2D(
        density=1.0,
        interaction_strength=0.5,
        interaction_range=0.7,
    )
    state = D2TripletHNC2D.gaussian(
        model,
        jastrow_amplitude=0.04,
        jastrow_range=1.4,
        transverse_amplitude=0.08,
        transverse_range=0.5,
        virtual_metric=-1.0,
        quadrature_points=32,
        angular_points=8,
    )
    first = state.virtual_kernel(0.3)
    second = state.virtual_kernel(0.9)

    def boundary(matrix):
        return matrix[0, 0]

    explicit = (
        0.5 * boundary(first @ second + second @ first)
        - boundary(first) * boundary(second)
    )
    commutator = first @ second - second @ first

    np.testing.assert_allclose(
        explicit,
        state.connected_virtual_cumulant(0.3, 0.9),
        atol=2.0e-18,
    )
    assert np.linalg.norm(commutator) > 1.0e-6

    positions = np.array(
        [
            [0.0, 0.0],
            [0.73, 0.0],
            [0.91 * np.cos(0.84), 0.91 * np.sin(0.84)],
        ]
    )

    def triplet_value(coordinates):
        first_distance = np.linalg.norm(coordinates[0] - coordinates[1])
        second_distance = np.linalg.norm(coordinates[0] - coordinates[2])
        third_distance = np.linalg.norm(coordinates[1] - coordinates[2])
        return state.triplet_kernel(
            first_distance,
            second_distance,
            third_distance,
        )

    step = 1.0e-4
    numerical_laplacian = 0.0
    for particle in range(3):
        for direction in range(2):
            plus = positions.copy()
            minus = positions.copy()
            plus[particle, direction] += step
            minus[particle, direction] -= step
            numerical_laplacian += (
                triplet_value(plus)
                - 2.0 * triplet_value(positions)
                + triplet_value(minus)
            ) / step**2
    distances = (
        np.linalg.norm(positions[0] - positions[1]),
        np.linalg.norm(positions[0] - positions[2]),
        np.linalg.norm(positions[1] - positions[2]),
    )
    np.testing.assert_allclose(
        state.triplet_total_laplacian(*distances),
        numerical_laplacian,
        rtol=2.0e-6,
        atol=2.0e-8,
    )


def test_d2_zero_transverse_channel_reduces_to_scalar_hnc():
    model = GaussianPotentialBoseGas2D(
        density=1.0,
        interaction_strength=0.5,
        interaction_range=0.7,
    )
    scalar = JastrowHNC2D.gaussian(
        model,
        jastrow_amplitude=0.046,
        jastrow_range=1.46,
        quadrature_points=64,
        tolerance=1.0e-9,
    ).solve()
    d2_state = D2TripletHNC2D.gaussian(
        model,
        jastrow_amplitude=0.046,
        jastrow_range=1.46,
        transverse_amplitude=0.0,
        transverse_range=0.5,
        quadrature_points=64,
        tolerance=1.0e-9,
    ).solve()

    assert scalar.success
    assert d2_state.success
    np.testing.assert_allclose(
        d2_state.energy_density,
        scalar.energy_density,
        rtol=0.0,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        d2_state.triplet_kinetic_energy_density,
        0.0,
        atol=0.0,
    )


def test_d2_triplet_hnc_lowers_the_scalar_hnc_estimate():
    model = GaussianPotentialBoseGas2D(
        density=1.0,
        interaction_strength=0.5,
        interaction_range=0.7,
    )
    pair_state = JastrowHNC2D.gaussian(
        model,
        jastrow_amplitude=0.04633314744327184,
        jastrow_range=1.460054540531644,
        quadrature_points=96,
        tolerance=1.0e-9,
    ).solve()
    state = optimize_d2_triplet_hnc(
        model,
        pair_state=pair_state,
        optimization_quadrature_points=64,
        optimization_angular_points=12,
        quadrature_points=96,
        angular_points=20,
        tolerance=1.0e-8,
        maxiter=35,
    )

    assert state.success
    assert state.d2_energy_gain_density > 5.0e-5
    assert state.triplet_kinetic_energy_density > 0.0
    assert state.potential_energy_density < pair_state.potential_energy_density
    assert state.optimized_parameters["transverse_amplitude"] > 0.0


def test_exact_squeezing_minimizes_quadratic_bogoliubov_energy():
    model = GaussianPotentialBoseGas2D(
        density=0.2,
        interaction_strength=1.5,
        interaction_range=0.7,
    )


def test_full_gaussian_interacting_energy_has_fixed_density_vacuum_limit():
    model = GaussianPotentialBoseGas2D(
        density=0.2,
        interaction_strength=1.5,
        interaction_range=0.7,
    )
    energy = model.full_gaussian_energy_density_for_squeezing(
        lambda momentum: np.zeros_like(momentum),
        quadrature_points=32,
    )

    np.testing.assert_allclose(
        energy,
        model.mean_field_energy_density,
        atol=1.0e-14,
    )


def test_full_gaussian_optimization_is_variational_and_finite():
    model = GaussianPotentialBoseGas2D(
        density=0.08,
        interaction_strength=0.7,
        interaction_range=1.0,
    ).optimize_full_gaussian(quadrature_points=36, maxiter=120)

    assert model.success
    assert np.isfinite(model.full_gaussian_energy_density)
    assert model.full_gaussian_energy_density <= model.mean_field_energy_density
    assert model.squeezing_amplitude > 0.0
    assert model.squeezing_momentum_scale > 0.0


def test_full_gaussian_radial_quadrature_converges():
    model = GaussianPotentialBoseGas2D(
        density=0.2,
        interaction_strength=1.5,
        interaction_range=0.7,
    )

    def trial(momentum):
        return 0.8 * model.squeezing(0.95 * momentum)

    coarse = model.full_gaussian_energy_density_for_squeezing(
        trial,
        quadrature_points=40,
    )
    fine = model.full_gaussian_energy_density_for_squeezing(
        trial,
        quadrature_points=72,
    )

    np.testing.assert_allclose(coarse, fine, rtol=5.0e-6, atol=2.0e-8)


def test_density_transfer_and_direct_wick_gaussian_energies_agree():
    model = GaussianPotentialBoseGas2D(
        density=0.12,
        interaction_strength=0.9,
        interaction_range=0.85,
    )

    def trial(momentum):
        return 0.22 * np.exp(-0.6 * np.asarray(momentum) ** 2)

    momenta = np.array([0.0, 0.4, 1.1])
    np.testing.assert_allclose(
        model.density_transfer_profile(momenta) ** 2,
        model.interaction_momentum(momenta),
        atol=2.0e-15,
    )

    direct = model.full_gaussian_energy_density_for_squeezing(
        trial,
        quadrature_points=72,
    )
    density_transfer = model.density_transfer_energy_density_for_squeezing(
        trial,
        radial_points=48,
        angular_points=48,
    )

    np.testing.assert_allclose(
        density_transfer,
        direct,
        rtol=2.0e-7,
        atol=2.0e-9,
    )
    variational = model.quadratic_energy_density_for_squeezing(model.squeezing)

    np.testing.assert_allclose(
        variational,
        model.bogoliubov_energy_density,
        atol=2.0e-11,
    )


def test_hierarchical_shell_contraction_integrates_constant_2d_generator():
    contraction = HierarchicalShellContraction(
        energy_cutoff=1.7,
        radial_points=8,
        angular_points=10,
    )
    generator = np.diag([-0.2, 0.07])

    def angular_generator(_energy, _theta, radial_width):
        return radial_width * generator

    left = np.array([1.0, 1.0])
    right = np.array([0.6, 0.4])
    value, environment = contraction.contract(
        angular_generator,
        left_boundary=left,
        right_boundary=right,
    )
    exact_environment = expm(2.0 * np.pi * 1.7 * generator) @ right

    np.testing.assert_allclose(environment, exact_environment, atol=2.0e-13)
    np.testing.assert_allclose(value, left @ exact_environment, atol=2.0e-13)


def test_hierarchical_shell_contraction_keeps_angular_ordering():
    contraction = HierarchicalShellContraction(
        energy_cutoff=0.3,
        radial_points=4,
        angular_points=48,
    )
    first = np.array([[0.0, 1.0], [0.0, 0.0]])
    second = np.array([[0.0, 0.0], [1.0, 0.0]])

    def ordered_generator(_energy, theta, radial_width):
        return radial_width * (first if theta < np.pi else second)

    def reversed_generator(_energy, theta, radial_width):
        return radial_width * (second if theta < np.pi else first)

    left = np.array([1.0, 0.0])
    right = np.array([1.0, 0.0])
    ordered, _ = contraction.contract(
        ordered_generator,
        left_boundary=left,
        right_boundary=right,
    )
    reversed_value, _ = contraction.contract(
        reversed_generator,
        left_boundary=left,
        right_boundary=right,
    )

    assert abs(ordered - reversed_value) > 1.0e-3


def test_d2_m1_hletta_has_explicit_virtual_memory_and_vacuum():
    contraction = HierarchicalShellContraction(
        energy_cutoff=0.4,
        radial_points=3,
        angular_points=6,
    )
    state = D2M1HierarchicalCLETTA2D(
        contraction=contraction,
        q_matrix=np.diag([-0.2, -0.5]),
        r_matrix=np.array([[0.0, 0.12], [0.08, 0.0]]),
        tie_matrix=np.array([[0.0, 0.04], [0.03, 0.0]]),
        memory_decay=0.7,
        radial_decay=0.9,
        angular_momentum=1,
    )
    q_combined, r_combined = state.combined_matrices()
    left, right = state.boundary_vectors()

    assert state.bond_dim == 2
    assert state.num_memory_modes == 1
    assert state.memory_dim == 2
    assert state.effective_bond_dim == 4
    assert state.transfer_dim == 16
    assert q_combined.shape == (4, 4)
    assert r_combined.shape == (4, 4)
    assert left.shape == right.shape == (16,)
    assert left[0] == right[0] == 1.0
    assert np.isfinite(state.norm())
    assert state.particle_number() >= 0.0
    assert state.kinetic_energy() >= 0.0
    assert np.isfinite(state.antipodal_pair_expectation())


def test_additive_observable_matches_counting_field_finite_difference():
    contraction = HierarchicalShellContraction(
        energy_cutoff=0.35,
        radial_points=3,
        angular_points=5,
    )
    state = D2M1HierarchicalCLETTA2D(
        contraction=contraction,
        q_matrix=np.diag([-0.2, -0.45]),
        r_matrix=np.array([[0.0, 0.1], [0.07, 0.0]]),
        tie_matrix=np.array([[0.0, 0.03], [0.02, 0.0]]),
        memory_decay=0.8,
        radial_decay=0.6,
    )
    left, right = state.boundary_vectors()

    def sourced_norm(source):
        def generator(energy, theta, radial_width):
            return state.angular_generator(energy, theta, radial_width) + (
                source
                * state.occupation_generator(energy, theta, radial_width)
            )

        value, _ = contraction.contract(
            generator,
            left_boundary=left,
            right_boundary=right,
        )
        return value

    step = 1.0e-5
    derivative = (sourced_norm(step) - sourced_norm(-step)) / (2.0 * step)
    finite_difference = derivative / sourced_norm(0.0)
    np.testing.assert_allclose(
        state.particle_number(),
        finite_difference.real,
        rtol=2.0e-8,
        atol=2.0e-10,
    )


def test_hletta_observables_match_coherent_continuum_state():
    cutoff = 0.7
    amplitude = 0.12
    radial_decay = 0.4
    state = D2M1HierarchicalCLETTA2D(
        contraction=HierarchicalShellContraction(
            energy_cutoff=cutoff,
            radial_points=12,
            angular_points=10,
        ),
        q_matrix=-0.1 * np.eye(2),
        r_matrix=amplitude * np.eye(2),
        tie_matrix=np.zeros((2, 2)),
        memory_decay=0.8,
        radial_decay=radial_decay,
        angular_momentum=0,
    )
    exponent = 2.0 * radial_decay / cutoff
    radial_norm = (1.0 - np.exp(-exponent * cutoff)) / exponent
    radial_kinetic = (
        1.0
        - np.exp(-exponent * cutoff) * (1.0 + exponent * cutoff)
    ) / exponent**2

    np.testing.assert_allclose(
        state.particle_number(),
        2.0 * np.pi * amplitude**2 * radial_norm,
        rtol=2.0e-12,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        state.kinetic_energy(),
        2.0 * np.pi * amplitude**2 * radial_kinetic,
        rtol=2.0e-12,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        state.antipodal_pair_expectation(),
        np.pi * amplitude**2 * radial_norm,
        rtol=2.0e-12,
        atol=2.0e-13,
    )

    energies, _ = state.contraction.radial_quadrature()
    insertions = [
        (2, 0.3, "creation"),
        (4, 1.2, "annihilation"),
        (7, 2.1, "creation"),
        (9, 5.0, "annihilation"),
    ]
    expected = 1.0 + 0.0j
    for radial_index, theta, operator in insertions:
        field = (
            amplitude
            * np.exp(-radial_decay * energies[radial_index] / cutoff)
            * np.exp(1.0j * state.angular_momentum * theta)
        )
        expected *= field.conjugate() if operator == "creation" else field
    np.testing.assert_allclose(
        state.field_correlation(insertions),
        expected,
        rtol=3.0e-12,
        atol=3.0e-13,
    )


def test_hletta_quadratic_bogoliubov_shell_functional_is_finite():
    model = GaussianPotentialBoseGas2D(
        density=0.15,
        interaction_strength=0.8,
        interaction_range=0.9,
    )
    state = D2M1HierarchicalCLETTA2D(
        contraction=HierarchicalShellContraction(
            energy_cutoff=2.0,
            radial_points=4,
            angular_points=6,
        ),
        q_matrix=np.diag([-0.25, -0.6]),
        r_matrix=np.array([[0.0, 0.08], [0.05, 0.0]]),
        tie_matrix=np.array([[0.0, 0.03], [0.02, 0.0]]),
        memory_decay=0.9,
        radial_decay=1.1,
    )

    energy = state.bogoliubov_shell_functional(model)

    assert np.isfinite(energy)


def test_nested_d2_m1_shell_channel_has_true_inner_and_outer_dimensions():
    state = D2M1NestedCLETTA2D(
        contraction=HierarchicalShellContraction(
            energy_cutoff=0.8,
            radial_points=6,
            angular_points=8,
        ),
        q_matrix=np.diag([-0.1, -0.3]),
        r_matrix=np.array([[0.0, 0.08], [0.05, 0.0]]),
        tie_matrix=np.array([[0.0, 0.04], [0.03, 0.0]]),
        angular_memory_decay=0.7,
        radial_decay=0.6,
    )
    energies, widths = state.contraction.radial_quadrature()
    channel = state.shell_channel(energies[0], widths[0])

    assert state.inner_bond_dim == 4
    assert state.inner_transfer_dim == 16
    assert state.outer_transfer_dim == 4
    assert channel.shape == (4, 4)
    assert np.isfinite(state.norm())


def test_nested_shell_channel_is_identity_plus_order_dE():
    state = D2M1NestedCLETTA2D(
        contraction=HierarchicalShellContraction(
            energy_cutoff=1.0,
            radial_points=4,
            angular_points=6,
        ),
        q_matrix=np.diag([-0.1, -0.25]),
        r_matrix=np.array([[0.0, 0.06], [0.04, 0.0]]),
        tie_matrix=np.array([[0.0, 0.03], [0.02, 0.0]]),
        angular_memory_decay=0.8,
        radial_decay=0.5,
    )
    identity = np.eye(state.outer_transfer_dim)
    coarse = np.linalg.norm(state.shell_channel(0.4, 2.0e-4) - identity)
    fine = np.linalg.norm(state.shell_channel(0.4, 1.0e-4) - identity)

    np.testing.assert_allclose(coarse / fine, 2.0, rtol=3.0e-3)


def test_nested_effective_generator_matches_small_finite_shell():
    state = D2M1NestedCLETTA2D(
        contraction=HierarchicalShellContraction(
            energy_cutoff=1.0,
            radial_points=5,
            angular_points=6,
        ),
        q_matrix=np.diag([-0.12, -0.28]),
        r_matrix=np.array([[0.0, 0.05], [0.03, 0.0]]),
        tie_matrix=np.array([[0.0, 0.025], [0.018, 0.0]]),
        angular_memory_decay=0.9,
        radial_decay=0.4,
        generator_step=5.0e-4,
    )
    width = 2.0e-5
    identity = np.eye(state.outer_transfer_dim)
    finite_generator = (state.shell_channel(0.35, width) - identity) / width

    np.testing.assert_allclose(
        state.effective_outer_generator(0.35),
        finite_generator,
        rtol=2.0e-4,
        atol=1.0e-6,
    )


def test_nested_one_body_sources_match_coherent_continuum_state():
    cutoff = 0.75
    amplitude = 0.11
    radial_decay = 0.45
    state = D2M1NestedCLETTA2D(
        contraction=HierarchicalShellContraction(
            energy_cutoff=cutoff,
            radial_points=10,
            angular_points=8,
        ),
        q_matrix=-0.12 * np.eye(2),
        r_matrix=amplitude * np.eye(2),
        tie_matrix=np.zeros((2, 2)),
        angular_memory_decay=0.9,
        radial_decay=radial_decay,
        generator_step=7.0e-4,
    )
    exponent = 2.0 * radial_decay / cutoff
    radial_norm = (1.0 - np.exp(-exponent * cutoff)) / exponent
    radial_kinetic = (
        1.0
        - np.exp(-exponent * cutoff) * (1.0 + exponent * cutoff)
    ) / exponent**2

    np.testing.assert_allclose(
        state.particle_number(),
        2.0 * np.pi * amplitude**2 * radial_norm,
        rtol=8.0e-6,
        atol=2.0e-8,
    )
    np.testing.assert_allclose(
        state.kinetic_energy(),
        2.0 * np.pi * amplitude**2 * radial_kinetic,
        rtol=8.0e-6,
        atol=2.0e-8,
    )

    insertions = [
        (0.11, 0.2, "creation"),
        (0.27, 1.1, "annihilation"),
        (0.46, 2.4, "creation"),
        (0.69, 5.2, "annihilation"),
    ]
    expected = 1.0 + 0.0j
    for energy, theta, operator in insertions:
        field = (
            amplitude
            * np.exp(-radial_decay * energy / cutoff)
            * np.exp(1.0j * state.angular_momentum * theta)
        )
        expected *= field.conjugate() if operator == "creation" else field
    np.testing.assert_allclose(
        state.field_correlation(insertions, propagation_points=8),
        expected,
        rtol=2.0e-5,
        atol=2.0e-8,
    )
    np.testing.assert_allclose(
        state.local_density(0.31, 0.7, propagation_points=8),
        amplitude**2 * np.exp(-2.0 * radial_decay * 0.31 / cutoff),
        rtol=2.0e-5,
        atol=2.0e-8,
    )

    momentum_transfer = 0.25
    density_radial_points = 2
    density_angular_points = 3

    def coherent_density_overlap(num_radial, transfer_sign):
        radial_nodes, radial_weights = leggauss(num_radial)
        energies = 0.5 * cutoff * (radial_nodes + 1.0)
        energy_weights = 0.5 * cutoff * radial_weights
        angular_nodes, angular_weights = leggauss(density_angular_points)
        angles = np.pi * (angular_nodes + 1.0)
        angle_weights = np.pi * angular_weights
        total = 0.0
        for energy, energy_weight in zip(energies, energy_weights):
            momentum = np.sqrt(energy)
            shifted_energy = (
                momentum**2
                + momentum_transfer**2
                + transfer_sign
                * 2.0
                * momentum
                * momentum_transfer
                * np.cos(angles)
            )
            inside = shifted_energy < cutoff
            total += energy_weight * np.sum(
                angle_weights[inside]
                * amplitude**2
                * np.exp(
                    -radial_decay
                    * (energy + shifted_energy[inside])
                    / cutoff
                )
            )
        return total

    expected_structure = coherent_density_overlap(2, 1.0) * (
        coherent_density_overlap(3, -1.0)
    )
    np.testing.assert_allclose(
        state.normal_ordered_density_structure(
            momentum_transfer,
            radial_points=density_radial_points,
            angular_points=density_angular_points,
            propagation_points=3,
        ),
        expected_structure,
        rtol=3.0e-5,
        atol=3.0e-8,
    )


def test_nested_hletta_evaluates_rank_one_density_transfer_energy():
    state = D2M1NestedCLETTA2D(
        contraction=HierarchicalShellContraction(
            energy_cutoff=0.8,
            radial_points=6,
            angular_points=6,
        ),
        q_matrix=np.diag([-0.16, -0.42]),
        r_matrix=np.array([[0.0, 0.09], [0.06, 0.0]]),
        tie_matrix=np.array([[0.0, 0.035], [0.025, 0.0]]),
        angular_memory_decay=0.75,
        radial_decay=0.7,
    )
    channel = RankOneDensityTransferChannel2D(
        radial_profile=lambda momentum: 0.5 * np.exp(-0.2 * momentum**2),
        momentum_cutoff=2.0 * np.sqrt(0.8),
        radial_points=8,
    )
    result = state.evaluate_rank_one_density_transfer(
        channel,
        structure_radial_points=2,
        structure_angular_points=3,
        propagation_points=2,
    )

    assert result is state
    assert state.density_transfer_structure.shape == (8,)
    assert state.particle_density >= 0.0
    assert state.kinetic_energy_density >= 0.0
    assert state.interaction_energy_density >= -1.0e-10
    np.testing.assert_allclose(
        state.energy_density,
        state.kinetic_energy_density + state.interaction_energy_density,
        atol=1.0e-14,
    )


def test_nested_hletta_field_scale_solves_density_as_hard_constraint():
    state = fixed_density_nested_hletta_state(
        HierarchicalShellContraction(
            energy_cutoff=0.8,
            radial_points=5,
            angular_points=5,
        ),
        target_density=0.05,
        area=2.0,
        tie_ratio=0.25,
        radial_decay=0.8,
    )

    np.testing.assert_allclose(state.particle_density, 0.05, rtol=2.0e-7)
    assert state.field_amplitude > 0.0


def test_fixed_density_optimizer_reports_extensivity_audit():
    model = GaussianPotentialBoseGas2D(
        density=0.08,
        interaction_strength=0.7,
        interaction_range=1.0,
    )
    state = optimize_nested_hletta_fixed_density(
        model,
        target_density=0.08,
        area=1.0,
        energy_cutoff=0.8,
        radial_points=4,
        angular_points=4,
        channel_points=8,
        structure_radial_points=2,
        structure_angular_points=3,
        propagation_points=2,
        maxiter=2,
    )

    np.testing.assert_allclose(state.particle_density, 0.08, rtol=3.0e-7)
    assert np.isfinite(state.energy_density)
    assert state.energy_density <= state.initial_energy_density + 1.0e-10
    assert np.isfinite(state.area_drift)
    assert isinstance(state.thermodynamic_valid, bool)


def test_nested_antipodal_pair_phase_rotates_without_changing_density():
    contraction = HierarchicalShellContraction(
        energy_cutoff=0.7,
        radial_points=4,
        angular_points=4,
    )
    real_state = fixed_density_nested_hletta_state(
        contraction,
        target_density=0.015,
        area=1.0,
        tie_ratio=0.3,
        radial_decay=0.8,
        field_phase=0.0,
    )
    shifted_state = fixed_density_nested_hletta_state(
        contraction,
        target_density=0.015,
        area=1.0,
        tie_ratio=0.3,
        radial_decay=0.8,
        field_phase=0.5 * np.pi,
    )
    real_pair = real_state.antipodal_pair_expectation(
        angular_points=3, propagation_points=2
    )
    shifted_pair = shifted_state.antipodal_pair_expectation(
        angular_points=3, propagation_points=2
    )

    np.testing.assert_allclose(
        shifted_state.particle_density, real_state.particle_density, rtol=2.0e-7
    )
    np.testing.assert_allclose(shifted_pair, -real_pair, rtol=2.0e-5, atol=2.0e-8)


def test_condensate_shifted_vacuum_is_the_pure_mean_field_state():
    density = 0.08
    model = GaussianPotentialBoseGas2D(
        density=density,
        interaction_strength=0.7,
        interaction_range=1.0,
    )
    state = fixed_density_nested_hletta_state(
        HierarchicalShellContraction(
            energy_cutoff=0.8,
            radial_points=4,
            angular_points=4,
        ),
        target_density=0.0,
        area=1.0,
        field_phase=0.5 * np.pi,
    )
    channel = RankOneDensityTransferChannel2D(
        radial_profile=model.density_transfer_profile,
        momentum_cutoff=2.0 * np.sqrt(0.8),
        radial_points=8,
    )
    state.evaluate_condensate_shifted_rank_one(
        channel,
        condensate_density=density,
        structure_radial_points=2,
        structure_angular_points=3,
        pair_angular_points=3,
        propagation_points=2,
    )

    np.testing.assert_allclose(state.fluctuation_density, 0.0, atol=2.0e-12)
    np.testing.assert_allclose(state.particle_density, density, atol=2.0e-12)
    np.testing.assert_allclose(
        state.energy_density, model.mean_field_energy_density, atol=2.0e-10
    )


def test_nested_replication_scale_makes_density_extensive():
    cutoff = 0.7
    amplitude = 0.08
    radial_decay = 0.5
    densities = []
    for area in (1.0, 6.0):
        state = D2M1NestedCLETTA2D(
            contraction=HierarchicalShellContraction(
                energy_cutoff=cutoff,
                radial_points=7,
                angular_points=4,
            ),
            q_matrix=-0.1 * np.eye(2),
            r_matrix=amplitude * np.eye(2),
            tie_matrix=np.zeros((2, 2)),
            angular_memory_decay=0.8,
            radial_decay=radial_decay,
            replication_scale=area,
        )
        densities.append(state.particle_number() / area)

    np.testing.assert_allclose(densities[1], densities[0], rtol=2.0e-6)


def test_fixed_density_gns_state_has_no_vacuum_boundary_density_constraint():
    state = fixed_density_gns_nested_hletta_state(
        HierarchicalShellContraction(
            energy_cutoff=0.8,
            radial_points=4,
            angular_points=4,
        ),
        target_density=0.03,
        field_phase=0.5 * np.pi,
    )

    np.testing.assert_allclose(state.gns_particle_density(), 0.03, rtol=3.0e-7)
    assert state.uses_gns_boundaries
    assert state.minimum_gns_transfer_gap > 0.0


def test_condensate_gns_hletta_optimizer_enforces_density_without_area_drift():
    model = GaussianPotentialBoseGas2D(
        density=1.0,
        interaction_strength=0.5,
        interaction_range=0.7,
    )
    state = optimize_condensate_gns_hletta_fixed_density(
        model,
        energy_cutoff=0.8,
        radial_points=4,
        angular_points=4,
        channel_points=8,
        structure_radial_points=2,
        structure_angular_points=3,
        pair_angular_points=3,
        maxiter=2,
    )

    np.testing.assert_allclose(
        state.condensate_density + state.fluctuation_density,
        model.density,
        rtol=3.0e-7,
    )
    assert 0.0 <= state.condensate_fraction <= 1.0
    assert state.condensate_anomalous_interaction_density <= 1.0e-10
    assert np.isfinite(state.energy_density)
    np.testing.assert_allclose(state.area_drift, 0.0, atol=0.0)
    np.testing.assert_allclose(
        state.thermodynamic_energy_density, state.energy_density, atol=0.0
    )
    assert state.minimum_gns_transfer_gap > 1.0e-7
    assert state.thermodynamic_valid
    assert isinstance(state.thermodynamic_valid, bool)
