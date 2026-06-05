import numpy as np

from pyqed.narg.functional import (
    ConditionalGaussianWavefunctionNARG,
    Phi4LogShellNARG,
    Phi4PeriodicSincNARG,
    Phi4TwoSiteNARG,
    Yukawa1DWavefunctionalNARG,
    periodic_real_fourier_transform,
    sine_basis_values,
    sine_dvr_grid,
    sine_dvr_kinetic_matrix,
    sine_dvr_transform,
    slater_sector_vector,
)


def test_conditional_gaussian_wavefunction_narg_schmidt_kernel():
    toy = ConditionalGaussianWavefunctionNARG(nbasis=32, quadrature_order=100)

    result = toy.schmidt_compress(rank=2)

    np.testing.assert_allclose(np.sum(result.singular_values**2), 1.0, atol=1e-12)
    np.testing.assert_allclose(result.discarded_weight, 0.0, atol=1e-12)
    np.testing.assert_allclose(
        result.boson_branches.T @ (result.weights[:, None] * result.boson_branches),
        np.eye(2),
        atol=1e-11,
    )
    np.testing.assert_allclose(result.fermion_branches @ result.fermion_branches.T, np.eye(2), atol=1e-12)


def test_conditional_gaussian_wavefunction_narg_energy_is_variational():
    toy = ConditionalGaussianWavefunctionNARG(nbasis=40, quadrature_order=140)

    rank1 = toy.schmidt_compress(rank=1)
    rank2 = toy.schmidt_compress(rank=2)

    assert rank1.energy >= rank1.exact_energy - 1e-10
    assert rank2.energy >= rank2.exact_energy - 1e-10
    assert rank2.energy < rank1.energy - 1e-2
    assert rank2.coefficient_norm > 0.999


def test_conditional_gaussian_wavefunction_narg_values_match_kept_weight():
    toy = ConditionalGaussianWavefunctionNARG(nbasis=32, quadrature_order=100)

    result = toy.schmidt_compress(rank=1)
    norm = np.sum(result.weights * np.sum(np.abs(result.wavefunction_values) ** 2, axis=1))

    np.testing.assert_allclose(norm, result.kept_weight, atol=1e-12)
    assert 0.0 < result.discarded_weight < 1.0


def test_sine_dvr_kinetic_has_particle_in_box_spectrum():
    npoints = 5
    length = 3.0

    transform = sine_dvr_transform(npoints)
    kinetic = sine_dvr_kinetic_matrix(npoints, length)
    expected = 0.5 * (np.pi * np.arange(1, npoints + 1) / length) ** 2

    np.testing.assert_allclose(transform.T @ transform, np.eye(npoints), atol=1e-12)
    np.testing.assert_allclose(np.linalg.eigvalsh(kinetic), expected, atol=1e-12)


def test_yukawa_1d_sine_dvr_uses_local_field_vertices():
    toy = Yukawa1DWavefunctionalNARG(
        coupling=0.5,
        scalar_modes=2,
        fermion_modes=4,
        fermion_regulator="sine_dvr",
        oscillator_nbasis=4,
        field_quadrature_order=6,
    )
    x, weights = sine_dvr_grid(4, toy.length)
    scalar_values = sine_basis_values(x, toy.scalar_modes, toy.length)

    np.testing.assert_allclose(toy.x, x, atol=1e-12)
    np.testing.assert_allclose(toy.x_weights, weights, atol=1e-12)
    for mode, vertex in enumerate(toy.scalar_vertices):
        np.testing.assert_allclose(vertex, np.diag(scalar_values[:, mode]), atol=1e-12)
    np.testing.assert_allclose(
        np.linalg.eigvalsh(toy.single_electron_keo),
        0.5 * (np.pi * np.arange(1, 5) / toy.length) ** 2,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        -1j * toy.derivative_matrix,
        (-1j * toy.derivative_matrix).T.conj(),
        atol=1e-12,
    )


def test_periodic_real_fourier_transform_groups_degenerate_pairs():
    transform, wave_numbers, labels = periodic_real_fourier_transform(4, 6.0)

    np.testing.assert_allclose(transform.T @ transform, np.eye(4), atol=1e-12)
    assert labels == [("zero", 0), ("cos", 1), ("sin", 1), ("nyquist", 2)]
    np.testing.assert_allclose(wave_numbers[1], wave_numbers[2], atol=1e-12)


def test_phi4_narg_heff_full_conditional_basis_matches_full_spectrum():
    toy = Phi4TwoSiteNARG(
        active_npoints=5,
        environment_npoints=6,
        field_range=4.0,
        mass2=0.6,
        coupling=0.7,
        stiffness=0.4,
    )

    result = toy.narg_effective_hamiltonian(nbranches=toy.environment_npoints)
    exact = toy.exact_energies()

    np.testing.assert_allclose(result.effective_energies, exact, atol=1e-12)
    np.testing.assert_allclose(
        result.kinetic_dressing[0, :, 0, :],
        np.eye(toy.environment_npoints),
        atol=1e-12,
    )


def test_phi4_narg_heff_improves_with_more_conditional_branches():
    toy = Phi4TwoSiteNARG(
        active_npoints=7,
        environment_npoints=8,
        field_range=4.5,
        mass2=0.5,
        coupling=0.8,
        stiffness=0.5,
    )

    exact = toy.exact_energies(1)[0]
    branch1 = toy.narg_effective_hamiltonian(nbranches=1)
    branch2 = toy.narg_effective_hamiltonian(nbranches=2)
    branch3 = toy.narg_effective_hamiltonian(nbranches=3)

    assert branch1.effective_energies[0] >= exact - 1e-12
    assert branch2.effective_energies[0] >= exact - 1e-12
    assert branch3.effective_energies[0] >= exact - 1e-12
    assert branch2.effective_energies[0] <= branch1.effective_energies[0] + 1e-12
    assert branch3.effective_energies[0] <= branch2.effective_energies[0] + 1e-12


def test_phi4_periodic_sinc_narg_full_conditional_basis_matches_full_spectrum():
    toy = Phi4PeriodicSincNARG(
        spatial_npoints=4,
        amplitude_npoints=4,
        field_range=4.5,
        length=6.0,
        mass2=0.5,
        coupling=0.8,
        active_mode_count=3,
    )

    result = toy.narg_effective_hamiltonian(nbranches=toy.environment_configs.shape[0])
    exact = toy.exact_energies()

    np.testing.assert_allclose(result.effective_energies, exact, atol=1e-11)
    assert [result.mode_labels[index] for index in result.active_modes] == [
        ("zero", 0),
        ("cos", 1),
        ("sin", 1),
    ]


def test_phi4_periodic_sinc_narg_improves_with_more_conditional_branches():
    toy = Phi4PeriodicSincNARG(
        spatial_npoints=4,
        amplitude_npoints=4,
        field_range=4.5,
        length=6.0,
        mass2=0.5,
        coupling=0.8,
        active_mode_count=3,
    )

    exact = toy.exact_energies(1)[0]
    branch1 = toy.narg_effective_hamiltonian(nbranches=1)
    branch3 = toy.narg_effective_hamiltonian(nbranches=3)
    full = toy.narg_effective_hamiltonian(nbranches=toy.environment_configs.shape[0])

    assert branch1.effective_energies[0] >= exact - 1e-12
    assert branch3.effective_energies[0] >= exact - 1e-12
    assert full.effective_energies[0] >= exact - 1e-12
    assert branch3.effective_energies[0] <= branch1.effective_energies[0] + 1e-12
    np.testing.assert_allclose(full.effective_energies[0], exact, atol=1e-11)


def test_phi4_log_shell_narg_groups_shell_pairs_and_weights():
    toy = Phi4LogShellNARG(
        cutoff=4.0,
        log_factor=2.0,
        nshells=2,
        active_shells=1,
        amplitude_npoints=3,
        quadrature_order=96,
    )

    np.testing.assert_allclose(toy.shell_edges, np.array([4.0, 2.0, 1.0]), atol=1e-12)
    np.testing.assert_allclose(toy.shell_representatives, np.sqrt([8.0, 2.0]), atol=1e-12)
    assert toy.mode_labels == [
        ("cos", 0),
        ("sin", 0),
        ("cos", 1),
        ("sin", 1),
    ]
    np.testing.assert_allclose(toy.mode_wave_numbers[0], toy.mode_wave_numbers[1], atol=1e-12)
    np.testing.assert_allclose(toy.mode_wave_numbers[2], toy.mode_wave_numbers[3], atol=1e-12)
    assert [toy.mode_labels[index] for index in toy.active_modes] == [
        ("cos", 1),
        ("sin", 1),
    ]
    assert [toy.mode_labels[index] for index in toy.environment_modes] == [
        ("cos", 0),
        ("sin", 0),
    ]


def test_phi4_log_shell_narg_full_conditional_basis_matches_full_spectrum():
    toy = Phi4LogShellNARG(
        cutoff=4.0,
        log_factor=2.0,
        nshells=2,
        active_shells=1,
        amplitude_npoints=3,
        field_range=4.0,
        mass2=0.5,
        coupling=0.6,
        quadrature_order=96,
    )

    result = toy.narg_effective_hamiltonian(nbranches=toy.environment_configs.shape[0])
    exact = toy.exact_energies()

    np.testing.assert_allclose(result.effective_energies, exact, atol=1e-11)


def test_phi4_log_shell_narg_shell_flow_reports_exact_final_cutoff():
    toy = Phi4LogShellNARG(
        cutoff=4.0,
        log_factor=2.0,
        nshells=2,
        active_shells=0,
        amplitude_npoints=3,
        field_range=4.0,
        mass2=0.5,
        coupling=0.8,
        quadrature_order=96,
    )

    rows = toy.shell_flow_summary(nbranches=2)

    assert len(rows) == 3
    assert [row["active_shells"] for row in rows] == [0, 1, 2]
    assert rows[-1]["branches"] == 1
    assert rows[-1]["dimension"] == toy.amplitude_npoints**toy.nmodes
    assert abs(rows[-1]["error"]) <= 1e-11


def test_phi4_log_shell_iterative_narg_large_kept_space_matches_exact():
    toy = Phi4LogShellNARG(
        cutoff=4.0,
        log_factor=2.0,
        nshells=2,
        active_shells=0,
        amplitude_npoints=3,
        field_range=4.0,
        mass2=0.5,
        coupling=0.6,
        quadrature_order=96,
    )

    result = toy.iterative_shell_narg(kept_dim=300, max_exact_dim=1000)

    np.testing.assert_allclose(result.energies, toy.exact_energies(result.energies.size), atol=1e-11)
    assert result.records[-1]["basis_dim"] == toy.amplitude_npoints**toy.nmodes
    assert result.records[-1]["projected_dim"] == toy.amplitude_npoints**toy.nmodes


def test_phi4_log_shell_iterative_narg_handles_larger_shell_count():
    toy = Phi4LogShellNARG(
        cutoff=8.0,
        log_factor=2.0,
        nshells=4,
        active_shells=0,
        amplitude_npoints=3,
        field_range=4.0,
        mass2=0.5,
        coupling=0.5,
        quadrature_order=160,
    )

    kept8 = toy.iterative_shell_narg(kept_dim=8, max_exact_dim=1000)
    kept16 = toy.iterative_shell_narg(kept_dim=16, max_exact_dim=1000)

    assert kept8.records[-1]["basis_dim"] == 3**8
    assert max(record["projected_dim"] for record in kept8.records) == 8 * 3**2
    assert max(record["projected_dim"] for record in kept16.records) == 16 * 3**2
    assert kept8.records[0]["shell"] == 0
    assert kept8.records[-1]["shell"] == toy.nshells - 1
    assert kept16.energies[0] <= kept8.energies[0] + 1e-12
    assert kept8.exact_energies.size == 0


def test_phi4_log_shell_iterative_narg_diagnostics_are_available():
    toy = Phi4LogShellNARG(
        cutoff=4.0,
        log_factor=2.0,
        nshells=2,
        active_shells=0,
        amplitude_npoints=3,
        field_range=4.0,
        mass2=0.5,
        coupling=0.6,
        quadrature_order=96,
    )

    uv = toy.iterative_shell_narg(kept_dim=8, direction="uv_to_ir", max_exact_dim=1000)
    ir = toy.iterative_shell_narg(kept_dim=8, direction="ir_to_uv", max_exact_dim=1000)
    d_scan = toy.iterative_kept_dim_scan([4, 8], direction="uv_to_ir", max_exact_dim=1000)
    moments = toy.iterative_mode_moments(uv, power=2)
    fit = toy.fit_ir_shell_effective_potential()
    b_scan = toy.log_factor_scan([1.5, 2.0], kept_dim=4)
    cutoff_scan = toy.cutoff_scan([3.0, 4.0], kept_dim=4)

    assert uv.records[0]["shell"] == 0
    assert uv.records[-1]["shell"] == toy.nshells - 1
    assert ir.records[0]["shell"] == toy.nshells - 1
    assert ir.records[-1]["shell"] == 0
    assert d_scan[1]["energy"] <= d_scan[0]["energy"] + 1e-12
    assert ("cos", 0) in moments
    assert ("sin", 1) in moments
    assert {"c0", "omega2_eff", "lambda_eff", "c6"} <= set(fit["coefficients"])
    assert fit["rms_error"] >= 0.0
    assert len(b_scan) == 2
    assert len(cutoff_scan) == 2


def test_phi4_periodic_sinc_dvr_free_theory_converges_to_analytic_spectrum():
    toy = Phi4PeriodicSincNARG(
        spatial_npoints=3,
        amplitude_npoints=9,
        field_range=5.0,
        length=6.0,
        mass2=0.7,
        coupling=0.0,
        active_mode_count=1,
    )
    energies = toy.exact_energies(2)

    np.testing.assert_allclose(energies[0], toy.free_analytic_ground_energy(), atol=7e-3)
    np.testing.assert_allclose(energies[1] - energies[0], toy.free_analytic_gap(), atol=1e-3)


def test_phi4_periodic_sinc_dvr_preserves_z2_parity():
    toy = Phi4PeriodicSincNARG(
        spatial_npoints=4,
        amplitude_npoints=4,
        field_range=4.5,
        length=6.0,
        mass2=0.5,
        coupling=0.8,
        active_mode_count=3,
    )

    parities = toy.z2_parity_expectations(nroots=4)

    np.testing.assert_allclose(np.abs(parities), np.ones(4), atol=1e-10)


def test_phi4_periodic_sinc_dvr_matches_weak_coupling_energy_shift():
    free = Phi4PeriodicSincNARG(
        spatial_npoints=3,
        amplitude_npoints=9,
        field_range=5.0,
        length=6.0,
        mass2=0.7,
        coupling=0.0,
        active_mode_count=1,
    )
    weak = Phi4PeriodicSincNARG(
        spatial_npoints=3,
        amplitude_npoints=9,
        field_range=5.0,
        length=6.0,
        mass2=0.7,
        coupling=0.05,
        active_mode_count=1,
    )

    dvr_shift = weak.exact_energies(1)[0] - free.exact_energies(1)[0]
    perturbative_shift = weak.weak_coupling_first_order_ground_energy() - weak.free_analytic_ground_energy()

    np.testing.assert_allclose(dvr_shift, perturbative_shift, atol=5e-5)


def test_yukawa_1d_wavefunctional_narg_free_limit_matches_exact():
    toy = Yukawa1DWavefunctionalNARG(
        coupling=0.0,
        scalar_modes=2,
        fermion_modes=2,
        oscillator_nbasis=6,
        field_quadrature_order=10,
    )

    result = toy.schmidt_compress(rank=1)

    np.testing.assert_allclose(result.energy, result.exact_energy, atol=1e-12)
    np.testing.assert_allclose(result.kept_weight, 1.0, atol=1e-12)
    np.testing.assert_allclose(result.coefficient_norm, 1.0, atol=1e-12)


def test_yukawa_1d_wavefunctional_narg_schmidt_branches_are_orthonormal():
    toy = Yukawa1DWavefunctionalNARG(
        coupling=0.4,
        scalar_modes=2,
        fermion_modes=2,
        oscillator_nbasis=8,
        field_quadrature_order=10,
    )

    rank1 = toy.schmidt_compress(rank=1)
    rank2 = toy.schmidt_compress(rank=2)

    np.testing.assert_allclose(np.sum(rank2.singular_values**2), 1.0, atol=1e-12)
    np.testing.assert_allclose(
        rank2.boson_branches.conj().T @ (rank2.weights[:, None] * rank2.boson_branches),
        np.eye(2),
        atol=1e-11,
    )
    np.testing.assert_allclose(rank2.fermion_branches @ rank2.fermion_branches.conj().T, np.eye(2), atol=1e-12)
    assert rank2.kept_weight > rank1.kept_weight
    assert rank2.energy >= rank2.exact_energy - 1e-10


def test_yukawa_1d_wavefunctional_narg_rank_can_improve_interacting_energy():
    toy = Yukawa1DWavefunctionalNARG(
        coupling=0.5,
        scalar_modes=1,
        fermion_modes=2,
        oscillator_nbasis=14,
        field_quadrature_order=40,
    )

    rank1 = toy.schmidt_compress(rank=1)
    rank2 = toy.schmidt_compress(rank=2)

    assert rank1.energy >= rank1.exact_energy - 1e-10
    assert rank2.energy >= rank2.exact_energy - 1e-10
    assert rank2.energy < rank1.energy - 1e-3
    assert rank2.kept_weight > 0.99


def test_yukawa_1d_wavefunctional_narg_variational_rank1_optimizes_chi():
    toy = Yukawa1DWavefunctionalNARG(
        coupling=0.5,
        scalar_modes=1,
        fermion_modes=2,
        oscillator_nbasis=10,
        field_quadrature_order=28,
    )

    fixed = toy.schmidt_compress(rank=1)
    variational = toy.variational_rank1(maxiter=80)

    assert variational.success
    assert variational.energy >= variational.exact_energy - 1e-10
    assert variational.energy < fixed.energy - 0.04
    assert variational.centers[0] > 0.1
    assert variational.coefficient_norm > 0.999


def test_yukawa_1d_gaussian_response_free_limit_matches_exact():
    toy = Yukawa1DWavefunctionalNARG(
        coupling=0.0,
        scalar_modes=1,
        fermion_modes=2,
        oscillator_nbasis=10,
    )

    response = toy.gaussian_response_rank1_energy()

    np.testing.assert_allclose(response.energy, response.exact_energy, atol=1e-12)
    np.testing.assert_allclose(response.fluctuation_energy, 0.0, atol=1e-12)
    np.testing.assert_allclose(response.born_huang_energy, 0.0, atol=1e-12)


def test_yukawa_1d_gaussian_response_matches_finite_difference_curvature():
    toy = Yukawa1DWavefunctionalNARG(
        coupling=0.5,
        scalar_modes=1,
        fermion_modes=2,
        oscillator_nbasis=10,
    )
    center = np.array([0.2])
    step = 1e-4

    e0, gradient, hessian, metric = toy.fermion_vacuum_response(center)
    e_plus = toy.fermion_vacuum_response(center + step)[0]
    e_minus = toy.fermion_vacuum_response(center - step)[0]

    np.testing.assert_allclose(gradient[0], (e_plus - e_minus) / (2.0 * step), atol=1e-8)
    np.testing.assert_allclose(hessian[0, 0], (e_plus - 2.0 * e0 + e_minus) / step**2, atol=1e-7)
    assert metric[0, 0] > 0.0


def test_yukawa_1d_conditional_vacuum_overlap_matches_fock_overlap():
    toy = Yukawa1DWavefunctionalNARG(
        coupling=0.5,
        scalar_modes=1,
        fermion_modes=2,
        oscillator_nbasis=10,
    )
    left = np.array([0.1])
    right = np.array([0.35])

    det_overlap = toy.fermion_vacuum_overlap(left, right)
    left_vec = slater_sector_vector(toy.conditional_occupied_orbitals(left), toy.fermion_sector_basis)
    right_vec = slater_sector_vector(toy.conditional_occupied_orbitals(right), toy.fermion_sector_basis)

    np.testing.assert_allclose(det_overlap, np.vdot(left_vec, right_vec), atol=1e-12)


def test_yukawa_1d_overlap_metric_matches_response_metric():
    toy = Yukawa1DWavefunctionalNARG(
        coupling=0.5,
        scalar_modes=1,
        fermion_modes=2,
        oscillator_nbasis=10,
    )

    _, _, _, response_metric = toy.fermion_vacuum_response(np.array([0.2]))
    overlap_metric = toy.fermion_overlap_metric(np.array([0.2]), step=3e-4)

    np.testing.assert_allclose(overlap_metric, response_metric, atol=1e-5)


def test_yukawa_1d_gaussian_response_variational_rank1_optimizes_chi_without_sampling():
    toy = Yukawa1DWavefunctionalNARG(
        coupling=0.5,
        scalar_modes=1,
        fermion_modes=2,
        oscillator_nbasis=10,
        field_quadrature_order=28,
    )

    fixed = toy.gaussian_response_rank1_energy()
    response = toy.variational_rank1_response(maxiter=80)
    sampled = toy.variational_rank1(maxiter=80)

    assert response.success
    assert response.metric_source == "overlap"
    assert response.energy >= response.exact_energy - 1e-10
    assert response.energy < fixed.energy - 0.04
    assert response.centers[0] > 0.1
    np.testing.assert_allclose(response.energy, sampled.energy, atol=3e-3)


def test_yukawa_1d_ts_regulated_rank1_free_limit_matches_gaussian_chi():
    toy = Yukawa1DWavefunctionalNARG(
        coupling=0.0,
        scalar_modes=1,
        fermion_modes=2,
        oscillator_nbasis=10,
        field_quadrature_order=24,
    )

    result = toy.ts_regulated_rank1_energy(cutoff=np.inf, shift=1e-3)
    response = toy.gaussian_response_rank1_energy(include_born_huang=False, metric_source="none")

    np.testing.assert_allclose(result.energy, response.energy, atol=1e-6)
    np.testing.assert_allclose(result.kinetic_weights, np.ones(1), atol=1e-12)
    np.testing.assert_allclose(result.norm, 1.0, atol=1e-12)


def test_yukawa_1d_ts_regulator_changes_only_kinetic_term():
    toy = Yukawa1DWavefunctionalNARG(
        coupling=0.0,
        scalar_modes=1,
        fermion_modes=2,
        oscillator_nbasis=10,
        field_quadrature_order=24,
    )

    unregulated = toy.ts_regulated_rank1_energy(cutoff=np.inf, shift=1e-3)
    regulated = toy.ts_regulated_rank1_energy(cutoff=0.7, shift=1e-3)

    assert 0.0 < regulated.kinetic_weights[0] < 1.0
    assert regulated.kinetic_energy < unregulated.kinetic_energy
    np.testing.assert_allclose(
        regulated.boson_potential_energy,
        unregulated.boson_potential_energy,
        atol=1e-12,
    )
    np.testing.assert_allclose(regulated.fermion_energy, unregulated.fermion_energy, atol=1e-12)


def test_yukawa_1d_ts_regulated_rank1_matches_sampled_interacting_branch():
    toy = Yukawa1DWavefunctionalNARG(
        coupling=0.5,
        scalar_modes=1,
        fermion_modes=2,
        oscillator_nbasis=10,
        field_quadrature_order=28,
    )
    widths = np.array([0.92])
    centers = np.array([0.35])

    ts = toy.ts_regulated_rank1_energy(widths, centers, cutoff=np.inf, shift=1e-3)
    sampled_energy = toy.rank1_energy(widths=widths, centers=centers)[0]

    np.testing.assert_allclose(ts.energy, sampled_energy, atol=1e-6)
    assert ts.kinetic_energy < 1.0


def test_yukawa_1d_gaussian_packet_overlap_dresses_boson_kinetic():
    toy = Yukawa1DWavefunctionalNARG(
        coupling=0.5,
        scalar_modes=1,
        fermion_modes=2,
        oscillator_nbasis=10,
    )
    widths = toy.scalar_frequencies.copy()
    centers = np.array([[-0.1], [0.35], [0.8]])

    _, overlap, parts = toy.gaussian_packet_matrices(widths, centers, return_parts=True)

    np.testing.assert_allclose(
        overlap,
        parts["boson_overlap"] * parts["fermion_overlap"],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        parts["boson_dressed"],
        parts["boson_hamiltonian"] * parts["fermion_overlap"],
        atol=1e-12,
    )
    assert abs(parts["fermion_overlap"][0, 1]) < 0.9999
    assert abs(parts["boson_dressed"][0, 1]) < abs(parts["boson_hamiltonian"][0, 1])


def test_yukawa_1d_gaussian_packet_ground_state_solves_generalized_problem():
    toy = Yukawa1DWavefunctionalNARG(
        coupling=0.5,
        scalar_modes=1,
        fermion_modes=2,
        oscillator_nbasis=10,
    )
    widths = toy.scalar_frequencies.copy()
    centers = np.array([[-0.1], [0.35], [0.8]])

    result = toy.gaussian_packet_ground_state(widths, centers)
    single = toy.gaussian_packet_ground_state(widths, np.array([[0.35]]))

    assert np.isfinite(result.energy)
    assert np.linalg.eigvalsh(result.overlap)[0] > 1e-8
    np.testing.assert_allclose(
        np.vdot(result.coefficients, result.overlap @ result.coefficients),
        1.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.hamiltonian @ result.coefficients,
        result.energy * (result.overlap @ result.coefficients),
        atol=1e-10,
    )
    assert result.energy <= single.energy + 1e-10
