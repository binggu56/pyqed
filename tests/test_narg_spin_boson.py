import numpy as np
import sys
from pathlib import Path

EXAMPLES_MPS = Path(__file__).resolve().parents[1] / "examples" / "mps"
if str(EXAMPLES_MPS) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_MPS))

from spin_boson_narg_field_scaling_s07 import grid_collapse_over_y_h

from pyqed.narg import (
    CKLocalPotentialParameters,
    CKLocalPotentialState,
    ContinuumAnalyticalNARGParameters,
    ContinuumAnalyticalNARGState,
    MatrixPolynomialPESState,
    OneModeNARGShellParameters,
    PolynomialPESState,
    SpinBosonWilsonChain,
    SpinBosonWilsonAdiabaticNARG,
    SpinBosonWilsonNARG,
    analytical_harmonic_narg_flow,
    analytical_landau_narg_flow,
    boson_displaced_dvr_operators,
    boson_dvr_operators,
    ck_local_potential_beta,
    ck_local_potential_linearization,
    conditional_one_mode_narg_shell_projection,
    continuum_landau_beta,
    continuum_landau_linearization,
    critical_harmonic_impurity_stiffness,
    discrete_matrix_wilson_narg_flow,
    extract_spin_boson_fpes_observable,
    estimate_harmonic_thermal_exponent,
    fit_order_parameter_exponent,
    harmonic_oscillator_displacement_overlap,
    integrate_ck_local_potential_flow,
    integrate_continuum_landau_flow,
    integrate_matrix_one_mode_narg_flow,
    log_discretized_spin_boson_wilson_chain,
    matrix_one_mode_narg_effective_hamiltonian,
    matrix_one_mode_narg_beta,
    matrix_one_mode_narg_linearization,
    narg_rescaled_spectrum_flow,
    overlap_dressed_phi_kinetic,
    landau_critical_exponents,
    one_mode_narg_beta,
    one_mode_narg_linearization,
    one_mode_narg_shell_energy,
    polynomial_one_mode_narg_beta,
    polynomial_one_mode_narg_shell_energy,
    scan_spin_boson_alpha,
    scan_spin_boson_fixed_point,
    scan_spin_boson_fixed_point_flows,
    scan_spin_boson_fpes_alpha,
    scan_spin_boson_fpes_observables,
    scan_spin_boson_gap_thresholds,
    sine_dvr_boson_operators,
    spin_boson_mode_pes,
    spin_boson_narg_step_observables,
    spin_boson_wilson_dmrg,
    spin_boson_wilson_exact,
    spin_boson_wilson_exact_magnetization,
    spin_boson_wilson_hamiltonian,
    spin_boson_wilson_mpo,
    star_to_wilson_chain,
)
from pyqed.narg.analytical import _matrix_one_mode_hamiltonian


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


def test_spin_boson_iterative_diagonalizer_matches_dense_low_states():
    rng = np.random.default_rng(1234)
    raw = rng.normal(size=(24, 24)) + 1.0j * rng.normal(size=(24, 24))
    hamiltonian = raw + raw.T.conj()

    dense_values, dense_vectors = SpinBosonWilsonNARG._diagonalize(
        hamiltonian,
        5,
        method="dense",
    )
    iterative_values, iterative_vectors = SpinBosonWilsonNARG._diagonalize(
        hamiltonian,
        5,
        method="iterative",
        tol=1.0e-12,
    )

    np.testing.assert_allclose(iterative_values, dense_values, atol=1.0e-9)
    projected = iterative_vectors.conj().T @ hamiltonian @ iterative_vectors
    np.testing.assert_allclose(
        projected,
        np.diag(iterative_values),
        atol=1.0e-8,
    )


def test_spin_boson_structured_product_diagonalizer_matches_dense_kron():
    rng = np.random.default_rng(5678)
    raw_block = rng.normal(size=(6, 6)) + 1.0j * rng.normal(size=(6, 6))
    raw_site = rng.normal(size=(5, 5)) + 1.0j * rng.normal(size=(5, 5))
    block = raw_block + raw_block.T.conj()
    site = raw_site + raw_site.T.conj()
    left = rng.normal(size=(6, 6)) + 1.0j * rng.normal(size=(6, 6))
    right = rng.normal(size=(5, 5)) + 1.0j * rng.normal(size=(5, 5))
    coupling_terms = ((left, right), (left.T.conj(), right.T.conj()))

    dense_values, dense_vectors = SpinBosonWilsonNARG._diagonalize_product(
        block,
        site,
        coupling_terms,
        4,
        method="dense",
    )
    iterative_values, iterative_vectors = SpinBosonWilsonNARG._diagonalize_product(
        block,
        site,
        coupling_terms,
        4,
        method="iterative",
        tol=1.0e-12,
        ncv=9,
    )
    lobpcg_values, lobpcg_vectors = SpinBosonWilsonNARG._diagonalize_product(
        block,
        site,
        coupling_terms,
        4,
        method="lobpcg",
        tol=1.0e-10,
        maxiter=40,
        initial_vectors=dense_vectors,
    )

    dense_hamiltonian = SpinBosonWilsonNARG._dense_product_hamiltonian(
        block,
        site,
        coupling_terms,
    )
    np.testing.assert_allclose(iterative_values, dense_values, atol=1.0e-9)
    np.testing.assert_allclose(
        iterative_vectors.conj().T @ dense_hamiltonian @ iterative_vectors,
        np.diag(iterative_values),
        atol=1.0e-8,
    )
    np.testing.assert_allclose(lobpcg_values, dense_values, atol=1.0e-8)
    np.testing.assert_allclose(
        lobpcg_vectors.conj().T @ dense_hamiltonian @ lobpcg_vectors,
        np.diag(lobpcg_values),
        atol=1.0e-7,
    )

    site_probe = rng.normal(size=(5, 5)) + 1.0j * rng.normal(size=(5, 5))
    block_probe = rng.normal(size=(6, 6)) + 1.0j * rng.normal(size=(6, 6))
    np.testing.assert_allclose(
        SpinBosonWilsonNARG._project_product_operator(dense_vectors, 6, site_probe),
        dense_vectors.conj().T @ np.kron(np.eye(6), site_probe) @ dense_vectors,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        SpinBosonWilsonNARG._project_block_operator(dense_vectors, block_probe, 5),
        dense_vectors.conj().T @ np.kron(block_probe, np.eye(5)) @ dense_vectors,
        atol=1.0e-12,
    )


def test_analytical_harmonic_narg_one_mode_critical_stiffness():
    chain = SpinBosonWilsonChain(
        onsite=np.array([2.0]),
        hopping=np.array([]),
        impurity_coupling=np.sqrt(2.0) * 0.3,
    )

    critical = critical_harmonic_impurity_stiffness(chain)
    np.testing.assert_allclose(critical, 0.3**2 / 2.0, atol=1e-14)

    result = analytical_harmonic_narg_flow(chain, impurity_stiffness=0.09)
    np.testing.assert_allclose(result.curvatures, [1.0], atol=1e-14)
    np.testing.assert_allclose(result.susceptibilities, [1.0], atol=1e-14)


def test_analytical_landau_narg_one_mode_quartic_response():
    chain = SpinBosonWilsonChain(
        onsite=np.array([2.0]),
        hopping=np.array([]),
        impurity_coupling=np.sqrt(2.0) * 0.3,
    )

    result = analytical_landau_narg_flow(
        chain,
        impurity_stiffness=1.0,
        impurity_quartic=0.2,
    )

    np.testing.assert_allclose(result.curvatures, [2.0 - 0.3**2], atol=1e-14)
    np.testing.assert_allclose(result.quartics, [0.3**4 * 0.2], atol=1e-14)
    np.testing.assert_allclose(result.steps[0].field_scale, 0.3, atol=1e-14)
    np.testing.assert_allclose(result.steps[0].quartic_scale, 0.3**4, atol=1e-14)


def test_landau_critical_exponents_at_upper_critical_s():
    exponents = landau_critical_exponents(0.5)

    assert exponents.y_t == 0.5
    assert exponents.y_u == 0.0
    assert exponents.y_h == 0.75
    assert exponents.nu == 2.0
    assert exponents.beta == 0.5
    assert exponents.gamma == 1.0
    assert exponents.delta == 3.0
    assert exponents.hyperscaling_beta == 0.5
    assert exponents.hyperscaling_gamma == 1.0
    assert exponents.hyperscaling_delta == 3.0


def test_continuum_analytical_narg_beta_and_linearization():
    params = ContinuumAnalyticalNARGParameters(s=0.5, quartic_self=1.0)
    state = ContinuumAnalyticalNARGState(thermal=2.0, quartic=0.2, field=4.0)

    beta = continuum_landau_beta(state, params)
    np.testing.assert_allclose(beta, [1.0, -0.04, 3.0], atol=1e-14)

    linearization = continuum_landau_linearization(params, kind="gaussian")
    np.testing.assert_allclose(
        linearization.jacobian,
        np.diag([0.5, 0.0, 0.75]),
        atol=1e-14,
    )
    np.testing.assert_allclose(np.sort(linearization.eigenvalues), [0.0, 0.5, 0.75])


def test_continuum_analytical_narg_integrates_marginal_quartic():
    params = ContinuumAnalyticalNARGParameters(s=0.5, quartic_self=1.0)
    initial = ContinuumAnalyticalNARGState(thermal=1e-3, quartic=0.2, field=1e-3)

    flow = integrate_continuum_landau_flow(initial, params, lmax=2.0, nsteps=200)

    expected_t = initial.thermal * np.exp(0.5 * 2.0)
    expected_h = initial.field * np.exp(0.75 * 2.0)
    expected_u = initial.quartic / (1.0 + initial.quartic * 2.0)
    np.testing.assert_allclose(flow.thermal[-1], expected_t, rtol=1e-10)
    np.testing.assert_allclose(flow.field[-1], expected_h, rtol=1e-10)
    np.testing.assert_allclose(flow.quartic[-1], expected_u, rtol=1e-10)


def test_ck_local_potential_beta_and_linearization():
    params = CKLocalPotentialParameters(s=0.5, shell_measure=1.0 / (2.0 * np.pi))
    state = CKLocalPotentialState(mass=2.0, quartic=0.2, field=4.0)

    beta = ck_local_potential_beta(state, params)
    shell = params.shell_measure
    expected_beta = np.array(
        [
            0.5 * state.mass + shell * state.quartic / (1.0 + state.mass),
            -3.0 * shell * state.quartic**2 / (1.0 + state.mass) ** 2,
            0.75 * state.field,
        ]
    )
    np.testing.assert_allclose(beta, expected_beta, atol=1e-14)

    linearization = ck_local_potential_linearization(params, kind="gaussian")
    expected_jacobian = np.array(
        [
            [0.5, shell, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.75],
        ]
    )
    np.testing.assert_allclose(linearization.jacobian, expected_jacobian, atol=1e-14)
    np.testing.assert_allclose(np.sort(linearization.eigenvalues), [0.0, 0.5, 0.75])


def test_ck_local_potential_integrates_gaussian_subspace():
    params = CKLocalPotentialParameters(s=0.5)
    initial = CKLocalPotentialState(mass=1e-3, quartic=0.0, field=1e-3)

    flow = integrate_ck_local_potential_flow(initial, params, lmax=2.0, nsteps=200)

    expected_mass = initial.mass * np.exp(0.5 * 2.0)
    expected_field = initial.field * np.exp(0.75 * 2.0)
    np.testing.assert_allclose(flow.mass[-1], expected_mass, rtol=1e-10)
    np.testing.assert_allclose(flow.field[-1], expected_field, rtol=1e-10)
    np.testing.assert_allclose(flow.quartic[-1], 0.0, atol=1e-14)


def test_one_mode_narg_shell_vanishes_for_zero_potential():
    params = OneModeNARGShellParameters(
        s=0.5,
        basis_size=8,
        fit_radius=0.2,
        n_fit_points=9,
    )
    state = CKLocalPotentialState(mass=0.0, quartic=0.0, field=0.0)

    grid = np.linspace(-0.2, 0.2, 5)
    np.testing.assert_allclose(one_mode_narg_shell_energy(grid, state, params), 0.0, atol=1e-14)
    np.testing.assert_allclose(one_mode_narg_beta(state, params), 0.0, atol=1e-14)


def test_one_mode_narg_linearization_has_gaussian_exponents():
    params = OneModeNARGShellParameters(
        s=0.5,
        shell_measure=1.0 / (2.0 * np.pi),
        basis_size=10,
        fit_radius=0.2,
        n_fit_points=11,
    )

    linearization = one_mode_narg_linearization(params, kind="gaussian")

    np.testing.assert_allclose(
        linearization.jacobian,
        np.array(
            [
                [0.5, params.shell_measure, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.75],
            ]
        ),
        atol=5e-5,
    )
    np.testing.assert_allclose(np.sort(linearization.eigenvalues), [0.0, 0.5, 0.75], atol=5e-5)


def test_polynomial_one_mode_narg_reduces_to_quartic_projection():
    params = OneModeNARGShellParameters(
        s=0.5,
        shell_measure=1.0 / (2.0 * np.pi),
        basis_size=10,
        fit_radius=0.2,
        n_fit_points=13,
    )
    quartic_state = CKLocalPotentialState(mass=1e-3, quartic=0.2, field=0.0)
    polynomial_state = PolynomialPESState.from_ck(quartic_state, order=4)
    grid = np.linspace(-0.2, 0.2, 7)

    np.testing.assert_allclose(
        polynomial_one_mode_narg_shell_energy(grid, polynomial_state, params),
        one_mode_narg_shell_energy(grid, quartic_state, params),
        atol=1e-14,
    )

    quartic_beta = one_mode_narg_beta(quartic_state, params)
    polynomial_beta = polynomial_one_mode_narg_beta(polynomial_state, params)
    np.testing.assert_allclose(polynomial_beta[1], quartic_beta[0], atol=1e-10)
    np.testing.assert_allclose(polynomial_beta[3], quartic_beta[1], atol=1e-8)
    np.testing.assert_allclose(polynomial_beta[[0, 2]], 0.0, atol=1e-10)


def test_conditional_one_mode_narg_projection_keeps_matrix_gaps():
    params = OneModeNARGShellParameters(
        s=0.5,
        shell_measure=1.0 / (2.0 * np.pi),
        basis_size=8,
        fit_radius=0.2,
        n_fit_points=9,
    )
    zero_state = PolynomialPESState(np.zeros(4))

    projection = conditional_one_mode_narg_shell_projection(
        zero_state,
        params,
        n_conditional_states=3,
        order=4,
    )

    assert projection.shell_matrices.shape == (9, 3, 3)
    np.testing.assert_allclose(
        projection.shell_matrices,
        np.swapaxes(projection.shell_matrices, 1, 2),
        atol=1e-14,
    )
    center = len(projection.grid) // 2
    expected_gaps = params.shell_measure * np.diag([0.0, 1.0, 2.0])
    np.testing.assert_allclose(projection.shell_matrices[center], expected_gaps, atol=1e-14)
    np.testing.assert_allclose(projection.shell_couplings[0], expected_gaps, atol=1e-14)
    np.testing.assert_allclose(projection.shell_couplings[1:], 0.0, atol=1e-11)


def test_matrix_one_mode_narg_normalized_gaussian_fixed_point():
    params = OneModeNARGShellParameters(
        s=0.5,
        shell_measure=1.0 / (2.0 * np.pi),
        basis_size=8,
        fit_radius=0.2,
        n_fit_points=9,
    )
    state = MatrixPolynomialPESState.gaussian(order=2, dimension=2, gap=1.0)

    beta = matrix_one_mode_narg_beta(state, params)
    np.testing.assert_allclose(beta, 0.0, atol=1e-12)

    linearization = matrix_one_mode_narg_linearization(state, params, step=1e-5)
    assert linearization.jacobian.shape == (9, 9)
    assert len(linearization.packed_labels) == 9
    assert np.all(np.isfinite(linearization.eigenvalues))


def test_matrix_one_mode_narg_flow_keeps_gaussian_fixed_point_stationary():
    params = OneModeNARGShellParameters(
        s=0.5,
        shell_measure=1.0 / (2.0 * np.pi),
        basis_size=8,
        fit_radius=0.2,
        n_fit_points=9,
    )
    state = MatrixPolynomialPESState.gaussian(order=2, dimension=2, gap=1.0)

    flow = integrate_matrix_one_mode_narg_flow(state, params, lmax=0.4, nsteps=4)

    np.testing.assert_allclose(flow.states[-1], state.coefficients, atol=1e-12)
    np.testing.assert_allclose(flow.ground_energies, 0.0, atol=1e-12)
    np.testing.assert_allclose(flow.first_gaps, 1.0, atol=1e-12)
    assert isinstance(flow.state(), MatrixPolynomialPESState)


def test_matrix_one_mode_narg_flow_preserves_normalized_multistate_gauge():
    params = OneModeNARGShellParameters(
        s=0.5,
        shell_measure=1.0 / (2.0 * np.pi),
        basis_size=8,
        fit_radius=0.2,
        n_fit_points=9,
    )
    coefficients = MatrixPolynomialPESState.gaussian(order=3, dimension=2, gap=1.0).asarray()
    coefficients[1] = np.array([[0.0, 0.02], [0.02, 0.01]])
    coefficients[2] = np.array([[0.1, 0.0], [0.0, -0.03]])
    coefficients[3] = np.array([[0.0, 0.01], [0.01, 0.0]])

    flow = integrate_matrix_one_mode_narg_flow(
        MatrixPolynomialPESState(coefficients),
        params,
        lmax=0.2,
        nsteps=4,
    )

    np.testing.assert_allclose(flow.states, np.swapaxes(flow.states, 2, 3), atol=1e-12)
    np.testing.assert_allclose(flow.ground_energies, 0.0, atol=1e-10)
    np.testing.assert_allclose(flow.first_gaps, 1.0, atol=1e-10)
    assert np.all(np.isfinite(flow.coefficients))


def test_discrete_matrix_wilson_narg_flow_keeps_retained_matrix_pes():
    chain = SpinBosonWilsonChain(
        onsite=np.array([1.2, 0.9]),
        hopping=np.array([0.15]),
        impurity_coupling=0.2,
    )
    initial = MatrixPolynomialPESState.gaussian(order=4, dimension=1)
    initial.coefficients[2, 0, 0] = 1.0
    initial.coefficients[4, 0, 0] = 0.1

    flow = discrete_matrix_wilson_narg_flow(
        chain,
        initial,
        n_conditional_states=2,
        polynomial_order=4,
        coordinate_basis_size=6,
        fit_radius=0.4,
        n_fit_points=9,
        nrg_rescale=False,
    )

    assert len(flow.steps) == 2
    assert flow.steps[-1].coefficients.shape == (5, 2, 2)
    assert np.all(np.isfinite(flow.coefficients))
    assert np.all(np.isfinite(flow.residuals))
    np.testing.assert_allclose(flow.steps[-1].coefficients[0, 0, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(flow.steps[-1].coefficients[0, 1, 1], 1.0, atol=1e-12)


def test_overlap_dressed_phi_kinetic_reduces_to_product_for_constant_basis():
    kinetic = np.array(
        [
            [1.0, -0.5, 0.0],
            [-0.5, 1.0, -0.5],
            [0.0, -0.5, 1.0],
        ],
        dtype=float,
    )
    basis = np.tile(np.eye(2), (3, 1, 1))

    dressed, overlaps = overlap_dressed_phi_kinetic(kinetic, basis)

    np.testing.assert_allclose(overlaps, np.einsum("ij,ab->iajb", np.ones((3, 3)), np.eye(2)), atol=1e-14)
    np.testing.assert_allclose(dressed, np.kron(kinetic, np.eye(2)), atol=1e-14)


def test_harmonic_displacement_overlap_has_closed_form_limits():
    overlap0 = harmonic_oscillator_displacement_overlap(0.0, 4)
    overlap = harmonic_oscillator_displacement_overlap(0.3, 4)

    np.testing.assert_allclose(overlap0, np.eye(4), atol=1e-14)
    np.testing.assert_allclose(overlap[0, 0], np.exp(-0.5 * 0.3**2), atol=1e-14)
    np.testing.assert_allclose(overlap[0, 1], -0.3 * np.exp(-0.5 * 0.3**2), atol=1e-14)
    np.testing.assert_allclose(overlap[1, 0], 0.3 * np.exp(-0.5 * 0.3**2), atol=1e-14)


def test_overlap_dressed_phi_kinetic_accepts_analytic_overlap_callable():
    grid = np.array([-0.2, 0.0, 0.2])
    kinetic = np.array(
        [
            [1.0, -0.5, 0.0],
            [-0.5, 1.0, -0.5],
            [0.0, -0.5, 1.0],
        ],
        dtype=float,
    )
    slope = 0.7

    def overlap(phi_i, phi_j):
        return harmonic_oscillator_displacement_overlap(slope * (phi_j - phi_i), 2)

    dressed, dressing = overlap_dressed_phi_kinetic(
        kinetic,
        phi_grid=grid,
        overlap=overlap,
        nstates=2,
    )

    expected = np.empty((3, 2, 3, 2), dtype=float)
    for i, phi_i in enumerate(grid):
        for j, phi_j in enumerate(grid):
            expected[i, :, j, :] = overlap(phi_i, phi_j)
    np.testing.assert_allclose(dressing, expected, atol=1e-14)
    np.testing.assert_allclose(dressed, dressed.T, atol=1e-14)


def test_matrix_one_mode_narg_overlap_dressed_kinetic_matches_full_primitive_basis():
    params = OneModeNARGShellParameters(
        s=0.5,
        shell_measure=1.0 / (2.0 * np.pi),
        basis_size=4,
        fit_radius=0.2,
        n_fit_points=9,
    )
    coefficients = MatrixPolynomialPESState.gaussian(order=3, dimension=2, gap=0.7).asarray()
    coefficients[1] = np.array([[0.0, 0.03], [0.03, -0.02]])
    coefficients[2] = np.array([[0.15, 0.01], [0.01, -0.04]])
    state = MatrixPolynomialPESState(coefficients)
    grid = np.linspace(-0.3, 0.3, 5)
    dx = grid[1] - grid[0]
    kinetic = np.diag(np.full(grid.size, 1.0 / (dx * dx)))
    kinetic += np.diag(np.full(grid.size - 1, -0.5 / (dx * dx)), k=1)
    kinetic += np.diag(np.full(grid.size - 1, -0.5 / (dx * dx)), k=-1)

    full_dim = params.basis_size * state.dimension
    dressed = matrix_one_mode_narg_effective_hamiltonian(
        state,
        grid,
        kinetic,
        params,
        n_conditional_states=full_dim,
    )
    primitive = np.kron(kinetic, np.eye(full_dim))
    for index, point in enumerate(grid):
        rows = slice(index * full_dim, (index + 1) * full_dim)
        primitive[rows, rows] += _matrix_one_mode_hamiltonian(point, state, params)

    np.testing.assert_allclose(dressed.hamiltonian, dressed.hamiltonian.T, atol=1e-12)
    np.testing.assert_allclose(
        dressed.kinetic_dressing[np.arange(grid.size), :, np.arange(grid.size), :],
        np.tile(np.eye(full_dim), (grid.size, 1, 1)),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        dressed.effective_energies,
        np.linalg.eigvalsh(primitive),
        atol=1e-10,
    )


def test_analytical_harmonic_narg_rescaled_curvatures_match_physical_flow():
    chain = log_discretized_spin_boson_wilson_chain(
        5,
        alpha=0.03,
        Lambda=2.0,
        s=0.5,
        delta=0.1,
    )

    physical = analytical_harmonic_narg_flow(chain, impurity_stiffness=1.0)
    rescaled = analytical_harmonic_narg_flow(
        chain,
        impurity_stiffness=1.0,
        nrg_rescale=True,
        Lambda=2.0,
    )

    scales = 2.0 ** np.arange(chain.nmodes)
    np.testing.assert_allclose(rescaled.curvatures, scales * physical.curvatures, atol=1e-12)
    assert rescaled.stable


def test_analytical_harmonic_narg_exponent_estimate_is_well_formed():
    Lambda = 1.5
    chain = log_discretized_spin_boson_wilson_chain(
        28,
        alpha=0.08,
        Lambda=Lambda,
        s=0.5,
        delta=0.1,
    )
    critical = critical_harmonic_impurity_stiffness(chain)

    estimate = estimate_harmonic_thermal_exponent(
        chain,
        critical * (1.0 + 1e-10),
        relative_step=1e-2,
        Lambda=Lambda,
        tail=6,
    )

    assert len(estimate.window) == 6
    assert np.isfinite(estimate.mean_y)
    assert 0.2 < estimate.mean_y < 0.8


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
    assert result.steps[-1].sigma_z is not None


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


def test_spin_boson_adiabatic_narg_full_conditional_basis_matches_wilson_update():
    chain = log_discretized_spin_boson_wilson_chain(
        2,
        alpha=0.04,
        Lambda=2.0,
        s=0.7,
        omegac=1.0,
        epsilon=0.02,
        delta=0.1,
    )

    wilson = SpinBosonWilsonNARG(
        chain,
        nboson=4,
        bond_dim=128,
        basis="sine-dvr",
        dvr_qmax=5.0,
    ).run(nroots=4)
    narg = SpinBosonWilsonAdiabaticNARG(
        chain,
        nboson=4,
        bond_dim=128,
        n_conditional_states=None,
        dvr_qmax=5.0,
    ).run(nroots=4)
    narg_literal = SpinBosonWilsonAdiabaticNARG(
        chain,
        nboson=4,
        bond_dim=128,
        n_conditional_states=None,
        dvr_qmax=5.0,
        full_conditional_shortcut=False,
    ).run(nroots=4)

    np.testing.assert_allclose(narg.energies, wilson.energies, atol=1e-10)
    np.testing.assert_allclose(narg.magnetizations, wilson.magnetizations[:4], atol=1e-10)
    np.testing.assert_allclose(narg_literal.energies, wilson.energies, atol=1e-10)
    np.testing.assert_allclose(narg_literal.magnetizations, wilson.magnetizations[:4], atol=1e-10)
    assert narg.steps[-1].product_dim == wilson.steps[-1].product_dim
    assert narg.steps[-1].conditional_dim == wilson.steps[-2].kept


def test_spin_boson_adiabatic_narg_reduced_conditional_basis_runs():
    chain = log_discretized_spin_boson_wilson_chain(
        3,
        alpha=0.04,
        Lambda=2.0,
        s=0.7,
        epsilon=0.0,
        delta=0.1,
    )

    result = SpinBosonWilsonAdiabaticNARG(
        chain,
        nboson=5,
        bond_dim=12,
        n_conditional_states=3,
        dvr_qmax=6.0,
        nrg_rescale=True,
        nrg_Lambda=2.0,
    ).run(nroots=4)

    assert result.nrg_rescaled
    assert result.steps[0].conditional_dim == 2
    assert result.steps[-1].conditional_dim == 3
    assert result.steps[-1].product_dim == 15
    assert result.steps[-1].kept == 12
    assert result.steps[-1].boundary_annihilation.shape == (12, 12)
    assert result.steps[-1].sigma_z.shape == (12, 12)
    assert np.all(np.isfinite(result.energies))


def test_spin_boson_narg_step_observables_match_prefix_runs():
    chain = log_discretized_spin_boson_wilson_chain(
        3,
        alpha=0.03,
        Lambda=2.0,
        s=0.7,
        epsilon=0.01,
        delta=0.1,
    )
    narg = SpinBosonWilsonAdiabaticNARG(
        chain,
        nboson=4,
        bond_dim=16,
        n_conditional_states=None,
        dvr_qmax=5.0,
    ).run(nroots=2)

    nvalues = np.array([1, 2, 3], dtype=int)
    observables = spin_boson_narg_step_observables(narg, nvalues=nvalues, nlevels=2)

    for row, nmodes in enumerate(nvalues):
        prefix = SpinBosonWilsonChain(
            onsite=chain.onsite[: int(nmodes)],
            hopping=chain.hopping[: max(0, int(nmodes) - 1)],
            impurity_coupling=chain.impurity_coupling,
            epsilon=chain.epsilon,
            delta=chain.delta,
        )
        direct = SpinBosonWilsonAdiabaticNARG(
            prefix,
            nboson=4,
            bond_dim=16,
            n_conditional_states=None,
            dvr_qmax=5.0,
        ).run(nroots=2)

        np.testing.assert_allclose(observables.energies[row], direct.energies[:2], atol=1e-11)
        np.testing.assert_allclose(observables.gaps[row, 0], direct.energies[1] - direct.energies[0], atol=1e-11)
        np.testing.assert_allclose(
            observables.magnetizations[row],
            np.asarray(direct.magnetizations[:2], dtype=float),
            atol=1e-11,
        )


def test_spin_boson_narg_can_recycle_product_eigenvectors():
    chain = log_discretized_spin_boson_wilson_chain(
        3,
        alpha=0.03,
        Lambda=2.0,
        s=0.7,
        epsilon=0.01,
        delta=0.1,
    )
    reference = SpinBosonWilsonAdiabaticNARG(
        chain,
        nboson=6,
        bond_dim=8,
        n_conditional_states=None,
        dvr_qmax=5.0,
        diagonalization_method="dense",
        store_step_vectors=True,
    ).run(nroots=2)
    recycled_vectors = [step.product_vectors for step in reference.steps]

    recycled = SpinBosonWilsonAdiabaticNARG(
        chain,
        nboson=6,
        bond_dim=8,
        n_conditional_states=None,
        dvr_qmax=5.0,
        diagonalization_method="lobpcg",
        diagonalization_tol=1.0e-10,
        diagonalization_maxiter=30,
        initial_product_vectors=recycled_vectors,
    ).run(nroots=2)

    assert all(vector is not None for vector in recycled_vectors)
    np.testing.assert_allclose(recycled.energies, reference.energies, atol=1.0e-8)
    np.testing.assert_allclose(recycled.magnetizations, reference.magnetizations, atol=1.0e-8)


def test_spin_boson_synthetic_collapse_recovers_yh_scan():
    alpha_c = 0.011
    inv_nu = 0.30
    beta_over_nu = 0.15
    y_h = 0.45
    Lambda = 1.5
    step = 2
    nvalues = np.array([2, 4, 6, 8])
    alpha_ratio = Lambda ** (-step * inv_nu)
    epsilon_ratio = Lambda ** (-step * y_h)
    alpha_offsets = 0.01 * alpha_ratio ** np.arange(5)
    alphas = np.sort(np.concatenate([alpha_c - alpha_offsets, [alpha_c], alpha_c + alpha_offsets]))
    epsilons = 1e-3 * epsilon_ratio ** np.arange(6)

    aligned = np.empty((len(epsilons), len(alphas), len(nvalues)), dtype=float)
    for eindex, epsilon in enumerate(epsilons):
        for aindex, alpha in enumerate(alphas):
            for nindex, nmode in enumerate(nvalues):
                scale = Lambda ** float(nmode)
                x_alpha = (alpha - alpha_c) * scale**inv_nu
                x_field = epsilon * scale**y_h
                collapsed = np.exp(1.0 * x_alpha + 2.0 * x_alpha**2 + 0.4 * np.sin(np.log(x_field)))
                aligned[eindex, aindex, nindex] = collapsed * scale ** (-beta_over_nu)

    fit = grid_collapse_over_y_h(
        alphas,
        epsilons,
        nvalues,
        aligned,
        alpha_c_grid=np.array([0.0107, 0.0110, 0.0113]),
        inv_nu_grid=np.array([0.20, 0.30, 0.40]),
        beta_over_nu_grid=np.array([0.10, 0.15, 0.20]),
        y_h_grid=np.array([0.35, 0.45, 0.55]),
        Lambda=Lambda,
        min_m=1e-8,
        max_m=10.0,
        width=0.05,
    )

    assert np.isclose(fit.alpha_c, alpha_c)
    assert np.isclose(fit.inv_nu, inv_nu)
    assert np.isclose(fit.beta_over_nu, beta_over_nu)
    assert np.isclose(fit.y_h, y_h)


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


def test_spin_boson_wilson_narg_can_use_explicit_nrg_rescaling():
    chain = log_discretized_spin_boson_wilson_chain(
        4,
        alpha=0.04,
        Lambda=2.0,
        s=0.5,
        epsilon=0.0,
        delta=0.1,
    )
    result = SpinBosonWilsonNARG(
        chain,
        nboson=3,
        bond_dim=20,
        basis="dvr",
        nrg_rescale=True,
        nrg_Lambda=2.0,
    ).run(nroots=4)

    assert result.nrg_rescaled
    np.testing.assert_allclose(result.steps[-1].energies[0], 0.0, atol=1e-12)
    assert result.steps[-1].rescale_factor == 2.0 ** result.steps[-1].site

    flow = narg_rescaled_spectrum_flow(result, Lambda=2.0, nlevels=4)
    final_gaps = result.steps[-1].energies[1:4] - result.steps[-1].energies[0]
    np.testing.assert_allclose(flow[-1], final_gaps, atol=1e-12)


def test_spin_boson_wilson_narg_can_scale_by_current_onsite_energy():
    chain = log_discretized_spin_boson_wilson_chain(
        4,
        alpha=0.04,
        Lambda=2.0,
        s=0.5,
        epsilon=0.0,
        delta=0.1,
    )
    result = SpinBosonWilsonNARG(
        chain,
        nboson=3,
        bond_dim=20,
        basis="sine-dvr",
        dvr_qmax=6.0,
        nrg_rescale=True,
        nrg_Lambda=2.0,
        nrg_scale="onsite",
    ).run(nroots=4)

    assert result.nrg_rescaled
    assert result.nrg_scale == "onsite"
    np.testing.assert_allclose(
        [step.rescale_factor for step in result.steps],
        1.0 / chain.onsite,
        rtol=1e-14,
    )
    flow = narg_rescaled_spectrum_flow(result, Lambda=2.0, nlevels=4)
    final_gaps = result.steps[-1].energies[1:4] - result.steps[-1].energies[0]
    np.testing.assert_allclose(flow[-1], final_gaps, atol=1e-12)


def test_spin_boson_adiabatic_narg_can_scale_by_current_onsite_energy():
    chain = log_discretized_spin_boson_wilson_chain(
        3,
        alpha=0.04,
        Lambda=2.0,
        s=0.7,
        epsilon=0.0,
        delta=0.1,
    )
    result = SpinBosonWilsonAdiabaticNARG(
        chain,
        nboson=5,
        bond_dim=12,
        n_conditional_states=3,
        dvr_qmax=6.0,
        nrg_rescale=True,
        nrg_Lambda=2.0,
        nrg_scale="onsite",
    ).run(nroots=4)

    assert result.nrg_rescaled
    assert result.nrg_scale == "onsite"
    np.testing.assert_allclose(
        [step.rescale_factor for step in result.steps],
        1.0 / chain.onsite,
        rtol=1e-14,
    )
    assert result.steps[-1].conditional_dim == 3
    assert np.all(np.isfinite(result.energies))


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


def test_spin_boson_wilson_mpo_matches_dense_hamiltonian():
    from pyqed.mps.mps import _mpo_to_dense_operator
    from pyqed.tn import MPO

    chain = log_discretized_spin_boson_wilson_chain(
        2,
        alpha=0.03,
        Lambda=2.0,
        s=0.5,
        epsilon=0.01,
        delta=0.1,
    )
    mpo = MPO(spin_boson_wilson_mpo(chain, nboson=3, basis="fock"))
    dense = spin_boson_wilson_hamiltonian(chain, nboson=3, basis="fock")

    np.testing.assert_allclose(_mpo_to_dense_operator(mpo), dense, atol=1e-12)


def test_spin_boson_wilson_dmrg_matches_tiny_exact_ground_energy():
    chain = log_discretized_spin_boson_wilson_chain(
        1,
        alpha=0.02,
        Lambda=2.0,
        s=0.5,
        epsilon=0.01,
        delta=0.1,
    )
    exact, exact_vectors = spin_boson_wilson_exact(
        chain,
        nboson=3,
        nroots=1,
        basis="fock",
    )
    result = spin_boson_wilson_dmrg(
        chain,
        nboson=3,
        bond_dim=8,
        nsweeps=4,
        basis="fock",
        noise=0.0,
        sweep_tol=1e-10,
        davidson_tol=1e-10,
    )

    np.testing.assert_allclose(result.energies[0], exact[0], atol=1e-8)
    assert -1.0 <= result.magnetization <= 1.0
    assert exact_vectors.shape[0] == 2 * 3


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
