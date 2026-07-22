import numpy as np
import scipy.linalg as la

from examples.mps.bose_hubbard_disentangler_benchmark import (
    dense_bose_hubbard,
    fixed_number_indices,
    run_block_site_benchmark,
    run_stepwise_narg_benchmark,
    stepwise_growth_pretruncation_basis,
)


def test_bose_hubbard_boundary_disentangler_improves_block_site_ritz_values():
    result = run_block_site_benchmark(
        nbosons=3,
        nmax=3,
        t=1.0,
        U=1.0,
        Dblock=6,
        nroots=3,
        theta_points=31,
        optimize=True,
        optimizer_xatol=1.0e-5,
    )

    assert result.optimized_rms < 0.25 * result.bare_rms
    assert result.optimized_fidelity > result.bare_fidelity
    assert abs(result.optimized_theta) > 0.05
    assert result.optimized_energies.shape == (3,)


def test_stepwise_bose_hubbard_narg_disentangler_improves_each_growth_layer():
    result = run_stepwise_narg_benchmark(
        nsites=4,
        nbosons=4,
        nmax=2,
        t=1.0,
        U=1.0,
        keep=2,
        nroots=2,
        theta_points=11,
        optimize=False,
    )

    assert result.best_scan_rms < 0.7 * result.bare_rms
    assert result.best_scan_fidelity > result.bare_fidelity + 0.1
    assert result.bare_dim == result.best_scan_dim


def test_stepwise_disentangler_is_applied_before_conditional_truncation():
    bare = stepwise_growth_pretruncation_basis(
        nsites=4,
        nbosons=4,
        nmax=2,
        t=1.0,
        U=1.0,
        mu=0.0,
        keep=2,
        theta=0.0,
    )
    disentangled = stepwise_growth_pretruncation_basis(
        nsites=4,
        nbosons=4,
        nmax=2,
        t=1.0,
        U=1.0,
        mu=0.0,
        keep=2,
        theta=0.45,
    )

    assert len(disentangled.steps) == 3
    assert disentangled.basis.shape[0] == 3**4
    assert np.all(disentangled.qn == 4)
    assert all(step.kept <= 2 * 3 for step in disentangled.steps)

    bare_projector = bare.basis @ bare.basis.conj().T
    disentangled_projector = disentangled.basis @ disentangled.basis.conj().T
    assert np.linalg.norm(disentangled_projector - bare_projector) > 1.0e-3


def test_recursive_renormalized_disentangler_is_exact_without_truncation():
    nsites = 3
    nbosons = 2
    nmax = 2
    state = stepwise_growth_pretruncation_basis(
        nsites=nsites,
        nbosons=nbosons,
        nmax=nmax,
        t=1.0,
        U=1.0,
        mu=0.0,
        keep=32,
        theta=0.37,
    )
    hamiltonian, _ = dense_bose_hubbard(nsites, nmax, t=1.0, U=1.0)
    fixed_indices, _ = fixed_number_indices(nsites, nbosons, nmax)
    exact = np.linalg.eigvalsh(hamiltonian[np.ix_(fixed_indices, fixed_indices)])
    target_columns = np.flatnonzero(state.qn == nbosons)
    projected = state.hamiltonian[np.ix_(target_columns, target_columns)]
    values = np.linalg.eigvalsh(0.5 * (projected + projected.conj().T))

    assert values.shape == exact.shape
    np.testing.assert_allclose(values, exact, atol=1.0e-10)


def test_stepwise_per_step_theta_optimizer_runs_and_records_vector():
    result = run_stepwise_narg_benchmark(
        nsites=3,
        nbosons=2,
        nmax=2,
        t=1.0,
        U=1.0,
        keep=2,
        nroots=2,
        theta_points=5,
        theta_mode="per-step",
        optimize=True,
        optimizer_xatol=1.0e-3,
        optimizer_maxiter=4,
    )

    assert result.theta_mode == "per-step"
    assert result.optimized_thetas.shape == (2,)
    assert result.optimized_rms <= result.best_scan_rms + 1.0e-10


def test_bose_hubbard_beam_splitter_preserves_total_number():
    nmax = 3
    local_dim = nmax + 1
    _, b_ops = dense_bose_hubbard(3, nmax, t=1.0, U=1.0)
    generator = b_ops[1].conj().T @ b_ops[2] - b_ops[1] @ b_ops[2].conj().T
    generator = 0.5 * (generator - generator.conj().T)
    unitary = la.expm(0.37 * generator)
    nloc = np.diag(np.arange(local_dim, dtype=float))
    total_number = sum(
        np.asarray(
            np.kron(
                np.kron(
                    nloc if site == 0 else np.eye(local_dim),
                    nloc if site == 1 else np.eye(local_dim),
                ),
                nloc if site == 2 else np.eye(local_dim),
            ),
            dtype=complex,
        )
        for site in range(3)
    )

    np.testing.assert_allclose(unitary.conj().T @ unitary, np.eye(local_dim**3), atol=1.0e-12)
    np.testing.assert_allclose(
        unitary.conj().T @ total_number @ unitary,
        total_number,
        atol=1.0e-12,
    )
