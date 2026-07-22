from types import SimpleNamespace

import numpy as np

from examples.mps.cletta_frustrated_bose_gas_continuum import (
    connected_density_correlation,
    preferred_wavevector,
    product_state,
    quadratic_spectrum,
    run,
)


def test_frustrated_exponential_kernel_has_finite_wavevector_minimum():
    rates = np.array([0.45, 2.2])
    strengths = np.array([0.8, -1.6])
    wavevector, value = preferred_wavevector(rates, strengths, density=1.0, momentum_max=3.0)

    assert 0.7 < wavevector < 0.8
    assert value < quadratic_spectrum(0.0, rates, strengths, density=1.0)


def test_noncanonical_connected_correlation_gives_zero_for_product_state():
    state = product_state(1.0)
    state.scale = 1.0
    values = connected_density_correlation(state, [0.0, 1.0, 2.0])
    assert np.allclose(values, 0.0, atol=1.0e-10)


def test_frustrated_cletta_cutoff_smoke():
    args = SimpleNamespace(
        density=1.0,
        decay_rates=[0.45, 2.2],
        strengths=[0.8, -1.6],
        contact_coupling=0.5,
        momentum_max=3.0,
        memory_decay_rate=None,
        cmps_bond_dims=[],
        cmps_restarts=1,
        cmps_maxiter=2,
        bond_dim=1,
        cutoffs=[1, 2],
        cmps_noise_cutoff=3,
        cmps_noise_seeds=2,
        cmps_seed_noise=0.01,
        skip_zero_cmps_seed=False,
        optimize_memory_poles=False,
        restarts=1,
        maxiter=4,
        seed=5,
        regularization=1.0e-10,
        density_gauge_penalty=1.0e-4,
        tie_scale=0.02,
        no_jax=True,
        contraction_backend="auto",
        iterative_tolerance=1.0e-8,
        iterative_maxiter=None,
        frequency_bound=8.0,
        output="",
        figure="",
        correlation_range=4.0,
        correlation_points=20,
    )
    rows = run(args)
    cletta = [row for row in rows if row["cutoff_L"] != ""]

    assert [row["memory_dim"] for row in cletta] == [3, 6]
    assert all(row["parameter_count"] == 2 for row in cletta)
    assert all(np.isfinite(row["energy"]) for row in rows)
