import numpy as np

from examples.ldr.high_low_polaron_disentangler import run_scan


def test_high_low_polaron_disentangler_recovers_conditional_shift():
    result = run_scan(
        omega_high=4.0,
        omega_low=1.0,
        coupling=2.0,
        qmax=6.0,
        npoints=22,
        nstates=3,
        nroots=4,
        eta_points=61,
    )

    assert abs(result.best_eta - result.expected_eta) < 0.04
    assert result.best_weighted_fidelity > result.bare_weighted_fidelity + 0.01
    assert result.best_weighted_fidelity > 0.99
    assert result.best_energy_rms < result.bare_energy_rms
    np.testing.assert_allclose(result.exact_dvr_energies, result.analytic_energies, atol=1.5e-3)


def test_quartic_coupled_energy_optimizer_refines_scan():
    result = run_scan(
        omega_high=4.0,
        omega_low=1.0,
        coupling=1.2,
        x4=0.05,
        y4=0.02,
        x2y2=0.05,
        qmax=5.5,
        npoints=16,
        nstates=2,
        nroots=3,
        eta_points=21,
        optimize_energy=True,
        optimizer_xatol=5.0e-4,
    )

    assert np.isfinite(result.optimized_energy_eta)
    assert result.optimized_energy_rms <= result.minimum_energy_rms + 1.0e-10
    assert result.optimized_energy_energies.shape == (3,)
