import numpy as np

from examples.lgt.schwinger_dvr_benchmark import (
    dispersion_benchmark,
    fourier_dirac_hamiltonian,
    schwinger_gap_benchmark,
    staggered_dirac_hamiltonian,
)


def test_dirac_regulator_hamiltonians_are_hermitian():
    mass = lambda x: 0.8 + 0.2 * np.cos(x)
    dvr = fourier_dirac_hamiltonian(11, 2.0 * np.pi, mass)
    staggered = staggered_dirac_hamiltonian(22, 2.0 * np.pi, mass)

    np.testing.assert_allclose(dvr, dvr.conj().T, atol=1.0e-13)
    np.testing.assert_allclose(staggered, staggered.conj().T, atol=1.0e-13)


def test_fourier_dvr_recovers_free_dirac_dispersion():
    result = dispersion_benchmark(npts=15, length=10.0, mass=0.7)

    assert np.max(result["dvr_error"]) < 1.0e-13
    assert result["staggered_error"][-1] > 0.2


def test_bosonized_schwinger_gap_is_spectral_in_sine_dvr():
    result = schwinger_gap_benchmark(
        npts_values=(7, 13, 25), timing_repeats=1
    )

    assert np.max(result["dvr_vector_error"]) < 1.0e-12
    assert np.max(result["dvr_scalar_error"]) < 2.0e-12
    assert np.all(np.diff(result["fd_vector_error"]) < 0.0)
