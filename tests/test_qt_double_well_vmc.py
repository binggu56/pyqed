import numpy as np

from pyqed.qt import (
    ThreeParticleDoubleWellVMC,
    double_well_jastrow_terms,
    optimize_symmetric_double_well_jastrow,
)


PARAMETERS = np.array((-1.8, np.log(0.6), 0.0, np.log(0.8), -0.3))


def test_three_body_jastrow_spatial_derivatives_match_finite_differences():
    coordinates = np.array(((0.31, -0.72, 1.14),))
    _, gradient, laplacian = double_well_jastrow_terms(coordinates, PARAMETERS)
    step = 2.0e-5
    numerical_gradient = np.empty(3)
    numerical_laplacian = 0.0
    center = double_well_jastrow_terms(coordinates, PARAMETERS)[0][0]
    for particle in range(3):
        plus, minus = coordinates.copy(), coordinates.copy()
        plus[0, particle] += step
        minus[0, particle] -= step
        amplitude_plus = double_well_jastrow_terms(plus, PARAMETERS)[0][0]
        amplitude_minus = double_well_jastrow_terms(minus, PARAMETERS)[0][0]
        numerical_gradient[particle] = (amplitude_plus - amplitude_minus) / (
            2.0 * step
        )
        numerical_laplacian += (
            amplitude_plus - 2.0 * center + amplitude_minus
        ) / step**2
    np.testing.assert_allclose(gradient[0], numerical_gradient, rtol=2e-8, atol=2e-9)
    np.testing.assert_allclose(laplacian[0], numerical_laplacian, rtol=2e-6, atol=2e-6)


def test_reflection_move_connects_symmetric_occupation_sectors():
    local = ThreeParticleDoubleWellVMC(PARAMETERS, nwalkers=4, seed=3).run(
        burn_in=0,
        sweeps=10,
        proposal_scale=0.0,
        reflection_probability=0.0,
        initialize_sector=1,
    )
    reflected = ThreeParticleDoubleWellVMC(PARAMETERS, nwalkers=4, seed=3).run(
        burn_in=0,
        sweeps=10,
        proposal_scale=0.0,
        reflection_probability=1.0,
        initialize_sector=1,
    )
    np.testing.assert_allclose(local.occupation_probability, (0.0, 1.0, 0.0, 0.0))
    np.testing.assert_allclose(
        reflected.occupation_probability, (0.0, 0.5, 0.5, 0.0)
    )
    assert reflected.reflection_acceptance == 1.0


def test_three_body_control_cannot_raise_optimized_grid_energy():
    start = ((-1.5, np.log(0.55), np.log(0.8), 0.0),)
    pair = optimize_symmetric_double_well_jastrow(
        ngrid=19, include_three_body=False, starts=start
    )
    three = optimize_symmetric_double_well_jastrow(
        ngrid=19, include_three_body=True, starts=start
    )
    assert three.fun <= pair.fun + 1.0e-8
