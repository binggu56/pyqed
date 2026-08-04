import numpy as np
from scipy.integrate import quad

from examples.mps.cletta_dipolar_bose_gas_continuum import (
    dipolar_kernel,
    pair_correlation,
    product_state,
)


def test_shifted_core_dipolar_kernel_has_expected_tail_and_integral():
    strength = 3.0
    softening = 0.4
    distances = np.array([10.0, 20.0, 40.0])
    values = dipolar_kernel(distances, strength, softening)

    np.testing.assert_allclose(
        values * distances**3,
        strength,
        rtol=0.12,
    )
    integral = quad(
        lambda distance: dipolar_kernel(distance, strength, softening),
        0.0,
        np.inf,
    )[0]
    np.testing.assert_allclose(
        integral,
        strength / (2.0 * softening**2),
        rtol=1.0e-11,
    )


def test_product_state_pair_correlation_is_normalized():
    state = product_state(1.3)
    state.scale = 1.0
    values = pair_correlation(state, [0.0, 0.2, 1.0, 5.0])

    np.testing.assert_allclose(values, 1.0, atol=1.0e-11)
