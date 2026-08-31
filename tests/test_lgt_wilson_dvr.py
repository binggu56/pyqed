import numpy as np

from pyqed.dvr import ExponentialDVR
from pyqed.lgt import WilsonFourierDVR


def _model(npts=15, length=9.0):
    x = -0.5 * length + length / npts * (np.arange(npts) + 0.5)
    phases = length / npts * (
        0.17 + 0.23 * np.cos(2.0 * np.pi * x / length)
    )
    return WilsonFourierDVR(phases, length)


def test_prefix_factorization_reconstructs_links_and_holonomy():
    model = _model()

    np.testing.assert_allclose(
        [model.wilson_line(site, site + 1) for site in range(model.npts)],
        model.link_variables,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        np.prod(model.link_variables),
        np.exp(-1j * model.holonomy_phase),
        atol=1.0e-14,
    )


def test_fft_derivative_matches_dense_wilson_matrix_and_is_antihermitian():
    model = _model()
    rng = np.random.default_rng(7)
    state = rng.normal(size=(model.npts, 3)) + 1j * rng.normal(
        size=(model.npts, 3)
    )
    dense = model.dense_derivative()
    bare = ExponentialDVR(npts=model.npts, L=model.length).derivative()
    wilson = np.asarray(
        [
            [model.wilson_line(site, source) for source in range(model.npts)]
            for site in range(model.npts)
        ]
    )

    np.testing.assert_allclose(dense, bare * wilson, atol=2.0e-13)
    np.testing.assert_allclose(model.apply_derivative(state), dense @ state, atol=2.0e-13)
    np.testing.assert_allclose(dense.conj().T, -dense, atol=2.0e-13)


def test_wilson_derivative_is_exactly_gauge_covariant():
    model = _model()
    beta = 0.4 * np.sin(2.0 * np.pi * model.x / model.length)
    transformed = model.gauge_transform(beta)
    gauge = np.exp(1j * beta)

    expected = gauge[:, None] * model.dense_derivative() * gauge.conjugate()[None, :]
    np.testing.assert_allclose(transformed.dense_derivative(), expected, atol=3.0e-13)

    rng = np.random.default_rng(11)
    state = rng.normal(size=model.npts) + 1j * rng.normal(size=model.npts)
    np.testing.assert_allclose(
        transformed.apply_derivative(gauge * state),
        gauge * model.apply_derivative(state),
        atol=3.0e-13,
    )


def test_fft_dirac_operator_matches_dense_and_is_gauge_covariant():
    model = _model()
    mass = 0.8 + 0.2 * np.cos(2.0 * np.pi * model.x / model.length)
    dense = model.dense_dirac(mass)
    np.testing.assert_allclose(dense, dense.conj().T, atol=3.0e-13)

    rng = np.random.default_rng(13)
    state = rng.normal(size=(model.npts, 2)) + 1j * rng.normal(
        size=(model.npts, 2)
    )
    np.testing.assert_allclose(
        model.apply_dirac(state, mass),
        (dense @ state.reshape(-1)).reshape(model.npts, 2),
        atol=3.0e-13,
    )

    beta = rng.normal(scale=0.3, size=model.npts)
    transformed = model.gauge_transform(beta)
    gauge = np.exp(1j * beta)[:, None]
    np.testing.assert_allclose(
        transformed.apply_dirac(gauge * state, mass),
        gauge * model.apply_dirac(state, mass),
        atol=4.0e-13,
    )
