import numpy as np

from pyqed.models.impurity import SBM, spin_boson_spectral_density, star_to_wilson_chain
from pyqed.models.impurity.sbm import SBM as LegacySBM


def test_star_to_wilson_chain_methods_agree_up_to_gauge():
    frequencies = np.array([1.0, 0.5, 0.25, 0.125])
    couplings = np.array([0.4, 0.2, 0.1, 0.05])

    onsite_l, hopping_l, t0_l, transform_l = star_to_wilson_chain(
        frequencies,
        couplings,
        method="lanczos",
    )
    onsite_h, hopping_h, t0_h, transform_h = star_to_wilson_chain(
        frequencies,
        couplings,
        method="householder",
    )

    np.testing.assert_allclose(onsite_h, onsite_l, atol=1e-12)
    np.testing.assert_allclose(np.abs(hopping_h), np.abs(hopping_l), atol=1e-12)
    assert np.isclose(abs(t0_h), abs(t0_l))

    tridiagonal_l = transform_l @ np.diag(frequencies) @ transform_l.T
    tridiagonal_h = transform_h @ np.diag(frequencies) @ transform_h.T
    np.testing.assert_allclose(np.diag(tridiagonal_l), onsite_l, atol=1e-12)
    np.testing.assert_allclose(np.diag(tridiagonal_h), onsite_h, atol=1e-12)


def test_clean_sbm_discretize_to_chain_is_chainable():
    model = SBM(alpha=0.05, delta=0.1, nmodes=4).discretize()
    chain = model.to_chain()
    onsite, hopping = model

    assert chain.nmodes == 4
    assert chain.delta == 0.1
    np.testing.assert_allclose(onsite, chain.onsite)
    np.testing.assert_allclose(hopping, chain.hopping)
    np.testing.assert_allclose(model.xi, chain.star_frequencies)
    np.testing.assert_allclose(model.g, chain.star_couplings)


def test_spin_boson_spectral_density_is_explicit():
    model = SBM(alpha=0.05, s=1.0, omegac=2.0)

    assert np.isclose(model.spectral_density(1.0), 0.1 * np.pi)
    assert model.spectral_density(-0.1) == 0.0
    assert model.spectral_density(2.1) == 0.0
    np.testing.assert_allclose(
        model.spectral_density(np.array([0.0, 1.0, 2.1])),
        spin_boson_spectral_density(np.array([0.0, 1.0, 2.1]), alpha=0.05, omegac=2.0),
    )


def test_clean_sbm_to_chain_supports_orthogonal_polynomial_scheme():
    alpha = 0.05
    model = SBM(alpha=alpha, delta=0.1)

    chain = model.to_chain(scheme="orthogonal-polynomial", nmodes=6)

    assert chain.nmodes == 6
    assert len(chain.star_frequencies) > chain.nmodes
    assert model.chain_scheme == "orthogonal-polynomial"
    assert np.isclose(chain.impurity_coupling**2, np.pi * alpha, atol=1e-12)
    np.testing.assert_allclose(chain.star_frequencies, model.xi)
    np.testing.assert_allclose(chain.star_couplings, model.g)


def test_orthogonal_polynomial_ohmic_last_onsite_uses_overquadrature():
    model = SBM(alpha=0.05, delta=0.1, omegac=1.0, s=1.0)

    chain = model.to_chain(scheme="orthogonal-polynomial", nmodes=20)

    n = 19
    expected = 0.5 + 1.0 / (2.0 * (2 * n + 1) * (2 * n + 3))
    assert np.isclose(chain.onsite[-1], expected, atol=1e-12)
    assert not np.isclose(chain.onsite[-1], 0.25641025641026)


def test_legacy_sbm_discretize_to_chain_is_chainable():
    model = LegacySBM(Himp=None, alpha=0.05, delta=0.1)
    returned = model.discretize(4)
    chain = model.to_chain()
    onsite, hopping = returned

    assert returned is model
    assert chain.nmodes == 4
    assert chain.delta == 0.1
    np.testing.assert_allclose(onsite, chain.onsite)
    np.testing.assert_allclose(hopping, chain.hopping)


def test_legacy_sbm_to_chain_supports_scheme_keyword():
    model = LegacySBM(Himp=None, alpha=0.05, delta=0.1)

    chain = model.to_chain(scheme="ortho", nmodes=5)

    assert chain.nmodes == 5
    assert model.chain_scheme == "orthogonal-polynomial"
