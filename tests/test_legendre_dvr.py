"""Physical regressions for the flat-coordinate Legendre kinetic form."""

import numpy as np
import pytest
from numpy.polynomial import Polynomial

from pyqed.dvr.dvr_1d import LegendreDVR


@pytest.mark.parametrize("npts", [1, 2, 3, 8, 16])
@pytest.mark.parametrize("bounds", [(-1.0, 1.0), (0.3, 3.7)])
def test_kinetic_is_positive_and_annihilates_constant(npts, bounds):
    dvr = LegendreDVR(*bounds, npts, mass=2.3)
    t = dvr.t()
    tol = 1e-12 * max(1.0, np.linalg.norm(t, 2))
    np.testing.assert_allclose(t, t.conj().T, atol=tol)
    assert np.linalg.eigvalsh(t)[0] >= -tol
    constant = np.sqrt(dvr.w / dvr.L)
    np.testing.assert_allclose(t @ constant, 0.0, atol=tol)
    assert dvr.T is t


def test_two_node_flat_kinetic_is_nonzero():
    dvr = LegendreDVR(-1.0, 1.0, 2, mass=2.0)
    np.testing.assert_allclose(dvr.t(), 0.375 * np.array([[1., -1.], [-1., 1.]]))


def test_kinetic_matches_independent_cardinal_derivative_integrals():
    dvr = LegendreDVR(-0.7, 2.1, 6, mass=1.7)
    derivatives = []
    for j, xj in enumerate(dvr.x):
        other_nodes = np.delete(dvr.x, j)
        cardinal = Polynomial.fromroots(other_nodes)
        cardinal /= np.prod(xj - other_nodes) * np.sqrt(dvr.w[j])
        derivatives.append(cardinal.deriv())
    # Overintegrate using independently differentiated polynomial coefficients.
    y, weights = np.polynomial.legendre.leggauss(12)
    x = dvr.xmin + (y + 1) * dvr.L / 2
    values = np.stack([p(x) for p in derivatives], axis=1)
    reference = values.T @ ((weights * dvr.L / 2)[:, None] * values) / (2 * dvr.mass)
    np.testing.assert_allclose(dvr.t(), reference, rtol=1e-11, atol=1e-11)


def test_neumann_free_particle_spectrum_converges():
    mass, length = 2.1, 3.4
    exact = (np.arange(5) * np.pi / length) ** 2 / (2 * mass)
    errors = []
    for npts in [8, 12, 20]:
        dvr = LegendreDVR(0.4, 0.4 + length, npts, mass=mass)
        energies = np.linalg.eigvalsh(dvr.t())[:5]
        # Polynomial Ritz values approach the Neumann spectrum from above.
        assert np.all(energies >= exact - 1e-10)
        errors.append(np.max(np.abs(energies - exact)))
    assert errors[1] < errors[0]
    assert errors[2] < errors[1]
    assert errors[2] < 1e-9


def test_mass_length_and_unit_scaling():
    dvr = LegendreDVR(-1., 1., 5, mass=3.)
    t = dvr.t().copy()
    np.testing.assert_allclose(dvr.t(hc=2.), 4 * t)
    np.testing.assert_allclose(dvr.t(hc=2., mc2=6.), 2 * t)
    dvr.mass = 6.
    np.testing.assert_allclose(dvr.t(), t / 2)
    stretched = LegendreDVR(1., 5., 5, mass=3.)
    np.testing.assert_allclose(stretched.t(), t / 4)


@pytest.mark.parametrize("mass", [0., -1., np.nan, np.inf])
def test_invalid_kinetic_mass_is_rejected(mass):
    dvr = LegendreDVR(-1., 1., 3, mass=mass)
    with pytest.raises(ValueError, match="mass"):
        dvr.t()
    with pytest.raises(ValueError, match="mass"):
        dvr.t(mc2=mass)


def test_momentum_and_grid_unchanged_by_kinetic_call():
    dvr = LegendreDVR(0., np.pi, 7)
    x, w = dvr.x.copy(), dvr.w.copy()
    p = dvr.momentum().copy()
    np.testing.assert_allclose(p @ (np.sqrt(w) * x**3), -3j * np.sqrt(w) * x**2,
                               rtol=1e-12, atol=1e-12)
    dvr.t()
    np.testing.assert_array_equal(dvr.x, x)
    np.testing.assert_array_equal(dvr.w, w)
    np.testing.assert_array_equal(dvr.momentum(), p)


def test_legendre_potential_and_hamiltonian_methods_are_callable():
    dvr = LegendreDVR(-1., 1., 5)
    potential = lambda x: 0.5 * x**2 + 0.2
    reference = np.diag(potential(dvr.x))
    np.testing.assert_allclose(dvr.v(potential), reference)
    np.testing.assert_allclose(dvr.h(potential), dvr.t() + reference)


def test_legendre_run_uses_inherited_hamiltonian_builder():
    dvr = LegendreDVR(-1., 1., 5)
    potential = lambda x: 0.5 * x**2 + 0.2
    reference = np.linalg.eigvalsh(dvr.t() + np.diag(potential(dvr.x)))
    energies, states = dvr.run(potential, num_eigs=dvr.npts)
    np.testing.assert_allclose(energies, reference, atol=1e-12)
    np.testing.assert_allclose(states.conj().T @ states, np.eye(dvr.npts), atol=1e-12)
