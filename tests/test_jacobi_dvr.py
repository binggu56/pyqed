import numpy as np
import pytest

from pyqed.dvr import DVR, JacobiDVR, PTDVR


def test_jacobi_dvr_builds_orthogonal_endpoint_adapted_basis():
    dvr = JacobiDVR(0.0, np.pi, 10, alpha=0.4, beta=1.2, mass=2.0)

    assert np.all(np.diff(dvr.x) > 0.0)
    assert np.all(dvr.x > 0.0)
    assert np.all(dvr.x < np.pi)
    assert np.all(dvr.w > 0.0)
    np.testing.assert_allclose(dvr.U @ dvr.U.T, np.eye(dvr.npts), atol=2e-13)
    np.testing.assert_allclose(
        dvr.f(),
        np.diag(1.0 / np.sqrt(dvr.w)),
        atol=2e-12,
    )
    np.testing.assert_allclose(
        dvr.momentum(), dvr.momentum().conj().T, atol=2e-13
    )


def test_jacobi_dvr_recovers_poschl_teller_reference_spectrum():
    dvr = JacobiDVR(0.2, 2.7, 14, alpha=0.7, beta=1.1, mass=3.0)
    hamiltonian = dvr.t() + np.diag(dvr.reference_potential())
    energies = np.linalg.eigvalsh(hamiltonian)
    expected = (
        (np.pi / dvr.L) ** 2
        * (np.arange(dvr.npts) + 0.5 * (dvr.alpha + dvr.beta + 1.0)) ** 2
        / (2.0 * dvr.mass)
    )

    np.testing.assert_allclose(hamiltonian, hamiltonian.T, atol=2e-13)
    np.testing.assert_allclose(energies, expected, atol=2e-12)


def test_ptdvr_associated_legendre_constructor_has_symmetric_endpoint_term():
    dvr = PTDVR.associated_legendre(0.0, np.pi, 12, m=2, mass=1.5)
    expected = (
        (2.0**2 - 0.25)
        / np.sin(dvr.q) ** 2
        / (2.0 * dvr.mass)
    )

    assert isinstance(dvr, JacobiDVR)
    assert dvr.alpha == dvr.beta == 2.0
    np.testing.assert_allclose(dvr.reference_potential(), expected)


def test_jacobi_dvr_works_as_a_product_axis():
    angle = JacobiDVR(0.0, np.pi, 5, alpha=1.0, beta=1.0)
    product = DVR.from_axes((angle,), names=("theta",))

    assert product.shape == (5,)
    np.testing.assert_allclose(product.t().toarray(), angle.t())


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"npts": 0}, "positive integer"),
        ({"npts": 4, "alpha": -0.5}, "alpha"),
        ({"npts": 4, "beta": -0.5}, "beta"),
        ({"npts": 4, "mass": 0.0}, "mass"),
    ],
)
def test_jacobi_dvr_rejects_invalid_parameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        JacobiDVR(0.0, np.pi, **kwargs)
