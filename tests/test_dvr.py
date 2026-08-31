import numpy as np
import pytest

from pyqed.dvr import DVR, ExponentialDVR, HermiteDVR, LegendreDVR, SineDVR


def test_dvr_builds_named_product_grid():
    dvr = DVR(
        domains=((-1.0, 1.0), (0.0, 3.0)),
        npts=(3, 2),
        mass=(2.0, 4.0),
        names=("x", "q"),
    )

    assert dvr.ndim == 2
    assert dvr.names == ("x", "q")
    assert dvr.shape == (3, 2)
    assert dvr.mass == (2.0, 4.0)
    assert dvr.npts == dvr.shape == (3, 2)
    assert dvr.size == 6
    assert dvr.points.shape == (6, 2)
    np.testing.assert_allclose(dvr.points[0], (dvr.x[0][0], dvr.x[1][0]))
    np.testing.assert_allclose(dvr.points[-1], (dvr.x[0][-1], dvr.x[1][-1]))
    assert dvr.axis("q") == 1
    assert dvr.values((0.2, 0.3)) == {"x": 0.2, "q": 0.3}


def test_dvr_builds_product_operators():
    dvr = DVR(
        domains=((-2.0, 2.0), (-1.0, 1.0)),
        npts=(3, 2),
        mass=(1.0, 2.0),
    )

    kinetic = dvr.t()
    expected = np.kron(dvr.dvr[0].t(), np.eye(2))
    expected += np.kron(np.eye(3), dvr.dvr[1].t())
    np.testing.assert_allclose(kinetic.toarray(), expected)

    potential = dvr.v(lambda x, y: x**2 + 2.0 * y**2)
    expected_values = np.array(
        [x**2 + 2.0 * y**2 for x, y in dvr.points]
    )
    np.testing.assert_allclose(potential.diagonal(), expected_values)
    np.testing.assert_allclose(
        dvr.buildH(lambda x, y: x**2 + 2.0 * y**2).toarray(),
        expected + np.diag(expected_values),
    )


def test_dvr_combines_existing_one_dimensional_axes():
    radial = SineDVR(1.0, 3.0, 3, mass=2.0)
    angle = LegendreDVR(0.0, np.pi, 4, mass=5.0)

    dvr = DVR.from_axes((radial, angle), names=("R", "theta"))

    assert dvr.names == ("R", "theta")
    assert dvr.axes == (radial, angle)
    assert dvr.shape == (3, 4)
    assert dvr.mass == (2.0, 5.0)
    assert dvr.kinetic().shape == (12, 12)


def test_exponential_dvr_has_consistent_periodic_momentum_and_kinetic():
    dvr = ExponentialDVR(3, L=2.0 * np.pi, x0=np.pi, mass=2.5)
    momentum = dvr.momentum()

    assert np.all(np.isfinite(momentum))
    np.testing.assert_allclose(momentum, momentum.conj().T, atol=1.0e-14)
    np.testing.assert_allclose(
        dvr.t(), momentum @ momentum / (2.0 * dvr.mass), atol=1.0e-13
    )


def test_exponential_dvr_supports_even_permutation_grids():
    dvr = ExponentialDVR(npts=12, L=2.0 * np.pi, x0=np.pi, mass=2.0)

    assert dvr.npts == 12
    np.testing.assert_allclose(dvr.momentum(), dvr.momentum().conj().T)
    np.testing.assert_allclose(dvr.t(), dvr.t().conj().T)
    assert np.all(np.isfinite(dvr.momentum()))


@pytest.mark.parametrize("npts", (11, 12))
def test_exponential_dvr_toeplitz_descriptor_matches_dense_kinetic(npts):
    dvr = ExponentialDVR(npts=npts, L=7.0, mass=2.3)
    column, row = dvr.kinetic_toeplitz()

    assert column.shape == row.shape == (npts,)
    np.testing.assert_allclose(column, dvr.t()[:, 0])
    np.testing.assert_allclose(row, dvr.t()[0])


def test_hermite_dvr_uses_harmonic_oscillator_scaling():
    omega = 0.004
    dvr = HermiteDVR(
        npts=25,
        mass=1.0 / omega,
        omega=omega,
        center=0.3,
    )

    np.testing.assert_allclose(dvr.alpha, 1.0)
    np.testing.assert_allclose(dvr.x, 0.6 - dvr.x[::-1])
    np.testing.assert_allclose(dvr.x[12], 0.3, atol=1.0e-14)
    assert dvr.x[0] < -5.8
    assert dvr.x[-1] > 6.4

    compressed = HermiteDVR(
        npts=25,
        mass=4.0 / omega,
        omega=omega,
        center=0.3,
    )
    np.testing.assert_allclose(
        compressed.x - 0.3,
        0.5 * (dvr.x - 0.3),
    )


def test_hermite_dvr_recovers_reference_oscillator_spectrum():
    omega = 0.007
    mass = 1.0 / omega
    dvr = HermiteDVR(npts=25, mass=mass, omega=omega)
    potential = 0.5 * mass * omega**2 * dvr.x**2
    energies, states = np.linalg.eigh(dvr.t() + np.diag(potential))

    np.testing.assert_allclose(
        energies[:10],
        omega * (np.arange(10) + 0.5),
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        np.abs(states[:, 0]),
        dvr.harmonic_state(0),
        atol=2.0e-13,
    )


def test_hermite_dvr_quadrature_and_operators_are_consistent():
    mass = 3.0
    omega = 0.2
    dvr = HermiteDVR(npts=17, mass=mass, omega=omega, center=-0.4)
    relative = dvr.x - dvr.center
    gaussian = (
        np.sqrt(dvr.alpha) / np.pi**0.25
        * np.exp(-0.5 * (dvr.alpha * relative) ** 2)
    )
    coefficients = np.sqrt(dvr.w) * gaussian

    np.testing.assert_allclose(
        coefficients,
        dvr.harmonic_state(),
        atol=2.0e-14,
    )
    np.testing.assert_allclose(np.linalg.norm(coefficients), 1.0)
    np.testing.assert_allclose(dvr.t(), dvr.t().T, atol=1.0e-14)
    np.testing.assert_allclose(
        dvr.momentum(),
        dvr.momentum().conj().T,
        atol=1.0e-14,
    )
    kinetic_propagator = dvr.expT(0.7)
    np.testing.assert_allclose(
        kinetic_propagator.conj().T @ kinetic_propagator,
        np.eye(dvr.npts),
        atol=2.0e-14,
    )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"npts": 0, "mass": 1.0, "omega": 1.0}, "positive integer"),
        ({"npts": 4, "mass": 0.0, "omega": 1.0}, "mass"),
        ({"npts": 4, "mass": 1.0, "omega": -1.0}, "omega"),
        (
            {"npts": 4, "mass": 1.0, "omega": 1.0, "center": np.inf},
            "center",
        ),
    ],
)
def test_hermite_dvr_rejects_invalid_setup(kwargs, message):
    with pytest.raises(ValueError, match=message):
        HermiteDVR(**kwargs)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"domains": ((1.0, -1.0),), "npts": (3,)}, "upper bound"),
        (
            {
                "domains": ((-1.0, 1.0), (-1.0, 1.0)),
                "npts": (3, 3),
                "names": ("q", "q"),
            },
            "unique",
        ),
        (
            {
                "domains": ((-1.0, 1.0), (-1.0, 1.0)),
                "npts": (3, 3),
                "mass": (1.0,),
            },
            "one value",
        ),
    ],
)
def test_dvr_rejects_invalid_setup(kwargs, message):
    with pytest.raises(ValueError, match=message):
        DVR(**kwargs)
