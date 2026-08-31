import numpy as np
import pytest

from pyqed.dvr import ExponentialDVR, LegendreDVR, SineDVR
from pyqed.ldr import keo


def test_aph_geometry_has_zero_com_and_requested_hyperradius():
    aph = keo.APH(
        ("H", "D", "T"), [1.0, 2.0, 3.0], jacobi_atoms=(0, (1, 2))
    )
    coordinates = np.array([4.2, 0.71, 1.1])
    geometry = aph.cartesian(coordinates)

    np.testing.assert_allclose(aph.masses @ geometry, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(aph.hyperradius(geometry), coordinates[0])
    assert tuple(atom for atom, _ in aph.geometry(coordinates)) == ("H", "D", "T")


def test_aph_inverse_preserves_geometry_up_to_rotation():
    aph = keo.APH(
        ("H", "D", "T"), [1.0, 2.0, 3.0], jacobi_atoms=(2, (0, 1))
    )
    coordinates = np.array([3.7, 0.83, 0.8])
    recovered = aph.from_cartesian(aph.cartesian(coordinates))

    np.testing.assert_allclose(recovered[:2], coordinates[:2], atol=1.0e-13)
    np.testing.assert_allclose(
        aph.pair_distances(recovered),
        aph.pair_distances(coordinates),
        atol=1.0e-13,
    )


def test_equal_mass_arrangement_sectors_are_permutations():
    aph = keo.APH(("H", "H", "H"), [1.0, 1.0, 1.0])
    coordinates = np.array([3.0, 0.76, 0.37])
    reference = np.sort(aph.pair_distances(coordinates))

    for sector in range(1, 6):
        shifted = coordinates + np.array([0.0, 0.0, sector * np.pi / 3.0])
        np.testing.assert_allclose(
            np.sort(aph.pair_distances(shifted)), reference, atol=1.0e-13
        )


def test_aph_dense_and_mpo_operators_match():
    pytest.importorskip("jax")
    aph = keo.APH(("H", "H", "H"), [1.0, 1.0, 1.0])
    dvrs = (
        SineDVR(2.0, 4.0, 2, mass=1.0),
        LegendreDVR(0.08, np.pi / 2.0 - 0.08, 2, mass=1.0),
        ExponentialDVR(1, L=2.0 * np.pi, x0=np.pi, mass=1.0),
    )

    dense, metric, pseudopotential = aph.matrix(dvrs, return_fields=True)
    mpo = aph.mpo(dvrs)

    assert metric.shape == (2, 2, 3, 3, 3)
    assert pseudopotential.shape == (2, 2, 3)
    assert np.all(np.isfinite(metric))
    assert np.all(np.isfinite(pseudopotential))
    np.testing.assert_allclose(metric[..., 0, 0], 1.0 / aph.mu, atol=1.0e-11)
    np.testing.assert_allclose(metric[..., 0, 1:], 0.0, atol=1.0e-11)
    np.testing.assert_allclose(dense, dense.conj().T, atol=1.0e-11)
    np.testing.assert_allclose(mpo.to_dense(), dense, atol=1.0e-11)


def test_aph_rejects_singular_inverse_geometries():
    aph = keo.APH(("H", "H", "H"), [1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="collinear"):
        aph.from_cartesian(aph.cartesian([3.0, np.pi / 2.0, 0.2]))
    with pytest.raises(ValueError, match="theta=0"):
        aph.from_cartesian(aph.cartesian([3.0, 0.0, 0.2]))


def test_aph_fixed_rho_angular_hamiltonian_is_hermitian():
    pytest.importorskip("jax")
    aph = keo.APH(("H", "H", "H"), [1.0, 1.0, 1.0])
    dvrs = (
        LegendreDVR(0.0, np.pi / 2.0, 2),
        ExponentialDVR(npts=3, L=2.0 * np.pi, x0=np.pi),
    )
    potential = np.zeros((2, 3))

    h_angular = aph.angular_hamiltonian(3.0, dvrs, potential)

    assert h_angular.shape == (6, 6)
    np.testing.assert_allclose(h_angular, h_angular.conj().T, atol=1.0e-11)
