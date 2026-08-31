import numpy as np

from pyqed.dvr.dvr_1d import LegendreDVR, PODVR


def test_podvr_builds_nonuniform_hermitian_operators():
    dvr = PODVR(0.6, 5.0, 8, De=0.18, a=1.1, re=1.4, mass=1836.0)

    assert dvr.x.shape == (8,)
    assert np.all(dvr.x > 0.6)
    assert np.all(dvr.x < 5.0)
    assert np.all(np.diff(dvr.x) > 0.0)
    assert not np.allclose(np.diff(dvr.x), np.diff(dvr.x)[0])

    np.testing.assert_allclose(dvr.t(), dvr.t().conj().T, atol=1e-12)
    np.testing.assert_allclose(dvr.momentum(), dvr.momentum().conj().T, atol=1e-12)

    assert np.all(dvr.w > 0.0)
    np.testing.assert_allclose(np.sum(dvr.w), dvr.L)


def test_podvr_solves_reference_hamiltonian():
    dvr = PODVR(0.6, 5.0, 10, De=0.18, a=1.1, re=1.4, mass=1836.0)
    energies, vecs = dvr.run(dvr.reference_potential, num_eigs=4)

    assert energies.shape == (4,)
    assert vecs.shape == (10, 4)
    assert np.all(np.isfinite(energies))
    assert np.all(np.diff(energies) >= 0.0)


def test_podvr_accepts_generic_reference_potential():
    def harmonic_reference(x):
        return 0.5 * (x - 1.4) ** 2

    dvr = PODVR(0.6, 5.0, 8, v_ref=harmonic_reference, mass=1836.0)

    np.testing.assert_allclose(
        dvr.reference_potential(dvr.x),
        harmonic_reference(dvr.x),
    )
    assert dvr.t().shape == (8, 8)


def test_legendre_dvr_builds_angular_grid_and_operators():
    dvr = LegendreDVR(0.0, np.pi, 6)

    assert dvr.x.shape == (6,)
    assert np.all(dvr.x > 0.0)
    assert np.all(dvr.x < np.pi)
    assert np.all(np.diff(dvr.x) > 0.0)
    np.testing.assert_allclose(np.sum(dvr.w), np.pi)
    assert dvr.t().shape == (6, 6)
    assert dvr.momentum().shape == (6, 6)


def test_legendre_dvr_kinetic_is_positive_and_recovers_free_rotor_levels():
    dvr = LegendreDVR(0.0, np.pi, 17, mass=2.0)
    kinetic = dvr.t()
    energies = np.linalg.eigvalsh(kinetic)

    np.testing.assert_allclose(kinetic, kinetic.conj().T, atol=1.0e-12)
    assert energies[0] >= -1.0e-12
    np.testing.assert_allclose(
        energies[:6],
        np.arange(6, dtype=float) ** 2 / 4.0,
        atol=1.0e-7,
    )
