import numpy as np

from pyqed.models.retinal import RetinalHahnStock
from pyqed.namd.retinal_dvr import RetinalDVRDynamics
from pyqed.units import au2ev, au2fs


def test_hahn_stock_default_parameters_and_periodicity():
    model = RetinalHahnStock()

    np.testing.assert_allclose(
        list(model.parameters_ev.values()),
        [4.84e-4, 2.48, 3.6, 1.09, 0.19, 0.10, 0.19],
    )
    phi = np.array([-1.2, 0.3, 2.4])
    q = np.array([0.1, -0.2, 0.7])
    np.testing.assert_allclose(
        model.dpes(phi, q),
        model.dpes(phi + 2.0 * np.pi, q),
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        model.dpes(phi, q),
        model.dpes(phi, q).swapaxes(-1, -2),
    )


def test_hahn_stock_conical_intersection_is_degenerate():
    model = RetinalHahnStock()
    phi, q = model.conical_intersection()
    energies = model.apes(phi, q) * au2ev

    np.testing.assert_allclose(energies[0], energies[1], atol=1.0e-13)
    np.testing.assert_allclose(
        phi,
        np.arccos(1.0 - 2.0 * 2.48 / (3.6 + 1.09)),
        atol=1.0e-14,
    )
    assert model.cis_mask(np.array([0.0, 2.0])).tolist() == [True, False]


def test_mixed_dvr_short_propagation_preserves_norm():
    dynamics = RetinalDVRDynamics(nphi=15, nq=10)

    assert dynamics.phi[0] == -0.5 * np.pi
    assert dynamics.phi[-1] < 1.5 * np.pi
    np.testing.assert_allclose(
        np.sum(np.abs(dynamics.initial_state) ** 2, axis=(0, 1)),
        [0.0, 1.0],
        atol=1.0e-14,
    )

    dynamics.run(tmax_fs=0.4, dt_fs=0.05, save_every=2)
    np.testing.assert_allclose(dynamics.norm, 1.0, atol=2.0e-13)
    np.testing.assert_allclose(
        dynamics.diabatic_populations.sum(axis=1),
        1.0,
        atol=2.0e-13,
    )


def test_periodic_fft_step_matches_dvr_kinetic_propagator():
    dynamics = RetinalDVRDynamics(nphi=15, nq=6)
    dynamics._prepare_propagators(0.03)
    rng = np.random.default_rng(12)
    state = rng.normal(size=dynamics.initial_state.shape) + 1j * rng.normal(
        size=dynamics.initial_state.shape
    )

    fft_state = np.fft.ifft(
        dynamics.u_phi_half[:, None, None]
        * np.fft.fft(state, axis=0, norm="ortho"),
        axis=0,
        norm="ortho",
    )
    dense_state = np.einsum(
        "ij,jqs->iqs",
        _dense_unitary(
            dynamics.t_phi,
            0.5 * 0.03 / au2fs,
        ),
        state,
    )
    np.testing.assert_allclose(fft_state, dense_state, atol=2.0e-13)


def _dense_unitary(hamiltonian, time):
    energies, states = np.linalg.eigh(hamiltonian)
    return (states * np.exp(-1j * time * energies)[None, :]) @ states.conj().T
