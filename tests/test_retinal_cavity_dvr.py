import numpy as np

from pyqed.namd.retinal_cavity_dvr import RetinalCavityDVRDynamics
from pyqed.namd.retinal_dvr import RetinalDVRDynamics


def test_zero_coupling_matches_bare_retinal_dynamics():
    bare = RetinalDVRDynamics(nphi=15, nq=8)
    bare.run(tmax_fs=0.4, dt_fs=0.05, save_every=2)
    cavity = RetinalCavityDVRDynamics(
        coupling_ev=0.0,
        nphotons=2,
        ntheta=15,
        nq=8,
    )
    cavity.run(tmax_fs=0.4, dt_fs=0.05, save_every=2)

    np.testing.assert_allclose(
        cavity.diabatic_populations,
        bare.diabatic_populations,
        atol=3.0e-13,
    )
    np.testing.assert_allclose(cavity.photon_number, 0.0, atol=1.0e-14)


def test_local_polaritonic_potential_is_hermitian():
    dynamics = RetinalCavityDVRDynamics(
        coupling_ev=0.04,
        nphotons=3,
        ntheta=15,
        nq=6,
    )

    np.testing.assert_allclose(
        dynamics.local_potential,
        dynamics.local_potential.swapaxes(-1, -2).conj(),
    )
    assert dynamics.local_potential.shape == (15, 6, 6, 6)


def test_photon_jump_maps_one_photon_to_vacuum():
    dynamics = RetinalCavityDVRDynamics(
        nphotons=3,
        ntheta=15,
        nq=6,
    )
    states = np.zeros((2,) + dynamics.initial_state.shape, dtype=complex)
    states[..., 0, 1] = dynamics.molecular_dvr.initial_state[..., 1]
    jumped = dynamics._apply_photon_jump(states)

    np.testing.assert_allclose(
        np.sum(np.abs(jumped[..., 0]) ** 2, axis=(1, 2, 3)),
        1.0,
    )
    np.testing.assert_allclose(jumped[..., 1:], 0.0)
