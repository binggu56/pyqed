import numpy as np

from pyqed.models.retinal_hs import RetinalHumphreySchulten
from pyqed.namd.retinal_hs_cavity import (
    RetinalHSTwoMoleculeCavityDynamics,
)
from pyqed.units import au2kcalmol


def test_hs_diabatic_topology_and_couplings():
    model = RetinalHumphreySchulten()
    potential = model.diabatic_potential(np.asarray([0.0, 0.5 * np.pi, np.pi]))
    diagonal = np.diagonal(potential, axis1=-2, axis2=-1) * au2kcalmol

    np.testing.assert_allclose(diagonal[0], [54.0, 0.0, 50.0])
    np.testing.assert_allclose(diagonal[1], [27.0, 27.0, 54.0])
    np.testing.assert_allclose(diagonal[2], [0.0, 54.0, 50.0], atol=1.0e-12)
    np.testing.assert_allclose(
        potential[0] * au2kcalmol,
        [[54.0, 0.5, 1.0], [0.5, 0.0, 1.0], [1.0, 1.0, 50.0]],
    )


def test_two_molecule_hamiltonian_is_hermitian_and_norm_is_conserved():
    dynamics = RetinalHSTwoMoleculeCavityDynamics(
        nphi=9,
        nphotons=3,
        coupling_ev=0.01,
    )
    assert dynamics.phi.size == 9
    np.testing.assert_allclose(
        dynamics.internal_hamiltonian,
        dynamics.internal_hamiltonian.conj().T,
    )
    dynamics.run(tmax_fs=0.4, dt_fs=0.1, save_every=1)
    np.testing.assert_allclose(dynamics.norm, 1.0, atol=2.0e-13)


def test_zero_coupling_two_molecule_dynamics_factorizes():
    common = dict(
        nphi=9,
        nphotons=2,
        coupling_ev=0.0,
        include_dse=False,
    )
    one = RetinalHSTwoMoleculeCavityDynamics(
        nmolecules=1,
        **common,
    ).run(tmax_fs=0.4, dt_fs=0.1, save_every=1)
    two = RetinalHSTwoMoleculeCavityDynamics(
        nmolecules=2,
        **common,
    ).run(tmax_fs=0.4, dt_fs=0.1, save_every=1)

    np.testing.assert_allclose(
        two.electronic_populations[:, 0],
        one.electronic_populations[:, 0],
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        two.reacted_population[:, 0],
        one.reacted_population[:, 0],
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        two.joint_a,
        one.electronic_populations[:, 0, 0] ** 2,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        two.connected_exchange_coherence,
        0.0,
        atol=2.0e-12,
    )


def test_zero_temperature_ensemble_matches_pure_initial_state():
    pure = RetinalHSTwoMoleculeCavityDynamics(
        nphi=9,
        nphotons=2,
        coupling_ev=0.01,
    ).run(tmax_fs=0.2, dt_fs=0.1, save_every=1)
    thermal = RetinalHSTwoMoleculeCavityDynamics(
        nphi=9,
        nphotons=2,
        coupling_ev=0.01,
    ).run_thermal_ensemble(
        temperature_k=0.0,
        samples=2,
        tmax_fs=0.2,
        dt_fs=0.1,
        save_every=1,
    )

    np.testing.assert_array_equal(thermal.sampled_thermal_levels, 0)
    np.testing.assert_allclose(
        thermal.electronic_populations,
        pure.electronic_populations,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        thermal.photon_number,
        pure.photon_number,
        atol=2.0e-12,
    )
