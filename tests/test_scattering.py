import numpy as np

from pyqed.mps.mpo import sop_to_mpo
from pyqed.mps.mps import MPS
from pyqed.namd import WavepacketScattering


def test_wavepacket_scattering_assembles_bo_hamiltonian_with_cap():
    dims = (2, 2)
    identity = np.eye(2)
    sigma_x = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    kinetic = sop_to_mpo(
        dims,
        [
            (0.3, (sigma_x, identity)),
            (-0.2, (identity, sigma_x)),
        ],
    )
    potential = np.asarray([[0.1, 0.2], [0.4, 0.7]])
    absorber = np.asarray([[0.0, 0.01], [0.02, 0.03]])

    calculation = WavepacketScattering(
        kinetic=kinetic,
        potential=potential,
        absorber=absorber,
    )

    expected = kinetic.to_dense() + np.diag(potential.reshape(-1))
    expected = expected - 1j * np.diag(absorber.reshape(-1))
    np.testing.assert_allclose(calculation.hamiltonian.to_dense(), expected)


def test_wavepacket_scattering_tddmrg_chain_preserves_cap_norm_loss():
    dims = (2, 2)
    gamma = 0.3
    kinetic = sop_to_mpo(dims, [])
    psi0 = MPS(
        [
            np.asarray([1.0, 0.0], dtype=complex).reshape(1, 2, 1),
            np.asarray([0.0, 1.0], dtype=complex).reshape(1, 2, 1),
        ]
    )
    calculation = WavepacketScattering(
        kinetic=kinetic,
        potential=np.zeros(dims),
        absorber=np.full(dims, gamma),
    )

    returned = calculation.tddmrg(max_bond=4, cutoff=1.0e-12).run(
        psi0,
        dt=0.1,
        steps=2,
        progress=False,
    )

    assert returned is calculation
    assert calculation.success
    np.testing.assert_allclose(calculation.times, [0.0, 0.1, 0.2])
    np.testing.assert_allclose(
        calculation.final_state.norm_squared(),
        np.exp(-2.0 * gamma * 0.2),
        rtol=1.0e-10,
    )
    np.testing.assert_allclose(calculation.norms[-1] ** 2, np.exp(-2.0 * gamma * 0.2))
