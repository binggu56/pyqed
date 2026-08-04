from itertools import product

import numpy as np

from examples.mps.unordered_group_letta_lieb_liniger import (
    boson_basis,
    creation_operators,
    lieb_liniger_hamiltonian,
    momentum_transfer_operators,
    unordered_projected_state,
    virtual_charge_shift,
)


def test_condensate_has_expected_contact_energy():
    particles = 3
    length = 3.0
    coupling = 0.7
    modes, basis, hamiltonian = lieb_liniger_hamiltonian(
        particles=particles,
        cutoff=1,
        length=length,
        coupling=coupling,
    )
    occupation = tuple(particles if mode == 0 else 0 for mode in modes)
    state = np.zeros(len(basis))
    state[basis.index(occupation)] = 1.0
    np.testing.assert_allclose(
        np.vdot(state, hamiltonian @ state),
        coupling * particles * (particles - 1) / length,
    )


def test_virtual_shift_cancels_physical_momentum_transfer():
    shift = virtual_charge_shift(2, 1)
    charges = np.arange(-2, 3)
    for column, charge in enumerate(charges):
        rows = np.flatnonzero(abs(shift[:, column]) > 0.0)
        if rows.size:
            assert charges[rows[0]] == charge - 1


def test_density_transfer_moves_one_particle_between_momentum_modes():
    basis = boson_basis(3, 2)
    creators = creation_operators(basis)
    rho = momentum_transfer_operators(creators, cutoff=1)[1]
    source = basis.index((0, 2, 0))
    target = basis.index((0, 1, 1))
    np.testing.assert_allclose(rho[target, source], np.sqrt(2.0))


def test_zero_transfer_strength_recovers_the_condensate():
    particles = 3
    cutoff = 1
    state = unordered_projected_state(
        np.zeros(5),
        particles=particles,
        cutoff=cutoff,
        virtual_cutoff=1,
    )
    fixed_basis = [
        occupation
        for occupation in product(range(particles + 1), repeat=2 * cutoff + 1)
        if sum(occupation) == particles
    ]
    fixed_basis.sort()
    condensate = np.zeros(len(fixed_basis))
    condensate[fixed_basis.index((0, particles, 0))] = 1.0
    np.testing.assert_allclose(abs(np.vdot(state, condensate)), 1.0, atol=1.0e-12)


def test_corrected_d5_transfer_state_reproduces_regulated_c1_ground_state():
    parameters = np.array(
        [
            -1.1539664139137884,
            -3.0559858185379847,
            -2.0609118051139568,
            -4.450202654603142,
            -2.4897087157506697,
            4.056912789297882,
            -4.54110517500333,
            2.028676426201901,
            -5.000248984164755,
        ]
    )
    _, _, hamiltonian = lieb_liniger_hamiltonian(
        particles=4,
        cutoff=2,
        length=4.0,
        coupling=1.0,
    )
    state = unordered_projected_state(
        parameters,
        particles=4,
        cutoff=2,
        virtual_cutoff=2,
    )
    energy = np.real(np.vdot(state, hamiltonian @ state))
    exact = np.linalg.eigvalsh(hamiltonian)[0]
    np.testing.assert_allclose(energy, exact, atol=2.0e-10)
