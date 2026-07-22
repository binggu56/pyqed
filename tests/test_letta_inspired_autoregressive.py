import jax
import numpy as np

from examples.mps.letta_inspired_autoregressive_crossing import (
    crossing_singlet_hamiltonian,
    initialize_letta,
    initialize_standard,
    letta_log_amplitude,
    standard_log_amplitude,
    state_vector,
)
from examples.mps.letta_inspired_autoregressive_square import (
    apply_hamiltonian,
    configurations,
    exact_sector_energy,
    hamiltonian_data,
    square_edges,
)


def test_crossing_singlet_ground_energy():
    spectrum = np.linalg.eigvalsh(np.asarray(crossing_singlet_hamiltonian()))
    assert np.isclose(spectrum[0], -3.0)


def test_autoregressive_states_are_normalized():
    standard = initialize_standard(jax.random.PRNGKey(3), hidden_dim=6)
    letta = initialize_letta(
        jax.random.PRNGKey(4), hidden_dim=6, virtual_dim=4, rank=2
    )

    standard_state = state_vector(standard, standard_log_amplitude)
    letta_state = state_vector(letta, letta_log_amplitude)

    assert np.isclose(np.vdot(standard_state, standard_state), 1.0)
    assert np.isclose(np.vdot(letta_state, letta_state), 1.0)


def test_matrix_free_two_spin_heisenberg_operator():
    configs = configurations(2)
    edges = square_edges(1, 2, j2=0.0)
    data = hamiltonian_data(configs, edges)
    columns = []
    for column in range(4):
        state = np.zeros(4)
        state[column] = 1.0
        columns.append(np.asarray(apply_hamiltonian(state, *data)))
    matrix = np.stack(columns, axis=1)

    assert np.allclose(
        matrix,
        np.asarray(
            [
                [0.25, 0.0, 0.0, 0.0],
                [0.0, -0.25, 0.5, 0.0],
                [0.0, 0.5, -0.25, 0.0],
                [0.0, 0.0, 0.0, 0.25],
            ]
        ),
    )
    energy, dimension = exact_sector_energy(2, edges)
    assert dimension == 2
    assert np.isclose(energy, -0.75)
