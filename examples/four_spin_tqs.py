"""Four-spin Heisenberg validation for the transformer quantum state."""

import numpy as np

from pyqed.ml import TQS, heisenberg_connections


def heisenberg_hamiltonian(n_visible):
    state = TQS(
        n_visible,
        d_model=4,
        n_heads=1,
        n_layers=1,
        seed=0,
        init_scale=0.0,
    )
    configurations = state.all_configurations()
    connected, elements = heisenberg_connections(configurations)
    hamiltonian = np.zeros((2**n_visible, 2**n_visible), dtype=complex)
    labels = (
        (np.asarray(connected) == -1) * 2 ** np.arange(n_visible - 1, -1, -1)
    ).sum(axis=2)
    rows = np.arange(2**n_visible)[:, None]
    np.add.at(hamiltonian, (rows, labels), np.asarray(elements))
    return hamiltonian


def main():
    hamiltonian = heisenberg_hamiltonian(4)
    exact_energy = np.linalg.eigvalsh(hamiltonian)[0]
    state = TQS(4, d_model=16, n_heads=4, n_layers=2, seed=7)

    for step in range(1, 301):
        state.train_step(
            heisenberg_connections,
            n_samples=2048,
            learning_rate=3.0e-3 if step <= 200 else 1.0e-3,
        )
        if step == 1 or step % 50 == 0:
            psi = state.state_vector()
            energy = np.vdot(psi, hamiltonian @ psi).real
            print(f"step {step:3d} | TQS {energy:.10f} | exact {exact_energy:.10f}")


if __name__ == "__main__":
    main()
