"""Optimize a small Heisenberg chain with an RBM and exact-weighted SR."""

import numpy as np

from pyqed.ml import RBM


def heisenberg_hamiltonian(n_sites: int) -> np.ndarray:
    """Return the open-chain spin-1/2 Heisenberg Hamiltonian with J=1."""

    dimension = 2**n_sites
    hamiltonian = np.zeros((dimension, dimension), dtype=complex)
    labels = np.arange(dimension)

    for label in labels:
        bits = (label >> np.arange(n_sites - 1, -1, -1)) & 1
        spins = 1 - 2 * bits
        for site in range(n_sites - 1):
            neighbor = site + 1
            hamiltonian[label, label] += 0.25 * spins[site] * spins[neighbor]
            if spins[site] != spins[neighbor]:
                flipped_label = label ^ (1 << (n_sites - 1 - site))
                flipped_label ^= 1 << (n_sites - 1 - neighbor)
                hamiltonian[label, flipped_label] += 0.5

    return hamiltonian


def main() -> None:
    n_sites = 4
    n_steps = 300
    rbm = RBM(n_sites, n_hidden=2 * n_sites, seed=7, init_scale=0.03)
    configurations = rbm.all_configurations()
    hamiltonian = heisenberg_hamiltonian(n_sites)
    exact_energies, exact_states = np.linalg.eigh(hamiltonian)

    for step in range(n_steps + 1):
        psi = rbm.state_vector(normalize=False)
        probabilities = np.abs(psi) ** 2
        probabilities /= np.sum(probabilities)
        local_energies = (hamiltonian @ psi) / psi
        energy = float(np.real(np.sum(probabilities * local_energies)))

        if step % 50 == 0:
            print(f"step {step:3d} | energy {energy:.10f}")
        if step == n_steps:
            break

        rbm.sr_step(
            configurations,
            local_energies,
            sample_weights=probabilities,
            learning_rate=0.05 if step < 100 else 0.02,
            diagonal_shift=max(1.0e-4, 1.0e-2 * 0.98**step),
        )

    fidelity = abs(np.vdot(exact_states[:, 0], rbm.state_vector())) ** 2
    print(f"exact ground energy: {exact_energies[0]:.10f}")
    print(f"ground-state fidelity: {fidelity:.10f}")


if __name__ == "__main__":
    main()
