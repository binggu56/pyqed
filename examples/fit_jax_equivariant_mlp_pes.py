"""Fit an equivariant MLP PES with JAX autodiff force matching.

Run from the repository root with

    PYTHONPATH=. python examples/fit_jax_equivariant_mlp_pes.py
"""

import numpy as np

from pyqed.ml import EquivariantMLP


def h3_symmetric_energy(geometries):
    r01 = geometries[:, 0] - geometries[:, 1]
    r02 = geometries[:, 0] - geometries[:, 2]
    r12 = geometries[:, 1] - geometries[:, 2]
    return np.sum(r01 * r01, axis=1) + np.sum(r02 * r02, axis=1) + np.sum(r12 * r12, axis=1)


def h3_symmetric_forces(geometries):
    forces = np.zeros_like(geometries)
    for i, j in ((0, 1), (0, 2), (1, 2)):
        rij = geometries[:, i] - geometries[:, j]
        forces[:, i] += -2.0 * rij
        forces[:, j] += 2.0 * rij
    return forces


def main():
    rng = np.random.default_rng(3)
    geometries = rng.normal(size=(64, 3, 3))
    energies = h3_symmetric_energy(geometries)
    forces = h3_symmetric_forces(geometries)

    model = EquivariantMLP(
        species=("H", "H", "H"),
        n_radial=8,
        angle_order=2,
        hidden_layers=(64, 64),
        learning_rate=0.003,
        batch_size=32,
        max_iter=500,
        validation_fraction=0.1,
        force_weight=0.1,
        random_state=2,
        verbose=True,
    ).fit(geometries, energies, forces=forces)

    geometry = geometries[0]
    print("fit result:", model.result_)
    print("E(R):", model.energy(geometry))
    print("F(R):", model.forces(geometry))


if __name__ == "__main__":
    main()
