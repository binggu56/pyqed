"""Fit an invariant/equivariant MLP PES from Cartesian geometries.

Run from the repository root with

    PYTHONPATH=. python examples/fit_equivariant_mlp_pes.py
"""

import numpy as np

from pyqed.ml import EquivariantMLP


def h3_symmetric_energy(geometries):
    r01 = np.linalg.norm(geometries[:, 0] - geometries[:, 1], axis=1)
    r02 = np.linalg.norm(geometries[:, 0] - geometries[:, 2], axis=1)
    r12 = np.linalg.norm(geometries[:, 1] - geometries[:, 2], axis=1)
    return 0.5 * (r01 - 1.0) ** 2 + 0.5 * (r02 - 1.0) ** 2 + 0.5 * (r12 - 1.0) ** 2


def main():
    rng = np.random.default_rng(3)
    geometries = rng.normal(size=(80, 3, 3))
    energies = h3_symmetric_energy(geometries)

    model = EquivariantMLP(
        species=("H", "H", "H"),
        n_radial=8,
        angle_order=2,
        hidden_layers=(64, 64),
        learning_rate=0.005,
        batch_size=32,
        max_iter=1000,
        validation_fraction=0.1,
        random_state=2,
        verbose=True,
    ).fit(geometries, energies)

    geometry = geometries[0]
    print("fit result:", model.result_)
    print("E(R):", model.energy(geometry))
    print("forces:", model.forces(geometry))

    permuted = geometry[[2, 1, 0]]
    print("E(permuted R):", model.energy(permuted))


if __name__ == "__main__":
    main()
