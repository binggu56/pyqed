"""Fit a scalar-vector equivariant message-passing PES.

Run from the repository root with

    PYTHONPATH=. python examples/fit_equivariant_mpnn_pes.py
"""

import numpy as np

from pyqed.ml import MPNN


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
    rng = np.random.default_rng(8)
    geometries = rng.normal(size=(64, 3, 3))
    energies = h3_symmetric_energy(geometries)
    forces = h3_symmetric_forces(geometries)

    model = MPNN(
        species=("H", "H", "H"),
        features=32,
        n_layers=3,
        n_radial=8,
        readout_hidden=32,
        learning_rate=0.002,
        batch_size=16,
        max_iter=300,
        force_weight=0.1,
        random_state=4,
        verbose=True,
    ).fit(geometries, energies, forces=forces)

    geometry = geometries[0]
    print("fit result:", model.result_)
    print("E(R):", model.energy(geometry))
    print("F(R):", model.forces(geometry))


if __name__ == "__main__":
    main()
