"""Fit a small ANN to a potential energy surface.

Run from the repository root with

    PYTHONPATH=. python examples/fit_ann_pes.py
"""

import numpy as np

from pyqed.ml import MLP, grid_to_samples


def main():
    # Example 2D single-state PES on a regular grid.
    x_axis = np.linspace(-1.5, 1.5, 31)
    y_axis = np.linspace(-1.0, 1.0, 25)
    x, y = np.meshgrid(x_axis, y_axis, indexing="ij")
    pes = 0.5 * x**2 + 0.25 * y**2 + 0.05 * np.sin(3.0 * x) * np.cos(2.0 * y)

    coordinates, energies = grid_to_samples((x_axis, y_axis), pes)
    model = MLP(
        hidden_layers=(64, 64),
        activation="tanh",
        learning_rate=0.005,
        batch_size=128,
        max_iter=2000,
        validation_fraction=0.1,
        random_state=2,
        verbose=True,
    ).fit(coordinates, energies)

    geometry = np.array([0.2, -0.3])
    print("fit result:", model.result_)
    print("E(q):", model.energy(geometry))
    print("dE/dq:", model.gradient(geometry))

    model.save("ann_pes_model.npz")


if __name__ == "__main__":
    main()
