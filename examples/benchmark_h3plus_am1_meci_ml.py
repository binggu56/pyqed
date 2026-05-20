"""Benchmark MLP, EquivariantMLP, and MPNN on H3+ AM1/MECI data.

Run from the repository root with

    PYTHONPATH=. MPLCONFIGDIR=/private/tmp/matplotlib-codex \
        python examples/benchmark_h3plus_am1_meci_ml.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.fit_mpnn_h3plus_am1_meci import load_or_generate, train_test_split
from pyqed.ml import EquivariantMLP, MLP, MPNN
from pyqed.qchem.semiempirical.am1 import HARTREE2EV


def pair_distances(geometries: np.ndarray) -> np.ndarray:
    r01 = np.linalg.norm(geometries[:, 0] - geometries[:, 1], axis=1)
    r02 = np.linalg.norm(geometries[:, 0] - geometries[:, 2], axis=1)
    r12 = np.linalg.norm(geometries[:, 1] - geometries[:, 2], axis=1)
    return np.sort(np.stack((r01, r02, r12), axis=1), axis=1)


def summarize(name: str, model, x_train, y_train, x_test, y_test):
    pred_train = model.energy(x_train)
    pred_test = model.energy(x_test)
    train_mae = np.mean(np.abs(pred_train - y_train), axis=0)
    test_mae = np.mean(np.abs(pred_test - y_test), axis=0)
    print(f"\n[{name}]")
    print("  fit:", model.result_)
    print("  train MAE / mEh:", np.array2string(1000.0 * train_mae, precision=4))
    print("  test  MAE / mEh:", np.array2string(1000.0 * test_mae, precision=4))
    print("  test  MAE / eV :", np.array2string(HARTREE2EV * test_mae, precision=4))
    return train_mae, test_mae


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-r", type=int, default=5)
    parser.add_argument("--n-theta", type=int, default=5)
    parser.add_argument("--r-min", type=float, default=1.25)
    parser.add_argument("--r-max", type=float, default=2.25)
    parser.add_argument("--theta-min-deg", type=float, default=45.0)
    parser.add_argument("--theta-max-deg", type=float, default=75.0)
    parser.add_argument("--nstates", type=int, default=3)
    parser.add_argument("--ncas", type=int, default=3)
    parser.add_argument("--scf-tol", type=float, default=1.0e-8)
    parser.add_argument("--max-cycle", type=int, default=100)
    parser.add_argument("--damping", type=float, default=0.0)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-iter", type=int, default=150)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).with_name("h3plus_am1_meci_mpnn"),
    )
    args = parser.parse_args()

    data_path, data = load_or_generate(args)
    geometries = np.asarray(data["geometries"], dtype=float)
    internals = np.asarray(data["internals"], dtype=float)
    energies = np.asarray(data["energies"], dtype=float)
    ok = np.all(np.isfinite(energies), axis=1)
    geometries = geometries[ok]
    internals = internals[ok]
    energies = energies[ok]
    distances = pair_distances(geometries)

    train_idx, test_idx = train_test_split(geometries.shape[0], args.test_fraction, args.seed)
    print("[dataset]", data_path)
    print("[samples]", geometries.shape[0], "train", train_idx.size, "test", test_idx.size)

    mlp = MLP(
        hidden_layers=(64, 64),
        activation="tanh",
        learning_rate=0.003,
        batch_size=32,
        max_iter=args.max_iter,
        validation_fraction=0.1,
        random_state=args.seed,
        verbose=False,
    ).fit(distances[train_idx], energies[train_idx])
    summarize(
        "MLP sorted distances",
        mlp,
        distances[train_idx],
        energies[train_idx],
        distances[test_idx],
        energies[test_idx],
    )

    equivariant_mlp = EquivariantMLP(
        species=("H", "H", "H"),
        n_radial=8,
        angle_order=2,
        hidden_layers=(64, 64),
        learning_rate=0.003,
        batch_size=32,
        max_iter=args.max_iter,
        validation_fraction=0.1,
        random_state=args.seed,
        verbose=False,
    ).fit(geometries[train_idx], energies[train_idx])
    summarize(
        "EquivariantMLP",
        equivariant_mlp,
        geometries[train_idx],
        energies[train_idx],
        geometries[test_idx],
        energies[test_idx],
    )

    mpnn = MPNN(
        species=("H", "H", "H"),
        features=16,
        n_layers=2,
        n_radial=6,
        readout_hidden=24,
        learning_rate=0.002,
        batch_size=32,
        max_iter=args.max_iter,
        validation_fraction=0.1,
        random_state=args.seed,
        verbose=False,
    ).fit(geometries[train_idx], energies[train_idx])
    summarize(
        "MPNN",
        mpnn,
        geometries[train_idx],
        energies[train_idx],
        geometries[test_idx],
        energies[test_idx],
    )

    probe = geometries[test_idx[0]]
    theta = 0.4
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    transformed = probe @ rotation.T + np.array([0.2, -0.3, 0.5])
    permuted = probe[[2, 1, 0]]
    print("\n[MPNN symmetry]")
    print("  |E(rot+shift)-E|:", np.max(np.abs(mpnn.energy(transformed) - mpnn.energy(probe))))
    print("  |E(perm)-E|     :", np.max(np.abs(mpnn.energy(permuted) - mpnn.energy(probe))))


if __name__ == "__main__":
    main()
