"""Train and save the H3PES model on cached H3+ AM1/MECI data.

Run from the repository root with

    PYTHONPATH=. python examples/train_h3pes_am1_meci.py
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message="AM1 model is under testing")

from pyqed.ml import H3PES
from pyqed.qchem.semiempirical.am1 import HARTREE2EV


def train_test_split(n_samples: int, test_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_samples)
    n_test = max(1, int(round(n_samples * test_fraction)))
    return order[n_test:], order[:n_test]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("examples/h3plus_am1_meci_mpnn/h3plus_am1_meci_nr9_ntheta9_r1.25_2.25_theta45_75.npz"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("examples/h3plus_am1_meci_mpnn/h3pes_am1_meci_nr9_ntheta9.npz"),
    )
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=3.0e-3)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=9)
    args = parser.parse_args()

    data = np.load(args.data, allow_pickle=True)
    geometries = np.asarray(data["geometries"], dtype=float)
    energies = np.asarray(data["energies"], dtype=float)
    ok = np.all(np.isfinite(energies), axis=1)
    geometries = geometries[ok]
    energies = energies[ok]

    train_idx, test_idx = train_test_split(geometries.shape[0], args.test_fraction, args.seed)
    model = H3PES(
        hidden_layers=(64, 64),
        learning_rate=args.learning_rate,
        batch_size=32,
        max_iter=args.max_iter,
        validation_fraction=0.1,
        random_state=args.seed,
        verbose=True,
    ).fit(geometries[train_idx], energies[train_idx])

    pred_train = model.energy(geometries[train_idx])
    pred_test = model.energy(geometries[test_idx])
    train_mae = np.mean(np.abs(pred_train - energies[train_idx]), axis=0)
    test_mae = np.mean(np.abs(pred_test - energies[test_idx]), axis=0)

    args.model.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model)
    loaded = H3PES.load(args.model)
    roundtrip_error = np.max(np.abs(loaded.energy(geometries[test_idx[:3]]) - model.energy(geometries[test_idx[:3]])))

    print("[data]", args.data)
    print("[model]", args.model)
    print("[fit]", model.result_)
    print("[train MAE] mEh =", np.array2string(1000.0 * train_mae, precision=4))
    print("[test  MAE] mEh =", np.array2string(1000.0 * test_mae, precision=4))
    print("[test  MAE] eV  =", np.array2string(HARTREE2EV * test_mae, precision=4))
    print("[roundtrip max error]", roundtrip_error)


if __name__ == "__main__":
    main()
