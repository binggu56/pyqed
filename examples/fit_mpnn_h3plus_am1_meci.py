"""Generate H3+ AM1/MECI data and train the MPNN PES model.

Run from the repository root with

    PYTHONPATH=. MPLCONFIGDIR=/private/tmp/matplotlib-codex \
        python examples/fit_mpnn_h3plus_am1_meci.py
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore", message="AM1 model is under testing")

from pyqed.ml import MPNN
from pyqed.qchem import Molecule
from pyqed.qchem.semiempirical.am1 import HARTREE2EV, RAM1


def h3plus_geometry(r1: float, r2: float, theta: float) -> np.ndarray:
    """Return body-frame H3+ Cartesian coordinates in bohr."""

    return np.array(
        [
            [r1, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [r2 * np.cos(theta), r2 * np.sin(theta), 0.0],
        ],
        dtype=float,
    )


def atom_string(geometry: np.ndarray) -> str:
    return "; ".join(f"H {x:.12f} {y:.12f} {z:.12f}" for x, y, z in geometry)


def run_am1_meci(geometry: np.ndarray, args) -> tuple[np.ndarray, float, int]:
    mol = Molecule(atom=atom_string(geometry), charge=1, spin=0, unit="bohr")
    mf = RAM1(mol).run(
        conv_tol=args.scf_tol,
        max_cycle=args.max_cycle,
        verbose=0,
        damping=args.damping,
    )
    meci = mf.MECI(nstates=args.nstates, ncas=args.ncas).run()
    if len(meci.e) < args.nstates:
        raise RuntimeError(f"MECI returned only {len(meci.e)} states.")
    return np.asarray(meci.e[: args.nstates], dtype=float), float(mf.e_tot), int(mf.cycles)


def generate_dataset(args):
    r_grid = np.linspace(args.r_min, args.r_max, args.n_r)
    theta_grid = np.deg2rad(np.linspace(args.theta_min_deg, args.theta_max_deg, args.n_theta))

    geometries = []
    internals = []
    energies = []
    scf_energies = []
    scf_cycles = []

    total = args.n_r * args.n_r * args.n_theta
    count = 0
    t0 = time.time()
    for r1 in r_grid:
        for r2 in r_grid:
            for theta in theta_grid:
                count += 1
                geometry = h3plus_geometry(float(r1), float(r2), float(theta))
                e, escf, cycles = run_am1_meci(geometry, args)
                geometries.append(geometry)
                internals.append([r1, r2, theta])
                energies.append(e)
                scf_energies.append(escf)
                scf_cycles.append(cycles)
                if count == 1 or count % args.progress_every == 0 or count == total:
                    elapsed = time.time() - t0
                    print(
                        f"[{count:4d}/{total}] r1={r1:.4f} r2={r2:.4f} "
                        f"theta={np.rad2deg(theta):.2f} deg "
                        f"E0={e[0]: .10f} Eh elapsed={elapsed:.1f}s"
                    )

    return {
        "geometries": np.asarray(geometries),
        "internals": np.asarray(internals),
        "energies": np.asarray(energies),
        "scf_energies": np.asarray(scf_energies),
        "scf_cycles": np.asarray(scf_cycles),
        "meta": {
            "n_r": args.n_r,
            "n_theta": args.n_theta,
            "r_min": args.r_min,
            "r_max": args.r_max,
            "theta_min_deg": args.theta_min_deg,
            "theta_max_deg": args.theta_max_deg,
            "nstates": args.nstates,
            "ncas": args.ncas,
            "method": "AM1/MECI",
        },
    }


def load_or_generate(args):
    args.outdir.mkdir(parents=True, exist_ok=True)
    data_path = args.outdir / (
        f"h3plus_am1_meci_nr{args.n_r}_ntheta{args.n_theta}_"
        f"r{args.r_min:g}_{args.r_max:g}_"
        f"theta{args.theta_min_deg:g}_{args.theta_max_deg:g}.npz"
    )
    if data_path.exists() and not args.force:
        data = np.load(data_path, allow_pickle=True)
        print(f"[cache] loaded {data_path}")
        return data_path, {key: data[key] for key in data.files}

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="AM1 model is under testing")
        data = generate_dataset(args)
    np.savez(data_path, **data)
    print(f"[data] wrote {data_path}")
    return data_path, data


def train_test_split(n_samples: int, test_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_samples)
    n_test = max(1, int(round(n_samples * test_fraction)))
    return order[n_test:], order[:n_test]


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
    parser.add_argument("--features", type=int, default=16)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--n-radial", type=int, default=6)
    parser.add_argument("--readout-hidden", type=int, default=24)
    parser.add_argument("--max-iter", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).with_name("h3plus_am1_meci_mpnn"),
    )
    args = parser.parse_args()

    data_path, data = load_or_generate(args)
    geometries = np.asarray(data["geometries"], dtype=float)
    energies = np.asarray(data["energies"], dtype=float)
    ok = np.all(np.isfinite(energies), axis=1)
    geometries = geometries[ok]
    energies = energies[ok]

    train_idx, test_idx = train_test_split(geometries.shape[0], args.test_fraction, args.seed)
    model = MPNN(
        species=("H", "H", "H"),
        features=args.features,
        n_layers=args.layers,
        n_radial=args.n_radial,
        readout_hidden=args.readout_hidden,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_iter=args.max_iter,
        validation_fraction=0.1,
        random_state=args.seed,
        verbose=True,
    ).fit(geometries[train_idx], energies[train_idx])

    pred_train = model.energy(geometries[train_idx])
    pred_test = model.energy(geometries[test_idx])
    train_mae = np.mean(np.abs(pred_train - energies[train_idx]), axis=0)
    test_mae = np.mean(np.abs(pred_test - energies[test_idx]), axis=0)

    print("[dataset]", data_path)
    print("[fit]", model.result_)
    print("[train MAE] mEh =", np.array2string(1000.0 * train_mae, precision=4))
    print("[test  MAE] mEh =", np.array2string(1000.0 * test_mae, precision=4))
    print("[test  MAE] eV  =", np.array2string(HARTREE2EV * test_mae, precision=4))

    theta = 0.4
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    probe = geometries[test_idx[0]]
    transformed = probe @ rotation.T + np.array([0.2, -0.3, 0.5])
    permuted = probe[[2, 1, 0]]
    print("[sym] |E(rot+shift)-E| =", np.max(np.abs(model.energy(transformed) - model.energy(probe))))
    print("[sym] |E(perm)-E|      =", np.max(np.abs(model.energy(permuted) - model.energy(probe))))
    print("[force] shape =", model.forces(probe).shape)


if __name__ == "__main__":
    main()
