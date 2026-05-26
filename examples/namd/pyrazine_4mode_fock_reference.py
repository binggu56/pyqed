#!/usr/bin/env python3
"""Sparse Fock-basis reference dynamics for the four-mode pyrazine model.

The nuclear basis is a direct product of harmonic-oscillator number states for
the four dimensionless normal coordinates ``nu1, nu6a, nu9a, nu10a``.  The
Hamiltonian matches the four-mode LVC model used by
``pyrazine_4mode_ldrfg.py``:

    H = H_ho I_el + E_i + kappa_i . Q + gamma Q_10a^2 + lambda Q_10a.

The script propagates a Franck--Condon-like initial state
``|0,0,0,0> x |S2>`` and stores the diabatic populations, coordinate moments,
and autocorrelation.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed.units import au2ev, au2fs, wavenum2au


MODE_LABELS = ("nu1", "nu6a", "nu9a", "nu10a")
MODE_FREQ_CM = np.array([1015.0, 596.0, 1230.0, 919.0])
MODE_FREQ_AU = MODE_FREQ_CM * wavenum2au
ESHIFT = np.array([3.94, 4.89]) / au2ev
KAPPA_1 = np.array([-0.0470, -0.0964, 0.1594, 0.0]) / au2ev
KAPPA_2 = np.array([-0.2012, 0.1193, 0.0484, 0.0]) / au2ev
GAMMA_10A = -0.018 / au2ev
LAMBDA_10A = 0.1825 / au2ev
NSTATES = 3


def q_operator(nbasis: int) -> sp.csr_matrix:
    """Dimensionless harmonic-oscillator coordinate operator."""

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for n in range(nbasis - 1):
        value = np.sqrt((n + 1) / 2.0)
        rows.extend([n, n + 1])
        cols.extend([n + 1, n])
        data.extend([value, value])
    return sp.csr_matrix((data, (rows, cols)), shape=(nbasis, nbasis))


def harmonic_operator(nbasis: int, omega: float) -> sp.csr_matrix:
    return sp.diags(omega * (np.arange(nbasis, dtype=float) + 0.5), format="csr")


def kron_all(operators: list[sp.spmatrix]) -> sp.csr_matrix:
    out = sp.csr_matrix(operators[0])
    for operator in operators[1:]:
        out = sp.kron(out, operator, format="csr")
    return out


def mode_operator(local: sp.spmatrix, mode: int, dims: tuple[int, int, int, int]) -> sp.csr_matrix:
    factors: list[sp.spmatrix] = []
    for axis, dim in enumerate(dims):
        factors.append(local if axis == mode else sp.eye(dim, format="csr"))
    return kron_all(factors)


def parse_mode_counts(value: str) -> tuple[int, int, int, int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if len(items) == 1:
        n = int(items[0])
        if n < 1:
            raise argparse.ArgumentTypeError("basis size must be positive")
        return (n, n, n, n)
    if len(items) != 4:
        raise argparse.ArgumentTypeError("basis counts must be one integer or four comma-separated integers")
    counts = tuple(int(item) for item in items)
    if any(n < 1 for n in counts):
        raise argparse.ArgumentTypeError("basis sizes must be positive")
    return counts


def build_fock_hamiltonian(dims: tuple[int, int, int, int]) -> tuple[sp.csr_matrix, list[sp.csr_matrix]]:
    """Return sparse diabatic Hamiltonian and coordinate operators."""

    nuclear_dim = int(np.prod(dims))
    identity_nuc = sp.eye(nuclear_dim, format="csr")
    identity_el = sp.eye(NSTATES, format="csr")

    q_ops: list[sp.csr_matrix] = []
    harmonic = sp.csr_matrix((nuclear_dim, nuclear_dim), dtype=float)
    for mode, (nbasis, omega) in enumerate(zip(dims, MODE_FREQ_AU)):
        h_local = harmonic_operator(nbasis, float(omega))
        q_local = q_operator(nbasis)
        harmonic += mode_operator(h_local, mode, dims)
        q_ops.append(mode_operator(q_local, mode, dims))

    h = sp.kron(harmonic, identity_el, format="csr")

    electronic_const = sp.diags([0.0, ESHIFT[0], ESHIFT[1]], format="csr")
    h += sp.kron(identity_nuc, electronic_const, format="csr")

    for mode, q_op in enumerate(q_ops):
        linear = sp.diags([0.0, KAPPA_1[mode], KAPPA_2[mode]], format="csr")
        h += sp.kron(q_op, linear, format="csr")

    q10 = q_ops[3]
    q10_sq = q10 @ q10
    gamma_el = sp.diags([0.0, GAMMA_10A, GAMMA_10A], format="csr")
    h += sp.kron(q10_sq, gamma_el, format="csr")

    coupling_el = sp.csr_matrix(
        (
            [LAMBDA_10A, LAMBDA_10A],
            ([1, 2], [2, 1]),
        ),
        shape=(NSTATES, NSTATES),
    )
    h += sp.kron(q10, coupling_el, format="csr")
    return h.tocsr(), q_ops


def initial_state(dims: tuple[int, int, int, int], state: int) -> np.ndarray:
    nuclear_dim = int(np.prod(dims))
    psi = np.zeros((nuclear_dim, NSTATES), dtype=complex)
    psi[0, state] = 1.0
    return psi.reshape(-1)


def populations(states: np.ndarray, nuclear_dim: int) -> np.ndarray:
    return np.sum(np.abs(states.reshape(states.shape[0], nuclear_dim, NSTATES)) ** 2, axis=1).real


def coordinate_moments(states: np.ndarray, q_ops: list[sp.csr_matrix]) -> tuple[np.ndarray, np.ndarray]:
    ntime = states.shape[0]
    nuclear_dim = q_ops[0].shape[0]
    q_mean = np.empty((ntime, 4), dtype=float)
    q2_mean = np.empty((ntime, 4), dtype=float)
    for mode, q_op in enumerate(q_ops):
        q_full = sp.kron(q_op, sp.eye(NSTATES, format="csr"), format="csr")
        q2_full = sp.kron(q_op @ q_op, sp.eye(NSTATES, format="csr"), format="csr")
        for t, psi in enumerate(states):
            q_mean[t, mode] = float(np.vdot(psi, q_full @ psi).real)
            q2_mean[t, mode] = float(np.vdot(psi, q2_full @ psi).real)
    return q_mean, q2_mean


def run_fock_reference(
    dims: tuple[int, int, int, int],
    tmax_fs: float,
    nsnapshots: int,
    initial_electronic_state: int,
) -> dict[str, np.ndarray]:
    hamiltonian, q_ops = build_fock_hamiltonian(dims)
    psi0 = initial_state(dims, initial_electronic_state)
    times_fs = np.linspace(0.0, tmax_fs, nsnapshots)
    states = expm_multiply(
        -1j * hamiltonian,
        psi0,
        start=0.0,
        stop=float(tmax_fs / au2fs),
        num=nsnapshots,
        traceA=-1j * hamiltonian.diagonal().sum(),
    )
    autocorrelation = np.einsum("i,ti->t", psi0.conj(), states)
    nuclear_dim = int(np.prod(dims))
    pops = populations(states, nuclear_dim)
    q_mean, q2_mean = coordinate_moments(states, q_ops)
    return {
        "times_fs": times_fs,
        "autocorrelation": autocorrelation,
        "populations_diabatic": pops,
        "q_mean": q_mean,
        "q2_mean": q2_mean,
        "q_variance": q2_mean - q_mean * q_mean,
        "basis_counts": np.asarray(dims, dtype=int),
        "hamiltonian_dim": np.asarray(hamiltonian.shape[0], dtype=int),
        "hamiltonian_nnz": np.asarray(hamiltonian.nnz, dtype=int),
        "initial_electronic_state": np.asarray(initial_electronic_state, dtype=int),
    }


def plot_result(result: dict[str, np.ndarray], outpath: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 8.2), sharex=True, constrained_layout=True)
    times = result["times_fs"]
    axes[0].plot(times, np.abs(result["autocorrelation"]), color="0.1", lw=2.2)
    axes[0].set_ylabel(r"$|C(t)|$")

    for state in range(NSTATES):
        axes[1].plot(times, result["populations_diabatic"][:, state], lw=2.0, label=f"S{state}")
    axes[1].set_ylabel("diabatic population")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].legend(frameon=False, ncol=3)

    for mode, label in enumerate(MODE_LABELS):
        axes[2].plot(times, result["q_variance"][:, mode], lw=1.9, label=label)
    axes[2].set_xlabel("time / fs")
    axes[2].set_ylabel(r"$\sigma_Q^2$")
    axes[2].legend(frameon=False, ncol=4, fontsize=8)

    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", type=parse_mode_counts, default=(9, 9, 9, 9))
    parser.add_argument("--tmax-fs", type=float, default=80.0)
    parser.add_argument("--nsnapshots", type=int, default=801)
    parser.add_argument("--initial-state", type=int, choices=range(NSTATES), default=2)
    parser.add_argument("--outdir", type=Path, default=Path("examples/namd/pyrazine_4mode_fock_reference"))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    result = run_fock_reference(
        dims=args.basis,
        tmax_fs=args.tmax_fs,
        nsnapshots=args.nsnapshots,
        initial_electronic_state=args.initial_state,
    )

    label = "x".join(str(n) for n in args.basis)
    data_path = args.outdir / f"pyrazine_4mode_fock_{label}_{args.tmax_fs:g}fs.npz"
    plot_path = args.outdir / f"pyrazine_4mode_fock_{label}_{args.tmax_fs:g}fs.png"
    np.savez_compressed(data_path, **result)
    plot_result(result, plot_path)

    print(f"[plot] {plot_path}")
    print(f"[data] {data_path}")
    print("[basis counts]", dict(zip(MODE_LABELS, result["basis_counts"])))
    print("[size] dim={} nnz={}".format(int(result["hamiltonian_dim"]), int(result["hamiltonian_nnz"])))
    print("[final diabatic populations]", np.array2string(result["populations_diabatic"][-1], precision=8))
    print("[final |C|]", float(abs(result["autocorrelation"][-1])))
    print("[final variances]", np.array2string(result["q_variance"][-1], precision=8))


if __name__ == "__main__":
    main()
