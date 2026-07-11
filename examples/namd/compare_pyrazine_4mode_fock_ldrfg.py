#!/usr/bin/env python3
"""Benchmark 4D pyrazine LDRFG observables against a Fock-basis reference."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODE_LABELS = ("nu1", "nu6a", "nu9a", "nu10a")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def normalize_ldrfg(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Accept either raw LDRFG output or an LDRFG-vs-exact comparison bundle."""

    if "ldrfg_times_fs" in data:
        return data
    required = ("times_fs", "autocorrelation", "populations", "q_mean", "q_variance")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"LDRFG data is missing required keys: {missing}")
    normalized = dict(data)
    normalized["ldrfg_times_fs"] = data["times_fs"]
    normalized["ldrfg_autocorrelation"] = data["autocorrelation"]
    normalized["ldrfg_populations"] = data["populations"]
    normalized["ldrfg_q_mean"] = data["q_mean"]
    normalized["ldrfg_q_variance"] = data["q_variance"]
    return normalized


def interp_complex(x: np.ndarray, y: np.ndarray, xnew: np.ndarray) -> np.ndarray:
    return np.interp(xnew, x, y.real) + 1j * np.interp(xnew, x, y.imag)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(a) - np.asarray(b)))))


def common_time_grid(fock: dict[str, np.ndarray], ldrfg: dict[str, np.ndarray]) -> np.ndarray:
    tf = np.asarray(fock["times_fs"], dtype=float)
    tl = np.asarray(ldrfg["ldrfg_times_fs"], dtype=float)
    tmax = min(float(tf[-1]), float(tl[-1]))
    return tl[tl <= tmax]


def compute_metrics(
    fock: dict[str, np.ndarray], ldrfg: dict[str, np.ndarray], t: np.ndarray
) -> dict[str, np.ndarray | float]:
    tf = np.asarray(fock["times_fs"], dtype=float)
    fock_autocorr = interp_complex(tf, fock["autocorrelation"], t)
    fock_pops = np.column_stack(
        [np.interp(t, tf, fock["populations_diabatic"][:, state]) for state in range(3)]
    )
    fock_q_mean = np.column_stack([np.interp(t, tf, fock["q_mean"][:, mode]) for mode in range(4)])
    fock_q_var = np.column_stack([np.interp(t, tf, fock["q_variance"][:, mode]) for mode in range(4)])

    ldrfg_autocorr = ldrfg["ldrfg_autocorrelation"][: len(t)]
    ldrfg_pops = ldrfg["ldrfg_populations"][: len(t)]
    ldrfg_q_mean = ldrfg["ldrfg_q_mean"][: len(t)]
    ldrfg_q_var = ldrfg["ldrfg_q_variance"][: len(t)]

    return {
        "times_fs": t,
        "ldrfg_representation": str(np.asarray(ldrfg["representation"]).item()),
        "ldrfg_overlap_method": str(np.asarray(ldrfg.get("overlap_method", "unknown")).item()),
        "fock_autocorrelation": fock_autocorr,
        "ldrfg_autocorrelation": ldrfg_autocorr,
        "fock_populations_diabatic": fock_pops,
        "ldrfg_populations": ldrfg_pops,
        "fock_q_mean": fock_q_mean,
        "ldrfg_q_mean": ldrfg_q_mean,
        "fock_q_variance": fock_q_var,
        "ldrfg_q_variance": ldrfg_q_var,
        "autocorr_abs_rmse": rmse(np.abs(fock_autocorr), np.abs(ldrfg_autocorr)),
        "population_rmse_s1_s2": rmse(fock_pops[:, 1:3], ldrfg_pops[:, 1:3]),
        "q_mean_rmse": rmse(fock_q_mean, ldrfg_q_mean),
        "q_variance_rmse": rmse(fock_q_var, ldrfg_q_var),
    }


def plot_benchmark(metrics: dict[str, np.ndarray | float], outpath: Path) -> None:
    t = metrics["times_fs"]
    ldrfg_representation = metrics["ldrfg_representation"]
    fig, axes = plt.subplots(4, 1, figsize=(8.2, 10.0), sharex=True, constrained_layout=True)

    axes[0].plot(t, np.abs(metrics["fock_autocorrelation"]), color="0.1", lw=2.2, label="Fock 18^4")
    axes[0].plot(t, np.abs(metrics["ldrfg_autocorrelation"]), color="C3", ls="--", lw=2.0, label="LDRFG")
    axes[0].set_ylabel(r"$|C(t)|$")
    axes[0].legend(frameon=False, ncol=2)

    for state in (1, 2):
        axes[1].plot(t, metrics["fock_populations_diabatic"][:, state], lw=2.0, label=f"Fock diabatic S{state}")
        axes[1].plot(
            t,
            metrics["ldrfg_populations"][:, state],
            lw=1.8,
            ls="--",
            label=f"LDRFG {ldrfg_representation} S{state}",
        )
    axes[1].set_ylabel("population")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].legend(frameon=False, ncol=2, fontsize=8)

    for mode, label in enumerate(MODE_LABELS):
        axes[2].plot(t, metrics["fock_q_mean"][:, mode], lw=1.9, label=f"Fock {label}")
        axes[2].plot(t, metrics["ldrfg_q_mean"][:, mode], lw=1.7, ls="--", label=f"LDRFG {label}")
    axes[2].set_ylabel(r"$\langle Q\rangle$")
    axes[2].legend(frameon=False, ncol=4, fontsize=7)

    for mode, label in enumerate(MODE_LABELS):
        axes[3].plot(t, metrics["fock_q_variance"][:, mode], lw=1.9, label=f"Fock {label}")
        axes[3].plot(t, metrics["ldrfg_q_variance"][:, mode], lw=1.7, ls="--", label=f"LDRFG {label}")
    axes[3].set_xlabel("time / fs")
    axes[3].set_ylabel(r"$\sigma_Q^2$")
    axes[3].legend(frameon=False, ncol=4, fontsize=7)

    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fock",
        type=Path,
        default=Path(
            "examples/namd/pyrazine_4mode_fock_reference/"
            "pyrazine_4mode_fock_18x18x18x18_160fs.npz"
        ),
    )
    parser.add_argument(
        "--ldrfg",
        type=Path,
        default=Path(
            "examples/namd/pyrazine_4mode_ldrfg_vs_exact_nu1n13_n9_80fs_matched_width/"
            "pyrazine_4mode_ldrfg_vs_exact_autocorr_moments.npz"
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("examples/namd/pyrazine_4mode_fock18_vs_ldrfg"),
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    fock = load_npz(args.fock)
    ldrfg = normalize_ldrfg(load_npz(args.ldrfg))
    t = common_time_grid(fock, ldrfg)
    metrics = compute_metrics(fock, ldrfg, t)

    plot_path = args.outdir / "pyrazine_4mode_fock18_vs_ldrfg_observables.png"
    data_path = args.outdir / "pyrazine_4mode_fock18_vs_ldrfg_observables.npz"
    plot_benchmark(metrics, plot_path)
    np.savez_compressed(
        data_path,
        **metrics,
        fock_basis_counts=fock["basis_counts"],
        fock_hamiltonian_dim=fock["hamiltonian_dim"],
        fock_hamiltonian_nnz=fock["hamiltonian_nnz"],
        ldrfg_npts_by_mode=ldrfg["npts_by_mode"],
        ldrfg_ldr_modes=ldrfg["ldr_modes"],
        ldrfg_fg_modes=ldrfg["fg_modes"],
    )

    print(f"[plot] {plot_path}")
    print(f"[data] {data_path}")
    print("[fock basis counts]", np.array2string(fock["basis_counts"]))
    print("[fock size] dim={} nnz={}".format(int(fock["hamiltonian_dim"]), int(fock["hamiltonian_nnz"])))
    print("[benchmark window fs] {:.6g} to {:.6g}".format(float(t[0]), float(t[-1])))
    print("[rmse |C|]", metrics["autocorr_abs_rmse"])
    print("[rmse populations S1/S2]", metrics["population_rmse_s1_s2"])
    print("[rmse q mean]", metrics["q_mean_rmse"])
    print("[rmse q variance]", metrics["q_variance_rmse"])
    print("[ldrfg representation]", metrics["ldrfg_representation"])
    print("[ldrfg overlap method]", metrics["ldrfg_overlap_method"])
    print("[fock final diabatic populations in window]", np.array2string(metrics["fock_populations_diabatic"][-1], precision=8))
    print("[ldrfg final populations in window]", np.array2string(metrics["ldrfg_populations"][-1], precision=8))
    print("[fock final |C| in window]", float(abs(metrics["fock_autocorrelation"][-1])))
    print("[ldrfg final |C| in window]", float(abs(metrics["ldrfg_autocorrelation"][-1])))


if __name__ == "__main__":
    main()
