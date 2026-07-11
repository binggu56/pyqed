#!/usr/bin/env python3
"""Compare 4D pyrazine Fock-basis and DVR reference observables."""

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


def plot_comparison(fock: dict[str, np.ndarray], dvr: dict[str, np.ndarray], outpath: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(8.2, 10.0), sharex=True, constrained_layout=True)
    tf = fock["times_fs"]
    td = dvr["exact_times_fs"]

    axes[0].plot(td, np.abs(dvr["exact_autocorrelation"]), color="0.1", lw=2.2, label="DVR exact")
    axes[0].plot(tf, np.abs(fock["autocorrelation"]), color="C3", ls="--", lw=2.0, label="Fock")
    axes[0].set_ylabel(r"$|C(t)|$")
    axes[0].legend(frameon=False, ncol=2)

    dvr_pop = dvr["exact_populations"]
    fock_pop = fock["populations_diabatic"]
    for state in (1, 2):
        axes[1].plot(td, dvr_pop[:, state], lw=2.0, label=f"DVR adiabatic S{state}")
        axes[1].plot(tf, fock_pop[:, state], lw=1.8, ls="--", label=f"Fock diabatic S{state}")
    axes[1].set_ylabel("population")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].legend(frameon=False, ncol=2, fontsize=8)

    for mode, label in enumerate(MODE_LABELS):
        axes[2].plot(td, dvr["exact_q_mean"][:, mode], lw=1.9, label=f"DVR {label}")
        axes[2].plot(tf, fock["q_mean"][:, mode], lw=1.7, ls="--", label=f"Fock {label}")
    axes[2].set_ylabel(r"$\langle Q\rangle$")
    axes[2].legend(frameon=False, ncol=4, fontsize=7)

    for mode, label in enumerate(MODE_LABELS):
        axes[3].plot(td, dvr["exact_q_variance"][:, mode], lw=1.9, label=f"DVR {label}")
        axes[3].plot(tf, fock["q_variance"][:, mode], lw=1.7, ls="--", label=f"Fock {label}")
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
            "pyrazine_4mode_fock_13x9x9x9_80fs.npz"
        ),
    )
    parser.add_argument(
        "--dvr",
        type=Path,
        default=Path(
            "examples/namd/pyrazine_4mode_ldrfg_vs_exact_nu1n13_n9_80fs_matched_width/"
            "pyrazine_4mode_ldrfg_vs_exact_autocorr_moments.npz"
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("examples/namd/pyrazine_4mode_fock_vs_dvr"),
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    fock = load_npz(args.fock)
    dvr = load_npz(args.dvr)

    plot_path = args.outdir / "pyrazine_4mode_fock_vs_dvr_observables.png"
    data_path = args.outdir / "pyrazine_4mode_fock_vs_dvr_observables.npz"
    plot_comparison(fock, dvr, plot_path)
    np.savez_compressed(
        data_path,
        fock_times_fs=fock["times_fs"],
        dvr_times_fs=dvr["exact_times_fs"],
        fock_autocorrelation=fock["autocorrelation"],
        dvr_autocorrelation=dvr["exact_autocorrelation"],
        fock_populations_diabatic=fock["populations_diabatic"],
        dvr_populations_adiabatic=dvr["exact_populations"],
        fock_q_mean=fock["q_mean"],
        dvr_q_mean=dvr["exact_q_mean"],
        fock_q_variance=fock["q_variance"],
        dvr_q_variance=dvr["exact_q_variance"],
        fock_basis_counts=fock["basis_counts"],
        dvr_npts_by_mode=dvr["npts_by_mode"],
        fock_hamiltonian_dim=fock["hamiltonian_dim"],
        dvr_hamiltonian_dim=dvr["exact_hamiltonian_dim"],
        fock_hamiltonian_nnz=fock["hamiltonian_nnz"],
        dvr_hamiltonian_nnz=dvr["exact_hamiltonian_nnz"],
    )

    print(f"[plot] {plot_path}")
    print(f"[data] {data_path}")
    print("[fock size] dim={} nnz={}".format(int(fock["hamiltonian_dim"]), int(fock["hamiltonian_nnz"])))
    print("[dvr size] dim={} nnz={}".format(int(dvr["exact_hamiltonian_dim"]), int(dvr["exact_hamiltonian_nnz"])))
    print("[fock final diabatic populations]", np.array2string(fock["populations_diabatic"][-1], precision=8))
    print("[dvr final adiabatic populations]", np.array2string(dvr["exact_populations"][-1], precision=8))
    print("[fock final |C|]", float(abs(fock["autocorrelation"][-1])))
    print("[dvr final |C|]", float(abs(dvr["exact_autocorrelation"][-1])))


if __name__ == "__main__":
    main()
