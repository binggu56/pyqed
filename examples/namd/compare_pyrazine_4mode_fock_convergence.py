#!/usr/bin/env python3
"""Plot Fock-basis convergence for 4D pyrazine reference runs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths",
        nargs="+",
        type=Path,
        default=[
            Path("examples/namd/pyrazine_4mode_fock_reference/pyrazine_4mode_fock_12x12x12x12_160fs.npz"),
            Path("examples/namd/pyrazine_4mode_fock_reference/pyrazine_4mode_fock_14x14x14x14_160fs.npz"),
            Path("examples/namd/pyrazine_4mode_fock_reference/pyrazine_4mode_fock_16x16x16x16_160fs.npz"),
        ],
    )
    parser.add_argument("--outdir", type=Path, default=Path("examples/namd/pyrazine_4mode_fock_convergence"))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    runs = [(path.stem.removeprefix("pyrazine_4mode_fock_").removesuffix("_160fs"), load(path)) for path in args.paths]

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 8.2), sharex=True, constrained_layout=True)
    for label, run in runs:
        t = run["times_fs"]
        axes[0].plot(t, np.abs(run["autocorrelation"]), lw=2.0, label=label)
        axes[1].plot(t, run["populations_diabatic"][:, 1], lw=2.0, label=label)
        axes[2].plot(t, run["q_variance"][:, 3], lw=2.0, label=label)

    axes[0].set_ylabel(r"$|C(t)|$")
    axes[1].set_ylabel("S1 population")
    axes[2].set_ylabel(r"$\sigma^2_{10a}$")
    axes[2].set_xlabel("time / fs")
    for ax in axes:
        ax.legend(frameon=False, ncol=3, fontsize=8)

    suffix = "_".join(label.replace("x", "-") for label, _ in runs)
    plot_path = args.outdir / f"pyrazine_4mode_fock_convergence_{suffix}.png"
    data_path = args.outdir / f"pyrazine_4mode_fock_convergence_{suffix}.npz"
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)

    labels = np.asarray([label for label, _ in runs])
    final_pop = np.asarray([run["populations_diabatic"][-1] for _, run in runs])
    final_abs_c = np.asarray([abs(run["autocorrelation"][-1]) for _, run in runs])
    final_var = np.asarray([run["q_variance"][-1] for _, run in runs])
    dims = np.asarray([run["hamiltonian_dim"] for _, run in runs])
    nnz = np.asarray([run["hamiltonian_nnz"] for _, run in runs])
    np.savez_compressed(
        data_path,
        labels=labels,
        final_populations=final_pop,
        final_abs_autocorrelation=final_abs_c,
        final_variances=final_var,
        hamiltonian_dim=dims,
        hamiltonian_nnz=nnz,
    )

    print(f"[plot] {plot_path}")
    print(f"[data] {data_path}")
    for label, run in runs:
        print(
            "[{}] dim={} nnz={} final_pop={} final_|C|={:.8f} final_var={}".format(
                label,
                int(run["hamiltonian_dim"]),
                int(run["hamiltonian_nnz"]),
                np.array2string(run["populations_diabatic"][-1], precision=8),
                float(abs(run["autocorrelation"][-1])),
                np.array2string(run["q_variance"][-1], precision=8),
            )
        )


if __name__ == "__main__":
    main()
