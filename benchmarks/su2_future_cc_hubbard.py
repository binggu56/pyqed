"""Benchmark reduced SU(2) future-CC dressing on a Hubbard chain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

from pyqed.narg.qchem.su2_chain import diagonalize_block, run_su2_narg_chain
from pyqed.narg.qchem.su2_core import su2_irrep_tensor_roots


def hubbard_integrals(nsites, *, hopping=0.7, interaction=2.0):
    h1e = np.zeros((nsites, nsites))
    for site in range(nsites - 1):
        h1e[site, site + 1] = h1e[site + 1, site] = -float(hopping)
    eri = np.zeros((nsites, nsites, nsites, nsites))
    for site in range(nsites):
        eri[site, site, site, site] = float(interaction)
    return h1e, eri


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("/private/tmp/su2_future_cc_hubbard"))
    parser.add_argument("--bond-dims", type=int, nargs="+", default=(3, 4, 6))
    parser.add_argument(
        "--strengths",
        type=float,
        nargs="+",
        default=(0.0, 0.05, 0.1, 0.2, 0.4),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    nsites = 5
    target_j2 = 1
    h1e, eri = hubbard_integrals(nsites)
    exact = float(
        su2_irrep_tensor_roots(
            h1e,
            eri,
            nsites,
            target_j2,
            nroots=1,
            m2=None,
        )[0][0]
    )
    records = []
    for bond_dim in args.bond_dims:
        for strength in args.strengths:
            options = {}
            if strength:
                options = {
                    "dressing": "future_cc",
                    "future_cc_strength": strength,
                }
            start = time.perf_counter()
            chain = run_su2_narg_chain(
                h1e,
                eri,
                {size: bond_dim for size in range(2, nsites)},
                final_size=nsites,
                target_nelec=nsites,
                target_j2=target_j2,
                backend="python",
                **options,
            )
            energy = float(
                diagonalize_block(
                    chain.final,
                    nelec=nsites,
                    j2=target_j2,
                    nroots=1,
                    backend="python",
                )[0][0]
            )
            diagnostics = chain.timings["future_cc_by_size"]
            records.append(
                {
                    "D": bond_dim,
                    "strength": strength,
                    "energy": energy,
                    "error": energy - exact,
                    "seconds": time.perf_counter() - start,
                    "maximum_response_mixing": max(
                        (item["response_mixing"] for item in diagnostics.values()),
                        default=0.0,
                    ),
                    "maximum_response_residual": max(
                        (item["maximum_response_residual"] for item in diagnostics.values()),
                        default=0.0,
                    ),
                }
            )

    payload = {
        "model": "open five-site Hubbard chain, t=0.7, U=2.0",
        "exact_energy": exact,
        "records": records,
    }
    (args.output_dir / "su2_future_cc_hubbard.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )

    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), constrained_layout=True)
    for bond_dim in args.bond_dims:
        selected = [item for item in records if item["D"] == bond_dim]
        axes[0].plot(
            [item["strength"] for item in selected],
            [item["error"] for item in selected],
            marker="o",
            label=f"D={bond_dim}",
        )
        axes[1].plot(
            [item["strength"] for item in selected],
            [item["maximum_response_mixing"] for item in selected],
            marker="o",
            label=f"D={bond_dim}",
        )
    axes[0].set(xlabel="future-CC damping", ylabel="energy error (Hartree)")
    axes[1].set(xlabel="future-CC damping", ylabel="maximum discarded mixing")
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    figure.savefig(args.output_dir / "su2_future_cc_hubbard.png", dpi=180)
    plt.close(figure)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
