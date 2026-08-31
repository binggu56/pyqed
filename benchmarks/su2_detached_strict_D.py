"""Compare strict-D detached SU2-NARG with ordinary NARG at the same D."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

from pyqed.narg.qchem.su2_chain import diagonalize_block, run_su2_narg_chain
from pyqed.narg.qchem.su2_core import su2_irrep_tensor_roots


def hubbard_integrals(nsites=6, hopping=0.7, interaction=2.0):
    h1e = np.zeros((nsites, nsites))
    eri = np.zeros((nsites, nsites, nsites, nsites))
    for site in range(nsites - 1):
        h1e[site, site + 1] = h1e[site + 1, site] = -float(hopping)
    for site in range(nsites):
        eri[site, site, site, site] = float(interaction)
    return h1e, eri


def solve(h1e, eri, bond, *, detached=False):
    nsites = h1e.shape[0]
    options = {}
    if detached:
        options.update(dressing="detached_frames", chi=16 * int(bond))
    start = time.perf_counter()
    chain = run_su2_narg_chain(
        h1e,
        eri,
        {size: int(bond) for size in range(2, nsites)},
        final_size=nsites,
        target_nelec=nsites,
        target_j2=0,
        backend="python",
        project_v1_packages=False,
        **options,
    )
    elapsed = time.perf_counter() - start
    energy = float(
        diagonalize_block(
            chain.final,
            nelec=nsites,
            j2=0,
            nroots=1,
            backend="python",
        )[0][0]
    )
    return energy, elapsed, chain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--D", type=int, nargs="+", default=(2, 4, 8))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/private/tmp/su2_detached_strict_D"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    h1e, eri = hubbard_integrals()
    exact = float(
        su2_irrep_tensor_roots(h1e, eri, nelec=6, j2=0, nroots=1, m2=None)[0][0]
    )
    records = []
    for D in args.D:
        detached_energy, detached_seconds, detached = solve(
            h1e, eri, D, detached=True
        )
        standard_energy, standard_seconds, _standard = solve(h1e, eri, D)
        standard_16D_energy, standard_16D_seconds, _standard_16D = solve(
            h1e, eri, 16 * D
        )
        diagnostics = detached.timings["detached_by_size"].values()
        records.append(
            {
                "D": int(D),
                "detached_energy": detached_energy,
                "detached_error": detached_energy - exact,
                "detached_seconds": detached_seconds,
                "standard_D_energy": standard_energy,
                "standard_D_error": standard_energy - exact,
                "standard_D_seconds": standard_seconds,
                "standard_16D_energy": standard_16D_energy,
                "standard_16D_error": standard_16D_energy - exact,
                "standard_16D_seconds": standard_16D_seconds,
                "maximum_eigensolve_order": max(
                    item["maximum_eigensolve_order"] for item in diagnostics
                ),
                "maximum_ambient_dimension": max(
                    item["maximum_ambient_dimension"] for item in diagnostics
                ),
            }
        )

    payload = {"exact_energy": exact, "records": records}
    (args.output_dir / "strict_D.json").write_text(json.dumps(payload, indent=2) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(8.8, 3.7), constrained_layout=True)
    D_values = [record["D"] for record in records]
    for prefix, label in (
        ("detached", "strict-D detached"),
        ("standard_D", "standard D"),
        ("standard_16D", "standard 16D reference"),
    ):
        axes[0].plot(
            D_values,
            [record[f"{prefix}_error"] for record in records],
            marker="o",
            label=label,
        )
        axes[1].plot(
            D_values,
            [record[f"{prefix}_seconds"] for record in records],
            marker="o",
            label=label,
        )
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.legend(frameon=False)
    axes[0].set(xlabel="conditional dimension D", ylabel="energy error (Hartree)")
    axes[1].set(xlabel="conditional dimension D", ylabel="wall time (s)")
    figure.savefig(args.output_dir / "strict_D.png", dpi=180)
    plt.close(figure)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
