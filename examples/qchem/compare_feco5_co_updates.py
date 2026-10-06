#!/usr/bin/env python3
"""Compare CO update trajectories with identical Fe(CO)5 physics and RHF start."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


def compare(folders, labels, output, *, limit=None, name="feco5_cocas88_co_updates"):
    if len(folders) != len(labels):
        raise ValueError("Provide one label for each calculation directory")
    records = [json.loads((folder / "result.json").read_text()) for folder in folders]
    keys = ("basis", "atom_angstrom", "ncas", "nelecas", "cd_tol", "macro_tol",
            "orbital_gradient_tol", "ci_tol")
    initial = records[0]["energy_history_hartree"][0]
    for folder, record in zip(folders, records):
        if record.get("continued_after_macro", 0):
            raise ValueError("This comparison requires fresh RHF-start trajectories")
        for key in keys:
            if record[key] != records[0][key]:
                raise ValueError(f"Calculation physics differs: {key}")
        np.testing.assert_allclose(record["energy_history_hartree"][0], initial,
                                   atol=1e-10, rtol=0)
        with np.load(folders[0] / "rhf_orbitals.npz") as first, np.load(folder / "rhf_orbitals.npz") as current:
            for key in ("mo_coeff", "mo_energy"):
                np.testing.assert_array_equal(first[key], current[key])
    output.mkdir(parents=True, exist_ok=True)
    summary = {"same_rhf_start_verified": True, "macro_limit": limit,
               "settings": {key: records[0][key] for key in keys},
               "timing_note": "Times cover full runs, including steps beyond a plot limit. Jobs overlap; no isolated speed comparison.",
               "variants": {}}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), constrained_layout=True)
    for label, folder, record in zip(labels, folders, records):
        rows = [row for row in record["macro_diagnostics"] if row.get("accepted") and "gn" in row]
        if limit is not None:
            rows = [row for row in rows if row["macro"] <= limit]
        energies = np.asarray(record["energy_history_hartree"][:len(rows) + 1])
        macro = [row["macro"] for row in rows]
        axes[0].plot(np.arange(len(energies)), energies - initial, "o-", ms=3, label=label)
        axes[1].semilogy(macro, [row["gn"] for row in rows], "o-", ms=3, label=label)
        axes[2].semilogy(np.arange(1, len(energies)), np.maximum(abs(np.diff(energies)), 1e-16),
                         "o-", ms=3, label=label)
        summary["variants"][label] = dict(
            output=str(folder), accepted_macros=len(rows), energy_hartree=float(energies[-1]),
            gradient_norm=rows[-1]["gn"], rejected_trials=sum(row.get("rej", 0) for row in rows),
            macro_converged=bool(record["macro_converged"] and
                                 (limit is None or record["macro_iterations"] <= limit)),
            solver_converged=record["solver_converged"],
            full_run_wall_time_s=record["wall_time_s"],
            full_run_macro_count=record["macro_iterations"],
        )
    axes[0].set(xlabel="Accepted CO macrostep", ylabel="Energy change from initial CASCI ($E_h$)")
    axes[1].set(xlabel="Accepted CO macrostep", ylabel="Physical orbital gradient norm ($E_h$)")
    axes[2].set(xlabel="Accepted CO macrostep", ylabel="Absolute energy change ($E_h$)")
    axes[1].axhline(records[0]["orbital_gradient_tol"], color="gray", ls="--")
    axes[2].axhline(records[0]["macro_tol"], color="gray", ls="--")
    for axis in axes:
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    budget = f" · first {limit} steps" if limit is not None else ""
    fig.suptitle(f"Fe(CO)₅ · def2-SVP · COCAS(8,8) · CD 10⁻⁸ · identical RHF frontier start{budget}")
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{name}.{suffix}", dpi=180)
    plt.close(fig)
    (output / f"{name}.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit-macro", type=int)
    parser.add_argument("--name", default="feco5_cocas88_co_updates")
    args = parser.parse_args()
    print(json.dumps(compare(args.runs, args.labels, args.output,
                             limit=args.limit_macro, name=args.name), indent=2))
