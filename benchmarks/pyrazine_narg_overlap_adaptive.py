"""Scan discarded-weight-adaptive SU(2)-NARG overlap compression."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYQED_MPS_DISABLE_CPP_DAVIDSON", "1")

import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np

from benchmarks.pyrazine_narg_overlap_casci import _geometry, _plain
from benchmarks.pyrazine_narg_overlap_compression import _complex_matrix, _run_narg


def _plot(records, output):
    budgets = np.asarray([record["discarded_weight_budget"] for record in records])
    timings = np.asarray([record["seconds"] for record in records])
    peaks = np.asarray([record["peak_reduced_bond_dimension"] for record in records])
    compression = np.asarray(
        [record["max_compression_error"] for record in records],
        dtype=float,
    )
    total = np.asarray([record["max_casci_error"] for record in records])

    order = np.argsort(budgets)[::-1]
    budgets = budgets[order]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    axes[0].plot(budgets, timings[order], "o-", color="#176B87")
    axes[0].set_xscale("log")
    axes[0].invert_xaxis()
    axes[0].xaxis.set_major_locator(ticker.FixedLocator(budgets))
    axes[0].xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2g"))
    axes[0].xaxis.set_minor_locator(ticker.NullLocator())
    axes[0].set_xlabel(r"Discarded-weight budget $\epsilon_{\rm tot}$")
    axes[0].set_ylabel("Time (s)")
    peak_axis = axes[0].twinx()
    peak_axis.plot(budgets, peaks[order], "s--", color="#D1495B")
    peak_axis.set_ylabel("Peak reduced bond")

    finite_compression = np.isfinite(compression[order])
    if np.any(finite_compression):
        axes[1].plot(
            budgets[finite_compression],
            compression[order][finite_compression],
            "o-",
            color="#6A4C93",
            label="Compression",
        )
    axes[1].plot(
        budgets,
        total[order],
        "s-",
        color="#D1495B",
        label="vs CASCI",
    )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].invert_xaxis()
    axes[1].xaxis.set_major_locator(ticker.FixedLocator(budgets))
    axes[1].xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2g"))
    axes[1].xaxis.set_minor_locator(ticker.NullLocator())
    axes[1].set_xlabel(r"Discarded-weight budget $\epsilon_{\rm tot}$")
    axes[1].set_ylabel("Maximum overlap-magnitude error")
    axes[1].legend(frameon=False)
    for label, axis in zip("ab", axes):
        axis.text(-0.16, 1.04, label, transform=axis.transAxes, fontweight="bold")
    fig.savefig(output, dpi=320)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--basis", default="6-31g")
    parser.add_argument("--ncas", type=int, default=14)
    parser.add_argument("--nelecas", type=int, default=14)
    parser.add_argument("--nstates", type=int, default=4)
    parser.add_argument("--D", type=int, default=256)
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.3, 0.1, 0.03])
    parser.add_argument("--adaptive-max-bond", type=int, default=8192)
    parser.add_argument("--cutoff", type=float, default=1.0e-12)
    parser.add_argument("--coupling-displacement", type=float, default=0.02)
    parser.add_argument("--cd-tol", type=float, default=1.0e-10)
    parser.add_argument("--su2-backend", default="python")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/private/tmp/pyrazine_narg_overlap_adaptive"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reference = json.loads(args.reference.read_text())
    casci_overlap = np.asarray(reference["casci_overlap"], dtype=complex)
    same_reference_D = int(reference.get("D", args.D)) == args.D
    exact_narg_overlap = (
        _complex_matrix(reference["narg_overlap"])
        if same_reference_D
        else None
    )
    reference_energies = (
        np.asarray(reference["narg_energies_hartree"])
        if same_reference_D
        else None
    )

    print("geometry 1", flush=True)
    first, first_seconds = _run_narg(_geometry(1.0), args)
    print(f"NARG: {first_seconds:.3f} s", flush=True)
    print("geometry 2", flush=True)
    second, second_seconds = _run_narg(
        _geometry(1.0, args.coupling_displacement),
        args,
    )
    print(f"NARG: {second_seconds:.3f} s", flush=True)
    energies = np.asarray([first.e_tot, second.e_tot])

    records = []
    matrices = []
    for budget in args.budgets:
        started = time.perf_counter()
        overlap, info = first.overlap(
            second,
            orbital_split="auto",
            orbital_map_threshold=0.0,
            cutoff=args.cutoff,
            max_bond="adaptive",
            discarded_weight_budget=budget,
            adaptive_max_bond=args.adaptive_max_bond,
            return_info=True,
        )
        seconds = time.perf_counter() - started
        overlap = np.asarray(overlap)
        transform = info["transforms"]["ket"][0]
        record = {
            "discarded_weight_budget": float(budget),
            "seconds": seconds,
            "max_compression_error": (
                float(np.max(np.abs(np.abs(overlap) - np.abs(exact_narg_overlap))))
                if exact_narg_overlap is not None
                else None
            ),
            "rms_compression_error": (
                float(
                    np.sqrt(
                        np.mean(
                            (np.abs(overlap) - np.abs(exact_narg_overlap)) ** 2
                        )
                    )
                )
                if exact_narg_overlap is not None
                else None
            ),
            "max_casci_error": float(
                np.max(np.abs(np.abs(overlap) - np.abs(casci_overlap)))
            ),
            "peak_reduced_bond_dimension": int(
                transform["peak_reduced_bond_dimension"]
            ),
            "sum_discarded_weight": float(transform["sum_gate_discarded_weight"]),
            "max_discarded_weight": float(transform["max_gate_discarded_weight"]),
            "adaptive_budget_satisfied": bool(
                transform["adaptive_budget_satisfied"]
            ),
            "minimum_gate_bond": int(min(transform["gate_kept_reduced_bonds"])),
            "maximum_gate_bond": int(max(transform["gate_kept_reduced_bonds"])),
            "compiled_channel_mix_batches": int(
                transform["compiled_channel_mix_batches"]
            ),
        }
        records.append(record)
        matrices.append(overlap)
        print(json.dumps(record), flush=True)

    payload = {
        "system": f"pyrazine/{args.basis} CAS({args.nelecas},{args.ncas})",
        "D": args.D,
        "cutoff": args.cutoff,
        "adaptive_max_bond": args.adaptive_max_bond,
        "reference": str(args.reference),
        "reference_narg_D": reference.get("D"),
        "reference_narg_compatible": same_reference_D,
        "narg_state_seconds": [first_seconds, second_seconds],
        "narg_energies_hartree": energies,
        "energy_reproduction_error_hartree": (
            float(np.max(np.abs(energies - reference_energies)))
            if reference_energies is not None
            else None
        ),
        "records": records,
    }
    (args.output_dir / "pyrazine_narg_overlap_adaptive.json").write_text(
        json.dumps(_plain(payload), indent=2) + "\n"
    )
    np.savez(
        args.output_dir / "pyrazine_narg_overlap_adaptive.npz",
        budgets=np.asarray(args.budgets),
        overlaps=np.asarray(matrices),
        casci_overlap=casci_overlap,
        exact_narg_overlap=(
            exact_narg_overlap
            if exact_narg_overlap is not None
            else np.empty((0, 0), dtype=complex)
        ),
    )
    _plot(records, args.output_dir / "pyrazine_narg_overlap_adaptive.png")
    print(json.dumps(_plain(payload), indent=2), flush=True)


if __name__ == "__main__":
    main()
