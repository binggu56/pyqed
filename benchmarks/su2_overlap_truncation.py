"""Benchmark SU(2) cross-geometry overlap truncation controls.

The script compares finite ``max_bond`` and ``cutoff`` settings against an
untruncated reduced-SU(2) reference.  Outputs are written outside the repository
by default so repeated numerical runs do not dirty the worktree.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace
import time

import matplotlib.pyplot as plt
import numpy as np

from pyqed.mps.nonabelian import MPS
from pyqed.mps.nonabelian.states import (
    build_random_reduced_spatial_mps,
    spatial_target_sector,
)
from pyqed.qchem.dmrg.overlap import (
    _reduced_su2_overlap_matrix,
    su2_biorthogonal_overlap,
)


def _normalize(state):
    norm = float(np.real(_reduced_su2_overlap_matrix([state], [state])[0, 0]))
    if norm <= 0.0:
        raise ValueError("Synthetic reduced MPS has non-positive norm.")
    scale = norm ** -0.5
    first = state.sites[0]
    first.data = {key: scale * block for key, block in first.data.items()}
    return state


def _solver(state, ncas):
    return SimpleNamespace(
        dmrg=SimpleNamespace(ground_state=state, states=[state]),
        ncas=ncas,
        ncore=0,
        mo_coeff=np.eye(ncas),
        mol=None,
    )


def _summarize_info(info):
    transforms = info["transforms"]["bra"] + info["transforms"]["ket"]
    return {
        "resolved_max_bond": max(
            (entry["max_bond"] or entry["peak_reduced_bond_dimension"])
            for entry in transforms
        ),
        "peak_bond": max(entry["peak_reduced_bond_dimension"] for entry in transforms),
        "discarded_weight": sum(
            entry["sum_gate_discarded_weight"] for entry in transforms
        ),
        "truncated_gates": sum(entry["truncated_gate_count"] for entry in transforms),
    }


def _evaluate(bra, ket, metric, reference, *, label, cutoff, max_bond):
    start = time.perf_counter()
    try:
        value, info = su2_biorthogonal_overlap(
            bra,
            ket,
            s=metric,
            cutoff=cutoff,
            max_bond=max_bond,
            return_info=True,
        )
    except ValueError as exc:
        return {
            "label": label,
            "status": f"failed: {exc}",
            "cutoff": float(cutoff),
            "requested_max_bond": str(max_bond),
            "overlap_real": np.nan,
            "overlap_imag": np.nan,
            "absolute_error": np.nan,
            "relative_error": np.nan,
            "seconds": time.perf_counter() - start,
            "resolved_max_bond": 0,
            "peak_bond": 0,
            "discarded_weight": np.nan,
            "truncated_gates": 0,
        }
    elapsed = time.perf_counter() - start
    scalar = complex(value[0, 0])
    absolute_error = abs(scalar - reference)
    relative_error = absolute_error / max(abs(reference), np.finfo(float).eps)
    summary = _summarize_info(info)
    return {
        "label": label,
        "status": "ok",
        "cutoff": float(cutoff),
        "requested_max_bond": str(max_bond),
        "overlap_real": scalar.real,
        "overlap_imag": scalar.imag,
        "absolute_error": absolute_error,
        "relative_error": relative_error,
        "seconds": elapsed,
        **summary,
    }


def _plot(rows, output):
    bond_rows = [
        row
        for row in rows
        if row["label"].startswith("bond=") and row["status"] == "ok"
    ]
    cutoff_rows = [
        row
        for row in rows
        if row["label"].startswith("cutoff=") and row["status"] == "ok"
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))

    def panel(ax, data, x, xlabel):
        errors = np.maximum([row["relative_error"] for row in data], 1.0e-16)
        times = [row["seconds"] for row in data]
        ax.loglog(x, errors, "o-", color="#3366cc", label="relative error")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Relative overlap error", color="#3366cc")
        ax.tick_params(axis="y", labelcolor="#3366cc")
        ax.grid(alpha=0.25, which="both")
        twin = ax.twinx()
        twin.plot(x, times, "s--", color="#cc5533", label="runtime")
        twin.set_ylabel("Runtime (s)", color="#cc5533")
        twin.tick_params(axis="y", labelcolor="#cc5533")

    panel(
        axes[0],
        bond_rows,
        [row["resolved_max_bond"] for row in bond_rows],
        "Maximum reduced SU(2) bond dimension",
    )
    panel(
        axes[1],
        cutoff_rows,
        [max(row["cutoff"], 1.0e-14) for row in cutoff_rows],
        "SVD cutoff (0 shown at 1e-14)",
    )
    fig.suptitle("SU(2) cross-geometry overlap truncation")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ncas", type=int, default=10)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("/private/tmp/su2_overlap_truncation"),
    )
    args = parser.parse_args()

    state = _normalize(
        MPS.from_sites(
            build_random_reduced_spatial_mps(
                args.ncas,
                target_sector=spatial_target_sector(args.ncas, 0),
                bond_multiplicity=2,
                seed=args.seed,
            )
        )
    )
    bra = _solver(state, args.ncas)
    ket = _solver(state.copy(), args.ncas)
    rng = np.random.default_rng(args.seed + 1)
    metric = np.eye(args.ncas) + 0.03 * rng.normal(size=(args.ncas, args.ncas))

    exact_start = time.perf_counter()
    exact, exact_info = su2_biorthogonal_overlap(
        bra,
        ket,
        s=metric,
        cutoff=0.0,
        max_bond=None,
        return_info=True,
    )
    exact_seconds = time.perf_counter() - exact_start
    reference = complex(exact[0, 0])
    exact_summary = _summarize_info(exact_info)
    rows = [
        {
            "label": "exact",
            "status": "ok",
            "cutoff": 0.0,
            "requested_max_bond": "None",
            "overlap_real": reference.real,
            "overlap_imag": reference.imag,
            "absolute_error": 0.0,
            "relative_error": 0.0,
            "seconds": exact_seconds,
            **exact_summary,
        }
    ]

    for max_bond in (16, 32, 64, 128, 256, 384, 448, 512):
        rows.append(
            _evaluate(
                bra,
                ket,
                metric,
                reference,
                label=f"bond={max_bond}",
                cutoff=0.0,
                max_bond=max_bond,
            )
        )
    for cutoff in (1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8, 1.0e-10, 0.0):
        rows.append(
            _evaluate(
                bra,
                ket,
                metric,
                reference,
                label=f"cutoff={cutoff:g}",
                cutoff=cutoff,
                max_bond=None,
            )
        )
    rows.append(
        _evaluate(
            bra,
            ket,
            metric,
            reference,
            label="default",
            cutoff=1.0e-10,
            max_bond="auto",
        )
    )

    csv_path = args.output_prefix.with_suffix(".csv")
    png_path = args.output_prefix.with_suffix(".png")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _plot(rows, png_path)

    print(f"reference_overlap={reference.real:.16e}{reference.imag:+.16e}j")
    print("label              rel_error      discarded       peak      seconds  status")
    for row in rows:
        if row["status"] != "ok":
            print(
                f"{row['label']:<18} {'--':>11} {'--':>13} {'--':>10} "
                f"{row['seconds']:>10.4f}  {row['status']}"
            )
            continue
        print(
            f"{row['label']:<18} {row['relative_error']:>11.3e} "
            f"{row['discarded_weight']:>13.3e} {row['peak_bond']:>10d} "
            f"{row['seconds']:>10.4f}  ok"
        )
    print(f"csv={csv_path}")
    print(f"figure={png_path}")


if __name__ == "__main__":
    main()
